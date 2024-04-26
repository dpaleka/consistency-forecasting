# %%
from typing import Literal, Union
import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai
from langchain.llms import OpenAI
import argparse
U, A, S = "user", "assistant", "system"

def api_keys():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_ORG_ID"):
        openai.organization = os.getenv("OPENAI_ORG_ID")

api_keys()


MODELS = ['gpt-4-0314', 'gpt-3.5-turbo-0301', 'text-davinci-003']
TYPES = ['negated_pair', 'bayes', 'paraphrase', 'precursor_event', 'compas_bail', 'monotonic_sequence']
SCALES = ['linear', 'log']

# The methods with no prefix before the number of shots are for prediction market questions
# Otherwise the task should be specified in the name
METHODS = ["1shot_china", "bail_0shot_expert", "recidivism_0shot_expert", "0shot_numerical", "1shot_climbers", "1shot_senate_neg", "1shot_senate_par"]

def filename_metadata(args):
    # if args is a dict, use it as a dict, otherwise assume it's an argparse.Namespace
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    
    #return f"{args.model}_method_{args.method}_tg_{'Yes' if args.together else 'No'}_T_{args.temperature}_times_{args.times}_mt_{args.max_tokens}"
    # remember to update this if you want to query the together stuff for sanity checks
    return f"{args.model}_method_{args.method}_T_{args.temperature}_times_{args.times}_mt_{args.max_tokens}"


#%%
# We'll just use strings as templates
class Template(str):
    def keys(self):
        """The string is something of the form 'foo{bar}baz{qux}' and this returns {'bar', 'qux'}"""
        import re
        return set(re.findall(r"{(\w+)}", self))
    
    class SafeDict(dict):
        def __missing__(self, key):
            return '{' + key + '}'

    def pformat(self, **kwargs):
        """ Partial string formatting, doesn't require all keys to be present """
        return Template(self.format_map(self.SafeDict(**kwargs)))
        
    pass

def test_template():
    x = Template("foo{bar}baz{qux}")
    print(x.keys())
    print(x.format(bar="BAR", qux="QUX"))
    print(x.pformat(qux="QUX"))

#test_template()

#%%
# Message
class Message(dict):
    def __init__(self, role : Literal["user", "assistant", "system"], content : str):
        assert role in ["user", "assistant", "system"]
        # we want to initialize a dict with keys "role" and "content"
        super().__init__(role=role, content=content)
        # if you want to initialize Message with a dict, use Message(**dict)

    
    
    def format(self, **kwargs):
        return Message(self["role"], self["content"].format(**kwargs))

    def pformat(self, **kwargs):
        return Message(self["role"], Template(self["content"]).pformat(**kwargs))
    

def chat_message(role : Literal["user", "assistant", "system"], content : str) -> dict[str, str]:
    return Message(role, content)

class Conversation():
    def __init__(self, messages : list, *args):
        self.messages = messages
        if len(messages) > 0:
            assert all(isinstance(msg, Message) for msg in self.messages)
            assert all(msg["role"] in ["user", "assistant", "system"] for msg in self.messages)
            assert self.messages[0]["role"] == "system"
    
    def format(self, **kwargs):
        return Conversation([msg.format(**kwargs) for msg in self.messages])
    
    def pformat(self, **kwargs):
        return Conversation([msg.pformat(**kwargs) for msg in self.messages])
    
    def __str__(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])


    
#%%
# TODO - figure out timeouts for model calls, we want to fail gracefully when the API is taking too long

#%%
def filter_model_kwargs(kwargs : Union[dict, argparse.Namespace]) -> dict:
    available_kwargs = ["max_tokens", "temperature"]
    #available_kwargs = ["max_tokens"]
    if isinstance(kwargs, argparse.Namespace):
        kwargs = vars(kwargs)
    return {k: v for k, v in kwargs.items() if k in available_kwargs}
#%%
# Define call_model function which adds the prompt to the returned JSON
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def call_chat_model(prompt : list[dict[str, str]], model : str = "gpt-3.5-turbo", temperature : float = 0., max_tokens : int = 400, **kwargs) -> dict:
    temperature = float(temperature)
    assert max_tokens > 0 and max_tokens <= 2048 and isinstance(max_tokens, int)
    assert temperature >= 0. and temperature <= 1.

    print(f"Temperature: {temperature}", f"Max tokens: {max_tokens}", f"Model: {model}")
    # print all the model parameters
    print(f"Prompt:\n", '\n'.join(map(str, prompt)))
    
    
    try:
        response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=temperature, **kwargs)
    except openai.error.OpenAIError as e:
        print("OpenAIError", e)
        """
              "choices": [
                        {
                            "finish_reason": "length",
                            "index": 0,
                            "message": {
                                "content": "To estimate the 100 meter men's sprint record in 2025, we can consider the following factors:\n1. Current record: As of 2021, the men's 100 meter sprint record is 9.58 seconds, set by Usain Bolt in 2009.\n2. Historical trend: The men's 100 meter sprint record has been decreasing steadily over the years, with significant improvements in the 20th century and smaller improvements in recent years.\n3. Recent performance:",
                                "role": "assistant"
                            }
                        }
                        ]
        """
        response = {"choices": [{"finish_reason": "error", "message": {"content": str(e), "role": "assistant"}}]}

    response["prompt"] = prompt
    return response


async def acall_model(prompt : str, model : str = "text-davinci-003", temperature : float = 0., max_tokens : int = 400, **kwargs) -> dict:
    llm = OpenAI(model_name=model, temperature=temperature, **kwargs)
    resp = await llm.agenerate([prompt])
    # you get generations=[[Generation(text='\n\nHello! Welcome to the world!', generation_info={'finish_reason': 'stop', 'logprobs': None})], [], [], [], [], [], [], [], [], [], []] llm_output={'token_usage': {'total_tokens': 11, 'prompt_tokens': 2, 'completion_tokens': 9}}
    response = {
        "output": resp.generations[0][0],
        "prompt": prompt,
    }
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def call_model(prompt : str, model : str = "text-davinci-003", temperature : float = 0., **kwargs) -> dict:
    # This is ditching some information about the number of tokens, hopefully that's not important right now
    llm = OpenAI(model_name=model, temperature=temperature, **kwargs)
    resp = llm.generate([prompt])
    # you get generations=[[Generation(text='\n\nHello! Welcome to the world!', generation_info={'finish_reason': 'stop', 'logprobs': None})], [], [], [], [], [], [], [], [], [], []] llm_output={'token_usage': {'total_tokens': 11, 'prompt_tokens': 2, 'completion_tokens': 9}}
    response = {
        "output": resp.generations[0][0].text,
        "generation_info": resp.generations[0][0].generation_info,
        "prompt": prompt,
    }
    return response

#%%
def test_async_call():
    api_keys()
    async def test_acall_model():
        response = await acall_model("Hello world")
        print(f"Response:\n{response}")
        return response
    
    return test_acall_model()

#await test_async_call()


#%%
# Basic functions

def get_content(response : dict) -> str:
    # Works for both chat and text models
    if "choices" in response:
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert len(response["choices"]) == 1
        return response["choices"][0]["message"]["content"]
    else:
        return response["output"]


#%%
# Example Prompt
orange_prompt = [
    chat_message("system", "You are a helpful assistant."),
    chat_message("user", "Knock knock."),
    chat_message("assistant", "Who's there?"),
    chat_message("user", "Orange."),
]


#%%
# Prob functions
def llm_get_prob(content : str) -> float:
    # Get the final probability from the response
    get_prob_prompt = [
        chat_message(S, """\
You are a helpful assistant who is good with extracting numbers from text.
You are given a text arguing for a given probability, and you must
extract the final probability in the form 0.XX, to two digits."""),
        chat_message(U, "Example: Blah blah blah\n[Answer] The probability of this happening is 0.746."),
        chat_message(A, "0.75"),
        chat_message(U, 
f"What is the final probability in the following text?\n{content}\n\Final probability: 0."),
    ]
    try:
        response = call_chat_model(get_prob_prompt, max_tokens=1, temperature=0.0) 
        prob_str = get_content(response)
        prob = float(prob_str)
    except:
        print("No final probability found in response; defaulting to 0.5")
        prob = 0.5
    return prob


def get_prob(content : str) -> float:
    # Get the probability from the last line of the response
    # or the latest line that starts with "[Answer]"
    # It should be in the format "[Answer] {"prob" : 0.65}"
    lines = content.splitlines()
    try:
        for line in reversed(lines):
            if line.startswith("[Answer]") and "prob" in line:
                print("line:", line)
                return float(line.split(":")[1].strip().strip("}"))
    except ValueError:
        return llm_get_prob(content)
    

#%%
# Write a LLM function to determine if the question resolves in a year that is year YYYY+1 or later
def is_future_chatpgt(question : str, YYYY : int) -> bool:
    # Query OpenAI API
    fewshot_prompt = [
        chat_message(S, """
You are a helpful assistant. You are given a prediction market question and a year YYYY. 
You must determine if the question resolves in a year that is in year YYYY+1 or later.
Respond with "Yes" or "No".
"""
        ),
        chat_message(U, "Example: [Question] Will the price of Bitcoin be greater than $100,000 on January 1, 2024? [Year] 2023"),
        chat_message(A, "Yes"),
        chat_message(U, "Example: [Question] Will Afghanistan be a member of the United Nations in 2025? [Year] 2025"),
        chat_message(A, "No"),
        chat_message(U, "Example: [Question] Will Afghanistan be a member of the United Nations in 2025? [Year] 2024"),
        chat_message(A, "Yes"),
        chat_message(U, "Example: [Question] Will the world population be greater than 8 billion in 2020? [Year] 2018"),
        chat_message(A, "Yes"),
    ]
    prompt = fewshot_prompt + [
        chat_message(U, question),
    ]

    response = call_chat_model(prompt, temperature=0.0)
    answer = get_content(response)
    print("Question:", question)
    print("Year:", YYYY)
    print("Answer:", answer)
    print("Response:", response)
    assert answer in ["Yes", "No"]
    return answer == "Yes"



#%%
# Write a legacy function to determine if the question resolves in a year that is year YYYY+1 or later, using regex
def is_future_regex(question : str, YYYY : int) -> bool:
    import re
    # Match a year in the question
    year_match = re.search(r"\d{4}", question)
    # discard if later than 2200
    if year_match is None or len(year_match.group()) != 4 or int(year_match.group()) > 2200:
        print(f"No year found in question: {question}")
        return False
    year = int(year_match.group())
    print(f"Year found in question: {year}")
    return year > YYYY

# %%
# Write something that filters the logprobs dict to only keep the most likely token whenever its logprob is above THRESH_SURE ~ -0.01
# And in general filters the logprob dict to contain only tokens with THRESH_POSSIBLE ~ -7.0
# The logprobs are like this
#<OpenAIObject at 0x7f80489d0b80> JSON: {
#  " Chain": -7.5711875,
#  "Answer": -6.827996,
#  "Chain": -0.25441772,
#  "Q": -1.5078415,
#  "Question": -7.8054614
#},
def filter_logprobs(logprobs : dict, THRESH_SURE : float = -0.01, THRESH_POSSIBLE : float = -7.0) -> dict:
    filtered_logprobs = {}
    for token, logprob in logprobs.items():
        if logprob > THRESH_SURE:
            filtered_logprobs[token] = logprob
            return filtered_logprobs
        elif logprob > THRESH_POSSIBLE:
            if token in filtered_logprobs:
                if filtered_logprobs[token] < logprob:
                    filtered_logprobs[token] = logprob
            else:
                filtered_logprobs[token] = logprob
    return filtered_logprobs

# %%
def get_prob_estimate(response : dict) -> tuple[float, float]:
    """
    Get the probability estimate from a response.
    Assumes the response is from a model that returns logprobs.

    :param response: the response from the API
        Has ["generation_info"]["logprobs"]["top_logprobs"] as a list of (str, float) dicts

    :return: a tuple of (average probability, standard deviation)
    """

    # The index of the last numeric token
    try:
        len_response = len(response["generation_info"]["logprobs"]["tokens"])
        answer_index = len_response - 1
        while True:
            cur_token = response["generation_info"]["logprobs"]["tokens"][answer_index]
            if cur_token.strip().isnumeric():
                # if next token starts with %, then it's a percentage
                if answer_index + 1 < len_response and response["generation_info"]["logprobs"]["tokens"][answer_index + 1][0] == "%":
                    break

                # also with the word percent
                if answer_index + 1 < len_response and response["generation_info"]["logprobs"]["tokens"][answer_index + 1].strip()[:3] == "per":
                    break

                # if previous token is a dot, then it's a decimal
                if answer_index - 1 >= 0 and response["generation_info"]["logprobs"]["tokens"][answer_index - 1] == ".":
                    break

            answer_index -= 1
            if answer_index < min(0, len_response - 15):
                raise ValueError("No numeric token found in the last 15 tokens")
            
        print("answer_index = ", answer_index)
        print(f'{response["generation_info"]["logprobs"]["tokens"][answer_index - 5:] = }')
        # hacky, need to assert high probability of it being a dot
        #assert response["generation_info"]["logprobs"]["tokens"][answer_index - 1]
        print(response.keys())
        #import code; code.interact(local=dict(globals(), **locals()))

        import numpy as np
        from collections import namedtuple
        Prediction = namedtuple("Prediction", ["answer", "weight"])
        predictions = []
        def make_twodigit_prob(strtoken : str) -> int:
            assert strtoken[0] in "0123456789"
            if len(strtoken) == 1:
                return int(strtoken) * 10  
            elif len(strtoken) >= 2:
                if len(strtoken) > 2:
                    print("Warning: probability token is longer than 2 digits, truncating")
                return int(strtoken[0]) * 10 + int(strtoken[1])
        
        for strtoken, logprob in response["generation_info"]["logprobs"]["top_logprobs"][answer_index].items():
            if strtoken.strip().isnumeric():
                predictions.append(Prediction(make_twodigit_prob(strtoken.strip()), np.exp(logprob)))
        
        if len(predictions) == 0:
            raise ValueError("No predictions found")

        # Get the average prediction and standard deviation
        print(predictions)
        assert sum(x.weight for x in predictions) <= 1.0001
        avg = sum(x.answer * x.weight for x in predictions) / sum(x.weight for x in predictions)
        var = sum(x.weight * (x.answer - avg)**2 for x in predictions)
        std = np.sqrt(var) # no bessel's correction for you!
    except:
        print("Warning: no predictions found, using uniform distribution")
        avg = 0.5
        std = np.sqrt(1/12) # standard deviation of a uniform distribution

    return avg, std


#%%
def get_numerical_answer(response : dict) -> float:
    """
    Get the numerical answer from a response.
    Assumes the response is from a model that returns logprobs.

    :param response: the response from the API
        Has ["generation_info"]["logprobs"]["top_logprobs"] as a list of (str, float) dicts
    
    :return: the numerical answer
    """

    # The index of the last numeric token
    try:
        len_response = len(response["generation_info"]["logprobs"]["tokens"])
        answer_index = len_response - 1
        while True:
            cur_token = response["generation_info"]["logprobs"]["tokens"][answer_index]
            if cur_token.strip().isnumeric():
                # if next token starts with %, then it's a percentage
                if answer_index + 1 < len_response and response["generation_info"]["logprobs"]["tokens"][answer_index + 1][0] == "%":
                    break

                # also with the word percent
                if answer_index + 1 < len_response and response["generation_info"]["logprobs"]["tokens"][answer_index + 1].strip()[:3] == "per":
                    break

                # if previous token is a dot, then it's a decimal
                if answer_index - 1 >= 0 and response["generation_info"]["logprobs"]["tokens"][answer_index - 1] == ".":
                    break

            answer_index -= 1
            if answer_index < min(0, len_response - 15):
                raise ValueError("No numeric token found in the last 15 tokens")
            
        print("answer_index = ", answer_index)
        print(f'{response["generation_info"]["logprobs"]["tokens"][answer_index - 5:] = }')
        # hacky, need to assert high probability of it being a dot
        #assert response["generation_info"]["logprobs"]["tokens"][answer_index - 1]
        print(response.keys())
        #import code; code.interact(local=dict(globals(), **locals()))

        def make_twodigit_prob(strtoken : str) -> int:
            assert strtoken[0] in "0123456789"
            if len(strtoken) == 1:
                return int(strtoken) * 10  
            elif len(strtoken) >= 2:
                if len(strtoken) > 2:
                    print("Warning: probability token is longer than 2 digits, truncating")
                return int(strtoken[0]) * 10 + int(strtoken[1])
        
        for strtoken, logprob in response["generation_info"]["logprobs"]["top_logprobs"][answer_index].items():
            if strtoken.strip().isnumeric():
                return make_twodigit_prob(strtoken.strip())
    except:
        print("Warning: no predictions found, using uniform distribution")
        return 50