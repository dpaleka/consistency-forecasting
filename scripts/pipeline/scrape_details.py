import sys
import os
import aiohttp
from bs4 import BeautifulSoup
import re

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../src"))


async def get_market_prob(question):
    pass


async def fetch_question_details_predictit(question):
    pass


async def fetch_question_details_metaculus(question):
    question_id = question["id"]
    url = f"https://www.metaculus.com/api2/questions/{question_id}/"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response_json = await response.json()

    resolution_criteria = response_json.get("resolution_criteria", "")
    background_info = response_json.get("description", "")

    community_prediction = response_json.get("community_prediction", {}).get("full", {})
    q2_value = community_prediction.get("q2")
    if q2_value:
        market_prob = q2_value
    else:
        market_prob = None

    question["body"] = {
        "resolution_criteria": resolution_criteria,
        "background_info": background_info,
    }
    question["metadata"]["market_prob"] = market_prob

    return question


async def fetch_question_details_manifold(question):
    url = "https://api.manifold.markets/v0/slug/{}".format(
        question["url"].split("/")[-1]
    )
    resolution_criteria_text = ""
    background_info_text = ""

    print("Fetching question details for {}".format(url))
    # flush
    sys.stdout.flush()

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response_json = await response.json()

    content = response_json.get("textDescription", {})

    probability = response_json.get("probability")
    if probability is not None:
        question["metadata"]["market_prob"] = probability
    else:
        question["metadata"]["market_prob"] = None

    # Divide content into resolution criteria and background info
    # use LLM WIP
    """
    command = ("I am going to give you a description of a prediction market question."
    "Your role is to separate out the parts of the question as either 'background_info' or 'resolution_criteria'. "
    "'resolution_criteria' should contain information regarding the conditions that are required for the question to resolve. "
    "'background_info' will contain everything else. "
    "Note it is EXTREMELY important that no information is added or removed. "
    "The exact words / phrases of 'background_info' and 'resolution' criteria when combined should be the same as the original description. "
    "Also note that it's possible that there isn't sufficient information to fill one or both of them, in which case it's OK to leave them blank. "
    "Your response should be structed such that background_info and resolution_criteria are separated by ' ~~~ ' "
    "For example  "
    "background_info: 'abc' and resolution_criteria: 'xyz', response: 'abc ~~~ xyz' "
    "background_info: '' and resolution_criteria: 'xyz', response: ' ~~~ xyz' "
    "background_info: 'abc' and resolution_criteria: '', response: 'abc ~~~ xyz' "
    "background_info: '' and resolution_criteria: '', response: ' ~~~ ' "

    "In your response, you should not include 'background_info:' or 'resolution_criteria' before the statement.  The ~~~ separation is enough. "
    "Sentences should NEVER be split.  However, you are welcome to rearrange the ordering of the sentences. "
    
    "Example: "
    "Resolves YES if the U.S. Bureau of Labor Statistics reports at least one Consumer Price Index (CPI) YoY reading below 0 until the end of the year. "
    "This is a description that is CLEARLY a resolution_criteria.  Notite that is includes words such as resolves, will resolve, settles etc.")

    command = ("Respond to the input in all CAPS. " 
    "If there are no letters in the input, just respond with the same thing. "
    "Do NOT add or delete any information. "
    "DO NOT add commentary about what you are thinking or what you did.")

    if content == '':
        content = ' '
    msg = [{'role': 'system', 'content': command}, {'role': 'user', 'content': content}] 

    #model_output = await query_api_chat(msg,verbose=True, response_model= None)
    print(msg)
    model_output = await query_api_chat(msg,verbose=False)
    #model_output = model_output.choices[0].message.content

    """

    background_info_text = ""
    resolution_criteria_text = content.strip()

    question["body"] = {
        "resolution_criteria": resolution_criteria_text,
        "background_info": background_info_text,
    }

    full_description = response_json.get("description", {})
    question["metadata"]["full_description"] = full_description

    return question


async def fetch_question_details_manifold_slow(question):
    url = question["url"]
    resolution_criteria_text = ""
    background_info_text = ""

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")

            meta_tag = soup.find("meta", attrs={"name": "description"})

            ##Removes the extra xx% chance if it exists
            content = meta_tag["content"] if meta_tag else ""

        chance_match = re.search(r"(\d+)% chance", content)

        if chance_match:
            prob = (
                float(chance_match.group(1)) * 0.01
            )  # Convert the matched string to an integer
            question["metadata"]["market_prob"] = prob
            content = re.sub(r"^\d+% chance\. ", "", content)
        else:
            question["metadata"]["market_prob"] = prob

        ## Make LLM divide up into resolution criteria or background text

        background_info_text = content
        resolution_criteria_text = "nini"

        """
        msgs = 


        async def query_api_chat(
            messages: list[dict[str, str]],
            verbose=False,
            model: str | None = None,
            **kwargs,
        ) -> BaseModel:       

        """

        question["body"] = {
            "resolution_criteria": resolution_criteria_text,
            "background_info": background_info_text,
        }
        return question
