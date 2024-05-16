#%%
import json
from common.datatypes import ForecastingQuestion_stripped, ForecastingQuestion, Prob_cot, Prob, PlainText
from common.llm_utils import query_api_chat_sync, query_api_chat_sync_native
import os

fq = ForecastingQuestion(
    title="Will Manhattan have a skyscraper a mile tall by 2030?",
    body=(
        "Resolves YES if at any point before 2030, there is at least "
        "one building in the NYC Borough of Manhattan (based on current "
        "geographic boundaries) that is at least a mile tall."
    ),
    resolution_date="2030-01-01T00:00:00",
    question_type="binary",
    data_source="manifold",
    url="https://www.metaculus.com/questions/12345/",
    metadata={"foo": "bar"},
    resolution=None,
)

fqs = ForecastingQuestion_stripped(
    title="Will Manhattan have a skyscraper a mile tall by 2030?",
    body=(
        "Resolves YES if at any point before 2030, there is at least "
        "one building in the NYC Borough of Manhattan (based on current "
        "geographic boundaries) that is at least a mile tall."
    ),
)

print(fqs.__str__())


# %%

#os.environ["USE_OPENROUTER"] = "True"
messages = [
    {"role": "system", "content": "You are a helpful assistant. Summarize the question for the user."},
    {"role": "user", "content": fq.__str__()}
]
#response = query_api_chat_sync(messages=messages, verbose=True, model="mistralai/mistral-large")
    


# %%
TEST_MANUAL=False
if TEST_MANUAL:
    from openai import OpenAI
    from os import getenv
    import instructor

    # gets API Key from environment variable OPENAI_API_KEY
    #_client = OpenAI(
    #base_url="https://openrouter.ai/api/v1",
    #api_key=getenv("OPENROUTER_API_KEY"),
    #)
    _client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
)

    #client = instructor.from_openai(_client, mode=instructor.Mode.MISTRAL_TOOLS)
    client = instructor.from_openai(_client,  mode=instructor.Mode.TOOLS)

    completion = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        #model="meta-llama/llama-3-70b-instruct",
        messages=[
            {
            "role": "user",
            "content": "Say this is a test",
            },
        ],
        response_model=PlainText,
    )
    print(completion)



#%%
#response = query_api_chat_sync_native(messages=messages, verbose=True, model="mistralai/mistral-large")
#print(response)
#%%
#response = query_api_chat_sync_native(messages=messages, verbose=True, model="meta-llama/llama-2-70b-chat-hf")
#print(response)
#%%
response = query_api_chat_sync_native(messages=messages, verbose=True, model="gpt-4o")
print(response)
# %%
