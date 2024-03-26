#%%
import json
from common.llm_utils import query_api_chat_sync

model = "gpt-3.5-turbo"
messages = [
    {
        "role": "system",
        "content": """\
Generate a question about the politics of the United States between 2024 and 2030.
Give the answer in the JSON object format.
Example: {"question": "What is the probability that Joe Biden will be the president of the United States on July 1 2025?", "answer_type": "Prob"}.
Answer type should always be Prob.
"""
    },
    {
        "role": "user",
        "content": "United States",
    },
]
response = query_api_chat_sync(model, messages, response_format={"type": "json_object"})
print(response)

response_dict = json.loads(response)

print(json.dumps(response_dict, indent=4))


# %%
