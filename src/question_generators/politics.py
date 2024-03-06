#%%
"""
This module contains functions for generating questions about politics.
"""

countries = [
    "United States",
    "United Kingdom",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "China",
    "Japan",
    "India",
    "Brazil",
]


"""
For each country, we're going to ask GPT-4 to generate 10 questions about the country's politics.
These questions should be about something that happens in 2024 or later, but not later than 2030.
"""

#%%
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.llm_utils import query_api_chat_sync, query_api_chat, parallelized_call

#model = "gpt-4-0125-preview"
model = "gpt-3.5-turbo"

def generate_politics_questions(country: str, k: int = 5) -> list[str]:
    kwargs = {"response_format": {"type": "json_object"}}
    messages = [
        {
            "role": "system",
            "content": """\
Generate {k} questions abot the politics of {country} between 2024 and 2030.
Give the answer in the JSON list format.
Example: "questions": ["text": "What is the probability that Joe Biden will be the president of the United States on July 1 2025?", "answer_type": "Prob"}.
Answer type should always be Prob.
"""
        },
        {
            "role": "user",
            "content": country,
        },
    ]
    response_text = query_api_chat_sync(model, messages, **kwargs)
    print(f"response_text:\n{response_text}\n")
    # now parse json
    try:
        response_json = json.loads(response_text)
        questions = [response_json["questions"][i]["text"] for i in range(k)]
        return questions
    except KeyError:
        print(f"Error parsing response for {country}")
        return []
    
from pathlib import Path
DATA_PATH = Path(__file__).parent.parent / "data"

# %%
if False:
    total_questions = []
    for country in countries:
        questions = generate_politics_questions(country, k=4)
        print(f"Questions for {country}:\n{questions}\n")
        total_questions += questions


    FILENAME = "politics_qs_1.json"
    with open(DATA_PATH / FILENAME, "w") as f:
        json.dump(total_questions, f, indent=4)


#%%

"""
Another approach.
We will have a list of topics and a list of countries.
We'll ask the model to generate questions about each topic for each country.
"""
topics = [
    "President/Prime Minister",
    "Gross Domestic Product",
    "Education (precise metrics)",
    "Environment (precise metrics)",
    "Wars and Conflicts",
    "Sports (precise achievements or metrics)"
]

def generate_politics_question_topic(topic, country, k : int = 5) -> list[str]:
    kwargs = {"response_format": {"type": "json_object"}}
    messages = [
        {
            "role": "system",
            "content": """\
Generate {k} questions abuot the politics of {country} between 2024 and 2030.
Give the answer in the JSON list format.
Example: "questions": ["text": "What is the probability that Joe Biden will be the president of the United States on July 1 2025?", "answer_type": "Prob"}.
Make sure the question is precise and unambigous.
Answer type should always be Prob.
"""
        },
        {
            "role": "user",
            "content": f"""\
The {k} questions should all be on the general topic of {topic}, in the country of {country}. """
        },
    ]
    response_text = query_api_chat_sync(model, messages, **kwargs)
    print(f"response_text:\n{response_text}\n")
    # now parse json
    try:
        response_json = json.loads(response_text)
        questions = [response_json["questions"][i]["text"] for i in range(k)]
        return questions
    except KeyError:
        print(f"Error parsing response for {country} and {topic}")
        return []

#%%
total_questions = []
for topic in topics:
    for country in countries:
        questions = generate_politics_question_topic(topic, country, k=1)
        total_questions.extend(questions)
        with open(DATA_PATH / "politics_qs_2.json", "w") as f:
            json.dump(total_questions, f, indent=4)
        print(f"Questions for {topic} in {country}: {questions}\n")

with open(DATA_PATH / "politics_qs_2.json", "w") as f:
    json.dump(total_questions, f, indent=4)

"""
{“title”: What is the probability that the United States will engage in a new major military conflict between 2024 and 2030”,
“details”: “... explain what doesn’t count as new, and what counts as major, exactly. Resolves on DATE if ...”,
“resolution_date”: 31 Dec 2030 (or 1 Jan 2023),
 } 

"""
# python politics.py  
# -> generates a file of question titles qtitles.json

# python sanity_check.py -i qtitles.json
# -> generates a file of qtitles_accepted.json, qtitles_rejected.json

# python generate_details.py -i qtitles_accepted.json -o qdetailed.json
# -> generates a file of full question dicts with details and resolution date, and whatever other metadata we need

# we'll need human parsing in the end, but for now just make sure these steps are sane











    








# %%
