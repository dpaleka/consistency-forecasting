import json
from dotenv import load_dotenv

import sys
from common.path_utils import get_src_path

sys.path.append(str(get_src_path()))
from common.datatypes import ForecastingQuestion

# llm_forecasting imports
from forecasters.llm_forecasting.prompts.prompts import PROMPT_DICT
from forecasters.llm_forecasting.utils.time_utils import (
    get_todays_date,
    subtract_days_from_date,
)
from forecasters.llm_forecasting import ranking, summarize, ensemble


import os

os.environ["LOCAL_CACHE"] = ".forecaster_cache"
import uuid
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger

# %%
from forecasters.advanced_forecaster import AdvancedForecaster
from forecasters.basic_forecaster import BasicForecaster


import asyncio

import datetime
import os
import requests
import re
import aiohttp

# Load .env file if it exists
dotenv_path = os.path.dirname(os.path.abspath(os.getcwd()))
dotenv_path = os.path.join(dotenv_path, ".env")
# print(dotenv_path)
if os.path.exists(dotenv_path):
    print(dotenv_path)
    load_dotenv(dotenv_path)
METACULUS_TOKEN = os.environ.get("METACULUS_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_KEY")
assert METACULUS_TOKEN is not None


AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api2"


## TODO (change)
TOURNAMENT_ID = (
    3294  # 3294 is WARMUP_TOURNAMENT_ID, fill this in with competition details
)
SUBMIT_PREDICTION = (True,)  # turn on when ready to submit
TOTAL_QUESTIONS = 100  # also get from competition details
LOG_FILE_PATH = "metaculus_submissions.log"  # log file
ERROR_LOG_FILE_PATH = "metaculus_submission_errors.log"  # error log file
SUBMIT_CHOICE = "adv"  # [adv, basic, meta], pick which result you actually want to submit, defaults to adv
NO_COMMENT = False  # if true, posts 'test' as comment, else will take long time to use news to make "real" comment


## paramaterize forecasters
ADVANCED_FORECASTER = AdvancedForecaster(
    MAX_WORDS_NEWSCATCHER=5,
    MAX_WORDS_GNEWS=8,
    SEARCH_QUERY_MODEL_NAME="gpt-4o",
    SUMMARIZATION_MODEL_NAME="gpt-4o",
    BASE_REASONING_MODEL_NAMES=["gpt-4o", "gpt-4o"],
    RANKING_MODEL_NAME="gpt-4o",
    AGGREGATION_MODEL_NAME="gpt-4o",
)

BASIC_FORECASTER = BasicForecaster()


##Param comments
RETRIEVAL_CONFIG = {
    "NUM_SEARCH_QUERY_KEYWORDS": 3,
    "MAX_WORDS_NEWSCATCHER": 5,
    "MAX_WORDS_GNEWS": 8,
    "SEARCH_QUERY_MODEL_NAME": "gpt-4o",
    "SEARCH_QUERY_TEMPERATURE": 0.0,
    "SEARCH_QUERY_PROMPT_TEMPLATES": [
        PROMPT_DICT["search_query"]["0"],
        PROMPT_DICT["search_query"]["1"],
    ],
    "NUM_ARTICLES_PER_QUERY": 5,
    "SUMMARIZATION_MODEL_NAME": "gpt-3.5-turbo-1106",
    "SUMMARIZATION_TEMPERATURE": 0.2,
    "SUMMARIZATION_PROMPT_TEMPLATE": PROMPT_DICT["summarization"]["9"],
    "PRE_FILTER_WITH_EMBEDDING": True,
    "PRE_FILTER_WITH_EMBEDDING_THRESHOLD": 0.32,
    "RANKING_MODEL_NAME": "gpt-3.5-turbo-1106",
    "RANKING_TEMPERATURE": 0.0,
    "RANKING_PROMPT_TEMPLATE": PROMPT_DICT["ranking"]["0"],
    "RANKING_RELEVANCE_THRESHOLD": 4,
    "RANKING_COSINE_SIMILARITY_THRESHOLD": 0.5,
    "SORT_BY": "date",
    "RANKING_METHOD": "llm-rating",
    "RANKING_METHOD_LLM": "title_250_tokens",
    "NUM_SUMMARIES_THRESHOLD": 20,
    "EXTRACT_BACKGROUND_URLS": True,
}

REASONING_CONFIG = {
    "BASE_REASONING_MODEL_NAMES": ["gpt-4o", "gpt-4o"],
    "BASE_REASONING_TEMPERATURE": 1.0,
    "BASE_REASONING_PROMPT_TEMPLATES": [
        [
            PROMPT_DICT["binary"]["scratch_pad"]["1"],
            PROMPT_DICT["binary"]["scratch_pad"]["2"],
        ],
        [
            PROMPT_DICT["binary"]["scratch_pad"]["new_3"],
            PROMPT_DICT["binary"]["scratch_pad"]["new_6"],
        ],
    ],
    "AGGREGATION_METHOD": "meta",
    "AGGREGATION_PROMPT_TEMPLATE": PROMPT_DICT["meta_reasoning"]["0"],
    "AGGREGATION_TEMPERATURE": 0.2,
    "AGGREGATION_MODEL_NAME": "gpt-4",
    "AGGREGATION_WEIGTHTS": None,
}


def post_question_comment(question_id, comment_text):
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/",
        json={
            "comment_text": comment_text,
            "submit_type": "N",
            "include_latest_prediction": True,
            "question": question_id,
        },
        **AUTH_HEADERS,
    )
    response.raise_for_status()


def post_question_prediction(question_id, prediction_percentage):
    """
    Post a prediction value (between 1 and 100) on the question.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/predict/"
    response = requests.post(
        url,
        json={"prediction": float(prediction_percentage) / 100},
        **AUTH_HEADERS,
    )
    response.raise_for_status()


def list_questions(tournament_id, offset, count):
    """
    List (all details) {count} questions from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": "binary",
        "project": tournament_id,
        "status": "active",
        "type": "forecast",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/questions/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    response.raise_for_status()
    data = json.loads(response.content)
    return data


async def fetch_question_details_metaculus(question):
    url = question["metadata"]["api_url"]

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

    question["body"] = resolution_criteria
    question["metadata"]["market_prob"] = market_prob
    question["metadata"]["background_info"] = background_info

    return question


def metaculus_to_jsonl(m_dict):
    res = {
        "id": str(uuid.uuid4()),  ##random or not?
        "title": m_dict["title"],
        "body": None,  ##resolution criteria
        "resolution_date": m_dict["resolve_time"],
        "question_type": m_dict["possibilities"]["type"],
        "data_source": "metaculus",
        "url": "https://www.metaculus.com".format(),
        "metadata": {
            "topics": [],
            "api_url": m_dict["url"],
            "market_prob": None,  # market prob
            "background_info": None,  ## background info
        },
        "resolution": m_dict["resolution"],
    }

    res = fetch_question_details_metaculus(res)
    return res


async def gen_comments(q):
    ##RETRIEVAL
    question = q.title
    background_info = q.metadata["background_info"]
    resolution_criteria = q.body  # resolution criteria and other info is in |body|

    today_date = get_todays_date()
    # If open date is set in data structure, change beginning of retrieval to question open date.
    # Retrieve from [today's date - 1 month, today's date].
    retrieval_dates = (
        subtract_days_from_date(today_date, 30),
        today_date,
    )

    (
        ranked_articles,
        all_articles,
        search_queries_list_gnews,
        search_queries_list_nc,
    ) = await ranking.retrieve_summarize_and_rank_articles(
        question,
        background_info,
        resolution_criteria,
        retrieval_dates,
        urls=[],
        config=RETRIEVAL_CONFIG,
        return_intermediates=True,
    )
    all_summaries = summarize.concat_summaries(
        ranked_articles[: RETRIEVAL_CONFIG["NUM_SUMMARIES_THRESHOLD"]]
    )

    ##REASONING
    close_date = "N/A"  # data doesn't have explicit close date, so set to N/A
    today_to_close_date = [today_date, close_date]

    ensemble_dict = await ensemble.meta_reason(
        question=question,
        background_info=background_info,
        resolution_criteria=resolution_criteria,
        today_to_close_date_range=today_to_close_date,
        retrieved_info=all_summaries,
        reasoning_prompt_templates=REASONING_CONFIG["BASE_REASONING_PROMPT_TEMPLATES"],
        base_model_names=REASONING_CONFIG["BASE_REASONING_MODEL_NAMES"],
        base_temperature=REASONING_CONFIG["BASE_REASONING_TEMPERATURE"],
        aggregation_method=REASONING_CONFIG["AGGREGATION_METHOD"],
        weights=REASONING_CONFIG["AGGREGATION_WEIGTHTS"],
        meta_model_name=REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
        meta_prompt_template=REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
        meta_temperature=REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
    )

    cleaned_text = re.sub(
        r"\n4\. Output your prediction \(a number between 0 and 1\) with an asterisk at the beginning and end of the decimal\.\n.*",
        "",
        ensemble_dict["meta_reasoning"],
        flags=re.DOTALL,
    )

    meta_prediction = ensemble_dict["meta_prediction"]
    meta_prediction = 100 * float(meta_prediction)
    return cleaned_text, meta_prediction


def submission_log(log_file_path, message):
    # Check if the log file exists
    if not os.path.exists(log_file_path):
        # If the file doesn't exist, create it
        open(log_file_path, "w").close()

    ##Write
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Open the log file in append mode
    with open(log_file_path, "a") as log_file:
        # Write the log entry with timestamp
        log_file.write(f"{timestamp} - {message}\n")


async def main():
    if TOTAL_QUESTIONS <= 100:
        questions_list = list_questions(
            tournament_id=TOURNAMENT_ID, offset=0, count=TOTAL_QUESTIONS
        )["results"]

    else:
        questions_list = []

        for off in range(0, 100, TOTAL_QUESTIONS):
            new_qs = list_questions(tournament_id=TOURNAMENT_ID, offset=off, count=100)[
                "results"
            ]
            questions_list.append(new_qs)

    res_dict = {}

    for q in questions_list:
        id = int(q["id"])
        title = q["title"]
        url = q["url"]

        ##make sure it's actually valid post place
        try:
            post_question_prediction(id, 1.0)
        except Exception as e:
            msg = "id: {}, post_error_msg: {}".format(id, e)
            print(msg)
            ## error logs ERROR_LOG_FILE_PATH
            submission_log(ERROR_LOG_FILE_PATH, msg)
            ## no need to waste time generating info
            continue

        q = await metaculus_to_jsonl(q)
        q = ForecastingQuestion(**q)

        adv_prob = await ADVANCED_FORECASTER.call_async(sentence=q)
        basic_prob = await BASIC_FORECASTER.call_async(sentence=q)

        adv_prob = min(max(100 * float(adv_prob), 1), 100)
        basic_prob = min(max(100 * float(basic_prob), 1), 100)

        comments, meta_prob = "Test", 1
        if not NO_COMMENT:
            comments, meta_prob = await gen_comments(q)

        prediction_dict = {"adv": adv_prob, "basic": basic_prob, "meta": meta_prob}
        res_dict[id] = {
            "title": title,
            "predictions": prediction_dict,
            "comments": comments,
            "url": url,
        }

    for id, data in res_dict.items():
        if SUBMIT_PREDICTION:
            ##submission

            try:
                post_question_prediction(id, data["predictions"][SUBMIT_CHOICE.lower()])
                post_question_comment(id, data["comments"])

                ##logging
                message = "id: {}, url: {}, title: {}, adv_prediction: {}%, basic_prediction: {}%, meta_prediction: {}%, submission: {}%,\ncomments:\n{}\n".format(
                    id,
                    data["url"],
                    data["title"],
                    data["predictions"]["adv"],
                    data["predictions"]["basic"],
                    data["predictions"]["meta"],
                    data["predictions"][SUBMIT_CHOICE.lower()],
                    data["comments"],
                )
                print(message)
                print("********************")
                print("")
                submission_log(LOG_FILE_PATH, message)

            except Exception as e:
                msg = "id: {}, post_error_msg: {}".format(id, e)
                print(msg)
                print("********************")
                print("")
                submission_log(ERROR_LOG_FILE_PATH, msg)


if __name__ == "__main__":
    asyncio.run(main())