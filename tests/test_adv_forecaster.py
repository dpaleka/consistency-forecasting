import sys
from common.path_utils import get_src_path, get_data_path
import asyncio

sys.path.append(str(get_src_path()))

import os
from common.datatypes import ForecastingQuestion
import json

from forecasters.advanced_forecaster import AdvancedForecaster


os.environ["USE_OPENROUTER"] = "True"
os.environ["SKIP_NEWSCATCHER"] = "True"

data = []
with open(get_data_path() / "other/forecaster_testing_q.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))

sample_question = data[0]

num_questions_to_run = 1

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger


af = AdvancedForecaster(
    MAX_WORDS_NEWSCATCHER=0,
    MAX_WORDS_GNEWS=2,
    NUM_ARTICLES_PER_QUERY=1,
    SEARCH_QUERY_MODEL_NAME="anthropic/claude-3-haiku",
    SUMMARIZATION_MODEL_NAME="anthropic/claude-3-haiku",
    BASE_REASONING_MODEL_NAMES=["anthropic/claude-3-haiku", "anthropic/claude-3-haiku"],
    RANKING_MODEL_NAME="anthropic/claude-3-haiku",
    AGGREGATION_MODEL_NAME="anthropic/claude-3-haiku",
)

for question in data[:num_questions_to_run]:
    fq = ForecastingQuestion(**question)
    print(
        f"\n{question['title']}\n{question['body']}\n{question['resolution_date']}\n\n{'%'*40}\n% Running Advanced Forecaster\n{'%'*40}\n\n"
    )

    final_prob = asyncio.run(af.call_async(sentence=fq))

    print("Final LLM probability", final_prob)
