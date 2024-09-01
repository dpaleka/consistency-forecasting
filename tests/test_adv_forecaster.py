import sys
from common.path_utils import get_src_path, get_data_path

sys.path.append(str(get_src_path()))

import os
from common.datatypes import ForecastingQuestion
import json
from dotenv import load_dotenv
import pytest
import logging

from forecasters.advanced_forecaster import AdvancedForecaster


old_openrouter = os.environ.get("USE_OPENROUTER", "False")
os.environ["USE_OPENROUTER"] = "True"
os.environ["SKIP_NEWSCATCHER"] = "True"

num_questions_to_run = 1

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger

load_dotenv()
pytest.mark.expensive = pytest.mark.skipif(
    os.getenv("TEST_ADV_FORECASTER", "False").lower() == "false",
    reason="Skipping advanced forecaster tests",
)
data = []
with open(get_data_path() / "other/forecaster_testing_q.jsonl", "r") as file:
    for line in file:
        data.append(json.loads(line))


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_advanced_forecaster():
    af = AdvancedForecaster(
        MAX_WORDS_NEWSCATCHER=0,
        MAX_WORDS_GNEWS=2,
        NUM_ARTICLES_PER_QUERY=1,
        SEARCH_QUERY_MODEL_NAME="gpt-4o-mini-2024-07-18",
        SUMMARIZATION_MODEL_NAME="gpt-4o-mini-2024-07-18",
        BASE_REASONING_MODEL_NAMES=["gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"],
        RANKING_MODEL_NAME="gpt-4o-mini-2024-07-18",
        AGGREGATION_MODEL_NAME="gpt-4o-mini-2024-07-18",
    )

    for question in data[:num_questions_to_run]:
        fq = ForecastingQuestion(**question)
        logging.info(
            f"\n{question['title']}\n{question['body']}\n{question['resolution_date']}\n\n{'%'*40}\n% Running Advanced Forecaster\n{'%'*40}\n"
        )

        final_prob = await af.call_async_full(sentence=fq)

        logging.info(f"Final LLM probability: {final_prob}")

        assert 0 <= final_prob <= 1, f"Probability {final_prob} is not between 0 and 1"


os.environ["USE_OPENROUTER"] = old_openrouter
