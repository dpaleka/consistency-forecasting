import sys
from common.path_utils import get_src_path

sys.path.append(str(get_src_path()))

import os
from common.datatypes import ForecastingQuestion
from dotenv import load_dotenv
import pytest
import logging

from forecasters.advanced_forecaster import AdvancedForecaster


old_openrouter = os.environ.get("USE_OPENROUTER", "False")
os.environ["USE_OPENROUTER"] = "True"
old_skip_newscatcher = os.environ.get("SKIP_NEWSCATCHER", "False")
os.environ["SKIP_NEWSCATCHER"] = "True"

num_questions_to_run = 1

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger

load_dotenv()
pytest.mark.expensive = pytest.mark.skipif(
    os.getenv("TEST_ADV_FORECASTER", "False").lower() == "false",
    reason="Skipping advanced forecaster tests",
)


@pytest.fixture
def sample_forecasting_question():
    return ForecastingQuestion(
        id="99e4f7dd-312a-405f-9239-d38081d584eb",
        title="Will any country that had nuclear weapons on July 1, 2017 give them up before 2035?",
        body="Resolution Criteria\nThis question will resolve as Yes if one of the 9 nations known to possess nuclear weapons on July 1, 2017 (U.S., China, Russia, U.K., France, Israel, India, Pakistan, or North Korea) is certified by the International Atomic Energy Agency to have dismantled all nuclear devices and signs the 2017 UN ban on nuclear weapons on or before January 1, 2035.\nFine Print\nFor the purposes of this question, in the case of territorial disputes or challenges to a government's legitimacy, successor governments will be recognized as those which hold over 50% of the nation's de facto controlled territory on July 1, 2017. The successor government must also hold a political capital city within that same territory. If any of the 9 countries no longer exist or have no succesor as defined by January 1, 2035, they will be ignored for the resolution of the question.\n",
        resolution_date="2035-01-01 00:00:00+00:00",
        created_date=None,
        question_type="binary",
        data_source="metaculus",
        url="https://www.metaculus.com/questions/492",
        metadata={
            "topics": [
                {
                    "id": 141,
                    "slug": "united-states",
                    "name": "United States",
                    "link_id": 2137,
                    "num_questions": 1759,
                },
                {
                    "id": 93,
                    "slug": "china",
                    "name": "China",
                    "link_id": 2136,
                    "num_questions": 511,
                },
                {
                    "id": 18,
                    "slug": "russia",
                    "name": "Russia",
                    "link_id": 2133,
                    "num_questions": 430,
                },
                {
                    "id": 174,
                    "slug": "united-kingdom",
                    "name": "United Kingdom",
                    "link_id": 2139,
                    "num_questions": 353,
                },
                {
                    "id": 8,
                    "slug": "israel",
                    "name": "Israel",
                    "link_id": 2138,
                    "num_questions": 160,
                },
                {
                    "id": 389,
                    "slug": "india",
                    "name": "India",
                    "link_id": 2140,
                    "num_questions": 138,
                },
                {
                    "id": 55,
                    "slug": "france",
                    "name": "France",
                    "link_id": 2132,
                    "num_questions": 105,
                },
                {
                    "id": 82,
                    "slug": "north-korea",
                    "name": "North Korea",
                    "link_id": 2134,
                    "num_questions": 87,
                },
                {
                    "id": 88,
                    "slug": "united-nations",
                    "name": "United Nations",
                    "link_id": 2135,
                    "num_questions": 80,
                },
                {
                    "id": 610,
                    "slug": "pakistan",
                    "name": "Pakistan",
                    "link_id": 2141,
                    "num_questions": 54,
                },
                {
                    "id": 41,
                    "slug": "international-atomic-energy-agency",
                    "name": "International Atomic Energy Agency",
                    "link_id": 2131,
                    "num_questions": 22,
                },
                {
                    "id": 1327,
                    "slug": "treaty-on-the-non-proliferation-of-nuclear-weapons",
                    "name": "Treaty on the Non-Proliferation of Nuclear Weapons",
                    "link_id": 38373,
                    "num_questions": 15,
                },
            ],
            "background_info": 'In July 2017, 122 member states of the United Nations adopted a ban on nuclear weapons. The participating states agreed to "never under any circumstances to develop, test, produce, manufacture, otherwise acquire, possess or stockpile nuclear weapons or other nuclear explosive devices."\nNotably, none of the nations that currently possess nuclear weapons participated in the negotiations of the ban or adopted the document.\nSeveral treaties prior to this aimed to curb the development of nuclear weapons, notably the 1968 Non-Proliferation Treaty (NPT), which sought to limit nuclear development beyond five nuclear powers - the U.S., Russia, China, the U.K., and France.\nArguments against nuclear disarmament typically cite the principle of deterrence, that the possession of nuclear weapons by some states precludes the development or use of weapons by other states, due to the threat of nuclear retaliation. Proponents of the ban argue that previous efforts have not prevented states such as North Korea from pursuing nuclear programs, and that disarmament, rather than deterrence, is the best way to prevent nuclear war.\nIt\'s not unprecedented for states to completely disarm, however. South Africa dismantled its nuclear weapons beginning in 1989 and joined the NPT as a non-nuclear state. Three former Soviet republics, previously part of a nuclear-capable nation, also joined the NPT as non-nuclear states.\n',
        },
    )


def log(message):
    print(message)
    logging.info(message)


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_advanced_forecaster(sample_forecasting_question):
    default_test_date = "2024-04-30"
    af = AdvancedForecaster(
        MAX_WORDS_NEWSCATCHER=0,
        MAX_WORDS_GNEWS=2,
        NUM_ARTICLES_PER_QUERY=1,
        SEARCH_QUERY_MODEL_NAME="gpt-4o-mini-2024-07-18",
        SUMMARIZATION_MODEL_NAME="gpt-4o-mini-2024-07-18",
        BASE_REASONING_MODEL_NAMES=["gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"],
        RANKING_MODEL_NAME="gpt-4o-mini-2024-07-18",
        AGGREGATION_MODEL_NAME="gpt-4o-mini-2024-07-18",
        forecaster_date=default_test_date,
    )

    fq = sample_forecasting_question
    question: dict = fq.to_dict()
    log(
        f"\n{question['title']=}"
        f"\n{question['body']=}"
        f"\n{question['resolution_date']=}"
    )
    if "background_info" in question.get("metadata", {}):
        log(f"\n{question['metadata']['background_info']=}")
    log(f"\n{'%'*40}\n% Running Advanced Forecaster\n{'%'*40}\n")

    forecast = await af.call_async(fq=fq)
    final_prob = forecast.prob

    log(f"Final LLM probability: {final_prob}")

    assert 0 <= final_prob <= 1, f"Probability {final_prob} is not between 0 and 1"


os.environ["USE_OPENROUTER"] = old_openrouter
os.environ["SKIP_NEWSCATCHER"] = old_skip_newscatcher
