import pytest
from datetime import datetime
import uuid

from forecasters.consistent_forecaster import ConsistentForecaster
from forecasters import BasicForecaster
from static_checks.Checker import NegChecker, CondCondChecker
from common.datatypes import ForecastingQuestion

import os


pytest.mark.expensive = pytest.mark.skipif(
    os.getenv("TEST_CONSISTENT_FORECASTER", "False").lower() == "false",
    reason="Skipping ConsistentForecaster tests",
)

default_small_model = "gpt-4o-mini-2024-07-18"


@pytest.fixture
def consistent_forecaster():
    basic_forecaster = BasicForecaster(model=default_small_model)
    return ConsistentForecaster(
        model=default_small_model,
        hypocrite=basic_forecaster,
        checks=[
            NegChecker(path=""),
            CondCondChecker(path=""),
        ],
        use_generate_related_questions=True,
    )

@pytest.fixture
def consistent_forecaster_nouse():
    basic_forecaster = BasicForecaster(model=default_small_model)
    return ConsistentForecaster(
        model=default_small_model,
        hypocrite=basic_forecaster,
        checks=[
            NegChecker(path=""),
            CondCondChecker(path=""),
        ],
        use_generate_related_questions=False,
    )


@pytest.fixture
def consistent_forecaster_single():
    basic_forecaster = BasicForecaster(model=default_small_model)
    return ConsistentForecaster(
        model=default_small_model,
        hypocrite=basic_forecaster,
        checks=[
            NegChecker(path=""),
        ],
        use_generate_related_questions=True,
    )

@pytest.fixture
def consistent_forecaster_single_nouse():
    basic_forecaster = BasicForecaster(model=default_small_model)
    return ConsistentForecaster(
        model=default_small_model,
        hypocrite=basic_forecaster,
        checks=[
            NegChecker(path=""),
        ],
        use_generate_related_questions=False,
    )


test_fq_around_fifty_fifty = ForecastingQuestion(
    id=uuid.uuid4(),
    title="Will the Democrats win the 2028 presidential election in the U.S.?",
    body="Resolves Yes if the Democratic candidate wins the 2028 presidential election in the United States, and No otherwise. "
    "If the candidate is not a member of the Democratic party, but the party endorses them, resolves Yes. "
    "If the party ceases to exist and a candidate from another party wins, resolves No. "
    "If the election is not held between October and December 2028, or if there ceases to be an office of "
    "President of the United States, resolves N/A.",
    question_type="binary",
    resolution_date=datetime(2028, 12, 31),
    data_source="synthetic",
    url=None,
    metadata={"topics": ["Politics", "Elections", "United States"]},
    resolution=None,
)

test_fq_two = ForecastingQuestion(
    id=uuid.uuid4(),
    title="Will a nationwide ban on smoking in public places be enacted in the U.S. by 2030?",
    body="Resolves Yes if a law is enacted that prohibits smoking in all public places in the United States by 2030, "
    "and No otherwise. If a law is enacted that prohibits smoking in some but not all public places, resolves No. "
    "If no such law is enacted by 2030, resolves No. If the United States ceases to exist, resolves N/A.",
    question_type="binary",
    resolution_date=datetime(2030, 12, 31),
    data_source="synthetic",
    url=None,
    metadata={"topics": ["Health", "Public Policy", "United States"]},
    resolution=None,
)

test_fq_three = ForecastingQuestion(
    id=uuid.uuid4(),
    title="Will the U.S. government default on its debt by 2030?",
    body="Resolves Yes if the U.S. government fails to make a payment on its debt by the end of 2030, and No otherwise. "
    "If the U.S. government makes all scheduled payments on its debt by the end of 2030, resolves No. "
    "If the U.S. government ceases to exist, resolves N/A.",
    question_type="binary",
    resolution_date=datetime(2030, 12, 31),
    data_source="synthetic",
    url=None,
    metadata={"topics": ["Economics", "United States"]},
    resolution=None,
)

test_fq_four = ForecastingQuestion(
    id=uuid.uuid4(),
    title="Will there be a successful human mission to Mars by 2030?",
    body="Resolves Yes if a human mission to Mars is launched and successfully lands on Mars by the end of 2030, "
    "and No otherwise. If a human mission to Mars is launched but does not successfully land on Mars by the end of 2030, "
    "resolves No. If no human mission to Mars is launched by the end of 2030, resolves No. "
    "If the human species ceases to exist, resolves N/A.",
    question_type="binary",
    resolution_date=datetime(2030, 12, 31),
    data_source="synthetic",
    url=None,
    metadata={"topics": ["Space Exploration", "Mars"]},
    resolution=None,
)

test_fq_five = ForecastingQuestion(
    id=uuid.uuid4(),
    title="Will there be an earthquake of magnitude 8.0 or greater in California by 2030?",
    body="Resolves Yes if an earthquake of magnitude 8.0 or greater occurs in California by the end of 2030, "
    "and No otherwise. If no such earthquake occurs by the end of 2030, resolves No. "
    "If California ceases to exist, resolves N/A.",
    question_type="binary",
    resolution_date=datetime(2030, 12, 31),
    data_source="synthetic",
    url=None,
    metadata={"topics": ["Natural Disasters", "Earthquakes", "California"]},
    resolution=None,
)


@pytest.mark.asyncio
@pytest.mark.expensive
async def test_consistent_forecaster_call_async(consistent_forecaster):
    bq_func_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    instantiation_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    prob = await consistent_forecaster.call_async(
        test_fq_around_fifty_fifty,
        bq_func_kwargs=bq_func_kwargs,
        instantiation_kwargs=instantiation_kwargs,
    )
    prob = prob.prob
    print("Probability: ", prob)

    assert (
        isinstance(prob, float) and 0.1 <= prob <= 0.9
    ), "The probability should be a float with a non-extreme value"


@pytest.mark.expensive
def test_consistent_forecaster_call_sync(consistent_forecaster_nouse):
    bq_func_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    instantiation_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    prob = consistent_forecaster_nouse.call(
        test_fq_around_fifty_fifty,
        bq_func_kwargs=bq_func_kwargs,
        instantiation_kwargs=instantiation_kwargs,
    )
    prob = prob.prob
    print("Probability: ", prob)
    assert (
        isinstance(prob, float) and 0.1 <= prob <= 0.9
    ), "The probability should be a float with a non-extreme value"
