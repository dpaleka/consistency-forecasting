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


@pytest.fixture
def consistent_forecaster():
    basic_forecaster = BasicForecaster()
    return ConsistentForecaster(
        basic_forecaster,
        checks=[
            NegChecker(),
            CondCondChecker(),
        ],
    )


@pytest.fixture
def consistent_forecaster_single():
    basic_forecaster = BasicForecaster()
    return ConsistentForecaster(
        basic_forecaster,
        checks=[
            NegChecker(),
        ],
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
async def test_consistent_forecaster_call_async(consistent_forecaster):
    call_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    bq_func_kwargs = {
        "model": "gpt-4o-mini-2024-07-18"
    }  # doesn't pass, unsure where to provide model
    prob = await consistent_forecaster.call_async(
        test_fq_around_fifty_fifty, bq_func_kwargs=bq_func_kwargs, **call_kwargs
    )
    print("Probability: ", prob)

    assert (
        isinstance(prob, float) and 0.1 <= prob <= 0.9
    ), "The probability should be a float with a non-extreme value"


def test_consistent_forecaster_call_sync(consistent_forecaster):
    call_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    bq_func_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    prob = consistent_forecaster.call(
        test_fq_around_fifty_fifty, bq_func_kwargs=bq_func_kwargs, **call_kwargs
    )
    print("Probability: ", prob)
    assert (
        isinstance(prob, float) and 0.1 <= prob <= 0.9
    ), "The probability should be a float with a non-extreme value"


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_consistent_forecaster_consistent_async(consistent_forecaster_single):
    # check that the ConsistentForecaster is actually consistent on the check that it is made consistent on

    call_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    instantiation_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    bq_func_kwargs = {"model": "gpt-4o-mini-2024-07-18"}

    checker = consistent_forecaster_single.checks[0]
    n = checker.num_base_questions

    keys = ["P", "Q", "R", "S", "T"]
    bq_list = [
        test_fq_around_fifty_fifty,
        test_fq_two,
        test_fq_three,
        test_fq_four,
        test_fq_five,
    ]
    bqs = {k: bq for k, bq in zip(keys[:n], bq_list[:n])}

    tup = await checker.instantiate(bqs, **instantiation_kwargs)

    if isinstance(tup, list):
        tup = tup[0]

    answers = await consistent_forecaster_single.elicit_async(
        tup,
        bq_func_kwargs=bq_func_kwargs,
        instantiation_kwargs=instantiation_kwargs,
        **call_kwargs,
    )
    print("Answers:\n", answers)
    v = checker.violation(answers)
    print("Violation: ", v)
    assert v < 0.0001, "The violation should be 0"


@pytest.mark.expensive
def test_consistent_forecaster_consistent_sync(consistent_forecaster_single):
    # check that the ConsistentForecaster is actually consistent on the check that it is made consistent on

    call_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    instantiation_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    bq_func_kwargs = {"model": "gpt-4o-mini-2024-07-18"}

    checker = consistent_forecaster_single.checks[0]
    n = checker.num_base_questions

    keys = ["P", "Q", "R", "S", "T"]
    bq_list = [
        test_fq_around_fifty_fifty,
        test_fq_two,
        test_fq_three,
        test_fq_four,
        test_fq_five,
    ]
    bqs = {k: bq for k, bq in zip(keys[:n], bq_list[:n])}

    tup = checker.instantiate_sync(bqs, **instantiation_kwargs)

    if isinstance(tup, list):
        tup = tup[0]

    answers = consistent_forecaster_single.elicit(
        tup,
        bq_func_kwargs=bq_func_kwargs,
        instantiation_kwargs=instantiation_kwargs,
        **call_kwargs,
    )
    print("Answers:\n", answers)
    v = checker.violation(answers)
    print("Violation: ", v)
    assert v < 0.0001, "The violation should be 0"


@pytest.mark.expensive
@pytest.mark.asyncio
async def test_consistent_forecaster_more_consistent_async(
    consistent_forecaster_single,
):
    # check that the ConsistentForecaster is more consistent than the hypocrite it improves upon

    call_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    instantiation_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    bq_func_kwargs = {"model": "gpt-4o-mini-2024-07-18"}

    checker = consistent_forecaster_single.checks[0]
    n = checker.num_base_questions

    keys = ["P", "Q", "R", "S", "T"]
    bq_list = [
        test_fq_around_fifty_fifty,
        test_fq_two,
        test_fq_three,
        test_fq_four,
        test_fq_five,
    ]
    bqs = {k: bq for k, bq in zip(keys[:n], bq_list[:n])}

    tup = await checker.instantiate(bqs, **instantiation_kwargs)

    if isinstance(tup, list):
        tup = tup[0]

    answers = await consistent_forecaster_single.elicit_async(
        tup,
        bq_func_kwargs=bq_func_kwargs,
        instantiation_kwargs=instantiation_kwargs,
        **call_kwargs,
    )
    print("Answers:\n", answers)
    v = checker.violation(answers)
    print("Violation: ", v)
    print("---")
    hypocrite_answers = await consistent_forecaster_single.hypocrite.elicit_async(
        tup, **call_kwargs
    )
    print("Hypocrite Answers:\n", hypocrite_answers)
    hv = checker.violation(hypocrite_answers)
    print("Hypocrite Violation: ", hv)
    assert (
        v <= hv
    ), "The ConsistentForecaster should be at least as consistent as the hypocrite"


@pytest.mark.expensive
def test_consistent_forecaster_more_consistent_sync(consistent_forecaster_single):
    # check that the ConsistentForecaster is more consistent than the hypocrite it improves upon

    call_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    instantiation_kwargs = {"model": "gpt-4o-mini-2024-07-18"}
    bq_func_kwargs = {"model": "gpt-4o-mini-2024-07-18"}

    checker = consistent_forecaster_single.checks[0]
    n = checker.num_base_questions

    keys = ["P", "Q", "R", "S", "T"]
    bq_list = [
        test_fq_around_fifty_fifty,
        test_fq_two,
        test_fq_three,
        test_fq_four,
        test_fq_five,
    ]
    bqs = {k: bq for k, bq in zip(keys[:n], bq_list[:n])}

    tup = checker.instantiate_sync(bqs, **instantiation_kwargs)

    if isinstance(tup, list):
        tup = tup[0]

    answers = consistent_forecaster_single.elicit(
        tup,
        bq_func_kwargs=bq_func_kwargs,
        instantiation_kwargs=instantiation_kwargs,
        **call_kwargs,
    )
    print("Answers:\n", answers)
    v = checker.violation(answers)
    print("Violation: ", v)
    print("---")
    hypocrite_answers = consistent_forecaster_single.hypocrite.elicit(
        tup, **call_kwargs
    )
    print("Hypocrite Answers:\n", hypocrite_answers)
    hv = checker.violation(hypocrite_answers)
    print("Hypocrite Violation: ", hv)
    assert (
        v <= hv
    ), "The ConsistentForecaster should be at least as consistent as the hypocrite"
