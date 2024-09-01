import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid

from common.datatypes import ForecastingQuestion
from forecasters import BasicForecaster, COT_Forecaster
from forecasters.consistent_forecaster import ConsistentForecaster
from static_checks.Checker import NegChecker, CondCondChecker

mock_q_and_a = "Will Manhattan have a skyscraper a mile tall by 2030?"
mock_response_list = ["0.03", "0.05", "0.02"]
mock_response = MagicMock(prob=0.09)


@pytest.fixture
def basic_forecaster():
    examples = [mock_q_and_a]
    return BasicForecaster(preface="Test preface")


@pytest.fixture
def mock_forecasting_question():
    return ForecastingQuestion(
        id=uuid.uuid4(),
        title="Test Title",
        body="Test Body",
        question_type="binary",
        resolution_date=datetime(2024, 1, 1),
        data_source="synthetic",
        url="http://example.com",
        metadata={"topics": ["test"]},
        resolution=True,
    )


@patch("forecasters.basic_forecaster.answer_sync", return_value=mock_response)
def test_basic_forecaster_call(
    mock_answer_sync, basic_forecaster, mock_forecasting_question
):
    expected_prob = mock_response.prob
    prob = basic_forecaster.call(mock_forecasting_question)
    assert prob == pytest.approx(
        expected_prob
    ), "The calculated probability does not match the expected value"
    mock_answer_sync.assert_called_once()


@pytest.mark.asyncio
@patch("forecasters.basic_forecaster.answer", return_value=mock_response)
async def test_basic_forecaster_call_async(
    mock_answer, basic_forecaster, mock_forecasting_question
):
    expected_prob = mock_response.prob
    prob = await basic_forecaster.call_async(mock_forecasting_question)
    assert prob == pytest.approx(
        expected_prob
    ), "The calculated probability does not match the expected value"
    mock_answer.assert_called_once()


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


def test_cot_forecaster_actual_call(mock_forecasting_question):
    # Prepare test data
    user_preface = (
        "What's the probability of this event occurring? Start with an initial estimate. You MUST mention the words 'Initial Estimate' in your answer\n"
        "Consider some counterarguments to your initial estimate. You MUST mention the word 'Counterarguments' in your answer\n"
        "Given these counterarguments, what's your final probability estimate?\n"
        "It MUST change from your initial estimate, at least a little bit. Even 0.001 of a percent change is enough.\n"
        "You MUST mention the word 'Final' in your answer\n"
        "Your chain of thought MUST end with a number between 0 and 1, inclusive."
    )

    # Call the forecaster with actual prompts
    forecaster = COT_Forecaster(preface=user_preface, examples=None)
    prob, chain_of_thought = forecaster.call(mock_forecasting_question)

    # Print the chain of thought for manual inspection
    print(f"\n{chain_of_thought=}")
    print(f"{prob=}\n")

    # Assert that we got a result
    assert isinstance(
        prob, float
    ), "Expected a float probability, but got a different type"
    assert 0 <= prob <= 1, f"Probability {prob} is not between 0 and 1"

    # Verify that the chain of thought was captured and shows a change in opinion
    assert (
        "initial estimate" in chain_of_thought.lower()
    ), "Chain of thought doesn't mention an initial estimate"
    assert (
        "counterarguments" in chain_of_thought.lower()
    ), "Chain of thought doesn't mention counterarguments"
    assert (
        "final" in chain_of_thought.lower()
    ), "Chain of thought doesn't mention a final estimate"

    # assert that the chain of thought final answer changes
    # Extract the final probability from the chain of thought
    import re

    cot_implied_prob_str = re.findall(r"\d+\.\d+", chain_of_thought)[-1]
    try:
        cot_implied_prob = float(cot_implied_prob_str)
        print(f"{cot_implied_prob=}")
        assert (
            abs(cot_implied_prob - prob) < 1e-6
        ), f"Final probability in chain of thought ({cot_implied_prob}) doesn't match returned probability ({prob})"
    except ValueError:
        pytest.fail(
            f"Failed to extract a valid final probability from the chain of thought. Last word was: {cot_implied_prob_str}"
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
