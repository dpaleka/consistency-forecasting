import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid

from common.datatypes import ForecastingQuestion, ForecastingQuestion_stripped, Prob_cot
from common.llm_utils import Example
from forecasters.cot_forecaster import COT_Forecaster, CoT_multistep_Forecaster
from forecasters.basic_forecaster import BasicForecaster

mock_q_and_a = "Will Manhattan have a skyscraper a mile tall by 2030?"
mock_response_list = ["0.03", "0.05", "0.02"]
mock_response = MagicMock(prob=0.09)
mock_cot_response = MagicMock(
    chain_of_thought="I think because of this and that, the probability is 0.09",
    prob=0.09,
)

pytest.mark.expensive = pytest.mark.skipif(
    os.getenv("TEST_CONSISTENT_FORECASTER", "False").lower() == "false",
    reason="Skipping ConsistentForecaster tests",
)


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
def cot_forecaster():
    examples = [
        Example(
            user=ForecastingQuestion_stripped(
                title="Will Manhattan have a skyscraper a mile tall by 2030?",
                body="Resolves YES if at any point before 2030, there is at least one building in the NYC Borough of Manhattan (based on current geographic boundaries) that is at least a mile tall.",
            ),
            assistant=Prob_cot(
                chain_of_thought="As of 2021, there are no skyscrapers a mile tall. There are also no plans to build any mile tall skyscraper in new york. The probability is: 0.03",
                prob=0.03,
            ),
        )
    ]
    return COT_Forecaster(preface="Test preface", examples=examples)


STEPS_MULTISTEP = 3


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
def cot_multistep_forecaster():
    examples = [
        Example(
            user=ForecastingQuestion_stripped(
                title="Will Manhattan have a skyscraper a mile tall by 2030?",
                body="Resolves YES if at any point before 2030, there is at least one building in the NYC Borough of Manhattan (based on current geographic boundaries) that is at least a mile tall.",
            ),
            assistant=Prob_cot(
                chain_of_thought="Step 1: Consider current tallest buildings...\nStep 2: Evaluate technological feasibility...\nStep 3: Assess economic factors...",
                prob=0.03,
            ),
        )
    ]
    return CoT_multistep_Forecaster(
        preface="Test preface", examples=examples, steps=STEPS_MULTISTEP
    )


@patch("forecasters.cot_forecaster.answer_sync", return_value=mock_cot_response)
def test_cot_forecaster_call(
    mock_answer_sync, cot_forecaster, mock_forecasting_question
):
    expected_prob = mock_cot_response.prob
    prob = cot_forecaster.call(mock_forecasting_question)
    assert prob == pytest.approx(
        expected_prob
    ), "The calculated probability does not match the expected value"
    mock_answer_sync.assert_called_once()


@pytest.mark.asyncio
@patch("forecasters.cot_forecaster.answer", return_value=mock_cot_response)
async def test_cot_forecaster_call_async(
    mock_answer, cot_forecaster, mock_forecasting_question
):
    expected_prob = mock_cot_response.prob
    prob = await cot_forecaster.call_async(mock_forecasting_question)
    assert prob == pytest.approx(
        expected_prob
    ), "The calculated probability does not match the expected value"
    mock_answer.assert_called_once()


@patch(
    "forecasters.cot_forecaster.answer_messages_sync", return_value=mock_cot_response
)
def test_cot_multistep_forecaster_call(
    mock_answer_messages_sync, cot_multistep_forecaster, mock_forecasting_question
):
    prob = cot_multistep_forecaster.call(mock_forecasting_question)
    assert prob == 0.09
    # assert called steps times
    assert mock_answer_messages_sync.call_count == STEPS_MULTISTEP


@pytest.mark.asyncio
@patch("forecasters.cot_forecaster.answer_messages", return_value=mock_cot_response)
async def test_cot_multistep_forecaster_call_async(
    mock_answer_messages, cot_multistep_forecaster, mock_forecasting_question
):
    prob = await cot_multistep_forecaster.call_async(mock_forecasting_question)
    assert prob == 0.09
    # assert called steps times
    assert mock_answer_messages.call_count == STEPS_MULTISTEP
