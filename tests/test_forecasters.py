import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid

from common.datatypes import ForecastingQuestion, ForecastingQuestion_stripped, Prob_cot
from common.llm_utils import Example
from forecasters import BasicForecaster
from forecasters.cot_forecaster import COT_Forecaster, CoT_multistep_Forecaster

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
