import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import uuid

from common.datatypes import Prob, ForecastingQuestion
from forecasters import BasicForecaster

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
        resolution=True
    )


@patch("forecasters.basic_forecaster.answer_sync", return_value=mock_response)
def test_basic_forecaster_call(mock_answer_sync, basic_forecaster, mock_forecasting_question):
    expected_prob = Prob(prob=mock_response.prob)
    prob = basic_forecaster.call(mock_forecasting_question)
    assert prob.prob == pytest.approx(
        expected_prob.prob
    ), "The calculated probability does not match the expected value"
    mock_answer_sync.assert_called_once()


@pytest.mark.asyncio
@patch("forecasters.basic_forecaster.answer", return_value=mock_response)
async def test_basic_forecaster_call_async(mock_answer, basic_forecaster, mock_forecasting_question):
    expected_prob = Prob(prob=mock_response.prob)
    prob = await basic_forecaster.call_async(mock_forecasting_question)
    assert prob.prob == pytest.approx(
        expected_prob.prob
    ), "The calculated probability does not match the expected value"
    mock_answer.assert_called_once()
