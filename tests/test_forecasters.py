import pytest
from unittest.mock import patch

from common.datatypes import Prob
from forecasters import BasicForecaster

mock_q_and_a = "Will Manhattan have a skyscraper a mile tall by 2030?"
mock_response_list = ["0.03", "0.05", "0.02"]
mock_response = "0.09"


@pytest.fixture
def basic_forecaster():
    examples = [mock_q_and_a]
    return BasicForecaster(preface="Test preface")


@patch("forecasters.basic_forecaster.answer_sync", return_value=mock_response)
def test_basic_forecaster_call(mock_answer_sync, basic_forecaster):
    sentence = "Test sentence"
    expected_prob = Prob(float(mock_response))
    prob = basic_forecaster.call(sentence)
    assert prob == pytest.approx(
        expected_prob
    ), "The calculated probability does not match the expected value"
    mock_answer_sync.assert_called_once()


@pytest.mark.asyncio
@patch("forecasters.basic_forecaster.answer", return_value=mock_response)
async def test_basic_forecaster_call_async(mock_answer, basic_forecaster):
    sentence = "Test sentence"
    expected_prob = Prob(float(mock_response))
    prob = await basic_forecaster.call_async(sentence)
    assert prob == pytest.approx(
        expected_prob
    ), "The calculated probability does not match the expected value"
    mock_answer.assert_called_once()
