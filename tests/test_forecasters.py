import pytest
from unittest.mock import MagicMock
from datetime import datetime
import uuid

from common.datatypes import ForecastingQuestion
from forecasters import BasicForecaster, ConsistentForecaster
from static_checks.Checker import NegChecker, CondCondChecker

mock_q_and_a = "Will Manhattan have a skyscraper a mile tall by 2030?"
mock_response_list = ["0.03", "0.05", "0.02"]
mock_response = MagicMock(prob=0.09)


# @pytest.fixture
# def basic_forecaster():
#    examples = [mock_q_and_a]
#    return BasicForecaster(preface="Test preface")
#
#
# @pytest.fixture
# def mock_forecasting_question():
#    return ForecastingQuestion(
#        id=uuid.uuid4(),
#        title="Test Title",
#        body="Test Body",
#        question_type="binary",
#        resolution_date=datetime(2024, 1, 1),
#        data_source="synthetic",
#        url="http://example.com",
#        metadata={"topics": ["test"]},
#        resolution=True
#    )
#
#
# @patch("forecasters.basic_forecaster.answer_sync", return_value=mock_response)
# def test_basic_forecaster_call(mock_answer_sync, basic_forecaster, mock_forecasting_question):
#    expected_prob = mock_response.prob
#    prob = basic_forecaster.call(mock_forecasting_question)
#    assert prob == pytest.approx(
#        expected_prob
#    ), "The calculated probability does not match the expected value"
#    mock_answer_sync.assert_called_once()
#
#
# @pytest.mark.asyncio
# @patch("forecasters.basic_forecaster.answer", return_value=mock_response)
# async def test_basic_forecaster_call_async(mock_answer, basic_forecaster, mock_forecasting_question):
#    expected_prob = mock_response.prob
#    prob = await basic_forecaster.call_async(mock_forecasting_question)
#    assert prob == pytest.approx(
#        expected_prob
#    ), "The calculated probability does not match the expected value"
#    mock_answer.assert_called_once()


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


@pytest.mark.asyncio
async def test_consistent_forecaster_call_async(consistent_forecaster):
    call_kwargs = {"model": "gpt-3.5-turbo"}
    bq_func_kwargs = {
        "model": "gpt-3.5-turbo"
    }  # doesn't pass, unsure where to provide model
    prob = await consistent_forecaster.call_async(
        test_fq_around_fifty_fifty, bq_func_kwargs=bq_func_kwargs, **call_kwargs
    )
    print("Probability: ", prob)

    assert (
        isinstance(prob, float) and 0.1 <= prob <= 0.9
    ), "The probability should be a float with a non-extreme value"


def test_consistent_forecaster_call_sync(consistent_forecaster):
    call_kwargs = {"model": "gpt-3.5-turbo"}
    bq_func_kwargs = {
        "model": "gpt-3.5-turbo"
    }  # doesn't pass, ucnlear where to provide model
    prob = consistent_forecaster.call(
        test_fq_around_fifty_fifty, bq_func_kwargs=bq_func_kwargs, **call_kwargs
    )
    print("Probability: ", prob)
    assert (
        isinstance(prob, float) and 0.1 <= prob <= 0.9
    ), "The probability should be a float with a non-extreme value"
