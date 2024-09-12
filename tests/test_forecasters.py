import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import uuid
from datetime import datetime

from common.datatypes import ForecastingQuestion, Forecast
from forecasters import Forecaster
from forecasters.crowd_forecaster import CrowdForecaster
from forecasters.basic_forecaster import BasicForecaster
from forecasters.cot_forecaster import COT_Forecaster

@pytest.fixture
def mock_q_and_a():
    return [
        {"role": "human", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]

@pytest.fixture
def mock_response_list():
    return [
        "The probability is 30%.",
        "I estimate the probability to be 50%.",
        "Based on the given information, the probability is approximately 70%.",
    ]

@pytest.fixture
def mock_response(mock_response_list):
    return mock_response_list[1]

@pytest.fixture
def mock_cot_response():
    return """
    Let's approach this step-by-step:

    1. First, we need to consider...
    2. Then, we should take into account...
    3. Based on historical data...
    4. Considering current trends...
    5. Taking all factors into account...

    Therefore, I estimate the probability to be 60%.
    """

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

@pytest.fixture
def mock_forecasters():
    return [
        MagicMock(spec=Forecaster, call=MagicMock(return_value=Forecast(prob=0.3))),
        MagicMock(spec=Forecaster, call=MagicMock(return_value=Forecast(prob=0.5))),
        MagicMock(spec=Forecaster, call=MagicMock(return_value=Forecast(prob=0.7))),
    ]

def test_crowd_forecaster_call(mock_forecasters, mock_forecasting_question):
    crowd_forecaster = CrowdForecaster(mock_forecasters)
    forecast = crowd_forecaster.call(mock_forecasting_question)
    assert isinstance(forecast, Forecast)
    assert forecast.prob == pytest.approx(0.5)  # Average of 0.3, 0.5, and 0.7

@pytest.mark.asyncio
async def test_crowd_forecaster_call_async(mock_forecasters, mock_forecasting_question):
    for forecaster in mock_forecasters:
        forecaster.call_async = AsyncMock(return_value=forecaster.call.return_value)
    crowd_forecaster = CrowdForecaster(mock_forecasters)
    forecast = await crowd_forecaster.call_async(mock_forecasting_question)
    assert isinstance(forecast, Forecast)
    assert forecast.prob == pytest.approx(0.5)  # Average of 0.3, 0.5, and 0.7

def test_crowd_forecaster_add_remove_forecaster(mock_forecasters, mock_forecasting_question):
    crowd_forecaster = CrowdForecaster(mock_forecasters[:2])
    assert len(crowd_forecaster.forecasters) == 2
    forecast = crowd_forecaster.call(mock_forecasting_question)
    assert forecast.prob == pytest.approx(0.4)  # Average of 0.3 and 0.5

    new_forecaster = MagicMock(spec=Forecaster, call=MagicMock(return_value=Forecast(prob=0.9)))
    crowd_forecaster.add_forecaster(new_forecaster)
    assert len(crowd_forecaster.forecasters) == 3
    forecast = crowd_forecaster.call(mock_forecasting_question)
    assert forecast.prob == pytest.approx((0.3 + 0.5 + 0.9) / 3)

    crowd_forecaster.remove_forecaster(new_forecaster)
    assert len(crowd_forecaster.forecasters) == 2
    forecast = crowd_forecaster.call(mock_forecasting_question)
    assert forecast.prob == pytest.approx(0.4)  # Back to average of 0.3 and 0.5

def test_crowd_forecaster_dump_load_config(mock_forecasters):
    crowd_forecaster = CrowdForecaster(mock_forecasters)
    config = crowd_forecaster.dump_config()
    assert isinstance(config, dict)
    assert "forecasters" in config
    assert len(config["forecasters"]) == len(mock_forecasters)

    loaded_forecaster = CrowdForecaster.load_config(config)
    assert isinstance(loaded_forecaster, CrowdForecaster)
    assert len(loaded_forecaster.forecasters) == len(mock_forecasters)

def test_basic_forecaster_call(mock_forecasting_question, mock_response):
    forecaster = BasicForecaster()
    forecaster._get_llm_response = MagicMock(return_value=mock_response)
    forecast = forecaster.call(mock_forecasting_question)
    assert isinstance(forecast, Forecast)
    assert forecast.prob == 0.5

@pytest.mark.asyncio
async def test_basic_forecaster_call_async(mock_forecasting_question, mock_response):
    forecaster = BasicForecaster()
    forecaster._get_llm_response_async = AsyncMock(return_value=mock_response)
    forecast = await forecaster.call_async(mock_forecasting_question)
    assert isinstance(forecast, Forecast)
    assert forecast.prob == 0.5

def test_basic_forecaster_actual_call(mock_forecasting_question):
    forecaster = BasicForecaster()
    forecast = forecaster.call(mock_forecasting_question)
    assert isinstance(forecast, Forecast)
    assert 0 <= forecast.prob <= 1

def test_cot_forecaster_actual_call(mock_forecasting_question):
    forecaster = COT_Forecaster()
    forecast = forecaster.call(mock_forecasting_question)
    assert isinstance(forecast, Forecast)
    assert 0 <= forecast.prob <= 1
    assert "chain_of_thought" in forecast.metadata
    assert isinstance(forecast.metadata["chain_of_thought"], str)
