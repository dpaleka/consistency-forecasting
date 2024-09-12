import pytest
from unittest.mock import MagicMock, AsyncMock
import uuid
from datetime import datetime

from common.datatypes import ForecastingQuestion, Forecast
from forecasters import Forecaster
from forecasters.crowd_forecaster import CrowdForecaster

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
