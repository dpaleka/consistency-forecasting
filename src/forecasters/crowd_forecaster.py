from typing import List, Any
from forecasters.forecaster import Forecaster
from common.datatypes import ForecastingQuestion, Forecast
import numpy as np
import asyncio

class CrowdForecaster(Forecaster):
    def __init__(self, forecasters: List[Forecaster]):
        self.forecasters = forecasters

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        forecasts = [forecaster.call(fq, **kwargs) for forecaster in self.forecasters]
        return self._combine_forecasts(forecasts)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        forecasts = await asyncio.gather(*[forecaster.call_async(fq, **kwargs) for forecaster in self.forecasters])
        return self._combine_forecasts(forecasts)

    def _combine_forecasts(self, forecasts: List[Forecast]) -> Forecast:
        # Simple average of probabilities
        combined_prob = np.mean([f.prob for f in forecasts])
        return Forecast(prob=combined_prob)

    def add_forecaster(self, forecaster: Forecaster):
        self.forecasters.append(forecaster)

    def remove_forecaster(self, forecaster: Forecaster):
        self.forecasters.remove(forecaster)

    def dump_config(self) -> dict[str, Any]:
        return {
            "forecasters": [f.dump_config() for f in self.forecasters]
        }

    @classmethod
    def load_config(cls, config: dict[str, Any]) -> "CrowdForecaster":
        forecasters = [Forecaster.load_config(f_config) for f_config in config["forecasters"]]
        return cls(forecasters)
