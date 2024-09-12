from typing import List, Any
from forecasters.forecaster import Forecaster
from common.datatypes import ForecastingQuestion, Forecast
import numpy as np
import asyncio
from pathlib import Path


class LoadForecaster(Forecaster):
    def load_data(self):
        data = {}
        for path in self.load_dir.iterdir():
            with open(path, "r") as f:
                fq = ForecastingQuestion.model_validate_json(f)
                data[fq] = Forecast.model_validate_json(f)
        return data

    def __init__(self, load_dir: Path):
        self.load_dir = load_dir
        self.data: dict[ForecastingQuestion, Forecast] = self.load_data()
        print(f"Loaded {len(self.data)} forecasts from {self.load_dir}")

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        return self.data.get(fq, None)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        return self.call(fq, **kwargs)


class CrowdForecaster(Forecaster):
    def __init__(self, forecasters: List[Forecaster]):
        self.forecasters = forecasters

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        forecasts = [forecaster.call(fq, **kwargs) for forecaster in self.forecasters]
        return self._combine_forecasts(forecasts)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        forecasts = await asyncio.gather(
            *[forecaster.call_async(fq, **kwargs) for forecaster in self.forecasters]
        )
        return self._combine_forecasts(forecasts)

    def _combine_forecasts(self, forecasts: List[Forecast]) -> Forecast:
        # Simple average of probabilities
        combined_prob = np.mean([f.prob for f in forecasts])
        return Forecast(prob=combined_prob)

    def dump_config(self) -> dict[str, Any]:
        return {"forecasters": [f.dump_config() for f in self.forecasters]}

    @classmethod
    def load_config(cls, config: dict[str, Any]) -> "CrowdForecaster":
        forecasters = [
            Forecaster.load_config(f_config) for f_config in config["forecasters"]
        ]
        return cls(forecasters)
