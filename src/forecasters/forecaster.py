from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Literal, Optional
import functools
from pathlib import Path
from common.datatypes import ForecastingQuestion, Forecast
from common.utils import shallow_dict
from common.llm_utils import parallelized_call
import numpy as np
import asyncio
import json


class Forecaster(ABC):
    def elicit(
        self, fqs: BaseModel | dict[str, ForecastingQuestion], **kwargs
    ) -> dict[str, Any]:
        if isinstance(fqs, BaseModel):
            fqs = shallow_dict(fqs)
        return {k: self.call_full(v, **kwargs) for k, v in fqs.items()}

    async def elicit_async(
        self, fqs: BaseModel | dict[str, ForecastingQuestion], **kwargs
    ) -> dict[str, Any]:
        if isinstance(fqs, BaseModel):
            fqs = shallow_dict(fqs)
        list_kv = fqs.items()
        keys, questions = zip(*list_kv)
        call_func = functools.partial(self.call_async_full, **kwargs)
        results = await parallelized_call(call_func, questions)
        return {k: v for k, v in zip(keys, results)}

    def pre_call(self, fq: ForecastingQuestion, **kwargs) -> ForecastingQuestion:
        fq_copy = fq.model_copy()
        fq_copy.resolution = None
        fq_copy.metadata = None
        return fq_copy

    def call_full(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        fq = self.pre_call(fq, **kwargs)
        return self.call(fq, **kwargs)

    async def call_async_full(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        fq = self.pre_call(fq, **kwargs)
        return await self.call_async(fq, **kwargs)

    @abstractmethod
    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        pass

    @abstractmethod
    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        pass

    @abstractmethod
    def dump_config(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def load_config(cls, config: dict[str, Any]) -> "Forecaster":
        pass


def try_load_tuple(line: str) -> dict[str, ForecastingQuestion] | None:
    """
    {"key1": FQ, "key2": FQ, ..., "keyN": FQ, "metadata": {...}}
    Load the FQs
    """
    ret = {}
    try:
        data = json.loads(line)
        for k, v in data.items():
            if k == "metadata":
                continue
            else:
                ret[k] = ForecastingQuestion.model_validate_json(v)
        return ret
    except json.JSONDecodeError:
        return None


class LoadForecaster(Forecaster):
    def __init__(self, load_dir: Path):
        self.load_dir = load_dir
        self.data: dict[str, Forecast] = self.load_data()
        print(f"Loaded {len(self.data)} forecasts from {self.load_dir}")

    def hash_key_info_from_fq(self, fq: ForecastingQuestion) -> str:
        return fq.to_str_forecast_mode()

    def load_data(self) -> dict[str, Forecast]:
        data: dict[str, Forecast] = {}
        for path in self.load_dir.iterdir():
            if path.suffix != ".jsonl":
                continue
            print(f"Loading forecasts from {path}")
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                is_fq = False
                is_tuple = False
                try:
                    fq = ForecastingQuestion.model_validate_json(lines[0])
                    is_fq = True
                except json.JSONDecodeError:
                    is_fq = False
                    if try_load_tuple(lines) is not None:
                        is_tuple = True

                if is_fq:
                    for line in lines:
                        fq = ForecastingQuestion.model_validate_json(line)
                        hashed_fq = self.hash_key_info_from_fq(fq)
                        forecast = Forecast.model_validate_json(line)
                        data[hashed_fq] = forecast
                elif is_tuple:
                    for line in lines:
                        tuple = try_load_tuple(line)
                        assert (
                            tuple is not None
                        ), "If the first line of a file is a valid tuple, all lines must be valid tuples."
                        for k, v in tuple.fq_dict.items():
                            hashed_fq = self.hash_key_info_from_fq(v)
                            data[hashed_fq] = tuple.metadata
                else:
                    raise ValueError(
                        f"Invalid file format for {path}: neither a ForecastingQuestion nor a valid tuple of ForecastingQuestions"
                    )
        return data

    def call(self, fq: ForecastingQuestion, **kwargs) -> Optional[Forecast]:
        return self.data.get(self.hash_key_info_from_fq(fq), None)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Optional[Forecast]:
        return self.data.get(self.hash_key_info_from_fq(fq), None)

    def dump_config(self) -> dict[str, Any]:
        return {"load_dir": str(self.load_dir)}

    @classmethod
    def load_config(cls, config: dict[str, Any]) -> "LoadForecaster":
        return cls(Path(config["load_dir"]))


class CrowdForecaster(Forecaster):
    def __init__(
        self,
        forecasters: list[Forecaster],
        method: Literal["mean", "median"] = "mean",
        weights: list[float] | None = None,
        extremize_alpha: float | None = None,
    ):
        self.forecasters = forecasters
        self.method = method
        self.weights = weights
        self.extremize_alpha = extremize_alpha

    def _extremize_forecast(self, p: float) -> float:
        """
        From https://faculty.wharton.upenn.edu/wp-content/uploads/2015/07/2015---two-reasons-to-make-aggregated-probability-forecasts_1.pdf
        They use alpha = 2.5
        No change is when alpha is 1
        """
        alpha = self.extremize_alpha
        if alpha is None:
            return p
        else:
            return (p**alpha) / (p**alpha + (1 - p) ** alpha)

    def _combine_forecasts(self, forecasts: list[Forecast]) -> Forecast:
        if self.method == "mean":
            if self.weights is None:
                combined_prob = np.mean([f.prob for f in forecasts])
            else:
                combined_prob = np.average(
                    [f.prob for f in forecasts], weights=self.weights
                )
        elif self.method == "median":
            assert self.weights is None, "Median does not support weights"
            combined_prob = np.median([f.prob for f in forecasts])
        else:
            raise ValueError(f"Invalid forecast combination method: {self.method}")

        combined_prob = self._extremize_forecast(combined_prob)

        return Forecast(
            metadata={
                "probs": [f.prob for f in forecasts],
                "method": self.method,
                "weights": self.weights,
                "extremize_alpha": self.extremize_alpha,
                "constituent_forecasts_metadata": [f.metadata for f in forecasts],
            },
            prob=combined_prob,
        )

    def call(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        forecasts = [forecaster.call(fq, **kwargs) for forecaster in self.forecasters]
        return self._combine_forecasts(forecasts)

    async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
        forecasts = await asyncio.gather(
            *[forecaster.call_async(fq, **kwargs) for forecaster in self.forecasters]
        )
        return self._combine_forecasts(forecasts)

    def dump_config(self) -> dict[str, Any]:
        return {
            "forecasters": [f.dump_config() for f in self.forecasters],
            "method": self.method,
            "weights": self.weights,
        }

    @classmethod
    def load_config(cls, config: dict[str, Any]) -> "CrowdForecaster":
        forecasters = [
            Forecaster.load_config(f_config) for f_config in config["forecasters"]
        ]
        return cls(forecasters, config["method"], config["weights"])
