from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any
import functools
from common.datatypes import ForecastingQuestion, Forecast
from common.utils import shallow_dict
from common.llm_utils import parallelized_call


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
