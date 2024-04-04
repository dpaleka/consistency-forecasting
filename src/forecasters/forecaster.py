from abc import ABC, abstractmethod
from pydantic import BaseModel
import asyncio
from common.datatypes import *


class Forecaster(ABC):

    def elicit(self, sentences: BaseModel, **kwargs) -> dict[str, Prob]:
        return {k: self.call(v, **kwargs) for k, v in sentences.model_fields.items()}

    async def elicit_async(self, sentences: BaseModel, **kwargs) -> dict[str, Prob]:
        keys, values = zip(*sentences.model_fields.items())
        tasks = [self.call_async(v, **kwargs) for v in values]
        results = await asyncio.gather(*tasks)
        return {k: v for k, v in zip(keys, results)}

    @abstractmethod
    def call(self, sentence: ForecastingQuestion, **kwargs) -> Prob:
        pass

    @abstractmethod
    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> Prob:
        pass
