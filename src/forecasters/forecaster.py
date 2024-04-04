from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any
import asyncio
from common.datatypes import *
from common.utils import shallow_dict

class Forecaster(ABC):

    def elicit(self, sentences: BaseModel, **kwargs) -> dict[str, Any]:
        return {k: self.call(v, **kwargs) for k, v in shallow_dict(sentences).items()}

    async def elicit_async(self, sentences: BaseModel, **kwargs) -> dict[str, Any]:
        keys, values = zip(*sentences.model_fields.items())
        tasks = [self.call_async(v, **kwargs) for v in values]
        results = await asyncio.gather(*tasks)
        return {k: v for k, v in zip(keys, results)}

    @abstractmethod
    def call(self, sentence: ForecastingQuestion, **kwargs) -> Any:
        pass

    @abstractmethod
    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> Any:
        pass
