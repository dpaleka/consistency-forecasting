from abc import ABC, abstractmethod
from pydantic import BaseModel
import asyncio
from common.datatypes import *


class Forecaster(ABC):

    def elicit(self, sentences: BaseModel) -> dict[str, Prob]:
        return {k: self.call(v) for k, v in sentences.model_fields.items()}

    async def elicit_async(self, sentences: BaseModel) -> dict[str, Prob]:
        keys, values = zip(*sentences.items())
        tasks = [self.call_async(v) for v in values]
        results = await asyncio.gather(*tasks)
        return {k: v for k, v in zip(keys, results)}

    @abstractmethod
    def call(self, sentence: str) -> Prob:
        pass

    @abstractmethod
    async def call_async(self, sentence: str) -> Prob:
        pass
