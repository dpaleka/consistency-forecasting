from abc import ABC, abstractmethod
import asyncio
from common.datatypes import *

class Forecaster(ABC):

    def elicit(self, sentences: ForecastingQuestionTemplate) -> ProbsTemplate:
        return {k: self.call(v) for k, v in sentences.items()}

    async def elicit_async(self, sentences: ForecastingQuestionTemplate) -> ProbsTemplate:
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
