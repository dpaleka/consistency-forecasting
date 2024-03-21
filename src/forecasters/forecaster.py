from abc import ABC, abstractmethod
import asyncio
from common.datatypes import *

class Forecaster(ABC):

    def elicit(self, sentences: ForecastingQuestionTuple) -> ProbsTuple:
        return {k: self.call(v) for k, v in sentences.items()}

    async def elicit_async(self, sentences: ForecastingQuestionTuple) -> ProbsTuple:
        keys, values = zip(*sentences.items())
        tasks = [self.call_async(v, **kwargs) for v in values]
        results = await asyncio.gather(*tasks)
        return {k: v for k, v in zip(keys, results)}

    @abstractmethod
    def call(self, sentence: str) -> Prob:
        pass

    @abstractmethod
    async def call_async(self, sentence: str) -> Prob:
        pass
