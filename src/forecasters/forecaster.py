from abc import ABC, abstractmethod
from typing import Dict
import asyncio

class Prob(float):
    def __new__(cls, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return super(Prob, cls).__new__(cls, value)


SentencesTemplate = Dict[str, str]
ProbsTemplate = Dict[str, Prob]

class Forecaster(ABC):

    def elicit(self, sentences: SentencesTemplate) -> ProbsTemplate:
        return {k: self.call(v) for k, v in sentences.items()}

    async def elicit_async(self, sentences: SentencesTemplate, **kwargs) -> ProbsTemplate:
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