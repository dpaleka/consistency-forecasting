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

    async def elicit_async(self, sentences: SentencesTemplate) -> ProbsTemplate:
        keys, values = zip(*sentences.items())
        tasks = [self.call_async(v) for v in values]
        results = await asyncio.gather(*tasks)
        return {k: v for k, v in zip(keys, results)}

    @abstractmethod
    def call(self, sentence: str, log: bool = False) -> Prob:
        """
        Calls the forecaster synchronously with a sentence to forecast.
    
        :param sentence: The sentence to forecast.
        :param log: If True, logs the input sentence and the forecasted probability. Logging is recommended to be enabled by default for synchronous calls.
        :return: The forecasted probability as a Prob object.
        """
        pass

    @abstractmethod
    async def call_async(self, sentence: str, log: bool = False) -> Prob:
        """
        Asynchronously calls the forecaster with a sentence to forecast.
    
        :param sentence: The sentence to forecast.
        :param log: If True, logs the input sentence and the forecasted probability. Logging is optional for asynchronous calls.
        :return: The forecasted probability as a Prob object.
        """
        pass 