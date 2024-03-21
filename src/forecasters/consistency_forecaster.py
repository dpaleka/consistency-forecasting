import numpy as np
import re
from common.datatypes import *
from common.llm_utils import answer, answer_sync, QandA
from .forecaster import Forecaster

class ConsistentAskForecaster(Forecaster):

    def __init__(self, temperature: float = 0.2, n: int = 5, preface: str = None, examples: list = None):
        self.temperature = temperature 
        self.n = n
        self.preface = preface or " ".join([
            "You are an informed and well-calibrated forecaster. I need you to give me",
            "your best probability estimate for the following sentence or question resolving YES.",
            "Your answer should be a float between 0 and 1, with nothing else in your response."
        ])
        self.examples = examples or [QandA("Will Manhattan have a skyscraper a mile tall by 2030?", "0.03")]
    
    def call(self, sentence: str, **kwargs) -> Prob:
        kwargs["temperature"] = kwargs.get("temperature", self.temperature)
        kwargs["n"] = kwargs.get("n", self.n)
        response = answer_sync(
            prompt = sentence,
            preface = self.preface,
            examples = self.examples,
            **kwargs)
        mean = np.mean([prob for prob in map(self.extract_prob, response) if prob is not None]) if response else None
        return Prob(mean)

    async def call_async(self, sentence: str, **kwargs) -> Prob:
        kwargs["temperature"] = kwargs.get("temperature", self.temperature)
        kwargs["n"] = kwargs.get("n", self.n)
        response = await answer(
            prompt = sentence,
            preface = self.preface,
            examples = self.examples,
            **kwargs)
        mean = np.mean([prob for prob in map(self.extract_prob, response) if prob is not None]) if response else None
        return Prob(mean)

    def extract_prob(self, s: str) -> Prob:
        pattern = r"-?\d*\.?\d+"
        match = re.search(pattern, s)
        if match:
            try:
                return Prob(float(match.group()))
            except Exception as e:
                #TODO: log error
                return None
        else:
            return None
