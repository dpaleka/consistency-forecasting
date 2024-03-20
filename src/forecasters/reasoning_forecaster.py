import re
from common.datatypes import *
from common.llm_utils import answer, answer_sync, QandA
from .forecaster import Forecaster

class ReasoningForecaster(Forecaster):

    def __init__(self, preface: str = None, examples: list = None):
        self.preface = preface or " ".join([
            "You are an informed and well-calibrated forecaster. I need you to give me",
            "your best probability estimate for the following sentence or question resolving YES.",
            "I want you to first provide a reasoning for your answer, and then give me the probability.",
            "Your last sentence should be, 'The probability is: <float between 0 and 1>'",
        ])
        self.examples = examples or [QandA("Will Manhattan have a skyscraper a mile tall by 2030?", 
                                           "As of 2021, there are no skyscrapers a mile tall. There are also no plans to build any mile tall skyscraper in new york. The probability is: 0.03")]
    
    def call(self, sentence: str, **kwargs) -> Prob:
        response = answer_sync(
            prompt = sentence,
            preface = self.preface,
            examples = self.examples,
            **kwargs)
        return self.extract_prob(response)

    async def call_async(self, sentence: str, **kwargs) -> Prob:
        response = await answer(
            prompt = sentence,
            preface = self.preface,
            examples = self.examples,
            **kwargs)
        return self.extract_prob(response)

    def extract_prob(self, s: str) -> float:
        # Adjusted pattern to capture a number that follows either a colon or the word "is" with optional spaces
        pattern = r"(?:\:|is)\s*(-?\d*\.?\d+)"
        match = re.search(pattern, s)
        if match:
            try:
                # Directly return the float value from the captured group
                return float(match.group(1))
            except Exception as e:
                # TODO: log error
                return None
        else:
            return None
