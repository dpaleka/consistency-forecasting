from .forecaster import Forecaster
import re
from common.datatypes import *
from common.llm_utils import answer, answer_sync, QandA


class BasicForecaster(Forecaster):

    def __init__(self, preface: str = None, examples: list = None):
        self.preface = preface or " ".join([
            "You are an informed and well-calibrated forecaster. I need you to give me",
            "your best probability estimate for the following sentence or question resolving YES.",
            "Your answer should be a float between 0 and 1, with nothing else in your response."
        ])
        self.examples = examples or [QandA("Will Manhattan have a skyscraper a mile tall by 2030?", "0.03")]
    
    def call(self, sentence: Sentence, **kwargs) -> Prob:
        response = answer_sync(
            prompt = sentence.__str__(),
            preface = self.preface,
            examples = self.examples,
            **kwargs)
        return self.extract_prob(response)

    async def call_async(self, sentence: Sentence, **kwargs) -> Prob:
        response = await answer(
            prompt = sentence.__str__(),
            preface = self.preface,
            examples = self.examples,
            **kwargs)
        return self.extract_prob(response)

    def extract_prob(self, s: str) -> float:
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




