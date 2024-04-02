from .forecaster import Forecaster
from common.datatypes import *
from common.llm_utils import answer, answer_sync, Example


class BasicForecaster(Forecaster):

    def __init__(self, preface: str = None, examples: list = None):
        self.preface = preface or (
            "You are an informed and well-calibrated forecaster. I need you to give me "
            "your best probability estimate for the following sentence or question resolving YES. "
            "Your answer should be a float between 0 and 1, with nothing else in your response."
        )

        self.examples = examples or [
            Example("Will Manhattan have a skyscraper a mile tall by 2030?", "0.03")
        ]

    def call(self, sentence: ForecastingQuestion, **kwargs) -> Prob:
        response = answer_sync(
            prompt=sentence.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type,
            **kwargs
        )
        return response

    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> Prob:
        response = await answer(
            prompt=sentence.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type,
            **kwargs
        )
        return self.extract_prob(response)