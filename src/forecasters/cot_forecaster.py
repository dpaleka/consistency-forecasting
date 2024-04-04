from common.datatypes import *
from common.llm_utils import answer, answer_sync, Example
from .forecaster import Forecaster


class COT_Forecaster(Forecaster):

    def __init__(self, preface: str = None, examples: list[Example] = None):
        self.preface = preface or (
            "You are an informed and well-calibrated forecaster. I need you to give me "
            "your best probability estimate for the following sentence or question resolving YES. "
            "I want you to first provide a reasoning for your answer, and then give me the probability. "
            "Your last sentence should be, 'The probability is: <float between 0 and 1>'"
        )
        self.examples = examples or [
            Example(
                "Will Manhattan have a skyscraper a mile tall by 2030?",
                "As of 2021, there are no skyscrapers a mile tall. There are also no plans to build any mile tall skyscraper in new york. The probability is: 0.03",
            )
        ]

    def call(self, sentence: ForecastingQuestion, **kwargs) -> float:
        response = answer_sync(
            prompt=sentence.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type(mode="cot"),
            **kwargs,
        )
        return response.prob

    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> float:
        response = await answer(
            prompt=sentence.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type(mode="cot"),
            **kwargs,
        )
        return response.prob