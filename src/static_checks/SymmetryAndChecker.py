""" Where f(x) is the model,
R(x1, x2)
    := x1 == P and Q; x2 == Q and P
S(f(x1), f(x2)) 
    :=  f(x1) == f(x2)
"""

from common.llm_utils import answer_sync, answer
from forecasters import ForecastingQuestionTuple, ProbsTuple
from .BaseChecker import BaseChecker
from .AndChecker import AndChecker

class SymmetryAndChecker(BaseChecker):
    preface = AndChecker.preface

    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(
        self, base_sentence_1: str, base_sentence_2: str, **kwargs
    ) -> ForecastingQuestionTuple:
        prompt_1 = self.stack(base_sentence_1, base_sentence_2)
        prompt_2 = self.stack(base_sentence_2, base_sentence_1)
        response_1 = answer_sync(prompt=prompt_1, preface=self.preface, **kwargs)
        response_2 = answer_sync(prompt=prompt_2, preface=self.preface, **kwargs)
        sentences = {"P_and_Q": response_1, "Q_and_P": response_2}
        return sentences

    async def instantiate(
        self, base_sentence_1: str, base_sentence_2: str, **kwargs
    ) -> ForecastingQuestionTuple:
        prompt_1 = self.stack(base_sentence_1, base_sentence_2)
        prompt_2 = self.stack(base_sentence_2, base_sentence_1)
        response_1 = await answer(prompt=prompt_1, preface=self.preface, **kwargs)
        response_2 = await answer(prompt=prompt_2, preface=self.preface, **kwargs)
        sentences = {"P_and_Q": response_1, "Q_and_P": response_2}
        return sentences

    def violation(self, answers: ProbsTuple) -> float:
        return abs(answers["P_and_Q"] - answers["Q_and_P"])
