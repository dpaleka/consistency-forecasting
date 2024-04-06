"""Where f(x) is the model,
R(x1, x2, x3, x4)
    := x1 == P, x2 == Q, x3 == P or Q; x4 == Q or P
S(f(x1), f(x2), f(x3), f(x4))
    := f(x1) + f(x2) = f(x3) + f(x4)
"""

from common.llm_utils import answer_sync, answer
from forecasters import ForecastingQuestionTuple, ProbsTuple
from .MiniInstantiator import BaseChecker
from .AndChecker import AndChecker
from .OrChecker import OrChecker

class AndOrChecker(BaseChecker):
    
    preface_and = AndChecker.preface
    preface_or = OrChecker.preface
    
    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)
    
    def instantiate_sync(self, base_sentence_1: str, base_sentence_2: str, **kwargs) -> ForecastingQuestionTuple:
        prompt = self.stack(base_sentence_1, base_sentence_2)
        response_and = answer_sync(
            prompt=prompt, preface=self.preface_and, **kwargs
        )
        response_or = answer_sync(
            prompt=prompt, preface=self.preface_or, **kwargs
        )
        sentences = {"P": base_sentence_1, "Q": base_sentence_2, "P_and_Q" : response_and, "P_or_Q" : response_or }
        return sentences
    
    async def instantiate(self, base_sentence_1: str, base_sentence_2: str, **kwargs) -> ForecastingQuestionTuple:
        prompt = self.stack(base_sentence_1, base_sentence_2)
        response_and = await answer(
            prompt=prompt, preface=self.preface_and, **kwargs
        )
        response_or = await answer(
            prompt=prompt, preface=self.preface_or, **kwargs
        )
        sentences = {"P": base_sentence_1, "Q": base_sentence_2, "P_and_Q" : response_and, "P_or_Q" : response_or }
        return sentences
    
    def violation(self, answers: ProbsTuple) -> float:
        return abs(answers["P"] + answers["Q"] - answers["P_and_Q"] - answers["P_or_Q"])
