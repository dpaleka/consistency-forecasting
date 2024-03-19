""" Where f(x) is the model,
R(x1, x2, x3)
    := x1 == P; x2 == ¬P and Q; x3 == P or Q
S(f(x1), f(x2), f(x3)) 
    :=  f(x1) + f(x2) = f(x3)
"""

from common.llm_utils import answer_sync, answer
from forecasters import ForecastingQuestionTemplate, ProbsTemplate
from .BaseChecker import BaseChecker
from .OrChecker import OrChecker

class ButNotChecker(BaseChecker):

    preface_butnot = (
        "You are a helpful assistant. I will give you two questions, q1 and q2. "
        "I want you to give me a single question that expresses 'q2 but not q1' "
        "in the most succint manner possible. Make sure your response is in the " 
        "form of a single question, and exactly expresses 'q2 but not q1'.")

    preface_or = OrChecker.preface
    
    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(self, base_sentence_1 : str, base_sentence_2 : str, **kwargs) -> ForecastingQuestionTemplate:
        prompt = self.stack(base_sentence_1, base_sentence_2)
        response_butnot = answer_sync(
            prompt = prompt, 
            preface = self.preface_butnot,
            **kwargs)
        response_or = answer_sync(
            prompt = prompt, 
            preface = self.preface_or,
            **kwargs)
        sentences = {"P": base_sentence_1, "Q_butnot_P": response_butnot, "P_or_Q" : response_or }
        return sentences

    async def instantiate(self, base_sentence_1: str, base_sentence_2 : str, **kwargs) -> ForecastingQuestionTemplate:
        prompt = self.stack(base_sentence_1, base_sentence_2)
        response_butnot = await answer(
            prompt = prompt, 
            preface = self.preface_butnot,
            **kwargs)
        response_or = await answer(
            prompt = prompt, 
            preface = self.preface_or,
            **kwargs)
        sentences = { "P": base_sentence_1, "Q_butnot_P": response_butnot, "P_or_Q" : response_or }
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(answers["P"] + answers["Q_butnot_P"] - answers["P_or_Q"])
