""" Where f(x) is the model,
R(x1, x2, x3)
    := x1 == P; x2 == Q|P and Q; x3 == P and Q
S(f(x1), f(x2), f(x3)) 
    :=  f(x1) * f(x2) == f(x3)
"""

import numpy as np
from common.llm_utils import answer_sync, answer
from forecasters import SentencesTemplate, ProbsTemplate
from .BaseChecker import BaseChecker
from .AndChecker import AndChecker


class CondChecker(BaseChecker):

    preface_cond = (
        "You are a helpful assistant. I will give you two questions, q1 and q2. "
        " I want you to give me a single question that expresses 'given q1, then q2' "
        "in the most succint manner possible. Make sure your response is in the "
        "form of a single question, and exactly expresses 'given q1, then q2'."
    )

    preface_and = AndChecker.preface

    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(
        self, base_sentence_1: str, base_sentence_2: str, **kwargs
    ) -> SentencesTemplate:
        prompt = self.stack(base_sentence_1, base_sentence_2)
        response_cond = answer_sync(prompt=prompt, preface=self.preface_cond, **kwargs)
        response_and = answer_sync(prompt=prompt, preface=self.preface_and, **kwargs)
        sentences = {
            "P": base_sentence_1,
            "Q_given_P": response_cond,
            "P_and_Q": response_and,
        }
        return sentences

    async def instantiate(
        self, base_sentence_1: str, base_sentence_2: str, **kwargs
    ) -> SentencesTemplate:
        prompt = self.stack(base_sentence_1, base_sentence_2)
        response_cond = await answer(prompt=prompt, preface=self.preface_cond, **kwargs)
        response_and = await answer(prompt=prompt, preface=self.preface_and, **kwargs)
        sentences = {
            "P": base_sentence_1,
            "Q_given_P": response_cond,
            "P_and_Q": response_and,
        }
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(np.log(answers["P"] * answers["Q_given_P"] / answers["P_and_Q"]))
