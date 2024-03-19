"""Where f(x) is the model,
R(x1, x2, x3, x4)
    := x1 == P; x2 == Q|P; x3 == R|(P and Q); x4 == P and Q and R
S(f(x1), f(x2), f(x3), f(x4))
    :=  f(x1) * f(x2) * f(x3) == f(x4)
"""

import numpy as np
from common.llm_utils import answer_sync, answer
from forecasters import ForecastingQuestionTemplate, ProbsTemplate
from .BaseChecker import BaseChecker
from .AndChecker import AndChecker
from .CondChecker import CondChecker


class TowerChecker(BaseChecker):
    preface_cond = CondChecker.preface_cond
    preface_and = AndChecker.preface

    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(
        self, base_sentence_1: str, base_sentence_2: str, base_sentence_3: str, **kwargs
    ) -> ForecastingQuestionTemplate:
        prompt_PQ = self.stack(base_sentence_1, base_sentence_2)

        response_Q_cond_P = answer_sync(
            prompt=prompt_PQ, preface=self.preface_cond, **kwargs
        )
        response_P_and_Q = answer_sync(
            prompt=prompt_PQ, preface=self.preface_and, **kwargs
        )

        prompt_P_and_Q_R = self.stack(response_P_and_Q, base_sentence_3)
        response_R_cond_P_and_Q = answer_sync(
            prompt=prompt_P_and_Q_R, preface=self.preface_cond, **kwargs
        )

        response_P_and_Q_and_R = answer_sync(
            prompt=prompt_P_and_Q_R, preface=self.preface_and, **kwargs
        )

        sentences = {
            "P": base_sentence_1,
            "Q_given_P": response_Q_cond_P,
            "R_given_(P_and_Q)": response_R_cond_P_and_Q,
            "P_and_Q_and_R": response_P_and_Q_and_R,
        }
        return sentences

    async def instantiate(
        self, base_sentence_1: str, base_sentence_2: str, base_sentence_3: str, **kwargs
    ) -> ForecastingQuestionTemplate:
        prompt_PQ = self.stack(base_sentence_1, base_sentence_2)

        response_Q_cond_P = await answer(
            prompt=prompt_PQ, preface=self.preface_cond, **kwargs
        )
        response_P_and_Q = await answer(
            prompt=prompt_PQ, preface=self.preface_and, **kwargs
        )

        prompt_P_and_Q_R = self.stack(response_P_and_Q, base_sentence_3)
        response_R_cond_P_and_Q = await answer(
            prompt=prompt_P_and_Q_R, preface=self.preface_cond, **kwargs
        )

        response_P_and_Q_and_R = await answer(
            prompt=prompt_P_and_Q_R, preface=self.preface_and, **kwargs
        )

        sentences = {
            "P": base_sentence_1,
            "Q_given_P": response_Q_cond_P,
            "R_given_(P_and_Q)": response_R_cond_P_and_Q,
            "P_and_Q_and_R": response_P_and_Q_and_R,
        }
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(
            np.log(
                answers["P"]
                * answers["Q_given_P"]
                * answers["R_given_(P_and_Q)"]
                / answers["P_and_Q_and_R"]
            )
        )
