"""Where f(x) is the model,
R(x1, x2, x3, x4)
    := x1 == P; x2 == Q|P; x3 == R|(P and Q); x4 == P and Q and R
S(f(x1), f(x2), f(x3), f(x4))
    :=  f(x1) * f(x2) * f(x3) == f(x4)
"""

from common.llm_utils import answer_sync, answer
from forecasters import SentencesTemplate, ProbsTemplate
from .BaseChecker import BaseChecker
from .AndChecker import AndChecker
from .CondChecker import CondChecker


class TowerChecker(BaseChecker):
    preface_cond = CondChecker.preface_cond
    preface_and = AndChecker.preface
    
    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)
    
    # def instantiate_sync(self, base_sentence_1: str, base_sentence_2: str, base_sentence_3: str, **kwargs) -> SentencesTemplate:
    #     prompt_1 = self.stack(base_sentence_1, base_sentence_2)
    #     prompt_2 = self.stack(base_sentence_1, base_sentence_2, base_sentence_3)
        
    #     response_1 = answer_sync(prompt=prompt_1, preface=self.preface_cond, **kwargs)
    #     response_2 = answer_sync(prompt=prompt_2, preface=self.preface_cond, **kwargs)
    #     response_3 = answer_sync(prompt=prompt_3, preface=self.preface_and, **kwargs)
    #     response_4 = answer_sync(prompt=prompt_4, preface=self.preface_and, **kwargs)
    #     sentences = {"P": base_sentence_1, "Q|P": response_1, "R|(P and Q)": response_2, "P and Q and R": response_3}
    #     return sentences