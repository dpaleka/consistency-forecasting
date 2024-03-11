"""Where f(x) is the model,
R(x1, x2)
    := x1 => x2
S(f(x1), f(x2))
    :=  f(x1) <= f(x2)
"""

from common.llm_utils import answer_sync, answer
from forecasters import SentencesTemplate, ProbsTemplate
from .BaseChecker import BaseChecker

class ImpChecker(BaseChecker):
    preface = (
        "You are a helpful assistant. I will give you a question. "
        "I want you to give me a guaranteed logical implication of that question, "
        "should it be true. Make sure your response is in the form of another question. "
    )
    
    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)
    
    def instantiate_sync(self, base_sentence: str, **kwargs) -> SentencesTemplate:
        response = answer_sync(
            prompt=base_sentence,
            preface=self.preface,
            **kwargs
        )
        sentences = {"P": base_sentence, "consequence_P": response}
        return sentences
    
    async def instantiate(self, base_sentence: str, **kwargs) -> SentencesTemplate:
        response = await answer(
            prompt=base_sentence,
            preface=self.preface,
            **kwargs
        )
        sentences = {"P": base_sentence, "consequence_P": response}
        return sentences
    
    def violation(self, answers: ProbsTemplate) -> float:
        return abs(answers["P"] - answers["consequence_P"])