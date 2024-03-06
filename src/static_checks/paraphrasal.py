""" Where f(x) is the model,
R(x1, x2)       :=  x2 == paraphrase(x1)
S(f(x1), f(x2)) :=  f(x1) == f(x2)
"""

from common.llm_utils import answer, answer_sync
from forecasters import SentencesTemplate, ProbsTemplate
from .Base import BaseChecker


class ParaphrasalChecker(BaseChecker):
    preface = " ".join([
        "You are a helpful assistant. I need you to paraphrase the question provided.",
        "Make sure the resulting question still means exactly the same thing."])    
    
    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate(self, base_sentence: str, **kwargs) -> SentencesTemplate:
        response = answer_sync(prompt = base_sentence, preface = self.preface, **kwargs)
        sentences = {"P": base_sentence, "P_alt": response}
        return sentences

    async def instantiate_async(self, base_sentence: str, **kwargs) -> SentencesTemplate:
        response = await answer(prompt = base_sentence, preface = self.preface, **kwargs)
        sentences = {"P": base_sentence, "P_alt": response}
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(answers["P"] - answers["P_alt"])
