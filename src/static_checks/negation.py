""" Where f(x) is the model,
R(x1, x2)       :=  x2 == ¬x1
S(f(x1), f(x2)) :=  f(x1) + f(x2) = 1
"""

from common.llm_utils import answer_sync, answer
from forecasters import SentencesTemplate, ProbsTemplate
from .Base import BaseChecker


class NegationChecker(BaseChecker):
    preface = " ".join([
        "You are a helpful assistant. I need you to negate the question provided.",
        "This should be done by adding / removing the word 'not' whenever possible.",
        "Demorgan's laws should be followed with and/or negation. It should return a",
        "question. Avoid using the word won't."])
    
    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(self, base_sentence: str, **kwargs) -> SentencesTemplate:
        response = answer_sync(prompt = base_sentence, preface = self.preface, **kwargs)
        sentences = {"P": base_sentence, "notP": response}
        return sentences

    async def instantiate(self, base_sentence: str, **kwargs) -> SentencesTemplate:
        response = await answer(prompt = base_sentence, preface = self.preface, **kwargs)
        sentences = {"P": base_sentence, "notP": response}
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(answers["P"] + answers["notP"] - 1)
