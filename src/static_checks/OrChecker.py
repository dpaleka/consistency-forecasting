"""Where f(x) is the model,
R(x1, x2, x3)
    := x1 == P; x2 == Q, x3 = P or Q
S(f(x1), f(x2), f(x3))
    :=  max(f(x1), f(x2)) <= f(x3) <= min(1, f(x1) + f(x2))
"""

from common.llm_utils import answer_sync, answer
from forecasters import SentencesTemplate, ProbsTemplate
from .BaseChecker import BaseChecker

class OrChecker(BaseChecker):
    preface = (
        "You are a helpful assistant. I will give you two questions, q1 and q2. "
        "I want you to give me a single question that expresses 'q1 or q2' "
        "in the most succint manner possible. Make sure your response is in the ",
        "form of a single question, and exactly expresses 'q1 or q2'.",
    )

    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)

    def instantiate_sync(
        self, base_sentence_1: str, base_sentence_2: str, **kwargs
    ) -> SentencesTemplate:
        prompt = self.stack(base_sentence_1, base_sentence_2)
        response = answer_sync(prompt=prompt, preface=self.preface, **kwargs)
        sentences = {"P": base_sentence_1, "Q": base_sentence_2, "P_or_Q": response}
        return sentences

    async def instantiate(
        self, base_sentence_1: str, base_sentence_2: str, **kwargs
    ) -> SentencesTemplate:
        prompt = self.stack(base_sentence_1, base_sentence_2)
        response = await answer(prompt=prompt, preface=self.preface, **kwargs)
        sentences = {"P": base_sentence_1, "Q": base_sentence_2, "P_or_Q": response}
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return max(
            max(answers["P"], answers["Q"]) - answers["P_or_Q"],
            answers["P_or_Q"] - min(1, answers["P"] + answers["Q"]),
        )

