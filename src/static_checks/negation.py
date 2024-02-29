# Path: static_checks/BaseChecker.py
import sys
from common.llm_utils import query_api_chat_sync
from common.forecasters import SentencesTemplate, ProbsTemplate
from static_checks.Base import BaseChecker

sys.path.append("..")


class NegationChecker(BaseChecker):
    def instantiate(self, base_sentence: str) -> SentencesTemplate:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. I need you to negate the question provided.  This should be done by adding / removing the word 'not' whenever possible.  Demorgan's laws should be followed with and/or negation.  It should return a question. Avoid using the word won't.",
            },
            {"role": "user", "content": base_sentence},
        ]
        response = query_api_chat_sync(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0.0,
        )
        sentences = {"P": base_sentence, "notP": response}
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(answers["P"] + answers["notP"] - 1)
