from common.llm_utils import query_api_chat_sync, query_api_chat
from common.forecasters import SentencesTemplate, ProbsTemplate
from .Base import BaseChecker


class NegationChecker(BaseChecker):

    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)
        self.first_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. I need you to negate the question provided.  This should be done by adding / removing the word 'not' whenever possible.  Demorgan's laws should be followed with and/or negation.  It should return a question. Avoid using the word won't.",
            },
        ]

    def instantiate(self, base_sentence: str, model = "gpt-4-1106-preview") -> SentencesTemplate:
        messages = self.first_messages + [{"role": "user", "content": base_sentence}] 
        response = query_api_chat_sync(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        sentences = {"P": base_sentence, "notP": response}
        return sentences

    async def instantiate_async(self, base_sentence: str, model = "gpt-4-1106-preview") -> SentencesTemplate:
        messages = self.first_messages + [{"role": "user", "content": base_sentence}] 
        response = await query_api_chat(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        sentences = {"P": base_sentence, "notP": response}
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(answers["P"] + answers["notP"] - 1)
