from typing import Dict, Callable
from common.llm_utils import query_api_chat_sync


class Prob(float):
    def __new__(cls, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return super(Prob, cls).__new__(cls, value)


Forecaster = Callable[[str], Prob]
SentencesTemplate = Dict[str, str]
ProbsTemplate = Dict[str, Prob]


def elicit(forecaster: Forecaster, sentences: SentencesTemplate) -> ProbsTemplate:
    return {k: forecaster(v) for k, v in sentences.items()}


def gpt4caster(sentence: str) -> Prob:
    messages = [
        {
            "role": "system",
            "content": "You are an informed and well-calibrated forecaster. I need you to give me your best probability estimate for the following sentence or question resolving YES. Your answer should be a float between 0 and 1, with nothing else in your response.",
        },
        {"role": "user", "content": sentence},
    ]
    response = query_api_chat_sync(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.0,
    )
    return float(response)
