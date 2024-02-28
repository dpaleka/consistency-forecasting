# Path: static_checks/checkers.py

from abc import ABC, abstractmethod
from typing import Dict, Callable
# path imports
import sys

sys.path.append("..")

from common.llm_utils import query_api_chat_sync
from common.models import Prob, Forecaster, SentencesTemplate, ProbsTemplate

def elicit(forecaster : Forecaster, sentences : SentencesTemplate) -> ProbsTemplate:
    return {k : forecaster(v) for k, v in sentences.items()}

class ConsistencyChecker(ABC):
    
    def __init__(self, tolerance = 0.1):
        self.tolerance = tolerance

    @abstractmethod
    def instantiate(self, *base_sentences : str) -> SentencesTemplate:
        pass
    
    @abstractmethod
    def violation(self, answers : ProbsTemplate) -> float:
        pass

    def check(self, answers : ProbsTemplate) -> bool:
        return self.violation(answers) < self.tolerance

    def instantiate_and_elicit(self, forecaster : Callable[[str], Prob], *base_sentences : str) -> ProbsTemplate:
        return elicit(forecaster, self.instantiate(*base_sentences))
    
    def elicit_and_violation(self, forecaster : Callable[[str], Prob], sentences : SentencesTemplate) -> float:
        return self.violation(elicit(forecaster, sentences))

    def elicit_and_check(self, forecaster : Callable[[str], Prob], sentences : SentencesTemplate) -> bool:
        return self.check(elicit(forecaster, sentences))
    
    def instantiate_and_elicit_and_violation(self, forecaster : Callable[[str], Prob], *base_sentences : str) -> float:
        return self.violation(self.instantiate_and_elicit(forecaster, *base_sentences))
    
    def instantiate_and_elicit_and_check(self, forecaster : Callable[[str], Prob], *base_sentences : str) -> float:
        return self.check(self.instantiate_and_elicit(forecaster, *base_sentences))

class NegationChecker(ConsistencyChecker):
    
    def __init__(self, tolerance=0.1):
        super().__init__(tolerance)
    
    def instantiate(self, base_sentence: str) -> Dict[str, str]:
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
        sentences = {
            "P" : base_sentence,
            "notP" : response
        }
        return sentences

    def violation(self, answers: ProbsTemplate) -> float:
        return abs(answers["P"] + answers["notP"] - 1)
