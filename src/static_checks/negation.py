"""
Formal constraint:
“desc”: “A -> B”
“probs_checker”: def check_condition(ans_A : Prob, ans_B : Prob):
return ans_A < ans_B
“probs_violation”: def viol_score(ans_A : Prob, ans_B):
return max(0, ans_A - ans_B)

def negate_simple(ques: str) -> str:
cons_check = {“tup”:  (“Will Dems win election in 2024”, make_negative_q(ques_str)), “desc”: ...,  “checker”: NegationChecker, )
cons_check_template = {“tuple_generator”: Callable = make_negative_q, “desc”: ...,  “output_parser”: Callable : str -> float, “probs_checker”: NegationChecker, )

make a class and subclass it for the negation thing
"""

# Path: static_checks/negation.py

from abc import ABC, abstractmethod
from typing import Type, Dict, NamedTuple, Callable
# path imports
import sys

sys.path.append("..")

from instantiators.negation import negate_simple
from common.llm_utils import query_api_chat_sync

class Prob(float):
    def __new__(cls, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return super(Prob, cls).__new__(cls, value)

Forecaster = Callable[[str], Prob]
SentencesTemplate = Dict[str, str]
ProbsTemplate = Dict[str, Prob]

def elicit(forecaster : Forecaster, sentences : SentencesTemplate) -> ProbsTemplate:
    return {k : forecaster(v) for k, v in sentences.items()}

def gpt4caster (sentence : str) -> Prob:
    messages = [
        {
            "role" : "system",
            "content" : "You are an informed and well-calibrated forecaster. I need you to give me your best probability estimate for the following sentence or question resolving YES. Your answer should be a float between 0 and 1, with nothing else in your response."
        },
        {
            "role" : "user",
            "content" : sentence
        }
    ]
    response = query_api_chat_sync(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.0,
    )
    return float(response)

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



########### old code

def negation_violation(ans_A: Prob, ans_B: Prob) -> float:
    """
    Returns the difference between the two.
    """
    return abs(ans_A + ans_B - 1)


def negation_checker(ans_A: Prob, ans_B: Prob, tolerance: float = 0.1) -> bool:
    """
    Checks if the two sum up to 1.
    """
    return negation_violation(ans_A, ans_B) < tolerance


def instantiate(base_q: str) -> tuple[str]:
    """
    Instantiates a negation constraint.
    """
    return (base_q, negate_simple(base_q))


negation_template = {
    "tuple_generator": instantiate,
    "desc": "A -> B",
    "violation_scorer": negation_violation,
    "probs_checker": negation_checker,
}
