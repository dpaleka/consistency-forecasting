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

# define type probability
from typing import Callable
from src.common.llm_utils import query_api_chat, query_api_text
from src.instantiators.negation import negate_simple

class Prob(float):
    def __new__(cls, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Probability must be between 0 and 1.")
        return super(Prob, cls).__new__(cls, value)



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

# Example usage
BASE_QS = [
    "What is the probability that the Democratic party will win the US Presidential election in 2024?",
    "What is the probability that Ebola will be eradicated by 2030?",
]
negation_checks = [
    negation_template(q, negate_simple, "A -> B", negation_checker)
    for q in BASE_QS
]


# %%
print("AAA")

# %%
