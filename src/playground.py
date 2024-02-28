#%%
from static_checks.negation import negation_template, negate_simple, negation_checker
from common.models import *
from static_checks.checkers import *

# Instantiate some base questions
BASE_QS = [
    "What is the probability that the Democratic party will win the US Presidential election in 2024?",
    "What is the probability that Ebola will be eradicated by 2030?",
]

# instantiate the negation checks
negation_check_qtuples = [
    negation_template["tuple_generator"](q)
    for q in BASE_QS
]

# Now call some LLM method on those qs to get answers
answers = [[0.5, 0.5], [0.6, 0.4]]
pass

# Now check the answers
violations = [
    negation_template["violation_scorer"](*ans)
    for ans in answers
]
passed = [
    negation_template["probs_checker"](*ans)
    for ans in answers
]

t=NegationChecker()
u = t.instantiate_and_elicit_and_check(gpt4caster, BASE_QS[1])
print(u)

# Now we can make a report
import matplotlib.pyplot as plt
pass
