#%%
from src.static_checks.NegChecker import *
from forecasters import *

# Instantiate some base questions
BASE_QS = [
    "What is the probability that the Democratic party will win the US Presidential election in 2024?",
    "What is the probability that Ebola will be eradicated by 2030?",
]

q_tuple = NegChecker().instantiate_sync(BASE_QS[1])
print(q_tuple)

# now use forecasters
from forecasters import ConsistentAskForecaster
f = ConsistentAskForecaster()
answers = {}
for k, v in q_tuple.items():
    answers[k] = f.call(v)

violation = NegChecker().violation(answers)
print(violation)



# %%
