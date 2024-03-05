#%%
from static_checks.negation import *
from forecasters import *

# Instantiate some base questions
BASE_QS = [
    "What is the probability that the Democratic party will win the US Presidential election in 2024?",
    "What is the probability that Ebola will be eradicated by 2030?",
]

u = NegationChecker().instantiate_and_elicit_and_check(BasicForecaster(), BASE_QS[1])
print(u)

# Now we can make a report
import matplotlib.pyplot as plt
pass

# %%
