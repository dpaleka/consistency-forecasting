# %%

from forecasters.consistent_forecaster import ConsistentForecaster
from forecasters.basic_forecaster import BasicForecaster
from common.datatypes import ForecastingQuestion
from static_checks.Checker import *
from common.llm_utils import query_api_chat_sync
from time import time


#%%
# Create a sample ForecastingQuestion
fq = ForecastingQuestion(
    title="Will Manhattan have a skyscraper a mile tall by 2030?",
    body=(
        "Resolves YES if at any point before 2030, there is at least "
        "one building in the NYC Borough of Manhattan (based on current "
        "geographic boundaries) that is at least a mile tall."
    ),
    resolution_date="2030-01-01T00:00:00",
    question_type="binary",
    created_date=None,
    data_source="manifold",
    url="https://www.metaculus.com/questions/12345/",
    metadata={"foo": "bar"},
    resolution=None,
)

#%%
# Run BasicForecaster
basic_forecaster = BasicForecaster(model="gpt-4o-mini")

# Print logging module current setup
import logging

print("Current logging configuration:")
print(f"Logging level: {logging.getLogger().getEffectiveLevel()}")
print(f"Logging handlers: {logging.getLogger().handlers}")
print(f"Logging formatter: {logging.getLogger().handlers[0].formatter if logging.getLogger().handlers else 'No formatter'}")

# Call the BasicForecaster synchronously
sync_result = basic_forecaster.call_full(fq)
print(f"Synchronous BasicForecaster result: {sync_result}")

exit()
#%%
cf = ConsistentForecaster(
    BasicForecaster(),
    checks=[
        NegChecker(),
        AndOrChecker(),
        CondChecker(),
        ParaphraseChecker(),
    ],
)
import asyncio

x = asyncio.run(cf.call_async_full(
    fq,
    bq_func_kwargs={"model": "gpt-4o-mini"},
    instantiation_kwargs={"model": "gpt-4o-mini"},
    model="gpt-4o-mini",
))

# %%
print(f"{x=}")

raise
# %%

from static_checks.Checker import *
from time import time

checkers = {
    "para": {"answers": {"P": 0.4, "para_P": 0.43}, "checker": ParaphraseChecker()},
    "neg": {"answers": {"P": 0.6, "not_P": 0.57}, "checker": NegChecker()},
    "cond": {
        "answers": {"P": 0.15, "Q_given_P": 0.6, "P_and_Q": 0.15},
        "checker": CondChecker(),
    },
    "andor": {
        "answers": {"P": 0.6, "Q": 0.6, "P_and_Q": 0.55, "P_or_Q": 0.6},
        "checker": AndOrChecker(),
    },
    "but": {
        "answers": {"P": 0.6, "Q_and_not_P": 0.3, "P_or_Q": 0.94},
        "checker": ButChecker(),
    },
    "condcond": {
        "answers": {
            "P": 0.5,
            "Q_given_P": 0.5,
            "R_given_P_and_Q": 0.5,
            "P_and_Q_and_R": 0.16,
        },
        "checker": CondCondChecker(),
    },
    "cons": {"answers": {"P": 0.5, "cons_P": 0.45}, "checker": ConsequenceChecker()},
}

for k, v in checkers.items():
    checker = v["checker"]
    answers = v["answers"]
    if k != "cons":
        continue
    print("Checker:", k)
    time0 = time()
    result_shgo = checker.max_min_arbitrage(answers, methods=("shgo",))
    time1 = time()
    print("SHGO:", result_shgo, "\nTime:", time1 - time0)
    result_diff_evol = checker.max_min_arbitrage(
        answers, methods=("differential_evolution",)
    )
    time2 = time()
    print("Diff Evolution:", result_diff_evol, "\nTime:", time2 - time1)
    result_dual_annealing = checker.max_min_arbitrage(
        answers, methods=("dual_annealing",)
    )
    time3 = time()
    print("Dual Annealing:", result_dual_annealing, "\nTime:", time3 - time2)
    result_basinhopping = checker.max_min_arbitrage(answers, methods=("basinhopping",))
    time4 = time()
    print("Basinhopping:", result_basinhopping, "\nTime:", time4 - time3)
    result_plain = checker.max_min_arbitrage(answers)
    time5 = time()
    print("No kwargs:", result_plain, "\nTime:", time5 - time4)
    print("\n\n")


# print('----------------')
# print(CondChecker().min_arbitrage(ex, exa))
# print('----------------')
# print(CondChecker().arbitrage(outcome1, ex, exa))
# print(CondChecker().arbitrage(outcome2, ex, exa))
# print(CondChecker().arbitrage(outcome3, ex, exa))
# print('----------------')

# %%
import json
from common.datatypes import (
    ForecastingQuestion_stripped,
    ForecastingQuestion,
    Prob_cot,
    Prob,
    PlainText,
)
from common.llm_utils import query_api_chat_sync, query_api_chat_sync_native
import os

fq = ForecastingQuestion(
    title="Will Manhattan have a skyscraper a mile tall by 2030?",
    body=(
        "Resolves YES if at any point before 2030, there is at least "
        "one building in the NYC Borough of Manhattan (based on current "
        "geographic boundaries) that is at least a mile tall."
    ),
    resolution_date="2030-01-01T00:00:00",
    question_type="binary",
    data_source="manifold",
    url="https://www.metaculus.com/questions/12345/",
    metadata={"foo": "bar"},
    resolution=None,
)

fqs = ForecastingQuestion_stripped(
    title="Will Manhattan have a skyscraper a mile tall by 2030?",
    body=(
        "Resolves YES if at any point before 2030, there is at least "
        "one building in the NYC Borough of Manhattan (based on current "
        "geographic boundaries) that is at least a mile tall."
    ),
)

print(fqs.__str__())


# %%

# os.environ["USE_OPENROUTER"] = "True"
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Summarize the question for the user.",
    },
    {"role": "user", "content": fq.__str__()},
]
# response = query_api_chat_sync(messages=messages, verbose=True, model="mistralai/mistral-large")
