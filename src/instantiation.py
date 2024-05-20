import itertools as it
import jsonlines
import asyncio

# from static_checks.MiniInstantiator import MiniInstantiator
from static_checks import Checker
from static_checks.Checker import (
    NegChecker,
    AndChecker,
    OrChecker,
    AndOrChecker,
    ButChecker,
    CondChecker,
    ConsequenceChecker,
    ParaphraseChecker,
    SymmetryAndChecker,
    SymmetryOrChecker,
    CondCondChecker,
)
from static_checks.tuple_relevance import relevance
from common.datatypes import ForecastingQuestion
from common.path_utils import get_data_path
from pathlib import Path
import random

# MODEL = "gpt-3.5-turbo"
MODEL = "gpt-4-turbo"
MODEL_RELEVANCE = "gpt-4o"
# MODEL = 'gpt-4o'
BASE_DATA_PATH: Path = (
    get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
)
TUPLES_PATH: Path = get_data_path() / "tuples/"

checkers: dict[str, Checker] = {
    "NegChecker": NegChecker(path=TUPLES_PATH / "NegChecker.jsonl"),
    "AndChecker": AndChecker(path=TUPLES_PATH / "AndChecker.jsonl"),
    "OrChecker": OrChecker(path=TUPLES_PATH / "OrChecker.jsonl"),
    "AndOrChecker": AndOrChecker(path=TUPLES_PATH / "AndOrChecker.jsonl"),
    "ButChecker": ButChecker(path=TUPLES_PATH / "ButChecker.jsonl"),
    "CondChecker": CondChecker(path=TUPLES_PATH / "CondChecker.jsonl"),
    "ConsequenceChecker": ConsequenceChecker(
        path=TUPLES_PATH / "ConsequenceChecker.jsonl"
    ),
    "ParaphraseChecker": ParaphraseChecker(
        path=TUPLES_PATH / "ParaphraseChecker.jsonl"
    ),
    "SymmetryAndChecker": SymmetryAndChecker(
        path=TUPLES_PATH / "SymmetryAndChecker.jsonl"
    ),
    "SymmetryOrChecker": SymmetryOrChecker(
        path=TUPLES_PATH / "SymmetryOrChecker.jsonl"
    ),
    "CondCondChecker": CondCondChecker(path=TUPLES_PATH / "CondCondChecker.jsonl"),
}


async def instantiate(path, checker_list, n_relevance=10, length=3):
    bqs = []
    print(f"Loading questions from {path}...")
    for line in jsonlines.open(path):
        try:
            bq = ForecastingQuestion(**line)
            bqs.append(bq)
        except Exception as e:
            print(e)
            continue
    print(f"Loaded {len(bqs)} questions.")

    possible_tuples = {} # {i: list of i-tuples}
    for i in [1, 2, 3]:
        if i > len(bqs):
            break

        print(f"Handling {i}-tuples...")
        sampled_tuples = [random.sample(bqs, i) for _ in range(n_relevance)]
        possible_ituples = [
            {chr(80 + j): tup[j] for j in range(i)} for tup in sampled_tuples
        ]

        if i > 1:
            print("Setting task to get relevance scores ...")
            tasks = [relevance(tup, model=MODEL_RELEVANCE) for tup in possible_ituples]
            print("Getting relevance scores ...")
            relevances = await asyncio.gather(*tasks)
            print("Sorting by relevance scores ...")
            possible_ituples = list(zip(possible_ituples, relevances))
            possible_ituples.sort(
                key=lambda x: x[1]["relevance"]["score"], reverse=True
            )
        
        possible_tuples[i] = possible_ituples

    for checker in checkers.values():
        print(f"Instantiating and writing {checker.__class__.__name__}")
        await checker.instantiate_and_write_many(
            possible_tuples[checker.num_base_questions][:length],
            model=MODEL,
            overwrite=True,
            verify_before=True,
            n_verification=3,
        )


asyncio.run(
    instantiate(path=BASE_DATA_PATH, checker_list=checkers, n_relevance=10, length=3)
)
