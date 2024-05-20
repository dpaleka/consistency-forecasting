import itertools as it
import jsonlines
import asyncio


# MODEL = "gpt-3.5-turbo"
MODEL = "gpt-4-turbo"
# MODEL = 'gpt-4o'

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

BASE_DATA_PATH: Path = get_data_path() / "tuples/"

checkers: dict[str, Checker] = {
    "NegChecker": NegChecker(path=BASE_DATA_PATH / "NegChecker.jsonl"),
    "AndChecker": AndChecker(path=BASE_DATA_PATH / "AndChecker.jsonl"),
    "OrChecker": OrChecker(path=BASE_DATA_PATH / "OrChecker.jsonl"),
    "AndOrChecker": AndOrChecker(path=BASE_DATA_PATH / "AndOrChecker.jsonl"),
    "ButChecker": ButChecker(path=BASE_DATA_PATH / "ButChecker.jsonl"),
    "CondChecker": CondChecker(path=BASE_DATA_PATH / "CondChecker.jsonl"),
    "ConsequenceChecker": ConsequenceChecker(
        path=BASE_DATA_PATH / "ConsequenceChecker.jsonl"
    ),
    "ParaphraseChecker": ParaphraseChecker(
        path=BASE_DATA_PATH / "ParaphraseChecker.jsonl"
    ),
    "SymmetryAndChecker": SymmetryAndChecker(
        path=BASE_DATA_PATH / "SymmetryAndChecker.jsonl"
    ),
    "SymmetryOrChecker": SymmetryOrChecker(
        path=BASE_DATA_PATH / "SymmetryOrChecker.jsonl"
    ),
    "CondCondChecker": CondCondChecker(path=BASE_DATA_PATH / "CondCondChecker.jsonl"),
}

async def instantiate(path, checker_list=checkers, n_relevance=10, length=3):
    bqs = []
    for line in jsonlines.open(path):
        try:
            bq = ForecastingQuestion(**line)
            bqs.append(bq)
        except Exception as e:
            print(e)
            continue
            
    possible_tuples = {
        1: [{"P": P} for P in bqs],
        2: [{"P": P, "Q": Q} for P, Q in it.combinations(bqs, 2)],
        3: [{"P": P, "Q": Q, "R": R} for P, Q, R in it.combinations(bqs, 3)],
    }
    
    for i, possible_ituples in possible_tuples.items():
        random.shuffle(possible_ituples)
        possible_ituples = possible_ituples[:n_relevance]
        if i > 1:
            tasks = [relevance(tup) for tup in possible_ituples]
            relevances = await asyncio.gather(*tasks)
            possible_ituples = list(zip(possible_ituples, relevances))
            possible_ituples.sort(key=lambda x: x[1]["relevance"]["score"], reverse=True)
    
    for checker in checkers.values():
        await checker.instantiate_and_write_many(
            possible_tuples[checker.num_base_questions][:length],
            model=MODEL,
            overwrite=True,
            validate_before=True,
            n_validation=3,
        )


asyncio.run(
    instantiate(
        path=get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
    )
)
