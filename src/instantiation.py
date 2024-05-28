import jsonlines
import asyncio

# from static_checks.MiniInstantiator import MiniInstantiator
import os
from static_checks import Checker
from static_checks.Checker import (
    NegChecker,  # noqa
    AndChecker,  # noqa
    OrChecker,  # noqa
    AndOrChecker,  # noqa
    ButChecker,  # noqa
    CondChecker,  # noqa
    ConsequenceChecker,  # noqa
    ParaphraseChecker,  # noqa
    SymmetryAndChecker,  # noqa
    SymmetryOrChecker,  # noqa
    CondCondChecker,  # noqa
)
from static_checks.tuple_relevance import relevance
from common.datatypes import ForecastingQuestion
from common.path_utils import get_data_path
from pathlib import Path
from common.llm_utils import parallelized_call
import functools
import random

# MODEL = "gpt-3.5-turbo"
MODEL = "gpt-4o"
MODEL_RELEVANCE = "gpt-4o"
# MODEL = 'gpt-4o'
BASE_DATA_PATH: Path = (
    get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
)

use_examples = os.getenv("USE_EXAMPLES", "False") == "True"
if use_examples:
    TUPLES_PATH: Path = get_data_path() / "tuples_alt/"
else:
    TUPLES_PATH: Path = get_data_path() / "tuples_alt_noex/"

checkers: dict[str, Checker] = {
    # "NegChecker": NegChecker(path=TUPLES_PATH / "NegChecker.jsonl"),
    # "AndChecker": AndChecker(path=TUPLES_PATH / "AndChecker.jsonl"),
    # "OrChecker": OrChecker(path=TUPLES_PATH / "OrChecker.jsonl"),
    # "AndOrChecker": AndOrChecker(path=TUPLES_PATH / "AndOrChecker.jsonl"),
    # "ButChecker": ButChecker(path=TUPLES_PATH / "ButChecker.jsonl"),
    # "CondChecker": CondChecker(path=TUPLES_PATH / "CondChecker.jsonl"),
    "ConsequenceChecker": ConsequenceChecker(
        path=TUPLES_PATH / "ConsequenceChecker.jsonl"
    ),
    # "ParaphraseChecker": ParaphraseChecker(
    #     path=TUPLES_PATH / "ParaphraseChecker.jsonl"
    # ),
    # "SymmetryAndChecker": SymmetryAndChecker(
    #     path=TUPLES_PATH / "SymmetryAndChecker.jsonl"
    # ),
    # "SymmetryOrChecker": SymmetryOrChecker(
    #     path=TUPLES_PATH / "SymmetryOrChecker.jsonl"
    # ),
    # "CondCondChecker": CondCondChecker(path=TUPLES_PATH / "CondCondChecker.jsonl"),
}


async def instantiate(
    BASE_DATA_PATH: Path,
    checker_list: dict[str, Checker],
    n_relevance: int = 10,
    n_top_relevance: int = 3,
    n_write: int = -1,
    **kwargs,
):
    """
    Tests n_relevance potential combinations for relevance, and sorts by relevance score.
    Writes the top n_top_relevance tuples to the Checker.
    Checker stops instantiating after n_write tuples have successfully passed verification.

    Args:
        BASE_DATA_PATH (Path): path to a jsonl file of ForecastingQuestions
        checker_list (dict[str, Checker]): dictionary of Checkers to instantiate with
        n_relevance (int, optional): _description_. number of possible tuples to test for relevance
        n_top_relevance (int, optional): _description_. top n relevant possible tuples;
            usually > n_write because some might fail verification
        n_write (int, optional): _description_. max number of tuples we actually want to write.
            Leave as -1 to write all tuples that pass verification
    """
    bqs = []
    print(f"Loading questions from {BASE_DATA_PATH}...")
    for line in jsonlines.open(BASE_DATA_PATH):
        try:
            bq = ForecastingQuestion(**line)
            bqs.append(bq)
        except Exception as e:
            print(e)
            continue
    print(f"Loaded {len(bqs)} questions.")

    possible_tuples = {}  # {i: list of i-tuples}
    i_set = {checker.num_base_questions for checker in checker_list.values()}
    for i in i_set:
        if i > len(bqs):
            break

        print(f"Handling {i}-tuples...")
        sampled_tuples = [random.sample(bqs, i) for _ in range(n_relevance)]
        possible_ituples = [
            {chr(80 + j): tup[j] for j in range(i)} for tup in sampled_tuples
        ]

        if i > 1:
            print("Setting task to get relevance scores ...")

            print("Getting relevance scores ...")
            func = functools.partial(relevance, model=MODEL_RELEVANCE)
            relevances = await parallelized_call(
                func=func, data=possible_ituples, max_concurrent_queries=25
            )
            print("Sorting by relevance scores ...")
            possible_ituples = list(zip(possible_ituples, relevances))
            possible_ituples.sort(
                key=lambda x: x[1]["relevance"]["score"], reverse=True
            )

        possible_tuples[i] = possible_ituples

    for checker in checker_list.values():
        print(f"Instantiating and writing {checker.__class__.__name__}")
        await checker.instantiate_and_write_many(
            possible_tuples[checker.num_base_questions][:n_top_relevance],
            model=MODEL,
            n_write=n_write,
            overwrite=True,
            n_verification=3,
            **kwargs,
        )


# this should probably go in scripts
if __name__ == "__main__":
    asyncio.run(
        instantiate(
            BASE_DATA_PATH=BASE_DATA_PATH,
            checker_list=checkers,
            n_relevance=50,
            n_top_relevance=16,
            n_write=10,
        )
    )
