import jsonlines
import asyncio
import click

# from static_checks.MiniInstantiator import MiniInstantiator
from static_checks import Checker
from static_checks.Checker import (
    choose_checkers,
)
from static_checks.tuple_relevance import relevance
from common.datatypes import ForecastingQuestion
from common.path_utils import get_data_path
from pathlib import Path
from common.llm_utils import parallelized_call
import functools
import random

# The following are defaults, but can be overriden in the script args
MODEL = "gpt-4o-mini-2024-07-18"
MODEL_RELEVANCE = "gpt-4o-mini-2024-07-18"
BASE_DATA_PATH: Path = (
    get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
)
# BASE_DATA_PATH: Path = (
#    get_data_path() / "fq" / "synthetic" / "high-quality-questions--all-domains.jsonl"
# )
# TUPLES_PATH: Path = get_data_path() / "tuples_playground/"
TUPLES_PATH: Path = get_data_path() / "tuples/"
# TUPLES_PATH: Path = get_data_path() / "tuples_synthetic"
RELEVANT_CHECKS = ["ConsequenceChecker"]


async def instantiate(
    BASE_DATA_PATH: Path,
    checker_list: dict[str, Checker],
    n_relevance: int = 10,
    n_write: int = -1,
    model: str = MODEL,
    model_relevance: str = MODEL_RELEVANCE,
    seed: int = 42,
    **kwargs,
):
    """
    Tests n_relevance potential combinations for relevance, and sorts by relevance score.
    Writes the n_write tuples to the Checker.
    Checker stops instantiating after n_write tuples have successfully passed verification.

    Args:
        BASE_DATA_PATH (Path): path to a jsonl file of ForecastingQuestions
        checker_list (dict[str, Checker]): dictionary of Checkers to instantiate with
        n_relevance (int, optional): _description_. number of possible tuples to test for relevance
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

    random.seed(seed)

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
            func = functools.partial(relevance, model=model_relevance)
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
            possible_tuples[checker.num_base_questions],
            model=model,
            n_write=n_write,
            overwrite=True,
            n_verification=3,
            **kwargs,
        )


@click.command()
@click.option("--data_path", "-d", type=click.Path(exists=True), default=BASE_DATA_PATH)
@click.option("--n_relevance", default=1000, help="Number of relevance samples.")
@click.option("--n_write", default=100, help="Number of writes.")
@click.option(
    "--model_main",
    default=MODEL,
    help="Model to use for instantiation and verification.",
)
@click.option(
    "--model_relevance",
    default=MODEL_RELEVANCE,
    help="Model to use for relevance scoring.",
)
@click.option(
    "--relevant_checks",
    "-k",
    default=RELEVANT_CHECKS,
    multiple=True,
    help='Relevant checks to perform. In case of "all", all checkers are used.',
)
@click.option(
    "--tuple_dir",
    "-t",
    type=click.Path(exists=True),
    default=TUPLES_PATH,
    help="Directory to read tuples from.",
)
@click.option(
    "--seed",
    default=42,
    help="Seed for reproducibility. Controls random sampling, not necessarily any randomness in external model calls.",
)
def main(
    data_path,
    n_relevance,
    n_write,
    model_main,
    model_relevance,
    relevant_checks,
    tuple_dir,
    seed,
):
    checkers = choose_checkers(relevant_checks, tuple_dir)
    asyncio.run(
        instantiate(
            BASE_DATA_PATH=data_path,
            checker_list=checkers,
            n_relevance=n_relevance,
            n_write=n_write,
            model=model_main,
            model_relevance=model_relevance,
            seed=seed,
        )
    )


if __name__ == "__main__":
    main()
