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
import itertools

# The following are defaults, but can be overriden in the script args
MODEL = "gpt-4o-mini-2024-07-18"
MODEL_RELEVANCE = "gpt-4o-mini-2024-07-18"
# BASE_DATA_PATH: Path = (
#     get_data_path() / "fq" / "real" / "questions_cleaned_formatted.jsonl"
# )

BASE_DATA_PATH: Path = (
    get_data_path() / "fq" / "synthetic" / "from-related-verified.jsonl"
)
# BASE_DATA_PATH: Path = (
#     get_data_path() / "fq" / "synthetic" / "high-quality-questions--all-domains.jsonl"
# )
# TUPLES_PATH: Path = get_data_path() / "tuples_playground/"
TUPLES_PATH: Path = get_data_path() / "tuples_rel/"
# TUPLES_PATH: Path = get_data_path() / "tuples_synthetic"
RELEVANT_CHECKS = ["AndChecker", "CondCondChecker", "NegChecker"]
# RELEVANT_CHECKS = ["AndChecker"]


async def instantiateRel(
    BASE_DATA_PATH: Path,
    checker_list: dict[str, Checker],
    n_write: int = -1,
    model: str = "gpt-4o-mini-2024-07-18",
    seed: int = 42,
    **kwargs,
):
    print(f"Loading questions from {BASE_DATA_PATH}...")
    question_sets = {}
    total_questions = 0
    source_questions_count = 0
    related_questions_count = 0
    for line in jsonlines.open(BASE_DATA_PATH):
        try:
            bq = ForecastingQuestion(**line)
            total_questions += 1
            source_question = bq.metadata.get("source_question")
            if source_question is None:
                source_questions_count += 1
                question_sets[bq.title] = {"source": bq, "related": []}
            else:
                related_questions_count += 1
                if source_question not in question_sets:
                    question_sets[source_question] = {"source": None, "related": []}
                question_sets[source_question]["related"].append(bq)
        except Exception as e:
            print(f"Error processing question: {e}")
            continue

    print(f"\nLoaded {total_questions} questions in total.")
    print(
        f"Found {source_questions_count} source questions and {related_questions_count} related questions."
    )
    print(f"Created {len(question_sets)} question sets.")

    valid_sets = {k: v for k, v in question_sets.items() if v["source"] is not None}
    print(
        f"\nFound {len(valid_sets)} valid question sets with both source and related questions."
    )

    random.seed(seed)

    possible_tuples = {}
    order_sensitive_checkers = ["ButChecker", "CondChecker", "CondCondChecker"]

    for checker_name, checker in checker_list.items():
        print(f"\nProcessing {checker_name}:")
        num_base_questions = checker.num_base_questions

        if checker_name in order_sensitive_checkers:
            possible_ituples = generate_order_sensitive_tuples(
                question_sets, num_base_questions
            )
        else:
            possible_ituples = generate_regular_tuples(
                question_sets, num_base_questions
            )

        print(f"Generated {len(possible_ituples)} possible {num_base_questions}-tuples")

        if possible_ituples:
            print("Sample tuple:")
            sample_tuple = random.choice(possible_ituples)
            for key, value in sample_tuple[0].items():
                print(f"    {key}: {value.title}")

            try:
                results = await checker.instantiate_and_write_many(
                    possible_ituples,
                    model=model,
                    n_write=n_write,
                    overwrite=True,
                    **kwargs,
                )
                print(f"Completed processing for {checker_name}")
            except Exception as e:
                print(f"Error in instantiate_and_write_many for {checker_name}: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("No tuples to process for this checker.")

    return possible_tuples


def generate_regular_tuples(question_sets, num_base_questions):
    possible_ituples = []
    for source_question, question_set in question_sets.items():
        if question_set["source"] is None:
            continue
        if num_base_questions == 1:
            possible_ituples.append(
                (
                    {"P": question_set["source"]},
                    {"P": {"source_question": source_question}},
                )
            )
        elif len(question_set["related"]) >= num_base_questions - 1:
            for related_combo in itertools.combinations(
                question_set["related"], num_base_questions - 1
            ):
                questions = {
                    "P": question_set["source"],
                    **{chr(81 + j): q for j, q in enumerate(related_combo)},
                }
                metadata = {
                    key: {"source_question": source_question}
                    for key in questions.keys()
                }
                possible_ituples.append((questions, metadata))
    return possible_ituples


def generate_order_sensitive_tuples(question_sets, num_base_questions):
    possible_ituples = []
    for source_question, question_set in question_sets.items():
        if (
            question_set["source"] is None
            or len(question_set["related"]) < num_base_questions - 1
        ):
            continue

        source_question_obj = question_set["source"]

        for related_combo in itertools.combinations(
            question_set["related"], num_base_questions - 1
        ):
            all_questions = [source_question_obj] + list(related_combo)

            for perm in itertools.permutations(all_questions):
                questions = {chr(80 + i): q for i, q in enumerate(perm)}
                metadata = {
                    key: {"source_question": source_question}
                    for key in questions.keys()
                }
                possible_ituples.append((questions, metadata))

    return possible_ituples


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
@click.option(
    "--use-instantiate-rel",
    "-r",
    is_flag=True,
    default=False,
    help="Use instantiateRel instead of instantiate",
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
    use_instantiate_rel,
):
    checkers = choose_checkers(relevant_checks, tuple_dir)

    if use_instantiate_rel:
        asyncio.run(
            instantiateRel(
                BASE_DATA_PATH=data_path,
                checker_list=checkers,
                n_write=n_write,
                model=model_main,
                seed=seed,
            )
        )

    else:
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
