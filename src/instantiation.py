import jsonlines
import asyncio
import click
from costly import Costlog

# from static_checks.MiniInstantiator import MiniInstantiator
from static_checks import Checker
from static_checks.Checker import choose_checkers
from static_checks.tuple_relevance import relevance
from common.datatypes import ForecastingQuestion
from common.path_utils import get_data_path
from pathlib import Path
from common.llm_utils import parallelized_call
import functools
import random
import itertools
import sys

# The following are defaults, but can be overriden in the script args
MODEL = "gpt-4o-2024-08-06"  # "gpt-4o-mini-2024-07-18"
MODEL_RELEVANCE = "gpt-4o-mini-2024-07-18"  # "gpt-4o-mini-2024-07-18"
BASE_DATA_PATH: Path = (
    get_data_path()
    / "fq"
    / "synthetic"
    / "news_api_generated_fqs"
    / "20240701_20240831_gpt-4o_spanned_resolved.jsonl"
)
# get_data_path() / "fq" / "real" / "20240501_20240815.jsonl"
# get_data_path() / "fq" / "real" / "20240501_20240815_unverified.jsonl"
# get_data_path() / "fq" / "synthetic" / "news_api_generated_fqs" / "20240701_20240831_gpt-4o_spanned_resolved.jsonl"
TUPLES_PATH: Path = get_data_path() / "tuples_newsapi/"

RELEVANT_CHECKS = ["all"]


def select_tuples(possible_ituples, max_tuples):
    """
    Randomly select up to max_tuples from possible_ituples.
    Maybe make this more sophisticated in future
    """
    if len(possible_ituples) <= max_tuples:
        return possible_ituples
    return random.sample(possible_ituples, max_tuples)


async def instantiateRel(
    BASE_DATA_PATH: Path,
    checker_list: dict[str, Checker],
    n_source_questions: int = 10,
    max_tuples_per_source: int = 10,
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
    # Print the number of related questions for each source question
    print("\nNumber of related questions for each source question:")
    for source_question, question_set in valid_sets.items():
        num_related = len(question_set["related"])
        print(f"  {source_question}: {num_related} related questions")
    random.seed(seed)
    # If n_source_questions is -1 or greater than the number of valid sets, use all valid sets
    if n_source_questions == -1 or n_source_questions > len(valid_sets):
        n_source_questions = len(valid_sets)

    # Randomly sample n_source_questions from valid_sets
    selected_sets = random.sample(list(valid_sets.items()), n_source_questions)

    order_sensitive_checkers = ["ButChecker", "CondChecker", "CondCondChecker"]

    for checker_name, checker in checker_list.items():
        print(f"\nProcessing {checker_name}:")
        num_base_questions = checker.num_base_questions

        all_tuples = []
        tuples_per_source = {}
        for source_question, question_set in selected_sets:
            if checker_name in order_sensitive_checkers:
                possible_ituples = generate_order_sensitive_tuples(
                    {source_question: question_set}, num_base_questions
                )
            else:
                possible_ituples = generate_regular_tuples(
                    {source_question: question_set}, num_base_questions
                )

            # Randomly select tuples up to max_tuples_per_source
            selected_ituples = select_tuples(possible_ituples, max_tuples_per_source)
            all_tuples.extend(selected_ituples)
            tuples_per_source[source_question] = len(selected_ituples)
        print(
            f"Generated {len(all_tuples)} possible {num_base_questions}-tuples from {n_source_questions} source questions"
        )

        print("Number of tuples generated for each source question:")
        for source_question, num_tuples in tuples_per_source.items():
            print(f"  {source_question}: {num_tuples} tuples", file=sys.stderr)
        if all_tuples:
            # print("Sample tuple:")
            # sample_tuple = random.choice(all_tuples)
            # for key, value in sample_tuple[0].items():
            #     print(f"    {key}: {value.title}")

            try:
                results = await checker.instantiate_and_write_many(
                    all_tuples,
                    model=model,
                    n_write=-1,  # Write all generated tuples
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
    return all_tuples


def generate_regular_tuples(question_sets, num_base_questions):
    possible_ituples = []
    for source_question, question_set in question_sets.items():
        if question_set["source"] is None:
            continue
        if num_base_questions == 1:
            possible_ituples.append(
                (
                    {"P": question_set["source"]},
                    {
                        "P": {
                            "source_question": source_question,
                            "source_id": question_set["source"].id,
                        }
                    },
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
                    key: {
                        "source_question": source_question,
                        "source_id": question_set["source"].id,
                    }
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
                    key: {
                        "source_question": source_question,
                        "source_id": source_question_obj.id,
                    }
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
    cost_log = kwargs.get("cost_log", None)
    simulate = kwargs.get("simulate", False)

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
            func = functools.partial(
                relevance, model=model_relevance, cost_log=cost_log, simulate=simulate
            )
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
@click.option("--n_relevance", default=50, help="Number of relevance samples.")
@click.option("--n_write", default=10, help="Number of writes.")
@click.option(
    "--n_source_questions",
    default=-1,
    help="Number of source questions to process for instantiateRel function.",
)
@click.option(
    "--max_tuples_per_source",
    default=10,
    help="Maximum number of tuples per source question for instantiateRel function.",
)
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
    type=str,
    default=TUPLES_PATH,
    help="Directory to write tuples to.",
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
@click.option(
    "--simulate",
    is_flag=True,
    default=False,
    help="Simulate the instantiation process, don't call the LLM or write to files.",
)
def main(
    data_path,
    n_relevance,
    n_write,
    n_source_questions,
    max_tuples_per_source,
    model_main,
    model_relevance,
    relevant_checks,
    tuple_dir,
    seed,
    use_instantiate_rel,
    simulate,
):
    print(f"Tuple dir: {tuple_dir}")
    tuple_dir = Path(tuple_dir)
    if not tuple_dir.exists():
        tuple_dir.mkdir(parents=True, exist_ok=True)

    checkers = choose_checkers(relevant_checks, tuple_dir)

    cl = Costlog(mode="jsonl")

    if use_instantiate_rel:
        asyncio.run(
            instantiateRel(
                BASE_DATA_PATH=data_path,
                checker_list=checkers,
                n_source_questions=n_source_questions,
                max_tuples_per_source=max_tuples_per_source,
                model=model_main,
                seed=seed,
                cost_log=cl,
                simulate=simulate,
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
                cost_log=cl,
                simulate=simulate,
            )
        )

    print("Costly log totals:")
    print("------------------")
    print(cl.totals)
    print(cl.totals_by_model)
    print("------------------")


if __name__ == "__main__":
    main()
