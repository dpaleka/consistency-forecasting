import json
import argparse
import asyncio
import os
import sys


from scrape_details import (
    fetch_question_details_metaculus,
    fetch_question_details_manifold,
    fetch_question_details_predictit,
)

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../src"))
from common.llm_utils import parallelized_call


"""
async def query_api_chat(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:
    
    Query the API (through instructor.Instructor) with the given messages.

    Order of precedence for model:
    1. `model` argument
    2. `model` in `kwargs`
    3. Default model
    
    def query_api_chat_sync(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:


async def parallelized_call(
    func: Coroutine,
    data: list[str],
    max_concurrent_queries: int = 100,
) -> list[any]:
    Run async func in parallel on the given data.
    func will usually be a partial which uses query_api or whatever in some way.

    Example usage:
        partial_eval_method = functools.partial(eval_method, model=model, **kwargs)
        results = await parallelized_call(partial_eval_method, [format_post(d) for d in data])
"""


def update_questions_with_details(file_path, source):
    with open(file_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    use_async = False
    click_scraping = False

    if source == "metaculus":
        fetch_question_details = fetch_question_details_metaculus
        use_async = 1
        click_scraping = True

    elif source == "manifold":
        fetch_question_details = fetch_question_details_manifold
        use_async = True

    elif source == "predictit":
        fetch_question_details = fetch_question_details_predictit

    else:
        print("Error: {} is not in known sources list".format(source))
        return

    if use_async:
        if click_scraping:
            questions = asyncio.run(
                parallelized_call(
                    fetch_question_details, questions, concurrent_queries_cap=2
                )
            )
        else:
            questions = asyncio.run(
                parallelized_call(fetch_question_details, questions)
            )

    else:
        ## add body linearly
        questions = [asyncio.run(fetch_question_details(q)) for q in questions]

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(questions, file, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Update scraped questions with resolution criteria and background info."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the JSON file containing the questions"
    )
    args = parser.parse_args()

    if "metaculus" in args.file_path.lower():
        update_questions_with_details(args.file_path, "metaculus")

    elif "manifold" in args.file_path.lower():
        update_questions_with_details(args.file_path, "manifold")

    elif "predictit" in args.file_path.lower():
        update_questions_with_details(args.file_path, "predictit")
    else:
        print("Filepath should contain one of [metaculus, manifold, predictit]")


if __name__ == "__main__":
    main()
