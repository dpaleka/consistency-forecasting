import json
import argparse
import asyncio
import os
import sys
from tqdm import tqdm


from scrape_details import (
    fetch_question_details_metaculus,
    fetch_question_details_manifold,
    fetch_question_details_predictit,
)

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../src"))
from common.llm_utils import parallelized_call


def update_questions_with_details(file_path, source, save_at_most: int | None = None):
    print(f"Updating questions with details for {source} in-place in {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    print(f"{len(questions)} questions to process")
    if save_at_most:
        questions = questions[:save_at_most]
        print(
            f"Only processing {len(questions)} questions, deleting any additional ones in the suffix"
        )

    use_async = False
    click_scraping = False

    if source == "metaculus":
        fetch_question_details = fetch_question_details_metaculus
        use_async = 1

    elif source == "manifold":
        fetch_question_details = fetch_question_details_manifold
        use_async = 1
        click_scraping = True

    elif source == "predictit":
        fetch_question_details = fetch_question_details_predictit

    else:
        print("Error: {} is not in known sources list".format(source))
        return

    save_every = 40

    grabbed_questions = []
    processed_count = 0

    max_concurrent_queries = 2 if click_scraping else 10

    if use_async:
        print(f"Retrieving body from {source} asynchronously")

        for i in range(0, len(questions), save_every):
            batch = questions[i : i + save_every]
            batch_results = asyncio.run(
                parallelized_call(
                    fetch_question_details,
                    batch,
                    max_concurrent_queries=max_concurrent_queries,
                )
            )
            grabbed_questions.extend(batch_results)
            processed_count += len(batch_results)

            # merge processed questions with the rest of the questions
            questions[:processed_count] = grabbed_questions
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(questions, file, indent=4, ensure_ascii=False)
    else:
        print(f"Retrieving body from {source} linearly")

        for i, q in enumerate(tqdm(questions)):
            if q.get("body") == "N/A":
                grabbed_questions.append(asyncio.run(fetch_question_details(q)))
            else:
                grabbed_questions.append(q)

            processed_count += 1

            questions[:processed_count] = grabbed_questions
            if processed_count % save_every == 0:
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(
                        questions[:processed_count], file, indent=4, ensure_ascii=False
                    )

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(questions, file, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Update scraped questions with resolution criteria and background info."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the JSON file containing the questions"
    )
    parser.add_argument(
        "-n", "--num_questions", type=int, help="Save at most this many questions"
    )
    args = parser.parse_args()
    print(args.file_path)

    source = None
    match args.file_path.lower():
        case s if "metaculus" in s:
            source = "metaculus"
        case s if "manifold" in s:
            source = "manifold"
        case s if "predictit" in s:
            source = "predictit"
        case _:
            print("Filepath should contain one of [metaculus, manifold, predictit]")
            return

    if source:
        update_questions_with_details(args.file_path, source, args.num_questions)


if __name__ == "__main__":
    main()
