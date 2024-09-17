import time
import os
import json
import asyncio
import argparse
from typing import Dict
from common.utils import write_jsonl_async
from perplexity_resolver import resolve_question

def ensure_directory_exists(file_path: str):
    """
    Ensure that the directory for the given file path exists.
    If it doesn't exist, create it.
    
    :param file_path: The path to the file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


async def process_question(question: Dict, retry: bool = False, n: int = 1) -> Dict:
    """
    Process a single question with error handling and retry.
    :param question: A dictionary containing the question data
    :param retry: Boolean indicating if this is a retry attempt
    :param n: Number of attempts for the resolve function
    :return: A dictionary with the processed question data or None if processing failed
    """
    try:
        response = await resolve_question(question["body"], question["title"], n=n)
        result = question.copy()
        result["chain_of_thought"] = response.chain_of_thought
        result["can_resolve_question"] = response.can_resolve_question
        result["answer"] = response.answer

        return result
    except Exception as e:
        print(f"Error processing question: {e}")
        if not retry:
            print("Retrying once...")
            return await process_question(question, retry=True, n=n)
        else:
            print("Failed after retry, ignoring this question.")
            return None


async def process_jsonl_file(
    input_file: str,
    output_file: str,
    max_questions: int = 200,
    model: str = "perplexity/llama-3.1-sonar-huge-128k-online",
    include_unresolvable: bool = False,
    n_attempts: int = 1,
):
    """
    Read forecasting questions from a JSONL file, resolve each question, and write results to a new JSONL file.
    :param input_file: Path to the input JSONL file
    :param output_file: Path to the output JSONL file
    :param max_questions: Maximum number of questions to process
    :param model: The model to use for the resolution function
    :param include_unresolvable: Whether to include questions that can't be resolved
    :param n_attempts: Number of attempts for the resolve function
    """
    try:
        # Read input JSONL file
        with open(input_file, "r") as f:
            questions = [json.loads(line) for line in f]
        questions = questions[:max_questions]
        print(f"Total questions to process: {len(questions)}")

        # Process questions concurrently using asyncio.gather
        results = await asyncio.gather(
            *[process_question(question, n=n_attempts) for question in questions]
        )

        # Filter out None results (failed questions) and optionally filter unresolvable questions
        if include_unresolvable:
            filtered_results = [result for result in results if result is not None]
        else:
            filtered_results = [
                result for result in results
                if result is not None and result.get("can_resolve_question", False)
            ]

        print(f"Filtered results: {len(filtered_results)}")
        print("----")
        if filtered_results:
            print(f"type of results: {type(filtered_results[0])}")
            print(f"Sample result: {filtered_results[0]}")

        ensure_directory_exists(output_file)
        # Write output JSONL file
        await write_jsonl_async(output_file, filtered_results)
        print(
            f"Successfully processed {len(filtered_results)} questions. "
            f"Results written to {output_file}"
        )
    except Exception as e:
        print(f"An error occurred while processing the JSONL file: {e}")


async def main():
    # Setup CLI argument parsing
    parser = argparse.ArgumentParser(
        description="Process a JSONL file of questions and resolve them using a specified model."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="src/data/news_feed_fq_generation/news_api/final_unverified/anthropic__claude-3.5-sonnet/2024-08-01_to_2024-08-31/num_pages_1/num_articles_all/lax_res_checking_fqs.jsonl",
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="src/data/news_feed_fq_generation/news_api/perplexity_resolved/anthropic__claude-3.5-sonnet/2024-08-01_to_2024-08-31/num_pages_1/num_articles_all/lax_res_checking_fqs_resolved.jsonl",
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=230,
        help="Maximum number of questions to process (default: 200)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=["perplexity/llama-3.1-sonar-huge-128k-online", "perplexity/llama-3.1-sonar-large-128k-online"],
        help="Model to use for resolving questions."
    )
    parser.add_argument(
        "--include_unresolvable",
        action="store_true",
        help="Include questions that can't be resolved in the output (default: False)."
    )
    parser.add_argument(
        "--n_attempts",
        type=int,
        default=1,
        help="Number of attempts for the resolve function (default: 1)."
    )

    args = parser.parse_args()

    t0 = time.time()
    await process_jsonl_file(
        input_file=args.input_file,
        output_file=args.output_file,
        max_questions=args.max_questions,
        model=args.model,
        include_unresolvable=args.include_unresolvable,
        n_attempts=args.n_attempts
    )
    print(f"time taken: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())