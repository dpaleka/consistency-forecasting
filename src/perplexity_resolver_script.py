import time
from typing import Dict
import json
import asyncio
from common.utils import write_jsonl_async
from perplexity_resolver import resolve_question


async def process_question(question: Dict, retry: bool = False, i = 0) -> Dict:
    """
    Process a single question with error handling and retry.
    :param question: A dictionary containing the question data
    :param retry: Boolean indicating if this is a retry attempt
    :return: A dictionary with the processed question data or None if processing failed
    """
    try:
        response = await resolve_question(question["body"], question["title"])
        print(f"retry = {retry}, response {i}: {response}")
        print("0-0-0-0-0-0-0-0-0-0-0-0-000")
        print("0-0-0-0-0-0-0-0-0-0-0-0-000")
        result = question.copy()
        result["chain_of_thought"] = response.chain_of_thought
        result["can_resolve_question"] = response.can_resolve_question
        result["answer"] = response.answer

        for key, value in result.items():
            print(f"{key}: {type(value)}")
            print(f"{key}: {value}")

        return result
    except Exception as e:
        print(f"Error processing question: {e}")
        if not retry:
            print("Retrying once...")
            return await process_question(question, retry=True, i=i)
        else:
            print("Failed after retry, ignoring this question.")
            return None


async def process_jsonl_file(
    input_file: str,
    output_file: str,
    model: str = "perplexity/llama-3.1-sonar-huge-128k-online",
):
    """
    Read forecasting questions from a JSONL file, resolve each question, and write results to a new JSONL file.
    :param input_file: Path to the input JSONL file
    :param output_file: Path to the output JSONL file
    :param model: The model to use for the resolution function
    """
    try:
        # Read input JSONL file
        with open(input_file, "r") as f:
            questions = [json.loads(line) for line in f]
        questions = questions[:200]

        # Process questions concurrently using asyncio.gather
        results = await asyncio.gather(
            *[process_question(question) for question in questions]
        )

        # Filter out None results (failed questions)
        results = [result for result in results if result is not None]

        print(results)
        print("----")
        print(f"type of results: {type(results[0])}")

        # Write output JSONL file
        await write_jsonl_async(output_file, results)
        print(
            f"Successfully processed {len(results)} questions. Results written to {output_file}"
        )
    except Exception as e:
        print(f"An error occurred while processing the JSONL file: {e}")


async def main():
    # Example usage of process_jsonl_file
    t0 = time.time()
    input_file = "src/data/news_feed_fq_generation/news_api/final_unverified/final_fq__claude-3.5-sonnet_lax_res_checking_from_July-1-2024_to_July-31-2024_num_pages_1_num_articles_all.jsonl"
    input_file = "src/input_file.jsonl"
    output_file = "src/output_results_5.jsonl"
    await process_jsonl_file(input_file, output_file)
    print(f"time taken: {time.time() - t0}")


if __name__ == "__main__":
    asyncio.run(main())
