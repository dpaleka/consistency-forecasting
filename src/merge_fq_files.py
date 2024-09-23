"""
Example question:
{"id": "a6612574-d9d3-4b46-8567-5fbe346589b1", "title": "Will the mpox outbreak in the Central African Republic be declared over by July 2024?", "body": "This question will resolve as YES if, by July 31, 2024, the mpox outbreak in the Central African Republic is officially declared over by the country's health authorities or the World Health Organization (WHO). The declaration must be reported by at least two reputable international news sources (such as Reuters, Associated Press, BBC, or CNN). If no such declaration is made by the resolution date, or if the outbreak is still ongoing, the question will resolve as NO.", "resolution_date": "2024-07-31T23:59:59", "question_type": "binary", "data_source": "synthetic", "created_date": "2024-06-30T23:59:59", "url": null, "metadata": {"article_information": {"article_url": "https://apnews.com/article/mpox-kenya-central-african-republic-outbreak-disease-7da16b2ccad88b7580318322bb0798ed", "article_date": "2024-07-31 22:57:06", "article_description": "Kenya and the Central African Republic have declared new outbreaks of mpox as Africa's health officials are racing to contain the spread of the disease in a region lacking vaccines. Kenya's health ministry confirmed an outbreak on Wednesday, after a case was \u2026", "article_title": "Mpox outbreaks declared in Kenya and Central African Republic. The race is on to contain the spread", "article_content": "ABUJA, Nigeria (AP) Kenya and the Central African Republic have declared new outbreaks of mpox as Africas health officials race to contain the spread of the disease in a region lacking vaccines.\r\nNai\u2026 [+2145 chars]"}, "pose_date": "2023-10-01 00:00:00", "scraped_date": "2024-09-19 20:38:13", "perplexity_verification": {"chain_of_thought": "To resolve this question, I've checked the available information up to the resolution date of July 31, 2024.\n    1. The WHO and Africa CDC reports up to July 2024 do not mention the mpox outbreak in the Central African Republic being declared over. Instead, they highlight the ongoing spread and increasing cases in the region.\n    2. The latest situation reports from WHO and other health organizations indicate that the mpox outbreak is still ongoing in the Central African Republic and other parts of Africa, with no declaration of the outbreak being over in the Central African Republic by July 31, 2024.\n    3. Given the information available, there is no indication that the mpox outbreak in the Central African Republic was declared over by July 31, 2024.\n\n    Therefore, based on the available data, the question can be resolved as NO because there was no declaration of the outbreak being over in the Central African Republic by the specified date.", "can_resolve_question": true, "answer": false, "agreement": true}}, "resolution": false}
"""

import argparse
import os
import json
from pathlib import Path
from typing import Optional
import asyncio
from fq_verification.question_verifier import verify_question
from common.utils import ensure_directory_exists
from common.llm_utils import query_api_chat_sync_native, parallelized_call
from fq_verification.question_verifier import ForecastingQuestion
from functools import partial


def get_a_good_output_filename(input_files: list[str]) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates a good filename for a given list of input files.",
        },
        # Example 1:
        {
            "role": "user",
            "content": "Input files: ['src/data/fq/synthetic/news_api_generated_fqs/openai__gpt-4o-2024-08-06/2024-07-01_to_2024-07-31/num_pages_1/num_articles_all/strict_res_checking_fqs_cleaned-ref-class-spanned-basic_resolved.jsonl']",
        },
        {
            "role": "assistant",
            "content": "src/data/fq/synthetic/news_api_generated_fqs/20240701_20240731_gpt-4o_spanned.jsonl",
        },
        # Example 2:
        {
            "role": "user",
            "content": "Input files: ['src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl', 'src/data/fq/real/manifold_cleaned_formatted_20240501_20240815.jsonl`]",
        },
        {"role": "assistant", "content": "src/data/fq/real/20240501_20240815.jsonl"},
        # Example 3:
        {
            "role": "user",
            "content": "Input files: ['src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815_unverified.jsonl', 'src/data/fq/real/manifold_cleaned_formatted_20240501_20240815_unverified.jsonl']",
        },
        {
            "role": "assistant",
            "content": "src/data/fq/real/20240501_20240815_unverified.jsonl",
        },
        {
            "role": "user",
            "content": f"Input files: [{', '.join([f'`{input_file}`' for input_file in input_files])}]",
        },
    ]

    response = query_api_chat_sync_native(messages, model="gpt-4o")
    return response


async def concatenate_and_filter(
    input_files: list[str], output_file: str, verify: bool = False
):
    aggregated_questions = []
    cnt_initial = 0
    for file in input_files:
        with open(file, "r") as f:
            for line in f:
                question = json.loads(line)
                cnt_initial += 1
                can_resolve = (
                    question.get("metadata", {})
                    .get("perplexity_verification", {})
                    .get("can_resolve_question", False)
                )
                resolution = question.get("resolution")
                if not can_resolve:
                    continue
                if (
                    can_resolve
                    and resolution is not None
                    and resolution
                    != question.get("metadata", {})
                    .get("perplexity_verification", {})
                    .get("answer", None)
                ):
                    continue
                if resolution is None:
                    question["resolution"] = question["metadata"][
                        "perplexity_verification"
                    ].get("answer", False)
                aggregated_questions.append(question)

    print(
        f"Number of questions after can_resolve and agreement filter: {len(aggregated_questions)}/{cnt_initial}"
    )

    ensure_directory_exists(output_file)  # Ensure output directory exists
    if verify:
        verified_questions = []
        batch_size = 50
        for i in range(0, len(aggregated_questions), batch_size):
            batch = aggregated_questions[i : i + batch_size]
            fq_batch = [ForecastingQuestion(**question) for question in batch]
            partial_func = partial(verify_question, model="gpt-4o")
            results = await parallelized_call(partial_func, fq_batch)
            for question, result in zip(batch, results):
                if result.valid:
                    verified_questions.append(question)

            with open(output_file, "w") as f:
                for question in verified_questions:
                    f.write(json.dumps(question) + "\n")

            print(
                f"Number of questions after verification: {len(aggregated_questions)}/{cnt_initial}"
            )
    else:
        with open(output_file, "w") as f:
            for question in aggregated_questions:
                f.write(json.dumps(question) + "\n")

    print(f"Concatenated and filtered data written to {output_file}")


def gather_jsonl_files(directory: str, substring: Optional[str] = None) -> list[str]:
    path = Path(directory)
    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError(f"File {path} is not a JSONL file.")
        return [str(path)]

    jsonl_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".jsonl"):
                if substring is None or substring in file:
                    jsonl_files.append(str(Path(root) / file))

    return jsonl_files


async def main():
    parser = argparse.ArgumentParser(description="Merge and filter JSONL files.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input JSONL files, or a single JSONL file.",
    )
    parser.add_argument(
        "-s",
        "--substring",
        type=str,
        default="",
        help="Substring to filter JSONL files in the input directory.",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run FQ verification on the filtered questions. Note that this might be redundant if the questions have already been verified.",
    )

    args = parser.parse_args()
    input_files = gather_jsonl_files(args.input_dir, args.substring)

    if args.output_file is None:
        args.output_file = get_a_good_output_filename(input_files)
        # Add a yes/no confirmation for the output file
        confirmation_message = f"Output will be written to: {args.output_file}"
        print(confirmation_message)
        confirm = input("Is this correct? (y/n): ").lower().strip()
        if confirm != "y":
            print("Operation cancelled.")
            exit(0)

    if os.path.exists(args.output_file):
        confirm = (
            input(f"Output file {args.output_file} already exists. Overwrite? (y/n): ")
            .lower()
            .strip()
        )
        if confirm != "y":
            print("Operation cancelled.")
            exit(0)

    print(f"Output will be written to: {args.output_file}")

    await concatenate_and_filter(
        input_files=input_files, output_file=args.output_file, verify=args.verify
    )

    print("Output written to", args.output_file)


if __name__ == "__main__":
    asyncio.run(main())

# python merge_verified.py -i /path/to/input/directory -s "substring" -o /path/to/output/file [--verify]

# Example:
# python src/merge_verified.py -i src/data/fq/synthetic/news_api_generated_fqs/openai__gpt-4o-2024-08-06/2024-07-01_to_2024-07-31/num_pages_1/num_articles_all/strict_res_checking_fqs_cleaned-ref-class-spanned-basic_resolved.jsonl -o src/data/fq/synthetic/news_api_generated_fqs/20240701_20240731__gpt-4o_spanned.jsonl
