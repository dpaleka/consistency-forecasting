import asyncio
import json
from typing import List, Optional
from common.datatypes import ForecastingQuestion
from question_generators import question_formatter
from common.utils import write_jsonl_async
from common.path_utils import get_data_path, get_scripts_path
from simple_parsing import ArgumentParser
from pathlib import Path


def read_json_or_jsonl(file_path: Path):
    if file_path.suffix == ".json":
        with open(file_path, "r") as file:
            return json.load(file)
    elif file_path.suffix == ".jsonl":
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]
    else:
        raise ValueError(
            "Unsupported file format. Only '.json' and '.jsonl' files are supported."
        )


async def validate_and_format_question(question: dict) -> Optional[ForecastingQuestion]:
    for i in range(2):
        forecasting_question = await question_formatter.from_string(
            question["title"],
            data_source=question["data_source"],
            question_type=question.get("question_type"),
            url=question.get("url", None),
            metadata=question.get("metadata", None),
            body=question.get("body", None),
            date=question.get("resolution_date", None),
        )
        if await question_formatter.validate_question(forecasting_question):
            break
        else:
            print(f"Invalid question: {question}")
            forecasting_question = None
            await asyncio.sleep(1)

    return forecasting_question


async def process_questions_from_file(
    file_path: Path, max_questions: Optional[int]
) -> List[ForecastingQuestion]:
    questions = read_json_or_jsonl(file_path)

    max_questions = max_questions if max_questions else len(questions)
    tasks = []

    for question in questions[:max_questions]:
        tasks.append(validate_and_format_question(question))

    forecasting_questions = await asyncio.gather(*tasks)
    count_none = forecasting_questions.count(None)
    forecasting_questions = [fq for fq in forecasting_questions if fq is not None]
    return forecasting_questions, count_none


async def main(
    file_path: Path, out_data_dir: str, out_file_name: str, max_questions: int
):
    forecasting_questions, none_count = await process_questions_from_file(
        file_path, max_questions
    )
    print(f"Number of invalid questions found: {none_count}")

    data_to_write = [fq.dict() for fq in forecasting_questions]
    for data in data_to_write:
        data["id"] = str(data["id"])
        data["resolution_date"] = str(data["resolution_date"])

    await write_jsonl_async(
        f"{get_data_path()}/fq/{out_data_dir}/{out_file_name}",
        data_to_write,
        append=False,
    )
    await write_jsonl_async(
        f"{get_scripts_path()}/pipeline/{out_file_name}",
        data_to_write,
        append=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--file_path",
        "-f",
        type=str,
        default=f"{get_scripts_path()}/pipeline/QUESTIONS_CLEANED_MODIFIED.jsonl",
        help="Path to the input file",
    )
    parser.add_argument(
        "--out_data_dir",
        "-d",
        type=str,
        default="real",
        choices=["real", "synthetic"],
        help="Data dir to write the output to",
    )
    # TODO name the output file better, somehow dependent on the input file name
    parser.add_argument(
        "--out_file_name",
        "-o",
        type=str,
        default="questions_cleaned_formatted.jsonl",
        help="Name of the output file",
    )
    parser.add_argument(
        "--max_questions",
        "-m",
        type=int,
        default=30,
        help="Maximum number of questions to process",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            Path(args.file_path),
            args.out_data_dir,
            args.out_file_name,
            args.max_questions,
        )
    )
