import asyncio
import functools
import json
from typing import List, Optional
from common.datatypes import ForecastingQuestion, SyntheticTagQuestion
from question_generators import question_formatter
from common.utils import write_jsonl_async
from common.llm_utils import parallelized_call
from common.path_utils import get_data_path, get_scripts_path
from simple_parsing import ArgumentParser
from pathlib import Path


def read_json_or_jsonl(file_path: Path):
    if not file_path.exists():
        return []

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


async def validate_and_format_question(
    question: dict,
    verify: bool = True,
    model: str = "gpt-4o-2024-05-13",
    fill_in_body: bool = False,
) -> Optional[ForecastingQuestion]:
    for i in range(2):
        forecasting_question = await question_formatter.from_string(
            question["title"],
            data_source=question["data_source"],
            question_type=question.get("question_type"),
            url=question.get("url", None),
            metadata=question.get("metadata", None),
            body=question.get("body", None),
            date=question.get("resolution_date", None),
            model=model,
            fill_in_body=fill_in_body,
        )
        if verify:
            verification = await question_formatter.verify_question(
                forecasting_question, model=model
            )
            print(f"Verification: {verification}")
            if verification.valid:
                break
            else:
                print(f"Invalid question: {question}")
                forecasting_question = None
                await asyncio.sleep(1)
        else:
            break

    return forecasting_question


async def validate_and_format_synthetic_question(
    question: SyntheticTagQuestion,
    verify: bool = True,
    fill_in_body: bool = False,
    **kwargs,
) -> Optional[ForecastingQuestion]:
    metadata = {"tags": question.tags, "category": question.category}
    for i in range(2):
        forecasting_question = await question_formatter.from_string(
            question.title,
            data_source="synthetic",
            question_type="binary",
            metadata=metadata,
            fill_in_body=fill_in_body,
            **kwargs,
        )
        if verify:
            verification = await question_formatter.verify_question(
                forecasting_question
            )
            if verification.valid:
                break
            else:
                print(f"Invalid question: {question}")
                forecasting_question = None
                await asyncio.sleep(1)
        else:
            break

    return forecasting_question


async def process_synthetic_questions_from_file(
    file_path: Path,
    output_path: Path,
    max_questions: Optional[int] = None,
    model: str = "gpt-4o-2024-05-13",
    fill_in_body: bool = False,
) -> List[ForecastingQuestion]:
    questions = read_json_or_jsonl(file_path)
    questions = [SyntheticTagQuestion(**q) for q in questions]
    print(f"number of questions before removing duplicates: {len(questions)}")
    questions = remove_repeated_questions(questions, output_path)
    print(f"number of questions after removing duplicates:  {len(questions)}")

    max_questions = max_questions if max_questions else len(questions)

    func = functools.partial(
        validate_and_format_synthetic_question, model=model, fill_in_body=fill_in_body
    )
    forecasting_questions = await parallelized_call(
        func=func, data=questions[:max_questions], max_concurrent_queries=50
    )

    count_none = forecasting_questions.count(None)
    forecasting_questions = [fq for fq in forecasting_questions if fq is not None]
    return forecasting_questions, count_none


async def process_questions_from_file(
    file_path: Path,
    max_questions: Optional[int],
    model: str = "gpt-4o-2024-05-13",
    fill_in_body: bool = False,
) -> List[ForecastingQuestion]:
    questions = read_json_or_jsonl(file_path)

    max_questions = max_questions if max_questions else len(questions)
    func = functools.partial(
        validate_and_format_question, model=model, fill_in_body=fill_in_body
    )
    forecasting_questions = await parallelized_call(
        func=func, data=questions[:max_questions], max_concurrent_queries=50
    )

    count_none = forecasting_questions.count(None)
    forecasting_questions = [fq for fq in forecasting_questions if fq is not None]
    return forecasting_questions, count_none


def remove_repeated_questions(
    questions: List[ForecastingQuestion], output_path
) -> List[ForecastingQuestion]:
    fq_questions = read_json_or_jsonl(Path(output_path))
    question_set = set([fq["title"] for fq in fq_questions])
    r = []
    for q in questions:
        if q.title not in question_set:
            r.append(q)

    return r


async def main(
    file_path: Path,
    out_data_dir: str,
    out_file_name: str,
    max_questions: int,
    model: str,
    synthetic: bool,
    fill_in_body: bool,
    overwrite: bool = False,
):
    output_path = Path(f"{get_data_path()}/fq/{out_data_dir}/{out_file_name}")
    if overwrite:
        if output_path.exists():
            confirmation = input(
                f"The file {output_path} already exists. The default is to append. Change to overwrite? (y/N): "
            )
            if confirmation.lower() != "y":
                print("Operation cancelled by the user.")
                return
    else:
        if output_path.exists():
            print(
                f"The file {output_path} already exists. Appending to it. If you want to overwrite, use the --overwrite flag."
            )

    if synthetic:
        forecasting_questions, none_count = await process_synthetic_questions_from_file(
            file_path,
            output_path,
            max_questions=max_questions,
            model=model,
            fill_in_body=fill_in_body,
        )
    else:
        forecasting_questions, none_count = await process_questions_from_file(
            file_path,
            max_questions=max_questions,
            model=model,
            fill_in_body=fill_in_body,
        )

    print(f"Number of invalid questions found: {none_count}")

    data_to_write = [fq.dict() for fq in forecasting_questions]
    for data in data_to_write:
        data["id"] = str(data["id"])
        data["resolution_date"] = str(data["resolution_date"])

    await write_jsonl_async(
        output_path,
        data_to_write,
        append=not overwrite,
    )
    await write_jsonl_async(
        f"{get_scripts_path()}/pipeline/{out_file_name}",
        data_to_write,
        append=not overwrite,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--file_path",
        "-f",
        type=str,
        default=f"{get_data_path()}/other/high-quality-questions-all-domains.jsonl",
        help="Path to the input file",
    )
    parser.add_argument(
        "--out_data_dir",
        "-d",
        type=str,
        default="synthetic",
        choices=["real", "synthetic"],
        help="Data dir to write the output to",
    )
    parser.add_argument(
        "--out_file_name",
        "-o",
        type=str,
        default="high-quality-questions--all-domains.jsonl",
        help="Name of the output file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to indicate whether to overwrite the output file if it exists, instead of appending. It will ask for confirmation.",
    )
    parser.add_argument(
        "--max_questions",
        "-m",
        type=int,
        default=30,
        help="Maximum number of questions to process",
    )
    parser.add_argument(
        "--model",
        "-M",
        type=str,
        default="gpt-4-0125-preview",
        help="Model to use",
    )
    parser.add_argument(
        "--synthetic",
        "-s",
        type=bool,
        default=False,
        help="Flag to indicate synthetic data processing",
    )
    parser.add_argument(
        "--fill_in_body",
        "-F",
        type=bool,
        default=False,
        help="If a real question does not have a question body, fill it in with an LLM call",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            Path(args.file_path),
            args.out_data_dir,
            args.out_file_name,
            args.max_questions,
            args.model,
            args.synthetic,
            args.fill_in_body,
            args.overwrite,
        )
    )
