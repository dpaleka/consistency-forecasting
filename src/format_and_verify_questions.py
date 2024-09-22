import asyncio
import functools
import json
from typing import List, Optional, Union
from common.datatypes import (
    ForecastingQuestion,
    SyntheticTagQuestion,
    SyntheticRelQuestion,
)
import fq_verification.question_verifier as question_verifier
import fq_generation.fq_body_generator as fq_body_generator
from common.utils import write_jsonl_async, recombine_filename
from common.llm_utils import parallelized_call
from common.path_utils import get_data_path
from simple_parsing import ArgumentParser
from pathlib import Path

SyntheticQuestion = Union[
    SyntheticTagQuestion, SyntheticRelQuestion
]  # help functions dynamically handle Synthetic Questions


def read_json_or_jsonl(file_path: Path):
    print(f"Reading file: {file_path}")
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
    model: str = "gpt-4o-mini-2024-07-18",
    fill_in_body: bool = False,
) -> Optional[ForecastingQuestion]:
    for i in range(2):
        forecasting_question = await fq_body_generator.from_string(
            question["title"],
            data_source=question["data_source"],
            created_date=question.get("created_date", None),
            question_type=question.get("question_type"),
            url=question.get("url", None),
            metadata=question.get("metadata", None),
            body=question.get("body", None),
            resolution_date=question.get("resolution_date", None),
            resolution=question.get("resolution", None),
            model=model,
            fill_in_body=fill_in_body,
        )
        if verify:
            verification = await question_verifier.verify_question(
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
    question: SyntheticQuestion,
    verify: bool = True,
    fill_in_body: bool = False,
    **kwargs,
) -> Optional[ForecastingQuestion]:
    if isinstance(question, SyntheticRelQuestion):
        metadata = {"source_question": question.source_question}
    else:
        metadata = {"tags": question.tags, "category": question.category}
    for i in range(2):
        forecasting_question = await fq_body_generator.from_string(
            question.title,
            data_source="synthetic",
            question_type="binary",
            metadata=metadata,
            fill_in_body=fill_in_body,
            body=question.body,
            resolution_date=question.resolution_date,
            **kwargs,
        )
        if verify:
            verification = await question_verifier.verify_question(forecasting_question)
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
    model: str = "gpt-4o-mini-2024-07-18",
    verification_level: str = "full",
    fill_in_body: bool = False,
    concurrent_queries=15,
) -> List[ForecastingQuestion]:
    questions = read_json_or_jsonl(file_path)

    # dynamically determine type of questions
    if "source_question" in questions[0]:
        question_type = "rel"
    else:
        question_type = "tag"

    if question_type == "rel":
        questions = [SyntheticRelQuestion(**q) for q in questions]
    elif question_type == "tag":
        questions = [SyntheticTagQuestion(**q) for q in questions]

    print(f"number of questions before removing duplicates: {len(questions)}")
    questions = remove_repeated_questions(questions, output_path)
    print(f"number of questions after removing duplicates:  {len(questions)}")

    max_questions = max_questions if max_questions else len(questions)

    verify = verification_level == "full"

    func = functools.partial(
        validate_and_format_synthetic_question,
        model=model,
        fill_in_body=fill_in_body,
        verify=verify,
    )
    forecasting_questions = await parallelized_call(
        func=func,
        data=questions[:max_questions],
        max_concurrent_queries=concurrent_queries,
    )

    count_none = forecasting_questions.count(None)
    forecasting_questions = [fq for fq in forecasting_questions if fq is not None]
    return forecasting_questions, count_none


async def process_questions_from_file(
    file_path: Path,
    max_questions: Optional[int],
    model: str = "gpt-4o-mini-2024-07-18",
    verification_level: str = "full",
    fill_in_body: bool = False,
    concurrent_queries: int = 15,
) -> List[ForecastingQuestion]:
    questions = read_json_or_jsonl(file_path)

    max_questions = max_questions if max_questions else len(questions)
    verify = verification_level == "full"
    func = functools.partial(
        validate_and_format_question,
        model=model,
        fill_in_body=fill_in_body,
        verify=verify,
    )
    forecasting_questions = await parallelized_call(
        func=func,
        data=questions[:max_questions],
        max_concurrent_queries=concurrent_queries,
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
    verification_level: str,
    synthetic: bool,
    fill_in_body: bool,
    overwrite: bool = False,
    concurrent_queries: int = 15,
):
    output_path = Path(f"{get_data_path()}/fq/{out_data_dir}/{out_file_name}")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if verification_level == "full":
        verification_suffix = ""
    elif verification_level == "light":
        verification_suffix = "_lightverified"
        assert NotImplementedError("Light verification not implemented")
    else:
        verification_suffix = "_unverified"

    output_path = recombine_filename(output_path, verification_suffix)

    if overwrite:
        if output_path.exists():
            print(
                f"The file {output_path} already exists. Overwriting as per the --overwrite flag."
            )
    else:
        if output_path.exists():
            print(
                f"The file {output_path} already exists. Appending to it. If you want to overwrite, use the --overwrite flag."
            )

    print("SYNTHETIC VALUE: ", synthetic)
    if synthetic:
        (
            forecasting_questions,
            none_count,
        ) = await process_synthetic_questions_from_file(  # LOOK HERE
            file_path,
            output_path,
            max_questions=max_questions,
            verification_level=verification_level,
            model=model,
            fill_in_body=fill_in_body,
            concurrent_queries=concurrent_queries,
        )

    else:
        print("Processing real questions")
        forecasting_questions, none_count = await process_questions_from_file(
            file_path,
            max_questions=max_questions,
            model=model,
            verification_level=verification_level,
            fill_in_body=fill_in_body,
            concurrent_queries=concurrent_queries,
        )

    print(f"Number of invalid questions found: {none_count}")

    data_to_write = [fq.dict() for fq in forecasting_questions]
    for field in ["resolution_date", "created_date", "id"]:
        for data in data_to_write:
            if data.get(field, None) is not None:
                data[field] = str(data[field])
            else:
                data[field] = None

    await write_jsonl_async(
        output_path,
        data_to_write,
        append=not overwrite,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--file_path",
        "-f",
        type=str,
        help="Path to the input file",
    )
    parser.add_argument(
        "--out_data_dir",
        "-d",
        type=str,
        default="synthetic",
        choices=["real", "synthetic", "test"],
        help="Data dir to write the output to",
    )
    parser.add_argument(
        "--out_file_name",
        "-o",
        type=str,
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
        default="gpt-4o-mini-2024-07-18",
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
    parser.add_argument(
        "--concurrent_queries",
        "-c",
        type=int,
        default=15,
        help="Max number of concurrent queries permitted to run",
    )
    parser.add_argument(
        "--verification_level",
        "-v",
        type=str,
        default="full",
        choices=["full", "light", "none"],
        help="Verification level",
    )

    args = parser.parse_args()

    print("Verification level:", args.verification_level)

    asyncio.run(
        main(
            file_path=Path(args.file_path),
            out_data_dir=args.out_data_dir,
            out_file_name=args.out_file_name,
            max_questions=args.max_questions,
            model=args.model,
            synthetic=args.synthetic,
            fill_in_body=args.fill_in_body,
            overwrite=args.overwrite,
            concurrent_queries=args.concurrent_queries,
            verification_level=args.verification_level,
        )
    )
