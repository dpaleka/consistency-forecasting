import json
import asyncio
from pathlib import Path
from typing import List
import click
from dotenv import load_dotenv

from common.datatypes import ForecastingQuestion
from common.path_utils import get_data_path

from question_generators.utils import (
    InstructionForQuestions,
    generate_questions_batched,
    deduplicate,
)

load_dotenv()
DATA_PATH: Path = get_data_path()

MODEL: str = "gpt-3.5-turbo"
EMBEDDING_MODEL: str = "text-embedding-3-small"


def load_questions(path: str) -> List[ForecastingQuestion]:
    with open(path, "r") as f:
        jsonl_content = f.read()
    return [
        ForecastingQuestion(**json.loads(jline)) for jline in jsonl_content.splitlines()
    ]


instruction = InstructionForQuestions(
    prompt="""\
- You are creating {num_questions} examples that follow the format of the example(s) provided, but with different content.
- The created examples **must** all be about US politics.
- The created examples **must** all have different answers.
- The output **must** be in unnumbered JSON format.
"""
)


def write_questions(questions: List[ForecastingQuestion], file_name: str):
    with open(DATA_PATH / "fq" / "synthetic" / file_name, "w") as f:
        for q in questions:
            f.write(f"{q.model_dump_json()}\n")


@click.command()
@click.option(
    "--input-file",
    default="politics_qs_2_formatted.jsonl",
    help="Input JSONL file (assumed to be in data/fq/synthetic)",
)
@click.option(
    "--output-file",
    default="politics_qs_3.jsonl",
    help="Output JSONL file (written to data/fq/synthetic)",
)
@click.option("--questions-per-batch", default=5, help="Number of questions per batch")
@click.option("--num-batches", default=20, help="Number of batches")
def main(input_file: str, output_file: str, questions_per_batch: int, num_batches: int):
    question_examples = load_questions(DATA_PATH / "fq" / "synthetic" / input_file)
    questions = generate_questions_batched(
        model=MODEL,
        instruction=instruction,
        questions_per_batch=questions_per_batch,
        num_batches=num_batches,
        question_examples=question_examples,
        verbose=True,
    )
    write_questions(questions, output_file)

    questions = load_questions(DATA_PATH / "fq" / "synthetic" / output_file)
    deduped_questions = asyncio.run(
        deduplicate(questions, embedding_model=EMBEDDING_MODEL)
    )
    write_questions(deduped_questions, f"{output_file.split('.')[0]}_deduped.jsonl")


if __name__ == "__main__":
    main()

# Example command:
# python src/question_generators/us_politics_reformat.py --input-file="politics_qs_2_formatted.jsonl" --output-file="politics_qs_3.jsonl" --questions-per-batch=5 --num-batches=20
