"""
Please use this script for any ForecastingQuestion JSONL files that you commit.
"""

import argparse
import json
from pathlib import Path

from common.datatypes import ForecastingQuestion
from pydantic import ValidationError


def validate_fq(data: list[dict]):
    for line in data:
        try:
            ForecastingQuestion.model_validate(line)
        except ValidationError as e:
            print(f"Error validating data: {e}\n")
            print(f"line: {line}\n")
            return False
    return True


def validate_fq_jsonl_file(filename: str):
    if filename is None:
        raise ValueError("Error: Please provide a filename.")
    if not Path(filename).is_file():
        raise ValueError(f"Error: File {filename} does not exist.")
    if not filename.endswith(".jsonl"):
        raise ValueError(f"Error: File {filename} is not a JSONL file.")
    with open(filename, "r") as file:
        data = []
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSONL: {e}")
                print(f"line: {line}")
                raise
    if not validate_fq(data):
        print(ForecastingQuestion.model_json_schema())
        raise ValueError("Error: file does not conform to the schema.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        help="Path to the JSONL file to validate, containing ForecastingQuestions",
    )
    args = parser.parse_args()
    validate_fq_jsonl_file(args.filename)
