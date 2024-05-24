"""
Please use this script for any ForecastingQuestion JSONL files that you commit.
"""

import argparse
import json
from pathlib import Path

from common.datatypes import ForecastingQuestion
from pydantic import ValidationError


def validate_fq(line: dict):
    try:
        ForecastingQuestion.model_validate(line)
    except ValidationError as e:
        print(f"Error validating data: {e}\n")
        print(f"line: {line}\n")
        return False
    return True


def validate_fq_tuple(line: dict):
    # Each line has multiple fields, most of which should contain ForecastingQuestions. We ignore the others.
    ignore_fields = ["metadata"]
    check_fields = [k for k, v in line.items() if k not in ignore_fields]
    for k in check_fields:
        if not validate_fq(line[k]):
            return False
    return True


def validate_fq_jsonl_file(filename: str, validate_tuple: bool = False):
    if filename is None:
        raise ValueError("Error: Please provide a filename.")
    if not Path(filename).is_file():
        raise ValueError(f"Error: File {filename} does not exist.")
    if not filename.endswith(".jsonl"):
        raise ValueError(f"Error: File {filename} is not a JSONL file.")
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = []
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSONL: {e}")
                    print(f"line: {line}")
                    raise
    except Exception as e:
        print(f"Error: {e}")
        raise
    for line in data:
        if validate_tuple:
            if not validate_fq_tuple(line):
                print(ForecastingQuestion.model_json_schema())
                raise ValueError(
                    "Error: one of the questions in the tuple does not conform to the schema."
                )
        else:
            if not validate_fq(line):
                print(ForecastingQuestion.model_json_schema())
                raise ValueError("Error: the question does not conform to the schema.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        help="Path to the JSONL file to validate, containing ForecastingQuestions",
    )
    parser.add_argument(
        "--tuple",
        "-t",
        action="store_true",
        help="Validate the file as a tuple of questions.",
    )
    args = parser.parse_args()
    res = validate_fq_jsonl_file(args.filename, args.tuple)
    if res:
        print(f"Validated {args.filename} successfully!")
