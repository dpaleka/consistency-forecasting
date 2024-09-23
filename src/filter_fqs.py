import argparse
import json
from pathlib import Path
from typing import Callable

from common.datatypes import ForecastingQuestion


def default_filter(fq: ForecastingQuestion) -> bool:
    return fq.resolution in [True, False]


def filter_fqs(
    input_file: Path,
    output_file: Path,
    filter_func: Callable[[ForecastingQuestion], bool] = default_filter,
    max_questions: int | None = None,
) -> None:
    good_lines: list[str] = []
    processed = 0
    cnt = 0

    with input_file.open("r") as f:
        for line in f:
            fq = ForecastingQuestion.model_validate(json.loads(line))
            cnt += 1
            if filter_func(fq):
                good_lines.append(line)
                processed += 1
                if max_questions and processed >= max_questions:
                    break

    with output_file.open("w") as f:
        for line in good_lines:
            f.write(line)

    print(f"Kept {processed}/{cnt} questions")
    print(f"Output written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter Forecasting Questions based on specified criteria."
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing Forecasting Questions",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Path to the output JSONL file. If not provided, will overwrite input_file",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default="default",
        help="Filter to apply. Options: default, resolved, unresolved, binary",
    )
    parser.add_argument(
        "-n",
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to process",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file
        if (
            input("Output file not provided. Overwrite input file? (y/n): ").lower()
            != "y"
        ):
            print("Operation cancelled.")
            return
        print(f"Proceeding to overwrite {input_file}...")

    filter_fqs(input_file, output_file, max_questions=args.max_questions)


if __name__ == "__main__":
    main()

# Example run command:
# python src/filter_fqs.py -i path/to/input.jsonl -o path/to/output_filtered.jsonl -n 1000
