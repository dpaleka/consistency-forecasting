import argparse
import json
import random
from pathlib import Path
from typing import Callable, List
from collections import namedtuple
import functools

from common.datatypes import ForecastingQuestion

"""
filter scoring takes an fq and returns a score, which can be used to sort and filter fqs
bad stuff should get 0 or worse, good stuff should get positive scores
strict=True means we only keep stuff with score > 0
strict=False means we keep all questions before taking prefix or random sample
"""


def default_filter_scoring(fq: ForecastingQuestion) -> int:
    return int(fq.resolution in [True, False])


def original_filter_scoring(fq: ForecastingQuestion) -> int:
    assert fq.metadata is not None
    if fq.metadata.get("generated_from_ref_class_spanner", False):
        return 0
    return 1


def filter_fqs(
    input_file: Path,
    output_file: Path,
    score_func: Callable[[ForecastingQuestion], float] = default_filter_scoring,
    max_questions: int | None = None,
    random_sample: int | None = None,
    strict: bool = True,
) -> None:
    # This seems like a nice pattern for sorting and filtering
    # TODO: refactor it to work on score_func: callable[[Any], float] and add to utils

    FQScore = namedtuple("FQScore", ["score", "index", "line"])
    FQScore.__annotations__ = {"score": int, "index": int, "line": str}
    all_lines: List[FQScore] = []
    cnt = 0

    with input_file.open("r") as f:
        for i, line in enumerate(f):
            fq = ForecastingQuestion.model_validate(json.loads(line))
            cnt += 1
            score = score_func(fq)
            all_lines.append(FQScore(score, i, line))

    # Stable sort based on score (highest first)
    all_lines.sort(
        key=functools.cmp_to_key(
            lambda x, y: y.score - x.score if y.score != x.score else x.index - y.index
        ),
        reverse=True,
    )

    cnt_positive_score = len([line for line in all_lines if line.score > 0])
    if strict:
        print(
            f"Filtering out {cnt - cnt_positive_score}/{cnt} questions with score <= 0"
        )
        all_lines = [line for line in all_lines if line.score > 0]
    else:
        print(
            f"There were {cnt_positive_score} questions with score > 0. We keep all {cnt} questions."
        )

    # Take top max_questions
    if max_questions:
        all_lines = all_lines[:max_questions]
        print(
            f"Taking top {max_questions} questions according to filter score (and secondarily by index)"
        )

    # Random sampling if specified
    if random_sample:
        random_sample = min(random_sample, len(all_lines))
        good_lines = [line for line in all_lines if line.score > 0]
        if len(good_lines) >= random_sample:
            print(
                f"We randomly sample {random_sample} questions from the {len(good_lines)} good questions."
            )
            all_lines = random.sample(good_lines, random_sample)
        else:
            print(
                f"We keep all {len(good_lines)} good questions, and take a random sample of {random_sample-len(good_lines)} bad questions."
            )
            bad_lines = [line for line in all_lines if line.score <= 0]
            all_lines = good_lines + random.sample(
                bad_lines, random_sample - len(good_lines)
            )
        random.shuffle(all_lines)

    with output_file.open("w") as f:
        for _, _, line in all_lines:
            f.write(line)

    print(f"Kept {len(all_lines)}/{cnt} questions")
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
        "--filter_score",
        type=str,
        default="default",
        help="Filter to apply. Options: default, resolved.",
    )
    parser.add_argument(
        "-n",
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to process",
    )
    parser.add_argument(
        "-r",
        "--random_sample",
        type=int,
        default=None,
        help="Randomly sample and reorder this many questions from the filtered set",
    )
    parser.add_argument(
        "-p",
        "--only_preference",
        action="store_true",
        help="Sets strict=False, so we can get more questions, still sorting by filter score",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator",
    )

    args = parser.parse_args()

    random.seed(args.seed)

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

    match args.filter_score:
        case "default":
            score_func = default_filter_scoring
        case "original":
            score_func = original_filter_scoring
        case _:
            raise ValueError(f"Invalid filter: {args.filter}")
    # Add other filter options here if needed

    if args.random_sample:
        if output_file == input_file:
            if (
                input(
                    "Output file will be overwritten. This is not usually what you want together with the random sampling parameter. Continue? (y/n): "
                ).lower()
                != "y"
            ):
                print("Operation cancelled.")
                return
        if args.max_questions:
            if (
                input(
                    f"You have specified both max_questions ({args.max_questions}) and random_sample ({args.random_sample}). This means filtering will take a prefix according to score and then random sample from that prefix. Continue? (y/n): "
                ).lower()
                != "y"
            ):
                print("Operation cancelled.")
                return

    filter_fqs(
        input_file,
        output_file,
        score_func,
        args.max_questions,
        args.random_sample,
        strict=not args.only_preference,
    )


if __name__ == "__main__":
    main()

# Example run command:
# python src/filter_fqs.py -i path/to/input.jsonl -o path/to/output_filtered.jsonl -n 1000 -f original -r 500 -s 123
