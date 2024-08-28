#!/usr/bin/env python3

import argparse
import subprocess


def run_command(command, dry_run=False):
    print(f"\n\033[1mRunning command: {command}\033[0m\n")
    if dry_run:
        print("Dry run, not executing command")
        return
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Error message: {stderr.decode('utf-8')}")
    else:
        print(stdout.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Scrape questions from various sources."
    )
    parser.add_argument(
        "-s", "--start_date", required=True, help="Start date in YYYYMMDD format"
    )
    parser.add_argument(
        "-e", "--end_date", required=True, help="End date in YYYYMMDD format"
    )
    parser.add_argument(
        "-n",
        "--num_questions",
        type=int,
        default=500,
        help="Number of questions to scrape",
    )
    parser.add_argument(
        "-o",
        "--output_infix",
        default="cleaned_formatted",
        help="Infix for output files",
    )
    parser.add_argument(
        "-m", "--model", default="gpt-4o", help="Model to use for formatting"
    )
    parser.add_argument(
        "-x", "--max_questions", type=int, help="Maximum number of questions to process"
    )
    parser.add_argument(
        "-d",
        "--data_source",
        choices=["manifold", "metaculus"],
        required=True,
        help="Data source to scrape from",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform a dry run without executing commands",
    )
    parser.add_argument(
        "--skip",
        required=False,
        nargs="+",
        help="List of steps to skip",
        default=[],
        choices=[
            "scrape",
            "count_entries",
            "add_body",
            "reshape_questions",
            "format_and_verify_questions",
        ],
    )

    args = parser.parse_args()

    # Set max_questions to num_questions if not provided
    max_questions = args.max_questions if args.max_questions else args.num_questions

    # Construct the base command
    if "scrape" not in args.skip:
        base_command = f"python {args.data_source}.py -start {args.start_date} -end {args.end_date} -num {args.num_questions}"
        run_command(base_command, dry_run=args.dry_run)

    # Run the commands
    if "count_entries" not in args.skip:
        run_command(
            f"python count_entries.py -f {args.data_source}/{args.data_source}_{args.start_date}_{args.end_date}.json",
            dry_run=args.dry_run,
        )
    if "add_body" not in args.skip:
        run_command(
            f"python add_body.py {args.data_source}/{args.data_source}_{args.start_date}_{args.end_date}.json",
            dry_run=args.dry_run,
        )
    if "reshape_questions" not in args.skip:
        run_command(
            f"python reshape_questions.py --filename {args.data_source}/{args.data_source}_{args.start_date}_{args.end_date}.json",
            dry_run=args.dry_run,
        )
    if "format_and_verify_questions" not in args.skip:
        format_command = (
            f"python ../../src/format_and_verify_questions.py "
            f"-f {args.data_source}/{args.data_source}_{args.start_date}_{args.end_date}.jsonl "
            f"-d real "
            f"-o {args.data_source}_{args.output_infix}_{args.start_date}_{args.end_date}.jsonl "
            f"--max_questions {max_questions} "
            f"--overwrite "
            f"-F True "
            f"-M {args.model}"
        )
        run_command(format_command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()


# Example usage:
# ./scrape_question.py -d manifold -s 20240301 -e 20240701 -n 500 -o manifold_cleaned_formatted -m gpt-4o
