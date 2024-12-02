import argparse
import subprocess
import json
import os
from collections import defaultdict


def check_tuple_generation(tuple_dir, checkers):
    tuple_counts = defaultdict(lambda: defaultdict(int))

    for checker in checkers:
        tuple_file = os.path.join(tuple_dir, f"{checker}.jsonl")
        if not os.path.exists(tuple_file):
            print(f"Warning: No tuple file found for {checker}")
            continue

        with open(tuple_file, "r") as f:
            for line in f:
                tuple_data = json.loads(line)
                metadata = tuple_data.get("metadata", {})
                base_sentences = metadata.get("base_sentences", {})

                # Count tuples for both P and Q
                for key in ["P", "Q"]:
                    source_id = base_sentences.get(key, {}).get("id")
                    if source_id:
                        tuple_counts[checker][source_id] += 1

    all_good = True
    for checker in checkers:
        if checker not in tuple_counts:
            print(f"Error: No tuples generated for {checker}")
            all_good = False
        else:
            zero_tuple_questions = [
                q for q, count in tuple_counts[checker].items() if count == 0
            ]
            if zero_tuple_questions:
                print(
                    f"Error: No tuples generated for {checker} for questions: {zero_tuple_questions}"
                )
                all_good = False

    if all_good:
        print("Success: At least one tuple generated for each check and each question.")
    else:
        print("Error: Some checks or questions have no tuples generated.")
        return False
    return True


# python consistency_evaluation_pipeline.py --input_file path/to/input_questions.jsonl --output_dir path/to/output
def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


starting_file = "src/data/fq/real/20240501_20240815.jsonl"
# starting_file = "src/data/other/high-quality-questions-all-domains.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Run consistency evaluation pipeline")
    parser.add_argument(
        "--input_file",
        default=starting_file,
        # required=True,
        help="Path to input file with questions",
    )
    parser.add_argument(
        "--data_dir",
        # required=True,
        default="src/data/fq/experiment/",
        help="Directory to store intermediary files",
    )

    parser.add_argument(
        "--tuple_dir",
        # required=True,
        default="src/data/tuples/experiment/",
        help="Directory to store tuple files",
    )

    parser.add_argument(
        "--eval_dir",
        # required=True,
        default="src/data/forecasts/experiment",
        help="Directory to store evaluation files",
    )

    parser.add_argument(
        "--num_source",
        type=int,
        default=3,
        help="Number of source questions to use",
    )
    parser.add_argument(
        "--related_questions",
        type=int,
        default=7,
        help="Number of related questions to generate for each source question",
    )
    parser.add_argument(
        "--tuples_per_source",
        type=int,
        default=4,
        help="Max number of tuples to generate per source question, per check",
    )
    parser.add_argument(
        "--checkers",
        nargs="+",
        default=[
            "NegChecker",
            "ParaphraseChecker",
            # "ConsequenceChecker",
            "CondChecker",
            "AndOrChecker",
            "ButChecker",
            "CondCondChecker",
        ],
        help="Checkers to use",
    )
    parser.add_argument(
        "--forecaster", default="BasicForecaster", help="Forecaster to use"
    )
    parser.add_argument(
        "--forecaster_options",
        default="model=gpt-4o-mini",
        help="Options for the forecaster",
    )
    args = parser.parse_args()

    # Create output directory
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Define pipeline steps
    steps = [
        # f"python src/format_and_verify_questions.py --file_path {args.input_file} -m 20 -d test -F True -o verified_questions.jsonl --overwrite",  # add -s True if input file is synthetic
        # f"python src/generate_related_questions.py -n {args.num_source} -q {args.related_questions} --input_file src/data/fq/test/verified_questions.jsonl --output_file src/data/fq/test/related_questions.jsonl",
        # "python src/format_and_verify_questions.py --file_path src/data/fq/test/related_questions.jsonl -m 30 -d test -o verified_related_questions.jsonl -s True -F True --overwrite -v none",
        # f"python src/instantiation.py --data_path src/data/fq/test/verified_related_questions_unverified.jsonl -r {' '.join(f'-k {checker}' for checker in args.checkers)} --max_tuples_per_source {args.tuples_per_source} --tuple_dir {args.tuple_dir}",
        # f"python src/evaluation.py --tuple_dir {args.tuple_dir} -f {args.forecaster} --forecaster_options {args.forecaster_options} --run {' '.join(f'-k {checker}' for checker in args.checkers)} --eval_by_source -t 4 --output_dir {args.eval_dir}",
        f"python src/eval_parser.py --input_file {args.eval_dir}/stats_by_source_question.json --output_file {args.eval_dir}/per_question_consistency.json",
    ]

    # Run pipeline steps
    for step in steps:
        print(f"Running: {step}")
        returncode, stdout, stderr = run_command(step)
        if returncode != 0:
            print(f"Error in step: {step}")
            print(f"STDERR:\n{stderr}")
            return
        print("Step completed successfully")

        # Check tuple generation after instantiation step
        if "instantiation.py" in step:
            print("Checking tuple generation...")
            if not check_tuple_generation(args.tuple_dir, args.checkers):
                print("Tuple generation check failed. Stopping pipeline.")
                return

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
