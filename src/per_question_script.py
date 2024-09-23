import argparse
import subprocess


# python consistency_evaluation_pipeline.py --input_file path/to/input_questions.jsonl --output_dir path/to/output
def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


# starting_file = "src/data/fq/real/20240501_20240815.jsonl"
starting_file = "src/data/other/high-quality-questions-all-domains.jsonl"


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
        default="src/data/tuples_experiment/",
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
            "ConsequenceChecker",
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
        # f"python src/format_and_verify_questions.py --file_path {args.input_file} -m 10 -d test -F True -s True -o verified_questions.jsonl --overwrite",  # add -s True if input file is synthetic
        # f"python src/generate_related_questions.py -n {args.num_source} -q {args.related_questions} --input_file src/data/fq/test/verified_questions.jsonl --output_file src/data/fq/test/related_questions.jsonl",
        # "python src/format_and_verify_questions.py --file_path src/data/fq/test/related_questions.jsonl -m 50 -d test -o verified_related_questions.jsonl -s True -F True --overwrite",
        f"python src/instantiation.py --data_path src/data/fq/test/verified_related_questions.jsonl -r {' '.join(f'-k {checker}' for checker in args.checkers)} --max_tuples_per_source {args.tuples_per_source} --tuple_dir {args.tuple_dir}",
        f"python src/evaluation.py --tuple_dir {args.tuple_dir} -f {args.forecaster} --forecaster_options {args.forecaster_options} --run {' '.join(f'-k {checker}' for checker in args.checkers)} --eval_by_source -t 4 --output_dir {args.eval_dir}",
    ]

    # Run pipeline steps
    for step in steps:
        print(f"Running: {step}")
        returncode, stdout, stderr = run_command(step)
        # if stderr:
        #     print("Debug Output:")
        #     print(stderr)
        if returncode != 0:
            print(f"Error in step: {step}")
            # print(f"STDOUT:\n{stdout}")
            print(f"STDERR:\n{stderr}")
            return
        print("Step completed successfully")

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
