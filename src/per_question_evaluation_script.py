import argparse
import subprocess


# python consistency_evaluation_pipeline.py --input_file path/to/input_questions.jsonl --output_dir path/to/output
def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


def main():
    parser = argparse.ArgumentParser(description="Run consistency evaluation pipeline")

    parser.add_argument(
        "-t",
        "--tuple_dir",
        # required=True,
        default="src/data/tuples/per-question-experiment/",
        help="Directory with tuple files",
    )

    parser.add_argument(
        "--eval_dir",
        # required=True,
        default="src/data/forecasts/per-question-experiment",
        help="Directory to store evaluation files",
    )

    parser.add_argument(
        "-k",
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
        "-f", "--forecaster", default="BasicForecaster", help="Forecaster to use"
    )

    parser.add_argument(
        "-o",
        "--forecaster_options",
        # default="model=gpt-4o-mini",
        help="Options for the forecaster",
    )

    parser.add_argument(
        "-c",
        "--config_path",
        # default="forecasters/forecaster_configs/advanced/cheap_haiku.yaml",
        help="Options for the forecaster",
    )

    args = parser.parse_args()

    # Create output directory
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Construct the evaluation command
    eval_command = f"python src/evaluation.py --tuple_dir {args.tuple_dir} -f {args.forecaster} --run {' '.join(f'-k {checker}' for checker in args.checkers)} --eval_by_source -t 4 --output_dir {args.eval_dir}"

    # Add optional arguments if they are provided
    if args.forecaster_options:
        eval_command += f" --forecaster_options {args.forecaster_options}"
    if args.config_path:
        eval_command += f" --config_path {args.config_path}"

    # Define pipeline steps
    steps = [
        eval_command,
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

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
