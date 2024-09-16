"""
(1) Test consistency evaluation pipeline (per-question) end-to-end.

(2) Test that LoadForecaster works on the output of a consistency evaluation pipeline.
"""

import subprocess
import pytest
from pathlib import Path
import filecmp


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


checkers = ["NegChecker", "AndChecker"]

starting_file = "src/data/other/high-quality-questions-all-domains.jsonl"  # has to be in this question stage instead of full FQs because of how the first part of the pipeline works. if we want to use full FQs here, start from the second step of the pipeline (verified FQs)

commands = [
    f"python src/format_and_verify_questions.py --file_path {starting_file} -d test -s True -F True -o verified_questions_test.jsonl",
    #
    "python src/generate_related_questions.py -n 3 -q 3 --input_file src/data/fq/test/verified_questions_test.jsonl --output_file src/data/fq/test/from_related_test.jsonl",
    #
    "python src/format_and_verify_questions.py --file_path src/data/fq/test/from_related_test.jsonl -d test -o from-related-verified_test.jsonl -s True -F True",
    #
    "python src/instantiation.py --data_path src/data/fq/test/from-related-verified_test.jsonl -r"
    + " ".join(f" -k {checker}" for checker in checkers)
    + " --n_source_questions 3 --max_tuples_per_source 3 --tuple_dir src/data/tuples_test",
    #
    "python src/evaluation.py --tuple_dir src/data/tuples_test -f BasicForecaster --forecaster_options model=gpt-4o-mini --run"
    + " ".join(f" -k {checker}" for checker in checkers)
    + " --eval_by_source -t 5 --output_dir src/data/forecasts/BasicForecaster_test",
    #
    "python src/evaluation.py --tuple_dir src/data/tuples_test -f LoadForecaster --forecaster_options load_dir=src/data/forecasts/BasicForecaster_test --run"
    + " ".join(f" -k {checker}" for checker in checkers)
    + " --eval_by_source -t 5 --output_dir src/data/forecasts/LoadForecaster_test",
]


def run_pipeline_command(command):
    returncode, stdout, stderr = run_command(command)

    print(f"Return Code: {returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")

    assert returncode == 0, f"Command failed: {command}\nError: {stderr}"


def expected_files(test_exist: bool = False):
    files = [
        "src/data/fq/test/verified_questions_test.jsonl",
        "src/data/fq/test/from_related_test.jsonl",
        "src/data/fq/test/from-related-verified_test.jsonl",
    ]

    for checker in checkers:
        files.extend(
            [
                f"src/data/tuples_test/{checker}.jsonl",
                f"src/data/forecasts/BasicForecaster_test/{checker}.jsonl",
                f"src/data/forecasts/BasicForecaster_test/stats_{checker}.json",
                f"src/data/forecasts/LoadForecaster_test/{checker}.jsonl",
                f"src/data/forecasts/LoadForecaster_test/stats_{checker}.json",
            ]
        )

    files.extend(
        [
            "src/data/forecasts/BasicForecaster_test/stats_by_source_question.json",
            "src/data/forecasts/LoadForecaster_test/stats_by_source_question.json",
        ]
    )

    if test_exist:
        for file_path in files:
            assert Path(
                file_path
            ).exists(), f"Expected output file does not exist: {file_path}"

        # assert that the stats files have the same content
        for checker in checkers:
            assert filecmp.cmp(
                f"src/data/forecasts/BasicForecaster_test/stats_{checker}.json",
                f"src/data/forecasts/LoadForecaster_test/stats_{checker}.json",
            ), f"Stats files differ for checker {checker}"

    return files


def test_pipeline_end_to_end():
    assert Path(
        starting_file
    ).exists(), f"Starting file does not exist: {starting_file}"
    print(f"Starting file: {starting_file}")

    output_files = expected_files(test_exist=False)
    print("\033[1mDeleting the following files:\033[0m")
    for file_path in output_files:
        print(f"  {file_path}")

    # delete all produced files
    for file_path in output_files:
        if Path(file_path).exists():
            Path(file_path).unlink()

    for command in commands:
        print(f"\033[1mRunning command: {command}\033[0m")
        run_pipeline_command(command)

    expected_files(test_exist=True)


if __name__ == "__main__":
    pytest.main([__file__])
