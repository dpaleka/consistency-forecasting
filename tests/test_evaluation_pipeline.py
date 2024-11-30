"""
(1) Test consistency evaluation pipeline (per-question) end-to-end.

(2) Test that LoadForecaster works on the output of a consistency evaluation pipeline.
(3) Test LoadForecaster shorthand and skip_validation functionality.
"""

import subprocess
import pytest
from pathlib import Path
import json
from src.common.path_utils import get_data_path


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
    + " --n_source_questions 3 --max_tuples_per_source 3 --tuple_dir src/data/test/tuples",
    #
    "python src/evaluation.py --tuple_dir src/data/test/tuples -f BasicForecaster --forecaster_options model=gpt-4o-mini --run"
    + " ".join(f" -k {checker}" for checker in checkers)
    + " --eval_by_source -t 5 --output_dir src/data/forecasts/BasicForecaster_test",
    #
    "python src/evaluation.py --tuple_dir src/data/test/tuples -f LoadForecaster --forecaster_options load_dir=src/data/forecasts/BasicForecaster_test --run"
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
                f"src/data/test/tuples/{checker}.jsonl",
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

        # We do not assert that the stats files have the same content, because arbitrage violation is not deterministic.
        # TODO: figure out a way to test this.
        # for checker in checkers:
        #    stats_dict_basic = json.load(open(f"src/data/forecasts/BasicForecaster_test/stats_{checker}.json"))
        #    stats_dict_load = json.load(open(f"src/data/forecasts/LoadForecaster_test/stats_{checker}.json"))
        #    TODO: figure out what fields *should* be the same and then assert those.

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


def test_load_shorthand():
    # Create a test directory
    temp_dir = get_data_path() / "test" / "load_forecaster_shorthand"
    temp_dir.mkdir(parents=True, exist_ok=True)
    expected_files = [
        "NegChecker.jsonl",
        "CondChecker.jsonl",
        "stats_NegChecker.json",
        "stats_CondChecker.json",
        "stats_summary.json",
    ]

    print("\033[1mDeleting the following files:\033[0m")
    for file_path in expected_files:
        print(f"  {file_path}")

    # Create a simple forecast file with P/Q format
    simple_neg_forecasts = [
        {
            "line": {
                "P": {
                    "question": {"title": "Test Question 1 (P)"},
                    "forecast": {"prob": 0.7},
                },
                "not_P": {
                    "question": {"title": "Test Question 1 (not P)"},
                    "forecast": {"prob": 0.3},
                },
            }
        },
        {
            "line": {
                "P": {
                    "question": {"title": "Test Question 2 (P)"},
                    "forecast": {"prob": 0.2},
                },
                "not_P": {
                    "question": {"title": "Test Question 2 (not P)"},
                    "forecast": {"prob": 0.5},
                },
            }
        },
    ]

    simple_cond_forecasts = [
        {
            "line": {
                "P": {
                    "question": {"title": "Test Question 3 (P)"},
                    "forecast": {"prob": 0.5},
                },
                "Q_given_P": {
                    "question": {"title": "Test Question 3 (Q_given_P)"},
                    "forecast": {"prob": 0.4},
                },
                "P_and_Q": {
                    "question": {"title": "Test Question 3 (P_and_Q)"},
                    "forecast": {"prob": 0.2},
                },
            }
        },
        {
            "line": {
                "P": {
                    "question": {"title": "Test Question 4 (P)"},
                    "forecast": {"prob": 0.3},
                },
                "Q_given_P": {
                    "question": {"title": "Test Question 4 (Q_given_P)"},
                    "forecast": {"prob": 0.6},
                },
                "P_and_Q": {
                    "question": {"title": "Test Question 4 (P_and_Q)"},
                    "forecast": {"prob": 0.3},
                },
            }
        },
    ]

    data = {
        "NegChecker.jsonl": simple_neg_forecasts,
        "CondChecker.jsonl": simple_cond_forecasts,
    }

    for file_name, forecasts in data.items():
        forecast_file = temp_dir / file_name
        with open(forecast_file, "w") as f:
            for forecast in forecasts:
                f.write(json.dumps(forecast) + "\n")

    # Test the --load shorthand (should work with minimal validation by default)
    command = f"python src/evaluation.py --load {temp_dir} -t 2"
    returncode, stdout, stderr = run_command(command)
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")
    assert returncode == 0, f"Command failed with --load shorthand: {stderr}"

    # Verify that specific output files were produced by the first command
    for file in expected_files:
        output_file = Path(temp_dir) / file
        assert output_file.exists(), f"Expected output file {file} was not produced"


if __name__ == "__main__":
    pytest.main([__file__])
