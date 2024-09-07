"""
Test evaluation pipeline around a single question.
"""

"""
Pipeline for going from pre-set source questions to evaluation score. 
Steps:
1. `python src/format_and_verify_questions.py -d synthetic -o synth-verified.jsonl -s True -F True`
(input file is src\data\other\high-quality-questions-all-domains.jsonl)
Output is new file src\data\fq\synthetic\synth-verified.jsonl. Questions from high-quality-question-all-domains.jsonl that pass verification from format_and_verify_questions.py
2. `python src/generate_related_questions.py -n 10 -q 5`
Input is synth-verified.jsonl. Outputs into src\data\other\from_related.jsonl
3. `python src/format_and_verify_questions.py -d synthetic -o from-related-verified.jsonl -s True -F True`
Input file is src\data\other\from_related.jsonl. Outputs to src\data\fq\synthetic\from-related-verified.jsonl
4. `python src/instantiation.py -r --n_source_questions 10 --max_tuples_per_source 10`
Input file is src\data\fq\synthetic\from-related-verified.jsonl. Outputs to src\data\tuples_rel
Added parameters to control by number of source questions and tuples per source question rather than line count
5. `python src/evaluation.py -f BasicForecaster -m gpt-4o-mini --run -k NegChecker -k AndChecker -k CondCondChecker -s -t 5 | tee see_eval.txt`
Input is src\data\tuples_rel. Outputs to src\data\forecasts\BasicForecaster_MM_DD_hh_mm
"""

import subprocess
import pytest
from pathlib import Path


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


checkers = ["NegChecker", "AndChecker"]

commands = [
    "python src/format_and_verify_questions.py --file_path src/data/other/high-quality-questions-all-domains.jsonl -d test -s True -F True -o high_quality_questions_all_domains_test.jsonl",
    "python src/generate_related_questions.py -n 3 -q 3 --input_file src/data/fq/test/high_quality_questions_all_domains_test.jsonl --output_file src/data/fq/test/from_related_test.jsonl",
    "python src/format_and_verify_questions.py --file_path src/data/fq/test/from_related_test.jsonl -d test -o from-related-verified_test.jsonl -s True -F True",
    "python src/instantiation.py --data_path src/data/fq/test/from-related-verified_test.jsonl -r"
    + " ".join(f" -k {checker}" for checker in checkers)
    + " --n_source_questions 3 --max_tuples_per_source 3 --tuple_dir src/data/tuples_test",
    "python src/evaluation.py --tuple_dir src/data/tuples_test -f BasicForecaster -m gpt-4o-mini --run"
    + " ".join(f" -k {checker}" for checker in checkers)
    + " -s -t 5 --output_dir src/data/forecasts/BasicForecaster_test",
]


def run_pipeline_command(command):
    returncode, stdout, stderr = run_command(command)

    print(f"Return Code: {returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")

    assert returncode == 0, f"Command failed: {command}\nError: {stderr}"


def expected_files(test_exist: bool = False):
    files = [
        "src/data/fq/test/high_quality_questions_all_domains_test.jsonl",
        "src/data/fq/test/from_related_test.jsonl",
        "src/data/fq/test/from-related-verified_test.jsonl",
    ]

    for checker in checkers:
        files.extend(
            [
                f"src/data/tuples_test/{checker}.jsonl",
                f"src/data/forecasts/BasicForecaster_test/{checker}.jsonl",
                f"src/data/forecasts/BasicForecaster_test/stats_{checker}.json",
            ]
        )

    files.append(
        "src/data/forecasts/BasicForecaster_test/stats_by_source_question.json"
    )

    if test_exist:
        for file_path in files:
            assert Path(
                file_path
            ).exists(), f"Expected output file does not exist: {file_path}"

    return files


def test_pipeline_end_to_end():
    starting_file = "src/data/other/high-quality-questions-all-domains.jsonl"
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
