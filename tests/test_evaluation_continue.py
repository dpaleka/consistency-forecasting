import json
import shutil
import subprocess
from pathlib import Path
import pytest
from common.path_utils import get_data_path

BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"
TEST_OUTPUT_DIR = BASE_FORECASTS_OUTPUT_PATH / "test_continue_feature"


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


def run_evaluation_command(command):
    returncode, stdout, stderr = run_command(command)

    print(f"Return Code: {returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")

    assert returncode == 0, f"Command failed: {command}\nError: {stderr}"


def expected_files(test_exist: bool = False):
    files = [
        TEST_OUTPUT_DIR / "NegChecker.jsonl",
        TEST_OUTPUT_DIR / "stats_summary.json",
    ]

    if test_exist:
        for file_path in files:
            assert (
                file_path.exists()
            ), f"Expected output file does not exist: {file_path}"

    return files


@pytest.fixture(scope="module")
def cleanup_test_output():
    # Setup: ensure the test output directory is empty
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    yield

    # Teardown: remove the test output directory
    shutil.rmtree(TEST_OUTPUT_DIR)


def test_continue_feature(cleanup_test_output):
    # Step 1: Run the evaluation for a small number of lines
    TUPLE_DIR = get_data_path() / "tuples" / "scraped"
    print(f"{TEST_OUTPUT_DIR=}")
    command1 = (
        "python src/evaluation.py -f BasicForecaster --run --async -n 2 "
        f"-k NegChecker --tuple_dir {TUPLE_DIR} "
        f"--output_dir {TEST_OUTPUT_DIR} "
        "-o model=gpt-4o-mini-2024-07-18"
    )
    run_evaluation_command(command1)

    # Check if the output files were created
    expected_files(test_exist=True)

    # Count the number of lines in the output file
    with open(TEST_OUTPUT_DIR / "NegChecker.jsonl", "r") as f:
        initial_line_count = sum(1 for _ in f)
    assert initial_line_count == 2, f"Expected 2 lines, but got {initial_line_count}"

    # Step 2: Run the evaluation again with --continue flag and more lines
    command2 = (
        "python src/evaluation.py -f BasicForecaster --run --async -n 5 "
        f"-k NegChecker --tuple_dir {TUPLE_DIR} "
        f"--output_dir {TEST_OUTPUT_DIR} "
        "-o model=gpt-4o-mini-2024-07-18"
    )
    run_evaluation_command(command2)

    # Check if the output file was updated
    with open(TEST_OUTPUT_DIR / "NegChecker.jsonl", "r") as f:
        final_line_count = sum(1 for _ in f)
    assert (
        final_line_count == 5
    ), f"Expected 5 lines after continue, but got {final_line_count}"

    # Check if the stats were updated
    with open(TEST_OUTPUT_DIR / "stats_summary.json", "r") as f:
        stats = json.load(f)
    assert (
        stats["NegChecker"]["overall"]["default"]["num_samples"] == 5
    ), f"Expected 5 samples in stats, but got {stats['NegChecker']['overall']['default']['num_samples']}"

    print("Test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__])
