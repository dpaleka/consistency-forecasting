import subprocess
import pytest
from pathlib import Path


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


commands = [
    "python src/ground_truth_run.py --forecaster_class BasicForecaster --forecaster_options model=gpt-4o-mini --input_file src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl --num_lines 8 --run --async --output_dir src/data/forecasts/test_output",
]


def run_ground_truth_command(command):
    returncode, stdout, stderr = run_command(command)

    print(f"Return Code: {returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")

    assert returncode == 0, f"Command failed: {command}\nError: {stderr}"


def expected_files(test_exist: bool = False):
    files = [
        "src/data/forecasts/test_output/ground_truth_results.jsonl",
        "src/data/forecasts/test_output/ground_truth_summary.json",
        "src/data/forecasts/test_output/calibration_plot_logit.png",
        "src/data/forecasts/test_output/calibration_plot_linear.png",
    ]

    if test_exist:
        for file_path in files:
            assert Path(
                file_path
            ).exists(), f"Expected output file does not exist: {file_path}"

    return files


def test_ground_truth_run():
    input_file = "src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl"
    assert Path(input_file).exists(), f"Input file does not exist: {input_file}"
    print(f"Using input file: {input_file}")

    output_files = expected_files(test_exist=False)
    print("\033[1mDeleting the following files:\033[0m")
    for file_path in output_files:
        print(f"  {file_path}")

    # Delete all produced files
    for file_path in output_files:
        if Path(file_path).exists():
            Path(file_path).unlink()

    for command in commands:
        print(f"\033[1mRunning command: {command}\033[0m")
        run_ground_truth_command(command)

    expected_files(test_exist=True)


if __name__ == "__main__":
    pytest.main([__file__])
