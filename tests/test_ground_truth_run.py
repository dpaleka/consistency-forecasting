"""
(1) Test the ground truth forecasting evaluation pipeline end-to-end.

(2) Test that LoadForecaster works on the output of the ground truth forecasting evaluation pipeline.
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


commands_forecaster_class = [
    "python src/ground_truth_run.py --forecaster_class BasicForecaster --forecaster_options model=gpt-4o-mini --input_file src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl --num_lines 3 --run --output_dir src/data/forecasts/test_output_basic",
    "python src/ground_truth_run.py --forecaster_class LoadForecaster --forecaster_options load_dir=src/data/forecasts/test_output_basic --input_file src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl --num_lines 3 --run --output_dir src/data/forecasts/test_output_load",
]

commands_custom_forecaster = [
    "python src/ground_truth_run.py --custom_path src/forecasters/basic_forecaster.py::BasicForecaster --forecaster_options model=gpt-4o-mini --input_file src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl --num_lines 3 --run --output_dir src/data/forecasts/test_output_custom",
]


def run_ground_truth_command(command):
    returncode, stdout, stderr = run_command(command)

    print(f"Return Code: {returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")

    assert returncode == 0, f"Command failed: {command}\nError: {stderr}"


def expected_files(output_dirs: list[str], test_exist: bool = False):
    files = [
        [
            f"{output_dir}/ground_truth_results.jsonl",
            f"{output_dir}/ground_truth_summary.json",
            f"{output_dir}/calibration_plot_logit.png",
            f"{output_dir}/calibration_plot_linear.png",
        ]
        for output_dir in output_dirs
    ]
    files = [file for sublist in files for file in sublist]

    print(files)

    if test_exist:
        for file_path in files:
            assert Path(
                file_path
            ).exists(), f"Expected output file does not exist: {file_path}"

        if (
            len(output_dirs) == 2
            and "basic" in output_dirs[0]
            and "load" in output_dirs[1]
        ):
            assert filecmp.cmp(
                f"{output_dirs[0]}/ground_truth_results.jsonl",
                f"{output_dirs[1]}/ground_truth_results.jsonl",
            ), "Results files differ"

    return files


@pytest.mark.parametrize(
    "commands,output_dirs",
    [
        pytest.param(
            commands_forecaster_class,
            [
                "src/data/forecasts/test_output_basic",
                "src/data/forecasts/test_output_load",
            ],
            id="forecaster_class",
        ),
        pytest.param(
            commands_custom_forecaster,
            ["src/data/forecasts/test_output_custom"],
            id="custom_forecaster",
        ),
    ],
)
def test_ground_truth_run(commands, output_dirs):
    input_file = "src/data/fq/real/metaculus_cleaned_formatted_20240501_20240815.jsonl"
    assert Path(input_file).exists(), f"Input file does not exist: {input_file}"
    print(f"Using input file: {input_file}")

    output_files = expected_files(output_dirs, test_exist=False)
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

    expected_files(output_dirs, test_exist=True)


if __name__ == "__main__":
    pytest.main([__file__])
