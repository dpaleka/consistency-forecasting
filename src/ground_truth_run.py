import sys
import io
import json
from pathlib import Path
import click
import logging

from common.path_utils import get_data_path
from common.utils import make_json_serializable
from common.datatypes import ForecastingQuestion
from evaluation_utils.utils import (
    load_forecaster,
    create_output_directory,
    write_to_dirs,
)
from evaluation_utils.common_options import common_options
from evaluation_utils.proper_scoring import (
    proper_scoring_rule,
    scoring_functions,
    plot_calibration,
    calculate_calibration,
)

BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


@click.command()
@common_options
@click.option(
    "--input_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input JSONL file containing Forecasting Questions",
)
@click.option(
    "--scoring_rule",
    type=click.Choice(list(scoring_functions.keys())),
    default="log_score",
    help="Scoring rule to use for evaluation",
)
def main(
    forecaster_class: str,
    config_path: str,
    model: str | None,
    input_file: str,
    scoring_rule: str,
    num_lines: int,
    run: bool,
    load_dir: str | None = None,
    is_async: bool = False,
    output_dir: str | None = None,
):
    forecaster = load_forecaster(forecaster_class, config_path, model)

    output_directory, most_recent_directory = create_output_directory(
        forecaster, model, BASE_FORECASTS_OUTPUT_PATH, output_dir
    )
    dirs_to_write = [output_directory, most_recent_directory]

    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    forecasting_questions = [ForecastingQuestion.model_validate(line) for line in data]
    print(f"Loaded {len(forecasting_questions)} forecasting questions")

    data, forecasting_questions = zip(
        *[
            (line, fq)
            for line, fq in zip(data, forecasting_questions)
            if fq.resolution is not None
        ]
    )
    assert len(data) == len(
        forecasting_questions
    ), "Data and forecasting questions have different lengths"
    print(f"Filtered to {len(forecasting_questions)} questions with resolutions")

    num_lines = min(num_lines, len(forecasting_questions))
    print(f"Running on {num_lines} questions")

    results = []
    for line, fq in zip(data[:num_lines], forecasting_questions[:num_lines]):
        forecast = forecaster.call_full(fq)
        score = proper_scoring_rule(fq, forecast, scoring_functions[scoring_rule])
        assert line == make_json_serializable(
            fq.to_dict()
        ), "line and make_json_serializable(fq.to_dict()) are not equal"
        results.append(
            {
                "question": line,
                "forecast": make_json_serializable(forecast.to_dict()),
                "prob": forecast.prob,
                "resolution": fq.resolution,
                "score": score,
            }
        )

    output_filename = "ground_truth_results.jsonl"
    write_to_dirs(
        results=results,
        filename=output_filename,
        dirs_to_write=dirs_to_write,
        overwrite=True,
    )
    print(f"Results written to {output_filename}")

    # Calculate and print summary statistics
    total_score = sum(result["score"] for result in results)
    average_score = total_score / len(results)
    calibration_error_data: dict = calculate_calibration(
        [fq.resolution for fq in forecasting_questions[:num_lines]],
        [result["prob"] for result in results],
        num_bins=10,
    )
    calibration_error = calibration_error_data["calibration_error"]

    summary = {
        "total_questions": len(results),
        "average_score": average_score,
        "calibration_error": calibration_error,
        "scoring_rule": scoring_rule,
        "forecaster": forecaster.__class__.__name__,
        "model": model,
    }

    print("\nGround Truth Summary:")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Average {scoring_rule}: {summary['average_score']:.4f}")
    print(f"Forecaster: {summary['forecaster']}")
    print(f"Model: {summary['model']}")

    # Write summary to file
    summary_filename = "ground_truth_summary.json"
    for dir in dirs_to_write:
        json.dump(summary, open(dir / summary_filename, "w"), indent=4)
        print(f"\nSummary written to {dir / summary_filename}")

    # Plot calibration error
    probs = [result["prob"] for result in results]
    outcomes = [result["resolution"] for result in results]
    to_plot = True
    if to_plot:
        plot = plot_calibration(probs, outcomes)
        plot_filename = "calibration_plot.png"
        for dir in dirs_to_write:
            plot.savefig(dir / plot_filename)
        print(f"Calibration plot written to {plot_filename}")


if __name__ == "__main__":
    main()

    # Example run command
print("Example run command:")
print(
    "python ground_truth_run.py --forecaster_class BasicForecaster --config_path path/to/config.json --model gpt-3.5-turbo --input_file path/to/input.jsonl --scoring_rule log_score --output_dir path/to/output"
)
