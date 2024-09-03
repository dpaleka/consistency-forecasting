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
from evaluation_utils.proper_scoring import proper_scoring_rule, scoring_functions

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


if __name__ == "__main__":
    main()

    # Example run command
print("Example run command:")
print(
    "python ground_truth_run.py --forecaster_class BasicForecaster --config_path path/to/config.json --model gpt-3.5-turbo --input_file path/to/input.jsonl --scoring_rule log_score --output_dir path/to/output"
)
