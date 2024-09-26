import sys
import io
import json
from pathlib import Path
import click
import logging
import asyncio
import functools
from tqdm import tqdm

from common.path_utils import get_data_path
from common.utils import make_json_serializable, compare_dicts
from common.llm_utils import parallelized_call, reset_global_semaphore
from common.datatypes import ForecastingQuestion, Forecast
from forecasters.create import make_forecaster
from evaluation_utils.utils import (
    create_output_directory,
    write_to_dirs,
)
from common.utils import round_floats
from evaluation_utils.common_options import common_options, get_forecaster_config
from evaluation_utils.proper_scoring import (
    proper_score,
    decompose_brier_score,
    scoring_functions,
    plot_calibration,
    calculate_calibration,
    scale_brier_score,
    platt_scaling,
)


BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def make_result_dict(line: dict, fq: ForecastingQuestion, forecast: Forecast):
    log_score = proper_score(
        probs=[forecast.prob],
        outcomes=[fq.resolution],
        scoring_function=scoring_functions["log_score"],
    )
    brier_score = proper_score(
        probs=[forecast.prob],
        outcomes=[fq.resolution],
        scoring_function=scoring_functions["brier_score"],
    )
    brier_score_scaled = scale_brier_score(brier_score)

    assert (
        (
            line_fq_differences := compare_dicts(
                line, make_json_serializable(fq.to_dict())
            )
        )
        == []
    ), f"line and make_json_serializable(fq.to_dict()) are not equal, differences: {line_fq_differences}"

    return {
        "question": line,
        "forecast": make_json_serializable(forecast.to_dict()),
        "prob": forecast.prob,
        "resolution": fq.resolution,
        "log_score": round_floats(log_score, precision=4),
        "brier_score": round_floats(brier_score, precision=4),
        "brier_score_scaled": round_floats(brier_score_scaled, precision=4),
    }


@click.command()
@common_options
@click.option(
    "--input_file",
    type=click.Path(exists=True),
    required=False,
    help="Path to the input JSONL file containing Forecasting Questions",
)
@click.option(
    "--platt_scaling_factor",
    type=float,
    default=None,
    help="Factor to use for Platt scaling. If not provided, will be calculated from the data. If the data does not have any non-None resolutions, it will be set to 1.",
)
def main(
    forecaster_class: str | None,
    custom_path: str | None,
    config_path: str | None,
    forecaster_options: list[str] | None,
    num_lines: int,
    run: bool,
    input_file: str | None = None,
    load_dir: str | None = None,
    is_async: bool = False,
    output_dir: str | None = None,
    platt_scaling_factor: float | None = None,
):
    output_filename = "ground_truth_results.jsonl"

    if run:
        forecaster_config = get_forecaster_config(config_path, forecaster_options)

        forecaster = make_forecaster(
            forecaster_class=forecaster_class,
            custom_path=custom_path,
            forecaster_config=forecaster_config,
        )

        # Print arguments
        print("Arguments:")
        print(f"  forecaster_class: {forecaster_class}")
        print(f"  custom_path: {custom_path}")
        print(f"  forecaster_config: {forecaster_config}")
        print(f"  input_file: {input_file}")
        print(f"  num_lines: {num_lines}")
        print(f"  run: {run}")
        print(f"  load_dir: {load_dir}")
        print(f"  is_async: {is_async}")
        print(f"  output_dir: {output_dir}")

        output_directory, most_recent_directory = create_output_directory(
            forecaster, BASE_FORECASTS_OUTPUT_PATH, output_dir
        )
        dirs_to_write = [output_directory, most_recent_directory]
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]
        forecasting_questions = [
            ForecastingQuestion.model_validate(line) for line in data
        ]
        print(f"Loaded {len(forecasting_questions)} forecasting questions")

        data, forecasting_questions = zip(
            *[
                (line, fq)
                for line, fq in zip(data, forecasting_questions)
                if fq.resolution is not None
            ]
        )

        print(
            f"Filtered to {len(forecasting_questions)}/{len(data)} questions with resolutions"
        )

        assert len(data) == len(
            forecasting_questions
        ), "Data and forecasting questions have different lengths"
        num_lines = min(num_lines, len(forecasting_questions))
        results = []
        print(f"Running on {num_lines} questions")
        assert load_dir is None, "load_dir must be None when run is True"

        forecasts = []
        results = []

        batch_size = 20
        for start in tqdm(range(0, num_lines, batch_size)):
            end = min(start + batch_size, num_lines)
            batch_tuples = forecasting_questions[start:end]
            if is_async:
                reset_global_semaphore()
                call_func = functools.partial(forecaster.call_async_full)
                forecasts_batch = asyncio.run(
                    parallelized_call(call_func, batch_tuples)
                )
            else:
                forecasts_batch = []
                for fq in batch_tuples:
                    forecast = forecaster.call_full(fq)
                    forecasts_batch.append(forecast)

            results_batch = []
            for line, fq, forecast in zip(
                data[start:end], batch_tuples, forecasts_batch
            ):
                results_batch.append(make_result_dict(line, fq, forecast))

            forecasts.extend(forecasts_batch)
            results.extend(results_batch)
            write_to_dirs(results, output_filename, dirs_to_write, overwrite=True)
    else:
        if (
            load_dir is None
            or not (
                (load_dir := Path(load_dir)) / "ground_truth_results.jsonl"
            ).exists()
        ):
            raise ValueError(
                "if --run argument is not set, load_dir must be provided and must contain ground_truth_results.jsonl"
            )
        output_directory = Path(load_dir)
        dirs_to_write = [output_directory]
        input_file = load_dir / "ground_truth_results.jsonl"
        with open(input_file, "r") as f:
            lines = [json.loads(line) for line in f]
        data = [line["question"] for line in lines]
        forecasting_questions = [
            ForecastingQuestion.model_validate(line) for line in data
        ]
        forecast_data = [line["forecast"] for line in lines]
        forecasts = [Forecast.model_validate(forecast) for forecast in forecast_data]
        print(
            f"Loaded {len(forecasting_questions)} forecasting questions and forecasts"
        )

        data, forecasting_questions, forecasts = zip(
            *[
                (line, fq, forecast)
                for line, fq, forecast in zip(data, forecasting_questions, forecasts)
                if fq.resolution is not None
            ]
        )
        print(
            f"Filtered to {len(forecasting_questions)}/{len(data)} questions with resolutions"
        )
        num_lines = min(num_lines, len(forecasting_questions))
        results = []

        with open(input_file, "r", encoding="utf-8") as f:
            results_loaded = [json.loads(line) for line in f]
            forecasts = [
                Forecast.model_validate(results_loaded[i]["forecast"])
                for i in range(num_lines)
            ]
            for i in range(num_lines):
                print(forecasting_questions[i].title)
                assert (
                    forecasting_questions[i].title
                    == results_loaded[i]["question"]["title"]
                ), "Questions do not match"

        for line, fq, forecast in zip(
            data[:num_lines], forecasting_questions[:num_lines], forecasts
        ):
            results.append(make_result_dict(line, fq, forecast))

    outcomes = [result["resolution"] for result in results]
    probs = [result["prob"] for result in results]
    if platt_scaling_factor is None:
        if all(o is None for o in outcomes):
            print("No outcomes to calibrate to. Skipping platt scaling.")
            platt_scaling_factor = 1
        else:
            platt_scaling_factor = platt_scaling(
                probs=probs,
                outcomes=outcomes,
            ).platt_scaling_a

    print(f"Platt scaling factor: {platt_scaling_factor}")
    calibrated_probs = platt_scaling(
        probs, outcomes, a=platt_scaling_factor
    ).calibrated_probs
    print(f"Calibrated probs: {calibrated_probs}")
    for result, calibrated_prob in zip(results, calibrated_probs):
        result["platt"] = {}
        result["platt"]["factor"] = round_floats(platt_scaling_factor, precision=4)
        result["platt"]["calibrated_prob"] = round_floats(calibrated_prob, precision=4)
        result["platt"]["brier_score"] = round_floats(
            proper_score(
                probs=[calibrated_prob],
                outcomes=[result["resolution"]],
                scoring_function=scoring_functions["brier_score"],
            ),
            precision=4,
        )
        result["platt"]["brier_score_scaled"] = round_floats(
            scale_brier_score(result["platt"]["brier_score"]), precision=1
        )
        result["platt"]["log_score"] = round_floats(
            proper_score(
                probs=[calibrated_prob],
                outcomes=[result["resolution"]],
                scoring_function=scoring_functions["log_score"],
            ),
            precision=4,
        )

    write_to_dirs(
        results=results,
        filename=output_filename,
        dirs_to_write=dirs_to_write,
        overwrite=True,
    )
    print(f"Results written to {output_filename}")

    # Calculate and print summary statistics
    avg_log_score = sum(result["log_score"] for result in results) / len(results)
    avg_brier_score = sum(result["brier_score"] for result in results) / len(results)
    avg_brier_score_scaled = sum(
        result["brier_score_scaled"] for result in results
    ) / len(results)
    avg_platt_brier_score = sum(
        result["platt"]["brier_score"] for result in results
    ) / len(results)
    avg_platt_brier_score_scaled = sum(
        result["platt"]["brier_score_scaled"] for result in results
    ) / len(results)

    calibration_error_data: dict = calculate_calibration(
        [fq.resolution for fq in forecasting_questions[:num_lines]],
        [result["prob"] for result in results],
    )
    calibration_error = calibration_error_data["calibration_error"]

    brier_score_decomposition = decompose_brier_score(
        [result["prob"] for result in results],
        [fq.resolution for fq in forecasting_questions[:num_lines]],
    )

    resolutions = [fq.resolution for fq in forecasting_questions[:num_lines]]
    avg_resolution = sum(resolutions) / len(resolutions)
    tuned_brier_baseline = sum(
        (resolution - avg_resolution) ** 2 for resolution in resolutions
    ) / len(resolutions)
    tuned_brier_baseline_scaled = scale_brier_score(tuned_brier_baseline)

    summary = {
        "total_questions": len(results),
        "avg_brier_score": round_floats(avg_brier_score, precision=4),
        "avg_platt_brier_score": round_floats(avg_platt_brier_score, precision=4),
        "tuned_brier_baseline": round_floats(tuned_brier_baseline, precision=4),
        "avg_brier_score_scaled": round_floats(avg_brier_score_scaled, precision=1),
        "avg_platt_brier_score_scaled": round_floats(
            avg_platt_brier_score_scaled, precision=1
        ),
        "tuned_brier_baseline_scaled": round_floats(
            tuned_brier_baseline_scaled, precision=1
        ),
        "avg_log_score": avg_log_score,
        "platt_scaling_factor": platt_scaling_factor,
        "brier_score_decomposition": brier_score_decomposition,
        "calibration_error": calibration_error,
        "calibration_error_data": calibration_error_data,
    }

    if run:
        summary["forecaster"] = forecaster.__class__.__name__
        summary["full_forecaster_config"] = forecaster.dump_config()
    else:
        # load ground_truth_summary.json
        summary_path = Path(load_dir) / "ground_truth_summary.json"
        if not summary_path.exists():
            print(f"ground_truth_summary.json not found in {load_dir}")
            summary["forecaster"] = None
            summary["full_forecaster_config"] = None
        else:
            summary_data = json.load(open(summary_path, "r"))
            summary["forecaster"] = summary_data["forecaster"]
            summary["full_forecaster_config"] = summary_data["full_forecaster_config"]
        summary["loaded_forecasts"] = f"{load_dir}/ground_truth_results.jsonl"

    print("\nGround Truth Summary:")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Average Brier Score:       {summary['avg_brier_score']:.4f}")
    print(f"Average Platt Brier Score: {summary['avg_platt_brier_score']:.4f}")
    print(f"Tuned Brier Baseline:      {summary['tuned_brier_baseline']:.4f}")
    print()

    print(f"Average Brier Score Scaled:       {summary['avg_brier_score_scaled']:.1f}")
    print(
        f"Average Platt Brier Score Scaled: {summary['avg_platt_brier_score_scaled']:.1f}"
    )
    print(
        f"Tuned Brier Baseline Scaled:      {summary['tuned_brier_baseline_scaled']:.1f}"
    )
    print()

    print(f"Average Log Score: {summary['avg_log_score']:.4f}")
    print(f"Platt Scaling Factor: {summary['platt_scaling_factor']:.2f}")
    print(f"Forecaster: {summary['forecaster']}")
    print(f"Forecaster Config: {summary['full_forecaster_config']}")

    # Write summary to file
    summary_filename = "ground_truth_summary.json"
    for dir in dirs_to_write:
        json.dump(round_floats(summary), open(dir / summary_filename, "w"), indent=4)
        print(f"\nSummary written to {dir / summary_filename}")

    # Plot calibration error
    probs = [result["prob"] for result in results]
    outcomes = [result["resolution"] for result in results]
    to_plot = True
    if to_plot:
        for spacing in ["logit", "linear"]:
            plot = plot_calibration(probs, outcomes, spacing=spacing)
            plot_filename = f"calibration_plot_{spacing}.png"
            for dir in dirs_to_write:
                plot.savefig(dir / plot_filename)
            print(f"Calibration plot written to {plot_filename}")


if __name__ == "__main__":
    main()

    # Example run command
print("Example run command:")
print(
    "python ground_truth_run.py --forecaster_class BasicForecaster --forecaster_options model=gpt-4o-mini --input_file path/to/input.jsonl --output_dir path/to/output"
)
# python src/ground_truth_run.py --load_dir="src/data/forecasts/recalc_test/groundtruth/"
