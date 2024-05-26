import sys
import io
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
import click
import yaml

from forecasters import Forecaster, AdvancedForecaster, BasicForecaster, COT_Forecaster
from static_checks.Checker import (
    Checker,
    NegChecker,
    AndChecker,
    OrChecker,
    AndOrChecker,
    ButChecker,
    CondChecker,
    ConsequenceChecker,
    ParaphraseChecker,
    SymmetryAndChecker,
    SymmetryOrChecker,
    CondCondChecker,
)
from common.path_utils import get_data_path, get_src_path

BASE_DATA_PATH: Path = get_data_path() / "tuples/"
BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"
CONFIGS_DIR: Path = get_src_path() / "forecasters/forecaster_configs"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

checkers: dict[str, Checker] = {
    "NegChecker": NegChecker(path=BASE_DATA_PATH / "NegChecker.jsonl"),
    "AndChecker": AndChecker(path=BASE_DATA_PATH / "AndChecker.jsonl"),
    "OrChecker": OrChecker(path=BASE_DATA_PATH / "OrChecker.jsonl"),
    "AndOrChecker": AndOrChecker(path=BASE_DATA_PATH / "AndOrChecker.jsonl"),
    "ButChecker": ButChecker(path=BASE_DATA_PATH / "ButChecker.jsonl"),
    "CondChecker": CondChecker(path=BASE_DATA_PATH / "CondChecker.jsonl"),
    "ConsequenceChecker": ConsequenceChecker(
        path=BASE_DATA_PATH / "ConsequenceChecker.jsonl"
    ),
    "ParaphraseChecker": ParaphraseChecker(
        path=BASE_DATA_PATH / "ParaphraseChecker.jsonl"
    ),
    "SymmetryAndChecker": SymmetryAndChecker(
        path=BASE_DATA_PATH / "SymmetryAndChecker.jsonl"
    ),
    "SymmetryOrChecker": SymmetryOrChecker(
        path=BASE_DATA_PATH / "SymmetryOrChecker.jsonl"
    ),
    "CondCondChecker": CondCondChecker(path=BASE_DATA_PATH / "CondCondChecker.jsonl"),
}


def get_stats(results: dict, label: str = "") -> dict:
    # Extract the violation and check results from the test
    violations = [result["violation"] for result in results]
    checks = [result["check"] for result in results]

    # Calculate the number of violations
    print(f"checks: {checks}")
    num_failed = sum(1 for c in checks if not c)

    # Calculate the average violation
    avg_violation = sum(v for v in violations) / len(violations)

    # Calculate the median violation
    sorted_violations = sorted(violations)
    n = len(sorted_violations)
    median_violation = (sorted_violations[n // 2] + sorted_violations[~n // 2]) / 2

    print(f"Number of violations: {num_failed}/{len(checks)}")
    print(f"Average violation: {avg_violation:.3f}")
    print(f"Median violation: {median_violation:.3f}")

    return {
        "label": label,
        "num_samples": len(violations),
        "num_violations": num_failed,
        "avg_violation": avg_violation,
        "median_violation": median_violation,
    }


def make_folder_name(forecaster: Forecaster, model: str, timestamp: datetime) -> str:
    return f"{forecaster.__class__.__name__}_{timestamp.strftime('%m-%d-%H-%M-%S')}"


def validate_result(result: dict, keys: list[str]) -> None:
    assert "line" in result, "results must contain a 'line' key"
    for key in keys:
        assert key in result["line"], f"line must contain a '{key}' key"
        assert (
            "elicited_prob" in result["line"][key]
        ), f"line[{key}] must contain an 'elicited_prob' key"
    return True


@click.command()
@click.option(
    "-f",
    "--forecaster_class",
    default="AdvancedForecaster",
    help="Forecaster to use. Can be BasicForecaster, COT_Forecaster, or AdvancedForecaster.",
)
@click.option(
    "-c",
    "--config_path",
    type=click.Path(),
    default=CONFIGS_DIR / "cheap_haiku.yaml",
    help="Path to the configuration file",
)
@click.option(
    "-m",
    "--model",
    default="gpt-4o-2024-05-13",
    help="Model to use for BasicForecaster and CoT_Forecaster. Is overridden by the config file in case of AdvancedForecaster.",
)
@click.option("-r", "--run", is_flag=True, help="Run the forecaster")
@click.option(
    "-l",
    "--load_dir",
    required=False,
    type=click.Path(),
    help="Directory to load results from in case run is False. Defaults to most_recent_directory",
)
@click.option(
    "-n",
    "--num_lines",
    default=3,
    help="Number of lines to process in each of the files",
)
@click.option(
    "-k",
    "--relevant_checks",
    multiple=True,
    default=["CondChecker"],
    help='Relevant checks to perform. In case of "all", all checkers are used.',
)
@click.option(
    "--async",
    "is_async",
    is_flag=True,
    default=False,
    help="Await gather the forecaster over all lines in a check",
)
def main(
    forecaster_class,
    config_path,
    model,
    run,
    load_dir,
    num_lines,
    relevant_checks,
    is_async,
):
    match forecaster_class:
        case "BasicForecaster":
            forecaster = BasicForecaster()
        case "COT_Forecaster":
            forecaster = COT_Forecaster()
        case "AdvancedForecaster":
            with open(config_path, "r") as f:
                config: dict[str, Any] = yaml.safe_load(f)
            forecaster = AdvancedForecaster(**config)
        case _:
            raise ValueError(f"Invalid forecaster class: {forecaster_class}")

    print(forecaster.dump_config())

    most_recent_directory = (
        BASE_FORECASTS_OUTPUT_PATH / f"A_{forecaster.__class__.__name__}_most_recent"
    )
    if not most_recent_directory.exists():
        most_recent_directory.mkdir(parents=True, exist_ok=True)

    timestamp_start_run = datetime.now()
    output_directory = BASE_FORECASTS_OUTPUT_PATH / make_folder_name(
        forecaster, model, timestamp_start_run
    )
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{output_directory}' created.")
    else:
        user_input = input(
            f"Directory '{output_directory}' already exists. Do you want to continue? (y/N): "
        )
        if user_input.lower() != "y":
            print("Operation aborted by the user.")
            exit(1)

    logged_config = {
        "forecaster_class": forecaster.__class__.__name__,
        "forecaster": forecaster.dump_config(),
        "model": model,
        "num_lines": num_lines,
    }

    with open(output_directory / "config.jsonl", "w") as f, open(
        most_recent_directory / "config.jsonl", "w"
    ) as f2:
        f.write(json.dumps(logged_config) + "\n")
        f2.write(json.dumps(logged_config) + "\n")

    if run:
        assert load_dir is None, "LOAD_DIR must be None if RUN is True"
    else:
        if load_dir is None:
            print(f"LOAD_DIR is None, using {most_recent_directory}")
            load_dir = most_recent_directory
        load_dir = Path(load_dir)
        assert (
            load_dir.exists() and load_dir.is_dir()
        ), "LOAD_DIR must be a valid directory"

    all_stats = {}
    if relevant_checks[0] == "all":
        relevant_checks = list(checkers.keys())

    for check_name in relevant_checks:
        print("Checker: ", check_name)
        with open(checkers[check_name].path, "r") as f:
            print(f"Path: {checkers[check_name].path}")
            checker_tuples = [json.loads(line) for line in f.readlines()[:num_lines]]

        keys = [key for key in checker_tuples[0].keys() if key not in ["metadata"]]
        print(f"keys: {keys}")

        if run:
            match forecaster_class:
                case "BasicForecaster" | "CoTForecaster":
                    if is_async:
                        results = asyncio.run(
                            checkers[check_name].test(
                                forecaster,
                                do_check=False,
                                num_lines=num_lines,
                                model=model,
                            )
                        )
                    else:
                        results = checkers[check_name].test_sync(
                            forecaster, do_check=False, num_lines=num_lines, model=model
                        )

                case "AdvancedForecaster":
                    if is_async:
                        results = asyncio.run(
                            checkers[check_name].test(
                                forecaster, do_check=False, num_lines=num_lines
                            )
                        )
                    else:
                        results = checkers[check_name].test_sync(
                            forecaster, do_check=False, num_lines=num_lines
                        )

            assert len(results) == num_lines, "results must be of length num_lines"
            assert all(validate_result(result, keys) for result in results)

            with open(output_directory / f"{check_name}.jsonl", "w") as f, open(
                most_recent_directory / f"{check_name}.jsonl", "w"
            ) as f2:
                for result in results:
                    f.write(json.dumps(result) + "\n")
                    f2.write(json.dumps(result) + "\n")
        else:
            with open(load_dir / f"{check_name}.jsonl", "r") as f:
                results = [json.loads(line) for line in f.readlines()]

            assert (
                len(results) >= num_lines
            ), "results must contain at least num_lines elements"
            assert all(validate_result(result, keys) for result in results)

        for result, checker_tuple in zip(results, checker_tuples):
            for key in keys:
                assert (
                    result["line"][key]["id"] == checker_tuple[key]["id"]
                ), f"result['line'][{key}]['id'] must match checker_tuple[{key}]['id']"
                assert (
                    result["line"][key]["title"] == checker_tuple[key]["title"]
                ), f"result['line'][{key}]['title'] must match checker_tuple[{key}]['title']"

        data = [result["line"] for result in results]
        all_answers = [
            {key: result["line"][key]["elicited_prob"] for key in keys}
            for result in results
        ]
        for line, answers, result in zip(data, all_answers, results):
            print(f"answers: {answers}")
            violation_data = checkers[check_name].check_from_elicited_probs(answers)
            print(f"violation_data: {violation_data}")
            result.update(violation_data)

        stats = get_stats(results, label=check_name)
        all_stats[check_name] = stats

    with open(output_directory / "stats_summary.json", "w") as f, open(
        most_recent_directory / "stats_summary.json", "w"
    ) as f2:
        json.dump(all_stats, f, indent=4)
        json.dump(all_stats, f2, indent=4)

    for check_name, stats in all_stats.items():
        print(f"{stats['label']}: {stats['num_violations']}/{stats['num_samples']}")

    print("\n\n")
    for check_name, stats in all_stats.items():
        print(
            f"{check_name} | avg: {stats['avg_violation']:.3f}, median: {stats['median_violation']:.3f}"
        )


if __name__ == "__main__":
    main()

# run the script with the following command:
# python evaluation.py -f AdvancedForecaster -c forecasters/forecaster_configs/cheap_haiku.yaml --run -n 3 --relevant_checks all | tee see_eval.txt
