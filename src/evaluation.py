import sys
import io
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
import click
import yaml
import logging
import functools
import concurrent.futures

from forecasters import (
    Forecaster,
    AdvancedForecaster,
    BasicForecaster,
    COT_Forecaster,
)
from forecasters.consistent_forecaster import ConsistentForecaster
from static_checks.Checker import (
    Checker,
    NegChecker,
    choose_checkers,
)
from common.path_utils import get_data_path, get_src_path
import common.llm_utils  # noqa
from common.llm_utils import reset_global_semaphore

BASE_TUPLES_PATH: Path = get_data_path() / "tuples/"
BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"
CONFIGS_DIR: Path = get_src_path() / "forecasters/forecaster_configs"

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

metrics = ["default", "frequentist"]


def get_stats(results: dict, label: str = "") -> dict:
    ret = {}
    for metric in metrics:
        print(f"{metric}")

        # Extract the violation and check results from the test
        violations = [result[metric]["violation"] for result in results]
        checks = [result[metric]["check"] for result in results]

        # Calculate the number of violations
        print(f"checks: {checks}")
        num_failed = sum(1 for c in checks if not c)

        # Calculate the average violation
        avg_violation = sum(v for v in violations) / len(violations)

        # Calculate the median violation
        sorted_violations = sorted(violations, key=abs)
        n = len(sorted_violations)
        median_violation = (sorted_violations[n // 2] + sorted_violations[~n // 2]) / 2

        outlier_tail: int = 1
        avg_violation_no_outliers = sum(
            sorted_violations[outlier_tail:-outlier_tail]
        ) / (len(sorted_violations) - 2 * outlier_tail)

        print(f"Number of violations: {num_failed}/{len(checks)}")
        print(f"Average violation: {avg_violation:.3f}")
        print(f"Average violation without outliers: {avg_violation_no_outliers:.3f}")
        print(f"Median violation: {median_violation:.3f}")

        ret[metric] = {
            "label": label,
            "num_samples": len(violations),
            "num_violations": num_failed,
            "avg_violation": round(avg_violation, 6),
            "avg_violation_no_outliers": round(avg_violation_no_outliers, 6),
            "median_violation": round(median_violation, 6),
        }

    return ret


def make_folder_name(forecaster: Forecaster, model: str, timestamp: datetime) -> str:
    # dirty hack: we explicitly don't round the seconds because we want the neighboring runs to write to the same dir
    return f"{forecaster.__class__.__name__}_{timestamp.strftime('%m-%d-%H-%M')}"


def validate_result(result: dict, keys: list[str]) -> None:
    assert "line" in result, "results must contain a 'line' key"
    for key in keys:
        assert key in result["line"], f"line must contain a '{key}' key"
        assert (
            "elicited_prob" in result["line"][key]
        ), f"line[{key}] must contain an 'elicited_prob' key"
    return True


def write_to_dirs(
    results: list[dict],
    filename: str,
    dirs_to_write: list[Path],
    overwrite: bool = False,
):
    for dir in dirs_to_write:
        if overwrite:
            with open(dir / filename, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
        else:
            with open(dir / filename, "a", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")


def process_check(
    check_name: str,
    checkers: dict[str, Checker],
    forecaster: Forecaster,
    model: str,
    num_lines: int,
    is_async: bool,
    output_directory: Path,
    most_recent_directory: Path,
    load_dir: Path,
    run: bool,
    forecaster_class: str,
) -> dict:
    print("Checker: ", check_name)
    with open(checkers[check_name].path, "r", encoding="utf-8") as f:
        print(f"Path: {checkers[check_name].path}")
        checker_tuples = [json.loads(line) for line in f.readlines()[:num_lines]]

    keys = [key for key in checker_tuples[0].keys() if key not in ["metadata"]]
    print(f"keys: {keys}")

    dirs_to_write = [output_directory, most_recent_directory]

    if run:
        # clear the file
        print(f"Clearing {check_name}.jsonl")
        for dir in dirs_to_write:
            if Path(dir / f"{check_name}.jsonl").exists():
                os.remove(dir / f"{check_name}.jsonl")

        if is_async:
            batch_of_qs_size = 5
            batches = [
                (i, min(i + batch_of_qs_size, num_lines))
                for i in range(0, num_lines, batch_of_qs_size)
            ]
        else:
            batches = [(i, i + 1) for i in range(0, num_lines)]

        results = []
        for batch_idx, batch in enumerate(batches):
            match forecaster_class:
                case (
                    "BasicForecaster"
                    | "CoTForecaster"
                    | "ConsistentForecaster"
                    | "RecursiveConsistentForecaster"
                ):
                    if is_async:
                        reset_global_semaphore()
                        results_batch = asyncio.run(
                            checkers[check_name].test(
                                forecaster,
                                do_check=False,
                                line_begin=batch[0],
                                line_end=batch[1],
                                model=model,
                            )
                        )
                    else:
                        results_batch = checkers[check_name].test_sync(
                            forecaster,
                            do_check=False,
                            line_begin=batch[0],
                            line_end=batch[1],
                            model=model,
                        )

                case "AdvancedForecaster":
                    # we don't pass model to the test function, it's specified in the config
                    if is_async:
                        reset_global_semaphore()
                        results_batch = asyncio.run(
                            checkers[check_name].test(
                                forecaster,
                                do_check=False,
                                line_begin=batch[0],
                                line_end=batch[1],
                            )
                        )
                    else:
                        results_batch = checkers[check_name].test_sync(
                            forecaster,
                            do_check=False,
                            line_begin=batch[0],
                            line_end=batch[1],
                        )

            print(f"results_batch: {results_batch}")
            print(f"len(results_batch): {len(results_batch)}")
            print(f"len(batch): {batch[1] - batch[0]}")
            assert (
                len(results_batch) == batch[1] - batch[0]
            ), "results must be of the same length as the batch"
            assert all(validate_result(result, keys) for result in results_batch)

            write_to_dirs(results_batch, f"{check_name}.jsonl", dirs_to_write)
            results.extend(results_batch)

    else:
        with open(load_dir / f"{check_name}.jsonl", "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f.readlines()]

        assert (
            len(results) >= num_lines
        ), "results must contain at least num_lines elements"
        results = results[:num_lines]
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
        violation_data = {}
        for metric in metrics:
            violation_data[metric] = checkers[check_name].check_from_elicited_probs(
                answers, metric
            )
        print(f"violation_data: {violation_data}")
        result.update(violation_data)

    stats = get_stats(results, label=check_name)
    return stats


@click.command()
@click.option(
    "-f",
    "--forecaster_class",
    default="AdvancedForecaster",
    help="Forecaster to use. Can be BasicForecaster, COT_Forecaster, AdvancedForecaster, ConsistentForecaster, RecursiveConsistentForecaster.",
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
    default="gpt-4o-mini",  # -2024-07-18
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
@click.option(
    "--threads",
    "use_threads",
    is_flag=True,
    default=False,
    help="Use threads to run the forecaster on different checks",
)
@click.option(
    "-p",
    "--tuple_dir",
    type=click.Path(),
    required=False,
    help="Path to the tuple file",
)
def main(
    forecaster_class: str,
    config_path: str,
    model: str,
    run: bool,
    load_dir: str,
    num_lines: int,
    relevant_checks: list[str],
    is_async: bool,
    use_threads: bool,
    tuple_dir: str | None = None,
):
    if tuple_dir is None:
        tuple_dir = BASE_TUPLES_PATH
    tuple_dir = Path(tuple_dir)

    checkers: dict[str, Checker] = choose_checkers(relevant_checks, tuple_dir)

    match forecaster_class:
        case "BasicForecaster":
            forecaster = BasicForecaster()
        case "COT_Forecaster":
            forecaster = COT_Forecaster()
        case "ConsistentForecaster":
            forecaster = ConsistentForecaster(
                hypocrite=BasicForecaster(),
                instantiation_kwargs={"model": model},
                bq_func_kwargs={"model": model},
            )
        case "RecursiveConsistentForecaster":
            forecaster = ConsistentForecaster.recursive(
                depth=4,
                hypocrite=BasicForecaster(),
                checks=[
                    NegChecker()
                ],  # , ParaphraseChecker(), ButChecker(), CondChecker()
                instantiation_kwargs={"model": model},
                bq_func_kwargs={"model": model},
            )
        case "AdvancedForecaster":
            with open(config_path, "r", encoding="utf-8") as f:
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
        pass
        # We do not actually care, for now
        # user_input = input(
        #    f"Directory '{output_directory}' already exists. Do you want to continue? (y/N): "
        # )
        # if user_input.lower() != "y":
        #    print("Operation aborted by the user.")
        #    exit(1)

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

    print(f"Relevant checks: {checkers.keys()}")

    logged_config = {
        "forecaster_class": forecaster.__class__.__name__,
        "forecaster": forecaster.dump_config(),
        "checkers": [checker.dump_config() for name, checker in checkers.items()],
        "model": model,
        "is_async": is_async,
        "use_threads": use_threads,
        "run": run,
        "load_dir": str(load_dir),
        "relevant_checks": list(checkers.keys()),
        "tuple_dir": str(tuple_dir),
        "num_lines": num_lines,
    }

    write_to_dirs(
        [logged_config],
        "config.jsonl",
        [output_directory, most_recent_directory],
        overwrite=True,
    )

    all_stats = {}

    if use_threads:
        print("Using threads")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(checkers.keys())
        ) as executor:
            process_check_func = functools.partial(
                process_check,
                checkers=checkers,
                forecaster=forecaster,
                model=model,
                num_lines=num_lines,
                is_async=is_async,
                output_directory=output_directory,
                most_recent_directory=most_recent_directory,
                load_dir=load_dir,
                run=run,
                forecaster_class=forecaster_class,
            )
            all_stats = {
                check_name: stats
                for check_name, stats in zip(
                    checkers.keys(), executor.map(process_check_func, checkers.keys())
                )
            }
    else:
        for check_name in checkers.keys():
            stats = process_check(
                check_name=check_name,
                checkers=checkers,
                forecaster=forecaster,
                model=model,
                num_lines=num_lines,
                is_async=is_async,
                output_directory=output_directory,
                most_recent_directory=most_recent_directory,
                load_dir=load_dir,
                run=run,
                forecaster_class=forecaster_class,
            )
            all_stats[check_name] = stats

    # TODO figure out how to write to the load_dir

    with open(output_directory / "stats_summary.json", "a", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=4)
    with open(
        most_recent_directory / "stats_summary.json", "a", encoding="utf-8"
    ) as f2:
        # TODO: this one append on old data if it exists in the dir
        json.dump(all_stats, f2, indent=4)

    for check_name, stats in all_stats.items():
        with (
            open(
                output_directory / f"stats_{check_name}.json", "w", encoding="utf-8"
            ) as f,
            open(
                most_recent_directory / f"stats_{check_name}.json",
                "w",
                encoding="utf-8",
            ) as f2,
        ):
            json.dump(stats, f, indent=4)
            json.dump(stats, f2, indent=4)

    for metric in metrics:
        print(f"{metric}")
        for check_name, stats in all_stats.items():
            print(
                f"{stats[metric]['label']}: {stats[metric]['num_violations']}/{stats[metric]['num_samples']}"
            )

        print("\n\n")
        for check_name, stats in all_stats.items():
            print(
                f"{check_name} | avg: {stats[metric]['avg_violation']:.3f}, avg_no_outliers: {stats[metric]['avg_violation_no_outliers']:.3f}, median: {stats[metric]['median_violation']:.3f}"
            )


if __name__ == "__main__":
    main()

# run the script with the following command:
# python evaluation.py -f AdvancedForecaster -c forecasters/forecaster_configs/cheap_haiku.yaml --run -n 3 --relevant_checks all | tee see_eval.txt
# python evaluation.py -f ConsistentForecaster -m gpt-4o-mini --run -n 50 --relevant_checks all | tee see_eval.txt
# python evaluation.py -f BasicForecaster -m gpt-4o-mini --run -n 50 -k ParaphraseChecker -k CondCondChecker | tee see_eval.txt
# python evaluation.py -f ConsistentForecaster -m gpt-4o-mini --run -n 25 -k CondCondChecker --async | tee see_eval.txt
# python evaluation.py -f ConsistentForecaster -m gpt-4o-mini-2024-07-18 --run -n 3 -k CondChecker -k ConsequenceChecker -k ParaphraseChecker -k CondCondChecker --async | tee see_eval.txt
# python evaluation.py -f RecursiveConsistentForecaster -m gpt-4o-mini --run -n 3 --relevant_checks all | tee see_eval.txt
# python evaluation.py -f ConsistentForecaster -m gpt-4o-mini --run -n 3 --relevant_checks all | tee see_eval.txt
# python evaluation.py -f RecursiveConsistentForecaster -m gpt-4o-mini -k NegChecker --run -n 20 --async
