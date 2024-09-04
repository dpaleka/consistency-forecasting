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
from forecasters.PromptedToCons_Forecaster import PromptedToCons_Forecaster  # Adam
from static_checks.Checker import (
    Checker,
    ExpectedEvidenceChecker,  # noqa
    NegChecker,  # noqa
    ParaphraseChecker,  # noqa
    choose_checkers,
)
from common.path_utils import get_data_path, get_src_path
import common.llm_utils  # noqa
from common.llm_utils import reset_global_semaphore

BASE_TUPLES_PATH: Path = get_data_path() / "tuples/"

# BASE_TUPLES_PATH: Path = get_data_path() / "tuples_source/"
BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"
CONFIGS_DIR: Path = get_src_path() / "forecasters/forecaster_configs"

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

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
        if violations:
            avg_violation = sum(violations) / len(violations)
        else:
            avg_violation = 0  # resolve division by zero error

        # Calculate the median violation
        if violations:
            sorted_violations = sorted(violations, key=abs)
            n = len(sorted_violations)
            median_violation = (
                sorted_violations[n // 2] + sorted_violations[~n // 2]
            ) / 2
        else:
            median_violation = 0

        outlier_tail: int = 1
        if len(violations) > 2 * outlier_tail:
            avg_violation_no_outliers = sum(
                sorted(violations)[outlier_tail:-outlier_tail]
            ) / (len(violations) - 2 * outlier_tail)
        else:
            avg_violation_no_outliers = avg_violation

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


def aggregate_stats_by_source(all_stats: dict, output_directory: Path):
    source_aggregated_stats = {}

    for checker_name, checker_stats in all_stats.items():
        if "by_source" in checker_stats:
            for source, source_stats in checker_stats["by_source"].items():
                source_id = source_stats.get("source_id")
                if source not in source_aggregated_stats:
                    source_aggregated_stats[source] = {
                        "source_id": source_id,
                        "overall": {},
                        "by_checker": {},
                    }

                # Store individual checker stats
                source_aggregated_stats[source]["by_checker"][checker_name] = {
                    k: v for k, v in source_stats.items() if k != "source_id"
                }

    # Calculate overall stats
    for source, stats in source_aggregated_stats.items():
        overall = {"default": {}, "frequentist": {}}
        checker_count = len(stats["by_checker"])

        for metric in ["default", "frequentist"]:
            overall[metric] = {
                "num_samples": 0,
                "num_violations": 0,
                "avg_violation": 0,
                "avg_violation_no_outliers": 0,
                "median_violation": 0,
            }

            for checker_stats in stats["by_checker"].values():
                checker_metric = checker_stats[metric]
                overall[metric]["num_samples"] += checker_metric["num_samples"]
                overall[metric]["num_violations"] += checker_metric["num_violations"]
                overall[metric]["avg_violation"] += checker_metric["avg_violation"]
                overall[metric]["avg_violation_no_outliers"] += checker_metric[
                    "avg_violation_no_outliers"
                ]
                overall[metric]["median_violation"] += checker_metric[
                    "median_violation"
                ]

            # Calculate averages
            for key in [
                "avg_violation",
                "avg_violation_no_outliers",
                "median_violation",
            ]:
                overall[metric][key] /= checker_count

            overall[metric]["label"] = f"Overall_{source}"

        stats["overall"] = overall

    # Write the aggregated stats to a file
    output_file = output_directory / "stats_by_source_question.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(source_aggregated_stats, f, indent=4)

    print(f"Aggregated stats by source question written to {output_file}")


def process_check(
    check_name: str,
    checkers: dict[str, Checker],
    forecaster: Forecaster,
    model: str,
    num_lines: int,
    tuples_per_source: int,
    is_async: bool,
    output_directory: Path,
    most_recent_directory: Path,
    load_dir: Path,
    run: bool,
    forecaster_class: str,
    eval_by_source: bool,
) -> dict:
    print(f"Debug: Starting process_check for {check_name}")
    try:
        print("Checker: ", check_name)
        with open(checkers[check_name].path, "r", encoding="utf-8") as f:
            print(f"Path: {checkers[check_name].path}")
            all_tuples = [json.loads(line) for line in f]

        if eval_by_source:
            source_questions = {}
            for tuple_data in all_tuples:
                source_question = (
                    tuple_data["P"]["metadata"].get("source_question")
                    or tuple_data["P"]["title"]
                )
                source_id = tuple_data["P"]["metadata"].get("source_id")
                if source_question not in source_questions:
                    source_questions[source_question] = {
                        "tuples": [],
                        "source_id": source_id,
                    }
                if len(source_questions[source_question]["tuples"]) < tuples_per_source:
                    source_questions[source_question]["tuples"].append(tuple_data)

            checker_tuples = [
                tuple
                for source_data in source_questions.values()
                for tuple in source_data["tuples"]
            ]
        else:
            checker_tuples = all_tuples[:num_lines]

        keys = [key for key in checker_tuples[0].keys() if key not in ["metadata"]]
        print(f"Debug: keys: {keys}")
        print(f"Debug: Number of checker_tuples: {len(checker_tuples)}")

        dirs_to_write = [output_directory, most_recent_directory]

        if run:
            # clear the file
            print(f"Clearing {check_name}.jsonl")
            for dir in dirs_to_write:
                if Path(dir / f"{check_name}.jsonl").exists():
                    os.remove(dir / f"{check_name}.jsonl")

            results = []
            for start in range(0, len(checker_tuples), 5):
                end = min(start + 5, len(checker_tuples))
                batch_tuples = checker_tuples[start:end]

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
                                    tuples=batch_tuples,
                                    model=model,
                                )
                            )
                        else:
                            results_batch = checkers[check_name].test_sync(
                                forecaster,
                                do_check=False,
                                tuples=batch_tuples,
                                model=model,
                            )
                    case "AdvancedForecaster":
                        if is_async:
                            reset_global_semaphore()
                            results_batch = asyncio.run(
                                checkers[check_name].test(
                                    forecaster,
                                    do_check=False,
                                    tuples=batch_tuples,
                                )
                            )
                        else:
                            results_batch = checkers[check_name].test_sync(
                                forecaster,
                                do_check=False,
                                tuples=batch_tuples,
                            )
                    case "PromptedToCons_Forecaster":  # Adam
                        if is_async:
                            reset_global_semaphore()
                            results_batch = asyncio.run(
                                checkers[check_name].test(
                                    forecaster,
                                    do_check=False,
                                    tuples=batch_tuples,
                                    model=model,
                                )
                            )
                        else:
                            results_batch = checkers[check_name].test_sync(
                                forecaster,
                                do_check=False,
                                tuples=batch_tuples,
                                model=model,
                            )

                print(f"results_batch: {results_batch}")
                print(f"len(results_batch): {len(results_batch)}")
                print(f"len(batch_tuples): {len(batch_tuples)}")
                assert len(results_batch) == len(
                    batch_tuples
                ), "results must be of the same length as the batch"
                assert all(validate_result(result, keys) for result in results_batch)

                write_to_dirs(results_batch, f"{check_name}.jsonl", dirs_to_write)
                results.extend(results_batch)

            print(f"Debug: Number of results after run: {len(results)}")
        else:
            with open(load_dir / f"{check_name}.jsonl", "r", encoding="utf-8") as f:
                results = [json.loads(line) for line in f]

            results = results[: len(checker_tuples)]
            assert all(validate_result(result, keys) for result in results)
            print(f"Debug: Number of results loaded: {len(results)}")

        print(f"Debug: Number of results: {len(results)}")
        print(f"Debug: Number of checker_tuples: {len(checker_tuples)}")

        if len(results) != len(checker_tuples):
            print("Warning: Number of results does not match number of checker_tuples")
            print(f"Results: {len(results)}, Checker tuples: {len(checker_tuples)}")

        # Ensure results match checker_tuples
        for i, (result, checker_tuple) in enumerate(zip(results, checker_tuples)):
            for key in keys:
                try:
                    assert result["line"][key]["id"] == checker_tuple[key]["id"], (
                        f"ID mismatch for key {key} at index {i}: "
                        f"result ID {result['line'][key]['id']} != checker tuple ID {checker_tuple[key]['id']}"
                    )
                    assert (
                        result["line"][key]["title"] == checker_tuple[key]["title"]
                    ), (
                        f"Title mismatch for key {key} at index {i}: "
                        f"result title {result['line'][key]['title']} != checker tuple title {checker_tuple[key]['title']}"
                    )
                except AssertionError as e:
                    print(f"Assertion failed: {str(e)}")
                    print(f"Result: {result}")
                    print(f"Checker tuple: {checker_tuple}")
                    raise

        data = [result["line"] for result in results]
        all_answers = [
            {key: result["line"][key]["elicited_prob"] for key in keys}
            for result in results
        ]
        for line, answers, result in zip(data, all_answers, results):
            violation_data = {}
            for metric in metrics:
                violation_data[metric] = checkers[check_name].check_from_elicited_probs(
                    answers, metric
                )
            result.update(violation_data)

        print(f"Debug: Calculating stats for {check_name}")
        if eval_by_source:
            stats = {"overall": get_stats(results, label=check_name), "by_source": {}}
            for source, source_data in source_questions.items():
                source_results = [
                    r
                    for r in results
                    if r["line"]["P"]["metadata"].get("source_question") == source
                    or r["line"]["P"]["title"] == source
                ]
                stats["by_source"][source] = get_stats(
                    source_results, label=f"{check_name}_{source}"
                )
                stats["by_source"][source]["source_id"] = source_data["source_id"]
        else:
            stats = {"overall": get_stats(results, label=check_name)}

        print(f"Debug: Finished process_check for {check_name}")
        return stats
    except Exception as e:
        print(f"Debug: Error in process_check for {check_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


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
    default=CONFIGS_DIR / "cheap_gpt4o-mini.yaml",
    help="Path to the configuration file",
)
@click.option(
    "-m",
    "--model",
    default=None,
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
    "-t",
    "--tuples_per_source",
    default=5,
    help="Max number of tuples to use for each source question",
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
@click.option(
    "--output_dir",
    type=click.Path(),
    required=False,
    help=f"Path to the output directory. Will default to timestamped directory in {BASE_FORECASTS_OUTPUT_PATH} otherwise",
)
@click.option(
    "--eval_by_source",
    "-s",
    is_flag=True,
    default=False,
    help="Evaluate consistency scores per source question",
)
def main(
    forecaster_class: str,
    config_path: str,
    model: str | None,
    run: bool,
    load_dir: str,
    num_lines: int,
    tuples_per_source: int,
    relevant_checks: list[str],
    is_async: bool,
    use_threads: bool,
    tuple_dir: str | None = None,
    output_dir: str | None = None,
    eval_by_source: bool = False,
):
    if tuple_dir is None:
        tuple_dir = BASE_TUPLES_PATH
    tuple_dir = Path(tuple_dir)
    if not tuple_dir.exists():
        assert tuple_dir.exists(), f"Tuple directory {tuple_dir} does not exist"

    match forecaster_class:
        case "AdvancedForecaster":
            if model is not None:
                raise ValueError(
                    "The 'model' parameter should not be set when using AdvancedForecaster. Model configuration should be done through the config file and the 'config_path' parameter."
                )
            print(f"Using AdvancedForecaster config file: {config_path}")
        case _:
            assert model is not None, "Model must be specified for forecaster class"
            print(f"Using model: {model}")

    checkers: dict[str, Checker] = choose_checkers(relevant_checks, tuple_dir)

    match forecaster_class:
        case "BasicForecaster":
            forecaster = BasicForecaster()
        case "COT_Forecaster":
            forecaster = COT_Forecaster()
        case "ConsistentForecaster":
            forecaster = ConsistentForecaster(
                hypocrite=BasicForecaster(),
                checks=[
                    NegChecker(),
                ],
                instantiation_kwargs={"model": model},
                bq_func_kwargs={"model": model},
            )
        case "RecursiveConsistentForecaster":
            forecaster = ConsistentForecaster.recursive(
                depth=4,
                hypocrite=BasicForecaster(),
                checks=[
                    NegChecker(),
                    # ParaphraseChecker(),
                ],  # , ParaphraseChecker(), ButChecker(), CondChecker()
                instantiation_kwargs={"model": model},
                bq_func_kwargs={"model": model},
            )
        case "AdvancedForecaster":
            with open(config_path, "r", encoding="utf-8") as f:
                config: dict[str, Any] = yaml.safe_load(f)
            forecaster = AdvancedForecaster(**config)
        case "PromptedToCons_Forecaster":
            forecaster = PromptedToCons_Forecaster()
        case _:
            raise ValueError(f"Invalid forecaster class: {forecaster_class}")

    # print(forecaster.dump_config())

    most_recent_directory = (
        BASE_FORECASTS_OUTPUT_PATH / f"A_{forecaster.__class__.__name__}_most_recent"
    )
    if not most_recent_directory.exists():
        most_recent_directory.mkdir(parents=True, exist_ok=True)

    timestamp_start_run = datetime.now()
    if output_dir is None:
        print("Using timestamped output directory for forecast evaluation outputs")
        output_directory = BASE_FORECASTS_OUTPUT_PATH / make_folder_name(
            forecaster, model, timestamp_start_run
        )
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)
            print(f"Directory '{output_directory}' created.")
    else:
        output_directory = Path(output_dir)
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)
            print(f"Directory '{output_directory}' created.")

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
                tuples_per_source=tuples_per_source,
                is_async=is_async,
                output_directory=output_directory,
                most_recent_directory=most_recent_directory,
                load_dir=load_dir,
                run=run,
                forecaster_class=forecaster_class,
                eval_by_source=eval_by_source,
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
                tuples_per_source=tuples_per_source,
                is_async=is_async,
                output_directory=output_directory,
                most_recent_directory=most_recent_directory,
                load_dir=load_dir,
                run=run,
                forecaster_class=forecaster_class,
                eval_by_source=eval_by_source,
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

    # Aggregate and write stats by source question
    if eval_by_source:
        aggregate_stats_by_source(all_stats, output_directory)
        aggregate_stats_by_source(all_stats, most_recent_directory)

    # Print summary
    for metric in metrics:
        print(f"\n{metric}")
        for check_name, stats in all_stats.items():
            print(f"\n{check_name}:")
            print(stats)
            overall_stats = stats["overall"]
            print(
                f"  Overall: {overall_stats[metric]['num_violations']}/{overall_stats[metric]['num_samples']}"
            )

            if eval_by_source and "by_source" in stats:
                for source, source_stats in stats["by_source"].items():
                    print(
                        f"  {source}: {source_stats[metric]['num_violations']}/{source_stats[metric]['num_samples']}"
                    )
                    print(
                        f"    avg: {source_stats[metric]['avg_violation']:.3f}, "
                        f"avg_no_outliers: {source_stats[metric]['avg_violation_no_outliers']:.3f}, "
                        f"median: {source_stats[metric]['median_violation']:.3f}"
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
# python evaluation.py -f PromptedToCons_Forecaster -m gpt-4o-mini --run -n 3 --relevant_checks all | tee see_eval.txt
# python evaluation.py -f ConsistentForecaster -m gpt-4o-mini --run -n 2 -k NegChecker
