import sys
import io
import json
import warnings
import asyncio
from pathlib import Path
import click
import logging
import functools
import concurrent.futures
from costly import Costlog
import numpy as np
from forecasters import Forecaster
from static_checks.Checker import (
    Checker,
    choose_checkers,
)
from common.path_utils import get_data_path
import common.llm_utils  # noqa
from common.llm_utils import reset_global_semaphore
from forecasters.create import make_forecaster
from evaluation_utils.utils import (
    create_output_directory,
    write_to_dirs,
)
from evaluation_utils.common_options import common_options, get_forecaster_config
from typing import Any

BASE_TUPLES_PATH: Path = get_data_path() / "tuples_scraped/"
# BASE_TUPLES_PATH: Path = get_data_path() / "tuples_newsapi/"
BASE_FORECASTS_OUTPUT_PATH: Path = get_data_path() / "forecasts"

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

metrics = ["default", "frequentist", "default_scaled"]


def get_stats(results: dict, label: str = "") -> dict:
    ret = {}
    for metric in metrics:
        print(f"{metric}")

        # Extract the violation and check results from the test
        violations = []
        checks = []
        for result in results:
            violation_data = result.get("violation_data", {})
            if metric in violation_data and isinstance(
                violation_data[metric]["violation"], (float, int)
            ):
                violations.append(violation_data[metric]["violation"])
            else:
                warnings.warn(
                    f"Violation {violation_data[metric]['violation']} is an error message not a number"
                )
            if metric in violation_data and isinstance(
                violation_data[metric]["check"], bool
            ):
                checks.append(violation_data[metric]["check"])
            else:
                warnings.warn(
                    f"Check {violation_data[metric]['check']} is an error message not a bool"
                )

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
            "num_samples_including_errors": len(results),
            "num_samples": len(violations),
            "num_violations": num_failed,
            "avg_violation": round(avg_violation, 6),
            "avg_violation_no_outliers": round(avg_violation_no_outliers, 6),
            "median_violation": round(median_violation, 6),
        }

    return ret


def validate_result(result: dict, keys: list[str]) -> bool:
    assert "line" in result, "results must contain a 'line' key"
    for key in keys:
        assert key in result["line"], f"line must contain a '{key}' key"
        assert (
            "forecast" in result["line"][key]
        ), f"line[{key}] must contain an 'forecast' key"
        assert (
            "prob" in result["line"][key]["forecast"]
        ), f"line[{key}]['forecast'] must contain a 'prob' key"
    return True


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
        overall = {"default": {}, "frequentist": {}, "default_scaled": {}}
        checker_count = len(stats["by_checker"])

        for metric in ["default", "frequentist", "default_scaled"]:
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


def aggregate_stats(all_stats: dict) -> dict:
    aggregate_stats = {}

    for metric in ["default", "frequentist", "default_scaled"]:
        aggregate_stats[metric] = {}
        tot_violation = 0.0
        n = 0
        aggregate_stats[metric]["avg_violation"] = round(
            np.mean(
                [
                    checker_stats["overall"][metric]["avg_violation"]
                    for checker_stats in all_stats.values()
                    if "overall" in checker_stats and metric in checker_stats["overall"]
                ]
            ),
            6,
        )
    return aggregate_stats


def load_existing_results(
    output_file: Path, max_lines: int | None = None
) -> list[dict[str, Any]]:
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f][:max_lines]
    return []


def process_check(
    check_name: str,
    checkers: dict[str, Checker],
    forecaster: Forecaster,
    tuples_per_source: int,
    is_async: bool,
    output_directory: Path,
    most_recent_directory: Path,
    load_dir: Path,
    run: bool,
    continue_run: bool,
    eval_by_source: bool,
    do_check: bool,
    num_lines: int | None = None,
    **kwargs,
) -> dict:
    print(f"Debug: Starting process_check for {check_name}")
    try:
        print("Checker: ", check_name)
        with open(checkers[check_name].path, "r", encoding="utf-8") as f:
            print(f"Path: {checkers[check_name].path}")
            all_tuples = [json.loads(line) for line in f]

        if eval_by_source:
            # TODO does this ignore num_lines? if yes, raise if num_lines is not None
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
            # if num_lines is None, this is all the tuples
            checker_tuples = all_tuples[:num_lines]

        keys = [key for key in checkers[check_name].TupleFormat.model_fields]
        print(f"Debug: keys: {keys}")
        print(f"Debug: Number of tuples loaded from dataset: {len(checker_tuples)}")

        dirs_to_write = [output_directory, most_recent_directory]
        output_file = output_directory / f"{check_name}.jsonl"

        if run:
            existing_results = []
            if continue_run and output_file.exists():
                existing_results = load_existing_results(
                    output_file, max_lines=num_lines
                )
                print(
                    f"Loaded {len(existing_results)} existing results for {check_name}"
                )

            start_index = len(existing_results)
            print(
                f"Number of checker_tuples to run: {len(checker_tuples) - start_index}"
            )

            results = existing_results
            batch_size = 25
            for start_batch in range(start_index, len(checker_tuples), batch_size):
                end_batch = min(start_batch + batch_size, len(checker_tuples))
                batch_tuples = checker_tuples[start_batch:end_batch]
                print(f"Number of batch_tuples: {len(batch_tuples)}")
                if is_async:
                    reset_global_semaphore()
                    results_batch = asyncio.run(
                        checkers[check_name].test(
                            forecaster,
                            do_check=do_check,
                            tuples=batch_tuples,
                            **kwargs,
                        )
                    )
                else:
                    results_batch = checkers[check_name].test_sync(
                        forecaster,
                        do_check=do_check,
                        tuples=batch_tuples,
                        **kwargs,
                    )

                print(f"results_batch: {results_batch}")
                print(f"len(results_batch): {len(results_batch)}")
                print(f"len(batch_tuples): {len(batch_tuples)}")
                assert len(results_batch) == len(batch_tuples)
                assert all(validate_result(result, keys) for result in results_batch)

                results.extend(results_batch)
                write_to_dirs(
                    results, f"{check_name}.jsonl", dirs_to_write, overwrite=True
                )

            print(f"Debug: Number of results after run: {len(results)}")
        else:
            results = load_existing_results(load_dir / f"{check_name}.jsonl")

        data = [result["line"] for result in results]
        all_answers = [
            {key: result["line"][key]["forecast"]["prob"] for key in keys}
            for result in results
        ]
        for line, answers, result in zip(data, all_answers, results):
            if "violation_data" in result and run:
                violation_data = result["violation_data"]
            else:
                violation_data = {}
                for metric in metrics:
                    violation_data[metric] = checkers[
                        check_name
                    ].check_from_elicited_probs(answers, metric)
            result["violation_data"] = violation_data

        print(f"Debug: Calculating stats for {check_name}")
        if eval_by_source:
            stats = {"overall": get_stats(results, label=check_name), "by_source": {}}
            for source, source_data in source_questions.items():
                source_results = [
                    r
                    for r in results
                    if r["line"]["P"]["question"]["metadata"].get("source_question")
                    == source
                    or r["line"]["P"]["question"]["title"] == source
                ]
                stats["by_source"][source] = get_stats(
                    source_results, label=f"{check_name}_{source}"
                )
                stats["by_source"][source]["source_id"] = source_data["source_id"]
        else:
            stats = {"overall": get_stats(results, label=check_name)}

        if not run:
            # write results back to file
            with open(load_dir / f"{check_name}.jsonl", "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

        print(f"Debug: Finished process_check for {check_name}")
        return stats
    except Exception as e:
        print(f"Debug: Error in process_check for {check_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


@click.command()
@common_options
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
    default=["all"],
    help='Relevant checks to perform. In case of "all", all checkers are used.',
)
@click.option(
    "--threads",
    "use_threads",
    is_flag=True,
    default=False,
    help="Use threads to run the forecaster on different checks",
)
@click.option(
    "--tuple_dir",
    type=click.Path(),
    required=False,
    help="Path to the tuple file",
)
@click.option(
    "--eval_by_source",
    "-s",
    is_flag=True,
    default=False,
    help="Evaluate consistency scores per source question",
)
@click.option(
    "--skip_check",
    is_flag=True,
    default=False,
    help="Compute and append violation data",
)
@click.option(
    "--simulate",
    is_flag=True,
    default=False,
    help="Simulate the evaluation",
)
@click.option(
    "--continue",
    "continue_run",
    is_flag=True,
    default=False,
    help="Continue from the last processed line when --run is used",
)
def main(
    forecaster_class: str | None,
    custom_path: str | None,
    config_path: str | None,
    forecaster_options: list[str] | None,
    run: bool,
    continue_run: bool,
    load_dir: str | None,
    tuples_per_source: int,
    relevant_checks: list[str],
    is_async: bool,
    use_threads: bool,
    num_lines: int,
    tuple_dir: str | None = None,
    output_dir: str | None = None,
    eval_by_source: bool = False,
    skip_check: bool = False,
    simulate: bool = False,
):
    do_check = not skip_check

    # IMPORTANT!!! If you remove this, it will prune your data down to 3 lines.
    if num_lines == -1 or not run:
        num_lines = None

    # Print arguments
    print("Arguments:")
    print(f"  forecaster_class: {forecaster_class}")
    print(f"  custom_path: {custom_path}")
    print(f"  num_lines: {num_lines}")
    print(f"  run: {run}")
    print(f"  load_dir: {load_dir}")
    print(f"  is_async: {is_async}")
    print(f"  output_dir: {output_dir}")
    cl = Costlog(mode="jsonl")

    if run:
        forecaster_config = get_forecaster_config(config_path, forecaster_options)
        print(f"  forecaster_config: {forecaster_config}")

        forecaster = make_forecaster(
            forecaster_class=forecaster_class,
            custom_path=custom_path,
            forecaster_config=forecaster_config,
        )
        if tuple_dir is None:
            tuple_dir = BASE_TUPLES_PATH
        tuple_dir = Path(tuple_dir)
        if not tuple_dir.exists():
            assert tuple_dir.exists(), f"Tuple directory {tuple_dir} does not exist"
        output_directory, most_recent_directory = create_output_directory(
            forecaster, BASE_FORECASTS_OUTPUT_PATH, output_dir
        )
        checkers: dict[str, Checker] = choose_checkers(relevant_checks, tuple_dir)
        logged_config = {
            "forecaster_class": forecaster.__class__.__name__,
            "full_forecaster_config": forecaster.dump_config(),
            "checkers": [checker.dump_config() for name, checker in checkers.items()],
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
    else:
        forecaster_config = None
        forecaster = None
        load_dir, output_directory, most_recent_directory = (
            Path(load_dir),
            Path(load_dir),
            Path(load_dir),
        )

        checkers: dict[str, Checker] = choose_checkers(relevant_checks, tuple_dir)

    # load_dir = validate_load_directory(run, load_dir, most_recent_directory)

    print(f"Relevant checks: {checkers.keys()}")
    print(f"Tuple dir: {tuple_dir}")
    print(f"Output dir: {output_directory}")

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
                num_lines=num_lines,
                tuples_per_source=tuples_per_source,
                is_async=is_async,
                output_directory=output_directory,
                most_recent_directory=most_recent_directory,
                load_dir=load_dir,
                run=run,
                continue_run=continue_run,
                eval_by_source=eval_by_source,
                do_check=do_check,
                cost_log=cl,
                simulate=simulate,
            )
            all_stats = {
                check_name: stats
                for check_name, stats in zip(
                    checkers.keys(), executor.map(process_check_func, checkers.keys())
                )
                if stats is not None
            }
    else:
        for check_name in checkers.keys():
            stats = process_check(
                check_name=check_name,
                checkers=checkers,
                forecaster=forecaster,
                num_lines=num_lines,
                tuples_per_source=tuples_per_source,
                is_async=is_async,
                output_directory=output_directory,
                most_recent_directory=most_recent_directory,
                load_dir=load_dir,
                run=run,
                continue_run=continue_run,
                eval_by_source=eval_by_source,
                do_check=do_check,
                cost_log=cl,
                simulate=simulate,
            )
            if stats is not None:
                all_stats[check_name] = stats

    # Recompute final stats
    for check_name, stats in all_stats.items():
        if check_name in ["forecaster", "full_forecaster_config", "aggregated"]:
            continue
        results = load_existing_results(output_directory / f"{check_name}.jsonl")
        recomputed_stats = get_stats(results, label=check_name)
        all_stats[check_name]["overall"] = recomputed_stats

    all_stats["aggregated"] = aggregate_stats(all_stats)
    if run:
        all_stats["forecaster"] = forecaster.__class__.__name__
        all_stats["full_forecaster_config"] = forecaster.dump_config()

    print("Cost log totals")
    print("---------------")
    print(cl.totals)
    print(cl.totals_by_model)
    print("---------------")

    with open(output_directory / "stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=4)
    with open(
        most_recent_directory / "stats_summary.json", "w", encoding="utf-8"
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
            if check_name in ["forecaster", "full_forecaster_config", "aggregated"]:
                continue
            print(f"\n{check_name}:")
            print(f"{stats=}")
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
        if run:
            print(f"Forecaster: {all_stats['forecaster']}")
            print(f"Forecaster Config: {all_stats['full_forecaster_config']}")

        print(f"Output written to {output_directory}")
        print(f"Summary written to {output_directory}/stats_summary.json")


if __name__ == "__main__":
    main()


# run the script with the following commands:

# Basic example with AdvancedForecaster:
# python evaluation.py -f AdvancedForecaster -c forecasters/forecaster_configs/advanced/cheap_haiku.yaml --run -n 3 --relevant_checks all | tee see_eval.txt

# Using BasicForecaster with specific checks:
# python evaluation.py -f BasicForecaster -o model=gpt-4o-mini --run -n 50 -k ParaphraseChecker -k CondCondChecker | tee see_eval.txt

# Using PromptedToCons_Forecaster:
# python evaluation.py -f PromptedToCons_Forecaster -o model=gpt-4o-mini --run -n 3 --relevant_checks all | tee see_eval.txt

# Using ConsistentForecaster, recursive
# python evaluation.py -f ConsistentForecaster -o model=gpt-4o-mini -o checks='[NegChecker, ParaphraseChecker]' -o depth=4 --run -n 100 --relevant_checks all --async -o use_generate_related_questions=True | tee see_eval.txt

# just recalculate

# python evaluation.py --load_dir data/forecasts/CF_NP4_test -k all
