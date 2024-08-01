import json
import argparse
from pathlib import Path
from typing import Any
from static_checks.Checker import (
    Checker,
    NegChecker,
    AndChecker,
    OrChecker,
    AndOrChecker,
    ButChecker,
    ParaphraseChecker,
    CondChecker,
    CondCondChecker,
    ConsequenceChecker,
)
from common.path_utils import get_data_path
from common.utils import round_floats
import plotnine as p9
import pandas as pd
import numpy as np

TUPLES_PATH: Path = get_data_path() / "tuples/"
FORECASTS_PATH: Path = get_data_path() / "forecasts"


checkers: dict[str, Checker] = {
    "NegChecker": NegChecker(path=TUPLES_PATH / "NegChecker.jsonl"),
    "AndChecker": AndChecker(path=TUPLES_PATH / "AndChecker.jsonl"),
    "OrChecker": OrChecker(path=TUPLES_PATH / "OrChecker.jsonl"),
    "AndOrChecker": AndOrChecker(path=TUPLES_PATH / "AndOrChecker.jsonl"),
    "ButChecker": ButChecker(path=TUPLES_PATH / "ButChecker.jsonl"),
    "CondChecker": CondChecker(path=TUPLES_PATH / "CondChecker.jsonl"),
    "ConsequenceChecker": ConsequenceChecker(
        path=TUPLES_PATH / "ConsequenceChecker.jsonl"
    ),
    "ParaphraseChecker": ParaphraseChecker(
        path=TUPLES_PATH / "ParaphraseChecker.jsonl"
    ),
    "CondCondChecker": CondCondChecker(path=TUPLES_PATH / "CondCondChecker.jsonl"),
}

paths = {
    "adv": [
        "AdvancedForecaster_05-30-02-55",  # real qs, n = 50
        "AdvancedForecaster_05-30-11-34",  # synthetic qs, n = 30
    ],
    "gpt_3_5": [
        "BasicForecaster_05-31-12-24",  # real qs, n = 50
        "BasicForecaster_05-31-12-18",  # synthetic qs, n = 30
    ],
    "gpt_4o": [
        "BasicForecaster_05-30-23-27",  # real qs, n = 50
        "BasicForecaster_05-30-23-26",  # synthetic qs, n = 30
    ],
    "cf_gpt_4omini_sample": [
        "ConsistentForecaster_07-19-18-59",  # real qs, n = 3
    ],
    "cf_gpt_4omini": [
        "ConsistentForecaster_07-24-14-58",  # real qs, n = 50
    ],
}


def append_violation(
    checker: Checker, tuple: dict[str, Any], metric=None
) -> dict[str, Any]:
    assert "line" in tuple
    answers = {name: fq["elicited_prob"] for name, fq in tuple["line"].items()}
    print(f"Computing violation for {answers} ...")
    if "violations" not in tuple:
        tuple["violations"] = {}
    if "checks" not in tuple:
        tuple["checks"] = {}
    v = checker.violation(answers, metric=metric)
    c = checker.check(answers, metric=metric)
    print(f"Violation: {v}. Check: {c}.")
    tuple["violations"][metric] = v
    tuple["checks"][metric] = c
    return tuple


def append_violations(
    checker: Checker,
    tuples_files: Path,
    metrics: list[str] | None = None,
    recalc: bool = False,
    write: bool = False,
) -> dict[str, list[float]]:
    """append_violation to each tuple in the jsonl file and write

    Args:
        checker (Checker): Checker object
        tuples_path (Path): Folder within FORECASTS_PATH or path to jsonl file
        metrics (list[str]): metric names
        recalc (bool): recalculate violations? Or just read them?
        write (bool): write the updated tuples back to the file? Or just return the violations?

    Returns:
        dict[str, list[float]]: {metric: [violations]}
    """
    print(f"Appending violations for {checker.name} in {tuples_files} ...")

    if metrics is None:
        metrics = ["default"]
    if isinstance(tuples_files, str):
        tuples_files = Path(tuples_files)
    if not tuples_files.suffix == ".jsonl":
        tuples_files = FORECASTS_PATH / tuples_files / f"{checker.name}.jsonl"
    else:
        tuples_files = FORECASTS_PATH / tuples_files
    with open(tuples_files, "r", encoding="utf-8") as f:
        tuples = [json.loads(line) for line in f]
        print(f"Number of tuples: {len(tuples)}")

    viols_all = {metric: [] for metric in metrics}
    checks_all = {metric: [] for metric in metrics}
    for metric in metrics:
        if recalc:
            tuples = [
                append_violation(checker, tuple, metric=metric) for tuple in tuples
            ]
            viols = [tuple["violations"][metric] for tuple in tuples]
            checks = [tuple["checks"][metric] for tuple in tuples]
        else:
            viols = []
            checks = []
            for tuple in tuples:
                if "violations" in tuple and metric in tuple["violations"]:
                    viols.append(tuple["violations"][metric])
                    checks.append(tuple["checks"][metric])
                else:
                    tuple = append_violation(checker, tuple, metric=metric)
                    viols.append(tuple["violations"][metric])
                    checks.append(tuple["checks"][metric])
        viols_all[metric] = viols
        checks_all[metric] = checks

    if write:
        with open(tuples_files, "w", encoding="utf-8") as f:
            for tuple in tuples:
                tuple = round_floats(tuple, precision=5)
                f.write(json.dumps(tuple) + "\n")
    return viols_all, checks_all


def append_violations_all(
    tuples_folders: list[Path],
    metrics=None,
    recalc=False,
    write=False,
) -> dict[str, dict[str, list[float]]]:
    """append_violations for all checkers in all tuples_folders

    Args:
        checkers (dict[str, Checker]): {name: Checker}
        tuples_folders (list[Path]): list of folders within FORECASTS_PATH
        metrics (list[str]): metric names
        recalc (bool): recalculate violations? Or just read them?
        write (bool): write the updated tuples back to the file? Or just return the violations?

    Returns:
        dict[str, dict[str, list[float]]]: {checker: {metric: [violations]}}
    """
    checker_viols = {
        checker: {metric: [] for metric in metrics} for checker in checkers
    }
    checker_checks = {
        checker: {metric: [] for metric in metrics} for checker in checkers
    }
    for tuples_folder in tuples_folders:
        for name, checker in checkers.items():
            viols, checks = append_violations(
                checker, tuples_folder, metrics=metrics, recalc=recalc, write=write
            )
            for metric, v in viols.items():
                checker_viols[name][metric].extend(v)
            for metric, c in checks.items():
                checker_checks[name][metric].extend(c)
    return checker_viols, checker_checks


def reset_appended_viols(tuples_folder: Path):
    for name, checker in checkers.items():
        if not isinstance(tuples_folder, Path):
            tuples_folder = FORECASTS_PATH / tuples_folder
        tuples_files = tuples_folder / f"{checker.name}.jsonl"
        with open(tuples_files, "r", encoding="utf-8") as f:
            tuples = [json.loads(line) for line in f]
        for tuple in tuples:
            if "violations" in tuple:
                del tuple["violations"]
            if "checks" in tuple:
                del tuple["checks"]
        with open(tuples_files, "w", encoding="utf-8") as f:
            for tuple in tuples:
                f.write(json.dumps(tuple) + "\n")
    return


def reset_all_appended_viols(tuples_folders: list[Path]):
    for tuples_folder in tuples_folders:
        reset_appended_viols(tuples_folder)
    return


def get_stats(checker_viols, checker_checks):
    """get stats from checker_viols, checker_checks

    Args:
        checker_viols (dict[str, dict[str, list[float]]): {checker: {metric: [violations]}}
        checker_checks (dict[str, dict[str, list[bool]]): {checker: {metric: [checks]}}

    Returns:
        dict[str, dict[str, float]]:
        {  checker:
            {
                {  metric:
                    {  n: int,
                       violated: int,
                       viol_avg: float,
                       viol_med: float
                    }
                }
            }
        }
    """
    stats = {}
    for checker in checkers:
        viols = checker_viols[checker]
        checks = checker_checks[checker]
        stats[checker] = {}
        for metric in viols:
            v = viols[metric]
            c = checks[metric]
            n = len(v)
            violated = len(c) - sum(c)
            viol_avg = sum(v) / n
            viol_med = sorted(v)[n // 2]
            stats[checker][metric] = {
                "n": n,
                "violated": violated,
                "viol_avg": viol_avg,
                "viol_med": viol_med,
            }
    return stats


def plot(viols: list[float], binwidth=0.005, cap=None):
    """Plot a histogram of violations."""
    df = pd.DataFrame({"violation": viols})
    if cap:
        df["Violation"] = df["violation"].apply(lambda x: min(x, cap))
    else:
        cap = max(viols)

    # Create custom breaks and labels
    breaks = np.arange(0, cap + binwidth, cap / 5)
    labels = [f"{b:.2f}" for b in breaks[:-1]] + [f"> {cap:.2f}"]

    return (
        p9.ggplot(df, p9.aes(x="Violation"))
        + p9.geom_density()
        + p9.scale_x_continuous(breaks=breaks, labels=labels)
        + p9.ylab("Density")
    )


def plot_all(checker_viols, metric="default", binwidth=0.005, cap=None, ymax=30):
    """Plot each checker's violations on the same overlapping plot."""

    df = pd.DataFrame()
    for checker, viols in checker_viols.items():
        df = df._append(pd.DataFrame({"violation": viols[metric], "checker": checker}))
    if not cap:
        cap = df["violation"].max()
    df["Violation"] = df["violation"].apply(lambda x: min(x, cap))

    # Create custom breaks and labels
    breaks = np.arange(0, cap + binwidth, cap / 5)
    labels = [f"{b:.2f}" for b in breaks[:-1]] + [f"> {cap:.2f}"]

    return (
        p9.ggplot(df, p9.aes(x="Violation", color="checker"))
        + p9.geom_density()
        + p9.ylim(0, ymax)
        + p9.scale_x_continuous(breaks=breaks, labels=labels)
        + p9.ylab("Density")
    )


def get_stats_from_paths(paths, metrics=None, write: Path | None = None):
    checker_viols, checker_checks = append_violations_all(
        paths, metrics=metrics, recalc=False, write=False
    )
    stats = get_stats(checker_viols, checker_checks)
    if write:
        if isinstance(write, str):
            write = FORECASTS_PATH / write
        with open(write, "w", encoding="utf-8") as f:
            stats = round_floats(stats, precision=4)
            f.write(json.dumps(stats, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process violations for checkers.")
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="*",
        help="Metrics to calculate violations for; by default ['default', 'frequentist']",
    )
    parser.add_argument(
        "-f",
        "--forecasters",
        nargs="*",
        help=(
            "Forecasters to calculate violations for; by default "
            "['adv', 'gpt_3_5', 'gpt_4o', 'cf_gpt_4omini_sample']"
        ),
    )
    parser.add_argument(
        "-c",
        "--checkers",
        nargs="*",
        help=(
            "Checkers to calculate violations for; by default "
            "['NegChecker', 'AndChecker', 'OrChecker', 'AndOrChecker', 'ButChecker', 'CondChecker', "
            "'ConsequenceChecker', 'ParaphraseChecker', 'CondCondChecker']"
        ),
    )
    parser.add_argument(
        "-r", "--recalculate", action="store_true", help="Recalculate violations"
    )
    parser.add_argument(
        "-x", "--reset", action="store_true", help="Reset all appended violations"
    )
    args = parser.parse_args()
    
    RECALC: bool = args.recalculate

    print(f"Recalculating violations: {RECALC}")

    # Ask for confirmation before modifying files
    if RECALC:
        confirm = input(
            "This will modify existing files and take some time to run. Are you sure you want to continue? (y/[n]): "
        )
        if confirm.lower() != "y":
            print("Operation cancelled.")
            exit()

    if args.metrics:
        metrics = args.metrics
    else:
        metrics = ["default", "frequentist"]
    
    if args.forecasters:
        paths = {k: v for k, v in paths.items() if k in args.forecasters}
    
    if args.checkers:
        checkers = {k: v for k, v in checkers.items() if k in args.checkers}

    if args.reset:
        reset_all_appended_viols(
            [
                FORECASTS_PATH / path / checker.name
                for path in paths.values()
                for checker in checkers.values()
            ]
        )
        exit()
        
    for forecaster, paths in paths.items():
        checker_viols, checker_checks = append_violations_all(
            paths, metrics=metrics, recalc=RECALC, write=True
        )
        print(get_stats_from_paths(paths, metrics=metrics, write=f"stats_{forecaster}.json"))
        
        # neaten up before plotting
        checker_viols.pop("AndChecker", None)
        checker_viols.pop("ConsequenceChecker", None)
        keys_to_rename = [k for k in checker_viols.keys() if k.endswith("Checker")]
        for k in keys_to_rename:
            checker_viols[k.replace("Checker", "")] = checker_viols.pop(k)
        
        # plot:
        for checker, viols in checker_viols.items():
            for metric, v in viols.items():
                plot(v, cap=0.1).save(get_data_path() / "figs" / f"{forecaster}_{checker}_{metric}.png")
        for metric in metrics:
            plot_all(checker_viols, metric=metric, cap=0.1).save(get_data_path() / "figs" / f"{forecaster}_all_{metric}.png")
            plot_all(checker_viols, metric=metric, cap=1.0, ymax=10).save(get_data_path() / "figs" / f"{forecaster}_all_{metric}_zoomed.png")
        
        
# Example usage:
# python src/reevaluation.py -m default frequentist -f adv gpt_3_5 gpt_4o cf_gpt_4omini_sample -c NegChecker AndChecker OrChecker AndOrChecker ButChecker CondChecker ConsequenceChecker ParaphraseChecker CondCondChecker -r        