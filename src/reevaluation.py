import json
from pathlib import Path
from typing import Any
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
    CondCondChecker,
)
from common.path_utils import get_data_path
from common.utils import round_floats
import plotnine as p9
import pandas as pd

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
                tuple = round_floats(tuple, precision=7)
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
        df["viol_capped"] = df["violation"].apply(lambda x: min(x, cap))
    else:
        cap = 999999
    return (
        p9.ggplot(df, p9.aes(x="viol_capped"))
        # + p9.geom_histogram(binwidth=binwidth, closed='left')
        + p9.geom_density()
        # + p9.scale_x_continuous(
        #     #breaks=range(int(df['CappedValues'].min()), int(cap) + 1),
        #     labels=lambda l: ["%.1f" % v if v != cap else ">{cap}" for v in l])
        # + p9.theme_minimal()
    )


def plot_all(checker_viols, metric="default", binwidth=0.005, cap=None, ymax=30):
    """Plot each checker's violations on the same overlapping plot."""

    df = pd.DataFrame()
    for checker, viols in checker_viols.items():
        df = df._append(pd.DataFrame({"violation": viols[metric], "checker": checker}))
    if not cap:
        cap = 999999
    df["viol_capped"] = df["violation"].apply(lambda x: min(x, cap))
    return (
        p9.ggplot(df, p9.aes(x="viol_capped", color="checker"))
        + p9.geom_density()
        + p9.ylim(0, ymax)
        # + p9.scale_x_continuous(
        #     #breaks=range(int(df['CappedValues'].min()), int(cap) + 1),
        #     labels=lambda l: ["%.1f" % v if v != cap else ">{cap}" for v in l])
        # + p9.theme_minimal()
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
            stats = round_floats(stats, precision=7)
            f.write(json.dumps(stats, indent=4))


paths_adv = [
    "AdvancedForecaster_05-30-02-55",  # real qs
    "AdvancedForecaster_05-30-11-34",  # synthetic qs
]
paths_gpt_3_5 = [
    "BasicForecaster_05-31-12-24",  # real qs
    "BasicForecaster_05-31-12-18",  # synthetic qs
]
paths_gpt_4o = [
    "BasicForecaster_05-30-23-27",  # real qs
    "BasicForecaster_05-30-23-26",  # synthetic qs
]

paths = paths_adv + paths_gpt_3_5 + paths_gpt_4o
metrics = ["default", "frequentist"]

if __name__ == "__main__":
    # TODO add some notifications about what files will get modified, and y/n. ideally together with the edits that introduces cli args to this
    append_violations_all(paths_adv, metrics=metrics, recalc=True, write=True)
    append_violations_all(paths_gpt_3_5, metrics=metrics, recalc=True, write=True)
    append_violations_all(paths_gpt_4o, metrics=metrics, recalc=True, write=True)
    print(get_stats_from_paths(paths_adv, metrics=metrics, write="stats_adv.json"))
    print(
        get_stats_from_paths(paths_gpt_3_5, metrics=metrics, write="stats_gpt_3_5.json")
    )
    print(
        get_stats_from_paths(paths_gpt_4o, metrics=metrics, write="stats_gpt_4o.json")
    )
    # checker_viols, checker_checks = append_violations_all(
    #     paths_adv, metrics=metrics, recalc=False, write=False
    # )
    # print(plot(checker_viols["CondChecker"]["default"], cap=0.1))
    # print(plot_all(checker_viols, metric="default", cap=0.1))
    # print(plot_all(checker_viols, metric="frequentist", cap=1.0, ymax=10))
