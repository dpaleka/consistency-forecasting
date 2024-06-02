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
    v = checker.violation(answers, metric=metric)
    print(f"It's {v}. is_inconsistent: {v > 0.01}.")
    tuple["violations"][metric] = v
    return tuple


def append_violations(
    checker: Checker,
    tuples_files: Path,
    metrics: list[str] | None = None,
    recalc: bool = False,
    write: bool = False,
):
    """append_violation to each tuple in the jsonl file and write

    Args:
        checker (Checker): Checker object
        tuples_path (Path): Folder within FORECASTS_PATH or path to jsonl file
        metrics (list[str]): metric names
        recalc (bool): recalculate violations? Or just read them?
        write (bool): write the updated tuples back to the file? Or just return the violations?
    """
    print(f"Appending violations for {checker.name} in {tuples_files} ...")
    viols = []
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

    for metric in metrics:
        if recalc:
            tuples = [
                append_violation(checker, tuple, metric=metric) for tuple in tuples
            ]
            viols = [tuple["violations"][metric] for tuple in tuples]
        else:
            for tuple in tuples:
                if "violations" in tuple and metric in tuple["violations"]:
                    viols.append(tuple["violations"][metric])
                else:
                    tuple = append_violation(checker, tuple, metric=metric)
                    viols.append(tuple["violations"][metric])

    if write:
        with open(tuples_files, "w", encoding="utf-8") as f:
            for tuple in tuples:
                f.write(json.dumps(tuple) + "\n")
    return viols


def append_violations_all(
    tuples_folders: list[Path], metrics=None, recalc=False, write=False
):
    checker_viols = {checker: [] for checker in checkers}
    for tuples_folder in tuples_folders:
        for name, checker in checkers.items():
            viols = append_violations(
                checker, tuples_folder, metrics=metrics, recalc=recalc, write=write
            )
            checker_viols[name].extend(viols)
    return checker_viols


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
        with open(tuples_files, "w", encoding="utf-8") as f:
            for tuple in tuples:
                f.write(json.dumps(tuple) + "\n")
    return


def reset_all_appended_viols(tuples_folders: list[Path]):
    for tuples_folder in tuples_folders:
        reset_appended_viols(tuples_folder)
    return


def get_stats(checker_viols):
    stats = {}
    for checker, viols in checker_viols.items():
        stats[checker] = {
            "total": len(viols),
            "violated": len([v for v in viols if v > 0.01]),
            "viol_avg": sum(viols) / len(viols),
            "viol_med": sorted(viols)[len(viols) // 2],
        }
    print(stats)
    return stats


def get_stats_from_paths(paths, metrics=None, write: Path | None = None):
    checker_viols = append_violations_all(
        paths, metrics=metrics, recalc=False, write=False
    )
    stats = get_stats(checker_viols)
    if write:
        if isinstance(write, str):
            write = FORECASTS_PATH / write
        with open(write, "w", encoding="utf-8") as f:
            f.write(json.dumps(stats, indent=4))


paths_adv = [
    "AdvancedForecaster_05-30-02-55",  # real qs
    "AdvancedForecaster_05-30-11-34",  # synthetic qs
]
paths_basic = [
    "BasicForecaster_05-31-12-24",  # real qs
    "BasicForecaster_05-31-12-18",  # synthetic qs
]
paths = paths_adv + paths_basic
# metrics = ["default", "frequentist"]
metrics = ["default"]

if __name__ == "__main__":
    print(get_stats_from_paths(paths_basic, metrics=metrics, write="stats_basic.json"))
    print(get_stats_from_paths(paths_adv, metrics=metrics, write="stats_adv.json"))
