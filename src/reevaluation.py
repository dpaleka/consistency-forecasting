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
    tuple["violations"][metric] = v
    return tuple


def append_violations(
    checker: Checker,
    tuples_files: Path,
    metric=None,
    write=False,
):
    """append_violation to each tuple in the jsonl file and write

    Args:
        checker (Checker): Checker object
        tuples_path (Path): Folder within FORECASTS_PATH or path to jsonl file
        metric (str): metric name
    """
    print(f"Appending violations for {checker.name} in {tuples_files} ...")
    viols = []
    if metric is None:
        metric = "default"
    if isinstance(tuples_files, str):
        tuples_files = Path(tuples_files)
    if not tuples_files.suffix == ".jsonl":
        tuples_files = FORECASTS_PATH / tuples_files / f"{checker.name}.jsonl"
    else:
        tuples_files = FORECASTS_PATH / tuples_files
    with open(tuples_files, "r", encoding="utf-8") as f:
        tuples = [json.loads(line) for line in f]
    tuples = [append_violation(checker, tuple, metric=metric) for tuple in tuples]
    viols = [tuple["violations"][metric] for tuple in tuples]
    if write:
        with open(tuples_files, "w", encoding="utf-8") as f:
            for tuple in tuples:
                f.write(json.dumps(tuple) + "\n")
    return viols


def append_violations_all(tuples_folders: list[Path], metric=None, write=False):
    checker_viols = {checker: [] for checker in checkers}
    for tuples_folder in tuples_folders:
        for name, checker in checkers.items():
            viols = append_violations(
                checker, tuples_folder, metric=metric, write=write
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


paths = [
    "AdvancedForecaster_05-30-02-55",  # real qs
    "AdvancedForecaster_05-30-11-34",  # synthetic qs
    "BasicForecaster_05-31-12-24",  # real qs
    "BasicForecaster_05-31-12-18",  # synthetic qs
]

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python reevaluation.py <tuples_folder>")
    #     sys.exit(1)
    # tuples_folder = Path(sys.argv[1])
    # append_violations_all(tuples_folder)
    # print("Done.")
    # reset_all_appended_viols(paths)
    append_violations_all(paths, write=True)
