import sys
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from forecasters import Forecaster, AdvancedForecaster
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

model = "gpt-3.5-turbo"
# model = "gpt-4-turbo-2024-04-09"

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


relevant_checks = ["CondChecker"]
# relevant_checks = list(checkers.keys())

# load config
config_path = CONFIGS_DIR / "cheap_haiku.yaml"
# load all of yaml into a dict
import yaml

with open(config_path, "r") as f:
    config: dict[str, Any] = yaml.safe_load(f)

forecaster = AdvancedForecaster(**config)
print(forecaster.dump_config())


def make_folder_name(forecaster: Forecaster, model: str, timestamp: datetime) -> str:
    return f"{forecaster.__class__.__name__}_{timestamp.strftime('%m-%d-%H-%M-%S')}"


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


NUM_LINES = 3


logged_config = {
    "forecaster_class": forecaster.__class__.__name__,
    "forecaster": forecaster.dump_config(),
    "model": model,
    "num_lines": NUM_LINES,
}

with open(output_directory / "config.jsonl", "w") as f, open(
    most_recent_directory / "config.jsonl", "w"
) as f2:
    f.write(json.dumps(logged_config) + "\n")
    f2.write(json.dumps(logged_config) + "\n")

RUN = False
# LOAD_DIR = BASE_FORECASTS_OUTPUT_PATH / "AdvancedForecaster_05-25-22-32-52"
LOAD_DIR = most_recent_directory


def validate_result(result: dict, keys: list[str]) -> None:
    assert "line" in result, "results must contain a 'line' key"
    for key in keys:
        assert key in result["line"], f"line must contain a '{key}' key"
        assert (
            "elicited_prob" in result["line"][key]
        ), f"line[{key}] must contain an 'elicited_prob' key"
    return True


all_stats = {}
for check_name in relevant_checks:
    print("Checker: ", check_name)
    with open(checkers[check_name].path, "r") as f:
        checker_tuples = [json.loads(line) for line in f.readlines()]
        checker_tuples = checker_tuples[:NUM_LINES]

    keys = [key for key in checker_tuples[0].keys() if key not in ["metadata"]]
    print(f"keys: {keys}")

    if RUN:
        assert LOAD_DIR is None, "LOAD_DIR must be None if RUN is True"
        if isinstance(forecaster, AdvancedForecaster):
            results = checkers[check_name].test_sync(
                forecaster, do_check=False, num_lines=NUM_LINES, **config
            )
        else:
            results = checkers[check_name].test_sync(
                forecaster, do_check=False, num_lines=NUM_LINES, model=model
            )

        assert len(results) == NUM_LINES, "results must be of length NUM_LINES"
        assert all(validate_result(result, keys) for result in results)

        with open(output_directory / f"{check_name}.jsonl", "w") as f, open(
            most_recent_directory / f"{check_name}.jsonl", "w"
        ) as f2:
            for result in results:
                f.write(json.dumps(result) + "\n")
                f2.write(json.dumps(result) + "\n")
    else:
        assert LOAD_DIR is not None, "LOAD_DIR must be set if RUN is False"
        LOAD_DIR = Path(LOAD_DIR)

        with open(LOAD_DIR / f"{check_name}.jsonl", "r") as f:
            results = [json.loads(line) for line in f.readlines()]

        assert (
            len(results) >= NUM_LINES
        ), "results must contain at least NUM_LINES elements"
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
        # print(f"line: {line}")
        print(f"answers: {answers}")
        violation_data = checkers[check_name].check_from_elicited_probs(answers)
        print(f"violation_data: {violation_data}")
        result.update(violation_data)

    stats = get_stats(results, label=check_name)
    all_stats[check_name] = stats

for check_name, stats in all_stats.items():
    print(f"{stats['label']}: {stats['num_violations']}/{stats['num_samples']}")

print("\n\n")
for check_name, stats in all_stats.items():
    print(
        f"{check_name} | avg: {stats['avg_violation']:.3f}, median: {stats['median_violation']:.3f}"
    )

# to save the output to a file, run this script as
# python src/evaluation.py | tee src/data/evaluation.txt
