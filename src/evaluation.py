from forecasters import BasicForecaster
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
from pathlib import Path
from common.path_utils import get_data_path

basic_forecaster = BasicForecaster()

BASE_DATA_PATH: Path = get_data_path() / "tuples/"

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


# relevant_keys = ["ConsequenceChecker"]
relevant_keys = list(checkers.keys())

all_stats = {}
for key in relevant_keys:
    print("Checker: ", key)
    results = checkers[key].test(basic_forecaster, model=model)
    # Log the messages being sent to the OpenAI API if results is a dict
    if isinstance(results, dict) and "messages" in results:
        print(f"Messages sent to OpenAI API for {key}: {results['messages']}")
    elif isinstance(results, list) and results:
        # Assuming each item in results is a dict that contains a 'messages' key
        messages = [result.get("messages", "No messages found") for result in results]
        print(f"Messages sent to OpenAI API for {key}: {messages}")
    else:
        print(f"No messages sent to OpenAI API for {key}")
    stats = get_stats(results, label=key)
    all_stats[key] = stats

for key, stats in all_stats.items():
    print(f"{stats['label']}: {stats['num_violations']}/{stats['num_samples']}")

print("\n\n")
for key, stats in all_stats.items():
    print(
        f"{key} | avg: {stats['avg_violation']:.3f}, median: {stats['median_violation']:.3f}"
    )

# to save the output to a file, run this script as
# python src/evaluation.py | tee src/data/evaluation.txt
