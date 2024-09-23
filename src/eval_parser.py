import json
import csv
import argparse
from typing import Dict, Any


def load_json_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return json.load(file)


def calculate_metrics(
    data: Dict[str, Any], weighted: bool = False
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results = {}
    for question, question_data in data.items():
        results[question] = {"default": {}, "frequentist": {}}
        for stat_type in ["default", "frequentist"]:
            overall = question_data["overall"][stat_type]
            checkers = question_data["by_checker"]

            # Calculate raw metrics
            raw_avg_violation = overall["avg_violation"]
            raw_num_violations = overall["num_violations"]

            # Calculate weighted metrics
            if weighted:
                weighted_violations = []
                weighted_num_violations = []
                for checker in checkers.values():
                    if stat_type in checker:
                        weighted_violations.append(checker[stat_type]["avg_violation"])
                        weighted_num_violations.append(
                            checker[stat_type]["num_violations"]
                        )
                weighted_avg_violation = (
                    sum(weighted_violations) / len(weighted_violations)
                    if weighted_violations
                    else 0
                )
                weighted_num_violations = (
                    sum(weighted_num_violations) / len(weighted_num_violations)
                    if weighted_num_violations
                    else 0
                )
            else:
                weighted_avg_violation = raw_avg_violation
                weighted_num_violations = raw_num_violations

            results[question][stat_type] = {
                "raw_avg_violation": raw_avg_violation,
                "raw_num_violations": raw_num_violations,
                "weighted_avg_violation": weighted_avg_violation,
                "weighted_num_violations": weighted_num_violations,
            }

    return results


def save_results_json(results: Dict[str, Dict[str, Dict[str, float]]], file_path: str):
    with open(file_path, "w") as file:
        json.dump(results, file, indent=2)


def save_results_csv(results: Dict[str, Dict[str, Dict[str, float]]], file_path: str):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Question", "Stat Type", "Metric", "Value"])
        for question, metrics in results.items():
            for stat_type in ["default", "frequentist"]:
                for metric, value in metrics[stat_type].items():
                    writer.writerow([question, stat_type, metric, value])


def main():
    parser = argparse.ArgumentParser(
        description="Process forecasting metrics from JSON input."
    )
    parser.add_argument(
        "--input",
        default="src/data/forecasts/experiment/stats_by_source_question.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output",
        default="src/data/forecasts/experiment/per_question_consistency",
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument("--weighted", action="store_true", help="Use weighted metrics")
    args = parser.parse_args()

    data = load_json_data(args.input)
    results = calculate_metrics(data, args.weighted)

    if args.format == "json":
        save_results_json(results, args.output)
    else:
        save_results_csv(results, args.output)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
