import json
import argparse


def parse_metrics(data):
    results = {}
    for question, question_data in data.items():
        overall_metrics = question_data["overall"]
        checker_metrics = question_data["by_checker"]

        results[question] = {"default": {}, "frequentist": {}}

        for metric_type in ["default", "frequentist"]:
            overall = overall_metrics[metric_type]
            results[question][metric_type] = {
                "avg_violation": overall["avg_violation"],
                "num_violations": overall["num_violations"],
                "num_samples": overall["num_samples"],
                "violation_rate": overall["num_violations"] / overall["num_samples"]
                if overall["num_samples"] > 0
                else 0,
            }

            # Calculate weighted violations
            num_checkers = len(checker_metrics)
            weighted_violations = (
                sum(
                    checker_data[metric_type]["num_violations"]
                    / checker_data[metric_type]["num_samples"]
                    for checker_data in checker_metrics.values()
                    if checker_data[metric_type]["num_samples"] > 0
                )
                / num_checkers
            )

            results[question][metric_type]["weighted_violations"] = weighted_violations

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse forecast metrics from JSON input and save results to JSON output."
    )
    parser.add_argument("--input_file", help="Path to the input JSON file")
    parser.add_argument("--output_file", help="Path to the output JSON file")
    args = parser.parse_args()

    # Load and parse the JSON data
    with open(args.input_file, "r") as f:
        data = json.load(f)

    results = parse_metrics(data)

    # Save results to JSON file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results have been saved to {args.output_file}")


if __name__ == "__main__":
    main()
