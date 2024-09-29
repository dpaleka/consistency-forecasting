import argparse
import json
import os
from typing import List, Dict, TypedDict
import matplotlib.pyplot as plt
import numpy as np


class ForecasterPair(TypedDict):
    name: str
    short_name: str
    ground_truth_dir: str
    eval_dir: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot consistency checks against ground truth Brier score."
    )
    parser.add_argument(
        "-d",
        "--directories",
        nargs="+",
        help="List of forecaster directory pairs (ground_truth eval).",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a file containing list of forecaster directory pairs.",
    )
    parser.add_argument(
        "-o", "--output_dir", default="output_plots", help="Directory to save plots."
    )
    parser.add_argument(
        "--all", action="store_true", help="Use all forecasters for a single dataset"
    )
    parser.add_argument(
        "--include_perplexity",
        action="store_true",
        help="Include perplexity in the plots",
    )
    parser.add_argument(
        "-m",
        "--gt_metric",
        choices=[
            "avg_brier_score",
            "avg_platt_brier_score",
            "tuned_brier_baseline",
            "avg_brier_score_scaled",
            "avg_platt_brier_score_scaled",
            "tuned_brier_baseline_scaled",
            "avg_log_score",
        ],
        default="avg_brier_score",
        help="Ground truth metric to use",
    )
    parser.add_argument(
        "-t",
        "--cons_metric_type",
        choices=["default", "frequentist", "default_scaled"],
        default="default",
        help="Consistency metric type to use",
    )
    parser.add_argument(
        "-c",
        "--cons_metric",
        choices=["avg_violation", "avg_violation_no_outliers", "median_violation"],
        default="avg_violation",
        help="Consistency metric to use",
    )
    parser.add_argument(
        "--dataset",
        choices=["newsapi", "scrape"],
        default="newsapi",
        help="Choose the dataset to use: newsapi or scrape",
    )
    return parser.parse_args()


# TODO: check ground truth dirs
forecaster_pairs_newsapi: list[ForecasterPair] = [
    ForecasterPair(
        name="BaselineForecaster_p0.4",
        short_name="Baseline",
        ground_truth_dir="src/data/forecasts/BaselineForecaster_p0.4_20240701_20240831",
        eval_dir="src/data/forecasts/BaselineForecaster_p0.4_tuples_newsapi",
    ),
    # ForecasterPair(name="UniformRandomForecaster_n_buckets100", short_name="Uniform", ground_truth_dir="src/data/forecasts/UniformRandomForecaster_n_buckets100_20240701_20240831", eval_dir="src/data/forecasts/UniformRandomForecaster_n_buckets100_tuples_newsapi"),
    ForecasterPair(
        name="BasicForecaster_gpt4o_2024-08-06",
        short_name="GPT-4o-08",
        ground_truth_dir="src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_20240701_20240831",
        eval_dir="src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_tuples_newsapi",
    ),
    ForecasterPair(
        name="BasicForecaster_gpt4o_2024-05-13",
        short_name="GPT-4o-05",
        ground_truth_dir="src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_20240701_20240831",
        eval_dir="src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_tuples_newsapi",
    ),
    ForecasterPair(
        name="BasicForecaster_gpt4o_mini_2024-07-18",
        short_name="GPT-4o-mini",
        ground_truth_dir="src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_20240701_20240831",
        eval_dir="src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_tuples_newsapi",
    ),
    ForecasterPair(
        name="BasicForecaster_claude-3.5-sonnet",
        short_name="Sonnet",
        ground_truth_dir="src/data/forecasts/BasicForecaster_claude-3.5-sonnet_20240701_20240831",
        eval_dir="src/data/forecasts/BasicForecaster_claude-3.5-sonnet_tuples_newsapi",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_o1-mini",
        short_name="CoT-o1-mini",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_20240701_20240831",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_tuples_newsapi",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_o1-preview",
        short_name="CoT-o1-preview",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_20240701_20240831",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_tuples_newsapi",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06",
        short_name="CoT-GPT-4o-08",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_20240701_20240831",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_newsapi",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18",
        short_name="CoT-GPT-4o-mini",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_20240701_20240831",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_newsapi",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet",
        short_name="CoT-Sonnet",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_20240701_20240831",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_newsapi",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_llama-3.1-70B",
        short_name="CoT-L3-70B",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-70B_20240701_20240831",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-70B_tuples_newsapi",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_llama-3.1-405B",
        short_name="CoT-L3-405B",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-405B_20240701_20240831",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-405B_tuples_newsapi",
    ),
    ForecasterPair(
        name="ResolverBasedForecaster_large",
        short_name="Perplexity",
        ground_truth_dir="src/data/forecasts/ResolverBasedForecaster_large_20240701_20240831",
        eval_dir="src/data/forecasts/ResolverBasedForecaster_large_tuples_newsapi",
    ),
]

forecaster_pairs_scrape: list[ForecasterPair] = []


def load_directory_pairs(args: argparse.Namespace) -> List[ForecasterPair]:
    if args.all:
        if not args.include_perplexity:
            if args.dataset == "newsapi":
                return [
                    pair
                    for pair in forecaster_pairs_newsapi
                    if pair["short_name"] != "Perplexity"
                ]
            else:
                return [
                    pair
                    for pair in forecaster_pairs_scrape
                    if pair["short_name"] != "Perplexity"
                ]
        else:
            if args.dataset == "newsapi":
                return forecaster_pairs_newsapi
            else:
                return forecaster_pairs_scrape
    elif args.directories:
        return [
            ForecasterPair(
                name=os.path.basename(gt_dir),
                ground_truth_dir=gt_dir,
                eval_dir=eval_dir,
            )
            for gt_dir, eval_dir in zip(args.directories[::2], args.directories[1::2])
        ]
    elif args.file:
        try:
            with open(args.file, "r") as file:
                lines = [line.strip() for line in file if line.strip()]
                return [
                    ForecasterPair(
                        name=os.path.basename(gt_dir),
                        ground_truth_dir=gt_dir,
                        eval_dir=eval_dir,
                    )
                    for gt_dir, eval_dir in zip(lines[::2], lines[1::2])
                ]
        except FileNotFoundError:
            print(f"Error: File {args.file} not found.")
            exit(1)
    else:
        print("Error: No directories or file provided.")
        exit(1)


def load_json_file(file_path: str) -> Dict:
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        return {}
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except Exception as e:
        print(f"Unexpected error when loading {file_path}: {str(e)}")
        return {}


def extract_metrics(
    forecaster_pair: ForecasterPair,
    gt_metric_key: str,
    cons_metric_type: str,
    cons_metric_key: str,
) -> tuple[str, float, dict[str, float]]:
    ground_truth_path = os.path.join(
        forecaster_pair["ground_truth_dir"], "ground_truth_summary.json"
    )
    stats_path = os.path.join(forecaster_pair["eval_dir"], "stats_summary.json")

    ground_truth_data = load_json_file(ground_truth_path)
    stats_data = load_json_file(stats_path)

    if not ground_truth_data or not stats_data:
        print(
            f"Warning: Missing or empty data for {forecaster_pair['ground_truth_dir']} or {forecaster_pair['eval_dir']}"
        )
        return "", 0.0, {}

    avg_brier_score = ground_truth_data.get(gt_metric_key, 0.0)

    checker_metrics = {}
    if isinstance(stats_data, dict):
        for checker, data in stats_data.items():
            if checker != "aggregated" and isinstance(data, dict):
                overall = data.get("overall", {})
                if isinstance(overall, dict):
                    match cons_metric_type:
                        case "default":
                            metric_dict = overall.get("default", {})
                        case "frequentist":
                            metric_dict = overall.get("frequentist", {})
                        case "default_scaled":
                            metric_dict = overall.get("default_scaled", {})
                        case _:
                            raise ValueError(
                                f"Invalid consistency metric type: {cons_metric_type}"
                            )

                    avg_violation = metric_dict.get(cons_metric_key)
                    if avg_violation is not None:
                        checker_metrics[checker] = avg_violation
    else:
        print(
            f"Warning: stats_data is not a dictionary for {forecaster_pair['eval_dir']}"
        )

    return forecaster_pair["short_name"], avg_brier_score, checker_metrics


def plot_metrics(
    data: list[tuple[str, float, dict[str, float]]],
    output_dir: str,
    gt_metric_key: str,
    cons_metric_type: str,
    cons_metric_key: str,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Identify all checker names
    checker_names = set()
    for _, _, metrics in data:
        checker_names.update(metrics.keys())

    for checker in checker_names:
        x = []
        y = []
        labels = []
        for short_name, brier_score, checker_metrics in data:
            if checker in checker_metrics:
                x.append(brier_score)
                y.append(checker_metrics[checker])
                labels.append(short_name)

        if x and y:
            plt.figure(figsize=(12, 8))
            plt.scatter(x, y)
            for i, label in enumerate(labels):
                plt.annotate(
                    label,
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

            plt.xlabel(f"{gt_metric_key}")
            plt.ylabel(f"{cons_metric_key}")
            plt.title(
                f"{checker}.{cons_metric_type}.{cons_metric_key} vs {gt_metric_key}"
            )
            plt.grid(True)

            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8)

            # Add correlation coefficient
            correlation = np.corrcoef(x, y)[0, 1]
            plt.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.2f}",
                transform=plt.gca().transAxes,
            )

            plot_path = os.path.join(
                output_dir,
                f"{checker}_vs_{gt_metric_key}_{cons_metric_type}_{cons_metric_key}.png",
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Plot saved to {plot_path}")
        else:
            print(f"No data available for checker {checker}.")


def main() -> None:
    args = parse_arguments()
    directory_pairs = load_directory_pairs(args)

    metrics_data = []
    for forecaster_pair in directory_pairs:
        if not os.path.isdir(forecaster_pair["ground_truth_dir"]) or not os.path.isdir(
            forecaster_pair["eval_dir"]
        ):
            print(
                f"Warning: {forecaster_pair['ground_truth_dir']} or {forecaster_pair['eval_dir']} is not a directory. Skipping."
            )
            continue
        short_name, brier_score, checker_metrics = extract_metrics(
            forecaster_pair,
            gt_metric_key=args.gt_metric,
            cons_metric_type=args.cons_metric_type,
            cons_metric_key=args.cons_metric,
        )
        if brier_score != 0.0 and checker_metrics:
            metrics_data.append((short_name, brier_score, checker_metrics))

    if not metrics_data:
        print("No valid metric data found. Exiting.")
        exit(1)

    plot_metrics(
        metrics_data,
        args.output_dir,
        args.gt_metric,
        args.cons_metric_type,
        args.cons_metric,
    )


if __name__ == "__main__":
    main()


# Example command:
# and  not --include_perplexity
# python src/plot_consistency_vs_brier.py --gt_metric avg_brier_score -t frequentist -c avg_violation
