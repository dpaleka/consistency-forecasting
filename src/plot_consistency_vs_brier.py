import argparse
import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

from forecaster_metrics import (
    ForecasterPair,
    get_forecaster_pairs,
    extract_all_metrics,
    get_cons_metric_label,
)


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
        "--include_baseline",
        action="store_true",
        help="Include baseline in the plots",
    )
    parser.add_argument(
        "--cfcasters",
        nargs="*",
        help="N, P, NP, EE, O, others",
    )
    parser.add_argument(
        "-m",
        "--gt_metric",
        choices=[
            "avg_brier_score",
            "avg_platt_brier_score",
            "avg_brier_score_scaled",
            "avg_platt_brier_score_scaled",
            "avg_log_score",
            "calibration_error",
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
        choices=[
            "avg_violation",
            "avg_violation_no_outliers",
            "median_violation",
            "frac_violations",
        ],
        default="avg_violation",
        help="Consistency metric to use",
    )
    parser.add_argument(
        "--remove_gt_outlier",
        type=float,
        default=None,
        help="Remove ground truth outlier from the plots, specify the outlier threshold in the ground truth metric.",
    )
    parser.add_argument(
        "--remove_cons_outlier",
        type=float,
        default=None,
        help="Remove consistency metric outlier from the plots, specify the outlier threshold in the consistency metric.",
    )
    parser.add_argument(
        "--dataset",
        choices=["newsapi", "scraped"],
        default="newsapi",
        help="Choose the dataset to use: newsapi or scrape",
    )
    parser.add_argument(
        "-p",
        "--plot_type",
        choices=["correlation", "bar", "gt_bar"],
        default="correlation",
        help="Type of plot to generate: correlation, bar chart of consistency violations, or bar chart of ground truth.",
    )
    return parser.parse_args()


def match_cfcaster_names(short_name: str, cfcasters: list[str]) -> bool:
    if not cfcasters:
        return not short_name.startswith("CF-")
    if short_name in cfcasters:
        return True
    if short_name.startswith("CF-") and short_name[3:] in cfcasters:
        return True
    if (
        "N" in cfcasters
        and short_name.startswith("CF-N")
        and not short_name.startswith("CF-NP")
    ):
        return True
    if "P" in cfcasters and short_name.startswith("CF-P"):
        return True
    if "NP" in cfcasters and short_name.startswith("CF-NP"):
        return True
    if "EE" in cfcasters and short_name.startswith("CF-") and "EE" in short_name:
        return True
    if "O" in cfcasters and short_name == "Basic-GPT-4o-mini":
        return True
    if "allcfs" in cfcasters and short_name.startswith("CF-"):
        return True
    if "others" in cfcasters and not short_name.startswith("CF-"):
        return True
    if "all" in cfcasters:
        return True
    return False


def load_directory_pairs(args: argparse.Namespace) -> List[ForecasterPair]:
    if args.all:
        forecaster_pairs = get_forecaster_pairs(args.dataset)
        if not args.include_perplexity:
            forecaster_pairs = [
                pair for pair in forecaster_pairs if pair["short_name"] != "Perplexity"
            ]
        if not args.include_baseline:
            forecaster_pairs = [
                pair for pair in forecaster_pairs if pair["short_name"] != "Baseline"
            ]
        forecaster_pairs = [
            pair
            for pair in forecaster_pairs
            if match_cfcaster_names(pair["short_name"], args.cfcasters)
        ]
        return forecaster_pairs

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


def plot_metrics(
    data: list[tuple[str, dict[str, float]]],
    output_dir: str,
    dataset_key: str,
    gt_metric_key: str,
    cons_metric_type: str,
    cons_metric_key: str,
    remove_gt_outlier: float = None,
    remove_cons_outlier: float = None,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Identify all checker names
    checker_names = [
        "NegChecker",
        "ParaphraseChecker",
        "CondCondChecker",
        "ExpectedEvidenceChecker",
        "ConsequenceChecker",
        "AndChecker",
        "OrChecker",
        "AndOrChecker",
        "ButChecker",
        "CondChecker",
        "aggregated",
    ]

    fig, axs = plt.subplots(
        nrows=int(math.ceil(len(checker_names) / 3)),
        ncols=3,
        figsize=(30, 25),
        layout="constrained",
    )
    if len(checker_names) == 1:
        axs = [axs]  # Ensure axs is a list even if there's only one subplot

    for idx, checker in tqdm(enumerate(checker_names), desc="Plotting checkers"):
        x = []
        y = []
        labels = []
        for short_name, metrics in data:
            cons_metric_label = get_cons_metric_label(
                checker, cons_metric_type, cons_metric_key
            )
            if cons_metric_label in metrics:
                if remove_gt_outlier is not None:
                    if metrics[gt_metric_key] > remove_gt_outlier:
                        continue
                if remove_cons_outlier is not None:
                    if metrics[cons_metric_label] > remove_cons_outlier:
                        continue
                x.append(metrics[gt_metric_key])
                y.append(metrics[cons_metric_label])
                labels.append(short_name)

        i, j = divmod(idx, 3)
        if x and y:
            axs[i, j].scatter(x, y)
            for idx, label in enumerate(labels):
                axs[i, j].annotate(
                    label,
                    (x[idx], y[idx]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

            axs[i, j].set_xlabel(f"{gt_metric_key}")
            axs[i, j].set_ylabel(f"{cons_metric_key}")
            axs[i, j].set_title(
                f"{checker}.{cons_metric_type}.{cons_metric_key} vs {gt_metric_key} ({dataset_key})"
            )
            axs[i, j].grid(False)

            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axs[i, j].plot(x, p(x), "r--", alpha=0.8)

            correlation = np.corrcoef(x, y)[0, 1]
            axs[i, j].text(
                0.05,
                0.95,
                f"Correlation: {correlation:.2f}",
                transform=axs[i, j].transAxes,
            )

            # Now the mini figure
            plt.figure(figsize=(12, 8))
            plt.scatter(x, y)
            for idx, label in enumerate(labels):
                plt.annotate(
                    label,
                    (x[idx], y[idx]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

            plt.plot(x, p(x), "r-", alpha=0.3)
            plt.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.2f}",
                transform=plt.gca().transAxes,
            )
            plt.xlabel(f"{gt_metric_key}")
            plt.ylabel(f"{cons_metric_key}")
            plt.title(
                f"{checker}.{cons_metric_type}.{cons_metric_key} vs {gt_metric_key} ({dataset_key})"
            )
            plt.grid(False)
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{checker}_vs_{gt_metric_key}_{cons_metric_type}_{cons_metric_key}_{dataset_key}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Plot saved to {os.path.join(output_dir, f'{checker}_vs_{gt_metric_key}_{cons_metric_type}_{cons_metric_key}_{dataset_key}.png')}"
            )
            plt.close()

    plt.savefig(
        os.path.join(
            output_dir,
            f"all_checkers_vs_{gt_metric_key}_{cons_metric_type}_{cons_metric_key}_{dataset_key}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Plot saved to {os.path.join(output_dir, f'all_checkers_vs_{gt_metric_key}_{cons_metric_type}_{cons_metric_key}_{dataset_key}.png')}\n"
    )
    exit(0)


def plot_bar_chart(
    data: list[tuple[str, dict[str, float]]],
    output_dir: str,
    dataset_key: str,
    cons_metric_type: str,
    cons_metric_key: str,
    remove_cons_outlier: float = None,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checker_names = [
        "NegChecker",
        "ParaphraseChecker",
        "CondCondChecker",
        "ExpectedEvidenceChecker",
        "ConsequenceChecker",
        "AndChecker",
        "OrChecker",
        "AndOrChecker",
        "ButChecker",
        "CondChecker",
        "aggregated",
    ]

    for checker in tqdm(checker_names, desc="Plotting bar charts"):
        labels = []
        values = []
        for short_name, metrics in data:
            cons_metric_label = get_cons_metric_label(
                checker, cons_metric_type, cons_metric_key
            )
            if cons_metric_label in metrics:
                if remove_cons_outlier is not None:
                    if metrics[cons_metric_label] > remove_cons_outlier:
                        continue
                labels.append(short_name)
                values.append(metrics[cons_metric_label])

        if labels and values:
            plt.figure(figsize=(12, 8))
            plt.bar(labels, values)
            plt.xlabel("Forecasters")
            plt.ylabel(f"{cons_metric_key}")
            plt.title(f"{checker}.{cons_metric_type}.{cons_metric_key} ({dataset_key})")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{checker}_bar_{cons_metric_type}_{cons_metric_key}_{dataset_key}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Bar chart saved to {os.path.join(output_dir, f'{checker}_bar_{cons_metric_type}_{cons_metric_key}_{dataset_key}.png')}"
            )
            plt.close()


def plot_gt_bar_chart(
    data: list[tuple[str, dict[str, float]]],
    output_dir: str,
    dataset_key: str,
    gt_metric_key: str,
    remove_gt_outlier: float = None,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labels = []
    values = []
    for short_name, metrics in data:
        if gt_metric_key in metrics:
            if remove_gt_outlier is not None:
                if metrics[gt_metric_key] > remove_gt_outlier:
                    continue
            labels.append(short_name)
            values.append(metrics[gt_metric_key])

    if labels and values:
        plt.figure(figsize=(12, 8))
        plt.bar(labels, values)
        plt.xlabel("Forecasters")
        plt.ylabel(f"{gt_metric_key}")
        plt.title(f"Ground Truth Metric: {gt_metric_key} ({dataset_key})")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"gt_bar_{gt_metric_key}_{dataset_key}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Ground truth bar chart saved to {os.path.join(output_dir, f'gt_bar_{gt_metric_key}_{dataset_key}.png')}"
        )
        plt.close()


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
        metrics = extract_all_metrics(forecaster_pair)
        if metrics:
            metrics_data.append((forecaster_pair["short_name"], metrics))

    if not metrics_data:
        print("No valid metric data found. Exiting.")
        exit(1)

    if args.plot_type == "correlation":
        plot_metrics(
            metrics_data,
            args.output_dir,
            dataset_key=args.dataset,
            gt_metric_key=args.gt_metric,
            cons_metric_type=args.cons_metric_type,
            cons_metric_key=args.cons_metric,
            remove_gt_outlier=args.remove_gt_outlier,
            remove_cons_outlier=args.remove_cons_outlier,
        )
    elif args.plot_type == "bar":
        plot_bar_chart(
            metrics_data,
            args.output_dir,
            dataset_key=args.dataset,
            cons_metric_type=args.cons_metric_type,
            cons_metric_key=args.cons_metric,
            remove_cons_outlier=args.remove_cons_outlier,
        )
    elif args.plot_type == "gt_bar":
        plot_gt_bar_chart(
            metrics_data,
            args.output_dir,
            dataset_key=args.dataset,
            gt_metric_key=args.gt_metric,
            remove_gt_outlier=args.remove_gt_outlier,
        )


if __name__ == "__main__":
    main()


# Example command:
# python src/plot_consistency_vs_brier.py --all --dataset newsapi --gt_metric avg_platt_brier_score -t frequentist -c avg_violation --remove_gt_outlier 0.25
