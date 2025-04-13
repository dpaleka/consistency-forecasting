import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from tqdm import tqdm
from adjustText import adjust_text

from forecaster_metrics import (
    ForecasterPair,
    load_dataset_directory_pairs,
    load_json_file,
    get_brier_score_metrics,
    get_consistency_metrics,
    get_consistency_metric_types,
    get_cons_metric_label,
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def rename_cons_metric_type(cons_metric_type: str) -> str:
    if cons_metric_type == "default":
        return "arbitrage"
    elif cons_metric_type == "default_scaled":
        return "arbitrage_scaled"
    else:
        return cons_metric_type


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
        "-o",
        "--output_dir",
        default="src/data/output_plots",
        help="Directory to save plots.",
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
        choices=["newsapi", "scraped", "2028"],
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
    parser.add_argument(
        "--fontsize",
        type=int,
        default=15,
        help="Font size for axes and titles",
    )
    parser.add_argument(
        "--point_fontsize",
        type=int,
        default=None,
        help="Font size for data point labels (if not provided, uses --fontsize value)",
    )
    parser.add_argument(
        "--corr_fontsize",
        type=int,
        default=None,
        help="Font size for correlation text (if not provided, uses --fontsize value)",
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="Do not display title on plots",
    )
    parser.add_argument(
        "--hide_names",
        action="store_true",
        help="Hide forecaster names in the plots",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: hide all text except axis names and increase point sizes",
    )
    parser.add_argument(
        "--axes",
        type=str,
        help='Custom axis names in format "x:name,y:name" (e.g. "x:Error,y:Consistency"). When used with --transpose, axis names are assigned before transposition.',
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose the plot by swapping x and y axes. Applied after --axes, so custom axis names will also be transposed.",
    )
    parser.add_argument(
        "--consistency_csv",
        help="Path to CSV file containing consistency scores in format: short_name,score",
    )
    return parser.parse_args()


def load_directory_pairs(args: argparse.Namespace) -> list[ForecasterPair]:
    if args.all:
        return load_dataset_directory_pairs(
            args.dataset,
            include_perplexity=args.include_perplexity,
            include_baseline=args.include_baseline,
            cfcasters=args.cfcasters,
        )

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


def get_axis_names(
    args: argparse.Namespace,
    gt_metric_key: str,
    cons_metric_key: str,
) -> tuple[str, str]:
    """Get x and y axis names based on command line args or defaults."""
    if args.axes:
        try:
            # Parse the axes string (e.g. "x:Error,y:Consistency")
            axes_dict = dict(pair.split(":") for pair in args.axes.split(","))
            return axes_dict.get("x", gt_metric_key), axes_dict.get(
                "y", cons_metric_key
            )
        except (ValueError, KeyError):
            print("Warning: Invalid axes format. Using default names.")
            return gt_metric_key, cons_metric_key
    return gt_metric_key, cons_metric_key


def plot_metrics(
    data: list[tuple[str, dict[str, float]]],
    output_dir: str,
    dataset_key: str,
    gt_metric_key: str,
    cons_metric_type: str,
    cons_metric_key: str,
    remove_gt_outlier: float = None,
    remove_cons_outlier: float = None,
    fontsize: int | None = None,
    point_fontsize: int | None = None,
    corr_fontsize: int | None = None,
    hide_names: bool = False,
    preview: bool = False,
    axes: str | None = None,
    transpose: bool = False,
    no_title: bool = False,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get axis names based on transpose setting
    if transpose:
        y_label, x_label = get_axis_names(
            argparse.Namespace(axes=axes), gt_metric_key, cons_metric_key
        )
    else:
        x_label, y_label = get_axis_names(
            argparse.Namespace(axes=axes), gt_metric_key, cons_metric_key
        )

    print(f"Fontsize: {fontsize}")
    # Identify all checker names

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
                if transpose:
                    y.append(metrics[gt_metric_key])
                    x.append(metrics[cons_metric_label])
                else:
                    x.append(metrics[gt_metric_key])
                    y.append(metrics[cons_metric_label])
                labels.append(short_name)

        i, j = divmod(idx, 3)
        if x and y:
            # Determine point size based on preview mode
            point_size = 100 if preview else 20
            axs[i, j].scatter(x, y, s=point_size)
            if not hide_names and not preview:
                for idx, label in enumerate(labels):
                    axs[i, j].annotate(
                        label,
                        (x[idx], y[idx]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                    )

            axs[i, j].set_xlabel(x_label)
            axs[i, j].set_ylabel(y_label)
            if not preview:
                title_parts = [
                    checker,
                    cons_metric_type,
                    cons_metric_key,
                    "vs",
                    gt_metric_key,
                ]
                if transpose:
                    # Swap the order in the title when transposed
                    title_parts = [
                        checker,
                        gt_metric_key,
                        "vs",
                        cons_metric_type,
                        cons_metric_key,
                    ]
                axs[i, j].set_title(f"{'.'.join(title_parts)} ({dataset_key})")
            axs[i, j].grid(False)

            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axs[i, j].plot(x, p(x), "r--", alpha=0.8)

            correlation = np.corrcoef(x, y)[0, 1]
            if not preview:
                # Use corr_fontsize if provided, otherwise fall back to fontsize
                correlation_fontsize = (
                    corr_fontsize if corr_fontsize is not None else fontsize
                )
                axs[i, j].text(
                    0.05,
                    0.95,
                    f"Correlation: {correlation:.2f}",
                    transform=axs[i, j].transAxes,
                    fontsize=correlation_fontsize,
                )

            if preview:
                # Remove ticks in preview mode
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

            # Now the mini figure
            plt.figure(figsize=(12, 8))
            plt.scatter(x, y, s=point_size)
            texts = []
            if not hide_names and not preview:
                # Use point_fontsize if provided, otherwise fall back to fontsize
                label_fontsize = (
                    point_fontsize if point_fontsize is not None else fontsize
                )
                for idx, label in enumerate(labels):
                    texts.append(
                        plt.text(x[idx], y[idx], label, fontsize=label_fontsize)
                    )

            plt.plot(x, p(x), "r-", alpha=0.3)
            if not preview:
                # Use corr_fontsize if provided, otherwise fall back to fontsize
                correlation_fontsize = (
                    corr_fontsize if corr_fontsize is not None else fontsize
                )
                plt.text(
                    0.05,
                    0.95,
                    f"Correlation: {correlation:.2f}",
                    transform=plt.gca().transAxes,
                    fontsize=correlation_fontsize,
                )
            plt.xlabel(x_label, fontsize=fontsize)
            plt.ylabel(y_label, fontsize=fontsize)
            if not preview and not no_title:
                title_parts = [
                    checker,
                    cons_metric_type,
                    cons_metric_key,
                    "vs",
                    gt_metric_key,
                ]
                if transpose:
                    # Swap the order in the title when transposed
                    title_parts = [
                        checker,
                        gt_metric_key,
                        "vs",
                        cons_metric_type,
                        cons_metric_key,
                    ]
                title_str = ".".join(title_parts)
                figure_name = f"{title_str}_{dataset_key}.png"
                plt.title(
                    f"{title_str} ({dataset_key})",
                    fontsize=fontsize,
                )
            elif preview:
                figure_name = f"{checker}_vs_{gt_metric_key}_{cons_metric_type}_{cons_metric_key}_{dataset_key}_preview.png"
                plt.xticks([])
                plt.yticks([])
            else:
                # If we have no title but aren't in preview mode
                figure_name = f"{checker}_vs_{gt_metric_key}_{cons_metric_type}_{cons_metric_key}_{dataset_key}.png"
            plt.grid(False)

            # Use adjust_text to prevent overlapping labels if names are shown
            if not hide_names and not preview:
                adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))

            plt.savefig(
                os.path.join(
                    output_dir,
                    figure_name,
                ),
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Plot saved to {os.path.join(output_dir, figure_name)}")
            plt.savefig(
                os.path.join(
                    output_dir,
                    figure_name.replace(".png", ".pdf"),
                ),
                dpi=300,
                bbox_inches="tight",
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
    hide_names: bool = False,
    preview: bool = False,
    axes: str | None = None,
    no_title: bool = False,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _, y_label = get_axis_names(
        argparse.Namespace(axes=axes), "Forecasters", cons_metric_key
    )

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
            figure_name = (
                f"{checker}_bar_{cons_metric_type}_{cons_metric_key}_{dataset_key}.png"
            )
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(values)), values)
            if not hide_names and not preview:
                plt.xticks(range(len(labels)), labels, rotation=0, fontsize=20)
            else:
                plt.xticks([])
            if preview:
                plt.yticks([])
            plt.ylabel(y_label, fontsize=20)  # Keep axis names
            if not preview and not no_title:
                plt.title(
                    f"{checker}.{cons_metric_type}.{cons_metric_key} ({dataset_key})",
                    fontsize=20,
                )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir,
                    figure_name,
                ),
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Bar chart saved to {os.path.join(output_dir, figure_name)}")
            plt.savefig(
                os.path.join(
                    output_dir,
                    figure_name.replace(".png", ".pdf"),
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def plot_gt_bar_chart(
    data: list[tuple[str, dict[str, float]]],
    output_dir: str,
    dataset_key: str,
    gt_metric_key: str,
    remove_gt_outlier: float = None,
    hide_names: bool = False,
    preview: bool = False,
    axes: str | None = None,
    no_title: bool = False,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    _, y_label = get_axis_names(
        argparse.Namespace(axes=axes), "Forecasters", gt_metric_key
    )

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
        plt.bar(range(len(values)), values)
        if not hide_names and not preview:
            plt.xticks(range(len(labels)), labels, rotation=0, fontsize=20)
        else:
            plt.xticks([])
        if preview:
            plt.yticks([])
        plt.ylabel(y_label, fontsize=16)  # Keep axis names
        if not preview and not no_title:
            plt.title(
                f"Ground Truth Metric: {gt_metric_key} ({dataset_key})", fontsize=20
            )
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


def load_consistency_scores(csv_path: str) -> dict[str, float]:
    """Load consistency scores from a CSV file."""
    import csv

    scores = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            short_name, score = row
            try:
                scores[short_name] = float(score)
            except ValueError:
                continue
    return scores


def extract_all_metrics(
    forecaster_pair: ForecasterPair, csv_scores: dict[str, float] | None = None
) -> dict[str, float] | None:
    """Extract all metrics for a forecaster pair, optionally using CSV scores for consistency."""
    # If we're using CSV scores and this forecaster isn't in them, skip it
    if csv_scores is not None and forecaster_pair["short_name"] not in csv_scores:
        return None

    ground_truth_path = (
        os.path.join(forecaster_pair["ground_truth_dir"], "ground_truth_summary.json")
        if forecaster_pair["ground_truth_dir"] is not None
        else None
    )

    stats_path = os.path.join(forecaster_pair["eval_dir"], "stats_summary.json")

    # Always load ground truth data for Brier scores
    ground_truth_data = load_json_file(ground_truth_path)

    metrics = {}

    # Add ground truth metrics
    for brier_metric in get_brier_score_metrics():
        metrics[brier_metric] = (
            ground_truth_data.get(brier_metric, 0.0)
            if ground_truth_data is not None
            else None
        )

    # If we have CSV scores, use those scores
    if csv_scores is not None:
        score = csv_scores[forecaster_pair["short_name"]]
        for checker in checker_names:
            for metric_type in get_consistency_metric_types():
                for metric in get_consistency_metrics():
                    key = get_cons_metric_label(checker, metric_type, metric)
                    metrics[key] = score
        return metrics

    # Otherwise use the original extraction logic for consistency metrics
    stats_data = load_json_file(stats_path)
    if not stats_data:
        print(f"Warning: Missing or empty data for {forecaster_pair['eval_dir']}")
        return metrics if metrics else None

    if isinstance(stats_data, dict):
        for checker, data in stats_data.items():
            if checker in ["forecaster", "full_forecaster_config"]:
                continue
            elif checker == "aggregated":
                overall = data
            elif isinstance(data, dict):
                overall = data.get("overall", {})

            for metric_type in get_consistency_metric_types():
                metric_dict = overall.get(metric_type, {})
                for cons_metric in get_consistency_metrics():
                    if cons_metric == "frac_violations":
                        value = (
                            metric_dict.get("num_violations", 0)
                            / metric_dict.get("num_samples", 1)
                            if metric_dict.get("num_samples")
                            else None
                        )
                    else:
                        value = metric_dict.get(cons_metric)
                    if value is not None:
                        label = get_cons_metric_label(checker, metric_type, cons_metric)
                        metrics[label] = value

    return metrics


def main() -> None:
    args = parse_arguments()
    directory_pairs = load_directory_pairs(args)

    # Print all forecaster names we're looking for
    print("\nLooking for scores for these forecasters:")
    for pair in directory_pairs:
        print(f"  {pair['short_name']}")
    print()

    # Load CSV scores if provided
    csv_scores = None
    if args.consistency_csv:
        csv_scores = load_consistency_scores(args.consistency_csv)
        if not csv_scores:
            print("Warning: No valid scores found in CSV file")
        else:
            # Print which forecasters are missing from CSV
            missing_forecasters = [
                pair["short_name"]
                for pair in directory_pairs
                if pair["short_name"] not in csv_scores
            ]
            if missing_forecasters:
                print(
                    "Warning: The following forecasters are missing from the CSV file:"
                )
                for name in missing_forecasters:
                    print(f"  {name}")
                print()

    metrics_data = []
    for forecaster_pair in directory_pairs:
        if not os.path.isdir(forecaster_pair["ground_truth_dir"]) or not os.path.isdir(
            forecaster_pair["eval_dir"]
        ):
            print(
                f"Warning: {forecaster_pair['ground_truth_dir']} or {forecaster_pair['eval_dir']} is not a directory. Skipping."
            )
            continue
        metrics = extract_all_metrics(forecaster_pair, csv_scores)
        if metrics:
            metrics_data.append((forecaster_pair["short_name"], metrics))

    if not metrics_data:
        print("No valid metric data found. Exiting.")
        exit(1)

    # Print which forecasters we're actually using
    print(f"\nUsing data for {len(metrics_data)} forecasters:")
    for name, _ in metrics_data:
        print(f"  {name}")
    print()

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
            fontsize=args.fontsize,
            point_fontsize=args.point_fontsize,
            corr_fontsize=args.corr_fontsize,
            hide_names=args.hide_names,
            preview=args.preview,
            axes=args.axes,
            transpose=args.transpose,
            no_title=args.no_title,
        )
    elif args.plot_type == "bar":
        plot_bar_chart(
            metrics_data,
            args.output_dir,
            dataset_key=args.dataset,
            cons_metric_type=args.cons_metric_type,
            cons_metric_key=args.cons_metric,
            remove_cons_outlier=args.remove_cons_outlier,
            hide_names=args.hide_names,
            preview=args.preview,
            axes=args.axes,
            no_title=args.no_title,
        )
    elif args.plot_type == "gt_bar":
        plot_gt_bar_chart(
            metrics_data,
            args.output_dir,
            dataset_key=args.dataset,
            gt_metric_key=args.gt_metric,
            remove_gt_outlier=args.remove_gt_outlier,
            hide_names=args.hide_names,
            preview=args.preview,
            axes=args.axes,
            no_title=args.no_title,
        )


if __name__ == "__main__":
    main()


# Example commands:
# python src/plot_consistency_vs_brier.py --all --dataset newsapi --gt_metric avg_platt_brier_score -t frequentist -c avg_violation --remove_gt_outlier 0.25
# python src/plot_consistency_vs_brier.py --all --dataset scraped --gt_metric avg_brier_score -t default -c avg_violation --point_fontsize 12 --no_title
# python src/plot_consistency_vs_brier.py --all --dataset scraped --gt_metric avg_brier_score -t default -c avg_violation --fontsize 14 --point_fontsize 10 --corr_fontsize 20
# python src/plot_consistency_vs_brier.py --all --dataset scraped --gt_metric avg_brier_score -t default -c avg_violation --remove_gt_outlier 0.25 --axes "x:Brier score,y:Consistency violation" --transpose --fontsize 16 --point_fontsize 11 --corr_fontsize 13 --no_title
