import argparse
import os
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from forecaster_metrics import (
    load_dataset_directory_pairs,
    get_brier_score_metrics,
    extract_all_metrics,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a correlation table between consistency checks and ground truth metrics."
    )
    parser.add_argument(
        "--dataset",
        choices=["newsapi", "scraped", "2028"],
        default="newsapi",
        help="Choose the dataset to use: newsapi or scraped",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="src/data/output_tables",
        help="Directory to save the correlation table.",
    )
    parser.add_argument(
        "--include_perplexity",
        action="store_true",
        help="Include perplexity in the analysis.",
    )
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        help="Include baseline in the analysis.",
    )
    parser.add_argument(
        "--cfcasters",
        nargs="+",
        choices=["all", "others", "cf-all", "cf-others"],
        help="Choose the cfcasters to include in the analysis.",
    )
    parser.add_argument(
        "--remove_gt_outlier",
        type=float,
        default=None,
        help="Remove ground truth outlier from the plots, specify the outlier threshold in the ground truth metric.",
    )
    return parser.parse_args()


def create_correlation_table(
    data: List[Dict[str, float]], remove_gt_outlier: float = None
) -> pd.DataFrame:
    df = pd.DataFrame(data)

    brier_metrics = get_brier_score_metrics()
    consistency_metrics = [col for col in df.columns if col not in brier_metrics]

    if remove_gt_outlier is not None:
        print(f"Removing ground truth outliers greater than {remove_gt_outlier}")
        print(df)
        outlier_indices = df["avg_brier_score"] > remove_gt_outlier
        for index in outlier_indices:
            print(f"Removing {index}")
        df = df[df["avg_brier_score"] <= remove_gt_outlier]

    correlation_table = pd.DataFrame(index=consistency_metrics, columns=brier_metrics)

    for cons_metric in consistency_metrics:
        for brier_metric in brier_metrics:
            correlation = df[cons_metric].corr(df[brier_metric])
            correlation_table.loc[cons_metric, brier_metric] = round(correlation, 2)

    return correlation_table


def make_latex_table(df: pd.DataFrame) -> str:
    latex_table = r"\begin{tabular}{lc}"
    latex_table += r"\hline"
    latex_table += (
        r"\textbf{Consistency Metric} & \textbf{Correlation with Brier Score} \\"
    )
    latex_table += r"\hline"

    for index, row in df.iterrows():
        print(index)
        if "avg_violation" in index and "no_outliers" not in index:
            if "default_scaled" in index:
                continue
            metric_name = index.replace("avg_violation", "")
            correlation = row["avg_brier_score"]
            latex_table += f"{metric_name} & {correlation:.2f} \\\\"

    latex_table += r"\hline"
    latex_table += r"\end{tabular}"

    return latex_table


def main() -> None:
    args = parse_arguments()
    forecaster_pairs = load_dataset_directory_pairs(
        args.dataset, include_perplexity=False, include_baseline=False, cfcasters=[]
    )

    all_metrics_data = []
    for forecaster_pair in tqdm(forecaster_pairs, desc="Extracting metrics"):
        if not os.path.isdir(forecaster_pair["ground_truth_dir"]) or not os.path.isdir(
            forecaster_pair["eval_dir"]
        ):
            print(
                f"Warning: {forecaster_pair['ground_truth_dir']} or {forecaster_pair['eval_dir']} is not a directory. Skipping."
            )
            continue
        metrics = extract_all_metrics(forecaster_pair)
        if metrics:
            all_metrics_data.append(metrics)

    if not all_metrics_data:
        print("No valid metric data found. Exiting.")
        exit(1)

    correlation_table = create_correlation_table(
        all_metrics_data, remove_gt_outlier=args.remove_gt_outlier
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save the correlation table to a CSV file
    output_file = os.path.join(args.output_dir, f"correlation_table_{args.dataset}.csv")
    correlation_table.to_csv(output_file)
    print(f"Correlation table saved to {output_file}")

    # Display the correlation table
    print("\nCorrelation Table:")
    # print(correlation_table.to_string(index=True, header=True, na_rep='N/A'))

    latex_table = make_latex_table(correlation_table)
    print("\nLaTeX Table:")
    print(latex_table)


if __name__ == "__main__":
    main()

# Example command:
# python src/create_correlation_table.py --dataset newsapi
