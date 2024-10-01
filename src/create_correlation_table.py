import argparse
import os
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from forecaster_metrics import (
    get_forecaster_pairs,
    get_brier_score_metrics,
    extract_all_metrics,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a correlation table between consistency checks and ground truth metrics."
    )
    parser.add_argument(
        "--dataset",
        choices=["newsapi", "scraped"],
        default="newsapi",
        help="Choose the dataset to use: newsapi or scraped",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="src/data/output_tables",
        help="Directory to save the correlation table.",
    )
    return parser.parse_args()


def create_correlation_table(data: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(data)

    brier_metrics = get_brier_score_metrics()
    consistency_metrics = [col for col in df.columns if col not in brier_metrics]

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
        if "frequentist" in index and "avg_violation" in index:
            metric_name = index.replace("frequentist.", "").replace("avg_violation", "")
            correlation = row["avg_brier_score"]
            latex_table += f"{metric_name} & {correlation:.2f} \\\\"

    latex_table += r"\hline"
    latex_table += r"\end{tabular}"

    return latex_table


def main() -> None:
    args = parse_arguments()
    forecaster_pairs = get_forecaster_pairs(args.dataset)

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

    correlation_table = create_correlation_table(all_metrics_data)

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
