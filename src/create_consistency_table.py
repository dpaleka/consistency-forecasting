import argparse
import os
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import math

from forecaster_metrics import (
    load_dataset_directory_pairs,
    extract_all_metrics,
    get_cons_metric_label,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a consistency table with checks as rows and forecasters as columns."
    )
    parser.add_argument(
        "--dataset",
        choices=["newsapi", "scraped"],
        default="newsapi",
        help="Choose the dataset to use: newsapi or scraped",
    )
    parser.add_argument(
        "-f",
        "--forecasters",
        nargs="+",
        help="List of forecasters short names to include in the table.",
    )
    parser.add_argument(
        "-t",
        "--cons_metric_types",
        nargs="+",
        choices=["default", "frequentist", "default_scaled"],
        default=["default"],
        help="Consistency metric type to use",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="src/data/output_tables",
        help="Directory to save the consistency table.",
    )
    parser.add_argument(
        "--max_columns",
        type=int,
        default=8,
        help="Maximum number of columns (forecasters) per table",
    )
    parser.add_argument(
        "--cfcasters",
        nargs="*",
        help="N, P, NP, EE, O, others",
    )
    parser.add_argument(
        "-g",
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
    return parser.parse_args()


def create_consistency_table(
    data: List[Tuple[str, Dict[str, float]]],
    max_columns: int,
    cons_metric_types: List[str],
    gt_metric: str,
) -> List[pd.DataFrame]:
    forecasters = [item[0] for item in data]

    checks = [
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

    num_tables = math.ceil(len(forecasters) / max_columns)
    full_table = []
    tables = []

    if len(cons_metric_types) == 1:
        cons_metric_type = cons_metric_types[0]
        for i in range(num_tables):
            start_idx = i * max_columns
            end_idx = min((i + 1) * max_columns, len(forecasters))
            current_forecasters = forecasters[start_idx:end_idx]

            table_data = []
            for check in checks:
                row = [check]
                for forecaster in current_forecasters:
                    metrics = next(item[1] for item in data if item[0] == forecaster)
                    metric_label = get_cons_metric_label(
                        check, cons_metric_type, "avg_violation"
                    )
                    row.append(metrics.get(metric_label, float("nan")))
                    frac_violations_label = get_cons_metric_label(
                        check, cons_metric_type, "frac_violations"
                    )
                    frac_violations = metrics.get(frac_violations_label, float("nan"))
                    if not math.isnan(frac_violations):
                        percent_violations = int(frac_violations * 100)
                        row.append(f"{percent_violations}%")
                    else:
                        row.append("")

                table_data.append(row)

            # Add ground truth row
            ground_truth_row = ["Ground Truth"]
            for forecaster in current_forecasters:
                metrics = next(item[1] for item in data if item[0] == forecaster)
                ground_truth_row.append(metrics.get(gt_metric, float("nan")))
                ground_truth_row.append("")  # No percentage for ground truth

            table_data.append(ground_truth_row)

            df = pd.DataFrame(
                table_data,
                columns=["Check"]
                + [
                    f"{forecaster}_{cons_metric_type}_avg_violation"
                    for forecaster in current_forecasters
                ]
                + [
                    f"{forecaster}_{cons_metric_type}_frac_violations"
                    for forecaster in current_forecasters
                ],
            )
            tables.append(df)
    else:
        print(cons_metric_types)
        raise ValueError("Not implemented")
    return tables


def save_table_as_csv(df: pd.DataFrame, output_file: str) -> None:
    df.to_csv(output_file, index=False)
    print(f"CSV table saved to {output_file}")


def save_table_as_latex(df: pd.DataFrame, output_file: str) -> None:
    latex_table = df.to_latex(index=False, float_format="%.4f")
    with open(output_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {output_file}")


def main() -> None:
    args = parse_arguments()
    forecaster_pairs = load_dataset_directory_pairs(
        args.dataset,
        include_perplexity=False,
        include_baseline=False,
        cfcasters=args.cfcasters,
    )
    if args.forecasters:
        forecaster_pairs = [
            pair for pair in forecaster_pairs if pair["short_name"] in args.forecasters
        ]
    print(forecaster_pairs)

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
            all_metrics_data.append((forecaster_pair["short_name"], metrics))

    if not all_metrics_data:
        print("No valid metric data found. Exiting.")
        exit(1)

    consistency_tables = create_consistency_table(
        all_metrics_data, args.max_columns, args.cons_metric_types, args.gt_metric
    )
    joint_table = pd.concat(consistency_tables, ignore_index=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    csv_output_file = os.path.join(
        args.output_dir, f"consistency_table_{args.dataset}.csv"
    )
    save_table_as_csv(joint_table, csv_output_file)

    for i, table in enumerate(consistency_tables):
        latex_output_file = os.path.join(
            args.output_dir,
            f"consistency_table_{args.dataset}_{args.cons_metric_types[0]}_{i+1}.tex",
        )
        save_table_as_latex(table, latex_output_file)


if __name__ == "__main__":
    main()

# Example command:
# python src/create_consistency_table.py --dataset newsapi --max_columns 8 --cons_metric_type default frequentist default_scaled
