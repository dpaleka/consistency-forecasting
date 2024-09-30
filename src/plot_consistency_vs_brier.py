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
        "--only_cfcaster",
        action="store_true",
        default=False,
        help="Only include consistent forecasters in the plots",
    )
    parser.add_argument(
        "--include_cfcaster",
        action="store_true",
        default=False,
        help="Include consistent forecasters in the plots",
    )
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        help="Include baseline in the plots",
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
        name="CoT_ForecasterTextBeforeParsing_llama-3.1-8B",
        short_name="CoT-L3-8B",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-8B_20240701_20240831",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-8B_tuples_newsapi",
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

forecaster_pairs_scraped: list[ForecasterPair] = [
    ForecasterPair(
        name="BaselineForecaster_p0.4",
        short_name="Baseline",
        ground_truth_dir="src/data/forecasts/BaselineForecaster_09-23-13-41",
        eval_dir="src/data/forecasts/BaselineForecaster_p0.4_tuples_scraped",
    ),
    ForecasterPair(
        name="ResolverBasedForecaster_large",
        short_name="Perplexity",
        ground_truth_dir="src/data/forecasts/ResolverBasedForecaster_09-23-21-55",
        eval_dir="src/data/forecasts/ResolverBasedForecaster_large_tuples_scraped",
    ),
    ForecasterPair(
        name="BasicForecaster_gpt4o_2024-08-06",
        short_name="Basic-GPT-4o-08",
        ground_truth_dir="src/data/forecasts/BasicForecaster_09-23-13-46",
        eval_dir="src/data/forecasts/BasicForecaster_gpt4o_2024-08-06_tuples_scraped",
    ),
    ForecasterPair(
        name="BasicForecaster_gpt4o_2024-05-13",
        short_name="Basic-GPT-4o-05",
        ground_truth_dir="src/data/forecasts/BasicForecaster_09-24-23-30",
        eval_dir="src/data/forecasts/BasicForecaster_gpt4o_2024-05-13_tuples_scraped",
    ),
    ForecasterPair(
        name="BasicForecaster_gpt4o_mini_2024-07-18",
        short_name="Basic-GPT-4o-mini",
        ground_truth_dir="src/data/forecasts/BasicForecaster_09-24-19-10",
        eval_dir="src/data/forecasts/BasicForecaster_gpt4o_mini_2024-07-18_tuples_scraped",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06",
        short_name="CoT-GPT-4o-08",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-30",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_2024-08-06_tuples_scraped",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18",
        short_name="CoT-GPT-4o-mini",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-44",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_gpt4o_mini_2024-07-18_tuples_scraped",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_o1-mini",
        short_name="CoT-o1-mini",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-23-22-25",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-mini_tuples_scraped",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_o1-preview",
        short_name="CoT-o1-preview",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-19-12",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_o1-preview_tuples_scraped",
    ),
    ForecasterPair(
        name="BasicForecaster_claude-3.5-sonnet",
        short_name="Basic-Sonnet",
        ground_truth_dir="src/data/forecasts/BasicForecaster_09-24-19-09",
        eval_dir="src/data/forecasts/BasicForecaster_claude-3.5-sonnet_tuples_scraped",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet",
        short_name="CoT-Sonnet",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-22-42",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_claude-3.5-sonnet_tuples_scraped",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_llama-3.1-8B",
        short_name="CoT-L3-8B",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-36",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-8B_tuples_scraped",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_llama-3.1-70B",
        short_name="CoT-L3-70B",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-09",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-70B_tuples_scraped",
    ),
    ForecasterPair(
        name="CoT_ForecasterTextBeforeParsing_llama-3.1-405B",
        short_name="CoT-L3-405B",
        ground_truth_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_09-24-23-25",
        eval_dir="src/data/forecasts/CoT_ForecasterTextBeforeParsing_llama-3.1-405B_tuples_scraped",
    ),
    ForecasterPair(
        name="BasicForecaster_llama-3.1-8B",
        short_name="Basic-L3-8B",
        ground_truth_dir="src/data/forecasts/BasicForecaster_09-24-19-12",
        eval_dir="src/data/forecasts/BasicForecaster_llama-3.1-8B_tuples_scraped",
    ),
    ForecasterPair(
        name="BasicForecaster_llama-3.1-70B",
        short_name="Basic-L3-70B",
        ground_truth_dir="src/data/forecasts/BasicForecaster_09-24-19-29",
        eval_dir="src/data/forecasts/BasicForecaster_llama-3.1-70B_tuples_scraped",
    ),
    ForecasterPair(
        name="BasicForecaster_llama-3.1-405B",
        short_name="Basic-L3-405B",
        ground_truth_dir="src/data/forecasts/BasicForecaster_09-24-22-40",
        eval_dir="src/data/forecasts/BasicForecaster_llama-3.1-405B_tuples_scraped",
    ),
]

consistentforecaster_pairs_newsapi: list[ForecasterPair] = [
    ForecasterPair(
        name="ConsistentForecaster_4xEE1",
        short_name="CF-4xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4",
        short_name="CF-N4",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_20240701_20240831",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4",
        short_name="CF-P4",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_20240701_20240831",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4",
        short_name="CF-NP4",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi",
    ),
    # Intermediate forecasters
    ForecasterPair(
        name="ConsistentForecaster_4xEE1_3x",
        short_name="CF-3xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_3x",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_3x",
    ),
    ForecasterPair(
        name="ConsistentForecaster_4xEE1_2x",
        short_name="CF-2xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_2x",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_2x",
    ),
    ForecasterPair(
        name="ConsistentForecaster_4xEE1_1x",
        short_name="CF-1xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_1x",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_1x",
    ),
    ForecasterPair(
        name="ConsistentForecaster_4xEE1_0x",
        short_name="CF-0xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_20240701_20240831_0x",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_newsapi_0x",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4_3",
        short_name="CF-N3",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_3",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_3",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4_2",
        short_name="CF-N2",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_2",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_2",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4_1",
        short_name="CF-N1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_1",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_1",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4_0",
        short_name="CF-N0",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_20240701_20240831_0",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_newsapi_0",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4_3",
        short_name="CF-P3",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_3",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_3",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4_2",
        short_name="CF-P2",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_2",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_2",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4_1",
        short_name="CF-P1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_1",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_1",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4_0",
        short_name="CF-P0",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_20240701_20240831_0",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_newsapi_0",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4_3",
        short_name="CF-NP3",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_3",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_3",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4_2",
        short_name="CF-NP2",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_2",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_2",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4_1",
        short_name="CF-NP1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_1",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_1",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4_0",
        short_name="CF-NP0",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_20240701_20240831_0",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_newsapi_0",
    ),
]

consistentforecaster_pairs_scraped: list[ForecasterPair] = [
    ForecasterPair(
        name="ConsistentForecaster_4xEE1",
        short_name="CF-4xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_scraped",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4",
        short_name="CF-N4",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_scraped",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4",
        short_name="CF-P4",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_scraped",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4",
        short_name="CF-NP4",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_scraped",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped",
    ),
    # Intermediate forecasters
    ForecasterPair(
        name="ConsistentForecaster_4xEE1_3x",
        short_name="CF-3xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_scraped_3x",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_3x",
    ),
    ForecasterPair(
        name="ConsistentForecaster_4xEE1_2x",
        short_name="CF-2xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_scraped_2x",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_2x",
    ),
    ForecasterPair(
        name="ConsistentForecaster_4xEE1_1x",
        short_name="CF-1xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_scraped_1x",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_1x",
    ),
    ForecasterPair(
        name="ConsistentForecaster_4xEE1_0x",
        short_name="CF-0xEE1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_4xEE1_scraped_0x",
        eval_dir="src/data/forecasts/ConsistentForecaster_4xEE1_tuples_scraped_0x",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4_3",
        short_name="CF-N3",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_scraped_3",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_3",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4_2",
        short_name="CF-N2",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_scraped_2",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_2",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4_1",
        short_name="CF-N1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_scraped_1",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_1",
    ),
    ForecasterPair(
        name="ConsistentForecaster_N4_0",
        short_name="CF-N0",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_N4_scraped_0",
        eval_dir="src/data/forecasts/ConsistentForecaster_N4_tuples_scraped_0",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4_3",
        short_name="CF-P3",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_scraped_3",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_3",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4_2",
        short_name="CF-P2",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_scraped_2",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_2",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4_1",
        short_name="CF-P1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_scraped_1",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_1",
    ),
    ForecasterPair(
        name="ConsistentForecaster_P4_0",
        short_name="CF-P0",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_P4_scraped_0",
        eval_dir="src/data/forecasts/ConsistentForecaster_P4_tuples_scraped_0",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4_3",
        short_name="CF-NP3",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_scraped_3",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_3",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4_2",
        short_name="CF-NP2",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_scraped_2",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_2",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4_1",
        short_name="CF-NP1",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_scraped_1",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_1",
    ),
    ForecasterPair(
        name="ConsistentForecaster_NP4_0",
        short_name="CF-NP0",
        ground_truth_dir="src/data/forecasts/ConsistentForecaster_NP4_scraped_0",
        eval_dir="src/data/forecasts/ConsistentForecaster_NP4_tuples_scraped_0",
    ),
]


def load_directory_pairs(args: argparse.Namespace) -> List[ForecasterPair]:
    if args.all:
        if args.only_cfcaster:
            match args.dataset:
                case "newsapi":
                    forecaster_pairs = consistentforecaster_pairs_newsapi
                case "scraped":
                    forecaster_pairs = consistentforecaster_pairs_scraped
                case _:
                    raise ValueError(f"Invalid dataset: {args.dataset}")
        else:
            match args.dataset:
                case "newsapi":
                    forecaster_pairs = forecaster_pairs_newsapi
                    if args.include_cfcaster:
                        forecaster_pairs += consistentforecaster_pairs_newsapi
                case "scraped":
                    forecaster_pairs = forecaster_pairs_scraped
                    if args.include_cfcaster:
                        forecaster_pairs += consistentforecaster_pairs_scraped
                case _:
                    raise ValueError(f"Invalid dataset: {args.dataset}")
            if not args.include_perplexity:
                forecaster_pairs = [
                    pair
                    for pair in forecaster_pairs
                    if pair["short_name"] != "Perplexity"
                ]
            if not args.include_baseline:
                forecaster_pairs = [
                    pair
                    for pair in forecaster_pairs
                    if pair["short_name"] != "Baseline"
                ]

def load_directory_pairs(args: argparse.Namespace) -> List[ForecasterPair]:
    if args.all:
        if args.only_cfcaster:
            match args.dataset:
                case "newsapi":
                    forecaster_pairs = consistentforecaster_pairs_newsapi
                case "scraped":
                    forecaster_pairs = consistentforecaster_pairs_scraped
                case _:
                    raise ValueError(f"Invalid dataset: {args.dataset}")
        else:
            forecaster_pairs = get_forecaster_pairs(args.dataset)
            if not args.include_perplexity:
                forecaster_pairs = [
                    pair for pair in forecaster_pairs if pair["short_name"] != "Perplexity"
                ]
            if not args.include_baseline:
                forecaster_pairs = [
                    pair for pair in forecaster_pairs if pair["short_name"] != "Baseline"
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


if __name__ == "__main__":
    main()


# Example command:
# python src/plot_consistency_vs_brier.py --all --dataset newsapi --gt_metric avg_platt_brier_score -t frequentist -c avg_violation --remove_gt_outlier 0.25
