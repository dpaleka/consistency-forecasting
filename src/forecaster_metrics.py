import json
import os
from typing import List, Dict, TypedDict


class ForecasterPair(TypedDict):
    name: str
    short_name: str
    ground_truth_dir: str
    eval_dir: str


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


def get_forecaster_pairs(dataset: str) -> List[ForecasterPair]:
    match dataset:
        case "newsapi":
            return forecaster_pairs_newsapi
        case "scraped":
            return forecaster_pairs_scraped
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")


def get_brier_score_metrics() -> List[str]:
    return [
        "avg_brier_score",
        "avg_platt_brier_score",
        #        "avg_brier_score_scaled",
        #        "avg_platt_brier_score_scaled",
        "avg_log_score",
        "calibration_error",
    ]


def get_consistency_metrics() -> List[str]:
    return [
        "avg_violation",
        "avg_violation_no_outliers",
        "median_violation",
        "frac_violations",
    ]


def get_consistency_metric_types() -> List[str]:
    return ["default", "frequentist", "default_scaled"]


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


def get_cons_metric_label(
    checker_name: str, cons_metric_type: str, cons_metric_key: str
) -> str:
    if checker_name == "aggregate":
        return f"aggregate.{cons_metric_type}.{cons_metric_key}"
    return f"{checker_name}.{cons_metric_type}.{cons_metric_key}"


def extract_all_metrics(forecaster_pair: ForecasterPair) -> Dict[str, float]:
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
        return {}

    metrics = {}
    for brier_metric in get_brier_score_metrics():
        metrics[brier_metric] = ground_truth_data.get(brier_metric, 0.0)

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
                    print(f"cons_metric: {cons_metric}")
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
