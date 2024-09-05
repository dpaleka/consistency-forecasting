from typing import Callable, Any
import math
import numpy as np
import matplotlib.pyplot as plt
from common.datatypes import ForecastingQuestion, Forecast
from matplotlib.figure import Figure


def log_score(probability: float, outcome: bool) -> float:
    epsilon = 1e-8
    if outcome:
        return math.log2(probability + epsilon)
    else:
        return math.log2(1 - probability + epsilon)


def brier_score(probability: float, outcome: bool) -> float:
    return (probability - int(outcome)) ** 2


scoring_functions = {"log_score": log_score, "brier_score": brier_score}


def proper_scoring_rule(
    forecasting_question: ForecastingQuestion,
    forecast: Forecast,
    scoring_function: Callable[[float, bool], float] | str,
) -> float:
    """
    Calculate the score for a given forecasting question and forecast using a proper scoring rule.

    Args:
        forecasting_question (ForecastingQuestion): The forecasting question object.
        forecast (Forecast): The forecast object containing the probability estimate.
        scoring_function (Callable[[float, bool], float] | str):
        A function that takes a probability and a boolean outcome and returns a score.
        If a string, it must be one of the keys in the scoring_functions dictionary.

    Returns:
        float: The calculated score.

    Raises:
        ValueError: If the forecasting question's resolution is not available.
    """
    if forecasting_question.resolution is None:
        print(f"{forecasting_question = }")
        raise ValueError(
            "The forecasting question must have a resolution to calculate the score."
        )

    probability = forecast.prob
    outcome = forecasting_question.resolution

    return scoring_function(probability, outcome)


def calculate_calibration(
    outcomes: list[bool],
    probs: list[float],
    num_bins: int = 10,
) -> dict[str, Any]:
    """
    Calculate calibration metrics for a list of forecasts and corresponding questions.

    Returns:
        Dict[str, Any]: A dictionary containing calibration metrics:
            - 'bins': List of bin edges
            - 'bin_accuracies': List of actual accuracies for each bin
            - 'bin_confidences': List of average confidences for each bin
            - 'num_samples': List of number of samples in each bin
            - 'calibration_error': Overall calibration error (lower is better)
    """
    print(f"{len(outcomes)=}, {len(probs)=}")
    if len(outcomes) != len(probs):
        raise ValueError("The number of outcomes and probabilities must be the same.")

    # Sort forecasts and questions by forecast probability
    sorted_pairs = sorted(zip(probs, outcomes), key=lambda x: x[0])
    sorted_probs = [p for p, _ in sorted_pairs]
    sorted_outcomes = [q for _, q in sorted_pairs]

    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(sorted_probs, bins) - 1

    bin_accuracies = []
    bin_confidences = []
    num_samples = []

    for i in range(num_bins):
        bin_mask = bin_indices == i
        bin_outcomes = np.array(sorted_outcomes)[bin_mask]
        bin_probs = np.array(sorted_probs)[bin_mask]

        if len(bin_outcomes) > 0:
            bin_accuracies.append(np.mean(bin_outcomes))
            bin_confidences.append(np.mean(bin_probs))
            num_samples.append(len(bin_outcomes))
        else:
            bin_accuracies.append(None)
            bin_confidences.append(None)
            num_samples.append(0)

    # Calculate calibration error
    valid_bins = [i for i in range(num_bins) if bin_accuracies[i] is not None]
    calibration_error = np.mean(
        [abs(bin_accuracies[i] - bin_confidences[i]) for i in valid_bins]
    )

    return {
        "bins": bins.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "num_samples": num_samples,
        "calibration_error": calibration_error,
    }


def get_calibration_points(probs, outcomes):
    points = [1, 3, 5] + list(range(10, 100, 10)) + [95, 97, 99]
    prob_buckets = {p: 0 for p in points}
    count_buckets = {p: 0 for p in points}

    for prob, outcome in zip(probs, outcomes):
        raw_p = prob * 100

        p = min(points, key=lambda x: abs(x - raw_p))

        if outcome:
            raw_p = prob * 100

        p = min(points, key=lambda x: abs(x - raw_p))

        if outcome:
            prob_buckets[p] += 1
        count_buckets[p] += 1

    buckets = {
        p: prob_buckets[p] / count_buckets[p] if count_buckets[p] else None
        for p in points
    }
    return buckets


def get_xy(prob_buckets: dict[float, float]) -> list[dict[str, float]]:
    return [
        {"x": p / 100, "y": prob_buckets[p]}
        for p in prob_buckets
        if prob_buckets[p] is not None
    ]


def plot_calibration(probs: list[float], outcomes: list[bool]) -> Figure:
    buckets = get_calibration_points(probs, outcomes)
    points = get_xy(buckets)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--")
    ax.scatter([p["x"] for p in points], [p["y"] for p in points], color="blue")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Plot")

    return fig


# Example usage:
# log_score_result = proper_scoring_rule(forecasting_question, forecast, "log_score")
# brier_score_result = proper_scoring_rule(forecasting_question, forecast, "brier_score")
