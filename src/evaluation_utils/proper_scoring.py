from typing import Callable, Any
import math
import numpy as np
import matplotlib.pyplot as plt
from common.datatypes import ForecastingQuestion, Forecast


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
        raise ValueError(
            "The forecasting question must have a resolution to calculate the score."
        )

    probability = forecast.prob
    outcome = forecasting_question.resolution

    return scoring_function(probability, outcome)


def calculate_calibration(
    forecasting_questions: list[ForecastingQuestion],
    forecasts: list[Forecast],
    num_bins: int = 10,
) -> dict[str, Any]:
    """
    Calculate calibration metrics for a list of forecasts and corresponding questions.

    Args:
        forecasting_questions (List[ForecastingQuestion]): List of forecasting questions.
        forecasts (List[Forecast]): List of forecasts corresponding to the questions.
        num_bins (int): Number of bins to use for calibration calculation. Default is 10.

    Returns:
        Dict[str, Any]: A dictionary containing calibration metrics:
            - 'bins': List of bin edges
            - 'bin_accuracies': List of actual accuracies for each bin
            - 'bin_confidences': List of average confidences for each bin
            - 'num_samples': List of number of samples in each bin
            - 'calibration_error': Overall calibration error (lower is better)
    """
    if len(forecasting_questions) != len(forecasts):
        raise ValueError("The number of questions and forecasts must be the same.")

    # Sort forecasts and questions by forecast probability
    sorted_pairs = sorted(
        zip(forecasts, forecasting_questions), key=lambda x: x[0].prob
    )
    sorted_probs = [f.prob for f, _ in sorted_pairs]
    sorted_outcomes = [q.resolution for _, q in sorted_pairs]

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


def plot_calibration(
    probs: list[float],
    outcomes: list[bool],
) -> plt.figure:
    # make the calibration plot
    raise NotImplementedError


# Example usage:
# log_score_result = proper_scoring_rule(forecasting_question, forecast, "log_score")
# brier_score_result = proper_scoring_rule(forecasting_question, forecast, "brier_score")
