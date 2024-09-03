from typing import Callable
import math
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


# Example usage:
# log_score_result = proper_scoring_rule(forecasting_question, forecast, "log_score")
# brier_score_result = proper_scoring_rule(forecasting_question, forecast, "brier_score")
