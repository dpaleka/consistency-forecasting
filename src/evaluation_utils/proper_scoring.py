from typing import Callable, Any
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogitLocator, LogitFormatter
from matplotlib.figure import Figure
import scipy.optimize
import scipy.special


def get_bucket_anchors(num_bins: int | None = None) -> list[float]:
    if num_bins is None:
        anchors = [1, 3, 5] + list(range(10, 100, 10)) + [95, 97, 99]
        anchors = [a / 100 for a in anchors]
    else:
        eps = 1e-3
        anchors = scipy.special.expit(
            np.linspace(
                scipy.special.logit(eps), scipy.special.logit(1 - eps), num_bins
            )
        )
    print(anchors)
    return anchors


def assign_bins(probs: list[float], bins: list[float]) -> list[int]:
    assert sorted(bins) == bins, "Bins must be sorted."
    return [min(range(len(bins)), key=lambda i: abs(bins[i] - p)) for p in probs]


def log_score(probability: float, outcome: bool) -> float:
    epsilon = 1e-8
    if outcome:
        return -math.log2(probability + epsilon)
    else:
        return -math.log2(1 - probability + epsilon)


def brier_score(probability: float, outcome: bool) -> float:
    return (probability - int(outcome)) ** 2


def scale_brier_score(brier_score: float) -> float:
    """
    Scale the Brier score linearly where:
    .25 is mapped to 0,
    .00 is mapped to 100
    allow negative scores for worse than random

    Args:
        brier_score (float): The original Brier score to be scaled

    Returns:
        float: The scaled Brier score
    """
    assert 0 <= brier_score <= 1

    # Perform linear interpolation
    scaled_score = 100 * (0.25 - brier_score) / 0.25

    return round(scaled_score, 2)


def decompose_brier_score(
    probs: list[float], outcomes: list[bool], num_bins: int | None = None
) -> dict[str, float]:
    """
    Decompose the Brier score into its three components: Uncertainty, Reliability, and Resolution.

    Args:
        probs (list[float]): List of probabilities.
        outcomes (list[bool]): List of outcomes corresponding to the probabilities.
        num_classes (int, optional): Number of classes in the event. Defaults to 2.

    Returns:
        dict[str, float]: A dictionary containing the decomposed Brier score components.
    """
    assert len(probs) == len(outcomes)

    bins = get_bucket_anchors(num_bins)
    bin_indices = assign_bins(probs, bins)

    # Initialize variables for decomposition
    REL = 0
    RES = 0
    UNC = 0

    # Calculate the observed climatological base rate for the event to occur
    observed_base_rate = sum(outcomes) / len(probs)

    # Calculate the number of unique forecasts issued
    unique_forecasts = set(probs)

    # Initialize a dictionary to hold the number of forecasts and observed frequencies for each probability category
    forecast_counts = {
        bin_index: {"count": 0, "observed_frequency": 0}
        for bin_index in range(len(bins))
    }

    # Count the number of forecasts and observed frequencies for each probability category
    for prob, outcome, bin_index in zip(probs, outcomes, bin_indices):
        forecast_counts[bin_index]["count"] += 1
        if outcome:
            forecast_counts[bin_index]["observed_frequency"] += 1

    # Calculate the components of the Brier score decomposition
    for bin_index, details in forecast_counts.items():
        n_k = details["count"]
        o_bar_k = details["observed_frequency"] / n_k if n_k > 0 else 0
        REL += n_k * (prob - o_bar_k) ** 2
        RES += n_k * (o_bar_k - observed_base_rate) ** 2

    # Calculate the Uncertainty component
    UNC = observed_base_rate * (1 - observed_base_rate)

    # Normalize the components by the total number of forecasts
    REL /= len(probs)
    RES /= len(probs)

    # Return the decomposed Brier score components
    return {
        "uncertainty": UNC,
        "reliability": REL,
        "resolution": RES,
    }


scoring_functions = {"log_score": log_score, "brier_score": brier_score}


def proper_score(
    probs: list[float],
    outcomes: list[bool | None],
    scoring_function: Callable[[float, bool], float] | str,
):
    assert isinstance(probs, list) and isinstance(
        outcomes, list
    ), "Score expects lists of probabilities and outcomes."
    for o in outcomes:
        assert o in [True, False], "Score expects outcomes to be boolean."
    assert len(probs) == len(
        outcomes
    ), "The number of probabilities and outcomes must be the same."
    if isinstance(scoring_function, str):
        scoring_function = scoring_functions[scoring_function]
    return sum(scoring_function(p, o) for p, o in zip(probs, outcomes))


from collections import namedtuple

PlattScalingResult = namedtuple(
    "PlattScalingResult", ["calibrated_probs", "platt_scaling_a"]
)


def platt_scaling(
    probs: list[float],
    outcomes: list[bool] | None = None,
    a: float | None = None,
) -> PlattScalingResult:
    """
    Implement Platt scaling to calibrate probabilities.

    Args:
        outcomes (list[bool]): List of boolean outcomes.
        probs (list[float]): List of initial probabilities.

    Returns:
        PlattScalingResult: A named tuple containing:
            - calibrated_probs: List of calibrated probabilities
            - platt_scaling_a: The optimal hyperparameter 'a'
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    eps = 1e-7

    def loss_function(a, probs, outcomes):
        logits = np.log(np.array(probs) + eps / (1 + eps - np.array(probs)))
        scaled_probs = sigmoid(a * logits)
        return np.mean((scaled_probs - np.array(outcomes)) ** 2)

    if a is None:
        assert outcomes is not None and len(outcomes) == len(
            probs
        ), "Must provide outcomes to calibrate."
        # Filter out None values from outcomes and corresponding probs
        filtered_outcomes = [o for o in outcomes if o is not None]
        filtered_probs = [p for p, o in zip(probs, outcomes) if o is not None]
        if len(filtered_probs) == 0:
            print("No non-None outcomes to calibrate to. Skipping Platt scaling.")
            return PlattScalingResult(calibrated_probs=probs, platt_scaling_a=1)

        # Find optimal 'a' using scipy's minimize_scalar
        result = scipy.optimize.minimize_scalar(
            loss_function, args=(filtered_probs, filtered_outcomes)
        )
        a = result.x
    else:
        assert isinstance(a, float), "Must provide a float for a."

    # Calculate calibrated probabilities
    logits = np.log(np.array(probs) + eps / (1 + eps - np.array(probs)))
    calibrated_probs = sigmoid(a * logits).tolist()

    return PlattScalingResult(calibrated_probs=calibrated_probs, platt_scaling_a=a)


def calculate_calibration(
    outcomes: list[bool],
    probs: list[float],
    num_bins: int | None = None,
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

    sorted_pairs = sorted(zip(probs, outcomes), key=lambda x: x[0])
    sorted_probs, sorted_outcomes = zip(*sorted_pairs)

    # Create bins
    bins = get_bucket_anchors(num_bins)
    bin_indices = assign_bins(sorted_probs, bins)

    bin_accuracies = []
    bin_confidences = []
    num_samples = []

    for i in range(len(bins)):
        bin_mask = np.array(bin_indices) == i
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
    valid_bins = [i for i in range(len(bins)) if bin_confidences[i] is not None]
    calibration_error = np.mean(
        [abs(bin_accuracies[i] - bin_confidences[i]) for i in valid_bins]
    )
    print(f"{bin_accuracies=}")
    print(f"{calibration_error=}")

    return {
        "bins": [bins[i] for i in valid_bins],
        "bin_accuracies": [bin_accuracies[i] for i in valid_bins],
        "bin_confidences": [bin_confidences[i] for i in valid_bins],
        "num_samples": [num_samples[i] for i in valid_bins],
        "calibration_error": calibration_error,
    }


def get_plot_calibration_points(probs, outcomes, num_bins: int | None = None):
    points = get_bucket_anchors(num_bins)
    prob_buckets = {p: 0 for p in points}
    count_buckets = {p: 0 for p in points}

    for prob, outcome in zip(probs, outcomes):
        p = min(points, key=lambda x: abs(x - prob))
        if outcome:
            prob_buckets[p] += 1
        count_buckets[p] += 1

    buckets = {
        p: prob_buckets[p] / count_buckets[p] if count_buckets[p] > 0 else None
        for p in points
    }
    return buckets


def get_xy(prob_buckets: dict[float, float]) -> list[dict[str, float]]:
    return [
        {"x": p, "y": prob_buckets[p]}
        for p in prob_buckets
        if prob_buckets[p] is not None
    ]


# display all buckets on x-axis
# those are in logit space
def plot_calibration(
    probs: list[float],
    outcomes: list[bool],
    num_bins: int | None = None,
    spacing: str = "logit",
    title_info: str = "Calibration Plot",
) -> Figure:
    buckets = get_plot_calibration_points(probs, outcomes, num_bins)
    points = get_xy(buckets)

    fig, ax = plt.subplots(figsize=(10, 10))  # Square figure for equal aspect ratio
    ax.set_aspect("equal")
    # Set labels and title
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")

    ax.set_title(f"{title_info}, {spacing} scale")

    match spacing:
        case "logit":
            # Set both axes to logit scale
            ax.set_xscale("logit")
            ax.set_yscale("logit")

            # Set custom ticks and labels for both axes
            locator = LogitLocator()
            formatter = LogitFormatter()

            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(locator)
                axis.set_minor_locator(locator)
                axis.set_major_formatter(formatter)

            # Plot the diagonal line
            ax.plot([0.001, 0.999], [0.001, 0.999], "k--", linewidth=1)

            # Plot the data points
            ax.scatter([p["x"] for p in points], [p["y"] for p in points], color="blue")

            x_min, x_max = min([p["x"] for p in points]), max([p["x"] for p in points])
            x_min = min(x_min, 0.001)
            x_max = max(x_max, 0.999)
            # Set axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, 1)

            # Add grid
            ax.grid(True, which="both", linestyle="--", alpha=0.7)
        case "linear":
            ax.set_aspect("equal")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.scatter([p["x"] for p in points], [p["y"] for p in points], color="blue")
            ax.grid(True, which="both", linestyle="--", alpha=0.7)
        case _:
            raise ValueError(f"Unsupported spacing: {spacing}")

    return fig


# Example usage:
# log_score_result = proper_scoring_rule(forecasting_question, forecast, "log_score")
# brier_score_result = proper_scoring_rule(forecasting_question, forecast, "brier_score")
