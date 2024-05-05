# Standard library imports
import logging

# Related third-party imports
import numpy as np
from numpy.linalg import norm

# Local application/library-specific imports
from utils import time_utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def brier_score(probabilities, answer_idx):
    """
    Calculate the Brier score for a set of probabilities and the correct answer
    index.

    Args:
    - probabilities (numpy array): The predicted probabilities for each class.
    - answer_idx (int): Index of the correct answer.

    Returns:
    - float: The Brier score.
    """
    answer = np.zeros_like(probabilities)
    answer[answer_idx] = 1
    return ((probabilities - answer) ** 2).sum() / 2


def cosine_similarity(u, v):
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(u, v) / (norm(u) * norm(v))


def get_average_forecast(date_pred_list):
    """
    Retrieve the average forecast value from the list of predictions.

    Args:
    - date_pred_list (list of tuples): list contain tuples of (date str, pred).

    Returns:
    - float: The average prediction.
    """
    if not date_pred_list or len(date_pred_list) == 0:
        return 0.5  # Return a default value of 0.5 if there is no history
    return sum(tup[1] for tup in date_pred_list) / len(date_pred_list)


def compute_bs_and_crowd_bs(pred, date_pred_list, retrieve_date, answer):
    """
    Computes Brier scores for individual prediction and community prediction.

    Parameters:
    - pred (float): The individual's probability prediction for an event.
    - date_pred_list (list of tuples): A list of tuples containing dates
        and community predictions. Each tuple is in the format (date, prediction).
    - retrieve_date (date): The date for which the community prediction is to be retrieved.
    - answer (int): The actual outcome of the event, where 0 indicates the event
        did not happen, and 1 indicates it did.

    Returns:
    - bs (float): The Brier score for the individual prediction.
    - bs_comm (float): The Brier score for the community prediction closest to the specified retrieve_date.
    """
    pred_comm = time_utils.find_closest_date(retrieve_date, date_pred_list)[-1]
    bs = brier_score([1 - pred, pred], answer)
    bs_comm = brier_score([1 - pred_comm, pred_comm], answer)

    return bs, bs_comm
