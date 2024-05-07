# Related third-party imports
import numpy as np
from numpy.linalg import norm

def cosine_similarity(u, v):
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(u, v) / (norm(u) * norm(v))