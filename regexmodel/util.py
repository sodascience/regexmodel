"""Utilities for the regex model."""
from enum import Enum

import numpy as np


# Used for log likelihood estimation to signify this branch isn't used (for that particular string).
UNVIABLE_REGEX = -1000000


class Dir(Enum):
    """Direction of links."""

    LEFT = 1  # Backward direction
    RIGHT = 2  # Forward direction
    BOTH = 3  # Both directions


def sum_prob_log(probs, log_likes):
    """Weighted sum for log likelihoods.

    Same as sum(probs*np.exp(log_likes)), but taking care of over/underflows.
    """
    log_likes = np.array(log_likes)
    probs = np.array(probs)
    max_log = np.max(log_likes)
    rel_probs = np.exp(log_likes-max_log)
    sum_prob = np.sum(probs*rel_probs)
    if sum_prob > 0:
        return np.log(sum_prob) + max_log
    # Sometimes the sum is basically zero, because all links are not possible.
    return UNVIABLE_REGEX + max_log
