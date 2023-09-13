"""Utilities for the regex model."""
import numpy as np


# Log likelihood penalty per character if the value cannot be matched.
LOG_LIKE_PER_CHAR = np.log(1e-3)


def sum_log(log_likes):
    """Sum of log likelihoods."""
    log_likes = np.array(log_likes)
    max_log = np.max(log_likes)
    rel_probs = np.exp(log_likes-max_log)
    sum_prob = np.sum(rel_probs)
    return np.log(sum_prob) + max_log


class NotFittedError(ValueError):
    """Signal that the regex could not be fitted."""
