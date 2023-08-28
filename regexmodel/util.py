from enum import Enum

import numpy as np


UNVIABLE_REGEX = -1000000


class Dir(Enum):
    LEFT = 1
    RIGHT = 2
    BOTH = 3


def sum_prob_log(probs, log_likes):
    log_likes = np.array(log_likes)
    probs = np.array(probs)
    max_log = np.max(log_likes)
    rel_probs = np.exp(log_likes-max_log)
    assert len(probs) > 0
    # assert np.sum(probs) > 0
    # assert np.sum(rel_probs) > 0, (probs, log_likes, rel_probs, max_log)
    # assert np.sum(probs*rel_probs) > 0, (probs, log_likes, rel_probs, max_log)
    sum_prob = np.sum(probs*rel_probs)
    if sum_prob > 0:
        return np.log(sum_prob) + max_log
    return UNVIABLE_REGEX + max_log
