from enum import Enum

import numpy as np


class Dir(Enum):
    LEFT = 1
    RIGHT = 2
    BOTH = 3


def sum_prob_log(probs, log_likes):
    assert len(probs) > 0

    log_likes = np.array(log_likes)
    probs = np.array(probs)
    max_log = np.max(log_likes)
    rel_probs = np.exp(log_likes-max_log)
    return np.log(np.sum(probs*rel_probs)) + max_log
