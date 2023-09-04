import numpy as np
from regexmodel.util import Dir, sum_prob_log
from regexmodel.regexclass import BaseRegex


class BaseNode():
    @property
    def full_regex(self):
        return self.regex + self.next.regex


class RegexNode(BaseNode):
    def __init__(self, regex, next_edge):
        # self.super_edge = edge if super_edges is None else super_edge
        self._regex = regex
        self.next = next_edge
        assert isinstance(regex, BaseRegex)
        assert isinstance(self.next, Edge)

    @property
    def regex(self):
        return self._regex.regex

    @property
    def count(self):
        return self.next.count

    @property
    def n_param(self):
        return self.next.n_param

    def log_likelihood(self, value):
        all_probs = []
        log_prob_next = []
        for post_str, prob in self._regex.fit_value(value):
            all_probs.append(prob)
            log_prob_next.append(self.next.log_likelihood(post_str))
        return sum_prob_log(all_probs, log_prob_next)


class OrNode(BaseNode):
    def __init__(self, edges, next_edge):
        self.edges = edges
        self.count = np.sum([edge.count for edge in self.edges])
        self.next = next_edge

    @property
    def regex(self):
        regex_str_list = [edge.regex for edge in self.edges]
        regex_str = "|".join(regex_str_list)
        if len(self.edges) > 1:
            return r"(" + regex_str + ")"
        return regex_str

    def add_edge(self, new_edge):
        self.count += new_edge.count
        self.edges.append(new_edge)

    def __repr__(self):
        return f"{self.__class__.__name__} <{self.regex}> -> <{self.next.regex}>"

    @property
    def n_param(self):
        sub_param = np.sum([edge.n_param for edge in self.edges])
        return sub_param + self.next.n_param

    def log_likelihood(self, value):
        probs = np.array([edge.count for edge in self.edges])
        probs /= np.sum(probs)
        log_likes = np.array([edge.log_like(value) for edge in self.edges])
        return sum_prob_log(probs, log_likes)


class Edge():
    def __init__(self, destination, count=None):
        if count is None:
            assert destination is not None
            count = destination.count
        self.count = count
        self.destination = destination

    @property
    def regex(self):
        if self.destination is None:
            return ""
        return self.destination.full_regex

    @property
    def n_param(self):
        if self.destination is None:
            return 1
        return self.destination.n_param + 1
