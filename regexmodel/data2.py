import numpy as np
from regexmodel.util import Dir
from regexmodel.regexclass import BaseRegex


class BaseNode():
    @property
    def full_regex(self):
        return self.regex + self.next.regex


class RegexNode(BaseNode):
    def __init__(self, regex, next_edge):
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

    def __str__(self):
        return "Regex " + self.regex

    def log_likelihood(self, value):
        for post_str, prob in self._regex.fit_value(value, direction=Dir.RIGHT):
            for next_str, next_log in self.next.log_likelihood(post_str):
                yield next_str, np.log(prob) + next_log


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

    def __str__(self):
        return f"{self.__class__.__name__} <{self.regex}> -> <{self.next.regex}>"

    @property
    def n_param(self):
        sub_param = np.sum([edge.n_param for edge in self.edges])
        return sub_param + self.next.n_param

    def log_likelihood(self, value):
        probs = np.array([edge.count for edge in self.edges])
        probs = probs/np.sum(probs)
        for prob, edge in zip(probs, self.edges):
            for post_str, log_prob in edge.log_likelihood(value):
                if self.next:
                    for next_post_str, next_log_prob in self.next.log_likelihood(post_str):
                        yield next_post_str, next_log_prob + log_prob + np.log(prob)
                else:
                    yield post_str, log_prob + np.log(prob)


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

    def log_likelihood(self, value):
        if self.destination is None:
            yield value, 0
        else:
            yield from self.destination.log_likelihood(value)

    def __str__(self):
        ret_str = "Edge <"
        if self.destination is None:
            ret_str += "None"
        else:
            ret_str += self.destination.regex
        ret_str += ">"
        return ret_str
