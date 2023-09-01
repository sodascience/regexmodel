import numpy as np
from regexmodel.util import Dir
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


class Edge():
    def __init__(self, destination, count=None, direction=Dir.RIGHT):
        if count is None:
            assert destination is not None
            count = destination.count
        self.count = count
        self.destination = destination
        self.direction = direction

    @property
    def regex(self):
        if self.destination is None:
            return ""
        return self.destination.full_regex
