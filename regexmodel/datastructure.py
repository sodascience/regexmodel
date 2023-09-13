"""Regex classes such as [0-9], [a-z], etc are defined in this module."""
from __future__ import annotations
from functools import cached_property
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import numpy as np
from regexmodel.regexclass import BaseRegex, OrRegex


class BaseNode(ABC):
    """Base class for nodes that are used in the finite automata."""

    def __init__(self, next_edge: Edge):
        self.next = next_edge
        assert isinstance(self.next, Edge)

    @property
    def full_regex(self) -> str:
        """Create regex from this node recursively."""
        return self.regex + self.next.regex

    @property
    @abstractmethod
    def regex(self) -> str:
        """Create the regex from this node, not next."""

    @property
    @abstractmethod
    def n_param(self) -> int:
        """Number of parameters for this node recursively."""

    def __str__(self):
        return f"{self.__class__.__name__} <{self.regex}> -> <{self.next.regex}>"

    @abstractmethod
    def log_likelihood(self, value: str) -> Iterable[tuple[str, float]]:
        """Calculate log likelihood recursively."""

    @abstractmethod
    def draw(self) -> str:
        """Draw a random string recursively."""

    @property
    @abstractmethod
    def count(self) -> int:
        """Weight of the node."""


class RegexNode(BaseNode):
    """Node for a single regex element.

    Arguments
    ---------
    regex:
        Regex element for the node.
    next_edge:
        Edge to the next node.
    """

    def __init__(self, regex: BaseRegex, next_edge: Edge):
        self._regex = regex
        self.next = next_edge
        assert isinstance(regex, BaseRegex), str(regex)
        super().__init__(next_edge)

    @property
    def regex(self) -> str:
        return self._regex.regex

    @property
    def count(self) -> int:
        """Get the weight of the node."""
        return self.next.count

    @property
    def n_param(self) -> int:
        return self.next.n_param + self._regex.n_param - 1

    def log_likelihood(self, value: str) -> Iterable[tuple[str, float]]:
        for post_str, prob in self._regex.fit_value(value):
            for next_str, next_log in self.next.log_likelihood(post_str):
                yield next_str, np.log(prob) + next_log

    def draw(self) -> str:
        return self._regex.draw() + self.next.draw()


class OrNode(BaseNode):
    """Node that encodes multiple options.

    In terms of regex, this is the equivalent of (a|bv|x).

    Arguments
    ---------
    edges:
        Edges towards the different options.
    next_edge:
        Edge towards the next node in the regex.
    """

    def __init__(self, edges, next_edge):
        self.edges = edges
        self.recount()
        self._count = 0
        super().__init__(next_edge)

    @property
    def regex(self):
        regex_str_list = [edge.regex for edge in self.edges]
        regex_str = "|".join(regex_str_list)
        if len(self.edges) > 1:
            return r"(" + regex_str + ")"
        return regex_str

    def add_edge(self, new_edge: Edge):
        """Add a new edge to the options.

        Arguments
        ---------
        new_edge:
            Edge to be added.
        """
        self._count += new_edge.count
        self.edges.append(new_edge)

    @property
    def count(self) -> int:
        return self._count

    @property
    def n_param(self):
        sub_param = np.sum([edge.n_param for edge in self.edges])
        return sub_param + self.next.n_param

    @cached_property
    def _probabilities(self):
        probs = np.array([edge.count for edge in self.edges])
        return probs/np.sum(probs)

    def draw(self) -> str:
        edge = np.random.choice(self.edges, p=self._probabilities)
        return edge.draw() + self.next.draw()

    def recount(self):
        """Update the count from the edges."""
        self._count = np.sum([edge.count for edge in self.edges]).astype(int)

    def log_likelihood(self, value: str) -> Iterable[tuple[str, float]]:
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
    """Edge used in the finite automata.

    An edge can point to 'None', in which case it signals the end
    of the string/regex.

    Arguments
    ---------
    destination:
        Node that the edge points to. If destination is None,
        signals the end of the regex.
    count:
        Weight of the edge.
    """

    def __init__(self, destination: Optional[BaseNode],
                 count: Optional[int] = None):
        if count is None:
            assert destination is not None
            count = destination.count
        self.count = count
        self.destination = destination

    @property
    def regex(self):
        """Get the regex for the edge recursively."""
        if self.destination is None:
            return ""
        return self.destination.full_regex

    @property
    def n_param(self):
        """Get the number of parameters of the model."""
        if self.destination is None:
            return 1
        return self.destination.n_param + 1

    def log_likelihood(self, value: str) -> Iterable[tuple[str, float]]:
        """Compute the log likelihood of the model."""
        if self.destination is None:
            yield value, 0.0
        else:
            yield from self.destination.log_likelihood(value)

    def draw(self) -> str:
        """Draw a new string from the model."""
        if self.destination is None:
            return ""
        return self.destination.draw()

    def __str__(self) -> str:
        ret_str = "Edge <"
        if self.destination is None:
            ret_str += "None"
        else:
            ret_str += self.destination.regex
        ret_str += ">"
        return ret_str

    @property
    def count_list(self) -> list:
        """Get a list of the weights for each of the regex nodes."""
        if self.destination is None:
            return [self.count]
        if isinstance(self.destination, OrNode):
            or_counts = [edge.count_list for edge in self.destination.edges]
            return [or_counts, *self.destination.next.count_list]
        if isinstance(self.destination, RegexNode):
            return [self.count, *self.destination.next.count_list]
        raise ValueError("Internal Error")

    @classmethod
    def from_string(cls, regex_str) -> tuple[Edge, str]:
        """Create edges and nodes from a regex.

        This method parses a regex from left to right.

        Arguments
        ---------
        regex_str:
            Regex string to parse.
        """
        # End of the regex string.
        if len(regex_str) == 0:
            return cls(None, 1), ""

        # Start of a regex class.
        if regex_str[0] == "[":
            new_regex, cur_regex_str = OrRegex.from_string(regex_str)
            new_edge, new_str = cls.from_string(cur_regex_str)
            return cls(RegexNode(new_regex, new_edge), 1), new_str

        # Start of an OrRegex construction.
        if regex_str[0] == "(":
            all_edges = []
            cur_regex_str = regex_str[1:]
            while cur_regex_str[0] != ")":
                new_edge, cur_regex_str = cls.from_string(cur_regex_str)
                all_edges.append(new_edge)
                if len(cur_regex_str) == 0:
                    raise ValueError("Unterminated ')' in regex.")
            next_edge, next_str = cls.from_string(cur_regex_str[1:])
            return cls(OrNode(all_edges, next_edge), 1), next_str

        # Continue with another branch of the OrRegex construction
        if regex_str[0] == "|":
            return cls(None, 1), regex_str[1:]

        # End of the OrRegex construction
        if regex_str[0] == ")":
            return cls(None, 1), regex_str
        raise ValueError(f"Failed parsing regex: currently still have: {regex_str}")

    def set_counts(self, counts: Union[int, list]):
        """Set the counts of the edges from the serialization procedure.

        counts:
            Counts for each of the regex nodes/edges.
        """
        if isinstance(counts, int):
            self.count = counts
            return
        if len(counts) == 1 and self.destination is None:
            self.count = counts[0]
            return
        assert self.destination is not None
        if isinstance(self.destination, OrNode):
            for cur_counts, edge in zip(counts[0], self.destination.edges):
                edge.set_counts(cur_counts)
            self.destination.recount()
            self.count = self.destination.count
        elif isinstance(self.destination, RegexNode):
            self.count = counts[0]
        self.destination.next.set_counts(counts[1:])
