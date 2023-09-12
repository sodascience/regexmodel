"""Regex model class for fitting structured strings."""
from __future__ import annotations

from functools import cached_property
from typing import Iterator, Optional, Iterable, Sequence, Union

import numpy as np
import polars as pl

from regexmodel.regexclass import BaseRegex
from regexmodel.regexclass import regex_list_from_string, fit_best_regex_class
from regexmodel.util import sum_prob_log, sum_log, LOG_LIKE_PER_CHAR
from regexmodel.datastructure import Edge, OrNode, RegexNode
# from regexmodel.model import fit_main_branch


def _preview(series, size=3):
    return series.drop_nulls()[:size].to_numpy()


def _simplify_edge(edge):
    node = edge.destination
    if node is None:
        return edge
    if len(node.edges) == 1:
        return node.edges[0]
    return edge


def fit_main_branch(series: pl.Series,
                    count_thres: int,
                    force_merge: bool = False) -> Edge:

    # Use the returnnode/edge for returning
    return_node = OrNode([], Edge(None, 0))
    # return_edge = Edge(return_node)

    # Add an END edge
    n_end_links = (series == "").sum()
    if n_end_links > count_thres:
        return_node.add_edge(Edge(None, n_end_links))
    cur_series = series.set(series == "", None)  # type: ignore

    while cur_series.drop_nulls().len() > count_thres:
        result = fit_best_regex_class(cur_series, count_thres, force_merge=force_merge)

        # If the regex fails the threshold, stop the search.
        if result is None:
            return _simplify_edge(Edge(return_node))

        new_edge = fit_main_branch(
            result["new_series"], count_thres=count_thres,
            force_merge=force_merge)

        # If there are no paths to the end of the string, quit.
        if new_edge.count == 0:
            if not force_merge:
                force_merge = True
                continue
            return _simplify_edge(Edge(return_node))

        # Remove the considered items from the current series
        cur_series = cur_series.set(result["new_series"].is_not_null(), None)  # type: ignore

        # Try to see if we need optional values.
        new_node = RegexNode(result["regex"], new_edge)
        main_edge = Edge(new_node, new_edge.count)
        cur_or_node = OrNode([Edge(None, new_edge.count)], main_edge)

        alt_series = result["alt_series"]
        if alt_series.drop_nulls().len() > count_thres:
            opt_series = alt_series.str.extract(r"(^[\S\s]*?)" + main_edge.regex + r"$")
            alt_edge = fit_main_branch(opt_series, count_thres, force_merge=force_merge)
            if alt_edge.count > 0:
                cur_or_node.add_edge(alt_edge)
                match_regex = r"^(" + alt_edge.regex + r")$"
                used_series = opt_series.str.extract(match_regex)
                cur_series = cur_series.set(used_series.is_not_null(), None)  # type: ignore

        if len(cur_or_node.edges) == 1:
            return_node.add_edge(main_edge)
        else:
            return_node.add_edge(Edge(cur_or_node))

    return _simplify_edge(Edge(return_node))


class RegexModel():
    """Model class to fit and create structured strings.

    This class models structured strings and new strings can be drawn from this model.
    It also features serialization and deserialization methods, so that it can be stored
    easily, for example in JSON files.

    Parameters
    ----------
    regex_data:
        Input for the regex model. Normally, the regex model would be initilized with the
        fit method, but it can also be directly initialized. It has different ways of initializing
        depending on the input type:
        - string: Create a regex model that follows exactly this regex. The language model of this
                  class is more limited than full regex, with only [0-9], [a-z], [A-Z], [xsd]
                  available.
        - list[Link]: This is the native underlying datastructure with which the model is
                      initialized.
        - RegexModel: Creates a shallow copy of another regex model.
        - list[dict]: A serialized version of the regex model.
    """

    def __init__(self, regex_data: Union[Edge, str, dict, RegexModel, list, tuple]):
        if isinstance(regex_data, Edge):
            self.regex_edge = regex_data
        elif isinstance(regex_data, str):
            self.regex_edge, regex_str = Edge.from_string(regex_data)
            assert regex_str == ""
        elif isinstance(regex_data, RegexModel):
            self.regex_edge = regex_data.regex_edge
        else:
            if isinstance(regex_data, dict):
                regex_str, counts = regex_data["regex"], regex_data["counts"]
            else:
                assert len(regex_data) == 2
                regex_str, counts = regex_data
            self.regex_edge, regex_str = Edge.from_string(regex_str)
            assert regex_str == ""
            self.regex_edge.set_counts(counts)

    @classmethod
    def fit(cls, values: Union[Iterable, Sequence], count_thres: int = 3):
        """Fit a sequence of values to create a regex model.

        Parameters
        ----------
        values:
            Sequence of values to fit the model on.
        count_thres:
            Only consider branches with (approximately) count_thres strings associated with it.

        Returns
        -------
            Fitted regex model.
        """
        values = pl.Series(values)
        return cls(fit_main_branch(values, count_thres=count_thres))

    @classmethod
    def from_regex(cls, regex_str: str):
        """Create a regex model from a regex string.

        It creates a simple model with no branches and with weights {1, 0}.

        Parameters
        ----------
        regex_str:
            Regex to create the model from.
        """
        return cls(regex_str)

    def serialize(self) -> dict:
        """Serialize the regex model.

        For example used to store the model in a JSON file.
        """
        regex_str = self.regex_edge.regex
        counts = self.regex_edge.count_list
        return {
            "regex": regex_str,
            "counts": counts,
        }

    @classmethod
    def deserialize(cls, regex_data: dict):
        """Create a regex model from the serialization.

        Parameters
        ----------
        regex_data:
            Serialized regex model.
        """
        return cls(regex_data)

    def draw(self) -> str:
        """Draw a structured string from the regex model."""
        return self.regex_edge.draw()
        # counts = np.array([link.count for link in self.root_links])
        # link = np.random.choice(self.root_links, p=counts/np.sum(counts))  # type: ignore
        # return link.draw()

    @cached_property
    def _root_prob(self):
        """Return the probability of each main branch."""
        counts = np.array([link.count for link in self.root_links])
        return counts/np.sum(counts)

    @cached_property
    def n_param(self):
        """Number of parameters of the model."""
        return self.regex_edge.n_param
        # return np.sum([link.n_param for link in self.root_links])

    @cached_property
    def regex(self):
        return self.regex_edge.regex

    def AIC(self, values) -> float:  # pylint: disable=invalid-name
        """Akaike Information Criterion for the given values."""
        return 2*self.n_param - 2*self.log_likelihood(values)

    def log_likelihood(self, values) -> float:
        """Log likelihood for the given values."""
        stats = self.fit_statistics(values)
        return stats["n_tot_char"]*stats["avg_log_like_per_char"]

    def _check_zero_links(self):
        """Debug method."""
        for link in self.root_links:
            link.check_zero_links()

    def fit_statistics(self, values) -> dict:
        """Get the performance of the regex model on some values.

        These can be the same values that the regex model was fitted on,
        or new values. This method can be quite slow depending on the size
        and complexity of the regex model.

        Parameters
        ----------
        values:
            Values to get the performance metrics for.

        Returns
        -------
            Dictionary with different metrics, such as failures/successes/parameters.
        """
        res = {
            "failed": 0,
            "success": 0,
            "n_tot_char": 0,
            "n_char_success": 0,
            "n_parameters": int(self.n_param),
            "avg_log_like_per_char": 0.0,
            "avg_log_like_pc_success": 0.0,
        }
        for val in values:
            log_likes = [x[1] for x in self.regex_edge.log_likelihood(val) if x[0] == ""]
            if len(log_likes) == 0:
                cur_log_like = max(len(val), 1)*LOG_LIKE_PER_CHAR
                res["failed"] += 1
            else:
                cur_log_like = sum_log(log_likes)
                res["n_char_success"] += len(val)
                res["avg_log_like_pc_success"] += cur_log_like
                res["success"] += 1
            res["n_tot_char"] += len(val)
            res["avg_log_like_per_char"] += cur_log_like
        if res["n_char_success"] > 0:
            res["avg_log_like_pc_success"] /= res["n_char_success"]
        if res["n_tot_char"] > 0:
            res["avg_log_like_per_char"] /= res["n_tot_char"]
        return res
