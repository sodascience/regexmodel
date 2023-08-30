"""Regex model class for fitting structured strings."""
from __future__ import annotations

import heapq
from functools import cached_property
from typing import Iterator, Optional, Iterable, Sequence, Union

import numpy as np
import polars as pl

from regexmodel.regexclass import BaseRegex
from regexmodel.regexclass import MultiRegex, regex_list_from_string, fit_best_regex_class
from regexmodel.datastructure import Link, Node, UNVIABLE_REGEX, LOG_LIKE_PER_CHAR
from regexmodel.util import Dir, sum_prob_log


def fit_main_branch(series: pl.Series,
                    score_thres: float,
                    direction=Dir.RIGHT) -> list[BaseRegex]:
    """Fit a main branch for a series.

    The main branch is the most likely/important branch. This uses a greedy algorithm
    to find the beste regex class progressively, until no more regexes that satisfy the
    score threshold can be found.

    Parameters
    ----------
    series:
        Series to get the main branch of.
    score_thres:
        Score threshold to determine whether to continue the main branch. If
        the best regex class is below this threshold, the fitting procedure will stop
        trying to add more regex classes.
    direction:
        Direction of the fit. Dir.RIGHT/Dir.BOTH will fit from left to right, while
        Dir.LEFT fits regex classes from right to left.

    Returns
    -------
        List of regex elements that fit the series.
    """
    # If it fails the threshold, stop the search.
    if len(series.drop_nulls())/len(series) < score_thres:
        return []
    result = fit_best_regex_class(series, score_thres, direction=direction)
    if result["score"] < score_thres:
        return []
    regex_list = fit_main_branch(result["new_series"], score_thres=score_thres,
                                 direction=direction)
    if direction == Dir.LEFT:
        return regex_list + [result["regex"]]
    return [result["regex"]] + regex_list


def generate_sub_regexes(regex_list: list[BaseRegex], direction: Dir
                         ) -> Iterator[tuple[str, int, int]]:
    """Iterate over all possible ways to fit a string to a main branch.

    For example, if the main branch (Dir.right) is A -> B -> C, then first
    all strings with ABC in them are fit to the main branch. Then, strings with
    AB and BC, and lastly strings with A, B, or C. This function is technically
    exponential in running time, but as long as the length of the main branch is small,
    this doesn't limit the algorithm in practice.

    This function iterates in a decreasing manner for the number of characters in the
    string to be matched. So for example if the main branch is A -> [0-9]{10} -> B,
    the it will iterate as A[0-9]{10}B, A[0-9]{10}, [0-9]{10}B, [0-9]{10}, ...

    Parameters
    ----------
    regex_list:
        Main branch to fit the structured strings to.
    direction:
        Direction of the main branch.

    Returns
    -------
        An iterator that iterates over all possible to fit a string:
        (regex to match the string, start index of the match, end index of the match).
    """
    # First compute the average number of characters to be matched per element.
    regex_len = []
    for regex in regex_list:
        if isinstance(regex, MultiRegex):
            regex_len.append((regex.min_len + regex.max_len)/2)
        else:
            regex_len.append(1)
    tot_length = np.sum(regex_len)
    if len(regex_len) == 0:
        return

    # Create a heap where the priority is -the total length of the match.
    # The starting position is constant, while after taking it off the heap, the end
    # position is decreased by one.
    heap: list[tuple] = []
    range_cumsum = np.cumsum(regex_len)
    range_cumsum[1:] = range_cumsum[:-1]
    range_cumsum[0] = 0
    for i in range(len(regex_list)):
        cur_range = tot_length - range_cumsum[i]
        heapq.heappush(heap, (-cur_range, (i, len(regex_list))))

    # Keep iterating, until the priority heap is empty.
    while len(heap) > 0:
        min_cur_range, (i_start, i_end) = heapq.heappop(heap)
        if i_end-1 > i_start:
            min_new_range = min_cur_range + regex_len[i_end-1]
            heapq.heappush(heap, (min_new_range, (i_start, i_end-1)))

        # If the direction is left, then all i_end values must be N-1.
        skip_left = (direction == Dir.LEFT and i_end != len(regex_list))
        # if the direction is right, then all i_start values must be 0.
        skip_right = (direction == Dir.RIGHT and i_start != 0)

        if not (skip_left or skip_right):
            # Build the regex for this selection.
            regex_str = ""
            for i in range(i_start, i_end):
                regex_str += regex_list[i].regex

            yield (regex_str, i_start, i_end)


def _merge_series(series_list: list[Optional[pl.Series]]) -> Optional[pl.Series]:
    """Merge polars series overwriting None values.

    This simply gets all the non-null values for each of the elements:
    [x, None], [None, y] -> [x, y]. If the first series is None, then it will
    not merge anything.

    Parameters
    ----------
    series_list:
        A list of polars series, or a list on None's, which are to be merged.
    """
    if series_list is None or len(series_list) == 0 or series_list[0] is None:
        return None
    cur_series = series_list[0]

    # This can probably be more efficient.
    for new_series in series_list[1:]:
        cur_series = pl.DataFrame({"a": cur_series, "b": new_series}).select(
            pl.when(pl.col("a").is_null())
            .then(pl.col("b"))
            .otherwise(pl.col("a"))
            .alias("result")
        )["result"]
    return cur_series


def extract_elements(cur_series: pl.Series,
                     regex: str,
                     direction: Dir) -> tuple[Optional[pl.Series], Optional[pl.Series]]:
    """Extract the before and after series using the regex in the middle.

    Parameters
    ----------
    cur_series:
        Series to apply the regex to.
    regex:
        Regex to extract the elements for.
    direction:
        Direction of the main branch that is being fitted to.

    Returns
    -------
        Series for before and after the regex.
    """
    if direction == Dir.BOTH:
        match_regex = r"^([\S\s]*?)" + regex + r"([\S\s]*?)$"
        start_elem = cur_series.str.extract(match_regex, 1)
        end_elem = cur_series.str.extract(match_regex, 2)
    elif direction == Dir.LEFT:
        start_elem = cur_series.str.extract(r"^([\S\s]*?)" + regex + r"$")
        end_elem = None
    elif direction == Dir.RIGHT:
        start_elem = None
        end_elem = cur_series.str.extract(r"^" + regex + r"([\S\s]*?)$")
    return start_elem, end_elem


def fit_series(  # pylint: disable=too-many-branches
        series: pl.Series,
        score_thres: float,
        direction: Dir = Dir.BOTH) -> list[Link]:
    """Fit the regex model to a series of structured strings.

    It does so in a greedy and recursive way. It first generates a main branch,
    which consists of a list of regex classes. This main branch is then used to incorporate
    the structure of as many structured strings as possible. This is done using side branches.
    These side branches are weighted by the number of times they are taken by the current
    series, so that the model knows how often this side branch is generated.

    After finishing one main branch and fitting as many side branches, the remaining structured
    strings that could not be fitted to the last main branch will then try to form a new and
    parallel main branch. This process is repeated until no new main branch can be formed anymore.
    (Either because there are no more left over structured strings, or the proposed regexes fall
    below the score threshold.)

    Parameters
    ----------
    series:
        The structured strings to model.
    score_thres:
        The score threshold for selecting regex classes.
    direction:
        Direction to fit the series for.

    Returns
    -------
        A list of links that point to all main branches that were found.
    """
    cur_series = series
    center_regex_list = fit_main_branch(cur_series, score_thres=score_thres,
                                        direction=direction)

    root_links: list[Link] = []

    # Keep making main branches until the main branch is empty.
    while len(center_regex_list) > 0:
        n_center_regex = len(center_regex_list)
        start_count = cur_series.drop_nulls().len()
        left_series: list[list[Optional[pl.Series]]] = [[] for _ in range(n_center_regex)]
        right_series: list[list[Optional[pl.Series]]] = [[] for _ in range(n_center_regex)]

        # Setup the nodes/links on the main regex line.
        new_link, nodes = Node.main_branch(center_regex_list, direction, 0)
        root_links.append(new_link)

        # Try all subsets of the main regex to fit the structured strings.
        for regex, i_start, i_end in generate_sub_regexes(center_regex_list,
                                                          direction=direction):
            start_elem, end_elem = extract_elements(cur_series, regex, direction)

            if start_elem is not None:
                cur_count = start_elem.drop_nulls().len()
            elif end_elem is not None:
                cur_count = end_elem.drop_nulls().len()
            else:
                raise ValueError("Internal Error")

            # Keep track of the weights of each regex node.
            if direction == Dir.LEFT:
                for i_cur in range(i_start+1, i_end):
                    nodes[i_cur].main_link.count += cur_count
            else:
                for i_cur in range(i_start, i_end-1):  # Note: -1 or not?
                    nodes[i_cur].main_link.count += cur_count

            # Keep structured strings that fit the current subset to create leaves.
            left_series[i_start].append(start_elem)
            right_series[i_end-1].append(end_elem)

            # Update the current series by removing the fitted strings.
            if start_elem is not None:
                cur_series = cur_series.set(start_elem.is_not_null(), None)  # type: ignore
            else:
                cur_series = cur_series.set(end_elem.is_not_null(), None)  # type: ignore

        # Add new leaf to root node structure
        for i_node in range(n_center_regex):
            # Compute incoming/left leaves
            cur_left_series = _merge_series(left_series[i_node])
            if cur_left_series is not None:
                new_links = fit_series(cur_left_series, score_thres, direction=Dir.LEFT)
                nodes[i_node].sub_links.extend(new_links)

            # Compute exiting/right leaves
            cur_right_series = _merge_series(right_series[i_node])
            if cur_right_series is not None:
                new_links = fit_series(cur_right_series, score_thres, direction=Dir.RIGHT)
                nodes[i_node].sub_links.extend(new_links)

        # Add new leaf to current leaf stack
        total_count = start_count - cur_series.drop_nulls().len()
        root_links[-1].count = total_count
        center_regex_list = fit_main_branch(cur_series, score_thres=score_thres,
                                            direction=direction)

    # If there are leftover structured strings, add a None link.
    n_left_over = cur_series.drop_nulls().len()
    if n_left_over:
        root_links.append(Link(n_left_over, direction=direction))
    return root_links


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

    def __init__(self, regex_data: Union[str, list[Link], list[dict], RegexModel]):
        self.root_links: list[Link]
        if isinstance(regex_data, str):
            self.root_links = self.__class__.from_regex(regex_data).root_links
        elif isinstance(regex_data, RegexModel):
            self.root_links = regex_data.root_links
        else:
            self.root_links = []
            for data in regex_data:
                if isinstance(data, dict):
                    self.root_links.append(Link.deserialize(data))
                else:
                    self.root_links.append(data)
        # self.root_links.
        # elif all(isinstance(data, dict) for data in regex_data):
        # elif not (len(regex_data) > 0 and isinstance(regex_data[0], Link)):
        # self.root_links = self.__class__.deserialize(regex_data).root_links
        # elif all(isinstance(data, Link) for data in regex_data):
        # self.root_links = regex_data

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
        series = pl.Series(list(values)).drop_nulls()  # pylint: disable=assignment-from-no-return
        root_links = fit_series(series, score_thres=count_thres/len(series))
        return cls(root_links)

    @classmethod
    def from_regex(cls, regex_str: str):
        """Create a regex model from a regex string.

        It creates a simple model with no branches and with weights {1, 0}.

        Parameters
        ----------
        regex_str:
            Regex to create the model from.
        """
        all_regexes = regex_list_from_string(regex_str)
        link, nodes = Node.main_branch(all_regexes, Dir.BOTH, 1)
        nodes[0].sub_links.append(Link(1, Dir.LEFT))
        nodes[-1].sub_links.append(Link(1, Dir.RIGHT))
        return cls([link])

    def serialize(self) -> list[dict]:
        """Serialize the regex model.

        For example used to store the model in a JSON file.
        """
        return [link.serialize() for link in self.root_links]

    @classmethod
    def deserialize(cls, regex_data: list):
        """Create a regex model from the serialization.

        Parameters
        ----------
        regex_data:
            Serialized regex model.
        """
        root_links = [Link.deserialize(link_data) for link_data in regex_data]
        return cls(root_links)

    def draw(self) -> str:
        """Draw a structured string from the regex model."""
        counts = np.array([link.count for link in self.root_links])
        link = np.random.choice(self.root_links, p=counts/np.sum(counts))  # type: ignore
        return link.draw()

    @cached_property
    def _root_prob(self):
        """Return the probability of each main branch."""
        counts = np.array([link.count for link in self.root_links])
        return counts/np.sum(counts)

    @cached_property
    def n_param(self):
        """Number of parameters of the model."""
        return np.sum([link.n_param for link in self.root_links])

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
            log_likes = [link.log_likelihood(val) for link in self.root_links]
            cur_log_like = sum_prob_log(self._root_prob, log_likes)
            res["n_tot_char"] += len(val)
            if cur_log_like < UNVIABLE_REGEX/2:
                cur_log_like = max(len(val), 1)*LOG_LIKE_PER_CHAR
                res["failed"] += 1
            else:
                res["n_char_success"] += len(val)
                res["avg_log_like_pc_success"] += cur_log_like
                res["success"] += 1
            res["avg_log_like_per_char"] += cur_log_like
        if res["n_char_success"] > 0:
            res["avg_log_like_pc_success"] /= res["n_char_success"]
        if res["n_tot_char"] > 0:
            res["avg_log_like_per_char"] /= res["n_tot_char"]
        return res
