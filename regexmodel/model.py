
from __future__ import annotations

import heapq
from functools import cached_property

import numpy as np
import polars as pl
from typing import Iterator, Optional, Iterable, Sequence, Union


from regexmodel.regexclass import UpperRegex, LowerRegex, DigitRegex, LiteralRegex, BaseRegex
from regexmodel.regexclass import MultiRegex, regex_list_from_string
from regexmodel.datastructure import Link, Node, UNVIABLE_REGEX, LOG_LIKE_PER_CHAR
from regexmodel.util import Dir, sum_prob_log


def fit_best_regex(series, score_thres, direction=Dir.RIGHT):
    best_regex = {"score": -1}
    for regex_class in [UpperRegex, LowerRegex, DigitRegex, LiteralRegex]:
        regex_inst, score, new_series = regex_class.fit(
            series, score_thres=score_thres, direction=direction)
        # print(regex_inst, score, new_series.drop_nulls()[:5].to_numpy())
        if score > best_regex["score"]:
            best_regex = {
                "score": score,
                "regex": regex_inst,
                "new_series": new_series,
            }
    return best_regex


def fit_center_branch(series: pl.Series, score_thres: float, direction=Dir.RIGHT) -> list:
    if len(series.drop_nulls())/len(series) < score_thres:
        return []
    result = fit_best_regex(series, score_thres, direction=direction)
    if result["score"] < score_thres:
        return []
    regex_list = fit_center_branch(result["new_series"], score_thres=score_thres,
                                   direction=direction)
    if direction == Dir.LEFT:
        return regex_list + [result["regex"]]
    else:
        return [result["regex"]] + regex_list


def generate_sub_regexes(regex_list: list[BaseRegex], direction: Dir
                         ) -> Iterator[tuple[str, int, int]]:
    regex_len = []
    for regex in regex_list:
        if isinstance(regex, MultiRegex):
            regex_len.append((regex.min_len + regex.max_len)/2)
        else:
            regex_len.append(1)
    tot_length = np.sum(regex_len)
    if len(regex_len) == 0:
        return
    # range_stack = defaultdict(list)
    heap: list[tuple] = []
    range_cumsum = np.cumsum(regex_len)
    range_cumsum[1:] = range_cumsum[:-1]
    range_cumsum[0] = 0
    for i in range(len(regex_list)):
        cur_range = tot_length - range_cumsum[i]
        heapq.heappush(heap, (-cur_range, (i, len(regex_list))))

    while len(heap):
        min_cur_range, (i_start, i_end) = heapq.heappop(heap)
        if i_end-1 > i_start:
            min_new_range = min_cur_range + regex_len[i_end-1]
            heapq.heappush(heap, (min_new_range, (i_start, i_end-1)))

        skip_left = (direction == Dir.LEFT and i_end != len(regex_list))
        skip_right = (direction == Dir.RIGHT and i_end != 0)

        if not skip_left or skip_right:
            regex_str = ""
            for i in range(i_start, i_end):
                regex_str += regex_list[i].regex

            yield (regex_str, i_start, i_end)


def _merge_series(series_list: list[pl.Series]):
    if series_list is None or len(series_list) == 0 or series_list[0] is None:
        return None
    cur_series = series_list[0]

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
                     i_start: int,
                     i_end: int,
                     direction: Dir) -> tuple[Optional[pl.Series], Optional[pl.Series]]:
    if direction == Dir.BOTH:
        start_elem = cur_series.str.extract(r"^(.*?)" + regex + r".*")
        end_elem = cur_series.str.extract(r"^.*" + regex + r"(.*?)$")
    elif direction == Dir.LEFT:
        start_elem = cur_series.str.extract(r"^(.*?)" + regex + r"$")
        end_elem = None
    elif direction == Dir.RIGHT:
        start_elem = None
        end_elem = cur_series.str.extract(r"^" + regex + r"(.*?)$")
    return start_elem, end_elem


def fit_series(series: pl.Series, score_thres, direction=Dir.BOTH) -> list[Link]:
    cur_series = series
    center_regex_list = fit_center_branch(cur_series, score_thres=score_thres,
                                          direction=direction)

    root_links: list[Link] = []

    while len(center_regex_list) > 0:
        n_center_regex = len(center_regex_list)
        start_count = cur_series.drop_nulls().len()
        left_series: list[list[Optional[pl.Series]]] = [[] for _ in range(n_center_regex)]
        right_series: list[list[Optional[pl.Series]]] = [[] for _ in range(n_center_regex)]

        # Setup the nodes/links on the main regex line.
        new_link, nodes = Node.main_branch(center_regex_list, direction, 0)
        root_links.append(new_link)

        for regex, i_start, i_end in generate_sub_regexes(center_regex_list,
                                                          direction=direction):
            start_elem, end_elem = extract_elements(cur_series, regex, i_start, i_end, direction)
            if start_elem is not None:
                cur_count = start_elem.drop_nulls().len()
            elif end_elem is not None:
                cur_count = end_elem.drop_nulls().len()
            else:
                raise ValueError("Internal Error")

            if direction == Dir.LEFT:
                for i_cur in range(i_start+1, i_end):
                    nodes[i_cur].main_link.count += cur_count
            else:
                for i_cur in range(i_start, i_end-1):  # Note: -1 or not?
                    # print(n_center_regex, i_cur)
                    nodes[i_cur].main_link.count += cur_count
            left_series[i_start].append(start_elem)
            right_series[i_end-1].append(end_elem)
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
        center_regex_list = fit_center_branch(cur_series, score_thres=score_thres,
                                              direction=direction)
    n_left_over = cur_series.drop_nulls().len()
    if n_left_over:
        root_links.append(Link(n_left_over, direction=direction))
    return root_links


class RegexModel():
    def __init__(self, regex_data: Union[str, list[Link], list[dict]]):
        if isinstance(regex_data, str):
            self.root_links = self.__class__.from_regex(regex_data).root_links
        elif isinstance(regex_data, RegexModel):
            self.root_links = regex_data.root_links
        elif not (len(regex_data) > 0 and isinstance(regex_data[0], Link)):
            self.root_links = self.__class__.deserialize(regex_data).root_links
        else:
            self.root_links = regex_data

    @classmethod
    def fit(cls, values: Union[Iterable, Sequence], count_thres: int = 3):
        series = pl.Series(list(values)).drop_nulls()
        root_links = fit_series(series, score_thres=count_thres/len(series))
        return cls(root_links)

    @classmethod
    def from_regex(cls, regex_str: str):
        all_regexes = regex_list_from_string(regex_str)
        link, nodes = Node.main_branch(all_regexes, Dir.BOTH, 1)
        nodes[0].sub_links.append(Link(1, Dir.LEFT))
        nodes[-1].sub_links.append(Link(1, Dir.RIGHT))
        return cls([link])

    @classmethod
    def deserialize(cls, regex_data: list):
        root_links = [Link.from_dict(link_data) for link_data in regex_data]
        return cls(root_links)

    def serialize(self):
        return [link._param_dict() for link in self.root_links]

    def draw(self) -> str:
        counts = np.array([link.count for link in self.root_links])
        link = np.random.choice(self.root_links, p=counts/np.sum(counts))
        return link.draw()

    @cached_property
    def root_prob(self):
        counts = np.array([link.count for link in self.root_links])
        return counts/np.sum(counts)

    @cached_property
    def n_param(self):
        return np.sum([link.n_param for link in self.root_links])

    def AIC(self, values):
        tot_log_like = 0
        for val in values:
            log_likes = [link.log_likelihood(val) for link in self.root_links]
            cur_log_like = sum_prob_log(self.root_prob, log_likes)
            if cur_log_like < UNVIABLE_REGEX/2:
                cur_log_like = max(len(val), 1)*LOG_LIKE_PER_CHAR
            tot_log_like += cur_log_like
        return 2*self.n_param - 2*tot_log_like

    # def regex(self) -> str:
    #
