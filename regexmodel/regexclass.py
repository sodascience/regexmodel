from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from typing import Optional, Iterable
import re
from functools import cached_property
from collections import defaultdict

import numpy as np
import polars as pl

from regexmodel.util import Dir


class BaseRegex(ABC):
    prefac: float = 1

    def __init__(self, digit: int = 1):
        self.digit = digit

    @property
    def string(self) -> str:
        return re.sub(r"\\", r"\\\\", self.regex)

    @abstractmethod
    def draw(self):
        pass

    @classmethod
    @abstractmethod
    def fit(cls, series, score_thres, direction: Dir):
        pass

    @abstractmethod
    def fit_value(self, value: str, direction: Dir):
        pass

    @classmethod
    @abstractmethod
    def from_string(cls, string) -> Optional[tuple[BaseRegex, str]]:
        pass

    @property
    @abstractmethod
    def regex(self) -> str:
        pass


class MultiRegex(BaseRegex):
    _base_regex = r"Some invalid rege\x //"
    n_possible = 100

    def __init__(self, min_len: int = 1, max_len: int = 1):
        self.min_len = min_len
        self.max_len = max_len

    @classmethod
    def fit(cls, series, score_thres, direction: Dir) -> tuple[MultiRegex, float, pl.Series]:
        if direction == Dir.LEFT:
            first_elem = series.str.extract(r"[\S\s]*("+cls._base_regex+r"+)$")
            first_char = first_elem.str.extract(r"[\S\s]*([\S\s])$")
        else:
            first_elem = series.str.extract(r"^("+cls._base_regex+r"+).*")
            first_char = first_elem.str.extract(r"^([\S\s])[\S\s]*")
        n_unique = len(first_char.drop_nulls().unique())
        score = n_unique/cls.n_possible * (len(series) - first_elem.null_count())/len(series)
        lengths = first_elem.str.lengths().drop_nulls().to_numpy()
        if len(lengths) == 0:
            lengths = np.array([1])
        unq_len, counts = np.unique(lengths, return_counts=True)
        cum_score = np.cumsum(counts)/len(series)
        try:
            i_min_len = np.where(cum_score >= score_thres)[0][0]
        except IndexError:
            i_min_len = 0
        try:
            i_max_len = np.where(
                cum_score >= len(first_elem.drop_nulls())/len(series)-score_thres)[0][0]
        except IndexError:
            i_max_len = i_min_len
        if i_min_len > i_max_len:
            i_min_len = 0
            i_max_len = len(unq_len)-1
        min_len = unq_len[i_min_len]
        max_len = unq_len[i_max_len]
        instance = cls(min_len, max_len)
        if direction == Dir.LEFT:
            first_elem = series.str.extract(r"[\S\s]*(" + instance.regex + r"})$")
            new_series = series.str.extract(r"([\S\s]*)"+instance.regex+r"$")
        else:
            first_elem = series.str.extract(r"^(" + instance.regex + r"})[\S\s]*")
            new_series = series.str.extract(r"^"+instance.regex+r"([\S\s]*)")

        return instance, score, new_series

    @cached_property
    def center_regexes(self):
        return {
            cur_len: re.compile(r"^" + self._base_regex + r"{" + str(cur_len) + r"}")
            for cur_len in range(self.min_len, self.max_len+1)
        }

    def fit_value(self, value: str, direction: Dir) -> Iterable:
        if direction == Dir.BOTH:
            return self.fit_value_center(value)
        elif direction == Dir.LEFT:
            return self.fit_value_left(value)
        return self.fit_value_right(value)

    def fit_value_center(self, value: str):
        for i_val in range(len(value)):
            for i_len in range(self.min_len, self.max_len+1):
                res = self.center_regexes[i_len].match(value[i_val:])
                if res is not None:
                    yield (value[:i_val], (self.n_possible)**-i_len, value[i_val+i_len:])

    def fit_value_left(self, value: str):
        for i_len in range(self.min_len, self.max_len+1):
            res = self.center_regexes[i_len].match(value[::-1])
            if res is not None:
                yield (value[:-i_len], (self.n_possible)**-i_len)

    def fit_value_right(self, value: str):
        for i_len in range(self.min_len, self.max_len+1):
            res = self.center_regexes[i_len].match(value)
            if res is not None:
                yield (value[i_len:], (self.n_possible)**-i_len)

    @property
    def regex(self):
        return self._base_regex + r"{" + f"{self.min_len},{self.max_len}" + r"}"

    def __repr__(self):
        return self.__class__.__name__ + self.regex

    def draw(self):
        n_draw = np.random.randint(self.min_len, self.max_len+1)
        return "".join(self.draw_once() for _ in range(n_draw))

    @classmethod
    def from_string(cls, string) -> Optional[tuple[MultiRegex, str]]:
        search_regex = r"^" + re.escape(cls._base_regex) + r"(?>{(\d+),(\d+)})?"
        res = re.search(search_regex, string)
        if res is None:
            return None
        if res.groups()[0] is None:
            return cls(), string[len(cls._base_regex):]
        regex_len = len("".join(res.groups())) + len(cls._base_regex) + 3
        return cls(*[int(x) for x in res.groups()]), string[regex_len:]


class UpperRegex(MultiRegex):
    _base_regex = r"[A-Z]"
    prefac = 0.25
    n_possible = 26

    def draw_once(self):
        return random.choice(string.ascii_uppercase)


class LowerRegex(MultiRegex):
    _base_regex = r"[a-z]"
    prefac = 0.25
    n_possible = 26

    def draw_once(self):
        return random.choice(string.ascii_lowercase)


class DigitRegex(MultiRegex):
    _base_regex = r"[0-9]"
    prefac = 0.5
    n_possible = 10

    def draw_once(self):
        return str(np.random.randint(10))


def _check_literal_compatible(unq_char, counts, unq_char_added, counts_added):
    if unq_char is None:
        return True
    tot_counts = np.sum(counts)
    tot_counts_added = np.sum(counts_added)
    orig_dict = {unq_char[i]: counts[i] for i in range(len(counts))}
    add_dict = {unq_char_added[i]: counts_added[i] for i in range(len(counts_added))}

    for char in (set(add_dict) | set(orig_dict)):
        if char in orig_dict:
            expected_count = (orig_dict[char]/tot_counts)*tot_counts_added
            min_bound = round(expected_count-np.sqrt(len(unq_char)*expected_count)-1)
            max_bound = round(expected_count+np.sqrt(len(unq_char)*expected_count)+1)
            if char not in add_dict and min_bound > 0:
                return False
            elif char not in add_dict:
                continue
            if add_dict[char] < min_bound or add_dict[char] > max_bound:
                return False
        else:
            if add_dict[char] >= np.sqrt(len(unq_char)) + 1:
                return False
    return True


def _add_literal_compatible(unq_char, counts, unq_char_added, counts_added):
    if unq_char is None:
        return unq_char_added, counts_added
    data = defaultdict(lambda: 0)
    for i_char, char in enumerate(unq_char):
        data[char] += counts[i_char]
    for i_char, char in enumerate(unq_char_added):
        data[char] += counts_added[i_char]
    return np.array(list(data.keys())), np.array(list(data.values()))


def _unescape(string: str) -> str:
    return re.sub(r'\\\\(.)', r'\1', string)


class LiteralRegex(BaseRegex):
    def __init__(self, literals: list[str]):
        self.literals = list(set(literals))

    @classmethod
    def fit(cls, series, score_thres, direction: Dir):
        if direction == Dir.LEFT:
            first_elem = series.str.extract(r"[\S\s]*([\S\s])$")
            second_elem = series.str.extract(r"[\S\s]*([\S\s])[\S\s]$")
        else:
            first_elem = series.str.extract(r"^([\S\s])[\S\s]*")
            second_elem = series.str.extract(r"^[\S\s]([\S\s])[\S\s]*")
        unq_char, counts = np.unique(first_elem.drop_nulls().to_numpy(), return_counts=True)
        thres_char = unq_char[counts/len(series) >= score_thres]
        thres_count = counts[counts/len(series) >= score_thres]
        cur_tot_counts = None
        cur_tot_chars = None
        use_chars = []
        for i_char in np.argsort(-thres_count):
            cur_char = thres_char[i_char]
            cur_second_elem = second_elem.filter(first_elem == cur_char)
            second_char, second_count = np.unique(cur_second_elem.drop_nulls().to_numpy(),
                                                  return_counts=True)
            if _check_literal_compatible(cur_tot_chars, cur_tot_counts, second_char, second_count):
                cur_tot_chars, cur_tot_counts = _add_literal_compatible(
                    cur_tot_chars, cur_tot_counts, second_char, second_count)
                use_chars.append(cur_char)

        if len(use_chars) == 0:
            use_chars = ["X"]
        instance = cls(use_chars)
        if direction == Dir.LEFT:
            new_series = series.str.extract(r"([\S\s]*)" + instance.regex + r"$")
        else:
            new_series = series.str.extract(r"^" + instance.regex + r"([\S\s]*)")
        score = (1/len(use_chars))*(new_series.drop_nulls().len()/len(series))

        return instance, score, new_series

    # @classmethod
    # def _fit_old(cls, series, score_thres, left=True):
    #     if left:
    #         first_elem = series.str.extract(r"^(.).*").drop_nulls()
    #     else:
    #         first_elem = series.str.extract(r".*(.)$").drop_nulls()

    #     unq_char, counts = np.unique(first_elem.to_numpy(), return_counts=True)
    #     thres_char = unq_char[counts/len(series) >= score_thres]
    #     if len(thres_char) == 0:
    #         thres_char = ["X"]
    #     instance = cls(thres_char)
    #     if left:
    #         new_series = series.str.extract(r"^" + instance.regex + r"(.*)")
    #     else:
    #         new_series = series.str.extract(r"(.*)" + instance.regex + r"$")
    #     score = (3/len(thres_char))*(new_series.drop_nulls().len()/len(series))

    #     return instance, score, new_series

    def __repr__(self):
        return f"Literal [{''.join(self.literals)}]"

    @property
    def regex(self):
        return r"[" + "".join(re.escape(x) if x not in [" "] else x for x in self.literals) + r"]"

    def draw(self):
        return np.random.choice(self.literals)

    def fit_value(self, value: str, direction: Dir) -> Iterable:
        if len(value) == 0:
            return
        if direction == Dir.BOTH:
            for i_val in range(len(value)):
                if value[i_val] in self.literals:
                    yield (value[:i_val], 1/len(self.literals), value[i_val+1:])
        elif direction == Dir.LEFT:
            if value[-1] in self.literals:
                yield (value[:-1], 1/len(self.literals))
        elif value[0] in self.literals:
            yield (value[1:], 1/len(self.literals))

    @classmethod
    def from_string(cls, string) -> Optional[tuple[LiteralRegex, str]]:
        # Find first unescaped ']'
        end_index = None
        for match in re.finditer(r"\]", string):
            start = match.span()[0]-2
            if start < 0:
                continue
            if string[start:start+3] != r"\\]":
                end_index = start+3
                break
        if end_index is None or string[0] != r"[":
            return None

        literals = list(_unescape(string[1:end_index-1]))
        return cls(literals), string[end_index:]


def regex_list_from_string(string) -> list[BaseRegex]:
    cur_data = string
    all_regexes = []
    all_regex_class: list[type[BaseRegex]] = [UpperRegex, LowerRegex, DigitRegex, LiteralRegex]
    while len(cur_data) > 0:
        found = False
        for regex_class in all_regex_class:
            res = regex_class.from_string(cur_data)
            if res is not None:
                found = True
                all_regexes.append(res[0])
                cur_data = res[1]
                break
        if not found:
            raise ValueError(f"Invalid regex: '{string}', at '{cur_data}'")
    return all_regexes
