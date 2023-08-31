"""Regex classes that are the individual nodes of the regex model."""
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
from numpy import typing as npt

from regexmodel.util import Dir


class BaseRegex(ABC):
    """Base regex element."""

    prefac: float = 1

    @property
    def string(self) -> str:
        """Create a string out of the regex (used for serialization)."""
        return re.sub(r"\\", r"\\\\", self.regex)

    @abstractmethod
    def draw(self) -> str:
        """Draw a string from the regex element."""

    @classmethod
    @abstractmethod
    def fit(cls, series, score_thres, direction: Dir):
        """Fit the regex class to a series of structured strings.

        Parameters
        ----------
        series:
            Series of structured strings.
        score_thres:
            Threshold for the fitting score.
        direction:
            Fit from the start of the strings if Dir.BOTH or Dir.RIGHT, fit from the end
            if direction is Dir.LEFT.
        """

    @abstractmethod
    def fit_value(self, value: str, direction: Dir) -> Iterable:
        """Try all possible ways to fit the regex to a single value.

        It yields a tuple of 3 in the case of direction == Dir.BOTH:
            start_string, probability of generation, end_string
        else:
            leftover string, probability of generation.
        For example if we have the value 12394X, and [0-9]{3,4}, and direction==Dir.BOTH,
        then we will get: ("", 0.5*10^-3, "94X"), ("", 0.5*10^-4, "4X"), etc.
        """

    @classmethod
    @abstractmethod
    def from_string(cls, regex_str) -> Optional[tuple[BaseRegex, str]]:
        """Create the regex class from a serialized string.

        It will return None if the string does not belong to the regex class.
        """

    @property
    @abstractmethod
    def regex(self) -> str:
        """Create a regex to be used for extracting it from a structured string."""


class MultiRegex(BaseRegex, ABC):
    """Base class for regex classes that have multiple repeating elements.

    Examples: [0-9]{3,4}, [A-Z]{1,2}

    Parameters
    ----------
    min_len:
        Minimum length of the number of repeating elements.
    max_len:
        Maximum length of the number of repeating elements.
    """

    _base_regex = r"Some invalid rege\x //"
    n_possible = 100

    def __init__(self, min_len: int = 1, max_len: int = 1):
        self.min_len = min_len
        self.max_len = max_len

    @classmethod
    def fit(cls, series, score_thres, direction: Dir) -> tuple[MultiRegex, float, pl.Series]:
        if direction == Dir.LEFT:
            first_elem = series.str.extract(r"(?:[\S\s]*?)*("+cls._base_regex+r"+)$")
            first_char = first_elem.str.extract(r"[\S\s]*([\S\s])$")
        else:
            first_elem = series.str.extract(r"^("+cls._base_regex+r"+)[\S\s]*")
            first_char = first_elem.str.extract(r"^([\S\s])[\S\s]*")
        n_unique = len(first_char.drop_nulls().unique())
        # Score is dependent on number of unique values of first character.
        score = n_unique/cls.n_possible * (len(series) - first_elem.null_count())/len(series)

        # Calculate the min_len and max_len using the score_thres.
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

        # Create an instance and return the score and the resulting series.
        instance = cls(min_len, max_len)
        if direction == Dir.LEFT:
            new_series = series.str.extract(r"([\S\s]*?)"+instance.regex+r"$")
        else:
            new_series = series.str.extract(r"^"+instance.regex+r"([\S\s]*)")

        return instance, score, new_series

    @cached_property
    def _center_regexes(self):
        """A dictionary for compiled regexes for all possible lengths."""
        return {
            cur_len: re.compile(r"^" + self._base_regex + r"{" + str(cur_len) + r"}")
            for cur_len in range(self.min_len, self.max_len+1)
        }

    def fit_value(self, value: str, direction: Dir) -> Iterable:
        if direction == Dir.BOTH:
            return self._fit_value_center(value)
        if direction == Dir.LEFT:
            return self._fit_value_left(value)
        return self._fit_value_right(value)

    def _fit_value_center(self, value: str):
        """Fit value in case direction == Dir.BOTH."""
        for i_val in range(len(value)):
            for i_len in range(self.min_len, self.max_len+1):
                res = self._center_regexes[i_len].match(value[i_val:])
                if res is not None:
                    yield (value[:i_val], self._draw_probability(i_len), value[i_val+i_len:])

    def _fit_value_left(self, value: str):
        """Fit value in case direction == Dir.LEFT."""
        for i_len in range(self.min_len, self.max_len+1):
            res = self._center_regexes[i_len].match(value[::-1])
            if res is not None:
                yield (value[:-i_len], self._draw_probability(i_len))

    def _fit_value_right(self, value: str):
        """Fit value in case direction == Dir.RIGHT."""
        for i_len in range(self.min_len, self.max_len+1):
            res = self._center_regexes[i_len].match(value)
            if res is not None:
                yield (value[i_len:], self._draw_probability(i_len))

    def _draw_probability(self, digit_len):
        """Probability to draw a certain string with length digit_len."""
        return self.n_possible**-digit_len/(self.max_len-self.min_len+1)

    @property
    def regex(self):
        """Regex for retrieving elements with the regex class."""
        return self._base_regex + r"{" + f"{self.min_len},{self.max_len}" + r"}"

    def __repr__(self):
        return self.__class__.__name__ + self.regex

    def draw(self):
        n_draw = np.random.randint(self.min_len, self.max_len+1)
        return "".join(self.draw_once() for _ in range(n_draw))

    @abstractmethod
    def draw_once(self) -> str:
        """Draw a single character from the multi regex."""

    @classmethod
    def from_string(cls, regex_str) -> Optional[tuple[MultiRegex, str]]:
        # It used to be, but changed for compatibility: r"(?>{(\d+),(\d+)})?"
        search_regex = r"^" + re.escape(cls._base_regex) + r"(?:{(\d+),(\d+)})?"
        res = re.search(search_regex, regex_str)
        if res is None:
            return None
        if res.groups()[0] is None:
            return cls(), regex_str[len(cls._base_regex):]
        regex_len = len("".join(res.groups())) + len(cls._base_regex) + 3
        return cls(*[int(x) for x in res.groups()]), regex_str[regex_len:]


class UpperRegex(MultiRegex):
    """Regex class that produces upper case characters."""

    _base_regex = r"[A-Z]"
    prefac = 0.25
    n_possible = 26

    def draw_once(self):
        return random.choice(string.ascii_uppercase)


class LowerRegex(MultiRegex):
    """Regex class that produces lower case characters."""

    _base_regex = r"[a-z]"
    prefac = 0.25
    n_possible = 26

    def draw_once(self):
        return random.choice(string.ascii_lowercase)


class DigitRegex(MultiRegex):
    """Regex class that produces digits."""

    _base_regex = r"[0-9]"
    prefac = 0.5
    n_possible = 10

    def draw_once(self):
        return str(np.random.randint(10))


def _check_literal_compatible(unq_char: Optional[str],
                              counts: Optional[npt.NDArray[np.int_]],
                              unq_char_added: npt.NDArray[np.string_],
                              counts_added: npt.NDArray[np.int_]):
    """Check whether to add a new literal to the current set of literals.

    It does so by looking ahead at the character after the current one.
    If the distribution of second characters for the current set of literals
    is similar enough to the distribution for the candidate literal, then return
    True, otherwise return false.

    This heuristic could use some study/work.
    """
    if unq_char is None or counts is None:
        return True
    tot_counts = np.sum(counts)
    tot_counts_added = np.sum(counts_added)
    # Convert both counts/characters to dictionary.
    orig_dict = {unq_char[i]: counts[i] for i in range(len(counts))}
    add_dict = {unq_char_added[i]: counts_added[i] for i in range(len(counts_added))}

    for char in (set(add_dict) | set(orig_dict)):
        if char in orig_dict:
            # Compute the expected count and bounds for the candidate
            expected_count = (orig_dict[char]/tot_counts)*tot_counts_added
            min_bound = round(expected_count-np.sqrt(len(unq_char)*expected_count)-1)
            max_bound = round(expected_count+np.sqrt(len(unq_char)*expected_count)+1)
            if char not in add_dict and min_bound > 0:
                return False
            if char not in add_dict:
                continue
            if add_dict[char] < min_bound or add_dict[char] > max_bound:
                return False
        else:
            # A new character should not have too high a count.
            if add_dict[char] >= np.sqrt(len(unq_char)) + 1:
                return False
    return True


def _add_literal_compatible(unq_char: Optional[str],
                            counts: Optional[npt.NDArray[np.int_]],
                            unq_char_added: npt.NDArray[np.string_],
                            counts_added: npt.NDArray[np.int_]):
    """Merge the characters and counts of multiple classes."""
    if unq_char is None or counts is None:
        return unq_char_added, counts_added
    data: dict[str, int] = defaultdict(lambda: 0)
    for i_char, char in enumerate(unq_char):
        data[char] += counts[i_char]
    for i_char, char in enumerate(unq_char_added):
        data[char] += counts_added[i_char]
    return np.array(list(data.keys())), np.array(list(data.values()))


def _unescape(regex_str: str) -> str:
    """Remove slashes that are in the serialized string."""
    return re.sub(r'\\\\(.)', r'\1', regex_str)


class LiteralRegex(BaseRegex):
    """Regex class that generates literals.

    Parameters
    ----------
    literals:
        List of all character that can be created using this regex class.
        This is not limited to ASCII characters.
    """

    def __init__(self, literals: list[str]):
        self.literals = list(set(literals))

    @classmethod
    def fit(cls, series, score_thres, direction: Dir):
        # Get the first and second characters.
        if direction == Dir.LEFT:
            first_elem = series.str.extract(r"[\S\s]*([\S\s])$")
            second_elem = series.str.extract(r"[\S\s]*([\S\s])[\S\s]$")
        else:
            first_elem = series.str.extract(r"^([\S\s])[\S\s]*")
            second_elem = series.str.extract(r"^[\S\s]([\S\s])[\S\s]*")

        # Find out which characters come up often and meet the score threshold.
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

        # Create the instance and return the score/resulting series.
        if len(use_chars) == 0:
            use_chars = ["X"]
        instance = cls(use_chars)
        if direction == Dir.LEFT:
            new_series = series.str.extract(r"([\S\s]*)" + instance.regex + r"$")
        else:
            new_series = series.str.extract(r"^" + instance.regex + r"([\S\s]*)")
        score = (1/len(use_chars))*(new_series.drop_nulls().len()/len(series))

        return instance, score, new_series

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
            for i_val, cur_char in enumerate(value):
                if cur_char in self.literals:
                    yield (value[:i_val], 1/len(self.literals), value[i_val+1:])
        elif direction == Dir.LEFT:
            if value[-1] in self.literals:
                yield (value[:-1], 1/len(self.literals))
        elif value[0] in self.literals:
            yield (value[1:], 1/len(self.literals))

    @classmethod
    def from_string(cls, regex_str) -> Optional[tuple[LiteralRegex, str]]:
        # Find first unescaped ']'
        end_index = None
        for match in re.finditer(r"\]", regex_str):
            start = match.span()[0]-2
            if start < 0:
                continue
            if regex_str[start:start+3] != r"\\]":
                end_index = start+3
                break
        if end_index is None or regex_str[0] != r"[":
            return None

        literals = list(_unescape(regex_str[1:end_index-1]))
        return cls(literals), regex_str[end_index:]


# All regex classes available, order matters.
ALL_REGEX_CLASS: list[type[BaseRegex]] = [UpperRegex, LowerRegex, DigitRegex, LiteralRegex]


def regex_list_from_string(regex_str) -> list[BaseRegex]:
    """Convert a regex string to a list of regex classes.

    Parameters
    ----------
    regex_str:
        String representation of the regex list, e.g. r'[0-9]{3,4}[A-Z]'

    Returns
    -------
        List of regex classes represented by the regex string.
    """
    cur_data = regex_str
    all_regexes = []
    # loop until we have parsed the whole string.
    while len(cur_data) > 0:
        found = False
        for regex_class in ALL_REGEX_CLASS:
            res = regex_class.from_string(cur_data)
            if res is not None:
                found = True
                all_regexes.append(res[0])
                cur_data = res[1]
                break
        if not found:
            raise ValueError(f"Invalid regex: '{regex_str}', at '{cur_data}'")
    return all_regexes


def fit_best_regex_class(series: pl.Series, score_thres: float,
                         direction: Dir = Dir.RIGHT) -> dict:
    """Find the optimal regex class for this series.

    Parameters
    ----------
    series:
        Series to fit regex classes for.
    score_thres:
        Score threshold for selection.
    direction:
        Direction of the fit.

    Returns
    -------
        Dictionary with score, regex, and new_series.
    """
    best_regex = {"score": -1}
    for regex_class in ALL_REGEX_CLASS:
        regex_inst, score, new_series = regex_class.fit(
            series, score_thres=score_thres, direction=direction)
        if score > best_regex["score"]:
            best_regex = {
                "score": score,
                "regex": regex_inst,
                "new_series": new_series,
            }
    return best_regex
