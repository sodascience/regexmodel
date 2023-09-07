"""Regex classes that are the individual nodes of the regex model."""
from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from typing import Optional, Iterable, Any
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

    def is_covered(self, regex):    
        return False

    def extract_first_elem(self, series) -> str:
        return series.str.extract(r"^(" + self.regex + r")[\S\s]*")

    def extract_after_first(self, series) -> str:
        print(self.regex)
        return series.str.extract(r"^" + self.regex + r"([\S\s]*)$")


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

    def covers(self, regex):
        if isinstance(regex, LiteralRegex):
            for literal in regex.literals:
                if literal not in self.all_possible:
                    return False
            return True
        return False

    @classmethod
    def score_single(cls, series, count_thres):
        next_series = series.str.extract(r"^[" + cls._base_regex + "]([\S\s]*)$")
        first_char = series.str.extract(r"^([" + cls._base_regex + "])[\S\s]*$")
        n_not_null = len(series) - next_series.null_count()
        n_unique = len(first_char.drop_nulls().unique())
        avg_len = next_series.drop_nulls().str.lengths().mean()
        if avg_len is None:
            split_penalty = 1
        else:
            expected_finish = 0.7**avg_len*n_not_null
            split_penalty = 1/(1 + np.exp(2*(count_thres - expected_finish)/count_thres))
        return split_penalty*n_unique/cls.n_possible*n_not_null/len(series), next_series

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
            cur_len: re.compile(r"^[" + self._base_regex + r"]{" + str(cur_len) + r"}")
            for cur_len in range(self.min_len, self.max_len+1)
        }

    def fit_value(self, value: str, direction: Dir) -> Iterable:
        for i_len in range(self.min_len, self.max_len+1):
            res = self._center_regexes[i_len].match(value)
            if res is not None:
                yield (value[i_len:], self._draw_probability(i_len))

    def _draw_probability(self, digit_len):
        """Probability to draw a certain string with length digit_len."""
        return float(self.n_possible)**-digit_len/(self.max_len-self.min_len+1)

    @property
    def regex(self):
        """Regex for retrieving elements with the regex class."""
        return r"[" + self._base_regex + r"]{" + f"{self.min_len},{self.max_len}" + r"}"

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

    _base_regex = r"A-Z"
    prefac = 0.25
    n_possible = 26
    all_possible = string.ascii_uppercase

    def draw_once(self):
        return random.choice(string.ascii_uppercase)


class LowerRegex(MultiRegex):
    """Regex class that produces lower case characters."""

    _base_regex = r"a-z"
    prefac = 0.25
    n_possible = 26
    all_possible = string.ascii_lowercase

    def draw_once(self):
        return random.choice(string.ascii_lowercase)


class DigitRegex(MultiRegex):
    """Regex class that produces digits."""

    _base_regex = r"0-9"
    prefac = 0.5
    n_possible = 10
    all_possible = string.digits

    def draw_once(self):
        return str(np.random.randint(10))


def get_class_stat(series, count_thres):
    score_list = []
    for rclass in [UpperRegex, LowerRegex, DigitRegex]:
        cur_class_stat = score_single(rclass._base_regex, series, count_thres, rclass.n_possible)
        if cur_class_stat[0] > 0 and cur_class_stat[1].drop_nulls().len() > count_thres:
            score_list.append((rclass(), *cur_class_stat))
    score_list.extend(LiteralRegex.get_candidates(series, count_thres))
    return sorted(score_list, key=lambda res: -res[1])


def _get_bounds(key, count_dict, count_thres, sigma, tot_counts):
    if key in count_dict:
        lower_bound = (count_dict[key] - sigma*np.sqrt(count_thres))/tot_counts
        upper_bound = (count_dict[key] + sigma*np.sqrt(count_thres))/tot_counts
    else:
        lower_bound = 0
        upper_bound = (count_thres + sigma*np.sqrt(count_thres))/tot_counts
    return lower_bound, upper_bound


def _check_stat_compatible(stat_a, tot_a, stat_b, tot_b, count_thres, sigma=3):
    dict_stat_a = {regex._base_regex: len(series)-series.null_count()
                   for regex, score, series in stat_a}
    dict_stat_b = {regex._base_regex: len(series)-series.null_count()
                   for regex, score, series in stat_b}
    # tot_a = np.sum([x for x in dict_stat_a.values()])
    # tot_b = np.sum([x for x in dict_stat_b.values()])
    for key in (set(dict_stat_a) | set(dict_stat_b)):
        lower_a, upper_a = _get_bounds(key, dict_stat_a, count_thres, sigma, tot_a)
        lower_b, upper_b = _get_bounds(key, dict_stat_b, count_thres, sigma, tot_b)
        print(key, lower_a, upper_a, lower_b, upper_b, lower_a > upper_b or lower_b > upper_a)
        if lower_a > upper_b or lower_b > upper_a:
            print({key: a/tot_a for key, a in dict_stat_a.items()})
            print({key: b/tot_b for key, b in dict_stat_b.items()})
            print(dict_stat_a)
            print(dict_stat_b)
            return False
    return True


class OrRegex(MultiRegex):
    def __init__(self, regex_instances, min_len=1, max_len=1):
        self._regex_instances = list(regex_instances)
        super().__init__(min_len, max_len)

    @property
    def n_possible(self):
        return np.sum([rc.n_possible for rc in self._regex_instances])

    @property
    def _base_regex(self):
        return "".join(rc._base_regex for rc in self._regex_instances)

    @property
    def regex(self):
        if self.min_len == 1 and self.max_len == 1:
            return r"[" + self._base_regex + r"]"
        return r"[" + self._base_regex + r"]{" + str(self.min_len) + "," + str(self.max_len) + "}"

    def draw_once(self):
        prob = np.array([inst.n_possible for inst in self._regex_instances])
        prob = prob/np.sum(prob)
        chosen_inst = np.random.choice(self._regex_instances, p=prob)
        return chosen_inst.draw_once()

    def covers(self, regex):
        for inst in self._regex_instances:
            if inst.covers(regex):
                return True
        return False

    def append(self, regex):
        self._regex_instances.append(regex)

    def check_compatibility(self, series: pl.Series, count_thres: int):
        if len(self._regex_instances) == 1:
            return True
        base_regex = self._regex_instances[0]
        base_series = base_regex.extract_after_first(series)
        base_stat = get_class_stat(base_series, count_thres)
        for cur_regex in self._regex_instances[1:]:
            cur_new_series = cur_regex.extract_after_first(series)
            cur_class_stat = get_class_stat(cur_new_series, count_thres)
            base_tot = len(series) - base_series.null_count()
            cur_tot = len(series) - cur_new_series.null_count()
            compatible = _check_stat_compatible(base_stat, base_tot, cur_class_stat, cur_tot,
                                                count_thres)
            if not compatible:
                return False
        return True


def score_single(regex, series, count_thres, n_possible):
    next_series = series.str.extract(r"^[" + regex + r"]([\S\s]*)$")
    first_char = series.str.extract(r"^([" + regex + r"])[\S\s]*$")
    n_not_null = len(series) - next_series.null_count()
    n_unique = len(first_char.drop_nulls().unique())
    avg_len = next_series.drop_nulls().str.lengths().mean()
    if avg_len is None:
        split_penalty = 1
    else:
        expected_finish = 0.7**avg_len*n_not_null
        split_penalty = 1/(1 + np.exp(2*(count_thres - expected_finish)/count_thres))
    return split_penalty*n_unique/n_possible*n_not_null/len(series), next_series


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

    @property
    def _base_regex(self):
        return "".join(self.literals)

    @property
    def n_possible(self):
        return len(self.literals)

    def covers(self, regex):
        return False

    @classmethod
    def get_candidates(cls, series, count_thres):
        first_elem = series.str.extract(r"^([\S\s])[\S\s]*")
        unq_char, counts = np.unique(first_elem.drop_nulls().to_numpy(), return_counts=True)
        thres_char = unq_char[counts >= count_thres]
        # thres_count = counts[counts >= count_thres]
        for cur_char in thres_char:
            next_series = series.str.extract(r"^" + cls(cur_char).regex + r"([\S\s]*)")
            # print(r"^" + cls(cur_char).regex + r"[\S\s]*")
            avg_len = next_series.drop_nulls().str.lengths().mean()

            n_not_null = len(series) - next_series.null_count()
            avg_len = next_series.drop_nulls().str.lengths().mean()
            if avg_len is None:
                split_penalty = 1
            else:
                expected_finish = 0.7**avg_len*n_not_null
                # print(expected_finish)
                split_penalty = 1/(1 + np.exp(2*(count_thres - expected_finish)/count_thres))
            # print(cur_char, split_penalty)
            yield (cls(cur_char), split_penalty*n_not_null/len(series), next_series)

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

    def draw_once(self):
        return self.draw()

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


def _preview(series):
    return series.drop_nulls()[:3].to_numpy()


def fit_best_regex_class(series: pl.Series, count_thres: int) -> Optional[dict]:
    class_stat = get_class_stat(series, count_thres)
    if len(class_stat) == 0:
        return None
    cur_best_regex, best_score, cur_best_series = class_stat[0]
    cur_best_regex = OrRegex([cur_best_regex])
    cur_series = series.set(cur_best_series.is_not_null(), None)  # type: ignore

    print(_preview(cur_series))
    print({rex: score for rex, score, _ in class_stat})
    for cur_regex, score, _ in class_stat[1:]:
        print("------", cur_regex.regex)
        if cur_best_regex.covers(cur_regex):
            print("covered")
            continue
        
        new_cur_series = cur_regex.extract_after_first(cur_series)
        # series.str.extract(r"^[^(" + cur_best_regex._base_regex + r")"
                                        # + cur_regex._base_regex + r"]([\S\s]*)")
        # print(r"^[^(" + cur_best_regex._base_regex + r")"
                                        # + cur_regex._base_regex + r"]([\S\s]*)")
        cur_non_null = len(series) - new_cur_series.null_count()
        if cur_non_null < count_thres:
            print("Failed threshold", cur_non_null)
            continue
        cur_new_class_stat = get_class_stat(new_cur_series, count_thres)
        best_new_class_stat = get_class_stat(cur_best_series, count_thres)

        best_count = len(series) - cur_best_series.null_count()
        cur_count = len(series) - new_cur_series.null_count()
        compatible = _check_stat_compatible(best_new_class_stat, best_count,
                                            cur_new_class_stat, cur_count, count_thres)
        if compatible:
            print("Compatible")
            cur_best_regex.append(cur_regex)
        cur_series = cur_series.set(new_cur_series.is_not_null(), None)  # type: ignore
        print(_preview(cur_series))

    cur_series = series
    cur_series = cur_best_regex.extract_after_first(series)
    print(series.str.lengths().mean(), cur_series.str.lengths().mean())
    start_non_null = len(series) - cur_series.null_count()
    i_start = None
    i_end = None
    for cur_i in range(1, 100):

        compatible = cur_best_regex.check_compatibility(cur_series, count_thres)
        cur_non_null = len(cur_series) - cur_series.null_count()
        if not compatible or cur_non_null < count_thres:
            print(compatible, cur_non_null, count_thres)
            if i_start is None:
                i_start = cur_i-1
            i_end = cur_i-1
            break

        if cur_non_null < start_non_null - count_thres and i_start is None:
            i_start = cur_i-1
        cur_series = cur_best_regex.extract_after_first(cur_series)

    if i_end is None:
        i_end = 100

    cur_best_regex.min_len = i_start
    cur_best_regex.max_len = i_end

    new_series = cur_best_regex.extract_after_first(series)
    alt_series = series.set(new_series.is_not_null(), None)  # type: ignore
    print("return", cur_best_regex.regex, best_score, count_thres/len(series))
    return {
        "score": best_score,
        "regex": cur_best_regex,
        "new_series": new_series,
        "alt_series": alt_series,
    }
