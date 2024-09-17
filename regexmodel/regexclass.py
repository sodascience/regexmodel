"""Regex classes that are the individual nodes of the regex model."""
from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from typing import Optional, Iterable
import re
from functools import cached_property

import numpy as np
import polars as pl


class BaseRegex(ABC):
    """Base class for regex classes that have multiple repeating elements.

    Examples: [0-9]{3,4}, [A-Z]{1,2}

    Parameters
    ----------
    min_len:
        Minimum length of the number of repeating elements.
    max_len:
        Maximum length of the number of repeating elements.
    """

    def __init__(self, min_len: int = 1, max_len: int = 1):
        self.min_len = min_len
        self.max_len = max_len

    prefac: float = 1

    @property
    def string(self) -> str:
        """Create a string out of the regex (used for serialization)."""
        return re.sub(r"\\", r"\\\\", self.regex)

    @classmethod
    @abstractmethod
    def from_string(cls, regex_str: str) -> Optional[tuple[BaseRegex, str]]:
        """Create the regex class from a serialized string.

        It will return None if the string does not belong to the regex class.
        """

    @property
    @abstractmethod
    def base_regex(self) -> str:
        """Base regex that returns the part inside the brackets.

        For example 'A-Z' in the case of an uppercase regex.
        """

    @classmethod
    @abstractmethod
    def get_candidates(cls, series: pl.Series, count_thres: int
                       ) -> Iterable[tuple[BaseRegex, float, pl.Series]]:
        """Return a list of candidate regexes for a given series.

        Parameters
        ----------
        series:
            Series to create the candidates for.
        count_thres:
            Threshold for creating candidates. No candidates will be created that
            have fewer matches with the series than this.

        Returns
        -------
        regex:
            Regex that matches the start of the series
        score:
            How well the regex matches the series.
        next_series:
            What is left after matching the series with the regex.
        """

    @staticmethod
    def get_class_length(regex_str: str):
        """Get the min and max length of the regex string."""
        res = re.search(r"(?:{(\d+),(\d+)})?", regex_str)
        if res is None:  # should never fire.
            raise ValueError("Cannot get class length.")
        if res.groups()[0] is None:
            res = re.search(r"^{(\d+)}", regex_str)
            if res is None:
                return (1, 1), regex_str
            all_len = res.groups()[0]
            return (int(all_len), int(all_len)), regex_str[len(all_len)+2:]
        regex_len = len("".join(res.groups())) + 3
        return (int(x) for x in res.groups()), regex_str[regex_len:]

    def draw(self):
        """Draw a string from the regex element."""
        n_draw = np.random.randint(self.min_len, self.max_len+1)
        return "".join(self.draw_once() for _ in range(n_draw))

    @property
    def regex(self):
        """Regex for retrieving elements with the regex class."""
        if isinstance(self, LiteralRegex):
            regex_start = self.base_regex
        else:
            regex_start = r"[" + self.base_regex + r"]"

        if self.min_len == 1 and self.max_len == 1:
            return regex_start
        if self.min_len == self.max_len:
            return regex_start + r"{" + f"{self.min_len}" + r"}"
        return regex_start + r"{" + f"{self.min_len},{self.max_len}" + "}"

    @abstractmethod
    def covers(self, regex: BaseRegex) -> bool:
        """Check whether self covers another regex.

        For example, [A-Z] covers [S], but [0-9] does not cover [a].
        """

    def extract_first_elem(self, series: pl.Series) -> pl.Series:
        """Get the values that are matched to the regex."""
        return series.str.extract(r"^(" + self.regex + r")[\S\s]*")

    def extract_after_first(self, series) -> pl.Series:
        """Get a new series after fitting the regex."""
        return series.str.extract(r"^" + self.regex + r"([\S\s]*)$")

    def __repr__(self):
        return self.__class__.__name__ + self.regex

    @abstractmethod
    def draw_once(self) -> str:
        """Draw a single character from the multi regex."""

    @property
    def n_param(self) -> int:
        """Number of parameters for the regex class."""
        return 1

    def _draw_probability(self, digit_len: int):
        """Probability to draw a certain string with length digit_len."""
        return float(self.n_possible)**-digit_len/(self.max_len-self.min_len+1)

    @cached_property
    def _center_regexes(self) -> dict:
        """A dictionary for compiled regexes for all possible lengths."""
        return {
            cur_len: re.compile(r"^[" + self.base_regex + r"]{" + str(cur_len) + r"}")
            for cur_len in range(self.min_len, self.max_len+1)
        }

    def fit_value(self, value: str) -> Iterable:
        """Try all possible ways to fit the regex to a single value.

        It yields a tuple of the new value and the probabilities.
        For example if we have the value 12394X, and [0-9]{3,4},
        then we will get: (0.5*10^-3, "94X"), (0.5*10^-4, "4X"), etc.
        """
        for i_len in range(self.min_len, self.max_len+1):
            res = self._center_regexes[i_len].match(value)
            if res is not None:
                yield (value[i_len:], self._draw_probability(i_len))

    @property
    def subrange_penalty(self) -> float:
        """Used to score the match.

        It is equivalent to the number of possibilties, but with modifiers.
        This is used in the character ranges, where the range is not maximal
        to give them a lower score. Thus, [A-Z] has generally a higher score than
        [A-Y], even if there is no Z present in the dataset.
        """
        return 1

    @property
    @abstractmethod
    def n_possible(self) -> int:
        """Get the number of possibilities of this regex."""


class CharRangeRegex(BaseRegex, ABC):
    """Base class for regex classes that have multiple repeating elements.

    Examples: [0-9]{3,4}, [A-Z]{1,2}, [b-d]{1,2}

    Parameters
    ----------
    min_len:
        Minimum length of the number of repeating elements.
    max_len:
        Maximum length of the number of repeating elements.
    """

    all_possible = ""

    def __init__(self, min_len: int = 1,
                 max_len: int = 1,
                 char_range: Optional[str] = None):
        if char_range is None:
            char_range = self.all_possible
        self.char_range = char_range
        super().__init__(min_len, max_len)

    @property
    def base_regex(self) -> str:
        return f"{self.char_range[0]}-{self.char_range[-1]}"

    def draw_once(self):
        return random.choice(self.char_range)

    @property
    def subrange_penalty(self) -> float:
        if self.char_range == self.all_possible:
            return super().subrange_penalty
        return 0.75

    def covers(self, regex: BaseRegex) -> bool:
        if isinstance(regex, LiteralRegex):
            for literal in regex.literals:
                if literal not in self.all_possible:
                    return False
            return True
        return False

    @property
    def n_possible(self):
        return len(self.char_range)

    @classmethod
    def from_string(cls, regex_str) -> Optional[tuple[BaseRegex, str]]:
        if len(regex_str) < 3:
            return None
        if (regex_str[0] in cls.all_possible
                and regex_str[2] in cls.all_possible
                and regex_str[1] == "-"):
            start_index = cls.all_possible.index(regex_str[0])
            end_index = cls.all_possible.index(regex_str[2])
            if start_index < end_index:
                return cls(char_range=cls.all_possible[start_index:end_index+1]), regex_str[3:]
        return None

    @classmethod
    def get_candidates(cls, series: pl.Series,
                       count_thres: int) -> Iterable[tuple[BaseRegex, float, pl.Series]]:
        """Get the score for the [A-Z, a-z, 0-9] regex classes."""
        score_full, next_series_full, first_char_full = score(series, cls(), count_thres)

        # Compute with sub range
        unique_chars, counts = np.unique(first_char_full.drop_nulls().to_numpy(),
                                         return_counts=True)
        thres_idx = np.where(counts >= count_thres)[0]
        if len(thres_idx) > 1:
            start_idx = thres_idx[0]
            end_idx = thres_idx[-1]
            start_char, end_char = unique_chars[start_idx], unique_chars[end_idx]
            ret = cls.from_string(f"{start_char}-{end_char}")
            assert ret is not None
            regex = ret[0]
            score_sub, next_series_sub, _first_char_sub = score(series, regex, count_thres)
        else:
            score_sub = 0
            regex = cls()
            next_series_sub = next_series_full

        if score_full >= score_sub:
            if score_full > 0:
                yield (cls(), score_full, next_series_full)
        else:
            yield (regex, score_sub, next_series_sub)


def score(series: pl.Series, regex: BaseRegex, count_thres: int,
          first_char: Optional[pl.Series] = None
          ) -> tuple[float, pl.Series, pl.Series]:
    """Compute the score of matching the regex.

    This attempts to create a universal measure of how good the regex fits the
    series. It consists of:
    - split_penalty: Some values in the series will not be match, and if this
        happens too often, no match will ever be found. This attempts to take this
        into account and will be lower if there is a higher likelihood of failing the
        to fit.
    - fraction_cover: For character classes, there are characters that are not matched.
        for example: [A-Z], but B is not the start of any of the series. In this case,
        there is a penalty for that.
    - fraction_match: Fraction of values matched. The higher the better.
    - subrange_penalty: Penalty for using subranges. We prefer [A-Z] over [A-Y] generally,
        even if Z is not present in the samples.

    The score is then the product of all those values. It is normalized by the length of
    the series to keep it < 1.

    Parameters
    ----------
    series:
        Series to be matched.
    regex:
        Regex that will match the series.
    count_thres:
        Threshold for allowing regexes to be used.
    first_char:
        Hack that is used to keep it from being recomputed in the case of the LiteralRegex.

    Returns
    -------
    score:
        Score that measures the goodness of fit.
    next_series:
        Series that is left over after matching the regex.
    first_char:
        Extracted match from the regex.
    """
    if first_char is None:
        first_char = series.str.extract(r"^([" + regex.base_regex + r"])[\S\s]*$")
    next_series = series.str.extract(r"^[" + regex.base_regex + r"]([\S\s]*)$")
    next_not_null = len(series) - next_series.null_count()
    cur_not_null = len(series) - series.null_count()
    if isinstance(regex, LiteralRegex):
        n_unique = regex.n_possible
    else:
        n_unique = len(first_char.drop_nulls().unique())
    avg_len_next = next_series.drop_nulls().str.len_chars().mean()
    if (next_not_null == 0 or next_not_null < count_thres or avg_len_next is None):
        return 0, next_series, first_char
    fraction_match = next_not_null/cur_not_null
    fraction_cover = n_unique/regex.n_possible

    expected_finish = fraction_match**avg_len_next*next_not_null  # type: ignore
    expected_finish = max(1e-12, expected_finish)
    split_penalty = 1/(1 + count_thres/expected_finish)
    cur_score = regex.subrange_penalty*split_penalty*fraction_cover*fraction_match
    return cur_score, next_series, first_char


class UpperRegex(CharRangeRegex):
    """Regex class that produces upper case characters."""

    _base_regex = r"A-Z"
    all_possible = string.ascii_uppercase


class LowerRegex(CharRangeRegex):
    """Regex class that produces lower case characters."""

    _base_regex = r"a-z"
    all_possible = string.ascii_lowercase


class DigitRegex(CharRangeRegex):
    """Regex class that produces digits."""

    _base_regex = r"0-9"
    all_possible = string.digits


def _unescape(regex_str: str) -> str:
    """Remove slashes that are in the serialized string."""
    return re.sub(r'\\\\(.)', r'\1', regex_str)


class LiteralRegex(BaseRegex):
    """Regex class that generates literals.

    Parameters
    ----------
    literals:
        List of all character that can be created using this regex class.
        This is not limited to ASCII characters. These literals are not escaped.
    """

    def __init__(self, literals: list[str], min_len: int = 1, max_len: int = 1):
        self.literals = list(set(literals))
        super().__init__(min_len, max_len)

    @property
    def base_regex(self):
        """Base regex to be used for regex matching."""
        return "".join(re.escape(x) if x not in [" "] else x for x in self.literals)

    @property
    def n_possible(self):
        """Number of possibilities."""
        return len(self.literals)

    def covers(self, regex):
        return False

    @classmethod
    def get_candidates(cls, series: pl.Series, count_thres: int
                       ) -> Iterable[tuple[LiteralRegex, float, pl.Series]]:
        """Get all possible candidate literal regexes.

        Arguments
        ---------
        series:
            Series to create candidates for.
        count_thres:
            Threshold for how many times the candidate must be present.

        Returns
        -------
            Generator for a tuple that contains the literal regex, score and new
            series that results from choosing that literal regex.
        """
        first_elem = series.str.extract(r"^([\S\s])[\S\s]*")
        unq_char, counts = np.unique(first_elem.drop_nulls().to_numpy(), return_counts=True)
        thres_char = unq_char[counts >= count_thres]
        for cur_char in thres_char:
            cur_score, next_series, _first_char = score(series, cls([cur_char]), count_thres,
                                                        first_char=first_elem)
            yield (cls(cur_char), cur_score, next_series)

    def __repr__(self):
        return f"Literal [{self.base_regex}]"

    def draw_once(self):
        return np.random.choice(self.literals)

    @classmethod
    def from_string(cls, regex_str) -> Optional[tuple[BaseRegex, str]]:
        _special_chars = [".", "+", "*", "?", "^", "$", "(", ")", "[", "]", "#", "&",
                          "{", "}", "|", "\\", "-", "~", "\r", "\t", "\x0b", "\x0c", "\n"]
        if len(regex_str) > 1 and regex_str[0] == "\\" and regex_str[1] in _special_chars:
            return cls([_unescape(regex_str[1])]), regex_str[2:]
        if len(regex_str) >= 1 and regex_str[0] != "\\" and regex_str[0] not in _special_chars:
            return cls([_unescape(regex_str[0])]), regex_str[1:]
        return None


def get_class_stat(series: pl.Series, count_thres: int) -> list:
    """Find all possible regex elements that can fit the series.

    Then sort them so that the first one is the best fit.
    """
    score_list: list[tuple[BaseRegex, float, pl.Series]] = []
    for rclass in ALL_REGEX_CLASS:
        score_list.extend(rclass.get_candidates(series, count_thres))

    return sorted(score_list, key=lambda res: -res[1])


def _get_bounds(key: str, count_dict: dict[str, int], count_thres: int,
                sigma: float, tot_counts: int) -> tuple[float, float]:
    if tot_counts == 0:
        return 0, 1
    count = count_dict.get(key, 0)
    rel_delta = sigma*np.sqrt(max(count_thres/2, count))/tot_counts
    rel_delta = max(0.15, rel_delta)
    if key in count_dict:
        lower_bound = count/tot_counts - rel_delta
        upper_bound = count/tot_counts + rel_delta
    else:
        lower_bound = 0
        upper_bound = (count_thres/2)/tot_counts + rel_delta
    return lower_bound, upper_bound


def _check_series_compatible(series_a, series_b, count_thres, sigma=2.0):
    stats_a = get_class_stat(series_a, count_thres)
    dict_stat_a = {}
    dict_stat_b = {}
    tot_a = len(series_a) - series_a.null_count()
    tot_b = len(series_b) - series_b.null_count()
    for regex, _score, next_series in stats_a:
        dict_stat_a[regex.base_regex] = len(series_a) - next_series.null_count()
        next_series_b = series_b.str.extract(r"^([" + regex.base_regex + r"])[\S\s]*")
        count_b = len(series_b) - next_series_b.null_count()
        if count_b >= count_thres:
            dict_stat_b[regex.base_regex] = count_b

    for key in (set(dict_stat_a) | set(dict_stat_b)):
        lower_a, upper_a = _get_bounds(key, dict_stat_a, count_thres, sigma, tot_a)
        lower_b, upper_b = _get_bounds(key, dict_stat_b, count_thres, sigma, tot_b)
        if lower_a > upper_b or lower_b > upper_a:
            return False
    return True


class OrRegex(CharRangeRegex):
    """Regex element that contains multiple regex elements.

    For example, a combination of A-Z and a-z to create [a-zA-Z].

    Arguments
    ---------
    regex_instances:
        List of base regexes to choose between.
    min_len:
        Minimum number of repeats.
    max_len:
        Maximum number of repeats.
    """

    def __init__(self, regex_instances: list[BaseRegex], min_len: int = 1, max_len: int = 1):
        self._regex_instances = list(regex_instances)
        super().__init__(min_len, max_len)

    @property
    def n_possible(self):
        """Number of possibilities for all sub regex elements."""
        return np.sum([rc.n_possible for rc in self._regex_instances])

    @property
    def base_regex(self):
        """Base regex for all subregexes."""
        return "".join(rc.base_regex for rc in self._regex_instances)

    @property
    def n_param(self) -> int:
        return np.sum([inst.n_param for inst in self._regex_instances]).astype(int)

    def draw_once(self) -> str:
        prob = np.array([inst.n_possible for inst in self._regex_instances])
        prob = prob/np.sum(prob)
        chosen_inst = np.random.choice(self._regex_instances, p=prob)  # type: ignore
        return chosen_inst.draw_once()

    def covers(self, regex: BaseRegex) -> bool:
        for inst in self._regex_instances:
            if inst.covers(regex):
                return True
        return False

    def append(self, regex: BaseRegex):
        """Add another regex element."""
        self._regex_instances.append(regex)

    @property
    def first_regex(self):
        """Get the first of the regex elements."""
        return self._regex_instances[0]

    def check_compatibility(self, series: pl.Series, count_thres: int) -> bool:
        """Check whether the regex elements are compatible with each other."""
        if len(self._regex_instances) == 1:
            return True
        base_regex = self._regex_instances[0]
        base_series = base_regex.extract_after_first(series)
        for cur_regex in self._regex_instances[1:]:
            cur_new_series = cur_regex.extract_after_first(series)
            compatible = _check_series_compatible(base_series, cur_new_series, count_thres)

            if not compatible:
                return False
        return True

    @classmethod
    def from_string(cls, regex_str) -> Optional[tuple[OrRegex, str]]:
        rex = r"^\[([\S\s]*?[^\\])\]"
        match = re.search(rex, regex_str)
        if match is None:
            return None
        options = match.group(1)
        new_regex_str = regex_str[len(options)+2:]
        inner_str = _unescape(options)
        all_regex = []
        while len(inner_str) > 0:
            last_inner = inner_str
            for regex_class in ALL_REGEX_CLASS:
                ret = regex_class.from_string(inner_str)
                if ret is not None:
                    cur_regex, inner_str = ret
                    all_regex.append(cur_regex)
                    break
            if inner_str == last_inner:
                return None

        (min_len, max_len), new_regex_str = cls.get_class_length(new_regex_str)
        return cls(all_regex, min_len, max_len), new_regex_str


# All regex classes available, order matters.
ALL_REGEX_CLASS: list[type[BaseRegex]] = [UpperRegex, LowerRegex, DigitRegex, LiteralRegex]


def _preview(series):
    return series.drop_nulls()[:3].to_numpy()


def fit_best_regex_class(series: pl.Series, count_thres: int,
                         force_merge: bool = False) -> Optional[dict]:
    """Fit the best regex class.

    This function is the main workhorse to fit the currently best (combination of)
    regex element.

    Arguments
    ---------
    series:
        Series to be fit with regex elements.
    count_thres:
        Threshold for how many values need to use the regex.
    force_merge:
        Whether to try harder to use multiple regex elements in an OR fashion.
        If true, this results in generally faster but worse performance.

    Returns
    -------
        Dictionary containing the score/regex/series if regex is used and
        series for all values that haven't used the regex.
    """
    # Get a ranking for all regex possibilities
    class_stat = get_class_stat(series, count_thres)
    if len(class_stat) == 0:
        return None
    cur_best_regex, best_score, cur_best_series = class_stat[0]
    cur_best_regex = OrRegex([cur_best_regex])
    cur_series = series.set(cur_best_series.is_not_null(), None)  # type: ignore
    class_stat = get_class_stat(cur_series, count_thres)

    # Attempt to combine the regex element with the highest score with the other candidates.
    while len(class_stat) > 0:
        cur_regex, _score, _next = class_stat[0]
        if cur_best_regex.covers(cur_regex):
            continue

        new_cur_series = cur_regex.extract_after_first(cur_series)

        cur_non_null = len(series) - new_cur_series.null_count()
        if cur_non_null < count_thres:
            continue

        # Check whether the look-one-ahead distributions are similar enough.
        compatible = _check_series_compatible(cur_best_series, new_cur_series, count_thres)

        if compatible:
            cur_best_regex.append(cur_regex)
            cur_series = cur_series.set(new_cur_series.is_not_null(), None)  # type: ignore
            class_stat = get_class_stat(cur_series, count_thres)

        class_stat = class_stat[1:]

    # Find the min_length and max_length for the regex class.
    cur_series = series
    cur_series = cur_best_regex.extract_after_first(series)
    start_non_null = len(series) - cur_series.null_count()
    i_start = None
    i_end = None
    for cur_i in range(1, 100):

        compatible = cur_best_regex.check_compatibility(cur_series, count_thres)
        cur_series = cur_best_regex.extract_after_first(cur_series)
        cur_non_null = len(cur_series) - cur_series.null_count()

        if not compatible or cur_non_null < count_thres:
            if i_start is None:
                i_start = cur_i
            i_end = cur_i
            break

        if cur_non_null < start_non_null - count_thres and i_start is None:
            i_start = cur_i

    # This should not happen
    if i_end is None:
        i_end = 100

    # Keep one character classes not together by default.
    if i_end == 1 and i_start == 1 and not force_merge:
        cur_best_regex = cur_best_regex.first_regex

    cur_best_regex.min_len = i_start
    cur_best_regex.max_len = i_end

    new_series = cur_best_regex.extract_after_first(series)
    alt_series = series.set(new_series.is_not_null(), None)  # type: ignore

    return {
        "score": best_score,
        "regex": cur_best_regex,
        "new_series": new_series,
        "alt_series": alt_series,
    }
