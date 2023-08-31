import string
from random import choice

import numpy as np
from pytest import mark, raises


from regexmodel import RegexModel
from regexmodel.regexclass import LiteralRegex, DigitRegex
from regexmodel.util import Dir


def test_regex_single_digit():
    series = ["R123", "R837", "R354", "R456", "R578", "R699"]
    dist = RegexModel.fit(series)

    def check_regex_dist(dist):
        assert len(dist.root_links) == 1
        assert dist.root_links[0].count == 6
        assert dist.root_links[0].count == 6
        first_node = dist.root_links[0].destination
        assert isinstance(first_node.regex, LiteralRegex)
        assert first_node.regex.string == r"[R]"
        assert len(first_node.sub_links) == 1
        assert first_node.sub_links[0].count == 6
        assert first_node.sub_links[0].destination is None
        assert first_node.sub_links[0].direction == Dir.LEFT
        assert first_node.main_link.count == 6
        assert first_node.main_link.direction == Dir.RIGHT

        second_node = first_node.main_link.destination
        assert isinstance(second_node.regex, DigitRegex)
        assert second_node.regex.string == r"[0-9]{3,3}"
        assert len(second_node.sub_links) == 1
        assert second_node.sub_links[0].count == 6
        assert second_node.sub_links[0].direction == Dir.RIGHT
        assert second_node.sub_links[0].destination is None
        assert second_node.main_link.count == 0
        assert second_node.main_link.direction == Dir.RIGHT
        assert second_node.main_link.destination is None

        for draw_str in [dist.draw() for _ in range(10)]:
            assert len(draw_str) == 4
            assert draw_str[0] == "R"

    check_regex_dist(dist)

    regex_data = dist.serialize()
    new_dist = RegexModel(regex_data)
    check_regex_dist(new_dist)
    print(-len(series)*3*np.log(10), dist.log_likelihood(series))
    assert np.isclose(-len(series)*3*np.log(10), dist.log_likelihood(series))

# @mark.parametrize("series_type", [pd.Series, pl.Series])
# def test_regex_unique(series_type):
#     series = series_type(["R1", "R2", "R3", "R4", "R5", "R6"])
#     dist = UniqueRegexDistribution.fit(series)
#     values = [dist.draw() for _ in range(10)]
#     assert len(set(values)) == 10
#     assert set(values) == set(["R" + x for x in string.digits])
#     with raises(ValueError):
#         dist.draw()
#     dist.draw_reset()
#     dist.draw()


# @mark.parametrize(
#     "dist_class,regex_str,regex_str_alt",
#     [
#         (AlphaNumericRegex, r"\w{1,1}", r"\w"),
#         (LettersRegex, r"[a-zA-Z]{1,1}",  r"[a-zA-Z]"),
#         (LowercaseRegex, r"[a-z]{1,1}", r"[a-z]"),
#         (UppercaseRegex, r"[A-Z]{1,1}", r"[A-Z]"),
#         (DigitRegex, r"\d{1,1}", r"\d"),
#         (AnyRegex, r".[]{1,1}", r".[]"),
#     ]
# )
# def test_optional_length(dist_class, regex_str, regex_str_alt):
#     dist, _ = dist_class.from_string(regex_str)
#     dist_alt, _ = dist_class.from_string(regex_str_alt)
#     assert str(dist) == str(dist_alt)


# @mark.parametrize(
#     "digit_set,dist_class,regex_str,n_digits",
#     [
#         (string.ascii_letters+string.digits, AlphaNumericRegex, r"\w{10,10}", 10),
#         (string.ascii_letters, LettersRegex, r"[a-zA-Z]{10,10}", 10),
#         (string.digits, DigitRegex, r"\d{10,10}", 10),
#         (string.ascii_lowercase, LowercaseRegex, r"[a-z]{10,10}", 10),
#         (string.ascii_uppercase, UppercaseRegex, r"[A-Z]{10,10}", 10),
#     ]
# )
# @mark.parametrize("series_type", [pd.Series, pl.Series])
# def test_digits(digit_set, dist_class, regex_str, n_digits, series_type):
#     def draw():
#         draw_str = ""
#         for _ in range(n_digits):
#             draw_str += choice(digit_set)
#         return draw_str

#     series = series_type([draw() for _ in range(100)])
#     dist = RegexDistribution.fit(series)
#     assert len(dist.re_list) == 1
#     assert isinstance(dist.re_list[0], dist_class)
#     assert dist.re_list[0].min_digit == n_digits
#     assert dist.re_list[0].max_digit == n_digits
#     assert dist.to_dict()["parameters"]["re_list"][0][0] == regex_str
#     assert np.all([len(dist.draw()) == n_digits for _ in range(100)])
#     assert np.all([c in digit_set for c in dist.draw()])
#     new_dist = dist_class.from_string(dist.regex_string, 1.0)[0]
#     with raises(ValueError):
#         dist_class.from_string(dist.regex_string, 3)[0]
#     assert isinstance(new_dist, dist_class)
#     assert new_dist.min_digit == dist.re_list[0].min_digit
#     assert new_dist.max_digit == dist.re_list[0].max_digit


