from pytest import mark
import string

from regexmodel import RegexModel


@mark.parametrize("series,regex,counts", [
        (["R123", "R837", "R354", "R456", "R578", "R699"], r"R[0-9]{3}",
            [6, 6, 6]),
        (["A123", "B123", "C123", "D123", "E123", "F123"], r"[A-F]123",
            [6, 6, 6, 6, 6]),
        (["123a", "123b", "123c", "123d", "123e", "123f"], r"123[a-f]",
            [6, 6, 6, 6, 6]),
        # (["1a3f", "b2af", "22b2", "ac2e", "5f2a", "45e3", "c23f", "de2d", "3a43"], r"[0-9a-z]{4,4}"),
        (["abc", "edf", "abc", "edf", "abc", "abc", "abc"], r"(abc|edf)",
            [[[5, 5, 5, 5], [2, 2, 2, 2]], 0]),
        # (["s123", "s345", "871", "231", "462", "720"], r"(|s)[0-9]{3,3}")
    ])
def test_full_pipeline(series, regex, counts):
    model = RegexModel.fit(series, count_thres=1)
    assert model.regex == regex
    assert model.serialize()["counts"] == counts
    new_series = [model.draw() for _ in range(100)]
    new_model = RegexModel.fit(new_series, count_thres=1)
    assert new_model.regex == model.regex
    new_model = RegexModel(model.serialize())
    assert new_model.regex == model.regex
    assert new_model.serialize()["counts"] == model.serialize()["counts"]

    # Test whether adding one non-conforming value and setting count_thres == 2
    # will ignore the added value.
    series.append("A123")
    new_model.fit(series, count_thres=2)
    assert model.regex == new_model.regex
    assert new_model.serialize()["counts"] == model.serialize()["counts"]

def test_all_chars():
    model = RegexModel.fit(list(string.printable), count_thres=1)
    new_model = RegexModel(model.serialize())
    assert isinstance(new_model, RegexModel)


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
