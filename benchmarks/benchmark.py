#!/usr/bin/env python

from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import time
import json
import sys

from faker import Faker

from regexmodel.model import RegexModel, NotFittedError
from regexmodel.util import LOG_LIKE_PER_CHAR


def run_bench(faker_type, count_thres, n_fake, locale="NL", method="accurate"):
    fake = Faker(locale=locale)
    Faker.seed(12345)
    fake_data = [getattr(fake, faker_type)() for _ in range(max(n_fake))]
    fake_data_2 = [getattr(fake, faker_type)() for _ in range(max(n_fake))]

    all_res = []
    for cur_n_fake in n_fake:
        for cur_count_thres in count_thres:
            start_time = time.time()
            try:
                model = RegexModel.fit(fake_data[:cur_n_fake], cur_count_thres, method=method)
                mid_time = time.time()
                stats = model.fit_statistics(fake_data_2[:cur_n_fake])
            except NotFittedError:
                stats = {
                    "failed": cur_n_fake,
                    "success": 0,
                    "n_tot_char": len("".join(fake_data_2[:cur_n_fake])),
                    "n_char_success": 0,
                    "n_parameters": 0,
                    "avg_log_like_per_char": LOG_LIKE_PER_CHAR,
                    "avg_log_like_pc_success": LOG_LIKE_PER_CHAR,
                }
            end_time = time.time()
            success_rate = stats["success"]/(stats["success"] + stats["failed"])
            stats.update({"n_fake": cur_n_fake, "threshold": cur_count_thres,
                          "success_rate": success_rate,
                          "fit_time": mid_time-start_time,
                          "statistics_time": end_time-mid_time})
            all_res.append(stats)
    return all_res


def standard_run(out_fp, method):
    locales = ["nl", "fr", "en", "de", "da"]
    faker_types = ["address", "phone_number", "pricetag", "timezone", "mime_type", "unix_partition",
                   "ascii_email", "isbn10", "job", "ssn", "user_agent", "color", "license_plate",
                   "iban", "company", "time", "ipv4", "uri", "name"]
    n_fake = [100, 200, 400, 600, 1000]
    count_thres = [2, 5, 10, 20]

    executor = ProcessPoolExecutor()
    future_results = defaultdict(dict)
    for cur_faker_type in faker_types:
        for cur_locale in locales:
            future_results[cur_faker_type][cur_locale] = executor.submit(
                run_bench, cur_faker_type, count_thres, n_fake, locale=cur_locale, method=method)

    results = {
        cur_faker_type: {locale: locale_data.result()
                         for locale, locale_data in cur_faker_data.items()}
        for cur_faker_type, cur_faker_data in future_results.items()
    }
    with open(out_fp, "w") as handle:
        json.dump(results, handle)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Need two arguments: output_fp, fit_method")
    standard_run(sys.argv[1], sys.argv[2])
