#!/usr/bin/env python

from collections import defaultdict
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


def create_dataframe(all_data, stat_name="success_rate"):
    results = defaultdict(list)
    results.update({"faker_type": list(all_data)})
    for faker_type, cur_faker_data in all_data.items():
        for cur_locale, stat_list in cur_faker_data.items():
            avg_data = np.mean([stat[stat_name] for stat in stat_list])
            results[cur_locale].append(avg_data)

    for locale in results:
        if locale != "faker_type":
            results[locale] = [np.mean(results[locale])] + results[locale]
    results["faker_type"] = ["average"] + results["faker_type"]
    return pl.DataFrame(results)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError("Need three arguments: benchmark file, output_dir, statistic name.")
    bench_fp, output_dir, stat_name = sys.argv[1:4]
    with open(bench_fp, "r") as handle:
        bench_data = json.load(handle)
    df = create_dataframe(bench_data, stat_name)
    df.write_csv(Path(sys.argv[2]) / f"{stat_name}.csv")
