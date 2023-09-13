#!/usr/bin/env python

import sys
import json
from pathlib import Path

from benchmark import standard_run
from create_dataframe import create_dataframe


if __name__ == "__main__":
    all_regex_method = ["accurate", "fast"]
    if len(sys.argv) == 2:
        all_regex_method = [sys.argv[1]]
    for regex_method in all_regex_method:
        output_dir = Path(regex_method)
        output_dir.mkdir(exist_ok=True)
        bench_fp = output_dir/"benchmark.json"
        standard_run(bench_fp, regex_method)

        for stat_name in ["avg_log_like_per_char", "n_parameters", "success_rate",
                          "fit_time", "statistics_time", "avg_log_like_pc_success"]:
            with open(bench_fp, "r") as handle:
                bench_data = json.load(handle)
            df = create_dataframe(bench_data, stat_name)
            df.write_csv(output_dir / f"{stat_name}.csv")
