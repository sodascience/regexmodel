#!/usr/bin/env python

import json
from pathlib import Path

from benchmark import standard_run
from create_dataframe import create_dataframe


if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    bench_fp = output_dir/"benchmark.json"
    standard_run(bench_fp)

    for stat_name in ["avg_log_like_per_char", "n_parameters", "success_rate",
                      "fit_time", "statistics_time"]:
        # bench_fp, output_dir, stat_name = sys.argv[1:4]
        with open(bench_fp, "r") as handle:
            bench_data = json.load(handle)
        df = create_dataframe(bench_data, stat_name)
        df.write_csv(output_dir / f"{stat_name}.csv")
