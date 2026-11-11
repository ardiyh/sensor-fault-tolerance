"""Native missing-data rate (readings coded as 0) per dataset, overall and on the test split.

Usage: python -m scripts.native_zeros --datasets metr-la pems-bay
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src import runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["metr-la", "pems-bay"])
    args = ap.parse_args()
    out = {}
    for ds in args.datasets:
        s = runner.load_split(ds)
        z = s["series"] == 0
        L = len(z)
        test = z[s["test_start"]:]
        per_sensor = test.mean(axis=0)
        out[ds] = {
            "timesteps": int(L), "sensors": int(z.shape[1]),
            "zero_rate_all": float(z.mean()),
            "zero_rate_train": float(z[:int(L * 0.7)].mean()),
            "zero_rate_test": float(test.mean()),
            "test_timesteps_all_sensors_zero": int(test.all(axis=1).sum()),
            "test_per_sensor_zero_rate_median": float(np.median(per_sensor)),
            "test_per_sensor_zero_rate_max": float(per_sensor.max()),
            "test_sensors_above_10pct_zero": int((per_sensor > 0.10).sum()),
            "test_target_entries_zero": float((s["y"] == 0).mean()),
            "test_input_entries_zero": float((s["x"] == 0).mean()),
        }
        print(ds, json.dumps(out[ds]))
    path = ROOT / "results" / "native_zeros.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
