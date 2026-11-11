"""Clean-input diagnostics behind two observations of the full grid.

1. Positive bias faults lower the students' error: test whether the students under-predict on
   non-zero targets (mean signed error, pred - true, physical units).
2. The repair front-end and dropout training lower the clean error: split the clean non-zero MAE by
   whether the sensor's own input window holds native zeros (none / 1-5 / at least 6).

Writes results/clean_bias_<ds>.json with one entry per model of configs/models_<ds>.json.

Usage: python -m scripts.clean_bias --dataset metr-la --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import runner
from scripts.run_grid import resolve

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--groups", nargs="+", default=None)
    args = ap.parse_args()
    ds, dev = args.dataset, torch.device(args.device)
    split = runner.load_split(ds)
    specs = json.loads((ROOT / "configs" / f"models_{ds}.json").read_text())
    if args.groups:
        specs = [s for s in specs if s["group"] in args.groups]
    x = torch.from_numpy(split["x"]).to(dev)
    xsn = torch.from_numpy(split["x_sn"]).to(dev)
    y = torch.from_numpy(split["y"])  # metrics on the CPU; the GPU only runs the forecasters
    nz = y != 0
    zeros_in = (x.cpu() == 0).sum(-1, keepdim=True).expand_as(y)  # native zeros in the sensor's own window
    bins = {"input_no_zero": zeros_in == 0, "input_1to5_zero": (zeros_in > 0) & (zeros_in < 6),
            "input_6plus_zero": zeros_in >= 6}
    out = {"dataset": ds, "share_nz_targets_by_input": {}, "models": []}
    for b, m in bins.items():
        out["share_nz_targets_by_input"][b] = float((m & nz).sum() / nz.sum())
    for s in specs:
        if s["path"] is None:
            pred = runner.naive_predict(s["name"], xsn if s["name"] == "seasonal_naive" else x)
        else:
            model, _ = runner.load_model(s["name"], ds, split["num_nodes"], dev, path=resolve(s["path"]))
            pred = runner.predict_t(model, x, split, args.batch_size)
            del model
        e = (pred.cpu() - y).double()
        row = {"label": s["label"], "group": s["group"], "model_seed": s.get("seed", -1),
               "mae_nz": float(e[nz].abs().mean()), "signed_error_nz": float(e[nz].mean()),
               "share_under_nz": float((e[nz] < 0).double().mean())}
        for b, m in bins.items():
            sel = m & nz
            row[f"mae_nz_{b}"] = float(e[sel].abs().mean())
            row[f"signed_error_nz_{b}"] = float(e[sel].mean())
        out["models"].append(row)
        print(f"{s['label']:28s} mae_nz={row['mae_nz']:.3f} signed={row['signed_error_nz']:+.3f} "
              f"no-zero={row['mae_nz_input_no_zero']:.3f} 6+={row['mae_nz_input_6plus_zero']:.3f}",
              flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    path = ROOT / "results" / f"clean_bias_{ds}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
