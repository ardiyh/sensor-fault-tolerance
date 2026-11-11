"""Inject sensor faults into the test inputs and score every model on healthy and faulty targets.

One fault mask per (pattern, fraction, mask seed) is shared by every model and fault type, so all
comparisons are paired. Each faulty configuration is scored against the same model's clean
predictions on the same node subset, which gives the degradation ratio directly.

Usage (from the SFT-code root):
    python -m scripts.run_faults --dataset metr-la --faults dropout stuck --patterns random \
        --out results/pilot/metr-la.csv
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src import runner
from src.faults import inject, select_faulty_nodes
from src.metrics import per_node_mae, split_metrics

ROOT = Path(__file__).resolve().parents[1]
FRACS = (0.05, 0.1, 0.2, 0.3, 0.5)


def cfg_key(dataset, pattern, frac, seed):
    """Masks (and noise draws) depend only on this key, so they are identical for all models."""
    return [sum(map(ord, dataset)), sum(map(ord, pattern)), int(round(frac * 1000)), seed]


def fault_kwargs(kind, split, args, key):
    if kind == "noise":
        return {"sigma": args.noise_sigma * split["train_std"],
                "rng": np.random.default_rng(key + [1])}
    if kind == "bias":
        return {"bias": args.bias_value * split["train_std"]}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+", default=list(runner.MODELS))
    ap.add_argument("--faults", nargs="+", default=["dropout", "stuck"])
    ap.add_argument("--patterns", nargs="+", default=["random"])
    ap.add_argument("--fracs", nargs="+", type=float, default=list(FRACS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise-sigma", type=float, default=0.5, help="multiple of train std")
    ap.add_argument("--bias-value", type=float, default=0.5, help="multiple of train std")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tag", default=None, help="use locally retrained checkpoints *_<tag>.pth")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds, device = args.dataset, torch.device(args.device)
    split = runner.load_split(ds)
    n = split["num_nodes"]
    configs = []
    for pattern in args.patterns:
        for frac in args.fracs:
            for seed in range(args.seeds):
                rng = np.random.default_rng(cfg_key(ds, pattern, frac, seed))
                mask = select_faulty_nodes(n, frac, pattern, rng, split["adj"])
                configs.append((pattern, frac, seed, mask))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pernode_path = out.with_suffix(".pernode.npz")
    pernode = {}
    rows = []

    for name in args.models:
        t0 = time.time()
        if name in runner.LEARNED:
            model, path = runner.load_model(name, ds, n, device, args.tag)
            ckpt = path.name

            def forecast(x):
                return runner.predict(model, x, split, device, args.batch_size)
        else:
            ckpt = None

            def forecast(x, _name=name):
                return runner.naive_predict(_name, x)
        x_clean = split["x_sn"] if name == "seasonal_naive" else split["x"]
        y = split["y"]
        pred_clean = forecast(x_clean)
        node_clean = per_node_mae(pred_clean, y)
        node_clean_nz = per_node_mae(pred_clean, y, nonzero_only=True)
        pernode[f"{name}|clean"] = node_clean
        pernode[f"{name}|clean_nz"] = node_clean_nz
        clean_all = split_metrics(pred_clean, y, np.zeros(n, dtype=bool))
        rows.append({"dataset": ds, "model": name, "checkpoint": ckpt, "fault": "none",
                     "pattern": "none", "frac": 0.0, "seed": -1, "n_faulty": 0,
                     **{k: v for k, v in clean_all.items() if k.startswith("all_")}})

        for kind in args.faults:
            for pattern, frac, seed, mask in configs:
                key = cfg_key(ds, pattern, frac, seed)
                x_f = inject(x_clean, mask, kind, **fault_kwargs(kind, split, args, key))
                pred = forecast(x_f)
                m = split_metrics(pred, y, mask)
                ref = split_metrics(pred_clean, y, mask)
                row = {"dataset": ds, "model": name, "checkpoint": ckpt, "fault": kind,
                       "pattern": pattern, "frac": frac, "seed": seed, "n_faulty": int(mask.sum()),
                       # share of faulty-sensor input readings that were already native zeros
                       "faulty_input_native_zero": float((x_clean[:, mask, :] == 0).mean()),
                       **m}
                for k, v in ref.items():
                    row[f"clean_{k}"] = v
                    row[f"ratio_{k}"] = m[k] / v if v else float("nan")
                rows.append(row)
                pernode[f"{name}|{kind}|{pattern}|{frac}|{seed}"] = per_node_mae(pred, y)
                pernode[f"mask|{pattern}|{frac}|{seed}"] = mask
        print(f"  {name}: {len(args.faults) * len(configs)} configs in {time.time() - t0:.0f}s",
              flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(pernode_path, **pernode)
    print(f"wrote {out} ({len(rows)} rows) and {pernode_path.name}")


if __name__ == "__main__":
    main()
