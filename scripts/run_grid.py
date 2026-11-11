"""Full fault grid: every model of a config file x faults x patterns x fractions x mask seeds,
with and without the training-free repair front-end.

Loop order is configuration-outer, model-inner, so each corrupted input (and its repaired version)
is built once and shared by all models. Masks and noise draws depend only on
(dataset, pattern, fraction, seed), exactly as in the pilot (scripts/run_faults.py).
Every faulty configuration is scored against the same model's clean, unrepaired predictions on the
same node subset.

Usage (from the SFT-code root):
    python -m scripts.run_grid --dataset metr-la --groups gru_p2 dcrnn_fixed --out results/full/metr-la__rnn.csv
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import runner, stc_bridge
from src.faults import inject, select_faulty_nodes
from src.metrics import node_sums, subset_metrics, sums_to_numpy
from src.repair import repair
from scripts.run_faults import cfg_key

ROOT = Path(__file__).resolve().parents[1]
FAULTS = ("dropout", "stuck", "stuck_long", "noise", "bias")
REPAIR_FAULTS = ("dropout", "stuck", "stuck_long")
FRACS = (0.05, 0.1, 0.2, 0.3, 0.5)


def resolve(path):
    if path is None:
        return None
    if path.startswith("p2:"):
        return stc_bridge.stc_root() / "models" / path[3:]
    return ROOT / path


def corrupt(x, mask, kind, split, args, key, hold, stream):
    kw = {}
    if kind == "noise":
        kw = {"sigma": args.noise_sigma * split["train_std"], "rng": np.random.default_rng(key + [stream])}
    elif kind == "bias":
        kw = {"bias": args.bias_value * split["train_std"]}
    elif kind == "stuck_long":
        kw = {"hold": hold}
    return inject(x, mask, kind, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--groups", nargs="+", default=None, help="subset of config groups")
    ap.add_argument("--faults", nargs="+", default=list(FAULTS))
    ap.add_argument("--patterns", nargs="+", default=["random", "clustered"])
    ap.add_argument("--fracs", nargs="+", type=float, default=list(FRACS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--repair-faults", nargs="*", default=list(REPAIR_FAULTS))
    ap.add_argument("--noise-sigma", type=float, default=0.5, help="multiple of train std")
    ap.add_argument("--bias-value", type=float, default=0.5, help="multiple of train std")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds, dev = args.dataset, torch.device(args.device)
    split = runner.load_split(ds)
    n = split["num_nodes"]
    adj = torch.from_numpy(split["adj"]).to(dev)
    specs = json.loads((ROOT / "configs" / f"models_{ds}.json").read_text())
    if args.groups:
        specs = [s for s in specs if s["group"] in args.groups]

    models = {}
    for s in specs:
        if s["path"] is None:
            models[s["label"]] = None
        else:
            models[s["label"]], _ = runner.load_model(s["name"], ds, n, dev, path=resolve(s["path"]))

    y = torch.from_numpy(split["y"]).to(dev)

    def forecast(label, x_t, x_sn_t):
        name = next(s["name"] for s in specs if s["label"] == label)
        if models[label] is None:
            return runner.naive_predict(name, x_sn_t if name == "seasonal_naive" else x_t)
        return runner.predict_t(models[label], x_t, split, args.batch_size)

    rows, pernode = [], {}
    base = {s["label"]: {"dataset": ds, "label": s["label"], "group": s["group"],
                         "model_seed": s.get("seed", -1),
                         "checkpoint": None if s["path"] is None else Path(s["path"]).name} for s in specs}

    def score(x_t, x_sn_t, fault, pattern, frac, seed, mask, rep, det):
        for s in specs:
            lab = s["label"]
            sums = sums_to_numpy(node_sums(forecast(lab, x_t, x_sn_t), y))
            ref = subset_metrics(base_sums[lab], mask)
            m = subset_metrics(sums, mask)
            row = {**base[lab], "fault": fault, "pattern": pattern, "frac": frac, "seed": seed,
                   "n_faulty": int(mask.sum()), "repair": rep, **det, **m}
            for k, v in ref.items():
                row[f"clean_{k}"] = v
                row[f"ratio_{k}"] = m[k] / v if v else float("nan")
            rows.append(row)
            key = f"{lab}|{fault}|{pattern}|{frac}|{seed}|{rep}"
            pernode[key + "|nz"] = (sums["err_nz"] / np.maximum(sums["cnt_nz"], 1)).astype(np.float32)
            pernode[key + "|all"] = (sums["err"] / sums["cnt"]).astype(np.float32)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        """Rewrite the CSV and per-node arrays so a crash loses at most one fault type."""
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(out.with_suffix(".pernode.npz"), **pernode)

    x_np, xsn_np = split["x"], split["x_sn"]
    x0, xsn0 = torch.from_numpy(x_np).to(dev), torch.from_numpy(xsn_np).to(dev)
    none = np.zeros(n, dtype=bool)
    t0 = time.time()
    base_sums = {}
    for s in specs:  # clean, unrepaired: the reference for every ratio
        base_sums[s["label"]] = sums_to_numpy(node_sums(forecast(s["label"], x0, xsn0), y))
    score(x0, xsn0, "none", "none", 0.0, -1, none, 0, {})
    if args.repair_faults:  # what the front-end does to healthy, clean input
        xr, fl = repair(x0, adj)
        xsr, _ = repair(xsn0, adj)
        score(xr, xsr, "none", "none", 0.0, -1, none, 1,
              {"flag_rate_faulty": float("nan"), "flag_rate_healthy": float(fl.float().mean())})
    print(f"  clean done ({time.time() - t0:.0f}s)", flush=True)

    for kind in args.faults:
        for pattern in args.patterns:
            for frac in args.fracs:
                for seed in range(args.seeds):
                    key = cfg_key(ds, pattern, frac, seed)
                    mask = select_faulty_nodes(n, frac, pattern, np.random.default_rng(key), split["adj"])
                    pernode[f"mask|{pattern}|{frac}|{seed}"] = mask
                    xf = torch.from_numpy(corrupt(x_np, mask, kind, split, args, key, split["hold"], 1)).to(dev)
                    xsnf = torch.from_numpy(
                        corrupt(xsn_np, mask, kind, split, args, key, split["hold_sn"], 2)).to(dev)
                    score(xf, xsnf, kind, pattern, frac, seed, mask, 0, {})
                    if kind in args.repair_faults:
                        xr, fl = repair(xf, adj)
                        xsr, _ = repair(xsnf, adj)
                        mk = torch.from_numpy(mask).to(dev)
                        det = {"flag_rate_faulty": float(fl[:, mk].float().mean()),
                               "flag_rate_healthy": float(fl[:, ~mk].float().mean())}
                        score(xr, xsr, kind, pattern, frac, seed, mask, 1, det)
                print(f"  {kind} {pattern} {frac}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        flush()

    flush()
    print(f"wrote {out} ({len(rows)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
