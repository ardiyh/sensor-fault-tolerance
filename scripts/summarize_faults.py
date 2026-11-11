"""Aggregate run_faults output: mean ± std over mask seeds, plus paired node-level tests.

Usage:
    python -m scripts.summarize_faults --runs results/pilot/metr-la__light.csv results/pilot/metr-la__dcrnn.csv \
        --out results/pilot/metr-la__summary
Writes <out>.csv (table) and <out>.json (tests).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.stats import bootstrap_ci, paired_node_test

ROOT = Path(__file__).resolve().parents[1]
COLS = ["all_mae", "ratio_all_mae", "healthy_mae", "ratio_healthy_mae", "faulty_mae",
        "ratio_faulty_mae", "all_mae_nz", "ratio_all_mae_nz", "healthy_mae_nz",
        "ratio_healthy_mae_nz", "faulty_mae_nz", "ratio_faulty_mae_nz"]
KD_PAIRS = [("student_attn_kd_at", "student_attn_baseline"),
            ("student_mlp_kd", "student_mlp_baseline")]


def load(runs):
    df = pd.concat([pd.read_csv(ROOT / r) for r in runs], ignore_index=True)
    pern = {}
    for r in runs:
        with np.load((ROOT / r).with_suffix(".pernode.npz")) as z:
            pern.update({k: z[k] for k in z.files})
    return df, pern


def table(df):
    f = df[df.fault != "none"]
    g = f.groupby(["model", "fault", "pattern", "frac"])[COLS]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    clean = df[df.fault == "none"].set_index("model")[["all_mae", "all_mae_nz"]]
    return out.merge(clean.add_prefix("clean_").reset_index(), on="model")


def healthy_tests(df, pern):
    """Per config: Wilcoxon of healthy-node MAE under fault vs clean (contamination test)."""
    res = []
    f = df[df.fault != "none"]
    for (model, fault, pattern, frac), grp in f.groupby(["model", "fault", "pattern", "frac"]):
        ps, deltas = [], []
        for seed in grp.seed:
            mask = pern[f"mask|{pattern}|{frac}|{seed}"].astype(bool)
            faulty = pern[f"{model}|{fault}|{pattern}|{frac}|{seed}"][~mask]
            clean = pern[f"{model}|clean"][~mask]
            t = paired_node_test(faulty, clean)
            ps.append(t["p_value"])
            deltas.append(t["median_delta"])
        res.append({"model": model, "fault": fault, "pattern": pattern, "frac": frac,
                    "seeds": len(ps), "seeds_p_below_0.05": int(np.sum(np.array(ps) < 0.05)),
                    "max_p": float(np.max(ps)), "median_node_delta_mean": float(np.mean(deltas))})
    return res


def kd_tests(df, pern):
    """KD vs task-only: per-node degradation (fault MAE - clean MAE), paired over nodes and seeds."""
    res = []
    f = df[df.fault != "none"]
    for kd, base in KD_PAIRS:
        if kd not in set(df.model) or base not in set(df.model):
            continue
        for (fault, pattern, frac), grp in f[f.model == kd].groupby(["fault", "pattern", "frac"]):
            for subset in ("healthy", "faulty"):
                a_all, b_all = [], []
                for seed in grp.seed:
                    mask = pern[f"mask|{pattern}|{frac}|{seed}"].astype(bool)
                    sel = ~mask if subset == "healthy" else mask
                    for m, acc in ((kd, a_all), (base, b_all)):
                        d = (pern[f"{m}|{fault}|{pattern}|{frac}|{seed}"]
                             - pern[f"{m}|clean"]) / pern[f"{m}|clean"]
                        acc.append(d[sel])
                a, b = np.concatenate(a_all), np.concatenate(b_all)
                t = paired_node_test(a, b)
                lo, hi = bootstrap_ci(a - b)
                res.append({"kd": kd, "task_only": base, "fault": fault, "pattern": pattern,
                            "frac": frac, "subset": subset,
                            "kd_rel_degradation_mean": float(a.mean()),
                            "task_rel_degradation_mean": float(b.mean()),
                            "diff_ci95": [lo, hi], **t})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df, pern = load(args.runs)
    tab = table(df)
    out = ROOT / args.out
    tab.to_csv(out.with_suffix(".csv"), index=False, float_format="%.5f")
    tests = {"healthy_vs_clean": healthy_tests(df, pern), "kd_vs_task_only": kd_tests(df, pern)}
    out.with_suffix(".json").write_text(json.dumps(tests, indent=2))
    print(f"wrote {out.with_suffix('.csv')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
