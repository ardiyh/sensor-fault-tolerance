"""Aggregate the full grid (results/full/<ds>__*.csv) into group summaries and tests.

Primary metric: MAE with native-zero targets excluded (`*_nz`). Outputs, per dataset:
  results/full/<ds>__groups.csv   mean and std over (model seed, mask seed) per group/config
  results/full/<ds>__tests.json   contamination, KD-pair, and fix tests

Usage: python -m scripts.analyze_full --dataset metr-la
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.stats import bootstrap_ci, paired_node_test

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["fault", "pattern", "frac", "repair"]
VALS = ["all_mae_nz", "healthy_mae_nz", "faulty_mae_nz", "ratio_all_mae_nz", "ratio_healthy_mae_nz",
        "ratio_faulty_mae_nz", "all_mae", "flag_rate_faulty", "flag_rate_healthy"]
PAIRS = {"kd_attn": ("attn_kd_rev1", "attn_task_rev1"), "aug_attn": ("attn_aug_p7", "attn_task_p7"),
         "aug_mlp": ("mlp_aug_p7", "mlp_task_p7")}


def load(ds):
    files = sorted(glob.glob(str(ROOT / "results" / "full" / f"{ds}__*.csv")))
    files = [f for f in files if not f.endswith(("__groups.csv",))]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pern = {}
    for f in files:
        with np.load(Path(f).with_suffix(".pernode.npz")) as z:
            pern.update({k: z[k] for k in z.files})
    return df, pern


def group_table(df):
    g = df.groupby(["group"] + KEYS)
    out = g[VALS].mean().add_suffix("_mean").join(g[VALS].std().add_suffix("_std"))
    # spread across training seeds: average over mask seeds first, then std over model seeds
    seed_lvl = df.groupby(["group", "model_seed"] + KEYS)[["all_mae_nz", "ratio_all_mae_nz"]].mean()
    out = out.join(seed_lvl.groupby(["group"] + KEYS).std().add_suffix("_seedstd"))
    out["n_rows"] = g.size()
    return out.reset_index()


def contamination(df, pern):
    """Healthy-node MAE (nz) under fault vs clean, Wilcoxon across healthy nodes, per (model, mask seed)."""
    res = []
    f = df[(df.fault != "none") & (df.repair == 0)]
    for (grp, fault, pattern, frac), sub in f.groupby(["group", "fault", "pattern", "frac"]):
        ps, rel = [], []
        for _, r in sub.iterrows():
            mask = pern[f"mask|{pattern}|{frac}|{int(r.seed)}"].astype(bool)
            a = pern[f"{r.label}|{fault}|{pattern}|{frac}|{int(r.seed)}|0|nz"][~mask]
            b = pern[f"{r.label}|none|none|0.0|-1|0|nz"][~mask]
            ps.append(paired_node_test(a, b)["p_value"])
            rel.append(float(np.nanmean(a) / np.nanmean(b)))
        res.append({"group": grp, "fault": fault, "pattern": pattern, "frac": frac, "runs": len(ps),
                    "share_p_below_0.05": float(np.mean(np.array(ps) < 0.05)),
                    "healthy_ratio_mean": float(np.mean(rel))})
    return res


def rel_degradation(pern, label, fault, pattern, frac, seed, rep=0):
    a = pern[f"{label}|{fault}|{pattern}|{frac}|{seed}|{rep}|nz"]
    b = pern[f"{label}|none|none|0.0|-1|0|nz"]
    return (a - b) / b


def pair_tests(df, pern, cand, ref):
    """cand vs ref groups (5 training seeds each). Seed-level Mann-Whitney + node-level Wilcoxon."""
    res = []
    f = df[(df.fault != "none") & (df.repair == 0)]
    labels = {g: sorted(df[df.group == g].label.unique()) for g in (cand, ref)}
    if not labels[cand] or not labels[ref]:
        return res
    for (fault, pattern, frac), sub in f[f.group == cand].groupby(["fault", "pattern", "frac"]):
        seeds = sorted(sub.seed.unique())
        for subset in ("healthy", "faulty", "all"):
            seed_means, node_vals = {cand: [], ref: []}, {cand: [], ref: []}
            for g in (cand, ref):
                per_label = []
                for lab in labels[g]:
                    vals = []
                    for s in seeds:
                        mask = pern[f"mask|{pattern}|{frac}|{int(s)}"].astype(bool)
                        sel = {"healthy": ~mask, "faulty": mask, "all": np.ones_like(mask)}[subset]
                        d = rel_degradation(pern, lab, fault, pattern, frac, int(s))
                        vals.append(d[sel])
                    seed_means[g].append(float(np.mean([np.nanmean(v) for v in vals])))
                    per_label.append(vals)
                # node level: average over training seeds, keep (mask seed, node) as the unit
                node_vals[g] = np.concatenate([np.nanmean([pl[i] for pl in per_label], axis=0)
                                               for i in range(len(seeds))])
            mw = sps.mannwhitneyu(seed_means[cand], seed_means[ref], alternative="two-sided")
            t = paired_node_test(node_vals[cand], node_vals[ref])
            lo, hi = bootstrap_ci(node_vals[cand] - node_vals[ref])
            res.append({"cand": cand, "ref": ref, "fault": fault, "pattern": pattern, "frac": frac,
                        "subset": subset,
                        "cand_seed_mean": float(np.mean(seed_means[cand])),
                        "cand_seed_std": float(np.std(seed_means[cand], ddof=1)),
                        "ref_seed_mean": float(np.mean(seed_means[ref])),
                        "ref_seed_std": float(np.std(seed_means[ref], ddof=1)),
                        "mannwhitney_p": float(mw.pvalue),
                        "seed_ranges_overlap": bool(min(seed_means[cand]) <= max(seed_means[ref])
                                                    and min(seed_means[ref]) <= max(seed_means[cand])),
                        "node_diff_ci95": [lo, hi], **{f"node_{k}": v for k, v in t.items()}})
    return res


def single_vs_band(df, single, band):
    """One checkpoint (e.g. Student-MLP KD) against the spread of a 5-seed task-only band."""
    res = []
    f = df[(df.fault != "none") & (df.repair == 0)]
    s = f[f.group == single].groupby(["fault", "pattern", "frac"])["ratio_all_mae_nz"].mean()
    b = f[f.group == band].groupby(["fault", "pattern", "frac", "model_seed"])["ratio_all_mae_nz"].mean()
    for key, v in s.items():
        vals = b.loc[key].values if key in b.index.droplevel(-1) else np.array([])
        if vals.size == 0:
            continue
        res.append({"single": single, "band": band, "fault": key[0], "pattern": key[1], "frac": key[2],
                    "single_ratio": float(v), "band_min": float(vals.min()), "band_max": float(vals.max()),
                    "outside_band": bool(v < vals.min() or v > vals.max())})
    return res


def repair_effect(df):
    """Recovery = share of the fault-induced MAE increase removed by the front-end (all targets, nz)."""
    res = []
    f = df[df.fault != "none"]
    clean = df[(df.fault == "none") & (df.repair == 0)].groupby("group")["all_mae_nz"].mean()
    clean_rep = df[(df.fault == "none") & (df.repair == 1)].groupby("group")["all_mae_nz"].mean()
    m = f.groupby(["group", "fault", "pattern", "frac", "repair"])[
        ["all_mae_nz", "flag_rate_faulty", "flag_rate_healthy"]].mean().reset_index()
    for (grp, fault, pattern, frac), sub in m.groupby(["group", "fault", "pattern", "frac"]):
        if set(sub.repair) != {0, 1}:
            continue
        raw = float(sub[sub.repair == 0].all_mae_nz.iloc[0])
        rep = sub[sub.repair == 1].iloc[0]
        c = float(clean[grp])
        res.append({"group": grp, "fault": fault, "pattern": pattern, "frac": frac, "clean": c,
                    "clean_with_frontend": float(clean_rep.get(grp, np.nan)), "faulty_raw": raw,
                    "faulty_repaired": float(rep.all_mae_nz),
                    "recovery": (raw - float(rep.all_mae_nz)) / (raw - c) if raw > c else float("nan"),
                    "flag_rate_faulty": float(rep.flag_rate_faulty),
                    "flag_rate_healthy": float(rep.flag_rate_healthy)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    df, pern = load(args.dataset)
    out = ROOT / "results" / "full"
    group_table(df).to_csv(out / f"{args.dataset}__groups.csv", index=False, float_format="%.5f")
    tests = {"contamination": contamination(df, pern),
             "pairs": {k: pair_tests(df, pern, *v) for k, v in PAIRS.items()},
             "mlp_kd_vs_task_band": single_vs_band(df, "mlp_kd_p2", "mlp_task_p7"),
             "mlp_task_p2_vs_task_band": single_vs_band(df, "mlp_task_p2", "mlp_task_p7"),
             "repair": repair_effect(df)}
    (out / f"{args.dataset}__tests.json").write_text(json.dumps(tests, indent=1))
    print(f"wrote {out / (args.dataset + '__groups.csv')} and __tests.json ({len(df)} rows)")


if __name__ == "__main__":
    main()
