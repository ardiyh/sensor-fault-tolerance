"""Manuscript figures 2-3 from results/full/<ds>__groups.csv (vector PDF into SFT-paper/figures).

Colour: at most three categorical hues per panel (validated all-pairs, light mode, with
dataviz/scripts/validate_palette.js); non-spatial references are grey with distinct line styles.
Every series also carries a marker, so identity never depends on colour alone.

Usage: python -m scripts.make_figures
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[0] / "SFT-paper" / "figures"
DATASETS = (("metr-la", "METR-LA"), ("pems-bay", "PEMS-BAY"))
CM = 1 / 2.54
WIDTH = 15.5 * CM

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # group -> (label, colour, linestyle, marker)
    "attn_task_rev1": ("Student-Attn", BLUE, "-", "o"),
    "mlp_task_p7": ("Student-MLP", ORANGE, "-", "s"),
    "dcrnn_fixed": ("DCRNN", AQUA, "-", "^"),
    "gru_p2": ("Recurrent, no mixing", INK2, "--", "D"),
    "persistence": ("Persistence", "#8a8984", ":", "v"),
}
FAULT_LABEL = {"dropout": "Dropout", "stuck": "Stuck", "stuck_long": "Long stuck",
               "noise": "Noise", "bias": "Bias"}

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})


def groups(ds):
    return pd.read_csv(ROOT / "results" / "full" / f"{ds}__groups.csv")


def curve(g, group, fault, pattern="random", repair=0, col="ratio_all_mae_nz_mean"):
    sub = g[(g.group == group) & (g.fault == fault) & (g.pattern == pattern) & (g.repair == repair)]
    sub = sub.sort_values("frac")
    return sub.frac.values * 100, sub[col].values


def panel_tag(ax, text):
    ax.text(0.03, 0.95, text, transform=ax.transAxes, va="top", ha="left", color=INK, fontsize=7.5)


def fig_contamination(fault="dropout"):
    """Healthy-sensor ratio R_H and faulty-sensor MAE vs fraction, random vs clustered faults."""
    cols = [("ratio_healthy_mae_nz_mean", "random"), ("ratio_healthy_mae_nz_mean", "clustered"),
            ("faulty_mae_nz_mean", "random"), ("faulty_mae_nz_mean", "clustered")]
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 6.6 * CM), sharex=True)
    for r, (ds, dname) in enumerate(DATASETS):
        g = groups(ds)
        for c, (col, pattern) in enumerate(cols):
            ax = axes[r, c]
            for grp, (lab, colr, ls, mk) in SERIES.items():
                x, y = curve(g, grp, fault, pattern=pattern, col=col)
                if len(x):
                    ax.plot(x, y, color=colr, ls=ls, marker=mk, label=lab)
            panel_tag(ax, f"{dname}, {pattern}")
            ax.set_xticks([10, 30, 50])
            if r == 1:
                ax.set_xlabel("Faulty sensors (%)")
        for c0 in (0, 2):  # random and clustered panels of one metric share the y range
            lo = min(axes[r, c0].get_ylim()[0], axes[r, c0 + 1].get_ylim()[0])
            hi = max(axes[r, c0].get_ylim()[1], axes[r, c0 + 1].get_ylim()[1])
            for c in (c0, c0 + 1):
                axes[r, c].set_ylim(0 if c0 == 2 else lo, hi * 1.3)
        axes[r, 0].set_ylabel("Healthy ratio $R_H$")
        axes[r, 2].set_ylabel("Faulty MAE (mph)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.9, h_pad=0.4)
    fig.savefig(OUT / "fig2_contamination.pdf")
    plt.close(fig)


def fig_remedies(faults=("dropout", "stuck_long")):
    """Absolute all-target MAE under faults: raw, with front-end, retrained with dropout."""
    spec = [("attn_task_p7", 0, "Student-Attn", BLUE, "-", "o"),
            ("attn_task_p7", 1, "Student-Attn + front-end", BLUE, "--", "o"),
            ("attn_aug_p7", 0, "Student-Attn, dropout-trained", ORANGE, "-", "s"),
            ("persistence", 1, "Persistence + front-end", "#8a8984", ":", "v")]
    fig, axes = plt.subplots(1, 2 * len(faults), figsize=(WIDTH, 6.2 * CM))
    k = 0
    for ds, dname in DATASETS:
        g = groups(ds)
        for fault in faults:
            ax = axes[k]
            k += 1
            for grp, rep, lab, colr, ls, mk in spec:
                x, y = curve(g, grp, fault, repair=rep, col="all_mae_nz_mean")
                if not len(x):
                    continue
                c0 = g[(g.group == grp) & (g.fault == "none") & (g.repair == rep)]["all_mae_nz_mean"]
                ax.plot(np.r_[0, x], np.r_[c0.iloc[0], y], color=colr, ls=ls, marker=mk, label=lab)
            panel_tag(ax, f"{dname}\n{FAULT_LABEL[fault].lower()}")
            ax.set_xticks([0, 10, 20, 30, 50])
            ax.set_xlabel("Faulty sensors (%)")
            if k == 1:
                ax.set_ylabel("MAE (mph)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=0.5)
    fig.savefig(OUT / "fig3_remedies.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_contamination()
    fig_remedies()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
