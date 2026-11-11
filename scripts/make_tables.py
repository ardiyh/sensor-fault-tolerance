"""Manuscript Table 2 as a LaTeX body (SFT-paper/tables/tab2_main.tex) from results/full.

Table 2: MAE (mph, native zeros excluded) on clean input and with 20% random faulty sensors,
         mean +/- std over mask seeds and training seeds, plus the two remedies. The distillation
         tests are reported in the text from results/full/<ds>__tests.json.

Usage: python -m scripts.make_tables
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[0] / "SFT-paper" / "tables"
DATASETS = (("metr-la", "METR-LA"), ("pems-bay", "PEMS-BAY"))
ROWS = [("persistence", 0, "Persistence"), ("seasonal_naive", 0, "Seasonal naive"),
        ("gru_p2", 0, "Recurrent, no mixing"), ("dcrnn_fixed", 0, "DCRNN"),
        ("mlp_task_p7", 0, "Student-MLP, task"), ("mlp_kd_p2", 0, "Student-MLP, distilled"),
        ("attn_task_rev1", 0, "Student-Attn, task"), ("attn_kd_rev1", 0, "Student-Attn, distilled"),
        ("attn_task_rev1", 1, "Student-Attn, task + front-end"),
        ("attn_aug_p7", 0, "Student-Attn, dropout-trained")]
FAULTS = ["dropout", "stuck", "stuck_long", "noise", "bias"]


def pm(m, s, nd=2):
    if np.isnan(m):
        return "--"
    return f"{m:.{nd}f}" if np.isnan(s) else f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def table_main(frac=0.2, pattern="random"):
    lines = []
    for ds, dname in DATASETS:
        g = pd.read_csv(ROOT / "results" / "full" / f"{ds}__groups.csv")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{dname}}}}} \\\\")
        for grp, rep, name in ROWS:
            sel = g[(g.group == grp) & (g.repair == rep)]
            if sel.empty:
                continue
            clean = sel[sel.fault == "none"]
            cells = [pm(clean.all_mae_nz_mean.iloc[0], clean.all_mae_nz_std.iloc[0])]
            for f in FAULTS:
                r = sel[(sel.fault == f) & (sel.pattern == pattern) & np.isclose(sel.frac, frac)]
                cells.append(pm(r.all_mae_nz_mean.iloc[0], r.all_mae_nz_std.iloc[0]) if len(r) else "--")
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    (OUT / "tab2_main.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table_main()
    print(f"wrote {OUT / 'tab2_main.tex'}")


if __name__ == "__main__":
    main()
