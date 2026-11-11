"""Gate before any fault injection: clean-input MAE must match Paper 6's dumps and Paper 2's JSON.

Usage (from the SFT-code root):
    python -m scripts.reproduce_clean --dataset metr-la [--device cpu] [--batch-size 16]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import runner, stc_bridge

ROOT = Path(__file__).resolve().parents[1]
DAMS_PREDS = ROOT.parents[1] / "paper6_DAMS" / "DAMS-code" / "results" / "preds"
P2_NAMES = {
    "dcrnn": "DCRNN (Baseline)",
    "student_mlp_baseline": "Student-MLP (Baseline)",
    "student_mlp_kd": "Student-MLP (KD)",
    "student_attn_baseline": "Student-Attn (Baseline)",
    "student_attn_kd_at": "Student-Attn (KD+AT)",
}
# Paper 2's "SeasonalNaive" (scripts/run_seasonal_naive.py) repeats the last 12-step input window,
# i.e. a lag-12 naive, not the daily seasonal naive used here and in Paper 6. It is checked as a
# separate row so its published number stays traceable.
P2_LAST_WINDOW = "last_window_naive"
TOL_REL = 0.005  # 0.5% relative MAE: Paper 6 reproduced Paper 2 within 0.17%


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    ds, device = args.dataset, torch.device(args.device)

    stc_bridge._ensure_on_path()
    from utils.common import compute_metrics

    split = runner.load_split(ds)
    p2 = {r["name"]: r for r in json.loads(
        (stc_bridge.stc_root() / "results" / f"results_{ds}.json").read_text())}
    p2_sn = json.loads((stc_bridge.stc_root() / "results" / "seasonal_naive_results.json")
                       .read_text()).get(ds)
    rows, ok_all = [], True
    for name in runner.MODELS + (P2_LAST_WINDOW,):
        t0 = time.time()
        if name in runner.LEARNED:
            try:
                model, path = runner.load_model(name, ds, split["num_nodes"], device)
            except FileNotFoundError as e:
                rows.append({"model": name, "status": "MISSING", "detail": str(e)})
                print(f"  {name}: MISSING")
                continue
            pred = runner.predict(model, split["x"], split, device, args.batch_size)
            ckpt = path.name
        elif name == P2_LAST_WINDOW:
            pred, ckpt = split["x"].copy(), None
        else:
            x_in = split["x_sn"] if name == "seasonal_naive" else split["x"]
            pred = runner.naive_predict(name, x_in)
            ckpt = None
        mae, rmse, mape, nmae = compute_metrics(pred, split["y"])
        row = {"model": name, "checkpoint": ckpt, "mae": float(mae), "nmae": float(nmae),
               "rmse": float(rmse), "seconds": round(time.time() - t0, 1)}

        dump = DAMS_PREDS / f"{ds}__{name}.npz"
        if dump.is_file():
            d = np.load(dump)
            row["dams_mae"] = float(np.mean(np.abs(d["pred"] - d["true"])))
            row["max_abs_pred_diff_vs_dams"] = float(np.nanmax(np.abs(d["pred"] - pred)))
            row["true_identical_to_dams"] = bool(np.allclose(d["true"], split["y"], atol=1e-4))
        ref = p2.get(P2_NAMES[name]) if name in P2_NAMES else (
            p2_sn if name == P2_LAST_WINDOW else None)
        if ref is not None:
            row["paper2_mae"] = float(ref["mae"])
        refs = [row[k] for k in ("dams_mae", "paper2_mae") if k in row]
        row["max_rel_dev"] = max(abs(mae - r) / r for r in refs) if refs else None
        row["status"] = ("PASS" if refs and row["max_rel_dev"] <= TOL_REL
                         else "NO_REFERENCE" if not refs else "FAIL")
        ok_all &= row["status"] != "FAIL"
        rows.append(row)
        print(f"  {name}: MAE={mae:.4f} dams={row.get('dams_mae')} p2={row.get('paper2_mae')} "
              f"dev={row['max_rel_dev']} {row['status']} ({row['seconds']}s)")

    out = ROOT / "results" / "clean" / f"{ds}__{args.device}_bs{args.batch_size}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": ds, "device": args.device,
                               "batch_size": args.batch_size, "tolerance_rel": TOL_REL,
                               "all_pass": bool(ok_all), "rows": rows}, indent=2))
    print(f"wrote {out}  all_pass={ok_all}")


if __name__ == "__main__":
    main()
