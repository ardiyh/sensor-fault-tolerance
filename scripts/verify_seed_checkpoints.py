"""Match multi-seed student checkpoints restored from Drive against Paper 2's published numbers.

Student-Attn seeds are compared with results/multiseed_summary.json (nMAE, 4 decimals);
single checkpoints with results/results_<ds>.json (MAE). Writes results/seed_checkpoints.json.

Usage: python -m scripts.verify_seed_checkpoints
"""
import json
import re
from pathlib import Path

import torch

from src import runner, stc_bridge
from src.metrics import mae_nmae

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT / "checkpoints" / "drive"


def classify(fname):
    ds = "metr-la" if "metr-la" in fname else "pems-bay" if "pems-bay" in fname else None
    seed = re.search(r"seed(\d)", fname)
    if fname.startswith("student_attention_baseline"):
        name, key = "student_attn_baseline", "attn_baseline"
    elif fname.startswith("best_student_attention"):
        name, key = "student_attn_kd_at", "attn_kdat"
    elif fname.startswith("student_baseline"):
        name, key = "student_mlp_baseline", "Student-MLP (Baseline)"
    else:
        name, key = "student_mlp_kd", "Student-MLP (KD)"
    return ds, name, key, int(seed.group(1)) if seed else None


def main():
    p2 = stc_bridge.stc_root() / "results"
    multi = json.loads((p2 / "multiseed_summary.json").read_text())
    rows = []
    for ds in ("metr-la", "pems-bay"):
        split = runner.load_split(ds)
        single = {r["name"]: r for r in json.loads((p2 / f"results_{ds}.json").read_text())}
        for path in sorted(DRIVE.rglob("*.pth")):
            fds, name, key, seed = classify(path.name)
            if fds != ds:
                continue
            model, _ = runner.load_model(name, ds, split["num_nodes"], torch.device("cpu"), path=path)
            pred = runner.predict(model, split["x"], split, torch.device("cpu"), 256)
            mae, nmae = mae_nmae(pred, split["y"])
            row = {"dataset": ds, "file": str(path.relative_to(ROOT)), "model": name, "seed": seed,
                   "mae": mae, "nmae": nmae}
            if seed is not None and key in multi.get(ds, {}):
                vals = multi[ds][key]["nmae_vals"]
                row["paper2_nmae_seed"] = vals[seed] if seed < len(vals) else None
                row["matches_paper2_seed"] = (row["paper2_nmae_seed"] is not None
                                              and abs(round(nmae, 4) - row["paper2_nmae_seed"]) <= 1e-4)
            elif key in single:
                row["paper2_mae"] = single[key]["mae"]
                row["matches_paper2"] = abs(mae - single[key]["mae"]) / single[key]["mae"] < 5e-3
            rows.append(row)
            print(f"{ds} {path.parent.name}/{path.name}: nMAE={nmae:.4f} MAE={mae:.4f} "
                  f"{ {k: v for k, v in row.items() if k.startswith(('paper2', 'matches'))} }")
    (ROOT / "results" / "seed_checkpoints.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
