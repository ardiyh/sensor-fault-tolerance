"""Does Paper 2's DCRNN mix information across sensors at all?

Zeroes the inputs of every sensor except one and measures how much that sensor's forecast moves.
With working diffusion convolution it must move. Paper 2's DiffusionConv uses
torch.einsum('nn,bnc->bnc', support, d), which takes the diagonal of the transition matrix; the
adjacency has no self-loops, so the diagonal is zero and every diffusion term vanishes.

Usage: python -m scripts.check_dcrnn_mixing --dataset metr-la
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import runner

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr-la")
    ap.add_argument("--windows", type=int, default=64)
    args = ap.parse_args()
    dev = torch.device("cpu")
    split = runner.load_split(args.dataset)
    x = split["x"][:args.windows].copy()
    out = {"dataset": args.dataset, "windows": args.windows}
    for name in ("dcrnn", "student_mlp_baseline", "student_attn_baseline"):
        model, path = runner.load_model(name, args.dataset, split["num_nodes"], dev)
        p0 = runner.predict(model, x, split, dev, args.windows)
        moves = []
        for keep in (0, 50, 100):  # three probe sensors
            xk = np.zeros_like(x)
            xk[:, keep, :] = x[:, keep, :]
            pk = runner.predict(model, xk, split, dev, args.windows)
            moves.append(float(np.abs(pk[:, keep] - p0[:, keep]).max()))
        out[name] = {"checkpoint": path.name,
                     "max_forecast_change_at_kept_sensor_when_all_others_zeroed": moves}
    model, _ = runner.load_model("dcrnn", args.dataset, split["num_nodes"], dev)
    runner.predict(model, x[:1], split, dev, 1)  # builds model.supports
    s = model.supports[0]
    out["dcrnn_support_diagonal_max"] = float(s.diag().abs().max())
    out["dcrnn_support_offdiagonal_max"] = float(s.abs().max())
    path = ROOT / "results" / f"dcrnn_mixing_{args.dataset}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
