"""Error on healthy-sensor and faulty-sensor targets.

MAE and nMAE follow Paper 2's utils/common.py::compute_metrics exactly (unmasked: the native
zeros of METR-LA count as targets). The `_nz` variants drop targets whose true value is 0, so
the dataset's own missing readings can be separated from injected faults.
"""
import numpy as np


def mae_nmae(pred: np.ndarray, true: np.ndarray):
    """Same arithmetic as compute_metrics: MAE, and MAE / mean(|true|)."""
    if pred.size == 0:
        return float("nan"), float("nan")
    mae = float(np.mean(np.abs(pred - true)))
    denom = float(np.mean(np.abs(true)))
    return mae, (mae / denom if denom > 1e-8 else 0.0)


def per_node_mae(pred: np.ndarray, true: np.ndarray, nonzero_only: bool = False) -> np.ndarray:
    """MAE of each sensor over windows and horizon steps. pred/true: [W, N, H] -> [N]."""
    err = np.abs(pred - true)
    if not nonzero_only:
        return err.mean(axis=(0, 2))
    keep = true != 0
    with np.errstate(invalid="ignore"):
        return (err * keep).sum(axis=(0, 2)) / keep.sum(axis=(0, 2))


def split_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict:
    """Metrics on all, healthy (~mask) and faulty (mask) target sensors."""
    out = {}
    for name, sel in (("all", np.ones_like(mask)), ("healthy", ~mask), ("faulty", mask)):
        p, t = pred[:, sel, :], true[:, sel, :]
        out[f"{name}_mae"], out[f"{name}_nmae"] = mae_nmae(p, t)
        nz = t != 0
        out[f"{name}_mae_nz"], out[f"{name}_nmae_nz"] = mae_nmae(p[nz], t[nz])
    return out


# ---- fast path: per-node sums, computed once per prediction (torch, any device) -------------

def node_sums(pred, true) -> dict:
    """pred/true [W, N, H] tensors -> float64 per-node sums from which every subset metric follows."""
    import torch
    err = (pred - true).abs()
    nz = true != 0
    dims = (0, 2)
    return {
        "err": err.sum(dims, dtype=torch.float64), "abs": true.abs().sum(dims, dtype=torch.float64),
        "cnt": torch.full((true.shape[1],), true.shape[0] * true.shape[2], dtype=torch.float64,
                          device=true.device),
        "err_nz": (err * nz).sum(dims, dtype=torch.float64),
        "abs_nz": (true.abs() * nz).sum(dims, dtype=torch.float64),
        "cnt_nz": nz.sum(dims, dtype=torch.float64),
    }


def sums_to_numpy(sums: dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in sums.items()}


def subset_metrics(sums: dict, mask: np.ndarray) -> dict:
    """Same keys as split_metrics(), from per-node sums (numpy) and a faulty-sensor mask [N]."""
    out = {}
    for name, sel in (("all", np.ones_like(mask)), ("healthy", ~mask), ("faulty", mask)):
        for suf, e, a, c in (("", "err", "abs", "cnt"), ("_nz", "err_nz", "abs_nz", "cnt_nz")):
            cnt = sums[c][sel].sum()
            if cnt == 0:
                out[f"{name}_mae{suf}"] = out[f"{name}_nmae{suf}"] = float("nan")
                continue
            err, ab = sums[e][sel].sum(), sums[a][sel].sum()
            out[f"{name}_mae{suf}"] = float(err / cnt)
            out[f"{name}_nmae{suf}"] = float(err / ab) if ab / cnt > 1e-8 else 0.0
    return out
