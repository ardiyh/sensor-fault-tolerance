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
