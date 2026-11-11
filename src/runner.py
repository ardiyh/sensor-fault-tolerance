"""Test split, model loading, and inference on (possibly corrupted) physical-unit windows.

The split reproduces Paper 2 exactly (evaluate_all.py: scaler fitted on the first 70%, test is
data[int(0.8*L):], 12-in / 12-out sliding windows). Windows are built in physical units so
faults can be injected before normalisation; the model then sees scaler.transform(x).
"""
from pathlib import Path

import numpy as np
import torch

from src import stc_bridge

INPUT_LEN = 12
HORIZON = 12
HOLD = 36  # stuck_long: the frozen value is the reading 36 steps (3 h) before the window start
MODELS = ("persistence", "seasonal_naive", "dcrnn", "student_mlp_baseline", "student_mlp_kd",
          "student_attn_baseline", "student_attn_kd_at")
LEARNED = MODELS[2:]

# Paper 2 checkpoint names (same table as paper6_DAMS/DAMS-code/scripts/dump_predictions.py)
CHECKPOINTS = {
    "dcrnn": "dcrnn_{ds}.pth",
    "student_mlp_baseline": "student_baseline_{ds}.pth",
    "student_mlp_kd": "best_student_{ds}.pth",
    "student_attn_baseline": "student_attention_baseline_{ds}.pth",
    "student_attn_kd_at": "best_student_attention_{ds}_v1.pth",
}
LOCAL_CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def sliding(series: np.ndarray, start: int, n_windows: int, length: int) -> np.ndarray:
    """series [T, N] -> windows [W, N, length] with window w = series[start+w : start+w+length]."""
    idx = start + np.arange(n_windows)[:, None] + np.arange(length)[None, :]
    return np.ascontiguousarray(np.transpose(series[idx], (0, 2, 1)))


def load_split(dataset: str) -> dict:
    load_traffic_data, _, get_scaler = stc_bridge.data_api()
    data_tensor, edge_index, edge_weight, timestamps = load_traffic_data(dataset)
    per_node = stc_bridge.norm_recommendation()[dataset] == "per_node"
    if per_node:
        raise NotImplementedError("per-node scaling is not used for the traffic datasets here")
    scaler = get_scaler(data_tensor, train_ratio=0.7, per_node=False)
    series = data_tensor.squeeze(-1).numpy()  # [T, N] float32, physical units
    L, n = series.shape
    test_start = int(L * 0.8)
    n_windows = (L - test_start) - INPUT_LEN - HORIZON + 1
    delta = (timestamps[1] - timestamps[0]).total_seconds()
    period = int(round(86400 / delta))
    adj = np.zeros((n, n), dtype=np.float32)
    adj[edge_index[0].numpy(), edge_index[1].numpy()] = edge_weight.numpy()
    return {
        "dataset": dataset,
        "series": series,
        "x": sliding(series, test_start, n_windows, INPUT_LEN),
        "y": sliding(series, test_start + INPUT_LEN, n_windows, HORIZON),
        # seasonal-naive input: the HORIZON readings one period before each target block
        "x_sn": sliding(series, test_start + INPUT_LEN - period, n_windows, HORIZON),
        # stuck_long frozen values for each input window: [W, N]
        "hold": series[test_start - HOLD + np.arange(n_windows)],
        "hold_sn": series[test_start + INPUT_LEN - period - HOLD + np.arange(n_windows)],
        "edge_index": edge_index,
        "adj": adj,
        "scaler": scaler,
        "test_start": test_start,
        "n_windows": n_windows,
        "num_nodes": n,
        "period": period,
        "train_std": float(scaler.std),
        "train_mean": float(scaler.mean),
    }


def checkpoint_path(name: str, dataset: str, tag: str = None) -> Path:
    """Paper 2's checkpoint by default; a locally retrained one when `tag` is given."""
    fname = CHECKPOINTS[name].format(ds=dataset)
    if tag is None:
        return stc_bridge.stc_root() / "models" / fname
    return LOCAL_CKPT_DIR / fname.replace(".pth", f"_{tag}.pth")


def load_model(name: str, dataset: str, num_nodes: int, device, tag: str = None, path=None):
    """Load a checkpoint of model type `name`; `path` (if given) overrides the default name."""
    if name == "dcrnn_fixed":
        from src.dcrnn_fixed import DCRNNTeacher
        path = Path(path)
        model = DCRNNTeacher(num_nodes=num_nodes, in_channels=1, hidden_channels=64, out_channels=1,
                             num_layers=2, K=2, prediction_length=12, dropout=0.1).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        return model, path
    path = Path(path) if path is not None else checkpoint_path(name, dataset, tag)
    if not path.is_file():
        raise FileNotFoundError(f"{name} checkpoint missing for {dataset}: {path}")
    load_dcrnn, load_student_mlp, load_student_attention = stc_bridge.model_loaders()
    if name == "dcrnn":
        model = load_dcrnn(num_nodes, dataset, device, str(path))
    elif name.startswith("student_mlp"):
        model = load_student_mlp(num_nodes, str(path), device)
    else:
        model = load_student_attention(num_nodes, str(path), device)
    if model is None:
        raise RuntimeError(f"failed to load {path}")
    model.eval()
    return model, path


def predict(model, x_phys: np.ndarray, split: dict, device, batch_size: int = 64) -> np.ndarray:
    """x_phys [W, N, In] physical -> predictions [W, N, H] physical."""
    scaler = split["scaler"]
    edge_index = split["edge_index"].to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(x_phys), batch_size):
            xb = scaler.transform(torch.from_numpy(np.ascontiguousarray(x_phys[i:i + batch_size])))
            out.append(model(xb.to(device), edge_index).float().cpu())
    return scaler.inverse_transform(torch.cat(out).numpy())


def predict_t(model, x_phys: torch.Tensor, split: dict, batch_size: int = 1024) -> torch.Tensor:
    """Torch version of predict(); x_phys [W, N, In] on the model's device -> physical forecasts."""
    mean, std = float(split["scaler"].mean), float(split["scaler"].std)
    edge_index = split["edge_index"].to(x_phys.device)
    out = []
    with torch.no_grad():
        for i in range(0, len(x_phys), batch_size):
            xb = (x_phys[i:i + batch_size] - mean) / std
            out.append(model(xb, edge_index).float())
    return torch.cat(out) * std + mean


def naive_predict(name: str, x_phys: np.ndarray) -> np.ndarray:
    """Persistence repeats the last input sample; seasonal naive returns its (period-old) input."""
    if name == "persistence":
        return x_phys[:, :, -1:].repeat_interleave(HORIZON, 2) if torch.is_tensor(x_phys) \
            else np.repeat(x_phys[:, :, -1:], HORIZON, axis=2)
    if name == "seasonal_naive":
        return x_phys.clone() if torch.is_tensor(x_phys) else x_phys.copy()
    raise ValueError(name)
