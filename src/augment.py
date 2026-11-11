"""Sensor-dropout augmentation for retraining task-only students.

Each training sample draws its own dropout rate r ~ U(0, max_frac); every sensor of that sample is
then dropped with probability r, i.e. its whole input window is replaced by `fill`, the normalised
value of a physical zero reading (the dataset's own missing-data code). Targets are not touched.
"""
import torch


def zero_fill_value(scaler) -> float:
    """Normalised value of a physical reading of 0."""
    if scaler is None:
        return 0.0
    return float(scaler.transform(torch.zeros(1)).reshape(-1)[0])


def sensor_dropout(x: torch.Tensor, max_frac: float, fill: float,
                   generator: torch.Generator = None) -> torch.Tensor:
    """x [B, N, T] -> copy with a per-sample random share of sensors set to `fill`."""
    if not 0.0 <= max_frac <= 1.0:
        raise ValueError(f"max_frac must be in [0, 1], got {max_frac}")
    if max_frac == 0.0:
        return x
    B, N, _ = x.shape
    rate = torch.rand(B, 1, device=x.device, generator=generator) * max_frac
    drop = torch.rand(B, N, device=x.device, generator=generator) < rate
    return x.masked_fill(drop.unsqueeze(-1), fill)
