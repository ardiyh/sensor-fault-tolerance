"""Zero-parameter reference forecasters.

Both operate on numpy arrays only — they never touch torch or a checkpoint.
"""
import numpy as np


def persistence_forecast(x: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value across the horizon.

    x: [W, N, In] input windows. Returns [W, N, horizon].
    """
    last = x[:, :, -1:]
    return np.repeat(last, horizon, axis=2)


def seasonal_naive_forecast(
    series: np.ndarray,
    window_starts: np.ndarray,
    input_len: int,
    horizon: int,
    period: int,
):
    """Forecast target t+k as the value observed one seasonal period earlier.

    series: [T, N] the FULL series (train+test) so the lookback is available.
    window_starts: indices s such that inputs are series[s : s+input_len]
        and targets are series[s+input_len : s+input_len+horizon].
    Returns (forecast [W, N, horizon], valid mask [W]).
    Windows whose lookback falls before the start of the series are marked invalid
    and their forecast rows are filled with NaN.
    """
    n_nodes = series.shape[1]
    out = np.full((len(window_starts), n_nodes, horizon), np.nan, dtype=float)
    valid = np.zeros(len(window_starts), dtype=bool)
    for i, s in enumerate(window_starts):
        t0 = s + input_len
        src0 = t0 - period
        if src0 < 0:
            continue
        out[i] = series[src0 : src0 + horizon].T
        valid[i] = True
    return out, valid
