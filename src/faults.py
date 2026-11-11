"""Sensor fault injection on input windows, in physical units.

A fault is a property of a sensor, not of a window: the same set of faulty sensors is
corrupted in every input window of the evaluation period. Targets are never touched.

Fault models (x is a window of physical readings, faulty sensor n):
    dropout  x[n, t] = 0                    (the dataset's own code for a missing reading)
    stuck    x[n, t] = x[n, 0]              (sensor froze at the first sample of the window)
    stuck_long x[n, t] = h[n]               (sensor froze HOLD steps before the window start;
                                             h is that older reading, supplied by the caller)
    noise    x[n, t] = x[n, t] + e,  e ~ N(0, sigma^2)
    bias     x[n, t] = x[n, t] + b

Spatial patterns:
    random     k sensors drawn uniformly without replacement
    clustered  one seed sensor, then its nearest neighbours on the graph (a station outage)
"""
import numpy as np

FAULT_TYPES = ("dropout", "stuck", "stuck_long", "noise", "bias")
PATTERNS = ("random", "clustered")


def n_faulty(n_nodes: int, frac: float) -> int:
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"fault fraction must be in [0, 1], got {frac}")
    return int(round(frac * n_nodes))


def _clustered(adj: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Grow a connected set from a random seed, always adding the outside node with the
    strongest adjacency weight to the set. The weights are Gaussian kernels of road distance,
    so this adds the nearest remaining neighbour. A new seed is drawn if the component runs out.
    """
    n = adj.shape[0]
    w = np.maximum(adj, adj.T).astype(float)
    np.fill_diagonal(w, 0.0)
    chosen = np.zeros(n, dtype=bool)
    link = np.zeros(n)  # strongest weight from each node to the chosen set
    while chosen.sum() < k:
        cand = np.where(~chosen & (link > 0))[0]
        if cand.size == 0:
            nxt = rng.choice(np.where(~chosen)[0])
        else:
            nxt = cand[np.argmax(link[cand])]
        chosen[nxt] = True
        link = np.maximum(link, w[nxt])
        link[chosen] = 0.0
    return chosen


def select_faulty_nodes(n_nodes: int, frac: float, pattern: str, rng: np.random.Generator,
                        adj: np.ndarray = None) -> np.ndarray:
    """Return a boolean mask [N] of faulty sensors."""
    k = n_faulty(n_nodes, frac)
    if pattern == "random":
        mask = np.zeros(n_nodes, dtype=bool)
        mask[rng.choice(n_nodes, size=k, replace=False)] = True
        return mask
    if pattern == "clustered":
        if adj is None or adj.shape != (n_nodes, n_nodes):
            raise ValueError("clustered pattern needs an [N, N] adjacency matrix")
        return _clustered(adj, k, rng)
    raise ValueError(f"unknown pattern '{pattern}', expected one of {PATTERNS}")


def inject(x: np.ndarray, mask: np.ndarray, kind: str, rng: np.random.Generator = None,
           sigma: float = None, bias: float = None, hold: np.ndarray = None) -> np.ndarray:
    """Corrupt the faulty sensors of every window. x: [W, N, L] physical units. Returns a copy."""
    if x.ndim != 3 or mask.shape != (x.shape[1],):
        raise ValueError(f"expected x [W, N, L] and mask [N], got {x.shape} and {mask.shape}")
    out = x.copy()
    if not mask.any():
        return out
    if kind == "dropout":
        out[:, mask, :] = 0.0
    elif kind == "stuck":
        out[:, mask, :] = x[:, mask, :1]
    elif kind == "stuck_long":
        if hold is None or hold.shape != x.shape[:2]:
            raise ValueError("stuck_long needs hold values of shape [W, N]")
        out[:, mask, :] = hold[:, mask, None]
    elif kind == "noise":
        if sigma is None or rng is None:
            raise ValueError("noise needs sigma and rng")
        out[:, mask, :] += rng.normal(0.0, sigma, size=out[:, mask, :].shape).astype(x.dtype)
    elif kind == "bias":
        if bias is None:
            raise ValueError("bias needs a bias value")
        out[:, mask, :] += bias
    else:
        raise ValueError(f"unknown fault '{kind}', expected one of {FAULT_TYPES}")
    return out
