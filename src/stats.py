"""Statistics policy for Paper 7 (copied from Paper 6, DAMS-code/src/stats.py).

Every model comparison reports p-value AND win rate AND median delta. A mean-only
comparison is not acceptable output.
"""
import numpy as np
from scipy import stats


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def win_rate(candidate, reference) -> float:
    a, b = np.asarray(candidate, float), np.asarray(reference, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() == 0:
        return float("nan")
    return float(np.mean(a[ok] < b[ok]))


def median_delta(candidate, reference) -> float:
    a, b = np.asarray(candidate, float), np.asarray(reference, float)
    return float(np.nanmedian(a - b))


def paired_node_test(candidate, reference) -> dict:
    """Wilcoxon signed-rank across nodes, plus the descriptive companions."""
    a, b = np.asarray(candidate, float), np.asarray(reference, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if np.allclose(a, b):
        p = 1.0
    else:
        p = float(stats.wilcoxon(a, b).pvalue)
    return {
        "n": int(a.size),
        "p_value": p,
        "win_rate": win_rate(a, b),
        "median_delta": median_delta(a, b),
        "mean_delta": float(np.mean(a - b)),
    }
