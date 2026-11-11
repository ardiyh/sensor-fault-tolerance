"""Training-free repair front-end: flag implausible sensor windows, replace them by neighbours.

Detection rule, per sensor and input window (physical units):
    flagged  if the window is constant (max - min <= tol)   -> stuck, and all-zero dropout
             or at least `min_zeros` readings are exactly 0  -> partial dropout / native outage
Imputation: a flagged window is replaced by the adjacency-weighted mean of the windows of its
unflagged neighbours. With no unflagged neighbour it falls back to the mean of all unflagged
sensors in that window; if every sensor is flagged the window is left unchanged.
"""
import torch


def detect(x: torch.Tensor, min_zeros: int = 6, tol: float = 1e-6) -> torch.Tensor:
    """x [W, N, L] -> flags [W, N] (bool)."""
    constant = (x.amax(dim=-1) - x.amin(dim=-1)) <= tol
    zeros = (x == 0).sum(dim=-1) >= min_zeros
    return constant | zeros


def impute(x: torch.Tensor, flags: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Replace flagged windows. x [W, N, L], flags [W, N], adj [N, N] non-negative weights."""
    a = torch.maximum(adj, adj.T).to(x.dtype)
    a = a * (1 - torch.eye(a.shape[0], device=a.device, dtype=a.dtype))
    ok = (~flags).to(x.dtype)                                   # [W, N]
    num = torch.einsum("nm,wml->wnl", a, x * ok.unsqueeze(-1))  # weighted sum over good neighbours
    den = ok @ a.T                                              # [W, N]
    neigh = num / den.clamp_min(1e-12).unsqueeze(-1)
    net = (x * ok.unsqueeze(-1)).sum(1, keepdim=True) / ok.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
    fill = torch.where((den > 0).unsqueeze(-1), neigh, net.expand_as(x))
    any_ok = (ok.sum(1) > 0).view(-1, 1, 1)
    replace = flags.unsqueeze(-1) & any_ok
    return torch.where(replace, fill, x)


def repair(x: torch.Tensor, adj: torch.Tensor, **detect_kw):
    """Detect and impute in one call. Returns (repaired x, flags)."""
    flags = detect(x, **detect_kw)
    return impute(x, flags, adj), flags
