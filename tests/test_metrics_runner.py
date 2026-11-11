import numpy as np
import torch

from src import runner, stc_bridge
from src.metrics import per_node_mae, split_metrics


def _pair(seed=0, w=5, n=8, h=12):
    rng = np.random.default_rng(seed)
    true = rng.uniform(0, 70, size=(w, n, h)).astype(np.float32)
    true[0, 1, :4] = 0.0  # a few native zeros
    pred = true + rng.normal(0, 3, size=true.shape).astype(np.float32)
    return pred, true


def test_all_split_matches_paper2_compute_metrics():
    stc_bridge._ensure_on_path()
    from utils.common import compute_metrics
    pred, true = _pair()
    m = split_metrics(pred, true, np.zeros(8, dtype=bool))
    mae, _, _, nmae = compute_metrics(pred, true)
    assert np.isclose(m["all_mae"], mae) and np.isclose(m["all_nmae"], nmae)
    assert np.isclose(m["healthy_mae"], mae)
    assert np.isnan(m["faulty_mae"])


def test_healthy_and_faulty_partition_the_targets():
    pred, true = _pair()
    mask = np.array([True, True, False, False, False, False, False, True])
    m = split_metrics(pred, true, mask)
    weighted = (m["healthy_mae"] * (~mask).sum() + m["faulty_mae"] * mask.sum()) / 8
    assert np.isclose(weighted, m["all_mae"])


def test_nonzero_variant_drops_native_zeros():
    pred, true = _pair()
    m = split_metrics(pred, true, np.zeros(8, dtype=bool))
    nz = true != 0
    assert np.isclose(m["all_mae_nz"], np.abs(pred[nz] - true[nz]).mean())
    pn = per_node_mae(pred, true, nonzero_only=True)
    assert np.isclose(pn[1], np.abs(pred[:, 1][true[:, 1] != 0] - true[:, 1][true[:, 1] != 0]).mean())


def test_sliding_windows_match_paper2_dataset():
    _, SlidingWindowDataset, _ = stc_bridge.data_api()
    series = np.random.default_rng(0).normal(size=(60, 4)).astype(np.float32)
    ds = SlidingWindowDataset(torch.from_numpy(series[:, :, None]), input_len=12, output_len=12)
    x = runner.sliding(series, 0, len(ds), 12)
    y = runner.sliding(series, 12, len(ds), 12)
    for i in (0, 7, len(ds) - 1):
        xi, yi = ds[i]
        assert np.array_equal(x[i], xi.numpy()) and np.array_equal(y[i], yi.numpy())


def test_persistence_repeats_last_input():
    x = np.arange(2 * 3 * 12, dtype=np.float32).reshape(2, 3, 12)
    p = runner.naive_predict("persistence", x)
    assert p.shape == (2, 3, 12) and (p == x[:, :, -1:]).all()


def test_subset_metrics_from_sums_match_split_metrics():
    from src.metrics import node_sums, subset_metrics, sums_to_numpy
    pred, true = _pair(seed=3)
    mask = np.array([True, False, False, True, False, False, False, True])
    ref = split_metrics(pred, true, mask)
    fast = subset_metrics(sums_to_numpy(node_sums(torch.from_numpy(pred), torch.from_numpy(true))), mask)
    for k, v in ref.items():
        assert np.isclose(fast[k], v, rtol=1e-5, equal_nan=True), k


def test_stuck_long_and_torch_naive():
    from src.faults import inject
    x = np.arange(2 * 3 * 12, dtype=np.float32).reshape(2, 3, 12)
    hold = np.full((2, 3), 7.0, dtype=np.float32)
    out = inject(x, np.array([False, True, False]), "stuck_long", hold=hold)
    assert (out[:, 1] == 7.0).all() and np.array_equal(out[:, [0, 2]], x[:, [0, 2]])
    p = runner.naive_predict("persistence", torch.from_numpy(x))
    assert torch.equal(p, torch.from_numpy(runner.naive_predict("persistence", x)))
