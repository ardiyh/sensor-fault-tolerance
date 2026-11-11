import numpy as np
import torch

from src.augment import sensor_dropout, zero_fill_value
from src.repair import detect, impute, repair


def _adj():
    # path 0-1-2-3, node 4 isolated
    a = torch.zeros(5, 5)
    for i, j, w in ((0, 1, 1.0), (1, 2, 0.5), (2, 3, 1.0)):
        a[i, j] = w
    return a


def _x():
    g = torch.Generator().manual_seed(0)
    return 30 + 30 * torch.rand(3, 5, 12, generator=g)


def test_detect_flags_constant_and_zero_windows_only():
    x = _x()
    x[0, 1] = 0.0            # dropout
    x[1, 2] = x[1, 2, 0]     # stuck
    x[2, 3, :6] = 0.0        # six zeros -> flagged
    x[2, 0, :5] = 0.0        # five zeros -> not flagged
    f = detect(x)
    expect = torch.zeros(3, 5, dtype=torch.bool)
    expect[0, 1] = expect[1, 2] = expect[2, 3] = True
    assert torch.equal(f, expect)


def test_impute_uses_weighted_good_neighbours_and_leaves_rest():
    x = _x()
    flags = torch.zeros(3, 5, dtype=torch.bool)
    flags[0, 1] = True
    out = impute(x, flags, _adj())
    # neighbours of 1: node 0 (w=1.0) and node 2 (w=0.5)
    expect = (1.0 * x[0, 0] + 0.5 * x[0, 2]) / 1.5
    assert torch.allclose(out[0, 1], expect)
    keep = ~flags.unsqueeze(-1).expand_as(x)
    assert torch.equal(out[keep], x[keep])


def test_impute_skips_flagged_neighbours_and_falls_back_to_network_mean():
    x = _x()
    flags = torch.zeros(3, 5, dtype=torch.bool)
    flags[0, 1] = flags[0, 0] = True          # 1's neighbours: 0 flagged, 2 good
    flags[0, 4] = True                         # isolated -> network mean of good sensors
    out = impute(x, flags, _adj())
    assert torch.allclose(out[0, 1], x[0, 2])
    good = [2, 3]
    assert torch.allclose(out[0, 4], x[0, good].mean(0))


def test_all_flagged_window_is_left_unchanged():
    x = _x()
    flags = torch.zeros(3, 5, dtype=torch.bool)
    flags[1] = True
    out = impute(x, flags, _adj())
    assert torch.equal(out[1], x[1])


def test_repair_restores_a_dropped_sensor_close_to_its_neighbours():
    x = _x()
    x[:, 1] = 0.0
    out, flags = repair(x, _adj())
    assert flags[:, 1].all() and not flags[:, [0, 2, 3]].any()
    assert (out[:, 1] > 0).all()


def test_sensor_dropout_fills_whole_windows_and_respects_rate():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4000, 50, 12, generator=g)
    out = sensor_dropout(x, 0.5, fill=-2.5, generator=g)
    dropped = (out == -2.5).all(-1)
    untouched = (out == x).all(-1)
    assert (dropped | untouched).all()
    assert abs(dropped.float().mean().item() - 0.25) < 0.01   # E[U(0, 0.5)] = 0.25
    assert sensor_dropout(x, 0.0, fill=-2.5) is x


def test_zero_fill_value_is_normalised_physical_zero():
    class S:
        def transform(self, t):
            return (t - 54.0) / 20.0
    assert np.isclose(zero_fill_value(S()), -2.7)
    assert zero_fill_value(None) == 0.0
