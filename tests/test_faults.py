import numpy as np
import pytest

from src.faults import inject, n_faulty, select_faulty_nodes


def _x(w=4, n=6, l=5, seed=0):
    return np.random.default_rng(seed).uniform(10, 70, size=(w, n, l)).astype(np.float32)


def _path_adj(n):
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 0.5
    return a


# ---- selection ------------------------------------------------------------------------------

def test_n_faulty_rounds_and_rejects_bad_fraction():
    assert n_faulty(207, 0.0) == 0
    assert n_faulty(207, 0.05) == 10
    assert n_faulty(207, 0.5) == 104
    with pytest.raises(ValueError):
        n_faulty(207, 1.5)


@pytest.mark.parametrize("pattern", ["random", "clustered"])
def test_selection_size_and_determinism(pattern):
    adj = _path_adj(50)
    m1 = select_faulty_nodes(50, 0.2, pattern, np.random.default_rng(3), adj)
    m2 = select_faulty_nodes(50, 0.2, pattern, np.random.default_rng(3), adj)
    assert m1.dtype == bool and m1.sum() == 10
    assert np.array_equal(m1, m2)


def test_clustered_on_a_path_is_contiguous():
    mask = select_faulty_nodes(30, 0.3, "clustered", np.random.default_rng(1), _path_adj(30))
    idx = np.where(mask)[0]
    assert idx.size == 9
    assert idx.max() - idx.min() == idx.size - 1


def test_clustered_takes_strongest_neighbour_first():
    # seed is forced to node 0 by making it the only node; then 0 links to 1 (0.9) and 2 (0.1)
    adj = np.zeros((3, 3), dtype=np.float32)
    adj[0, 1], adj[0, 2] = 0.9, 0.1

    class FirstNode:
        def choice(self, a, *args, **kwargs):
            return a[0]

    mask = select_faulty_nodes(3, 2 / 3, "clustered", FirstNode(), adj)
    assert mask.tolist() == [True, True, False]


def test_clustered_reseeds_when_component_is_exhausted():
    adj = np.zeros((6, 6), dtype=np.float32)
    adj[0, 1] = adj[1, 0] = 1.0  # component {0, 1}; nodes 2..5 isolated
    mask = select_faulty_nodes(6, 0.5, "clustered", np.random.default_rng(0), adj)
    assert mask.sum() == 3


def test_selection_rejects_unknown_pattern_and_missing_adjacency():
    with pytest.raises(ValueError):
        select_faulty_nodes(10, 0.2, "ring", np.random.default_rng(0))
    with pytest.raises(ValueError):
        select_faulty_nodes(10, 0.2, "clustered", np.random.default_rng(0), adj=None)


# ---- injection ------------------------------------------------------------------------------

MASK = np.array([True, False, True, False, False, False])


def test_dropout_zeros_only_faulty_sensors_and_copies():
    x = _x()
    before = x.copy()
    out = inject(x, MASK, "dropout")
    assert np.array_equal(x, before), "input must not be modified in place"
    assert (out[:, MASK, :] == 0).all()
    assert np.array_equal(out[:, ~MASK, :], x[:, ~MASK, :])


def test_stuck_repeats_first_sample():
    x = _x()
    out = inject(x, MASK, "stuck")
    assert np.array_equal(out[:, MASK, :], np.repeat(x[:, MASK, :1], x.shape[2], axis=2))
    assert np.array_equal(out[:, ~MASK, :], x[:, ~MASK, :])


def test_noise_has_requested_scale_on_faulty_sensors_only():
    x = _x(w=400, l=12)
    out = inject(x, MASK, "noise", rng=np.random.default_rng(0), sigma=5.0)
    d = out - x
    assert np.array_equal(d[:, ~MASK, :], np.zeros_like(d[:, ~MASK, :]))
    assert abs(d[:, MASK, :].std() - 5.0) < 0.1
    assert abs(d[:, MASK, :].mean()) < 0.1


def test_bias_adds_constant():
    x = _x()
    out = inject(x, MASK, "bias", bias=-7.5)
    assert np.allclose(out[:, MASK, :] - x[:, MASK, :], -7.5)
    assert np.array_equal(out[:, ~MASK, :], x[:, ~MASK, :])


@pytest.mark.parametrize("kind", ["dropout", "stuck", "noise", "bias"])
def test_empty_mask_is_identity(kind):
    x = _x()
    out = inject(x, np.zeros(6, dtype=bool), kind, rng=np.random.default_rng(0), sigma=1.0, bias=1.0)
    assert np.array_equal(out, x)


def test_inject_validates_arguments():
    x = _x()
    with pytest.raises(ValueError):
        inject(x, MASK, "melt")
    with pytest.raises(ValueError):
        inject(x, MASK, "noise")  # no sigma / rng
    with pytest.raises(ValueError):
        inject(x, MASK, "bias")  # no bias
    with pytest.raises(ValueError):
        inject(x[0], MASK, "dropout")  # wrong rank
