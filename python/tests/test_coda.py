import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(2)
parts = bt.closure(rng.uniform(0.1, 5, (40, 4)))


def test_closure_sums_to_total():
    np.testing.assert_allclose(bt.closure(parts * 7, total=100).sum(axis=1), 100)


@pytest.mark.parametrize(
    "forward, inverse", [(bt.clr, bt.clr_inverse), (bt.alr, bt.alr_inverse), (bt.ilr, bt.ilr_inverse)]
)
def test_log_ratios_invert(forward, inverse):
    np.testing.assert_allclose(inverse(forward(parts)), parts, atol=1e-12)


def test_ilr_is_an_isometry():
    a, b = parts[:20], parts[20:]
    euclid = np.linalg.norm(bt.ilr(a) - bt.ilr(b), axis=1)
    np.testing.assert_allclose(bt.aitchison_distance(a, b), euclid, atol=1e-10)
    assert bt.ilr(parts).shape == (40, 3)
