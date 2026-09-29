import ceres as cs
import numpy as np
import pytest


def channels(nx=80, ny=80):
    """Meandering channels (code 1) in a background (code 0)."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    codes = ((y - 6 * np.sin(x / 7)) % 16 < 5).astype(float)
    return cs.BlockModel((0, 0), (1, 1), (nx, ny), attributes={"facies": codes.ravel()})


def wave(nx=80, ny=80):
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    v = np.sin(x / 6) * np.cos(y / 9) + x / 80
    return cs.BlockModel((0, 0), (1, 1), (nx, ny), attributes={"grade": v.ravel()})


def grid(nx=60, ny=60):
    return cs.BlockModel((100, 200), (5, 5), (nx, ny))


def test_categorical_realizations_keep_the_training_image_proportions():
    ti = channels()
    iq = cs.ImageQuilting(ti, "facies", patch_size=16)
    assert iq.categorical
    s = iq.simulate(grid(), n=12, seed=3, keep=True, progress=False)
    assert isinstance(s, cs.CategoricalSummary)
    assert s.realizations.shape == (12, 3600)
    assert np.isin(s.realizations, [0, 1]).all()
    want = ti["facies"].mean()
    assert abs(s.proportions[:, 1].mean() - want) < 0.04


def test_continuous_histogram_matches_the_training_image():
    ti = wave()
    iq = cs.ImageQuilting(ti, "grade", patch_size=14)
    assert not iq.categorical
    s = iq.simulate(grid(), n=10, keep=True, progress=False)
    assert isinstance(s, cs.SimulationSummary)
    reference = np.asarray(ti["grade"])
    span = np.ptp(reference)
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    np.testing.assert_allclose(np.quantile(s.realizations, q), np.quantile(reference, q), atol=0.08 * span)


def test_integer_codes_can_be_continuous():
    iq = cs.ImageQuilting(channels(), "facies", categorical=False, patch_size=10)
    assert not iq.categorical
    s = iq.simulate(grid(20, 20), n=2, progress=False)
    assert isinstance(s, cs.SimulationSummary)


def test_hard_data_are_reproduced():
    ti = channels()
    targets = grid()
    rng = np.random.default_rng(1)
    rows = rng.choice(3600, 80, replace=False)
    xyz = targets.centroids[rows]
    codes = rng.integers(0, 2, 80)
    iq = cs.ImageQuilting(ti, "facies", patch_size=15).fit(xyz, codes)
    s = iq.simulate(targets, n=4, keep=True, progress=False)
    np.testing.assert_array_equal(s.realizations[:, rows], np.tile(codes, (4, 1)))
    with pytest.raises(cs.InvalidInput, match="code"):
        iq.fit(xyz[:1], [3])


def test_masked_targets_and_3d():
    masked = grid(30, 30).mask(np.arange(900) % 3 != 0)
    s = cs.ImageQuilting(channels(), "facies", patch_size=10).simulate(masked, n=2, keep=True, progress=False)
    assert s.realizations.shape == (2, 600)
    x, _, z = np.meshgrid(np.arange(16), np.arange(16), np.arange(8), indexing="ij")
    layers = ((x + 2 * z) // 4 % 2).astype(float)
    ti = cs.BlockModel((0, 0, 0), (1, 1, 1), (16, 16, 8), attributes={"f": layers.ravel(order="F")})
    targets = cs.BlockModel((0, 0, 0), (1, 1, 1), (12, 11, 8))
    s = cs.ImageQuilting(ti, "f", patch_size=(6, 6, 4)).simulate(targets, n=1, keep=True, progress=False)
    assert np.isin(s.realizations, [0, 1]).all()


def test_bad_parameters():
    ti = channels(30, 30)
    for kwargs, match in [
        ({"patch_size": (12, 0, 1)}, "patch_size"),
        ({"patch_size": "large"}, "patch_size"),
        ({"patch_size": 12, "overlap": 7}, "overlap"),
        ({"n_best": 0}, "n_best"),
        ({"data_weight": -1.0}, "data_weight"),
    ]:
        with pytest.raises(ValueError, match=match):
            cs.ImageQuilting(ti, "facies", **kwargs)
