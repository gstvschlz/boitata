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


def wide_channels(nx=80, ny=80):
    """Channels wide enough for a patch to lie inside one."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    codes = ((y - 6 * np.sin(x / 9)) % 24 < 10).astype(float)
    return cs.BlockModel((0, 0), (1, 1), (nx, ny), attributes={"facies": codes.ravel()})


def smooth(image, r=2):
    """Mean over each cell's box of 2r + 1 cells, edges repeated."""
    k = 2 * r + 1
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(image, r, mode="edge"), (k, k))
    return windows.mean(axis=(-1, -2))


def test_soft_probabilities_steer_the_realizations():
    side = 64
    west = np.tile(np.arange(side) < side // 2, side).astype(float)
    soft = np.column_stack([1 - west, west])
    targets = cs.BlockModel((0, 0), (1, 1), (side, side), attributes={"p0": 1 - west, "p1": west})
    iq = cs.ImageQuilting(wide_channels(), "facies", patch_size=12, n_best=3)
    free = iq.simulate(targets, n=6, keep=True, progress=False).realizations
    steered = iq.simulate(targets, n=6, soft=soft, keep=True, progress=False).realizations
    by_name = iq.simulate(targets, n=6, soft=["p0", "p1"], keep=True, progress=False).realizations
    np.testing.assert_array_equal(by_name, steered)
    is_west = west.astype(bool)
    assert steered[:, is_west].mean() > free[:, is_west].mean() + 0.1
    assert steered[:, ~is_west].mean() < free[:, ~is_west].mean() - 0.05
    # A row of nulls carries no probabilities.
    none = np.full((side * side, 2), np.nan)
    np.testing.assert_array_equal(
        iq.simulate(targets, n=6, soft=none, keep=True, progress=False).realizations, free
    )


def test_soft_probabilities_are_checked():
    targets = grid(10, 10)
    iq = cs.ImageQuilting(channels(), "facies", patch_size=8)
    for soft, match in [
        (np.ones((100, 3)), "shape"),
        (np.ones((99, 2)), "soft"),
        (np.array([[np.nan, 1.0]] * 100), "missing"),
        (np.array([[-1.0, 1.0]] * 100), "soft"),
    ]:
        with pytest.raises(ValueError, match=match):
            iq.simulate(targets, n=1, soft=soft, progress=False)
    with pytest.raises(ValueError, match="categorical"):
        cs.ImageQuilting(wave(), "grade", patch_size=8).simulate(
            targets, n=1, soft=np.ones((100, 2)), progress=False
        )


def test_a_secondary_variable_pulls_realizations_toward_its_pattern():
    ti = channels()
    facies = np.asarray(ti["facies"]).reshape(80, 80)
    ti = cs.BlockModel(
        (0, 0), (1, 1), (80, 80), attributes={"facies": facies.ravel(), "s": smooth(facies).ravel()}
    )
    x, y = np.meshgrid(np.arange(60), np.arange(60), indexing="xy")
    reference = ((y - 9 * np.cos((x + 20) / 11)) % 16 < 5).astype(float)
    targets = cs.BlockModel((0, 0), (1, 1), (60, 60), attributes={"s": smooth(reference).ravel()})

    def correlation(realizations):
        return np.mean([np.corrcoef(r, reference.ravel())[0, 1] for r in realizations])

    free = cs.ImageQuilting(ti, "facies", patch_size=16).simulate(targets, n=4, keep=True, progress=False)
    steered = cs.ImageQuilting(ti, "facies", patch_size=16, secondary="s", secondary_weight=2.0).simulate(
        targets, n=4, secondary="s", keep=True, progress=False
    )
    with_, without = correlation(steered.realizations), correlation(free.realizations)
    assert with_ > 0.5 and with_ > without + 0.4, (with_, without)
    # An array works as well as a column, and weight 0 leaves the variable out.
    iq = cs.ImageQuilting(ti, "facies", patch_size=16, secondary="s", secondary_weight=0.0)
    same = iq.simulate(targets, n=4, secondary=smooth(reference).ravel(), keep=True, progress=False)
    np.testing.assert_array_equal(same.realizations, free.realizations)


def test_a_continuous_primary_takes_a_secondary_variable():
    ti = wave()
    grade = np.asarray(ti["grade"])
    ti = cs.BlockModel((0, 0), (1, 1), (80, 80), attributes={"grade": grade, "s": -grade})
    iq = cs.ImageQuilting(ti, "grade", patch_size=10, secondary="s")
    s = iq.simulate(grid(20, 20), n=2, secondary=np.linspace(-1, 1, 400), progress=False)
    assert isinstance(s, cs.SimulationSummary)


def test_secondary_is_given_to_both_or_neither():
    ti = cs.BlockModel(
        (0, 0), (1, 1), (30, 30), attributes={"f": np.arange(900) % 2 * 1.0, "s": np.arange(900.0)}
    )
    targets = grid(10, 10)
    with pytest.raises(ValueError, match="neither"):
        cs.ImageQuilting(ti, "f", patch_size=5, secondary="s").simulate(targets, n=1, progress=False)
    with pytest.raises(ValueError, match="neither"):
        cs.ImageQuilting(ti, "f", patch_size=5).simulate(targets, n=1, secondary=np.ones(100), progress=False)
    with pytest.raises(ValueError, match="secondary"):
        cs.ImageQuilting(ti, "f", patch_size=5, secondary="s").simulate(
            targets, n=1, secondary=np.ones(99), progress=False
        )
    with pytest.raises(ValueError, match="secondary_weight"):
        cs.ImageQuilting(ti, "f", secondary="s", secondary_weight=-1.0)
    with pytest.raises(ValueError, match="soft_weight"):
        cs.ImageQuilting(ti, "f", soft_weight=float("nan"))
