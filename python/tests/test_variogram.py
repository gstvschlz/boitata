import sys

import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(11)


@pytest.fixture
def model():
    return bt.Variogram(
        [bt.Structure("spherical", 2.0, 30.0)], nugget=0.5, rotation=(45, 0, 0), ratios=(0.5, 1)
    )


def test_spherical_shape(model):
    g = model.gamma([0.0, 30.0, 60.0])
    assert g[0] == 0.0 and g[1] == pytest.approx(2.5) and g[2] == pytest.approx(2.5)
    assert model.covariance([0.0])[0] == pytest.approx(model.sill) == pytest.approx(2.5)


def test_anisotropy_follows_azimuth(model):
    h = 12.0
    along = np.array([[h * np.sin(np.radians(45)), h * np.cos(np.radians(45)), 0]])
    across = np.array([[h * 0.5 * np.sin(np.radians(135)), h * 0.5 * np.cos(np.radians(135)), 0]])
    origin = np.zeros((1, 3))
    expected = model.gamma([h])[0]
    assert model.gamma_between(origin, along)[0] == pytest.approx(expected)
    assert model.gamma_between(origin, across)[0] == pytest.approx(expected)


def test_json_round_trip(model):
    back = bt.Variogram.from_json(model.to_json())
    a, b = rng.uniform(0, 50, (20, 3)), rng.uniform(0, 50, (20, 3))
    np.testing.assert_allclose(back.gamma_between(a, b), model.gamma_between(a, b))
    assert back.rotation == (45, 0, 0) and back.ratios == (0.5, 1)


def test_white_noise_variogram_is_flat_at_the_variance():
    coords = rng.uniform(0, 100, (600, 2))
    values = rng.normal(size=600)
    exp = bt.experimental_variogram(coords, values, lag=5, max_lag=50)
    assert np.all(np.abs(exp.gammas - values.var()) < 0.25)
    fitted = exp.fit("spherical")
    assert fitted.sill == pytest.approx(values.var(), rel=0.2)


def test_nested_fit_with_fixed_and_bounded_parameters():
    x = np.linspace(0, 400, 800)
    values = np.sin(x / 6) + np.sin(x / 40) + 0.2 * rng.normal(size=x.size)
    exp = bt.experimental_variogram(np.c_[x, 0 * x], values, lag=4, max_lag=120)
    assert repr(exp.fit()) == repr(exp.fit("spherical", nugget=None, sills=None, ranges=None))
    two = exp.fit(["spherical", "spherical"], weighting="count/distance")
    assert len(two.structures) == 2 and two.structures[0].range < two.structures[1].range
    fixed = exp.fit(["spherical", "gaussian"], nugget=0.05, sills=[None, (0.1, 0.4)], ranges=[(5, 15), 60.0])
    assert fixed.nugget == 0.05 and fixed.structures[1].range == 60.0
    assert 0.1 <= fixed.structures[1].sill <= 0.4 and 5 <= fixed.structures[0].range <= 15
    assert repr(bt.Variogram.fit(exp, ("spherical", "spherical"), weighting="count/distance")) == repr(two)
    with pytest.raises(ValueError):
        exp.fit(["spherical"] * 4)
    with pytest.raises(ValueError):
        exp.fit(["spherical"] * 2, sills=[None])
    with pytest.raises(ValueError):
        exp.fit(nugget=(0.3, 0.1))


def test_directional_fit_finds_the_anisotropy():
    xy = np.stack(np.meshgrid(np.arange(0, 80, 2.0), np.arange(0, 80, 2.0)), -1).reshape(-1, 2)
    t = np.radians(30)
    major, minor = np.array([np.sin(t), np.cos(t)]), np.array([np.cos(t), -np.sin(t)])
    freq = rng.normal(size=(400, 1)) * major / 15 + rng.normal(size=(400, 1)) * minor / 5
    values = np.cos(xy @ freq.T + rng.uniform(0, 2 * np.pi, 400)).sum(1) / np.sqrt(200)
    azimuths = np.arange(0, 180, 22.5)
    exps = [bt.experimental_variogram(xy, values, 2, 40, azimuth=a) for a in azimuths]
    directions = [(a, 0) for a in azimuths]
    model = bt.Variogram.fit_directional(exps, directions, ["spherical", "spherical"])
    assert model.rotation[0] == pytest.approx(30, abs=10) and model.rotation[1:] == (0, 0)
    assert model.ratios[0] < 0.6 and model.ratios[1] == 1
    assert model.structures[-1].range <= max(e.lags.max() for e in exps)
    assert repr(bt.Variogram.fit_directional(exps, directions, ["spherical", "spherical"])) == repr(model)
    fixed = bt.Variogram.fit_directional(
        exps, directions, rotation=[45.0, None, None], ratios=[(0.2, 0.5), None]
    )
    assert fixed.rotation[0] == 45 and 0.2 <= fixed.ratios[0] <= 0.5
    with pytest.raises(ValueError):
        bt.Variogram.fit_directional(exps, directions[1:])
    with pytest.raises(ValueError):
        bt.Variogram.fit_directional(exps, directions, rotation=[None, None])


def test_estimators_and_standardize():
    coords = rng.uniform(0, 100, (400, 2))
    values = np.exp(rng.normal(size=400))
    args = (coords, values, 5, 50)
    cov = bt.experimental_variogram(*args, estimator="covariance")
    np.testing.assert_allclose(cov.gammas + cov.covariances, values.var())
    rho = bt.experimental_variogram(*args, estimator="correlogram", standardize=True)
    np.testing.assert_allclose(rho.gammas, 1 - rho.covariances)
    assert bt.experimental_variogram(*args).covariances is None
    std = bt.experimental_variogram(*args, standardize=True)
    np.testing.assert_allclose(std.gammas, bt.experimental_variogram(*args).gammas / values.var())
    pr = bt.experimental_variogram(*args, estimator="pairwise_relative")
    scaled = bt.experimental_variogram(coords, 7 * values, 5, 50, estimator="pairwise-relative")
    np.testing.assert_allclose(pr.gammas, scaled.gammas)
    with pytest.raises(ValueError):
        bt.experimental_variogram(coords, values - 5, 5, 50, estimator="pairwise-relative")
    m = bt.variogram_map(coords, values, lag=10, max_lag=60, steps=6, estimator="correlogram")
    assert m.gammas.shape == (6, len(m.lags))


def test_cross_variograms():
    coords = rng.uniform(0, 100, (300, 2))
    values = rng.normal(size=300)
    args = (coords, values, 5, 50)
    direct = bt.experimental_variogram(*args, azimuth=30)
    cross = bt.experimental_variogram(*args, azimuth=30, other=values)
    np.testing.assert_array_equal(cross.gammas, direct.gammas)
    affine = bt.experimental_variogram(*args, other=3 - 2 * values)
    np.testing.assert_allclose(affine.gammas, -2 * bt.experimental_variogram(*args).gammas)
    other = values + rng.normal(size=300)
    c12 = bt.experimental_variogram(*args, azimuth=30, estimator="covariance", other=other)
    c21 = bt.experimental_variogram(coords, other, 5, 50, azimuth=210, estimator="covariance", other=values)
    np.testing.assert_allclose(c12.covariances, c21.covariances, atol=1e-12)
    np.testing.assert_allclose(c12.gammas + c12.covariances, np.cov(values, other, bias=True)[0, 1])
    elsewhere = coords + 1
    hetero = bt.experimental_variogram(*args, estimator="covariance", other=other, other_coords=elsewhere)
    assert len(hetero.covariances) == len(hetero.gammas)
    with pytest.raises(ValueError):
        bt.experimental_variogram(*args, other=other, other_coords=elsewhere)
    with pytest.raises(ValueError):
        bt.experimental_variogram(*args, other_coords=elsewhere)
    with pytest.raises(ValueError):
        bt.experimental_variogram(*args, other=other[:10])


def test_directional_and_map_shapes():
    coords = rng.uniform(0, 100, (300, 2))
    values = coords[:, 1] / 10 + rng.normal(size=300)
    exp = bt.experimental_variogram(coords, values, lag=10, max_lag=60, azimuth=0)
    assert len(exp.lags) == len(exp.gammas) == len(exp.counts)
    m = bt.variogram_map(coords, values, lag=10, max_lag=60, steps=12)
    assert m.gammas.shape == (12, len(m.lags)) and m.ranges.shape == (12,)


def gaussian_field(coords, rotation, ratios, scale, seed, waves=400):
    """Gaussian-covariance field by a sum of random cosines, anisotropic as `rotation` and `ratios`."""
    to_isotropic = bt.plot._principal(rotation) / np.array([1.0, *ratios])[:, None] / scale
    rng = np.random.default_rng(seed)
    omega = rng.normal(scale=np.sqrt(2), size=(waves, 3))
    phase = rng.uniform(0, 2 * np.pi, waves)
    return np.sqrt(2 / waves) * np.cos(coords @ to_isotropic.T @ omega.T + phase).sum(axis=1)


def test_variogram_volume_recovers_a_rotated_anisotropy():
    coords = np.random.default_rng(5).uniform(0, 200, (4000, 3))
    values = gaussian_field(coords, (120.0, 35.0, 30.0), (0.5, 0.25), 25.0, seed=4)
    vol = bt.variogram_volume(coords, values, 5.0, 70.0, model="gaussian")
    n = vol.lags.size
    assert vol.gammas.shape == vol.counts.shape == (n, n, n) and vol.lags[n // 2] == 0
    np.testing.assert_array_equal(vol.counts, vol.counts[::-1, ::-1, ::-1])
    found, truth = bt.plot._principal(vol.rotation), bt.plot._principal((120.0, 35.0, 30.0))
    angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(found * truth, axis=1)), 0, 1)))
    assert np.all(angles < 6), (vol.rotation, angles)
    np.testing.assert_allclose(vol.ratios, (0.5, 0.25), atol=0.1)
    assert vol.ratios == pytest.approx((vol.ranges[1] / vol.ranges[0], vol.ranges[2] / vol.ranges[0]))
    assert vol.directions.shape == (200, 2) and vol.direction_ranges.shape == (200,)
    assert vol.axes[0] == pytest.approx(vol.rotation[:2])
    iso = bt.variogram_volume(coords, gaussian_field(coords, (0, 0, 0), (1, 1), 25.0, seed=3), 5.0, 70.0)
    assert iso.ratios[1] > 0.75
    with pytest.raises(ValueError):
        bt.variogram_volume(coords, values, 1.0, 70.0)


def test_coregionalization_at_zero_lag():
    lmc = bt.Coregionalization(
        [[0.1, 0.0], [0.0, 0.2]], structures=[("spherical", 20.0, [[1.0, 0.6], [0.6, 1.0]])]
    )
    p = np.zeros((1, 3))
    assert lmc.cross_covariance(0, 1, p, p)[0] == pytest.approx(0.6)
    assert lmc.cross_covariance(1, 1, p, p)[0] == pytest.approx(1.2)


def test_coregionalization_must_be_positive_semidefinite():
    zero = [[0.0, 0.0], [0.0, 0.0]]
    bt.Coregionalization(zero, structures=[("spherical", 20.0, [[1.0, 1.0 + 1e-14], [1.0, 1.0]])])
    with pytest.raises(
        ValueError, match=r"structure 0 sills .* smallest eigenvalue -2\.000e-1.*Coregionalization\.fit"
    ):
        bt.Coregionalization(zero, structures=[("spherical", 20.0, [[1.0, 1.2], [1.2, 1.0]])])
    with pytest.raises(bt.BoitataError, match="nugget is not symmetric"):
        bt.Coregionalization([[0.1, 0.05], [0.0, 0.1]], structures=[])
    with pytest.raises(ValueError, match="structure 0 sills is not 2 x 2"):
        bt.Coregionalization(zero, structures=[("spherical", 20.0, [[1.0]])])
    good = bt.Coregionalization(zero, structures=[("spherical", 20.0, [[1.0, 0.6], [0.6, 1.0]])]).to_json()
    with pytest.raises(ValueError, match="positive semi-definite"):
        bt.Coregionalization.from_json(good.replace("0.6", "1.6"))


def field(xy, scale, n=200):
    freq = rng.normal(size=(n, 2)) / scale
    return np.cos(xy @ freq.T + rng.uniform(0, 2 * np.pi, n)).sum(1) / np.sqrt(n / 2)


def test_coregionalization_fit_is_positive_semidefinite():
    xy = rng.uniform(0, 100, (500, 2))
    short, long = field(xy, 5), field(xy, 20)
    a, b = short + long, 1000 * (0.8 * long - 0.5 * short + 0.3 * rng.normal(size=500))
    exp = [
        [bt.experimental_variogram(xy, a, 4, 60), bt.experimental_variogram(xy, a, 4, 60, other=b)],
        [None, bt.experimental_variogram(xy, b, 4, 60)],
    ]
    lmc = bt.Coregionalization.fit(exp, ["spherical", "spherical"])
    assert lmc.nvar == 2 and lmc.nugget.shape == (2, 2) and len(lmc.structures) == 2
    for m in [lmc.nugget, *(s[2] for s in lmc.structures)]:
        assert np.linalg.eigvalsh(m).min() >= -1e-9 * np.abs(m).max()
    assert lmc.structures[0][1] < lmc.structures[1][1]
    assert lmc.structures[1][2][0, 1] > 0 > lmc.structures[0][2][0, 1]
    assert lmc.to_json() == bt.Coregionalization.fit(exp, ["spherical", "spherical"]).to_json()
    assert bt.Coregionalization.from_json(lmc.to_json()).to_json() == lmc.to_json()
    one = bt.Coregionalization.fit([[exp[0][0]]], ["spherical", "spherical"])
    v = exp[0][0].fit(["spherical", "spherical"])
    assert one.nugget[0, 0] == pytest.approx(v.nugget, abs=1e-6)
    assert [s[1] for s in one.structures] == pytest.approx([s.range for s in v.structures], rel=1e-6)
    none = bt.Coregionalization.fit(exp, "spherical", nugget=False, ranges=[(10, 30)])
    assert not none.nugget.any() and 10 <= none.structures[0][1] <= 30
    gap = bt.Coregionalization.fit([[exp[0][0], None], [None, exp[1][1]]], "spherical")
    assert np.linalg.eigvalsh(gap.structures[0][2]).min() >= -1e-9


def test_coregionalization_fit_with_fixed_anisotropy():
    xy = rng.uniform(0, 100, (500, 2))
    a = field(xy, 10)
    b = a + 0.5 * rng.normal(size=500)
    directions = [(0.0, 0.0), (90.0, 0.0)]

    def pair(u, v):
        return [bt.experimental_variogram(xy, u, 4, 60, azimuth=d, other=v) for d, _ in directions]

    exp = [[pair(a, None), pair(a, b)], [None, pair(b, None)]]
    lmc = bt.Coregionalization.fit(exp, directions=directions, rotation=(30.0, 0.0, 0.0), ratios=(0.5, 1.0))
    assert lmc.rotation == (30.0, 0.0, 0.0) and lmc.ratios == (0.5, 1.0)
    with pytest.raises(ValueError):
        bt.Coregionalization.fit(exp)
    with pytest.raises(ValueError):
        bt.Coregionalization.fit([[pair(a, None)[0]]], rotation=(30.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        bt.Coregionalization.fit([[None]])
    with pytest.raises(ValueError):
        bt.Coregionalization.fit(exp, directions=directions[:1])


@pytest.mark.xfail(sys.platform == "linux", reason="drifts to a wrong azimuth on Linux, #383", strict=False)
def test_coregionalization_fit_finds_the_anisotropy():
    xy = np.stack(np.meshgrid(np.arange(0, 80, 2.0), np.arange(0, 80, 2.0)), -1).reshape(-1, 2)
    t = np.radians(30)
    major, minor = np.array([np.sin(t), np.cos(t)]), np.array([np.cos(t), -np.sin(t)])
    freq = rng.normal(size=(400, 1)) * major / 15 + rng.normal(size=(400, 1)) * minor / 5
    a = np.cos(xy @ freq.T + rng.uniform(0, 2 * np.pi, 400)).sum(1) / np.sqrt(200)
    b = 10 * (a + 0.5 * rng.normal(size=len(a)))
    azimuths = np.arange(0, 180, 22.5)

    def cell(u, v):
        return [bt.experimental_variogram(xy, u, 2, 40, azimuth=d, other=v) for d in azimuths]

    exp = [[cell(a, None), cell(a, b)], [None, cell(b, None)]]
    directions = [(d, 0) for d in azimuths]
    lmc = bt.Coregionalization.fit(exp, ["spherical", "spherical"], directions=directions)
    alone = bt.Variogram.fit_directional(exp[0][0], directions, ["spherical", "spherical"])
    assert lmc.rotation[0] == pytest.approx(alone.rotation[0], abs=5) and lmc.rotation[1:] == (0, 0)
    assert lmc.ratios[0] == pytest.approx(alone.ratios[0], abs=0.05) and lmc.ratios[1] == 1
    for m in [lmc.nugget, *(s[2] for s in lmc.structures)]:
        assert np.linalg.eigvalsh(m).min() >= -1e-9 * np.abs(m).max()
    bounded = bt.Coregionalization.fit(
        exp, directions=directions, rotation=[45.0, None, None], ratios=[(0.2, 0.5), None]
    )
    assert bounded.rotation[0] == 45 and 0.2 <= bounded.ratios[0] <= 0.5


def test_variogram_sets_match_single_calls_and_feed_the_fit():
    xy = rng.uniform(0, 100, (300, 2))
    a = field(xy, 10)
    b, c = a + 0.5 * rng.normal(size=300), rng.normal(size=300)
    points = bt.PointSet(xy, attributes={"a": a, "b": b, "c": c})
    vs = bt.experimental_variograms(points, ["a", "b", c], 4, 60)
    assert vs.nvar == 3 and vs.names == ["a", "b", None] and vs.directions is None
    np.testing.assert_array_equal(vs[0, 0].gammas, bt.experimental_variogram(xy, a, 4, 60).gammas)
    np.testing.assert_array_equal(
        vs["a", "b"].gammas, bt.experimental_variogram(xy, a, 4, 60, other=b).gammas
    )
    swapped = bt.experimental_variograms(xy, [b, a], 4, 60)
    np.testing.assert_array_equal(vs[1, 0].gammas, swapped[0, 1].gammas)
    one = bt.experimental_variograms(xy, [a], 4, 60)
    np.testing.assert_array_equal(one[0, 0].gammas, bt.experimental_variogram(xy, a, 4, 60).gammas)
    matrix = [[vs[i, j] if j >= i else None for j in range(3)] for i in range(3)]
    assert bt.Coregionalization.fit(vs).to_json() == bt.Coregionalization.fit(matrix).to_json()
    directions = [(0.0, 0.0), (90.0, 0.0)]
    directed = bt.experimental_variograms(xy, [a, b], 4, 60, directions=directions)
    assert len(directed[0, 1]) == 2 and directed.directions == directions
    by_hand = [[directed[0, 0], directed[0, 1]], [None, directed[1, 1]]]
    fitted = bt.Coregionalization.fit(directed, directions=None)
    assert fitted.to_json() == bt.Coregionalization.fit(by_hand, directions=directions).to_json()
    with pytest.raises(ValueError):
        bt.Coregionalization.fit(directed, directions=directions)
    with pytest.raises(KeyError):
        vs["d", 0]
    with pytest.raises(ValueError):
        bt.experimental_variograms(xy, [a, b], 4, 60, estimator="correlogram")


def test_grid_variograms_match_the_pair_search():
    model = bt.BlockModel((10.0, 20.0), (2.0, 3.0), (30, 20), rotation=(25.0, 0.0, 0.0))
    xy = model.centroids
    a = field(xy[:, :2], 8)
    b = a + rng.normal(size=len(a))
    for kwargs in [{}, {"azimuth": 43.0, "tolerance": 14.0, "bandwidth": 6.1}]:
        by_grid = bt.experimental_variogram(model, a, 2.3, 30.7, **kwargs)
        by_pairs = bt.experimental_variogram(model, a, 2.3, 30.7, method="pairs", **kwargs)
        np.testing.assert_array_equal(by_grid.counts, by_pairs.counts)
        np.testing.assert_allclose(by_grid.gammas, by_pairs.gammas, rtol=1e-10)
        cross = bt.experimental_variogram(model, a, 2.3, 30.7, other=b, method="grid", **kwargs)
        np.testing.assert_allclose(
            cross.gammas, bt.experimental_variogram(xy, a, 2.3, 30.7, other=b, **kwargs).gammas, rtol=1e-10
        )
    keep = np.flatnonzero(np.arange(len(a)) % 5 != 2)
    masked = bt.BlockModel(
        (10.0, 20.0), (2.0, 3.0), (30, 20), rotation=(25.0, 0.0, 0.0), index=keep.astype(np.uint64)
    )
    grid_set = bt.experimental_variograms(masked, [a[keep], b[keep]], 2.3, 30.7)
    pair_set = bt.experimental_variograms(xy[keep], [a[keep], b[keep]], 2.3, 30.7)
    np.testing.assert_allclose(grid_set[0, 1].gammas, pair_set[0, 1].gammas, rtol=1e-10)
    with pytest.raises(ValueError):
        bt.experimental_variogram(xy, a, 2.3, 30.7, method="grid")
    with pytest.raises(ValueError):
        bt.experimental_variogram(model, a, 2.3, 30.7, method="nearest")


def test_transiogram_is_a_markov_matrix():
    t = bt.Transiogram([0.2, 0.3, 0.5], 10.0)
    np.testing.assert_allclose(t.matrix(0.0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t.matrix(5.0).sum(axis=1), 1.0)
    np.testing.assert_allclose(t.matrix(1e6)[0], [0.2, 0.3, 0.5], atol=1e-6)


def test_change_of_support_reduces_variance():
    anam = bt.HermiteAnamorphosis().fit(rng.lognormal(0, 0.7, 500))
    gaussian = bt.Variogram([("spherical", 1.0, 50.0)])
    r, block = bt.change_of_support(anam, gaussian, size=(10, 10))
    assert 0 < r < 1
    assert block.variance_ < anam.variance_
    assert block.mean_ == pytest.approx(anam.mean_)


def test_nugget_averages_out_of_the_block():
    assert bt.block_correlation(bt.Variogram([], nugget=1.0), size=(10, 10)) == 0.0
    structured = bt.block_correlation(bt.Variogram([("spherical", 1.0, 50.0)]), size=(10, 10))
    nugget = bt.block_correlation(bt.Variogram([("spherical", 1.0, 50.0)], nugget=0.5), size=(10, 10))
    assert nugget == pytest.approx(structured / 1.5)


def test_downhole_nugget():
    n_holes, depth = 200, 40
    phi = np.exp(-1 / 20)
    signal = np.empty((n_holes, depth))
    signal[:, 0] = rng.normal(size=n_holes)
    for k in range(1, depth):
        signal[:, k] = phi * signal[:, k - 1] + np.sqrt(1 - phi**2) * rng.normal(size=n_holes)
    values = (signal + np.sqrt(0.3) * rng.normal(size=signal.shape)).ravel()
    coords = np.stack(np.meshgrid(0.3 * np.arange(n_holes), 0.0, -np.arange(depth), indexing="ij"), -1)
    holes = np.repeat([f"DH{i}" for i in range(n_holes)], depth)
    exp = bt.experimental_variogram(coords.reshape(-1, 3), values, 1.0, 10.0, holes=holes)
    assert exp.lags[0] == pytest.approx(1.0)
    assert exp.nugget() == pytest.approx(0.3, abs=0.05)
    vertical = bt.experimental_variogram(
        coords.reshape(-1, 3), values, 1.0, 10.0, azimuth=0, dip=90, holes=holes
    )
    np.testing.assert_array_equal(vertical.gammas, exp.gammas)
    flat = bt.experimental_variogram(coords.reshape(-1, 3), values, 1.0, 10.0, azimuth=90, holes=holes)
    assert flat.counts.sum() == 0
    with pytest.raises(ValueError):
        bt.experimental_variogram(coords.reshape(-1, 3), values, 1.0, 10.0, other=values, holes=holes)


def test_names_resolve_against_the_container():
    xyz, v, w = rng.uniform(0, 100, (200, 3)), rng.normal(size=200), rng.normal(size=200)
    columns = {"v": v, "w": w, "hole": np.repeat(np.arange(20), 10), "c": np.arange(200) % 3}
    points = bt.PointSet(xyz, columns)
    pairs = [
        (
            bt.experimental_variogram(points, "v", 10.0, 50.0, other="w"),
            bt.experimental_variogram(xyz, v, 10.0, 50.0, other=w),
        ),
        (
            bt.experimental_variogram(points, "v", 10.0, 50.0, holes="hole"),
            bt.experimental_variogram(xyz, v, 10.0, 50.0, holes=columns["hole"]),
        ),
        (bt.variogram_map(points, "v", 10.0, 50.0), bt.variogram_map(xyz, v, 10.0, 50.0)),
        (bt.variogram_volume(points, "v", 10.0, 50.0), bt.variogram_volume(xyz, v, 10.0, 50.0)),
    ]
    for named, arrays in pairs:
        np.testing.assert_array_equal(named.gammas, arrays.gammas)
    named, arrays = (
        bt.experimental_transiogram(points, "c", 10.0, 50.0),
        bt.experimental_transiogram(xyz, columns["c"], 10.0, 50.0),
    )
    np.testing.assert_array_equal(named[1], arrays[1])
    with pytest.raises(bt.MissingColumn):
        bt.experimental_variogram(points, "x", 10.0, 50.0)


def test_options_are_keyword_only():
    xyz, v = rng.uniform(0, 100, (50, 3)), rng.normal(size=50)
    for call in (
        lambda: bt.experimental_variogram(xyz, v, 10.0, 50.0, 0.0),
        lambda: bt.variogram_map(xyz, v, 10.0, 50.0, (1, 0, 0)),
        lambda: bt.variogram_volume(xyz, v, 10.0, 50.0, 20.0),
        lambda: bt.Variogram([("spherical", 1.0, 30.0)], 0.1),
        lambda: bt.Search(50.0, 8),
        lambda: bt.Coregionalization([[0.0]], [("spherical", 20.0, [[1.0]])]),
    ):
        with pytest.raises(TypeError):
            call()


def test_intrinsic_coregionalization_recovers_its_sill_matrix():
    xy = np.random.default_rng(3).uniform(0, 200, (1500, 2))
    xyz = np.c_[xy, np.zeros(len(xy))]
    z = np.stack([gaussian_field(xyz, (0, 0, 0), (1, 1), 15.0, seed=s) for s in (1, 2)])
    truth = np.array([[4.0, 1.2], [1.2, 1.0]])
    a, b = np.linalg.cholesky(truth) @ z
    exp = bt.experimental_variograms(xy, [a, b], 3, 45)
    icm = bt.Coregionalization.fit(exp, "gaussian", intrinsic=True)
    sill = icm.nugget + sum(s for _, _, s in icm.structures)
    assert np.linalg.eigvalsh(sill).min() >= 0
    assert sill[0, 1] / np.sqrt(sill[0, 0] * sill[1, 1]) == pytest.approx(0.6, abs=0.1)
    assert sill[0, 0] / sill[1, 1] == pytest.approx(4.0, rel=0.3)
    for s in [icm.nugget, *(m for _, _, m in icm.structures)]:
        np.testing.assert_allclose(s, s[0, 0] / sill[0, 0] * sill, atol=1e-12)
    shape = bt.Variogram([bt.Structure("gaussian", 1.0, 20.0)])
    np.testing.assert_allclose(bt.Coregionalization.intrinsic(shape, sill).structures[0][2], sill)
    one = bt.Coregionalization.fit([[exp[0, 0]]], ["spherical", "spherical"], intrinsic=True)
    v = exp[0, 0].fit(["spherical", "spherical"])
    assert one.nugget[0, 0] == pytest.approx(v.nugget, abs=1e-3 * v.sill)
    assert [s[2][0, 0] for s in one.structures] == pytest.approx([s.sill for s in v.structures], rel=1e-3)
    with pytest.raises(ValueError, match="positive semi-definite"):
        bt.Coregionalization.intrinsic(shape, [[1, 2], [2, 1]])


def moving_gaussian(n, window, seed):
    noise = np.random.default_rng(seed).normal(size=n + window - 1)
    return np.c_[np.arange(n, dtype=float), np.zeros(n)], np.convolve(noise, np.ones(window), "valid")


def test_madogram_is_root_gamma_over_pi_for_gaussian_data():
    xy, z = moving_gaussian(8000, 15, seed=4)

    def run(values, estimator="matheron", **kwargs):
        return bt.experimental_variogram(xy, values, 1.0, 30.0, estimator=estimator, **kwargs)

    gamma, madogram = run(z), run(z, "madogram")
    np.testing.assert_allclose(madogram.gammas, np.sqrt(gamma.gammas / np.pi), rtol=0.04)
    np.testing.assert_allclose(bt.dissemination(madogram, gamma), 1.0, atol=0.04)
    np.testing.assert_allclose(run(z, "madogram", standardize=True).gammas, madogram.gammas / z.std())
    spiked = z.copy()
    spiked[::500] += 40.0
    ratio = {e: run(spiked, e).gammas / run(z, e).gammas for e in ("matheron", "madogram")}
    assert np.all(ratio["matheron"] - 1 > 3 * (ratio["madogram"] - 1))
    assert bt.dissemination(run(spiked, "madogram"), run(spiked))[0] < 0.7
    with pytest.raises(ValueError, match="same lags"):
        bt.dissemination(madogram, bt.experimental_variogram(xy, z, 2.0, 30.0))
