import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(11)


@pytest.fixture
def model():
    return cs.Variogram(
        [cs.Structure("spherical", 2.0, 30.0)], nugget=0.5, rotation=(45, 0, 0), ratios=(0.5, 1)
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
    back = cs.Variogram.from_json(model.to_json())
    a, b = rng.uniform(0, 50, (20, 3)), rng.uniform(0, 50, (20, 3))
    np.testing.assert_allclose(back.gamma_between(a, b), model.gamma_between(a, b))
    assert back.rotation == (45, 0, 0) and back.ratios == (0.5, 1)


def test_white_noise_variogram_is_flat_at_the_variance():
    coords = rng.uniform(0, 100, (600, 2))
    values = rng.normal(size=600)
    exp = cs.experimental_variogram(coords, values, lag=5, max_lag=50)
    assert np.all(np.abs(exp.gammas - values.var()) < 0.25)
    fitted = exp.fit("spherical")
    assert fitted.sill == pytest.approx(values.var(), rel=0.2)


def test_nested_fit_with_fixed_and_bounded_parameters():
    x = np.linspace(0, 400, 800)
    values = np.sin(x / 6) + np.sin(x / 40) + 0.2 * rng.normal(size=x.size)
    exp = cs.experimental_variogram(np.c_[x, 0 * x], values, lag=4, max_lag=120)
    assert repr(exp.fit()) == repr(exp.fit("spherical", nugget=None, sills=None, ranges=None))
    two = exp.fit(["spherical", "spherical"], weighting="count/distance")
    assert len(two.structures) == 2 and two.structures[0].range < two.structures[1].range
    fixed = exp.fit(["spherical", "gaussian"], nugget=0.05, sills=[None, (0.1, 0.4)], ranges=[(5, 15), 60.0])
    assert fixed.nugget == 0.05 and fixed.structures[1].range == 60.0
    assert 0.1 <= fixed.structures[1].sill <= 0.4 and 5 <= fixed.structures[0].range <= 15
    assert repr(cs.Variogram.fit(exp, ("spherical", "spherical"), "count/distance")) == repr(two)
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
    exps = [cs.experimental_variogram(xy, values, 2, 40, azimuth=a) for a in azimuths]
    directions = [(a, 0) for a in azimuths]
    model = cs.Variogram.fit_directional(exps, directions, ["spherical", "spherical"])
    assert model.rotation[0] == pytest.approx(30, abs=10) and model.rotation[1:] == (0, 0)
    assert model.ratios[0] < 0.6 and model.ratios[1] == 1
    assert model.structures[-1].range <= max(e.lags.max() for e in exps)
    assert repr(cs.Variogram.fit_directional(exps, directions, ["spherical", "spherical"])) == repr(model)
    fixed = cs.Variogram.fit_directional(
        exps, directions, rotation=[45.0, None, None], ratios=[(0.2, 0.5), None]
    )
    assert fixed.rotation[0] == 45 and 0.2 <= fixed.ratios[0] <= 0.5
    with pytest.raises(ValueError):
        cs.Variogram.fit_directional(exps, directions[1:])
    with pytest.raises(ValueError):
        cs.Variogram.fit_directional(exps, directions, rotation=[None, None])


def test_estimators_and_standardize():
    coords = rng.uniform(0, 100, (400, 2))
    values = np.exp(rng.normal(size=400))
    args = (coords, values, 5, 50)
    cov = cs.experimental_variogram(*args, estimator="covariance")
    np.testing.assert_allclose(cov.gammas + cov.covariances, values.var())
    rho = cs.experimental_variogram(*args, estimator="correlogram", standardize=True)
    np.testing.assert_allclose(rho.gammas, 1 - rho.covariances)
    assert cs.experimental_variogram(*args).covariances is None
    std = cs.experimental_variogram(*args, standardize=True)
    np.testing.assert_allclose(std.gammas, cs.experimental_variogram(*args).gammas / values.var())
    pr = cs.experimental_variogram(*args, estimator="pairwise_relative")
    scaled = cs.experimental_variogram(coords, 7 * values, 5, 50, estimator="pairwise-relative")
    np.testing.assert_allclose(pr.gammas, scaled.gammas)
    with pytest.raises(ValueError):
        cs.experimental_variogram(coords, values - 5, 5, 50, estimator="pairwise-relative")
    m = cs.variogram_map(coords, values, lag=10, max_lag=60, steps=6, estimator="correlogram")
    assert m.gammas.shape == (6, len(m.lags))


def test_cross_variograms():
    coords = rng.uniform(0, 100, (300, 2))
    values = rng.normal(size=300)
    args = (coords, values, 5, 50)
    direct = cs.experimental_variogram(*args, azimuth=30)
    cross = cs.experimental_variogram(*args, azimuth=30, other=values)
    np.testing.assert_array_equal(cross.gammas, direct.gammas)
    affine = cs.experimental_variogram(*args, other=3 - 2 * values)
    np.testing.assert_allclose(affine.gammas, -2 * cs.experimental_variogram(*args).gammas)
    other = values + rng.normal(size=300)
    c12 = cs.experimental_variogram(*args, azimuth=30, estimator="covariance", other=other)
    c21 = cs.experimental_variogram(coords, other, 5, 50, azimuth=210, estimator="covariance", other=values)
    np.testing.assert_allclose(c12.covariances, c21.covariances, atol=1e-12)
    np.testing.assert_allclose(c12.gammas + c12.covariances, np.cov(values, other, bias=True)[0, 1])
    elsewhere = coords + 1
    hetero = cs.experimental_variogram(*args, estimator="covariance", other=other, other_coords=elsewhere)
    assert len(hetero.covariances) == len(hetero.gammas)
    with pytest.raises(ValueError):
        cs.experimental_variogram(*args, other=other, other_coords=elsewhere)
    with pytest.raises(ValueError):
        cs.experimental_variogram(*args, other_coords=elsewhere)
    with pytest.raises(ValueError):
        cs.experimental_variogram(*args, other=other[:10])


def test_directional_and_map_shapes():
    coords = rng.uniform(0, 100, (300, 2))
    values = coords[:, 1] / 10 + rng.normal(size=300)
    exp = cs.experimental_variogram(coords, values, lag=10, max_lag=60, azimuth=0)
    assert len(exp.lags) == len(exp.gammas) == len(exp.counts)
    m = cs.variogram_map(coords, values, lag=10, max_lag=60, steps=12)
    assert m.gammas.shape == (12, len(m.lags)) and m.ranges.shape == (12,)


def test_coregionalization_at_zero_lag():
    lmc = cs.Coregionalization([[0.1, 0.0], [0.0, 0.2]], [("spherical", 20.0, [[1.0, 0.6], [0.6, 1.0]])])
    p = np.zeros((1, 3))
    assert lmc.cross_covariance(0, 1, p, p)[0] == pytest.approx(0.6)
    assert lmc.cross_covariance(1, 1, p, p)[0] == pytest.approx(1.2)


def test_transiogram_is_a_markov_matrix():
    t = cs.Transiogram([0.2, 0.3, 0.5], 10.0)
    np.testing.assert_allclose(t.matrix(0.0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t.matrix(5.0).sum(axis=1), 1.0)
    np.testing.assert_allclose(t.matrix(1e6)[0], [0.2, 0.3, 0.5], atol=1e-6)


def test_change_of_support_reduces_variance():
    anam = cs.HermiteAnamorphosis().fit(rng.lognormal(0, 0.7, 500))
    gaussian = cs.Variogram([("spherical", 1.0, 50.0)])
    r, block = cs.change_of_support(anam, gaussian, size=(10, 10))
    assert 0 < r < 1
    assert block.variance_ < anam.variance_
    assert block.mean_ == pytest.approx(anam.mean_)
