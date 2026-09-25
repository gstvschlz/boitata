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
