import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(3)
coords = rng.uniform(0, 100, (150, 2))
values = np.sin(coords[:, 0] / 15) + coords[:, 1] / 50
model = cs.Variogram([("spherical", 1.0, 40.0)], nugget=0.0)
search = cs.Search(radius=50, max_samples=16)


def test_ordinary_kriging_honours_data_with_zero_variance():
    est, var = cs.OrdinaryKriging(model, search).fit(coords, values).predict(coords, return_variance=True)
    np.testing.assert_allclose(est, values, atol=1e-8)
    np.testing.assert_allclose(var, 0.0, atol=1e-8)


def test_variance_grows_away_from_data():
    ok = cs.OrdinaryKriging(model, search).fit(coords, values)
    near = coords[:1] + [0.5, 0.0]
    far = np.array([[200.0, 200.0]])
    wide = cs.OrdinaryKriging(model, cs.Search(radius=400)).fit(coords, values)
    assert ok.predict(near, return_variance=True)[1][0] < wide.predict(far, return_variance=True)[1][0]


def test_too_few_neighbours_is_nan():
    ok = cs.OrdinaryKriging(model, cs.Search(radius=5, min_samples=3)).fit(coords, values)
    assert np.isnan(ok.predict([[500.0, 500.0]])[0])


def test_simple_kriging_reverts_to_the_mean_far_away():
    sk = cs.SimpleKriging(model, cs.Search(radius=1000), mean=7.0).fit(coords, values)
    assert sk.predict([[400.0, 400.0]])[0] == pytest.approx(7.0)


def test_indicator_kriging_gives_probabilities():
    ik = cs.IndicatorKriging(model, search, threshold=float(np.median(values))).fit(coords, values)
    p = ik.predict(rng.uniform(0, 100, (50, 2)))
    assert np.all((p >= -1e-9) & (p <= 1 + 1e-9))


def test_universal_kriging_reproduces_a_linear_drift():
    linear = 2.0 + 0.3 * coords[:, 0] - 0.1 * coords[:, 1]
    uk = cs.UniversalKriging(model, cs.Search(radius=100, max_samples=30), degree=1).fit(coords, linear)
    targets = rng.uniform(10, 90, (20, 2))
    expected = 2.0 + 0.3 * targets[:, 0] - 0.1 * targets[:, 1]
    np.testing.assert_allclose(uk.predict(targets), expected, atol=1e-6)


def test_block_kriging_of_a_constant_is_the_constant():
    bk = cs.BlockKriging(model, search, size=(10, 10)).fit(coords, np.full(len(coords), 3.0))
    np.testing.assert_allclose(bk.predict([[50.0, 50.0]]), 3.0)


def test_nearest_and_inverse_distance():
    nn = cs.NearestNeighbor(search).fit(coords, values)
    np.testing.assert_allclose(nn.predict(coords[:5] + 1e-6), values[:5])
    idw = cs.InverseDistance(search, power=2).fit(coords, values)
    est = idw.predict(rng.uniform(0, 100, (30, 2)))
    assert np.all((est >= values.min()) & (est <= values.max()))


def test_cross_validation_summary():
    cv = cs.OrdinaryKriging(model, search).fit(coords, values).cross_validate()
    assert cv.estimate.shape == values.shape
    assert abs(cv.mean_error) < 0.1 and cv.correlation > 0.8
    assert np.isfinite(cv.standardized_squared_error)


def test_targets_from_containers():
    ok = cs.OrdinaryKriging(model, search).fit(coords, values)
    grid = cs.BlockModel(origin=(0, 0), size=(10, 10), count=(10, 10))
    est = ok.predict(grid)
    assert est.shape == (100,)
    grid = grid.with_column("ok", est)
    np.testing.assert_array_equal(grid["ok"], est)
    points = cs.PointSet(coords[:3])
    np.testing.assert_allclose(ok.predict(points), values[:3], atol=1e-8)


def test_dual_kriging_interpolates():
    dual = cs.DualKriging(model, degree=1).fit(coords, values)
    np.testing.assert_allclose(dual.predict(coords), values, atol=1e-6)


def test_neighborhood_stats_columns():
    stats = cs.neighborhood_stats(coords[:4], coords, values, k=5)
    assert set(stats) >= {"nearest_dist", "value_mean", "n_within"}
    np.testing.assert_allclose(stats["nearest_dist"], 0.0)
