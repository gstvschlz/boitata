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


def test_search_passes_fill_what_earlier_passes_left():
    grid = np.column_stack([np.repeat(np.arange(0, 150, 5.0), 30), np.tile(np.arange(0, 150, 5.0), 30)])
    tight, wide = cs.Search(radius=10, min_samples=4), cs.Search(radius=80, min_samples=2)
    first = cs.OrdinaryKriging(model, tight).fit(coords, values).predict(grid)
    second = cs.OrdinaryKriging(model, wide).fit(coords, values).predict(grid)
    ok = cs.OrdinaryKriging(model, [tight, wide]).fit(coords, values)
    d = ok.predict(grid, diagnostics=True)
    np.testing.assert_array_equal(
        d["pass"], np.where(np.isfinite(first), 1.0, np.where(np.isfinite(second), 2.0, np.nan))
    )
    np.testing.assert_array_equal(d["value"], np.where(np.isfinite(first), first, second))
    assert {1.0, 2.0} <= set(d["pass"][np.isfinite(d["pass"])])
    assert np.isfinite(ok.cross_validate().estimate).all()
    with pytest.raises(ValueError):
        cs.OrdinaryKriging(model, [])


def test_high_grade_restriction():
    grid = rng.uniform(0, 100, (300, 2))
    plain = cs.OrdinaryKriging(model, search).fit(coords, values)
    wide = cs.OrdinaryKriging(model, cs.Search(radius=50, high_grade=(1.0, 50))).fit(coords, values)
    np.testing.assert_array_equal(plain.predict(grid), wide.predict(grid))
    np.testing.assert_array_equal(plain.cross_validate().estimate, wide.cross_validate().estimate)
    capped = cs.OrdinaryKriging(model, cs.Search(radius=50, high_grade=(1.0, 5))).fit(coords, values)
    assert not np.array_equal(plain.predict(grid), capped.predict(grid))
    assert not np.array_equal(plain.cross_validate().estimate, capped.cross_validate().estimate)
    far = np.array([[500.0, 500.0]])
    lone = cs.NearestNeighbor(cs.Search(radius=1000, max_samples=1, high_grade=(1.0, 5))).fit(
        [[1, 1], [0, 0]], [2.0, 0.5]
    )
    assert lone.predict(far)[0] == 0.5
    with pytest.raises(ValueError):
        cs.Search(radius=50, high_grade=(1.0, -1))


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


def test_cokriging_with_only_primary_data_is_ordinary_kriging():
    lmc = cs.Coregionalization([[0.0, 0.0], [0.0, 0.0]], [("spherical", 40.0, [[1.0, 0.7], [0.7, 1.0]])])
    ck = cs.Cokriging(lmc, search).fit(coords, values, [0] * len(values))
    targets = rng.uniform(0, 100, (25, 2))
    np.testing.assert_allclose(
        ck.predict(targets), cs.OrdinaryKriging(model, search).fit(coords, values).predict(targets)
    )


def test_collocated_cokriging_uses_the_secondary():
    lmc = cs.Coregionalization([[0.0, 0.0], [0.0, 0.0]], [("spherical", 40.0, [[1.0, 0.9], [0.9, 1.0]])])
    secondary = values + rng.normal(0, 0.1, len(values))
    xy = np.vstack([coords, coords])
    ck = cs.Cokriging(lmc, search, means=[values.mean(), secondary.mean()])
    ck.fit(xy, np.r_[values, secondary], [0] * len(values) + [1] * len(values))
    targets = rng.uniform(0, 100, (10, 2))
    plain = ck.predict(targets)
    with_secondary = ck.predict(targets, collocated={1: np.full(10, 5.0)})
    assert np.all(with_secondary > plain)


def test_disjunctive_kriging_tonnage_is_a_proportion():
    grades = rng.lognormal(0, 0.5, len(coords))
    anam = cs.HermiteAnamorphosis().fit(grades)
    dk = cs.DisjunctiveKriging(anam, model, search, order=15).fit(coords, grades)
    targets = rng.uniform(0, 100, (20, 2))
    t = dk.predict_tonnage(targets, cutoff=float(np.median(grades)))
    assert np.all((t > -0.05) & (t < 1.05))
    assert np.all(np.isfinite(dk.predict(targets)))


def test_shared_locations_keep_the_first_and_name_their_holes():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [0.0, 10.0]])
    v = np.array([1.0, 2.0, 5.0, 3.0])
    holes = ["DH1", "DH2", "DH3", "DH4"]
    search = cs.Search(radius=50, max_samples=8)
    with pytest.warns(UserWarning, match="holes DH1, DH3 at"):
        ok = cs.OrdinaryKriging(cs.Variogram([("spherical", 1.0, 30.0)]), search).fit(xy, v, holes)
    np.testing.assert_array_equal(ok.cross_validate().actual, [1.0, 2.0, 3.0])
    assert np.isfinite(ok.predict([[2.0, 2.0]])).all()
    with pytest.warns(UserWarning, match="rows 0, 2"):
        sgs = cs.SGS(cs.Variogram([("spherical", 1.0, 30.0)]), search).fit(xy, v)
    assert np.isfinite(sgs.simulate([[2.0, 2.0], [5.0, 5.0]], n=2, seed=1).mean).all()


def test_diagnostics_classification_and_smoothing():
    xy = rng.uniform(0, 100, (80, 2))
    v = np.sin(xy[:, 0] / 20) + rng.normal(0, 0.1, 80)
    grid = cs.BlockModel(origin=(0, 0), size=(10, 10), count=(10, 10))
    ok = cs.OrdinaryKriging(cs.Variogram([("spherical", 1.0, 40.0)]), cs.Search(radius=60, max_samples=12))
    d = ok.fit(xy, v).predict(grid, diagnostics=True)
    assert {"value", "variance", "efficiency", "slope", "n_samples", "pass"} <= set(d)
    assert np.all(d["slope"] > 0) and np.all(d["efficiency"] <= 1 + 1e-9)
    at_data = ok.predict(xy[:3], diagnostics=True)
    np.testing.assert_allclose(at_data["slope"], 1.0)
    np.testing.assert_allclose(at_data["efficiency"], 1.0)

    labels = cs.classify(
        d, [("measured", {"slope": (">=", 0.9)}), ("indicated", {"slope": (">=", 0.6)})], default="inferred"
    )
    assert set(labels) <= {"measured", "indicated", "inferred"}
    assert np.all(labels[d["slope"] >= 0.9] == "measured")
    spotted = np.where(np.arange(100) == 55, "inferred", "measured")
    np.testing.assert_array_equal(cs.smooth_classes(grid, spotted), np.full(100, "measured"))

    bias = cs.global_bias(d["value"], v)
    assert bias["relative"] == pytest.approx(d["value"].mean() / v.mean() - 1)
    assert ok.cross_validate().slope > 0


def test_measurement_error_blends_a_datum_with_its_neighbours():
    xy, v = coords[:20], values[:20]
    wide = cs.Search(radius=500, max_samples=50)
    plain = cs.OrdinaryKriging(model, wide).fit(xy, v)
    zero = cs.OrdinaryKriging(model, wide).fit(xy, v, error_variance=np.zeros(20))
    grid = rng.uniform(0, 100, (50, 2))
    np.testing.assert_array_equal(plain.predict(grid), zero.predict(grid))
    loo = plain.cross_validate()
    error = np.where(np.arange(20) == 0, 0.2, 0.0)
    noisy = cs.OrdinaryKriging(model, wide).fit(xy, v, error_variance=error)
    k = loo.variance[0] / (loo.variance[0] + 0.2)
    assert noisy.predict(xy[:1])[0] == pytest.approx(loo.estimate[0] + k * (v[0] - loo.estimate[0]), abs=1e-9)
    with pytest.raises(ValueError):
        cs.OrdinaryKriging(model, wide).fit(xy, v, error_variance=-error)
    with pytest.raises(ValueError):
        cs.InverseDistance(wide).fit(xy, v, error_variance=error)


def test_k_fold_cross_validation():
    ok = cs.OrdinaryKriging(model, search).fit(coords, values)
    np.testing.assert_array_equal(ok.cross_validate(folds=len(values)).estimate, ok.cross_validate().estimate)
    five = ok.cross_validate(folds=5)
    assert np.isfinite(five.estimate).all() and five.rmse > ok.cross_validate().rmse
    with pytest.raises(ValueError):
        ok.cross_validate(folds=1)


def test_neighbourhood_diagnostics():
    holes = np.arange(len(values)) // 3
    ok = cs.OrdinaryKriging(model, cs.Search(radius=30, max_samples=8)).fit(coords, values, holes=holes)
    d = ok.predict(rng.uniform(0, 100, (200, 2)), diagnostics=True)
    assert {"n_holes", "mean_distance", "negative_weight_sum", "lagrange", "max_samples_reached"} <= set(d)
    assert np.all((d["n_holes"] >= 1) & (d["n_holes"] <= d["n_samples"]))
    assert np.all((d["mean_distance"] > 0) & (d["mean_distance"] <= 30))
    assert np.all(d["negative_weight_sum"] <= 0) and np.any(d["negative_weight_sum"] < 0)
    assert np.isfinite(d["lagrange"]).all()
    np.testing.assert_array_equal(d["max_samples_reached"], d["n_samples"] == 8)
    idw = cs.InverseDistance(search).fit(coords, values).predict(coords[:3], diagnostics=True)
    assert np.isnan(idw["negative_weight_sum"]).all() and np.isnan(idw["lagrange"]).all()
