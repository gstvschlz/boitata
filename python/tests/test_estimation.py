import pickle

import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(3)
coords = rng.uniform(0, 100, (150, 2))
values = np.sin(coords[:, 0] / 15) + coords[:, 1] / 50
model = cs.Variogram([("spherical", 1.0, 40.0)], nugget=0.0)
search = cs.Search(radius=50, max_samples=16)


def test_ordinary_kriging_honors_data_with_zero_variance():
    est, var = cs.OrdinaryKriging(model, search).fit(coords, values).predict(coords, return_variance=True)
    np.testing.assert_allclose(est, values, atol=1e-8)
    np.testing.assert_allclose(var, 0.0, atol=1e-8)


def test_variance_grows_away_from_data():
    ok = cs.OrdinaryKriging(model, search).fit(coords, values)
    near = coords[:1] + [0.5, 0.0]
    far = np.array([[200.0, 200.0]])
    wide = cs.OrdinaryKriging(model, cs.Search(radius=400)).fit(coords, values)
    assert ok.predict(near, return_variance=True)[1][0] < wide.predict(far, return_variance=True)[1][0]


def test_too_few_neighbors_is_nan():
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


def test_high_grade_modes():
    grid = rng.uniform(0, 100, (300, 2))

    def run(high_grade):
        est = cs.OrdinaryKriging(model, cs.Search(radius=50, high_grade=high_grade)).fit(coords, values)
        return est.predict(grid), est.cross_validate().estimate

    threshold = np.quantile(values, 0.8)
    drop = run((threshold, 5.0))
    for same in [
        cs.HighGrade(threshold, 5.0),
        cs.HighGrade(threshold, (5.0, 5.0, 5.0), rotation=(30, 10, 0)),
    ]:
        for a, b in zip(drop, run(same)):
            np.testing.assert_array_equal(a, b)
    clamp = run(cs.HighGrade(threshold, 5.0, mode="clamp"))
    assert not np.array_equal(drop[0], clamp[0])
    np.testing.assert_array_equal(clamp[0], run(cs.HighGrade(threshold, 5.0, mode="clamp"))[0])
    passes = [
        cs.Search(radius=10, min_samples=6, high_grade=(threshold, 5.0)),
        cs.Search(radius=60, high_grade=(threshold, 20.0)),
    ]
    assert np.isfinite(cs.OrdinaryKriging(model, passes).fit(coords, values).predict(grid)).all()
    h = cs.HighGrade(2.0, (40.0, 20.0, 5.0), rotation=(30.0, 0.0, 0.0), mode="clamp")
    assert (h.threshold, h.radius, h.mode, h.rotation) == (2.0, 40.0, "clamp", (30.0, 0.0, 0.0))
    np.testing.assert_allclose(h.ranges, (40.0, 20.0, 5.0))
    assert cs.HighGrade.from_json(h.to_json()) == h
    assert cs.Search(radius=50, high_grade=h).high_grade == h
    with pytest.raises(ValueError):
        cs.HighGrade(1.0, 5.0, mode="cap")
    with pytest.raises(ValueError):
        cs.HighGrade(1.0, (1.0, 0.0, 1.0))
    clamped = cs.Search(radius=50, high_grade=cs.HighGrade(1.0, 5.0, mode="clamp"))
    with pytest.raises(ValueError):
        cs.MultigaussianKriging(model, clamped).fit(coords, np.exp(values)).predict(grid)


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
    assert set(stats.column_names) >= {"nearest_dist", "value_mean", "n_within"}
    np.testing.assert_allclose(stats["nearest_dist"], 0.0)


def test_cokriging_with_only_primary_data_is_ordinary_kriging():
    lmc = cs.Coregionalization(
        [[0.0, 0.0], [0.0, 0.0]], structures=[("spherical", 40.0, [[1.0, 0.7], [0.7, 1.0]])]
    )
    ck = cs.Cokriging(lmc, search).fit(coords, values, [0] * len(values))
    targets = rng.uniform(0, 100, (25, 2))
    np.testing.assert_allclose(
        ck.predict(targets), cs.OrdinaryKriging(model, search).fit(coords, values).predict(targets)
    )


def test_collocated_cokriging_uses_the_secondary():
    lmc = cs.Coregionalization(
        [[0.0, 0.0], [0.0, 0.0]], structures=[("spherical", 40.0, [[1.0, 0.9], [0.9, 1.0]])]
    )
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
        ok = cs.OrdinaryKriging(cs.Variogram([("spherical", 1.0, 30.0)]), search).fit(xy, v, holes=holes)
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
    assert d.column_names == [
        "value",
        "variance",
        "efficiency",
        "slope",
        "n_samples",
        "pass",
        "n_holes",
        "n_other_domain",
        "mean_distance",
        "negative_weight_sum",
        "lagrange",
        "support_variance",
        "estimate_variance",
        "max_samples_reached",
    ]
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


def test_hole_distance_classification():
    hole_xy = rng.uniform(0, 100, (30, 2))
    xyz = np.repeat(np.c_[hole_xy, np.zeros(30)], 4, axis=0)
    xyz[:, 2] = np.tile(np.arange(4.0), 30)
    holes = np.repeat([f"H{i}" for i in range(30)], 4)
    grid = cs.BlockModel(origin=(0, 0), size=(5, 5), count=(20, 20))
    both = cs.hole_distance(grid, xyz, holes, n=[1, 3])
    assert both.shape == (400, 2) and np.all(both[:, 0] <= both[:, 1])
    d = {3: both[:, 1]}
    np.testing.assert_array_equal(cs.hole_distance(grid, xyz, holes, 3), d[3])
    np.testing.assert_array_equal(cs.hole_distance(grid, xyz[::4], holes[::4], 3), d[3])

    three = np.array([[50.0, 50.0, 0.0]] * 3)
    at = cs.hole_distance([[50.0, 50.0]], np.r_[three, xyz], np.r_[["a", "b", "c"], holes], 3)
    assert at[0] == 0.0

    def labels(measured, indicated):
        rules = [("measured", {"d3": ("<=", measured)}), ("indicated", {"d3": ("<=", indicated)})]
        return cs.classify({"d3": d[3]}, rules, default="inferred")

    tight, loose = labels(10, 20), labels(15, 30)
    assert np.all(loose[tight == "measured"] == "measured")
    assert np.all(loose[tight == "indicated"] != "inferred")

    near = cs.hole_distance(grid, xyz, holes, 3, search=cs.Search(radius=15))
    assert np.all(np.isinf(near) | (near == d[3])) and np.isinf(near).any()
    stretched = cs.hole_distance(grid, xyz, holes, 3, search=cs.Search(radius=1e9, ratios=(0.5, 0.5)))
    assert np.all(stretched >= d[3] - 1e-9)

    east = grid.centroids[:, 0] > 50
    sample_east = xyz[:, 0] > 50
    split = cs.hole_distance(grid, xyz, holes, 3, domains=(east, sample_east))
    own = cs.hole_distance(grid.centroids[east], xyz[sample_east], holes[sample_east], 3)
    np.testing.assert_array_equal(split[east], own)
    assert np.all(split >= d[3])
    blocks = grid.with_column("zone", east * 1.0)
    samples = cs.PointSet(xyz, {"hole": holes, "zone": sample_east * 1.0})
    np.testing.assert_array_equal(cs.hole_distance(blocks, samples, "hole", 3, domain_column="zone"), split)
    with pytest.raises(cs.CeresError):
        cs.hole_distance(blocks, samples, "hole", 3, domains=(east, sample_east), domain_column="zone")

    zone = np.where(east, "E", "W")
    by_zone = cs.classify(
        {"d3": d[3]},
        {"E": [("measured", {"d3": ("<=", 10)})], None: [("measured", {"d3": ("<=", 20)})]},
        default="inferred",
        domains=zone,
    )
    np.testing.assert_array_equal(by_zone[east], labels(10, -1)[east])
    np.testing.assert_array_equal(by_zone[~east], labels(20, -1)[~east])
    only_e = cs.classify({"d3": d[3]}, {"E": [("measured", {"d3": ("<=", 1e9)})]}, domains=zone)
    assert set(only_e[~east]) == {"unclassified"}
    with pytest.raises(ValueError):
        cs.classify({"d3": d[3]}, {"E": []})

    stripes = np.where(np.arange(400) % 20 == 7, "A", "B")
    np.testing.assert_array_equal(cs.smooth_classes(grid, stripes, domains=stripes), stripes)
    assert set(cs.smooth_classes(grid, stripes)) == {"B"}
    with pytest.raises(ValueError):
        cs.hole_distance(grid, xyz, holes, 0)


def test_measurement_error_blends_a_datum_with_its_neighbors():
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


def test_k_fold_keeps_holes_whole():
    holes = np.arange(len(values)) // 5
    cv = cs.OrdinaryKriging(model, search).fit(coords, values, holes=holes).cross_validate(folds=4)
    expected = np.empty(len(values))
    for fold in range(4):
        test = holes % 4 == fold
        train = cs.OrdinaryKriging(model, search).fit(coords[~test], values[~test])
        expected[test] = train.predict(coords[test])
    np.testing.assert_array_equal(cv.estimate, expected)


def test_neighborhood_diagnostics():
    holes = np.arange(len(values)) // 3
    ok = cs.OrdinaryKriging(model, cs.Search(radius=30, max_samples=8)).fit(coords, values, holes=holes)
    d = ok.predict(rng.uniform(0, 100, (200, 2)), diagnostics=True)
    assert {"n_holes", "mean_distance", "negative_weight_sum", "lagrange", "max_samples_reached"} <= set(
        d.column_names
    )
    assert np.all((d["n_holes"] >= 1) & (d["n_holes"] <= d["n_samples"]))
    assert np.all((d["mean_distance"] > 0) & (d["mean_distance"] <= 30))
    assert np.all(d["negative_weight_sum"] <= 0) and np.any(d["negative_weight_sum"] < 0)
    assert np.isfinite(d["lagrange"]).all()
    np.testing.assert_array_equal(d["max_samples_reached"], d["n_samples"] == 8)
    np.testing.assert_array_equal(d["support_variance"], model.sill)
    covariance = d["support_variance"] - d["variance"] - d["lagrange"]
    np.testing.assert_allclose(covariance / d["estimate_variance"], d["slope"], rtol=1e-12)
    idw = cs.InverseDistance(search).fit(coords, values).predict(coords[:3], diagnostics=True)
    assert np.isnan(idw["negative_weight_sum"]).all() and np.isnan(idw["lagrange"]).all()


def test_multiple_indicator_kriging_distributions():
    thresholds = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    variograms = [cs.Variogram([("spherical", 1.0, r)]) for r in (15.0, 40.0, 25.0, 60.0)]
    targets = rng.uniform(0, 100, (60, 2))
    mik = cs.MultipleIndicatorKriging(variograms, search, thresholds, upper_tail=("hyperbolic", 2.0))
    s = mik.fit(coords, values - values.min() + 0.1).predict(
        targets, cutoffs=[1.0], quantiles=[0.1, 0.5, 0.9]
    )
    assert s.cdf.shape == (60, 4) and s.quantile_values.shape == (60, 3)
    assert np.all((s.cdf >= 0) & (s.cdf <= 1)) and np.all(np.diff(s.cdf, axis=1) >= 0)
    assert np.all(np.diff(s.quantile_values, axis=1) >= 0) and np.all(s.std >= 0)
    assert np.all((s.probability_above >= 0) & (s.probability_above <= 1))

    one = cs.MultipleIndicatorKriging(model, search, [thresholds[1]]).fit(coords, values).predict(targets)
    ik = cs.IndicatorKriging(model, search, threshold=thresholds[1]).fit(coords, values)
    np.testing.assert_array_equal(one.cdf[:, 0], ik.predict(targets))


def test_multiple_indicator_simple_form_far_away_gives_the_declustered_mean():
    weights = cs.cell_declustering(coords, values, cell_size=20.0).weights
    mik = cs.MultipleIndicatorKriging(
        model, cs.Search(radius=1e4), np.quantile(values, [0.25, 0.5, 0.75]), simple=True
    )
    s = mik.fit(coords, values, weights=weights).predict([[5000.0, 5000.0], [500.0, 900.0]])
    assert s.mean == pytest.approx(np.average(values, weights=weights), abs=1e-10)
    with pytest.raises(cs.InvalidInput, match="strictly increasing"):
        cs.MultipleIndicatorKriging(model, search, [1.0, 1.0])
    with pytest.raises(cs.InvalidInput, match="one per threshold"):
        cs.MultipleIndicatorKriging([model, model], search, [0.0, 1.0, 2.0])


def test_multiple_indicator_cross_validation_and_diagnostics():
    thresholds = np.quantile(values, [0.25, 0.5, 0.75])
    passes = [cs.Search(radius=6.0, max_samples=8), search]
    mik = cs.MultipleIndicatorKriging(model, passes, thresholds).fit(coords, values)
    cv = mik.cross_validate()
    assert isinstance(cv, cs.IndicatorCrossValidation) and cv.cdf.shape == (len(values), 3)
    assert np.all((cv.pit >= 0) & (cv.pit <= 1)) and cv.rmse < values.std()
    assert cv.brier.shape == (3,) and np.all(cv.brier < 0.25)
    assert cv.accuracy(1.0) == 1.0 and cv.accuracy([0.2, 0.8]).shape == (2,)
    assert 0.5 < cv.goodness <= 1.0

    one = (
        cs.MultipleIndicatorKriging(model, search, [thresholds[1]])
        .fit(coords, values)
        .cross_validate(folds=5)
    )
    ik = (
        cs.IndicatorKriging(model, search, threshold=thresholds[1])
        .fit(coords, values)
        .cross_validate(folds=5)
    )
    np.testing.assert_array_equal(one.cdf[:, 0], np.clip(ik.estimate, 0, 1))

    targets = rng.uniform(0, 100, (80, 2))
    assert mik.predict(targets).diagnostics is None
    s = mik.predict(targets, diagnostics=True)
    d = s.diagnostics
    assert d.column_names == [
        "n_samples",
        "pass",
        "n_holes",
        "mean_distance",
        "max_samples_reached",
        "correction",
        "n_order_violations",
    ]
    np.testing.assert_array_equal(d["correction"], s.correction)
    np.testing.assert_array_equal(d["n_order_violations"] > 0, s.correction > 0)
    np.testing.assert_array_equal(
        d["max_samples_reached"], np.where(d["pass"] == 1, d["n_samples"] == 8, d["n_samples"] == 16)
    )
    assert set(d["pass"]) == {1.0, 2.0}


def test_multigaussian_kriging_distributions():
    grades = np.exp(values)
    mg = cs.MultigaussianKriging(model, search).fit(coords, grades)
    exact = mg.predict(coords, quantiles=[0.1, 0.9])
    np.testing.assert_allclose(exact.mean, grades, rtol=1e-9)
    np.testing.assert_allclose(exact.quantile_values, np.c_[grades, grades], rtol=1e-6)
    cutoffs = np.quantile(grades, [0.2, 0.5, 0.8])
    s = mg.predict(rng.uniform(0, 100, (60, 2)), cutoffs=cutoffs, quantiles=[0.1, 0.5, 0.9], diagnostics=True)
    assert s.thresholds == [] and np.all(s.correction == 0) and np.all(s.std >= 0)
    assert np.all(np.diff(s.probability_above, axis=1) <= 0) and np.all(
        np.diff(s.quantile_values, axis=1) >= 0
    )
    assert np.all((s.mean_above >= cutoffs) | (s.probability_above == 0))
    assert s.diagnostics.column_names[0] == "n_samples"

    blocks = cs.BlockModel(origin=(0, 0, 0), size=(10, 10, 1), count=(10, 10, 1))
    point = mg.predict(blocks, cutoffs=cutoffs)
    np.testing.assert_array_equal(mg.predict(blocks, discretization=(1, 1, 1)).mean, point.mean)
    block = mg.predict(blocks, cutoffs=cutoffs, discretization=(3, 3, 1))
    assert np.nanmean(block.variance) > np.nanmean(point.variance)

    cv = mg.cross_validate(folds=5)
    assert isinstance(cv, cs.IndicatorCrossValidation) and cv.cdf.shape == (len(values), 0)
    assert np.all((cv.pit >= 0) & (cv.pit <= 1)) and cv.rmse < grades.std() and 0.5 < cv.goodness <= 1.0
    with pytest.raises(cs.InvalidInput, match="not fitted"):
        cs.MultigaussianKriging(model, search).predict(coords)
    with pytest.raises(cs.InvalidInput, match="tails"):
        cs.MultigaussianKriging(model, search, tails=(2.0, 1.0))


rock = np.where(values < -0.3, "shale", np.where(values < 0.7, "sand", "lime"))
scheme = cs.Categories(["sand", "shale", "lime"], colors=["gold", "gray", "skyblue"])


def test_categorical_indicator_kriging_gives_a_distribution_exact_at_the_data():
    variograms = [cs.Variogram([("spherical", 0.2, r)], nugget=0.02) for r in (15.0, 40.0, 25.0)]
    for simple in (False, True):
        cik = cs.CategoricalIndicatorKriging(variograms, search, simple=simple, scheme=scheme)
        s = cik.fit(coords, rock).predict(rng.uniform(0, 100, (200, 2)), diagnostics=True)
        assert s.probabilities.shape == (200, 3) and s.names == ["sand", "shale", "lime"]
        assert np.all((s.probabilities >= 0) & (s.probabilities <= 1))
        np.testing.assert_allclose(s.probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(s.most_likely, s.probabilities.argmax(axis=1))
        assert np.all((s.entropy >= 0) & (s.entropy <= 1 + 1e-12))
        assert s.diagnostics["n_order_violations"].max() > 0
        np.testing.assert_array_equal(s.diagnostics["correction"], s.correction)
        at = cik.predict(coords)
        np.testing.assert_allclose(at.probabilities, np.eye(3)[scheme.encode(rock).astype(int)], atol=1e-8)
        assert scheme.decode(at.most_likely) == list(rock)
    one = cs.CategoricalIndicatorKriging(model, search).fit(coords, np.zeros(len(coords), int))
    assert np.all(one.predict(rng.uniform(0, 100, (20, 2))).probabilities == 1.0)


def test_categorical_indicator_kriging_domains_and_proportions():
    weights = cs.cell_declustering(coords, values, cell_size=20.0).weights
    cik = cs.CategoricalIndicatorKriging(model, cs.Search(radius=1e4), simple=True, scheme=scheme)
    s = cik.fit(coords, rock, weights=weights).predict([[5000.0, 5000.0]])
    expected = scheme.shares(scheme.encode(rock), weights=weights)
    np.testing.assert_allclose(s.proportions, expected)
    np.testing.assert_allclose(s.probabilities[0], expected, atol=1e-9)
    points = cs.PointSet(coords, {"rock": rock, "zone": np.where(coords[:, 0] < 50, "w", "e")})
    zoned = cs.CategoricalIndicatorKriging(model, search, scheme=scheme).fit(
        points, "rock", domain_column="zone"
    )
    at = rng.uniform(0, 100, (30, 2))
    targets = cs.PointSet(at, {"zone": np.where(np.arange(30) < 20, np.where(at[:, 0] < 50, "w", "e"), "x")})
    p = zoned.predict(targets, domain_column="zone").probabilities
    assert np.isnan(p[20:]).all() and not np.isnan(p[:20]).any()
    with pytest.raises(cs.InvalidInput, match="needs domains"):
        zoned.predict(targets)
    with pytest.raises(cs.InvalidInput, match="one per category"):
        cs.CategoricalIndicatorKriging([model, model], search, scheme=scheme)
    with pytest.raises(cs.InvalidInput):
        cs.CategoricalIndicatorKriging(model, search, scheme=scheme).fit(
            coords, np.where(rock == "sand", "clay", rock)
        )


def test_categorical_indicator_cross_validation():
    cik = cs.CategoricalIndicatorKriging(model, search, scheme=scheme).fit(coords, rock)
    cv = cik.cross_validate()
    assert isinstance(cv, cs.CategoricalCrossValidation) and cv.probabilities.shape == (len(rock), 3)
    assert cv.brier.shape == (3,) and np.all(cv.brier < 0.2)
    assert np.mean(cv.most_likely == scheme.encode(rock)) > 0.7
    np.testing.assert_array_equal(cik.cross_validate(folds=len(rock)).probabilities, cv.probabilities)


def test_block_kriging_with_a_pure_nugget_has_no_block_variance():
    nugget = cs.Variogram([], nugget=1.0)
    d = (
        cs.BlockKriging(nugget, search, size=(10, 10))
        .fit(coords, values)
        .predict([[50.0, 50.0]], diagnostics=True)
    )
    assert d["variance"][0] == pytest.approx(1.0 / 16)


zone = np.where(coords[:, 0] + 0.3 * coords[:, 1] < 50, "MS", "SM")
grid = np.stack(np.meshgrid(np.arange(0, 100, 4.0), np.arange(0, 100, 4.0)), -1).reshape(-1, 2)
grid_zone = np.where(grid[:, 0] + 0.3 * grid[:, 1] < 50, "MS", "SM")


def zoned(soft=None, passes=((30, 16), (80, 8))):
    searches = [cs.Search(radius=r, max_samples=n, soft=soft) for r, n in passes]
    return cs.OrdinaryKriging(model, searches).fit(coords, values, domains=zone)


def test_hard_domains_are_separate_estimations():
    ok = zoned()
    at = ok.predict(grid, domains=grid_zone)
    cv = ok.cross_validate().estimate
    field = cs.LocalAnisotropy(
        grid, np.tile([35.0, 0.0, 0.0], (len(grid), 1)), np.tile([0.3, 1.0], (len(grid), 1))
    )
    local = ok.predict(grid, anisotropy=field, domains=grid_zone)
    passes = [cs.Search(radius=30, max_samples=16), cs.Search(radius=80, max_samples=8)]
    for name in ("MS", "SM"):
        alone = cs.OrdinaryKriging(model, passes).fit(coords[zone == name], values[zone == name])
        inside = grid_zone == name
        np.testing.assert_array_equal(at[inside], alone.predict(grid[inside]))
        np.testing.assert_array_equal(cv[zone == name], alone.cross_validate().estimate)
        np.testing.assert_array_equal(local[inside], alone.predict(grid[inside], anisotropy=field))
    assert np.isfinite(at).all()


def test_soft_distance_zero_is_hard_and_infinite_pools_the_domains():
    hard = zoned().predict(grid, domains=grid_zone)
    np.testing.assert_array_equal(zoned(soft=0.0).predict(grid, domains=grid_zone), hard)
    passes = [cs.Search(radius=30, max_samples=16), cs.Search(radius=80, max_samples=8)]
    free = cs.OrdinaryKriging(model, passes).fit(coords, values)
    pooled = zoned(soft=np.inf)
    np.testing.assert_array_equal(pooled.predict(grid, domains=grid_zone), free.predict(grid))
    np.testing.assert_array_equal(pooled.cross_validate().estimate, free.cross_validate().estimate)
    assert not np.array_equal(zoned(soft=10.0).predict(grid, domains=grid_zone), hard)


def test_soft_pairs_are_one_way_and_counted():
    one_way = zoned(soft={("SM", "MS"): 10.0}).predict(grid, diagnostics=True, domains=grid_zone)
    hard = zoned().predict(grid, domains=grid_zone)
    ms, sm = grid_zone == "MS", grid_zone == "SM"
    np.testing.assert_array_equal(one_way["value"][ms], hard[ms])
    assert np.all(one_way["n_other_domain"][ms] == 0) and one_way["n_other_domain"][sm].max() > 0
    near = one_way["n_other_domain"] > 0
    assert not np.array_equal(one_way["value"][near], hard[near])
    free = cs.OrdinaryKriging(model, search).fit(coords, values).predict(grid, diagnostics=True)
    assert np.all(free["n_other_domain"] == 0)


def test_domains_in_predict_and_cross_validation():
    ok = zoned(soft=8.0)
    single = ok.predict(grid[:5], domains="MS")
    np.testing.assert_array_equal(single, ok.predict(grid[:5], domains=["MS"] * 5))
    assert np.isnan(ok.predict(grid[:5], domains="QE")).all()
    holes = np.arange(len(values)) // 5
    ok = cs.OrdinaryKriging(model, cs.Search(radius=50, soft=8.0)).fit(
        coords, values, holes=holes, domains=zone
    )
    cv = ok.cross_validate(folds=4).estimate
    expected = np.empty(len(values))
    for fold in range(4):
        test = holes % 4 == fold
        train = cs.OrdinaryKriging(model, cs.Search(radius=50, soft=8.0))
        train.fit(coords[~test], values[~test], domains=zone[~test])
        expected[test] = train.predict(coords[test], domains=zone[test])
    np.testing.assert_array_equal(cv, expected)


def test_labels_of_any_simple_type_and_shared_locations_across_domains():
    codes = np.where(zone == "MS", 1, 2)
    a = cs.InverseDistance(cs.Search(radius=50, soft={(1, 2): 5.0})).fit(coords, values, domains=codes)
    b = cs.InverseDistance(cs.Search(radius=50, soft={(1.0, 2): 5.0})).fit(
        coords, values, domains=list(codes)
    )
    np.testing.assert_array_equal(a.predict(grid, domains=1), b.predict(grid, domains=np.int64(1)))
    twin = np.vstack([coords[:1], coords[:1]])
    kept = cs.NearestNeighbor(search).fit(twin, [1.0, 2.0], domains=["a", "b"])
    np.testing.assert_array_equal(kept.predict(twin, domains=["a", "b"]), [1.0, 2.0])


def test_a_shared_contact_location_keeps_the_target_domain_sample():
    both = np.vstack([coords, [[50.0, 50.0], [50.0, 50.0]]])
    labels = np.r_[zone, ["SM", "MS"]]
    data = np.r_[values, 5.0, -5.0]
    ok = cs.OrdinaryKriging(model, cs.Search(radius=30, soft=np.inf)).fit(both, data, domains=labels)
    at = ok.predict([[50.0, 50.0]] * 2, domains=["MS", "SM"])
    np.testing.assert_allclose(at, [-5.0, 5.0], atol=1e-8)
    near = ok.predict([[50.5, 50.0]] * 2, domains=["MS", "SM"], diagnostics=True)
    assert np.isfinite(near["value"]).all() and near["value"][0] < near["value"][1]
    assert np.isfinite(ok.cross_validate().estimate[-2:]).all()


def test_domain_errors():
    with pytest.raises(cs.InvalidInput, match="predict needs domains"):
        zoned().predict(grid)
    with pytest.raises(cs.InvalidInput, match="takes none"):
        cs.OrdinaryKriging(model, search).fit(coords, values).predict(grid, domains="MS")
    soft = cs.Search(radius=50, soft=5.0)
    with pytest.raises(cs.InvalidInput, match="needs domains at fit"):
        cs.OrdinaryKriging(model, soft).fit(coords, values)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        zoned(soft={("MS", "QE"): 5.0})
    with pytest.raises(cs.InvalidInput, match="expected 150"):
        cs.OrdinaryKriging(model, search).fit(coords, values, domains=zone[:10])
    with pytest.raises(cs.InvalidInput, match=">= 0"):
        cs.Search(radius=50, soft={("MS", "SM"): -1.0})
    lmc = cs.Coregionalization(
        [[0.1, 0.0], [0.0, 0.1]], structures=[("spherical", 40.0, [[1.0, 0.6], [0.6, 1.0]])]
    )
    for make in (
        lambda: cs.MultivariateSimulation(cs.PCA(), [cs.SGS(model, soft)]),
        lambda: cs.SIS([model], soft),
        lambda: cs.Cokriging(lmc, soft),
        lambda: cs.MultipleIndicatorKriging(model, soft, [1.0]),
        lambda: cs.DisjunctiveKriging(cs.HermiteAnamorphosis().fit(values + 2), model, soft),
    ):
        with pytest.raises(cs.InvalidInput, match="does not take domains"):
            make()


def test_search_getters():
    s = cs.Search(
        radius=50,
        octant=True,
        max_per_hole=3,
        rotation=(30, 10, 5),
        ratios=(0.5, 0.2),
        high_grade=(2.0, 10.0),
    )
    assert (s.octant, s.max_per_hole, s.high_grade) == (True, 3, cs.HighGrade(2.0, 10.0))
    assert (s.rotation, s.ratios) == ((30, 10, 5), (0.5, 0.2))
    plain = cs.Search(radius=50)
    assert not plain.octant
    assert plain.max_per_hole is plain.high_grade is plain.rotation is plain.ratios is plain.soft is None
    assert cs.Search(radius=50, soft=5.0).soft == 5.0
    pairs = {("MS", "SM"): 5.0, (1, True): 2.0}
    assert cs.Search(radius=50, soft=pairs).soft == pairs


def test_with_search_keeps_samples_and_domains():
    passes = [search, cs.Search(radius=200, max_samples=4)]
    ok = cs.OrdinaryKriging(model, passes).fit(coords, values)
    targets = rng.uniform(-50, 150, (300, 2))
    same = ok.with_search(passes)
    assert type(same) is cs.OrdinaryKriging
    np.testing.assert_array_equal(same.predict(targets), ok.predict(targets))
    np.testing.assert_array_equal(same.cross_validate(folds=5).estimate, ok.cross_validate(folds=5).estimate)
    other = ok.with_search(cs.Search(radius=50, max_samples=4)).predict(targets)
    assert not np.array_equal(other, ok.predict(targets), equal_nan=True)

    soft = zoned(soft=10.0)
    again = soft.with_search([cs.Search(radius=r, max_samples=n, soft=10.0) for r, n in ((30, 16), (80, 8))])
    np.testing.assert_array_equal(
        again.predict(grid, domains=grid_zone), soft.predict(grid, domains=grid_zone)
    )
    with pytest.raises(cs.InvalidInput, match="no samples"):
        soft.with_search(cs.Search(radius=30, soft={("MS", "XX"): 5.0}))


def gaussian_field(n=300, seed=11):
    g = np.random.default_rng(seed)
    at = g.uniform(0, 200, (n, 2))
    field = cs.Variogram([("spherical", 0.8, 60.0)], nugget=0.2)
    h = np.linalg.norm(at[:, None] - at[None], axis=-1)
    cov = field.sill - field.gamma(h.ravel()).reshape(n, n) + np.where(h == 0, field.nugget, 0.0)
    return field, at, np.linalg.cholesky(cov) @ g.standard_normal(n)


blocks = cs.BlockModel(origin=(5.0, 5.0), size=(10, 10), count=(20, 20))


def test_calibration_block_variance_excludes_the_nugget():
    one = cs.Search(radius=1e4, max_samples=4)
    nugget = cs.BlockKriging(cs.Variogram([], nugget=1.0), one, size=(10, 10)).fit(coords, values)
    assert cs.calibrate_search(nugget, [one], blocks, cross_validation=False)["block_variance"][0] == 0.0

    c0, c, a, length = 0.4, 1.0, 100.0, 60.0
    segment = cs.Variogram([("spherical", c, a)], nugget=c0)
    kriging = cs.BlockKriging(segment, one, size=(length, 1.0), discretization=(400, 1, 1)).fit(
        coords, values
    )
    gamma_bar = c * (length / (2 * a) - length**3 / (20 * a**3))
    table = cs.calibrate_search(kriging, [one], [[50.0, 50.0]], cross_validation=False)
    assert table["block_variance"][0] == pytest.approx(c - gamma_bar, abs=1e-4)


def test_simple_kriging_with_a_unique_neighborhood_has_slope_one():
    unique = cs.Search(radius=1e6, max_samples=len(values))
    sk = cs.SimpleKriging(model, unique, mean=float(values.mean())).fit(coords, values)
    inside = cs.BlockModel(origin=(5.0, 5.0), size=(10, 10), count=(10, 10))
    table = cs.calibrate_search(sk, [unique], inside, cross_validation=False)
    assert table["slope_mean"][0] == 1.0 and table["slope_p10"][0] == 1.0


def test_more_samples_raise_the_slope_and_smooth_the_estimates():
    field, at, z = gaussian_field()
    kriging = cs.BlockKriging(field, search, size=(10, 10), discretization=(3, 3, 1)).fit(at, z)
    scenarios = [cs.Search(radius=1e4, max_samples=n) for n in (2, 4, 8, 16)]
    table = cs.calibrate_search(kriging, scenarios, blocks)
    assert np.all(np.diff(table["slope_mean"]) > 0), table["slope_mean"]
    assert np.all(np.diff(table["variance_ratio"]) < 0), table["variance_ratio"]
    assert np.all(np.diff(table["model_variance_ratio"]) < 0), table["model_variance_ratio"]
    np.testing.assert_array_equal(table["scenario"], np.arange(4))


def test_calibration_scores_are_deterministic():
    field, at, z = gaussian_field()
    z = np.exp(z)
    weights = cs.cell_declustering(at, z, cell_size=20.0).weights
    kriging = cs.BlockKriging(field, search, size=(10, 10)).fit(at, z)
    anamorphosis = cs.HermiteAnamorphosis().fit(z, weights=weights)
    passes = [cs.Search(radius=12, max_samples=8, min_samples=4), cs.Search(radius=1e4, max_samples=8)]

    def run():
        return cs.calibrate_search(
            kriging,
            [search, passes],
            blocks,
            folds=5,
            weights=weights,
            cutoffs=[1.0, 2.0],
            anamorphosis=anamorphosis,
        )

    first, second = run(), run()
    assert first.column_names == second.column_names
    for name in first.column_names:
        np.testing.assert_array_equal(first[name], second[name])
    assert {"tonnage_ratio_1", "metal_ratio_2", "cv_rmse", "global_bias"} <= set(first.column_names)
    assert first["first_pass"][1] < 1.0 and first["estimated"][1] == 1.0
    assert all(np.isfinite(first[name]).all() for name in first.column_names)


def test_calibration_cross_validates_block_kriging_at_points():
    kriging = cs.BlockKriging(model, search, size=(10, 10)).fit(coords, values)
    table = cs.calibrate_search(kriging, [search], blocks)
    cv = cs.OrdinaryKriging(model, search).fit(coords, values).cross_validate()
    assert table["cv_rmse"][0] == pytest.approx(cv.rmse, rel=1e-12)
    assert table["cv_slope"][0] == pytest.approx(cv.slope, rel=1e-9)
    assert table["cv_mean_error"][0] == pytest.approx(cv.mean_error, rel=1e-9)


def test_calibration_errors():
    kriging = cs.BlockKriging(model, search, size=(10, 10)).fit(coords, values)
    with pytest.raises(ValueError, match="sequence of scenarios"):
        cs.calibrate_search(kriging, search, blocks)
    with pytest.raises(ValueError, match="HermiteAnamorphosis"):
        cs.calibrate_search(kriging, [search], blocks, cutoffs=[1.0])
    with pytest.raises(ValueError, match="weights for"):
        cs.calibrate_search(kriging, [search], blocks, weights=np.ones(3))
    idw = cs.InverseDistance(search).fit(coords, values)
    anamorphosis = cs.HermiteAnamorphosis().fit(values + 2)
    with pytest.raises(ValueError, match="variogram"):
        cs.calibrate_search(idw, [search], blocks, cutoffs=[1.0], anamorphosis=anamorphosis)
    with pytest.raises(ValueError, match="no target"):
        cs.calibrate_search(kriging, [cs.Search(radius=1, min_samples=3)], [[500.0, 500.0]])


def test_calibration_by_domain_matches_the_domain_alone():
    weights = cs.cell_declustering(coords, values, cell_size=20.0).weights
    scenarios = [
        cs.Search(radius=30, max_samples=8),
        [cs.Search(radius=30, max_samples=16), cs.Search(radius=80)],
    ]
    at, inside = grid[grid_zone == "MS"], zone == "MS"
    kriging = zoned()
    by_domain = cs.calibrate_search(kriging, scenarios, at, weights=weights, domains="MS")
    alone = cs.OrdinaryKriging(model, search).fit(coords[inside], values[inside])
    reference = cs.calibrate_search(alone, scenarios, at, weights=weights[inside])
    assert by_domain.column_names == reference.column_names
    for name in by_domain.column_names:
        np.testing.assert_allclose(by_domain[name], reference[name], rtol=1e-12, atol=1e-15, err_msg=name)

    hard, soft = cs.Search(radius=30, max_samples=16), cs.Search(radius=30, max_samples=16, soft=10.0)
    table = cs.calibrate_search(kriging, [hard, soft], grid, domains=grid_zone)
    assert np.all(table["estimated"] == 1.0)
    assert table["slope_mean"][0] != table["slope_mean"][1]
    with pytest.raises(cs.InvalidInput, match="predict needs domains"):
        cs.calibrate_search(kriging, [hard], grid)


def test_multiple_indicator_localization():
    thresholds = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    mik = cs.MultipleIndicatorKriging(model, search, thresholds, interpolation="linear").fit(coords, values)
    panels = cs.BlockModel(origin=(0, 0), size=(25, 25), count=(8, 4))
    smus = panels.discretize(5)
    smus = smus.with_column("rank", rng.normal(size=len(smus)))
    owner, rank = smus["block"].astype(int), smus["rank"]
    n, f = 25, 0.4
    k = np.arange(1, n)
    s = mik.predict(panels, quantiles=list(1 - k / n))
    out = mik.localize(smus, "rank", panels, variance_factor=f)["localized"]
    for p, m in enumerate(s.mean):
        mine = owner == p
        if np.isnan(m):
            assert np.isnan(out[mine]).all()
            continue
        by_rank = out[mine][np.argsort(rank[mine], kind="stable")]
        assert by_rank.mean() == pytest.approx(m, abs=1e-9)
        assert (np.diff(by_rank) >= -1e-12).all()
        point = mik.predict(panels.centroids[p : p + 1], cutoffs=list(s.quantile_values[p]))
        top = np.cumsum(by_rank[::-1])[:-1] / k
        np.testing.assert_allclose(top, m + np.sqrt(f) * (point.mean_above[0] - m), atol=1e-9)
    assert np.isnan(s.mean).any() and not np.isnan(s.mean).all()

    point = mik.localize(smus, "rank", panels, variance_factor=1.0)["localized"]
    derived = mik.localize(smus, "rank", panels, variance_factor=model)["localized"]
    default = mik.localize(smus, "rank", panels, name="g")["g"]
    np.testing.assert_array_equal(derived, default)
    ok = ~np.isnan(point)
    assert 0 < np.var((derived - s.mean[owner])[ok]) < np.var((point - s.mean[owner])[ok])
    with pytest.raises(cs.InvalidInput):
        mik.localize(smus, "rank", panels, variance_factor=1.5)
    with pytest.raises(cs.InvalidInput):
        shifted = cs.BlockModel(origin=(3, 0), size=(5, 5), count=(4, 4)).with_column("rank", np.zeros(16))
        mik.localize(shifted, "rank", panels)


def test_block_multiple_indicator_kriging():
    thresholds = np.quantile(values, [0.25, 0.5, 0.75])
    wide = cs.Search(radius=1e4)
    panels = cs.BlockModel(origin=(0, 0), size=(25, 25), count=(4, 4))
    for simple in (False, True):
        mik = cs.MultipleIndicatorKriging(model, wide, thresholds, simple=simple).fit(coords, values)
        point = mik.predict(panels)
        np.testing.assert_array_equal(mik.predict(panels, discretization=(1, 1, 1)).cdf, point.cdf)
        block = mik.predict(panels, discretization=(4, 4, 1), diagnostics=True)
        assert block.cdf[:, 1].var() < point.cdf[:, 1].var() and block.diagnostics is not None
    smus = panels.discretize(5).with_column("rank", rng.normal(size=400))
    local = mik.localize(smus, "rank", panels, discretization=(4, 4, 1))["localized"]
    owner = smus["block"].astype(int)
    for p, m in enumerate(block.mean):
        assert local[owner == p].mean() == pytest.approx(m, abs=1e-9)
    with pytest.raises(cs.InvalidInput, match="BlockModel"):
        mik.predict(coords[:3], discretization=(2, 2, 1))


def test_column_names_match_arrays():
    holes = np.arange(len(values)) // 3
    zone = np.where(coords[:, 0] < 50, 1, 2)
    samples = cs.PointSet(coords, {"grade": values, "hole": holes, "zone": zone})
    grid = cs.BlockModel(origin=(0, 0), size=(10, 10), count=(10, 10))
    blocks = grid.with_column("zone", np.where(grid.centroids[:, 0] < 50, 1, 2))
    ok = cs.OrdinaryKriging(model, search)
    arrays = ok.fit(coords, values, holes=holes, domains=zone).predict(grid, domains=blocks["zone"])
    names = ok.fit(samples, "grade", holes="hole", domain_column="zone").predict(blocks, domain_column="zone")
    np.testing.assert_array_equal(names, arrays)
    w = np.linspace(1, 2, len(values))
    weighted = samples.with_column("w", w)
    fitted = cs.OrdinaryKriging(model, search).fit(weighted, "grade")
    by_name = cs.calibrate_search(fitted, [search], grid, weights="w", data=weighted)
    by_array = cs.calibrate_search(fitted, [search], grid, weights=w)
    assert by_name["cv_rmse"][0] == by_array["cv_rmse"][0] != fitted.cross_validate().rmse
    with pytest.raises(cs.InvalidInput, match="data="):
        cs.calibrate_search(fitted, [search], grid, weights="w")
    assert b"PointSet" not in pickle.dumps(fitted) + pickle.dumps(fitted.with_search(search))
    a = cs.neighborhood_stats(grid, coords, values, k=5, holes=holes)
    b = cs.neighborhood_stats(grid, samples, "grade", k=5, holes="hole")
    for c in a.column_names:
        np.testing.assert_array_equal(a[c], b[c])
    mik = cs.MultipleIndicatorKriging(model, search, np.quantile(values, [0.3, 0.7]))
    first = mik.fit(coords, values, holes=holes).predict(grid).mean
    np.testing.assert_array_equal(mik.fit(samples, "grade", holes="hole").predict(grid).mean, first)
    criteria = cs.Table({"slope": np.linspace(0, 1, 100), "zone": blocks["zone"]})
    rules = {2: [("measured", {"slope": (">=", 0.5)})]}
    np.testing.assert_array_equal(
        cs.classify(criteria, rules, domain_column="zone"),
        cs.classify(criteria, rules, domains=blocks["zone"]),
    )


def test_domains_and_domain_column_are_exclusive():
    zone = np.where(coords[:, 0] < 50, "W", "E")
    samples = cs.PointSet(coords, {"grade": values, "zone": zone})
    ok = cs.OrdinaryKriging(model, search)
    with pytest.raises(cs.InvalidInput, match="one of domains or domain_column"):
        ok.fit(samples, "grade", domains=zone, domain_column="zone")
    ok.fit(samples, "grade", domain_column="zone")
    with pytest.raises(cs.InvalidInput, match="one of domains or domain_column"):
        ok.predict(samples, domains=zone, domain_column="zone")
    with pytest.raises(cs.MissingColumn):
        ok.predict(samples, domain_column="rock")
    with pytest.raises(cs.InvalidInput, match="one of domains or domain_column"):
        cs.classify({"zone": zone}, [], domains=zone, domain_column="zone")
    with pytest.raises(cs.InvalidInput, match="one of domains or domain_column"):
        cs.calibrate_search(ok, [search], samples, domains="W", domain_column="zone", cross_validation=False)


def test_defaulted_arguments_are_keyword_only():
    ok = cs.OrdinaryKriging(model, search).fit(coords, values)
    lmc = cs.Coregionalization([[0.0]], structures=[("spherical", 40.0, [[1.0]])])
    calls = [
        lambda: ok.fit(coords, values, None),
        lambda: ok.predict(coords, True),
        lambda: cs.neighborhood_stats(coords, coords, values, 5),
        lambda: cs.hole_distance(coords, coords, np.arange(len(coords)), 1, None),
        lambda: cs.global_bias(values, values, None),
        lambda: cs.classify({"a": values}, [], "x"),
        lambda: cs.calibrate_search(ok, [search], coords, 5),
        lambda: cs.MultipleIndicatorKriging(model, search, [1.0]).fit(coords, values, None),
        lambda: cs.Cokriging(lmc, search).fit(coords, values, np.zeros(len(values))).predict(coords, 0),
    ]
    for call in calls:
        with pytest.raises(TypeError):
            call()


def test_weight_declustering_downweights_clusters_and_shares_duplicates():
    grid = np.array([(x, y) for x in range(5, 100, 10) for y in range(5, 100, 10)], float)
    cluster = rng.uniform(15, 30, (60, 2))
    xy = np.vstack([grid, cluster, grid[:1]])
    z = np.sin(xy[:, 0] / 15) + xy[:, 1] / 50
    targets = cs.BlockModel((0, 0, 0), (2, 2, 1), (50, 50, 1))
    for estimator in [cs.OrdinaryKriging(model, search), cs.InverseDistance(search, power=2)]:
        with pytest.warns(UserWarning):
            d = cs.weight_declustering(xy, z, targets, estimator=estimator)
        assert d.weights.sum() == pytest.approx(len(xy))
        assert d.weights[100:160].mean() < 0.5 * d.weights[:100].mean()
        assert d.weights[0] == d.weights[-1]
        assert d.mean == pytest.approx(np.average(z, weights=d.weights))
        assert np.isnan(d.cell_size)
    near = cs.weight_declustering(grid, grid[:, 0], targets, estimator=cs.NearestNeighbor(search))
    np.testing.assert_allclose(near.weights, 1.0)
    with pytest.raises(ValueError, match="linear"):
        cs.weight_declustering(grid, grid[:, 0], targets, estimator=cs.MovingMedian(search))
