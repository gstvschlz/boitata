import json
import os
import pathlib
import pickle
import subprocess
import sys

import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(5)
coords = rng.uniform(0, 100, (60, 2))
values = rng.lognormal(0, 0.6, 60)
gaussian = bt.Variogram([("spherical", 1.0, 30.0)])
grid = bt.BlockModel(origin=(0, 0), size=(5, 5), count=(20, 20))


def test_sgs_is_reproducible_and_honors_data():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    a = sgs.simulate(grid, n=3, seed=7, keep=True).realizations
    assert a.shape == (3, 400)
    np.testing.assert_array_equal(a, sgs.simulate(grid, n=3, seed=7, keep=True).realizations)
    assert not np.array_equal(a[0], a[1])
    at_data = sgs.simulate(coords[:5], n=4, seed=1)
    np.testing.assert_allclose(at_data.mean, values[:5])
    np.testing.assert_allclose(at_data.variance, 0, atol=1e-12)


def test_sgs_summary_matches_its_realizations():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    cut = float(np.median(values))
    s = sgs.simulate(grid, n=20, seed=3, cutoffs=[cut], quantiles=[0.1, 0.5, 0.9], keep=True)
    reals = s.realizations
    np.testing.assert_allclose(s.mean, reals.mean(axis=0))
    np.testing.assert_allclose(s.variance, reals.var(axis=0), atol=1e-9)
    np.testing.assert_allclose(s.probability_above[:, 0], (reals > cut).mean(axis=0))
    np.testing.assert_allclose(s.quantile_values[:, 1], np.median(reals, axis=0), rtol=1e-6)
    np.testing.assert_allclose(s.realization_above[:, 0], (reals > cut).mean(axis=1))
    assert np.median(reals) == pytest.approx(np.median(values), rel=0.25)
    assert sgs.simulate(grid, n=2, seed=3).realizations is None


def test_derived_summary_statistics():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    s = sgs.simulate(grid, n=30, seed=4, quantiles=[0.05, 0.5, 0.95], keep=True)
    reals = s.realizations
    np.testing.assert_allclose(s.cv, reals.std(axis=0) / reals.mean(axis=0), rtol=1e-9)
    lo, mid, hi = s.quantile_values.T
    np.testing.assert_allclose(s.relative_error(), (hi - lo) / (2 * s.mean))
    np.testing.assert_allclose(s.relative_error(center="median"), (hi - lo) / (2 * mid))
    with pytest.raises(bt.InvalidInput, match="needs quantiles 0.1, 0.9"):
        s.relative_error(confidence=0.8)
    facies = np.random.default_rng(9).choice(3, 60, p=[0.2, 0.3, 0.5])
    pgs = bt.Plurigaussian(gaussian, proportions=[0.2, 0.3, 0.5]).fit(coords, facies)
    c = pgs.simulate(grid, n=8, seed=2)
    p, least = c.probabilities, c.least_likely
    seen = (p > 0).sum(axis=1)
    assert (least[seen < 2] == -1).all()
    rows = np.flatnonzero(seen >= 2)
    masked = np.where(p[rows] > 0, p[rows], np.inf)
    np.testing.assert_array_equal(least[rows], masked.argmin(axis=1))


def test_grade_tonnage_of_a_constant_ensemble_is_the_data_curve():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    domain = np.where(coords[:, 0] < 50, "west", "east")
    points = bt.PointSet(coords, {"d": np.full(60, 2.7), "domain": domain})
    cuts = [0.5, 1.0, 2.0]
    s = sgs.simulate(points, n=5, seed=1, grade_tonnage_cutoffs=cuts, density="d", categories="domain")
    curves = s.grade_tonnage(probabilities=[0.5])
    expected = bt.grade_tonnage(values, cuts, density=2.7, categories=domain)
    assert curves["category"].tolist() == expected["category"].tolist()
    for q in ("tonnage", "metal", "mean_grade"):
        np.testing.assert_allclose(curves[q], expected[q], rtol=1e-9)


def test_grade_tonnage_uncertainty_on_blocks(tmp_path):
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    blocks = bt.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    s = sgs.simulate(
        grid, n=20, seed=2, keep=True, blocks=blocks, grade_tonnage_cutoffs=[0.0, 1.0], density=2.5
    )
    reals = s.realizations
    above = reals >= 1.0
    tonnes = 2.5 * 400.0 * above.sum(axis=1)
    curves = s.grade_tonnage(probabilities=[0.1, 0.5, 0.9])
    rows = (curves["cutoff"] == 1.0) & (curves["category"] == "all")
    np.testing.assert_allclose(curves["tonnage"][rows], np.quantile(tonnes, [0.1, 0.5, 0.9]))
    np.testing.assert_allclose(
        curves["tonnage"][(curves["cutoff"] == 0.0) & (curves["probability"] == 0.5)], 2.5 * 400 * 25
    )
    s.to_parquet(tmp_path / "s.parquet")
    again = bt.SimulationSummary.from_parquet(tmp_path / "s.parquet")
    np.testing.assert_array_equal(again.grade_tonnage()["metal"], s.grade_tonnage()["metal"])
    import matplotlib

    matplotlib.use("Agg")
    _, ax = bt.plot.grade_tonnage(curves)
    assert len(ax.lines) == 1 and len(ax.collections) == 1
    with pytest.raises(bt.InvalidInput, match="one of density or tonnage"):
        sgs.simulate(grid, n=1, grade_tonnage_cutoffs=[1.0])
    with pytest.raises(bt.InvalidInput, match="go with grade_tonnage_cutoffs"):
        sgs.simulate(grid, n=1, density=2.5)
    with pytest.raises(bt.InvalidInput, match="no grade-tonnage"):
        sgs.simulate(grid, n=1).grade_tonnage()


def test_block_support_averages_each_realization():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    blocks = bt.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    nodes = sgs.simulate(grid, n=6, seed=2, keep=True).realizations
    s = sgs.simulate(grid, n=6, seed=2, cutoffs=[1.0], keep=True, blocks=blocks)
    xy = grid.centroids[:, :2] // 20
    rows = (xy[:, 0] + 5 * xy[:, 1]).astype(int)
    expected = np.stack([np.bincount(rows, r) / np.bincount(rows) for r in nodes])
    np.testing.assert_allclose(s.realizations, expected)
    np.testing.assert_allclose(s.mean, expected.mean(axis=0))
    assert (s.variance <= np.bincount(rows, nodes.var(axis=0)) / 16 + 1e-9).all()
    with pytest.raises(ValueError):
        sgs.simulate(grid, n=1, blocks=bt.BlockModel(origin=(0, 0), size=(20, 20), count=(6, 5)))

    cats = (values > np.median(values)).astype(int)
    sis = bt.SIS([gaussian, gaussian], bt.Search(radius=40, max_samples=12)).fit(coords, cats)
    c = sis.simulate(grid, n=3, seed=4, keep=True, blocks=blocks)
    assert c.realizations.shape == (3, 25) and c.probabilities.shape == (25, 2)


def _box_mean(xy, reals, half):
    near = (np.abs(xy[:, None, :] - xy[None, :, :]) <= half + 1e-9).all(axis=2)
    return reals @ near.T / near.sum(axis=1)


@pytest.mark.parametrize(
    "make",
    [
        lambda: bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)),
        lambda: bt.DSS(bt.Variogram([("spherical", 0.4, 30.0)]), bt.Search(radius=40, max_samples=12)),
        lambda: bt.TurningBands(gaussian, bands=100, step=1.0),
    ],
)
@pytest.mark.filterwarnings("ignore:.*no draw reaches")
def test_windows_and_groups_summarize_each_realization(make, tmp_path):
    sim = make().fit(coords, values)
    nodes = sim.simulate(grid, n=5, seed=2, keep=True, progress=False).realizations
    xy = grid.centroids[:, :2]
    s = sim.simulate(grid, n=5, seed=2, keep=True, quantiles=[0.5], window=(15, 10), progress=False)
    expected = _box_mean(xy, nodes, np.array([7.5, 5.0]))
    np.testing.assert_allclose(s.realizations, expected, rtol=1e-9)
    np.testing.assert_allclose(s.mean, expected.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(s.quantile_values[:, 0], np.median(expected, axis=0), rtol=1e-9)
    assert s.groups is None

    period = np.array(["b", "a", "c", "a"])[(xy[:, 0] // 25).astype(int)]
    with_period = grid.with_column("period", period)
    g = sim.simulate(
        with_period,
        n=5,
        seed=2,
        keep=True,
        groups="period",
        grade_tonnage_cutoffs=[0.0],
        density=2.0,
        progress=False,
    )
    labels = np.unique(period)
    expected = np.stack([nodes[:, period == p].mean(axis=1) for p in labels], axis=1)
    np.testing.assert_allclose(g.realizations, expected, rtol=1e-9)
    np.testing.assert_allclose(g.variance, expected.var(axis=0), atol=1e-9)
    assert g.groups == ["a", "b", "c"]
    g.to_parquet(tmp_path / "g.parquet")
    for again in (bt.SimulationSummary.from_parquet(tmp_path / "g.parquet"), pickle.loads(pickle.dumps(g))):
        assert again.groups == g.groups
        np.testing.assert_array_equal(again.realizations, g.realizations)
    total = g.grade_tonnage(probabilities=[0.5])
    assert total["tonnage"][total["category"] == "all"][0] == pytest.approx(2.0 * 25 * 400)


@pytest.mark.parametrize(
    "make",
    [
        lambda: bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)),
        lambda: bt.DSS(bt.Variogram([("spherical", 0.4, 30.0)]), bt.Search(radius=40, max_samples=12)),
        lambda: bt.TurningBands(gaussian, bands=100, step=1.0),
    ],
)
@pytest.mark.filterwarnings("ignore:.*no draw reaches")
def test_precision_counts_realizations_near_the_mean(make, tmp_path):
    sim = make().fit(coords, values)
    period = np.arange(400) % 3
    for rows in ({}, {"groups": period}, {"window": (15, 10)}):
        s = sim.simulate(grid, n=12, seed=2, keep=True, tolerances=[0.1, 0.3], progress=False, **rows)
        reals, mean = s.realizations, s.mean
        expected = np.stack(
            [(np.abs(reals - mean) <= r * np.abs(mean)).mean(axis=0) for r in (0.1, 0.3)], axis=1
        )
        assert s.tolerances == [0.1, 0.3] and s.precision.shape == (len(mean), 2)
        np.testing.assert_array_equal(s.precision, expected)
    s.to_parquet(tmp_path / "s.parquet")
    for again in (bt.SimulationSummary.from_parquet(tmp_path / "s.parquet"), pickle.loads(pickle.dumps(s))):
        assert again.tolerances == s.tolerances
        np.testing.assert_array_equal(again.precision, s.precision)
    plain = sim.simulate(grid, n=2, progress=False)
    assert plain.tolerances == [] and plain.precision.shape == (400, 0)


def test_validate_against_a_truth():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    s = sgs.simulate(grid, n=30, seed=1, cutoffs=[1.0], quantiles=[0.05, 0.95], progress=False)
    truth = sgs.simulate(grid, n=1, seed=99, keep=True, progress=False).realizations[0]
    t = s.validate(truth, cutoff=1.0)
    assert t.column_names == ["truth", "mean", "error", "covered", "type_1", "type_2"]
    np.testing.assert_allclose(t["error"], (s.mean - truth) / truth)
    mee = s.relative_error(confidence=0.9)
    np.testing.assert_array_equal(t["covered"], np.abs(truth - s.mean) / np.abs(s.mean) <= np.abs(mee))
    ore = s.probability_above[:, 0] >= 0.5
    np.testing.assert_array_equal(t["type_1"], ore & (truth <= 1.0))
    np.testing.assert_array_equal(t["type_2"], ~ore & (truth > 1.0))
    by_name = s.validate("truth", data=grid.with_column("truth", truth))
    assert by_name.column_names == ["truth", "mean", "error", "covered"]
    np.testing.assert_array_equal(by_name["covered"], t["covered"])

    with pytest.raises(bt.InvalidInput, match="needs quantiles 0.1, 0.9"):
        s.validate(truth, confidence=0.8)
    with pytest.raises(bt.InvalidInput, match="cutoffs="):
        s.validate(truth, cutoff=2.0)
    with pytest.raises(bt.InvalidInput, match="3 truth values for 400 rows"):
        s.validate(truth[:3])


def test_group_grade_is_tonnage_weighted():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    nodes = sgs.simulate(grid, n=4, seed=3, keep=True, progress=False).realizations
    period = np.arange(400) % 3
    density = 1.5 + (np.arange(400) % 7) * 0.4
    g = sgs.simulate(
        grid.with_column("density", density),
        n=4,
        seed=3,
        keep=True,
        groups=period,
        grade_tonnage_cutoffs=[0.0],
        density="density",
        progress=False,
    )
    for p in range(3):
        rows = period == p
        metal = (nodes[:, rows] * density[rows]).sum(axis=1)
        np.testing.assert_allclose(g.realizations[:, p] * density[rows].sum(), metal, rtol=1e-12)


def test_windows_over_blocks_points_and_bad_arguments():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    blocks = bt.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    on_blocks = sgs.simulate(grid, n=3, seed=1, keep=True, blocks=blocks, progress=False).realizations
    s = sgs.simulate(grid, n=3, seed=1, keep=True, blocks=blocks, window=(40, 40, 0), progress=False)
    np.testing.assert_allclose(s.realizations, _box_mean(blocks.centroids[:, :2], on_blocks, 20.0), rtol=1e-9)

    points = rng.uniform(0, 100, (50, 2))
    raw = sgs.simulate(points, n=3, seed=1, keep=True, progress=False).realizations
    s = sgs.simulate(points, n=3, seed=1, keep=True, window=(30, 30), progress=False)
    np.testing.assert_allclose(s.realizations, _box_mean(points, raw, 15.0), rtol=1e-9)
    ids = np.arange(50) % 3
    s = sgs.simulate(points, n=3, seed=1, keep=True, groups=ids, progress=False)
    np.testing.assert_allclose(
        s.realizations, np.stack([raw[:, ids == i].mean(axis=1) for i in range(3)], axis=1)
    )
    assert s.groups == [0, 1, 2]

    with pytest.raises(bt.InvalidInput, match="one of window or groups"):
        sgs.simulate(grid, n=1, window=(10, 10), groups=np.zeros(400))
    with pytest.raises(bt.InvalidInput, match="categories do not go with groups"):
        sgs.simulate(
            grid,
            n=1,
            groups=np.zeros(400),
            grade_tonnage_cutoffs=[0.0],
            density=1.0,
            categories=np.zeros(400),
        )
    with pytest.raises(bt.InvalidInput, match="2 or 3 values"):
        sgs.simulate(grid, n=1, window=(10,))
    with pytest.raises(bt.InvalidInput, match="non-negative"):
        sgs.simulate(grid, n=1, window=(10, -1))
    with pytest.raises(ValueError):
        sgs.simulate(grid, n=1, groups=np.zeros(3))


def test_localize_realizations_within_panels():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    panels = bt.BlockModel(origin=(0, 0), size=(50, 50), count=(2, 2))
    smus = panels.discretize(5)
    s = sgs.simulate(grid, n=8, seed=2, keep=True, blocks=smus)
    smus = smus.with_column("etype", s.mean)
    out = bt.localize(smus, "etype", panels, s.realizations)
    owner, local = smus["block"].astype(int), out["localized"]
    for p in range(4):
        mine = owner == p
        pooled = np.sort(s.realizations[:, mine].ravel())
        assert local[mine].mean() == pytest.approx(pooled.mean(), rel=1e-9)
        by_rank = local[mine][np.argsort(s.mean[mine], kind="stable")]
        np.testing.assert_allclose(by_rank, pooled.reshape(25, 8).mean(axis=1), rtol=1e-12)

    one = bt.localize(smus, "etype", panels, s.realizations[:1], name="one")["one"]
    for p in range(4):
        mine = owner == p
        by_rank = one[mine][np.argsort(s.mean[mine], kind="stable")]
        np.testing.assert_array_equal(by_rank, np.sort(s.realizations[0, mine]))
    with pytest.raises(bt.InvalidInput):
        bt.localize(smus, "etype", panels, s.realizations[:, 1:])
    with pytest.raises(KeyError):
        bt.localize(smus, "missing", panels, s.realizations)


def test_turning_bands_summary():
    tb = bt.TurningBands(gaussian, bands=100, step=1.0).fit(coords, values)
    s = tb.simulate(grid, n=2, seed=1, cutoffs=[1.0, 2.0])
    assert s.mean.shape == (400,) and s.probability_above.shape == (400, 2)


def test_sis_probabilities():
    cats = (values > np.median(values)).astype(int)
    sis = bt.SIS([gaussian, gaussian], bt.Search(radius=40, max_samples=12)).fit(coords, cats)
    s = sis.simulate(grid, n=4, seed=4, keep=True)
    assert s.probabilities.shape == (400, 2) and s.proportions.shape == (4, 2)
    np.testing.assert_allclose(s.probabilities.sum(axis=1), 1.0)
    assert set(np.unique(s.realizations)) <= {0, 1}
    assert ((s.entropy >= 0) & (s.entropy <= 1)).all()


def test_sis_follows_local_proportions():
    far = np.array([[-1e3, 0.0], [-1e3, 5.0]])
    line = np.c_[np.arange(200) + 0.5, np.zeros(200)]
    east = line[:, 0] / 200
    local = bt.Table({"west": 1 - east, "east": east})
    sis = bt.SIS([gaussian, gaussian], bt.Search(radius=30, max_samples=12))
    sis.fit(far, [0, 1], proportions=[[0.5, 0.5]] * 2)
    s = sis.simulate(line, n=60, seed=1, proportions=local)
    by_half = s.probabilities[:, 1].reshape(2, 100).mean(axis=1)
    np.testing.assert_allclose(by_half, [0.25, 0.75], atol=0.08)
    again = sis.simulate(line, n=60, seed=1, proportions=np.c_[1 - east, east])
    np.testing.assert_array_equal(again.probabilities, s.probabilities)
    with pytest.raises(bt.InvalidInput, match="both fit and simulate"):
        sis.simulate(line, n=1)
    with pytest.raises(bt.InvalidInput, match="shape"):
        sis.fit(far, [0, 1], proportions=[[1.0, 0.0, 0.0]] * 2)


def test_plurigaussian_proportions():
    facies = rng.choice(3, 60, p=[0.2, 0.3, 0.5])
    pgs = bt.Plurigaussian(gaussian, proportions=[0.2, 0.3, 0.5]).fit(coords, facies)
    s = pgs.simulate(grid, n=2, seed=2)
    assert s.probabilities.shape == (400, 3) and set(np.unique(s.most_likely)) <= {0, 1, 2}


def test_plurigaussian_hierarchy_honors_data_on_three_fields():
    facies = rng.choice(4, 60, p=[0.4, 0.3, 0.2, 0.1])
    rule = (0, [0, (1, [1, (2, [2, 3])])])
    pgs = bt.Plurigaussian([gaussian] * 3, proportions=[0.4, 0.3, 0.2, 0.1], rule=rule).fit(coords, facies)
    s = pgs.simulate(coords, n=2, seed=4, keep=True)
    np.testing.assert_array_equal(s.realizations, [facies, facies])
    with pytest.raises(ValueError, match="splits field 2"):
        bt.Plurigaussian([gaussian] * 2, proportions=[0.4, 0.3, 0.2, 0.1], rule=rule)
    with pytest.raises(ValueError, match="every facies"):
        bt.Plurigaussian(gaussian, proportions=[0.5, 0.5], rule=(0, [0]))
    regions = [
        ([(-np.inf, 0.0)], 0),
        ([(0.0, np.inf), (-np.inf, 0.0)], 1),
        ([(0.0, np.inf), (0.0, np.inf)], 2),
    ]
    with pytest.raises(ValueError, match="thresholds 2 fields"):
        bt.Plurigaussian(gaussian, regions=regions)
    two = bt.Plurigaussian([gaussian] * 2, regions=regions).fit(coords, facies % 3)
    assert set(np.unique(two.simulate(grid, n=1, seed=1).most_likely)) <= {0, 1, 2}


def test_plurigaussian_fitted_variograms_recover_the_latent_ranges():
    rule = (0, [0, (1, [1, 2])])
    truth = [bt.Variogram([("spherical", 1.0, 16.0)]), bt.Variogram([("exponential", 1.0, 8.0)])]
    fine = bt.BlockModel(origin=(0, 0), size=(2, 2), count=(60, 60))
    xy = fine.centroids[:, :2]
    pgs = bt.Plurigaussian(truth, proportions=[0.3, 0.4, 0.3], rule=rule).fit([[60.0, 60.0]], [1])
    facies = pgs.simulate(fine, n=1, seed=5, keep=True).realizations[0]
    experimental = [bt.experimental_variogram(xy, (facies == f).astype(float), 2.0, 24.0) for f in range(3)]
    start = [bt.Variogram([("spherical", 1.0, 5.0)]), bt.Variogram([("exponential", 1.0, 80.0)])]
    fitted = bt.Plurigaussian(start, proportions=[0.3, 0.4, 0.3], rule=rule).fit_variograms(experimental)
    ranges = [v.structures[0].range for v in fitted.variograms]
    assert ranges == pytest.approx([16.0, 8.0], rel=0.25)
    implied = fitted.indicator_variograms([0.0, 1e6])
    assert implied.shape == (3, 2)
    np.testing.assert_allclose(implied[:, 1], [0.21, 0.24, 0.21], atol=1e-6)
    with pytest.raises(ValueError, match="per facies"):
        fitted.fit_variograms(experimental[:2])


def test_plurigaussian_follows_local_proportions():
    def west_to_east(xy):
        p0 = 0.9 - 0.8 * xy[:, 0] / 100
        return np.column_stack([p0, (1 - p0) / 2, (1 - p0) / 2])

    xy = np.array([[50.0, 50.0]])
    rule = (0, [0, (1, [1, 2])])
    pgs = bt.Plurigaussian([gaussian] * 2, proportions=[0.5, 0.25, 0.25], rule=rule)
    pgs.fit(xy, [1], proportions=west_to_east(xy))
    nodes = grid.centroids[:, :2]
    s = pgs.simulate(grid, n=10, seed=1, proportions=west_to_east(nodes))
    west, east = nodes[:, 0] < 30, nodes[:, 0] > 70
    assert s.probabilities[west, 0].mean() > 0.6 and s.probabilities[east, 0].mean() < 0.4
    with pytest.raises(ValueError, match="at both fit and simulate"):
        pgs.simulate(grid, n=1)
    with pytest.raises(ValueError, match="shape"):
        pgs.simulate(grid, n=1, proportions=np.ones((400, 2)))
    with pytest.raises(ValueError, match="built from proportions"):
        bt.Plurigaussian(gaussian, regions=[([(-np.inf, np.inf)], 0)]).fit(xy, [0], proportions=[[1.0]])


def test_gibbs_respects_bounds():
    pts = rng.uniform(0, 50, (20, 2))
    bounds = np.column_stack([np.zeros(20), np.full(20, np.inf)])
    draw = bt.gibbs(pts, bounds, gaussian, seed=3)
    assert np.all(draw >= 0)


def trended_samples():
    r = np.random.default_rng(8)
    xy = r.uniform(0, 100, (400, 2))
    trend = xy[:, 0] / 100
    return xy, np.exp(1.5 * trend + 0.5 * r.normal(size=400)), trend


def test_simulation_with_a_trend_follows_it_and_honors_data():
    xy, z, trend = trended_samples()
    search = bt.Search(radius=30, max_samples=12)
    node_trend = grid.centroids[:, 0] / 100
    white = bt.Variogram([("spherical", 1.0, 4.0)])
    for simulator in (bt.SGS(white, search, classes=5), bt.TurningBands(white, bands=100, classes=5)):
        simulator.fit(xy, z, trend=trend)
        reals = simulator.simulate(grid, n=4, seed=2, keep=True, trend=node_trend).realizations
        want = np.corrcoef(np.log(z), trend)[0, 1]
        for r in reals:
            assert np.corrcoef(np.log(r), node_trend)[0, 1] == pytest.approx(want, abs=0.12)
        at_data = simulator.simulate(xy[:5], n=3, seed=1, trend=trend[:5])
        np.testing.assert_allclose(at_data.mean, z[:5])
        np.testing.assert_allclose(at_data.variance, 0, atol=1e-9)

    sgs = bt.SGS(gaussian, search).fit(xy, z, trend=trend)
    model = grid.with_column("drift", node_trend)
    by_array = sgs.simulate(grid, n=3, seed=4, keep=True, path="random", trend=node_trend).realizations
    np.testing.assert_array_equal(
        sgs.simulate(model, n=3, seed=4, keep=True, path="random", trend="drift").realizations, by_array
    )
    points = bt.PointSet(model.centroids, {"drift": node_trend})
    np.testing.assert_array_equal(
        sgs.simulate(points, n=3, seed=4, keep=True, path="random", trend="drift").realizations, by_array
    )
    blocks = bt.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    by_block = sgs.simulate(
        grid, n=3, seed=4, keep=True, path="random", blocks=blocks, trend=node_trend
    ).realizations
    xy_block = grid.centroids[:, :2] // 20
    rows = (xy_block[:, 0] + 5 * xy_block[:, 1]).astype(int)
    np.testing.assert_allclose(by_block, [np.bincount(rows, r) / np.bincount(rows) for r in by_array])

    flat = bt.SGS(gaussian, search).fit(xy, z, trend=np.ones(len(z)))
    plain = bt.SGS(gaussian, search).fit(xy, z)
    np.testing.assert_allclose(
        flat.simulate(grid, n=2, seed=3, trend=np.ones(400)).mean, plain.simulate(grid, n=2, seed=3).mean
    )

    with pytest.raises(bt.InvalidInput, match="give trend"):
        sgs.simulate(grid, n=1)
    with pytest.raises(bt.InvalidInput, match="needs trend at fit"):
        plain.simulate(grid, n=1, trend=node_trend)
    with pytest.raises(bt.InvalidInput, match="needs a container"):
        sgs.simulate(grid.centroids, n=1, trend="drift")
    with pytest.raises(bt.MissingColumn, match="drift"):
        sgs.simulate(grid, n=1, trend="drift")
    with pytest.raises(ValueError):
        sgs.simulate(grid, n=1, trend=node_trend[1:])
    with pytest.raises(ValueError):
        bt.SGS(gaussian, search).fit(xy, z, trend=trend[1:])
    with pytest.raises(bt.InvalidInput, match="give trend"):
        bt.TurningBands(gaussian).fit(xy, z, trend=trend).simulate_to_parquet("in.parquet", "out.parquet")


def test_simulators_with_a_trend_save_and_load_it(tmp_path):
    xy, z, trend = trended_samples()
    node_trend = grid.centroids[:, 0] / 100
    for simulator in (
        bt.SGS(gaussian, bt.Search(radius=30, max_samples=12), classes=4),
        bt.TurningBands(gaussian, bands=50, classes=4),
    ):
        simulator.fit(xy, z, trend=trend)
        want = simulator.simulate(grid, n=2, seed=6, keep=True, trend=node_trend).realizations
        path = tmp_path / "simulator.parquet"
        simulator.to_parquet(path)
        for back in (type(simulator).from_parquet(path), pickle.loads(pickle.dumps(simulator))):
            got = back.simulate(grid, n=2, seed=6, keep=True, trend=node_trend).realizations
            np.testing.assert_array_equal(got, want)


def test_multivariate_simulation_reproduces_correlation_and_honors_data():
    g = rng.standard_normal((60, 2))
    data = np.exp(np.column_stack([g[:, 0], 0.8 * g[:, 0] + 0.6 * g[:, 1]]))
    search = bt.Search(radius=40, max_samples=12)
    mv = bt.MultivariateSimulation(
        bt.PPMT(seed=3), [bt.SGS(gaussian, search), bt.TurningBands(gaussian, bands=100, step=1.0)]
    ).fit(coords, data)
    a, b = mv.simulate(grid, n=10, seed=4, keep=True)
    assert a.realizations.shape == b.realizations.shape == (10, 400)
    r = np.mean([np.corrcoef(np.log(x), np.log(y))[0, 1] for x, y in zip(a.realizations, b.realizations)])
    assert r == pytest.approx(np.corrcoef(np.log(data.T))[0, 1], abs=0.15)
    np.testing.assert_array_equal(a.realizations, mv.simulate(grid, n=10, seed=4, keep=True)[0].realizations)

    at_data = mv.simulate(coords[:5], n=3, seed=1)
    for v, s in enumerate(at_data):
        np.testing.assert_allclose(s.mean, data[:5, v], rtol=1e-6)

    blocks = bt.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    xy = grid.centroids[:, :2] // 20
    rows = (xy[:, 0] + 5 * xy[:, 1]).astype(int)
    by_block = mv.simulate(grid, n=10, seed=4, keep=True, blocks=blocks)
    for s, nodes in zip(by_block, (a, b)):
        expected = np.stack([np.bincount(rows, x) / np.bincount(rows) for x in nodes.realizations])
        np.testing.assert_allclose(s.realizations, expected)


def test_multivariate_simulation_drops_incomplete_samples_and_checks_inputs():
    data = np.column_stack([values, values**0.5])
    data[:4, 1] = np.nan
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12))
    with pytest.warns(UserWarning, match="4 samples miss a variable"):
        bt.MultivariateSimulation(bt.StepwiseConditional(), [sgs, sgs]).fit(coords, data)
    data[:, 1] *= rng.lognormal(0, 0.5, 60)
    data[4] = np.nan
    mv = bt.MultivariateSimulation(bt.PPMT(seed=3), [sgs, sgs])
    with pytest.warns(UserWarning, match="1 samples miss every variable"):
        mv.fit(coords, data, impute=True)
    at_data = mv.simulate(coords[:4], n=6, seed=2, keep=True)
    np.testing.assert_allclose(at_data[0].realizations, np.tile(values[:4], (6, 1)), rtol=1e-6)
    imputed = at_data[1].realizations
    assert np.isfinite(imputed).all() and np.ptp(imputed, axis=0).min() > 0
    with pytest.warns(UserWarning):
        mv.fit(coords, data, impute=bt.GaussianImputer(spatial=gaussian, neighbors=8))
    spatial = mv.simulate(coords[:4], n=6, seed=2, keep=True)
    assert np.isfinite(spatial[1].realizations).all()
    np.testing.assert_allclose(spatial[0].realizations, at_data[0].realizations)
    with pytest.raises(ValueError, match="impute must be"):
        mv.fit(coords, data, impute="yes")
    with pytest.raises(ValueError, match="transform must be"):
        bt.MultivariateSimulation(bt.NormalScore(), [sgs])
    with pytest.raises(ValueError, match="2 columns"):
        bt.MultivariateSimulation(bt.PCA(), [sgs, sgs]).fit(coords, data[:, :1])


@pytest.mark.parametrize("kind", ["SGS", "TurningBands", "SIS", "MultivariateSimulation"])
def test_max_per_hole_caps_the_data_of_one_hole(kind, tmp_path):
    down = np.column_stack([np.zeros(10), np.zeros(10), np.arange(10.0)])
    grades = np.arange(10.0) * 7 % 10 + 1
    data = {
        "SGS": grades,
        "TurningBands": grades,
        "SIS": np.arange(10) % 2,
        "MultivariateSimulation": np.column_stack([grades, grades**0.5 + np.arange(10) % 3]),
    }[kind]

    def simulator(max_samples, max_per_hole=None):
        search = bt.Search(50.0, max_samples=max_samples, max_per_hole=max_per_hole)
        if kind == "SGS":
            return bt.SGS(gaussian, search)
        if kind == "TurningBands":
            return bt.TurningBands(gaussian, bands=50, step=1.0, search=search)
        if kind == "SIS":
            return bt.SIS([gaussian, gaussian], search)
        bands = bt.TurningBands(gaussian, bands=50, step=1.0, search=search)
        return bt.MultivariateSimulation(bt.PCA(), [bt.SGS(gaussian, search), bands])

    def run(model):
        out = model.simulate([[3.0, 0.0, 4.4]], n=3, seed=2, keep=True)
        return [s.realizations for s in out] if isinstance(out, list) else [out.realizations]

    holes = ["DH1"] * 10
    capped = simulator(8, 1).fit(down, data, holes=holes)
    np.testing.assert_array_equal(run(capped), run(simulator(1).fit(down, data)))
    free = run(simulator(8).fit(down, data))
    np.testing.assert_array_equal(run(simulator(8).fit(down, data, holes=holes)), free)
    if kind != "SIS":
        assert not np.array_equal(run(capped), free)
    if kind != "MultivariateSimulation":
        capped.to_parquet(tmp_path / "capped.parquet")
        np.testing.assert_array_equal(
            run(type(capped).from_parquet(tmp_path / "capped.parquet")), run(capped)
        )
        meta, columns = capped._state()
        older = type(capped)._from_state(meta, [c for c in columns if c[0] != "hole"])
        np.testing.assert_array_equal(run(older), run(simulator(8, 1).fit(down, data)))


@pytest.mark.parametrize("kind", [bt.SGS, bt.TurningBands])
def test_max_per_hole_caps_the_data_of_one_hole_with_a_trend(kind):
    down = np.column_stack([np.zeros(10), np.zeros(10), np.arange(10.0)])
    grades, trend = np.arange(10.0) * 7 % 10 + 1, np.arange(10.0) / 10

    def run(max_samples, max_per_hole=None, holes=None):
        search = bt.Search(50.0, max_samples=max_samples, max_per_hole=max_per_hole)
        options = {"bands": 50, "step": 1.0} if kind is bt.TurningBands else {}
        model = kind(gaussian, search=search, classes=2, **options).fit(
            down, grades, holes=holes, trend=trend
        )
        return model.simulate([[3.0, 0.0, 4.4]], n=3, seed=2, keep=True, trend=[0.45]).realizations

    capped = run(8, 1, ["DH1"] * 10)
    np.testing.assert_array_equal(capped, run(1))
    assert not np.array_equal(capped, run(8))


def holes_in_clusters():
    r = np.random.default_rng(8)
    tops = np.vstack([r.uniform(0, 25, (20, 2)), r.uniform(50, 100, (10, 2))])
    xyz = np.column_stack([np.repeat(tops, 5, axis=0), np.tile(np.arange(5.0), 30)])
    grades = r.lognormal(0, 0.8, 150)
    passes = [
        bt.Search(r, max_samples=12, min_samples=m, max_per_hole=3, high_grade=(4.0, 6.0))
        for r, m in ((10, 8), (25, 4), (40, 2))
    ]
    return xyz, grades, np.repeat(np.arange(30), 5), passes


def test_sgs_passes_are_the_kriging_passes():
    xyz, grades, holes, passes = holes_in_clusters()
    targets = bt.BlockModel(origin=(0, 0, 2), size=(2.5, 2.5, 1), count=(40, 40, 1))
    kriged = bt.OrdinaryKriging(gaussian, passes).fit(xyz, grades, holes=holes)
    want = kriged.predict(targets, diagnostics=True)["pass"]
    sgs = bt.SGS(gaussian, passes).fit(xyz, grades, holes=holes)
    np.testing.assert_array_equal(sgs.passes(targets), want)
    assert {1.0, 2.0, 3.0} <= set(want[~np.isnan(want)]) and np.isnan(want).any()


def test_sgs_with_passes_is_reproducible_and_one_pass_is_the_search(tmp_path):
    xyz, grades, holes, passes = holes_in_clusters()
    targets = bt.BlockModel(origin=(0, 0, 2), size=(5, 5, 1), count=(20, 20, 1))

    def run(model):
        return model.fit(xyz, grades, holes=holes).simulate(targets, n=3, seed=4, keep=True).realizations

    np.testing.assert_array_equal(run(bt.SGS(gaussian, [passes[1]])), run(bt.SGS(gaussian, passes[1])))
    by_pass = bt.SGS(gaussian, passes)
    reals = run(by_pass)
    np.testing.assert_array_equal(reals, run(bt.SGS(gaussian, passes)))
    assert not np.array_equal(reals, run(bt.SGS(gaussian, passes[2])))
    by_pass.to_parquet(tmp_path / "sgs.parquet")
    np.testing.assert_array_equal(
        bt.SGS.from_parquet(tmp_path / "sgs.parquet").simulate(targets, n=3, seed=4, keep=True).realizations,
        reals,
    )
    meta, columns = bt.SGS(gaussian, passes[1]).fit(xyz, grades, holes=holes)._state()
    older = json.loads(meta)
    older["search"] = older["search"][0]
    np.testing.assert_array_equal(
        bt.SGS._from_state(json.dumps(older), columns).simulate(targets, n=3, seed=4, keep=True).realizations,
        run(bt.SGS(gaussian, passes[1])),
    )
    mv = bt.MultivariateSimulation(bt.PCA(), [bt.SGS(gaussian, passes), bt.SGS(gaussian, passes[0])])
    two = mv.fit(xyz, np.column_stack([grades, grades**0.5]), holes=holes).simulate(targets, n=2, seed=1)
    assert len(two) == 2 and np.isfinite(two[0].mean).all()
    with pytest.raises(ValueError, match="at least one"):
        bt.SGS(gaussian, [])


def zoned_holes():
    xyz, grades, holes, passes = holes_in_clusters()
    zone = np.where(xyz[:, 0] < 40, "MS", "SM")
    grades = np.where(zone == "SM", grades * 3, grades)
    weights = np.where(xyz[:, 0] < 25, 0.5, 2.0)
    x, y = np.meshgrid(np.arange(0, 100, 5.0), np.arange(0, 100, 5.0))
    targets = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 2.0)])
    return xyz, grades, holes, weights, zone, targets, passes


def softened(passes, soft):
    return [bt.Search(s.radius, max_samples=12, min_samples=s.min_samples, soft=soft) for s in passes]


def test_sgs_with_one_domain_is_sgs_without():
    xyz, grades, holes, weights, _, targets, passes = zoned_holes()
    plain = bt.SGS(gaussian, passes).fit(xyz, grades, weights=weights, holes=holes)
    one = bt.SGS(gaussian, passes).fit(xyz, grades, weights=weights, holes=holes, domains="MS")
    want = plain.simulate(targets, n=3, seed=4, keep=True).realizations
    got = one.simulate(targets, n=3, seed=4, keep=True, domains="MS").realizations
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(one.passes(targets, domains="MS"), plain.passes(targets))


def test_hard_sgs_domains_ignore_the_other_domain():
    xyz, grades, holes, weights, zone, targets, passes = zoned_holes()
    ms = zone == "MS"

    def run(search, domains=None):
        data = (
            (xyz, grades, weights, holes)
            if domains is not None
            else (xyz[ms], grades[ms], weights[ms], holes[ms])
        )
        sgs = bt.SGS(gaussian, search).fit(data[0], data[1], weights=data[2], holes=data[3], domains=domains)
        on = {"domains": "MS"} if domains is not None else {}
        return sgs.simulate(targets, n=3, seed=1, keep=True, **on).realizations

    np.testing.assert_array_equal(run(passes, zone), run(passes))
    alone = run(softened(passes, None))
    np.testing.assert_array_equal(run(softened(passes, 0.0), zone), alone)
    assert not np.array_equal(run(softened(passes, 10.0), zone), alone)


def test_each_sgs_domain_keeps_its_own_declustered_mean():
    xyz, grades, _, weights, zone, _, _ = zoned_holes()
    far = np.column_stack([1e4 + 100.0 * np.arange(40), np.full(40, 1e4), np.zeros(40)])
    labels = np.repeat(["MS", "SM"], 20)
    sgs = bt.SGS(gaussian, bt.Search(np.inf, max_samples=8)).fit(xyz, grades, weights=weights, domains=zone)
    mean = sgs.simulate(far, n=200, seed=3, domains=labels).mean
    for name in ("MS", "SM"):
        want = np.average(grades[zone == name], weights=weights[zone == name])
        assert mean[labels == name].mean() == pytest.approx(want, rel=0.1)


def test_a_contact_node_takes_its_own_domain_datum():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [-10.0, 0.0]])
    sgs = bt.SGS(gaussian, bt.Search(50.0, soft=np.inf)).fit(
        xy, [1.0, 9.0, 2.0, 8.0], domains=["A", "B", "A", "B"]
    )
    at = sgs.simulate([[0.0, 0.0], [0.0, 0.0]], n=4, seed=1, keep=True, domains=["A", "B"])
    np.testing.assert_array_equal(at.realizations, [[1.0, 9.0]] * 4)


def test_sgs_domains_errors_and_persistence(tmp_path):
    xyz, grades, holes, weights, zone, targets, passes = zoned_holes()
    soft = bt.Search(30.0, max_samples=12, soft={("MS", "SM"): 8.0})
    sgs = bt.SGS(gaussian, soft).fit(xyz, grades, weights=weights, holes=holes, domains=zone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")

    def run(model):
        return model.simulate(targets, n=2, seed=5, keep=True, domains=labels).realizations

    sgs.to_parquet(tmp_path / "sgs.parquet")
    for again in (bt.SGS.from_parquet(tmp_path / "sgs.parquet"), pickle.loads(pickle.dumps(sgs))):
        np.testing.assert_array_equal(run(again), run(sgs))
    with pytest.raises(bt.InvalidInput, match="simulate needs domains"):
        sgs.simulate(targets, n=1)
    with pytest.raises(bt.InvalidInput, match="passes needs domains"):
        sgs.passes(targets)
    with pytest.raises(bt.InvalidInput, match="has no samples"):
        sgs.simulate(targets, n=1, domains="QE")
    with pytest.raises(bt.InvalidInput, match="takes none"):
        bt.SGS(gaussian, passes).fit(xyz, grades).simulate(targets, n=1, domains="MS")
    with pytest.raises(bt.InvalidInput, match="needs domains at fit"):
        bt.SGS(gaussian, soft).fit(xyz, grades)
    with pytest.raises(bt.InvalidInput, match="has no samples"):
        bt.SGS(gaussian, soft).fit(xyz, grades, domains="MS")


def test_dss_is_reproducible_and_honors_data():
    dss = bt.DSS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    s = dss.simulate(grid, n=3, seed=7, keep=True, cutoffs=[1.0], quantiles=[0.5])
    assert s.realizations.shape == (3, 400) and s.probability_above.shape == (400, 1)
    np.testing.assert_array_equal(s.realizations, dss.simulate(grid, n=3, seed=7, keep=True).realizations)
    assert values.min() <= s.realizations.min() and s.realizations.max() <= values.max()
    at_data = dss.simulate(coords[:5], n=4, seed=1)
    np.testing.assert_allclose(at_data.mean, values[:5])
    np.testing.assert_allclose(at_data.variance, 0, atol=1e-12)
    np.testing.assert_array_equal(dss.passes(grid), np.ones(400))


def test_each_dss_domain_keeps_its_own_declustered_mean_and_range():
    xyz, grades, _, weights, zone, _, _ = zoned_holes()
    far = np.column_stack([1e4 + 100.0 * np.arange(40), np.full(40, 1e4), np.zeros(40)])
    labels = np.repeat(["MS", "SM"], 20)
    dss = bt.DSS(gaussian, bt.Search(np.inf, max_samples=8)).fit(xyz, grades, weights=weights, domains=zone)
    s = dss.simulate(far, n=200, seed=3, domains=labels, keep=True)
    for name in ("MS", "SM"):
        mine = grades[zone == name]
        want = np.average(mine, weights=weights[zone == name])
        assert s.mean[labels == name].mean() == pytest.approx(want, rel=0.1)
        reals = s.realizations[:, labels == name]
        assert mine.min() <= reals.min() and reals.max() <= mine.max()


def test_dss_warns_with_the_clamped_fraction():
    smooth = bt.Variogram([("gaussian", 1.0, 200.0)])
    dss = bt.DSS(smooth, bt.Search(radius=60, max_samples=16)).fit(coords, values)
    with pytest.warns(UserWarning, match=r"\d+\.\d+% of simulated nodes"):
        dss.simulate(grid, n=2, seed=1)
    with pytest.raises(bt.InvalidInput, match="positive sill"):
        bt.DSS(bt.Variogram([("power", 1.0, 1.5)]), bt.Search(radius=40))
    with pytest.raises(bt.InvalidInput, match="not fitted"):
        bt.DSS(gaussian, bt.Search(radius=40)).simulate(grid, n=1)


def test_dss_domains_persist(tmp_path):
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    soft = bt.Search(30.0, max_samples=12, soft={("MS", "SM"): 8.0})
    dss = bt.DSS(gaussian, soft).fit(xyz, grades, weights=weights, holes=holes, domains=zone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")

    def run(model):
        return model.simulate(targets, n=2, seed=5, keep=True, domains=labels).realizations

    dss.to_parquet(tmp_path / "dss.parquet")
    for again in (bt.DSS.from_parquet(tmp_path / "dss.parquet"), pickle.loads(pickle.dumps(dss))):
        np.testing.assert_array_equal(run(again), run(dss))
    with pytest.raises(bt.InvalidInput, match='expected a SGS, found "DSS"'):
        bt.SGS.from_parquet(tmp_path / "dss.parquet")
    with pytest.raises(bt.InvalidInput, match="simulate needs domains"):
        dss.simulate(targets, n=1)


def test_sgs_domains_with_a_trend():
    xyz, grades, holes, weights, zone, targets, passes = zoned_holes()
    trend, at = xyz[:, 1] / 100, targets[:, 1] / 100
    ms = zone == "MS"

    def run(search, rows, labels=None, **on):
        sgs = bt.SGS(gaussian, search, classes=3).fit(
            xyz[rows],
            grades[rows],
            weights=weights[rows],
            holes=holes[rows],
            trend=trend[rows],
            domains=labels,
        )
        return sgs.simulate(targets, n=3, seed=2, keep=True, trend=at, **on).realizations

    everything = np.ones(len(zone), bool)
    alone = run(passes, ms)
    np.testing.assert_array_equal(run(passes, everything, zone, domains="MS"), alone)
    np.testing.assert_array_equal(run(passes, ms, "MS", domains="MS"), alone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")
    soft = run(softened(passes, 10.0), everything, zone, domains=labels)
    hard = run(softened(passes, None), everything, zone, domains=labels)
    assert np.isfinite(soft).all() and not np.array_equal(soft, hard)


def banded(search=None):
    return bt.TurningBands(gaussian, bands=60, step=1.0, search=search, classes=3)


def test_turning_bands_domains_are_hard_unless_soft():
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    # Corners holding every datum: the bands cover one box whatever the data.
    targets = np.vstack([targets, [[0.0, 0.0, 0.0], [100.0, 100.0, 4.0]]])
    trend, at = xyz[:, 1] / 100, targets[:, 1] / 100
    ms, everything = zone == "MS", np.ones(len(zone), bool)
    hard = bt.Search(40.0, max_samples=12, max_per_hole=3, high_grade=(4.0, 6.0))

    def run(search, rows, labels=None, trended=False, **on):
        tb = banded(search).fit(
            xyz[rows],
            grades[rows],
            weights=weights[rows],
            holes=holes[rows],
            trend=trend[rows] if trended else None,
            domains=labels,
        )
        on |= {"trend": at} if trended else {}
        return tb.simulate(targets, n=3, seed=2, keep=True, **on).realizations

    for trended in (False, True):
        alone = run(hard, ms, trended=trended)
        np.testing.assert_array_equal(run(hard, everything, zone, trended, domains="MS"), alone)
        np.testing.assert_array_equal(run(hard, ms, "MS", trended, domains="MS"), alone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")
    soft = run(bt.Search(40.0, max_samples=12, soft=20.0), everything, zone, True, domains=labels)
    hard = run(bt.Search(40.0, max_samples=12), everything, zone, True, domains=labels)
    assert np.isfinite(soft).all() and not np.array_equal(soft, hard)


def test_each_turning_bands_domain_keeps_its_own_declustered_mean():
    xyz, grades, _, weights, zone, _, _ = zoned_holes()
    far = np.column_stack([25.0 * np.arange(40), np.full(40, 150.0), np.zeros(40)])
    labels = np.repeat(["MS", "SM"], 20)
    tb = banded(bt.Search(40.0, max_samples=8)).fit(xyz, grades, weights=weights, domains=zone)
    mean = tb.simulate(far, n=200, seed=3, domains=labels).mean
    for name in ("MS", "SM"):
        want = np.average(grades[zone == name], weights=weights[zone == name])
        assert mean[labels == name].mean() == pytest.approx(want, rel=0.1)


def test_turning_bands_domains_errors_and_persistence(tmp_path):
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    soft = bt.Search(30.0, max_samples=12, soft={("MS", "SM"): 8.0})
    tb = banded(soft).fit(xyz, grades, weights=weights, holes=holes, domains=zone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")

    def run(model):
        return model.simulate(targets, n=2, seed=5, keep=True, domains=labels).realizations

    tb.to_parquet(tmp_path / "tb.parquet")
    for again in (bt.TurningBands.from_parquet(tmp_path / "tb.parquet"), pickle.loads(pickle.dumps(tb))):
        np.testing.assert_array_equal(run(again), run(tb))
    with pytest.raises(bt.InvalidInput, match="simulate needs domains"):
        tb.simulate(targets, n=1)
    with pytest.raises(bt.InvalidInput, match="has no samples"):
        tb.simulate(targets, n=1, domains="QE")
    with pytest.raises(bt.InvalidInput, match="takes none"):
        banded().fit(xyz, grades).simulate(targets, n=1, domains="MS")
    with pytest.raises(bt.InvalidInput, match="needs domains at fit"):
        banded(soft).fit(xyz, grades)
    with pytest.raises(bt.InvalidInput, match="has no samples"):
        banded(soft).fit(xyz, grades, domains="MS")
    with pytest.raises(bt.InvalidInput, match="does not take domains"):
        bt.MultivariateSimulation(bt.PCA(), [banded(soft)])


@pytest.mark.parametrize(
    "model",
    [
        bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)),
        bt.TurningBands(gaussian, bands=60, search=bt.Search(radius=40, max_samples=12)),
    ],
)
def test_grades_follow_each_realization_of_simulated_domains(model):
    zone = np.where(coords[:, 0] < 50, "lean", "rich")
    grades = np.where(zone == "lean", values, 100 * values)
    model.fit(coords, grades, domains=zone)
    nodes = grid.centroids
    fixed = np.where(nodes[:, 0] < 50, "lean", "rich")

    def run(domains):
        path = {"path": "random"} if isinstance(model, bt.SGS) else {}
        return model.simulate(grid, n=3, seed=4, keep=True, domains=domains, **path).realizations

    np.testing.assert_array_equal(run(np.tile(fixed, (3, 1))), run(fixed))
    simulated = np.array([np.where(nodes[:, 0] < edge, "lean", "rich") for edge in (20, 50, 80)])
    reals = run(simulated)
    split = (values.max() + 100 * values.min()) / 2
    for real, domains in zip(reals, simulated):
        assert (real[domains == "lean"] < split).all() and (real[domains == "rich"] > split).all()
    with pytest.raises(bt.InvalidInput, match="expected 3 realizations of domains, got 2"):
        run(simulated[:2])


@pytest.mark.parametrize("make", [lambda s: bt.SGS(gaussian, s), banded], ids=["SGS", "TurningBands"])
def test_simulators_take_column_names_and_domain_column(make):
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    soft = bt.Search(30.0, max_samples=12, soft=8.0)
    trend, at = xyz[:, 1] / 100, targets[:, 1] / 100
    labels = np.where(targets[:, 0] < 40, "MS", "SM")
    samples = bt.PointSet(xyz, {"zn": grades, "w": weights, "hole": holes, "t": trend, "zone": zone})
    nodes = bt.PointSet(targets, {"t": at, "zone": labels})
    arrays = make(soft).fit(xyz, grades, weights=weights, holes=holes, trend=trend, domains=zone)
    names = make(soft).fit(samples, "zn", weights="w", holes="hole", trend="t", domain_column="zone")
    want = arrays.simulate(targets, n=2, seed=5, keep=True, trend=at, domains=labels).realizations
    for model in (arrays, names):
        got = model.simulate(nodes, n=2, seed=5, keep=True, trend="t", domain_column="zone")
        np.testing.assert_array_equal(got.realizations, want)
    if isinstance(names, bt.SGS):
        passes = names.passes(nodes, domain_column="zone")
        np.testing.assert_array_equal(passes, arrays.passes(targets, domains=labels))
    with pytest.raises(bt.InvalidInput, match="one of domains or domain_column"):
        names.simulate(nodes, n=1, trend="t", domains=labels, domain_column="zone")
    with pytest.raises(bt.InvalidInput, match="one of domains or domain_column"):
        make(soft).fit(samples, "zn", domains=zone, domain_column="zone")
    with pytest.raises(bt.InvalidInput, match="simulate needs domains"):
        names.simulate(nodes, n=1, trend="t")
    with pytest.raises(bt.MissingColumn, match="columns: t, zone"):
        names.simulate(nodes, n=1, trend="t", domain_column="rock")
    with pytest.raises(TypeError):
        make(soft).fit(samples, "zn", "w")
    with pytest.raises(TypeError):
        names.simulate(nodes, 1)


def test_categorical_and_multivariate_simulators_take_column_names():
    rock = (values > 1).astype(int)
    samples = bt.PointSet(coords, {"rock": rock, "v": values, "root": values**0.5, "w": np.ones(60)})
    near = bt.Search(radius=40, max_samples=12)
    for model in (bt.SIS([gaussian] * 2, near), bt.Plurigaussian(gaussian, proportions=[0.5, 0.5])):
        want = model.fit(coords, rock).simulate(grid, n=2, seed=1, keep=True).realizations
        got = model.fit(samples, "rock").simulate(grid, n=2, seed=1, keep=True).realizations
        np.testing.assert_array_equal(got, want)
        with pytest.raises(TypeError):
            model.fit(coords, rock, None)
    sgs = bt.SGS(gaussian, near)
    mv = bt.MultivariateSimulation(bt.PCA(), [sgs, sgs])
    want = mv.fit(coords, np.column_stack([values, values**0.5]), weights=np.ones(60)).simulate(grid, n=2)
    got = mv.fit(samples, ["v", "root"], weights="w").simulate(grid, n=2)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a.mean, b.mean)
    with pytest.raises(bt.MissingColumn):
        mv.fit(samples, ["v", "missing"])


def test_cdf_bands_contain_the_data_cdf_of_the_distribution_drawn_from():
    points = bt.PointSet(coords, {"v": values, "w": rng.uniform(0.5, 2.0, 60)})
    p = points["w"] / points["w"].sum()
    drawn = rng.choice(values, size=(50, 400), p=p)
    check = bt.check_realizations(grid, drawn, points, "v", weights="w")
    inner = (check.probabilities >= 0.05) & (check.probabilities <= 0.95)
    for q, target in [
        (check.quantiles[0], check.data_quantiles[0]),
        (check.score_quantiles[0], bt.normal_ppf(check.probabilities)),
    ]:
        assert np.all((q.min(axis=0) <= target)[inner] & (target <= q.max(axis=0))[inner])
    shifted = bt.check_realizations(grid, drawn * 1.5, points, "v", weights="w").quantiles[0]
    assert np.mean(shifted.min(axis=0)[inner] > check.data_quantiles[0][inner]) > 0.5
    stats = check.statistics
    assert stats.num_rows == 51 and list(stats["realization"][:2]) == [0, 1]
    np.testing.assert_allclose(stats["mean"][0], np.average(values, weights=p))
    np.testing.assert_allclose(stats["mean"][1:], drawn.mean(axis=1))


def test_unconditional_sgs_reproduces_the_model_variogram_at_short_lags():
    model = bt.Variogram([("spherical", 1.0, 15.0)])
    data = bt.PointSet(rng.uniform(1000, 2000, (1000, 2)), {"v": rng.normal(size=1000)})
    nodes = bt.BlockModel((0.5, 0.5), (1.0, 1.0), (60, 60))
    sgs = bt.SGS(model, bt.Search(radius=30, max_samples=16)).fit(data, "v")
    reals = sgs.simulate(nodes, n=20, seed=4, keep=True)
    check = bt.check_realizations(nodes, reals, data, "v", variogram=model, lag=1.0, max_lag=8.0)
    assert check.directions == [(0.0, 0.0), (90.0, 0.0)]
    for d in range(2):
        mean = np.mean([r[d].gammas for r in check.variograms[0]], axis=0)
        np.testing.assert_allclose(mean, model.gamma(check.variograms[0][0][d].lags), atol=0.1)


def test_multivariate_and_categorical_checks():
    nodes = bt.BlockModel((0.5, 0.5), (1.0, 1.0), (20, 20))
    a = rng.normal(size=(8, 400))
    b = 0.8 * a + 0.6 * rng.normal(size=(8, 400))
    x = rng.normal(size=60)
    points = bt.PointSet(coords, {"a": x, "b": 0.8 * x + 0.6 * rng.normal(size=60)})
    check = bt.check_realizations(nodes, [a, b], points, ["a", "b"], lag=2.0, max_lag=8.0)
    assert check.names == ["a", "b"] and check.correlations.shape == (8, 2, 2)
    np.testing.assert_allclose(check.correlations[3, 0, 1], np.corrcoef(a[3], b[3])[0, 1])
    np.testing.assert_allclose(check.data_correlation, bt.correlation(points, columns=["a", "b"]))
    assert len(check.variograms[1]) == 8 and check.directions is None
    codes = (values > 1).astype(int) + (values > 2)
    reals = rng.integers(0, 3, (6, 400))
    cats = bt.check_realizations(nodes, reals, coords, codes, lag=2.0, max_lag=8.0)
    assert cats.categorical and cats.quantiles is None and cats.proportions.shape == (6, 3)
    np.testing.assert_allclose(cats.data_proportions, np.bincount(codes, minlength=3) / 60)
    np.testing.assert_allclose(cats.proportions[0], np.bincount(reals[0], minlength=3) / 400)
    indicator = bt.experimental_variogram(nodes, (reals[2] == 1).astype(float), 2.0, 8.0)
    np.testing.assert_array_equal(cats.variograms[1][2][0].gammas, indicator.gammas)
    no_reals = bt.SGS(gaussian, bt.Search(40.0)).fit(coords, values).simulate(nodes, n=2)
    for args, options in [
        ([a, b], {}),
        (a, {"lag": 2.0}),
        (no_reals, {}),
    ]:
        with pytest.raises(ValueError):
            bt.check_realizations(nodes, args, points, "a", **options)


def _cosimulation_case():
    model = bt.Variogram([("spherical", 1.0, 10.0)])
    nodes = bt.BlockModel((0.5, 0.5), (1.0, 1.0), (40, 40))
    far = bt.PointSet(rng.uniform(1000, 2000, (500, 2)), {"v": rng.normal(size=500)})
    field = bt.SGS(model, bt.Search(radius=30, max_samples=16)).fit(far, "v")
    secondary = field.simulate(nodes, n=2, seed=9, keep=True, path="random").realizations
    rows = rng.choice(1600, 100, replace=False)
    s = secondary[0]
    scores = (s - s.mean()) / s.std()
    grade = np.exp(0.7 * scores[rows] + np.sqrt(0.51) * rng.normal(size=100))
    points = bt.PointSet(nodes.centroids[rows], {"v": grade, "s": s[rows]})
    return model, nodes, secondary, rows, points


def test_collocated_cosimulation_reproduces_correlation_histogram_and_variogram():
    model, nodes, secondary, rows, points = _cosimulation_case()
    s = secondary[0]
    search = bt.Search(radius=30, max_samples=16)
    sgs = bt.SGS(model, search).fit(points, "v", secondary="s")
    assert sgs.correlation == pytest.approx(0.7, abs=0.15)
    reals = sgs.simulate(nodes, n=10, seed=1, secondary=s, keep=True).realizations
    np.testing.assert_array_equal(reals[:, rows], np.tile(points["v"], (10, 1)))
    logs = bt.PointSet(points.coords, {"v": np.log(points["v"]), "s": points["s"]})
    check = bt.check_realizations(
        nodes,
        [np.log(reals), np.tile(s, (10, 1))],
        logs,
        ["v", "s"],
        variogram=[model, model],
        lag=1.0,
        max_lag=8.0,
    )
    assert check.correlations[:, 0, 1].mean() == pytest.approx(sgs.correlation, abs=0.15)
    inner = (check.probabilities >= 0.1) & (check.probabilities <= 0.9)
    median = np.median(check.score_quantiles[0], axis=0)
    np.testing.assert_allclose(median[inner], bt.normal_ppf(check.probabilities[inner]), atol=0.3)
    for d in range(2):
        mean = np.mean([r[d].gammas for r in check.variograms[0]], axis=0)
        np.testing.assert_allclose(mean, model.gamma(check.variograms[0][0][d].lags), atol=0.15)

    independent = bt.SGS(model, search).fit(points, "v").simulate(nodes, n=3, seed=1, keep=True)
    zero = bt.SGS(model, search).fit(points, "v", secondary="s", correlation=0.0)
    assert zero.correlation == 0.0
    same = zero.simulate(nodes, n=3, seed=1, secondary=s, keep=True).realizations
    np.testing.assert_array_equal(same, independent.realizations)


def test_cosimulation_takes_a_secondary_realization_per_realization(tmp_path):
    model, nodes, secondary, _, points = _cosimulation_case()
    sgs = bt.SGS(model, bt.Search(radius=30, max_samples=16)).fit(points, "v", secondary="s", correlation=0.8)
    both = sgs.simulate(nodes, n=2, seed=4, secondary=secondary, keep=True).realizations
    second = sgs.simulate(nodes, n=2, seed=4, secondary=secondary[[1, 1]], keep=True).realizations
    np.testing.assert_array_equal(both[1], second[1])
    assert not np.array_equal(both[0], second[0])
    sgs.to_parquet(tmp_path / "cosgs.parquet")
    loaded = bt.SGS.from_parquet(tmp_path / "cosgs.parquet")
    assert loaded.correlation == 0.8
    again = loaded.simulate(nodes, n=2, seed=4, secondary=secondary, keep=True).realizations
    np.testing.assert_array_equal(again, both)
    plain = bt.SGS(model, bt.Search(radius=30)).fit(points, "v")
    for call in [
        lambda: sgs.simulate(nodes, n=2),
        lambda: sgs.simulate(nodes, n=3, secondary=secondary),
        lambda: sgs.simulate(nodes, n=1, secondary=secondary[0][:10]),
        lambda: plain.simulate(nodes, n=1, secondary=secondary[0]),
        lambda: bt.SGS(model, bt.Search(radius=30)).fit(points, "v", correlation=0.5),
        lambda: bt.SGS(model, bt.Search(radius=30)).fit(points, "v", secondary="s", correlation=1.5),
    ]:
        with pytest.raises(ValueError):
            call()


def test_collocated_dss_follows_the_secondary(tmp_path):
    model, nodes, secondary, rows, points = _cosimulation_case()
    s = secondary[0]
    search = bt.Search(radius=30, max_samples=16)
    dss = bt.DSS(model, search).fit(points, "v", secondary="s")
    assert dss.correlation == pytest.approx(np.corrcoef(points["v"], points["s"])[0, 1])
    reals = dss.simulate(nodes, n=10, seed=1, secondary=s, keep=True).realizations
    np.testing.assert_array_equal(reals[:, rows], np.tile(points["v"], (10, 1)))
    got = np.mean([np.corrcoef(r, s)[0, 1] for r in reals])
    assert got == pytest.approx(dss.correlation, abs=0.15)

    plain = bt.DSS(model, search).fit(points, "v").simulate(nodes, n=2, seed=1, keep=True)
    zero = bt.DSS(model, search).fit(points, "v", secondary="s", correlation=0.0)
    assert zero.correlation == 0.0
    same = zero.simulate(nodes, n=2, seed=1, secondary=secondary, keep=True).realizations
    np.testing.assert_array_equal(same, plain.realizations)

    both = dss.simulate(nodes, n=2, seed=4, secondary=secondary, keep=True).realizations
    second = dss.simulate(nodes, n=2, seed=4, secondary=secondary[[1, 1]], keep=True).realizations
    np.testing.assert_array_equal(both[1], second[1])
    dss.to_parquet(tmp_path / "codss.parquet")
    loaded = bt.DSS.from_parquet(tmp_path / "codss.parquet")
    assert loaded.correlation == dss.correlation
    again = loaded.simulate(nodes, n=2, seed=4, secondary=secondary, keep=True).realizations
    np.testing.assert_array_equal(again, both)
    for call in [
        lambda: dss.simulate(nodes, n=2),
        lambda: dss.simulate(nodes, n=3, secondary=secondary),
        lambda: bt.DSS(model, search).fit(points, "v").simulate(nodes, n=1, secondary=s),
        lambda: bt.DSS(model, search).fit(points, "v", correlation=0.5),
        lambda: bt.DSS(model, search).fit(points, "v", secondary="s", correlation=1.5),
    ]:
        with pytest.raises(ValueError):
            call()


def test_correct_distribution_matches_the_target_and_keeps_ranks():
    rng = np.random.default_rng(11)
    reals = rng.normal(size=(4, 250))
    target = rng.lognormal(0.0, 0.7, 250)
    out = bt.correct_distribution(reals, target)
    np.testing.assert_allclose(np.sort(out, axis=1), np.tile(np.sort(target), (4, 1)), atol=1e-9)
    np.testing.assert_array_equal(np.argsort(out, axis=1), np.argsort(reals, axis=1))
    np.testing.assert_array_equal(bt.correct_distribution(reals, target, strength=0.0), reals)
    half = bt.correct_distribution(reals, target, strength=0.5)
    np.testing.assert_allclose(half, 0.5 * (reals + out))
    only = bt.correct_distribution(reals, target, realizations=[2])
    np.testing.assert_array_equal(only[[0, 1, 3]], reals[[0, 1, 3]])
    np.testing.assert_array_equal(only[2], out[2])
    one = bt.correct_distribution(reals[0], target)
    assert one.shape == (250,)
    np.testing.assert_array_equal(one, out[0])
    kde = bt.KernelDensity(lower=0.0).fit(target)
    smooth = bt.correct_distribution(reals, kde)
    np.testing.assert_allclose(np.median(smooth, axis=1), kde.quantile([0.5])[0], rtol=0.02)
    with pytest.raises(ValueError):
        bt.correct_distribution(reals, target, strength=2.0)
    with pytest.raises(ValueError):
        bt.correct_distribution(reals, target, realizations=[9])


def test_correct_distribution_takes_a_summary():
    model = bt.Variogram([("spherical", 1.0, 30.0)])
    rng = np.random.default_rng(2)
    points = bt.PointSet(rng.uniform(0, 100, (40, 2)), {"v": rng.lognormal(0.0, 0.5, 40)})
    nodes = bt.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(20, 20))
    sgs = bt.SGS(model, bt.Search(radius=40)).fit(points, "v")
    summary = sgs.simulate(nodes, n=3, seed=1, keep=True)
    out = bt.correct_distribution(summary, points["v"])
    assert out.shape == summary.realizations.shape
    np.testing.assert_allclose(out.mean(axis=1), points["v"].mean(), rtol=0.01)
    with pytest.raises(ValueError):
        bt.correct_distribution(sgs.simulate(nodes, n=2), points["v"])


def test_keep_selects_realizations():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    everything = sgs.simulate(grid, n=4, seed=7, keep=True)
    some = sgs.simulate(grid, n=4, seed=7, keep=[3, 1])
    assert everything.kept == [0, 1, 2, 3]
    assert some.kept == [1, 3]
    np.testing.assert_array_equal(some.realizations, everything.realizations[[1, 3]])
    np.testing.assert_array_equal(some.mean, everything.mean)
    none = sgs.simulate(grid, n=4, seed=7)
    assert none.kept == [] and none.realizations is None


def test_keep_rejects_bad_indices():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    for keep in ([4], [1, 1], [-1], "all"):
        with pytest.raises(bt.InvalidInput):
            sgs.simulate(grid, n=4, seed=7, keep=keep)


def test_kept_realizations_survive_parquet(tmp_path):
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    summary = sgs.simulate(grid, n=4, seed=7, keep=[2])
    summary.to_parquet(tmp_path / "s.parquet")
    back = bt.SimulationSummary.from_parquet(tmp_path / "s.parquet")
    assert back.kept == [2]
    np.testing.assert_array_equal(back.realizations, summary.realizations)


def _shared_case():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    return sgs, bt.BlockModel(origin=(0, 0), size=(5, 5), count=(20, 20))


def test_shared_realizations_do_not_depend_on_the_batch():
    sgs, blocks = _shared_case()
    one = sgs.simulate(blocks, n=5, seed=2, keep=True, path="shared", batch=1)
    together = sgs.simulate(blocks, n=5, seed=2, keep=True, path="shared", batch=5)
    np.testing.assert_array_equal(one.realizations, together.realizations)
    np.testing.assert_array_equal(one.mean, together.mean)


def test_block_models_default_to_the_shared_path():
    sgs, blocks = _shared_case()
    default = sgs.simulate(blocks, n=3, seed=1, keep=True)
    shared = sgs.simulate(blocks, n=3, seed=1, keep=True, path="shared")
    random = sgs.simulate(blocks, n=3, seed=1, keep=True, path="random")
    np.testing.assert_array_equal(default.realizations, shared.realizations)
    assert not np.array_equal(default.realizations, random.realizations)


def test_points_default_to_the_random_path_and_refuse_a_shared_one():
    sgs, blocks = _shared_case()
    points = np.asarray(blocks.centroids)[:, :2]
    default = sgs.simulate(points, n=2, seed=1, keep=True)
    random = sgs.simulate(points, n=2, seed=1, keep=True, path="random")
    np.testing.assert_array_equal(default.realizations, random.realizations)
    with pytest.raises(bt.InvalidInput, match="path='shared' does not support"):
        sgs.simulate(points, n=2, path="shared")
    with pytest.raises(bt.InvalidInput, match="path must be"):
        sgs.simulate(blocks, n=2, path="spiral")


def test_octant_searches_fall_back_to_the_random_path():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12, octant=True)).fit(coords, values)
    blocks = bt.BlockModel(origin=(0, 0), size=(5, 5), count=(20, 20))
    default = sgs.simulate(blocks, n=2, seed=1, keep=True)
    random = sgs.simulate(blocks, n=2, seed=1, keep=True, path="random")
    np.testing.assert_array_equal(default.realizations, random.realizations)
    with pytest.raises(bt.InvalidInput, match="octant"):
        sgs.simulate(blocks, n=2, path="shared")


def test_shared_path_honors_data_and_supports_masked_models_and_blocks():
    sgs, blocks = _shared_case()
    s = sgs.simulate(blocks, n=4, seed=3, keep=True, path="shared")
    assert s.realizations.shape == (4, 400)
    assert np.isfinite(s.realizations).all()
    coarse = bt.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    b = sgs.simulate(blocks, n=3, seed=3, blocks=coarse, path="shared")
    assert b.mean.shape == (25,)


def test_available_memory_is_a_positive_size():
    from boitata import _memory

    assert _memory.available() > 2**20


def test_progress_bar_shows_only_when_asked_and_changes_nothing(capsys):
    sgs, blocks = _shared_case()
    on = sgs.simulate(blocks, n=4, seed=3, keep=True, progress=True)
    assert "100%" in capsys.readouterr().err
    off = sgs.simulate(blocks, n=4, seed=3, keep=True, progress=False)
    assert capsys.readouterr().err == ""
    np.testing.assert_array_equal(on.realizations, off.realizations)
    on = sgs.simulate(coords[:20], n=3, seed=3, keep=True, progress=True)
    off = sgs.simulate(coords[:20], n=3, seed=3, keep=True, progress=False)
    np.testing.assert_array_equal(on.realizations, off.realizations)
    capsys.readouterr()


def test_every_simulator_accepts_progress():
    tb = bt.TurningBands(gaussian, bands=50, step=1.0).fit(coords, values)
    np.testing.assert_array_equal(
        tb.simulate(grid, n=3, seed=1, keep=True, progress=True).realizations,
        tb.simulate(grid, n=3, seed=1, keep=True, progress=False).realizations,
    )
    cats = (values > np.median(values)).astype(int)
    sis = bt.SIS([gaussian, gaussian], bt.Search(radius=40, max_samples=12)).fit(coords, cats)
    np.testing.assert_array_equal(
        sis.simulate(grid, n=3, seed=1, keep=True, progress=True).realizations,
        sis.simulate(grid, n=3, seed=1, keep=True, progress=False).realizations,
    )


CHANNELS = {"shape": "channel", "code": 1, "proportion": 0.25, "width": 8.0, "azimuth": (-10, 10)}
LOBES = {"shape": "ellipsoid", "code": 2, "proportion": 0.1, "radii": (6.0, 3.0)}


def test_object_training_image_is_seeded_and_hits_its_proportions():
    tpl = bt.BlockModel(origin=(0, 0), size=(1, 1), count=(80, 60), crs="EPSG:32722")
    ti = bt.object_training_image(tpl, [CHANNELS, LOBES], background=0, seed=4)
    assert ti.count == [80, 60, 1] and ti.crs == "EPSG:32722"
    codes = ti["facies"]
    assert set(np.unique(codes)) <= {0, 1, 2}
    assert np.mean(codes == 2) >= 0.1
    assert np.mean(codes >= 1) >= 0.25
    np.testing.assert_array_equal(codes, bt.object_training_image(tpl, [CHANNELS, LOBES], seed=4)["facies"])
    assert not np.array_equal(codes, bt.object_training_image(tpl, [CHANNELS, LOBES], seed=5)["facies"])


def test_object_training_image_rejects_bad_sets():
    tpl = bt.BlockModel(origin=(0, 0, 0), size=(1, 1, 1), count=(20, 20, 5))
    for bad in [
        {**CHANNELS, "shape": "blob"},
        {**CHANNELS, "length": 3},
        {**CHANNELS, "width": (5, 2)},
        {**CHANNELS, "proportion": 1.5},
        {**CHANNELS, "amplitude": 3},
        CHANNELS,
        {**LOBES, "radii": (6.0, 3.0)},
    ]:
        with pytest.raises(bt.InvalidInput):
            bt.object_training_image(tpl, [bad])


def test_select_realizations_picks_one_medoid_per_cluster():
    rng = np.random.default_rng(2)
    centers = np.repeat([0.0, 5.0, 20.0], [8, 5, 3])
    reals = centers[:, None] + rng.normal(scale=0.1, size=(16, 30))
    reals[4, 7] = np.nan
    picked = bt.select_realizations(reals, 3, seed=4)
    assert picked.dtype == np.int64
    np.testing.assert_array_equal(centers[picked], [0.0, 5.0, 20.0])
    np.testing.assert_array_equal(picked, bt.select_realizations(reals, 3, seed=4))
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    s = sgs.simulate(grid, n=6, seed=1, keep=True)
    np.testing.assert_array_equal(bt.select_realizations(s, 2), bt.select_realizations(s.realizations, 2))
    for call in [
        lambda: bt.select_realizations(reals, 0),
        lambda: bt.select_realizations(reals, 17),
        lambda: bt.select_realizations(sgs.simulate(grid, n=2, seed=1), 1),
    ]:
        with pytest.raises(ValueError):
            call()


def _study(**options):
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    options = {
        "truths": 3,
        "n": 20,
        "window": (20, 20),
        "composite_length": np.inf,
        "progress": False,
        **options,
    }
    return bt.spacing_study(sgs, grid, **options)


def test_spacing_study_mee_grows_with_spacing():
    study = _study(spacings=[10, 20, 40])
    assert study.column_names == [
        "plan",
        "spacing",
        "realization",
        "row",
        "truth",
        "mean",
        "mee",
        "cv",
        "precision_0.15",
        "error",
        "covered",
    ]
    plan, mee = np.asarray(study["plan"]), study["mee"]
    medians = [np.median(mee[plan == p]) for p in ("10", "20", "40")]
    assert medians == sorted(medians) and medians[0] < 0.5 * medians[-1]
    np.testing.assert_array_equal(np.unique(study["spacing"]), [10, 20, 40])
    assert study["covered"].dtype == bool and study.num_rows == 3 * 3 * 400


def test_spacing_study_plans_equal_spacings_and_truth_indices():
    by_spacing = _study(spacings=[(10, 20)], truths=[4, 1])
    plan = bt.planned_drillholes(grid, (10, 20))
    by_plan = _study(plans={"grid": plan}, truths=[4, 1])
    for name in by_spacing.column_names[2:]:
        np.testing.assert_array_equal(by_spacing[name], by_plan[name], err_msg=name)
    assert set(by_plan["plan"]) == {"grid"} and np.isnan(by_plan["spacing"]).all()
    assert set(by_spacing["plan"]) == {"10x20"} and by_spacing["spacing"][0] == pytest.approx(np.sqrt(200))
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12)).fit(coords, values)
    truth = sgs.simulate(grid, n=5, seed=0, keep=[4], window=(20, 20), progress=False).realizations[0]
    np.testing.assert_array_equal(by_plan["truth"][:400], truth)
    np.testing.assert_array_equal(by_plan["realization"], np.repeat([4.0, 1.0], 400))


def test_spacing_study_groups_existing_and_sampling_error():
    period = np.arange(400) % 3
    study = _study(spacings=[20], window=None, groups=period, existing=True, sampling_error=0.1)
    np.testing.assert_array_equal(study["row"], np.tile([0, 1, 2], 3))
    exact = _study(spacings=[20], window=None, groups=period, existing=True)
    assert not np.array_equal(study["mean"], exact["mean"])
    np.testing.assert_array_equal(study["truth"], exact["truth"])


def test_spacing_study_rejects_bad_input():
    sgs = bt.SGS(gaussian, bt.Search(radius=40, max_samples=12))
    plan = bt.planned_drillholes(grid, 20.0)
    for call, match in [
        (lambda: _study(), "one of spacings or plans"),
        (lambda: _study(spacings=[10], plans={"a": plan}), "one of spacings or plans"),
        (lambda: _study(plans={"a": plan}, dip=60.0), "go with spacings"),
        (lambda: _study(spacings=[10], quantiles=(0.1, 0.95)), "symmetric"),
        (lambda: _study(spacings=[10], truths=[1, 1]), "distinct"),
        (lambda: _study(spacings=[10], composite_length=None), "composite_length"),
        (lambda: bt.spacing_study(sgs, grid, spacings=[10]), "not fitted"),
        (lambda: bt.spacing_study(sgs.fit(coords, values), grid.centroids, spacings=[10]), "BlockModel"),
    ]:
        with pytest.raises(bt.InvalidInput, match=match):
            call()


def test_spacing_study_ignores_the_thread_count(tmp_path):
    here = _study(spacings=[10, 25], sampling_error=0.05)
    script = (
        "import sys, numpy as np; sys.path.insert(0, sys.argv[1]); import test_simulation as t; "
        "s = t._study(spacings=[10, 25], sampling_error=0.05); "
        "np.savez(sys.argv[2], **{c: np.asarray(s[c], dtype=float) for c in s.column_names[1:]})"
    )
    out = tmp_path / "one.npz"
    env = {**os.environ, "RAYON_NUM_THREADS": "1"}
    subprocess.run(
        [sys.executable, "-c", script, str(pathlib.Path(__file__).parent), str(out)], env=env, check=True
    )
    with np.load(out) as one:
        for name in here.column_names[1:]:
            np.testing.assert_array_equal(np.asarray(here[name], dtype=float), one[name], err_msg=name)
