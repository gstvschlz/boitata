import json
import pickle

import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(5)
coords = rng.uniform(0, 100, (60, 2))
values = rng.lognormal(0, 0.6, 60)
gaussian = cs.Variogram([("spherical", 1.0, 30.0)])
grid = cs.BlockModel(origin=(0, 0), size=(5, 5), count=(20, 20))


def test_sgs_is_reproducible_and_honors_data():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    a = sgs.simulate(grid, n=3, seed=7, keep=True).realizations
    assert a.shape == (3, 400)
    np.testing.assert_array_equal(a, sgs.simulate(grid, n=3, seed=7, keep=True).realizations)
    assert not np.array_equal(a[0], a[1])
    at_data = sgs.simulate(coords[:5], n=4, seed=1)
    np.testing.assert_allclose(at_data.mean, values[:5])
    np.testing.assert_allclose(at_data.variance, 0, atol=1e-12)


def test_sgs_summary_matches_its_realizations():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
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


def test_block_support_averages_each_realization():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    nodes = sgs.simulate(grid, n=6, seed=2, keep=True).realizations
    s = sgs.simulate(grid, n=6, seed=2, cutoffs=[1.0], keep=True, blocks=blocks)
    xy = grid.centroids[:, :2] // 20
    rows = (xy[:, 0] + 5 * xy[:, 1]).astype(int)
    expected = np.stack([np.bincount(rows, r) / np.bincount(rows) for r in nodes])
    np.testing.assert_allclose(s.realizations, expected)
    np.testing.assert_allclose(s.mean, expected.mean(axis=0))
    assert (s.variance <= np.bincount(rows, nodes.var(axis=0)) / 16 + 1e-9).all()
    with pytest.raises(ValueError):
        sgs.simulate(grid, n=1, blocks=cs.BlockModel(origin=(0, 0), size=(20, 20), count=(6, 5)))

    cats = (values > np.median(values)).astype(int)
    sis = cs.SIS([gaussian, gaussian], cs.Search(radius=40, max_samples=12)).fit(coords, cats)
    c = sis.simulate(grid, n=3, seed=4, keep=True, blocks=blocks)
    assert c.realizations.shape == (3, 25) and c.probabilities.shape == (25, 2)


def test_localize_realizations_within_panels():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    panels = cs.BlockModel(origin=(0, 0), size=(50, 50), count=(2, 2))
    smus = panels.discretize(5)
    s = sgs.simulate(grid, n=8, seed=2, keep=True, blocks=smus)
    smus = smus.with_column("etype", s.mean)
    out = cs.localize(smus, "etype", panels, s.realizations)
    owner, local = smus["block"].astype(int), out["localized"]
    for p in range(4):
        mine = owner == p
        pooled = np.sort(s.realizations[:, mine].ravel())
        assert local[mine].mean() == pytest.approx(pooled.mean(), rel=1e-9)
        by_rank = local[mine][np.argsort(s.mean[mine], kind="stable")]
        np.testing.assert_allclose(by_rank, pooled.reshape(25, 8).mean(axis=1), rtol=1e-12)

    one = cs.localize(smus, "etype", panels, s.realizations[:1], name="one")["one"]
    for p in range(4):
        mine = owner == p
        by_rank = one[mine][np.argsort(s.mean[mine], kind="stable")]
        np.testing.assert_array_equal(by_rank, np.sort(s.realizations[0, mine]))
    with pytest.raises(cs.InvalidInput):
        cs.localize(smus, "etype", panels, s.realizations[:, 1:])
    with pytest.raises(KeyError):
        cs.localize(smus, "missing", panels, s.realizations)


def test_turning_bands_summary():
    tb = cs.TurningBands(gaussian, bands=100, step=1.0).fit(coords, values)
    s = tb.simulate(grid, n=2, seed=1, cutoffs=[1.0, 2.0])
    assert s.mean.shape == (400,) and s.probability_above.shape == (400, 2)


def test_sis_probabilities():
    cats = (values > np.median(values)).astype(int)
    sis = cs.SIS([gaussian, gaussian], cs.Search(radius=40, max_samples=12)).fit(coords, cats)
    s = sis.simulate(grid, n=4, seed=4, keep=True)
    assert s.probabilities.shape == (400, 2) and s.proportions.shape == (4, 2)
    np.testing.assert_allclose(s.probabilities.sum(axis=1), 1.0)
    assert set(np.unique(s.realizations)) <= {0, 1}
    assert ((s.entropy >= 0) & (s.entropy <= 1)).all()


def test_sis_follows_local_proportions():
    far = np.array([[-1e3, 0.0], [-1e3, 5.0]])
    line = np.c_[np.arange(200) + 0.5, np.zeros(200)]
    east = line[:, 0] / 200
    local = cs.Table({"west": 1 - east, "east": east})
    sis = cs.SIS([gaussian, gaussian], cs.Search(radius=30, max_samples=12))
    sis.fit(far, [0, 1], proportions=[[0.5, 0.5]] * 2)
    s = sis.simulate(line, n=60, seed=1, proportions=local)
    by_half = s.probabilities[:, 1].reshape(2, 100).mean(axis=1)
    np.testing.assert_allclose(by_half, [0.25, 0.75], atol=0.08)
    again = sis.simulate(line, n=60, seed=1, proportions=np.c_[1 - east, east])
    np.testing.assert_array_equal(again.probabilities, s.probabilities)
    with pytest.raises(cs.InvalidInput, match="both fit and simulate"):
        sis.simulate(line, n=1)
    with pytest.raises(cs.InvalidInput, match="shape"):
        sis.fit(far, [0, 1], proportions=[[1.0, 0.0, 0.0]] * 2)


def test_plurigaussian_proportions():
    facies = rng.choice(3, 60, p=[0.2, 0.3, 0.5])
    pgs = cs.Plurigaussian(gaussian, proportions=[0.2, 0.3, 0.5]).fit(coords, facies)
    s = pgs.simulate(grid, n=2, seed=2)
    assert s.probabilities.shape == (400, 3) and set(np.unique(s.most_likely)) <= {0, 1, 2}


def test_plurigaussian_hierarchy_honors_data_on_three_fields():
    facies = rng.choice(4, 60, p=[0.4, 0.3, 0.2, 0.1])
    rule = (0, [0, (1, [1, (2, [2, 3])])])
    pgs = cs.Plurigaussian([gaussian] * 3, proportions=[0.4, 0.3, 0.2, 0.1], rule=rule).fit(coords, facies)
    s = pgs.simulate(coords, n=2, seed=4, keep=True)
    np.testing.assert_array_equal(s.realizations, [facies, facies])
    with pytest.raises(ValueError, match="splits field 2"):
        cs.Plurigaussian([gaussian] * 2, proportions=[0.4, 0.3, 0.2, 0.1], rule=rule)
    with pytest.raises(ValueError, match="every facies"):
        cs.Plurigaussian(gaussian, proportions=[0.5, 0.5], rule=(0, [0]))
    regions = [
        ([(-np.inf, 0.0)], 0),
        ([(0.0, np.inf), (-np.inf, 0.0)], 1),
        ([(0.0, np.inf), (0.0, np.inf)], 2),
    ]
    with pytest.raises(ValueError, match="thresholds 2 fields"):
        cs.Plurigaussian(gaussian, regions=regions)
    two = cs.Plurigaussian([gaussian] * 2, regions=regions).fit(coords, facies % 3)
    assert set(np.unique(two.simulate(grid, n=1, seed=1).most_likely)) <= {0, 1, 2}


def test_plurigaussian_fitted_variograms_recover_the_latent_ranges():
    rule = (0, [0, (1, [1, 2])])
    truth = [cs.Variogram([("spherical", 1.0, 16.0)]), cs.Variogram([("exponential", 1.0, 8.0)])]
    fine = cs.BlockModel(origin=(0, 0), size=(2, 2), count=(60, 60))
    xy = fine.centroids[:, :2]
    pgs = cs.Plurigaussian(truth, proportions=[0.3, 0.4, 0.3], rule=rule).fit([[60.0, 60.0]], [1])
    facies = pgs.simulate(fine, n=1, seed=5, keep=True).realizations[0]
    experimental = [cs.experimental_variogram(xy, (facies == f).astype(float), 2.0, 24.0) for f in range(3)]
    start = [cs.Variogram([("spherical", 1.0, 5.0)]), cs.Variogram([("exponential", 1.0, 80.0)])]
    fitted = cs.Plurigaussian(start, proportions=[0.3, 0.4, 0.3], rule=rule).fit_variograms(experimental)
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
    pgs = cs.Plurigaussian([gaussian] * 2, proportions=[0.5, 0.25, 0.25], rule=rule)
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
        cs.Plurigaussian(gaussian, regions=[([(-np.inf, np.inf)], 0)]).fit(xy, [0], proportions=[[1.0]])


def test_gibbs_respects_bounds():
    pts = rng.uniform(0, 50, (20, 2))
    bounds = np.column_stack([np.zeros(20), np.full(20, np.inf)])
    draw = cs.gibbs(pts, bounds, gaussian, seed=3)
    assert np.all(draw >= 0)


def trended_samples():
    r = np.random.default_rng(8)
    xy = r.uniform(0, 100, (400, 2))
    trend = xy[:, 0] / 100
    return xy, np.exp(1.5 * trend + 0.5 * r.normal(size=400)), trend


def test_simulation_with_a_trend_follows_it_and_honors_data():
    xy, z, trend = trended_samples()
    search = cs.Search(radius=30, max_samples=12)
    node_trend = grid.centroids[:, 0] / 100
    white = cs.Variogram([("spherical", 1.0, 4.0)])
    for simulator in (cs.SGS(white, search, classes=5), cs.TurningBands(white, bands=100, classes=5)):
        simulator.fit(xy, z, trend=trend)
        reals = simulator.simulate(grid, n=4, seed=2, keep=True, trend=node_trend).realizations
        want = np.corrcoef(np.log(z), trend)[0, 1]
        for r in reals:
            assert np.corrcoef(np.log(r), node_trend)[0, 1] == pytest.approx(want, abs=0.12)
        at_data = simulator.simulate(xy[:5], n=3, seed=1, trend=trend[:5])
        np.testing.assert_allclose(at_data.mean, z[:5])
        np.testing.assert_allclose(at_data.variance, 0, atol=1e-9)

    sgs = cs.SGS(gaussian, search).fit(xy, z, trend=trend)
    model = grid.with_column("drift", node_trend)
    by_array = sgs.simulate(grid, n=3, seed=4, keep=True, trend=node_trend).realizations
    np.testing.assert_array_equal(
        sgs.simulate(model, n=3, seed=4, keep=True, trend="drift").realizations, by_array
    )
    points = cs.PointSet(model.centroids, {"drift": node_trend})
    np.testing.assert_array_equal(
        sgs.simulate(points, n=3, seed=4, keep=True, trend="drift").realizations, by_array
    )
    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    by_block = sgs.simulate(grid, n=3, seed=4, keep=True, blocks=blocks, trend=node_trend).realizations
    xy_block = grid.centroids[:, :2] // 20
    rows = (xy_block[:, 0] + 5 * xy_block[:, 1]).astype(int)
    np.testing.assert_allclose(by_block, [np.bincount(rows, r) / np.bincount(rows) for r in by_array])

    flat = cs.SGS(gaussian, search).fit(xy, z, trend=np.ones(len(z)))
    plain = cs.SGS(gaussian, search).fit(xy, z)
    np.testing.assert_allclose(
        flat.simulate(grid, n=2, seed=3, trend=np.ones(400)).mean, plain.simulate(grid, n=2, seed=3).mean
    )

    with pytest.raises(cs.InvalidInput, match="give trend"):
        sgs.simulate(grid, n=1)
    with pytest.raises(cs.InvalidInput, match="needs trend at fit"):
        plain.simulate(grid, n=1, trend=node_trend)
    with pytest.raises(cs.InvalidInput, match="needs a container"):
        sgs.simulate(grid.centroids, n=1, trend="drift")
    with pytest.raises(cs.MissingColumn, match="drift"):
        sgs.simulate(grid, n=1, trend="drift")
    with pytest.raises(ValueError):
        sgs.simulate(grid, n=1, trend=node_trend[1:])
    with pytest.raises(ValueError):
        cs.SGS(gaussian, search).fit(xy, z, trend=trend[1:])
    with pytest.raises(cs.InvalidInput, match="give trend"):
        cs.TurningBands(gaussian).fit(xy, z, trend=trend).simulate_to_parquet("in.parquet", "out.parquet")


def test_simulators_with_a_trend_save_and_load_it(tmp_path):
    xy, z, trend = trended_samples()
    node_trend = grid.centroids[:, 0] / 100
    for simulator in (
        cs.SGS(gaussian, cs.Search(radius=30, max_samples=12), classes=4),
        cs.TurningBands(gaussian, bands=50, classes=4),
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
    search = cs.Search(radius=40, max_samples=12)
    mv = cs.MultivariateSimulation(
        cs.PPMT(seed=3), [cs.SGS(gaussian, search), cs.TurningBands(gaussian, bands=100, step=1.0)]
    ).fit(coords, data)
    a, b = mv.simulate(grid, n=10, seed=4, keep=True)
    assert a.realizations.shape == b.realizations.shape == (10, 400)
    r = np.mean([np.corrcoef(np.log(x), np.log(y))[0, 1] for x, y in zip(a.realizations, b.realizations)])
    assert r == pytest.approx(np.corrcoef(np.log(data.T))[0, 1], abs=0.15)
    np.testing.assert_array_equal(a.realizations, mv.simulate(grid, n=10, seed=4, keep=True)[0].realizations)

    at_data = mv.simulate(coords[:5], n=3, seed=1)
    for v, s in enumerate(at_data):
        np.testing.assert_allclose(s.mean, data[:5, v], rtol=1e-6)

    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    xy = grid.centroids[:, :2] // 20
    rows = (xy[:, 0] + 5 * xy[:, 1]).astype(int)
    by_block = mv.simulate(grid, n=10, seed=4, keep=True, blocks=blocks)
    for s, nodes in zip(by_block, (a, b)):
        expected = np.stack([np.bincount(rows, x) / np.bincount(rows) for x in nodes.realizations])
        np.testing.assert_allclose(s.realizations, expected)


def test_multivariate_simulation_drops_incomplete_samples_and_checks_inputs():
    data = np.column_stack([values, values**0.5])
    data[:4, 1] = np.nan
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12))
    with pytest.warns(UserWarning, match="4 samples miss a variable"):
        cs.MultivariateSimulation(cs.StepwiseConditional(), [sgs, sgs]).fit(coords, data)
    data[:, 1] *= rng.lognormal(0, 0.5, 60)
    data[4] = np.nan
    mv = cs.MultivariateSimulation(cs.PPMT(seed=3), [sgs, sgs])
    with pytest.warns(UserWarning, match="1 samples miss every variable"):
        mv.fit(coords, data, impute=True)
    at_data = mv.simulate(coords[:4], n=6, seed=2, keep=True)
    np.testing.assert_allclose(at_data[0].realizations, np.tile(values[:4], (6, 1)), rtol=1e-6)
    imputed = at_data[1].realizations
    assert np.isfinite(imputed).all() and np.ptp(imputed, axis=0).min() > 0
    with pytest.warns(UserWarning):
        mv.fit(coords, data, impute=cs.GaussianImputer(spatial=gaussian, neighbors=8))
    spatial = mv.simulate(coords[:4], n=6, seed=2, keep=True)
    assert np.isfinite(spatial[1].realizations).all()
    np.testing.assert_allclose(spatial[0].realizations, at_data[0].realizations)
    with pytest.raises(ValueError, match="impute must be"):
        mv.fit(coords, data, impute="yes")
    with pytest.raises(ValueError, match="transform must be"):
        cs.MultivariateSimulation(cs.NormalScore(), [sgs])
    with pytest.raises(ValueError, match="2 columns"):
        cs.MultivariateSimulation(cs.PCA(), [sgs, sgs]).fit(coords, data[:, :1])


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
        search = cs.Search(50.0, max_samples=max_samples, max_per_hole=max_per_hole)
        if kind == "SGS":
            return cs.SGS(gaussian, search)
        if kind == "TurningBands":
            return cs.TurningBands(gaussian, bands=50, step=1.0, search=search)
        if kind == "SIS":
            return cs.SIS([gaussian, gaussian], search)
        bands = cs.TurningBands(gaussian, bands=50, step=1.0, search=search)
        return cs.MultivariateSimulation(cs.PCA(), [cs.SGS(gaussian, search), bands])

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


@pytest.mark.parametrize("kind", [cs.SGS, cs.TurningBands])
def test_max_per_hole_caps_the_data_of_one_hole_with_a_trend(kind):
    down = np.column_stack([np.zeros(10), np.zeros(10), np.arange(10.0)])
    grades, trend = np.arange(10.0) * 7 % 10 + 1, np.arange(10.0) / 10

    def run(max_samples, max_per_hole=None, holes=None):
        search = cs.Search(50.0, max_samples=max_samples, max_per_hole=max_per_hole)
        options = {"bands": 50, "step": 1.0} if kind is cs.TurningBands else {}
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
        cs.Search(r, max_samples=12, min_samples=m, max_per_hole=3, high_grade=(4.0, 6.0))
        for r, m in ((10, 8), (25, 4), (40, 2))
    ]
    return xyz, grades, np.repeat(np.arange(30), 5), passes


def test_sgs_passes_are_the_kriging_passes():
    xyz, grades, holes, passes = holes_in_clusters()
    targets = cs.BlockModel(origin=(0, 0, 2), size=(2.5, 2.5, 1), count=(40, 40, 1))
    kriged = cs.OrdinaryKriging(gaussian, passes).fit(xyz, grades, holes=holes)
    want = kriged.predict(targets, diagnostics=True)["pass"]
    sgs = cs.SGS(gaussian, passes).fit(xyz, grades, holes=holes)
    np.testing.assert_array_equal(sgs.passes(targets), want)
    assert {1.0, 2.0, 3.0} <= set(want[~np.isnan(want)]) and np.isnan(want).any()


def test_sgs_with_passes_is_reproducible_and_one_pass_is_the_search(tmp_path):
    xyz, grades, holes, passes = holes_in_clusters()
    targets = cs.BlockModel(origin=(0, 0, 2), size=(5, 5, 1), count=(20, 20, 1))

    def run(model):
        return model.fit(xyz, grades, holes=holes).simulate(targets, n=3, seed=4, keep=True).realizations

    np.testing.assert_array_equal(run(cs.SGS(gaussian, [passes[1]])), run(cs.SGS(gaussian, passes[1])))
    by_pass = cs.SGS(gaussian, passes)
    reals = run(by_pass)
    np.testing.assert_array_equal(reals, run(cs.SGS(gaussian, passes)))
    assert not np.array_equal(reals, run(cs.SGS(gaussian, passes[2])))
    by_pass.to_parquet(tmp_path / "sgs.parquet")
    np.testing.assert_array_equal(
        cs.SGS.from_parquet(tmp_path / "sgs.parquet").simulate(targets, n=3, seed=4, keep=True).realizations,
        reals,
    )
    meta, columns = cs.SGS(gaussian, passes[1]).fit(xyz, grades, holes=holes)._state()
    older = json.loads(meta)
    older["search"] = older["search"][0]
    np.testing.assert_array_equal(
        cs.SGS._from_state(json.dumps(older), columns).simulate(targets, n=3, seed=4, keep=True).realizations,
        run(cs.SGS(gaussian, passes[1])),
    )
    mv = cs.MultivariateSimulation(cs.PCA(), [cs.SGS(gaussian, passes), cs.SGS(gaussian, passes[0])])
    two = mv.fit(xyz, np.column_stack([grades, grades**0.5]), holes=holes).simulate(targets, n=2, seed=1)
    assert len(two) == 2 and np.isfinite(two[0].mean).all()
    with pytest.raises(ValueError, match="at least one"):
        cs.SGS(gaussian, [])


def zoned_holes():
    xyz, grades, holes, passes = holes_in_clusters()
    zone = np.where(xyz[:, 0] < 40, "MS", "SM")
    grades = np.where(zone == "SM", grades * 3, grades)
    weights = np.where(xyz[:, 0] < 25, 0.5, 2.0)
    x, y = np.meshgrid(np.arange(0, 100, 5.0), np.arange(0, 100, 5.0))
    targets = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 2.0)])
    return xyz, grades, holes, weights, zone, targets, passes


def softened(passes, soft):
    return [cs.Search(s.radius, max_samples=12, min_samples=s.min_samples, soft=soft) for s in passes]


def test_sgs_with_one_domain_is_sgs_without():
    xyz, grades, holes, weights, _, targets, passes = zoned_holes()
    plain = cs.SGS(gaussian, passes).fit(xyz, grades, weights=weights, holes=holes)
    one = cs.SGS(gaussian, passes).fit(xyz, grades, weights=weights, holes=holes, domains="MS")
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
        sgs = cs.SGS(gaussian, search).fit(data[0], data[1], weights=data[2], holes=data[3], domains=domains)
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
    sgs = cs.SGS(gaussian, cs.Search(np.inf, max_samples=8)).fit(xyz, grades, weights=weights, domains=zone)
    mean = sgs.simulate(far, n=200, seed=3, domains=labels).mean
    for name in ("MS", "SM"):
        want = np.average(grades[zone == name], weights=weights[zone == name])
        assert mean[labels == name].mean() == pytest.approx(want, rel=0.1)


def test_a_contact_node_takes_its_own_domain_datum():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [-10.0, 0.0]])
    sgs = cs.SGS(gaussian, cs.Search(50.0, soft=np.inf)).fit(
        xy, [1.0, 9.0, 2.0, 8.0], domains=["A", "B", "A", "B"]
    )
    at = sgs.simulate([[0.0, 0.0], [0.0, 0.0]], n=4, seed=1, keep=True, domains=["A", "B"])
    np.testing.assert_array_equal(at.realizations, [[1.0, 9.0]] * 4)


def test_sgs_domains_errors_and_persistence(tmp_path):
    xyz, grades, holes, weights, zone, targets, passes = zoned_holes()
    soft = cs.Search(30.0, max_samples=12, soft={("MS", "SM"): 8.0})
    sgs = cs.SGS(gaussian, soft).fit(xyz, grades, weights=weights, holes=holes, domains=zone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")

    def run(model):
        return model.simulate(targets, n=2, seed=5, keep=True, domains=labels).realizations

    sgs.to_parquet(tmp_path / "sgs.parquet")
    for again in (cs.SGS.from_parquet(tmp_path / "sgs.parquet"), pickle.loads(pickle.dumps(sgs))):
        np.testing.assert_array_equal(run(again), run(sgs))
    with pytest.raises(cs.InvalidInput, match="simulate needs domains"):
        sgs.simulate(targets, n=1)
    with pytest.raises(cs.InvalidInput, match="passes needs domains"):
        sgs.passes(targets)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        sgs.simulate(targets, n=1, domains="QE")
    with pytest.raises(cs.InvalidInput, match="takes none"):
        cs.SGS(gaussian, passes).fit(xyz, grades).simulate(targets, n=1, domains="MS")
    with pytest.raises(cs.InvalidInput, match="needs domains at fit"):
        cs.SGS(gaussian, soft).fit(xyz, grades)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        cs.SGS(gaussian, soft).fit(xyz, grades, domains="MS")


def test_sgs_domains_with_a_trend():
    xyz, grades, holes, weights, zone, targets, passes = zoned_holes()
    trend, at = xyz[:, 1] / 100, targets[:, 1] / 100
    ms = zone == "MS"

    def run(search, rows, labels=None, **on):
        sgs = cs.SGS(gaussian, search, classes=3).fit(
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
    return cs.TurningBands(gaussian, bands=60, step=1.0, search=search, classes=3)


def test_turning_bands_domains_are_hard_unless_soft():
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    # Corners holding every datum: the bands cover one box whatever the data.
    targets = np.vstack([targets, [[0.0, 0.0, 0.0], [100.0, 100.0, 4.0]]])
    trend, at = xyz[:, 1] / 100, targets[:, 1] / 100
    ms, everything = zone == "MS", np.ones(len(zone), bool)
    hard = cs.Search(40.0, max_samples=12, max_per_hole=3, high_grade=(4.0, 6.0))

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
    soft = run(cs.Search(40.0, max_samples=12, soft=20.0), everything, zone, True, domains=labels)
    hard = run(cs.Search(40.0, max_samples=12), everything, zone, True, domains=labels)
    assert np.isfinite(soft).all() and not np.array_equal(soft, hard)


def test_each_turning_bands_domain_keeps_its_own_declustered_mean():
    xyz, grades, _, weights, zone, _, _ = zoned_holes()
    far = np.column_stack([25.0 * np.arange(40), np.full(40, 150.0), np.zeros(40)])
    labels = np.repeat(["MS", "SM"], 20)
    tb = banded(cs.Search(40.0, max_samples=8)).fit(xyz, grades, weights=weights, domains=zone)
    mean = tb.simulate(far, n=200, seed=3, domains=labels).mean
    for name in ("MS", "SM"):
        want = np.average(grades[zone == name], weights=weights[zone == name])
        assert mean[labels == name].mean() == pytest.approx(want, rel=0.1)


def test_turning_bands_domains_errors_and_persistence(tmp_path):
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    soft = cs.Search(30.0, max_samples=12, soft={("MS", "SM"): 8.0})
    tb = banded(soft).fit(xyz, grades, weights=weights, holes=holes, domains=zone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")

    def run(model):
        return model.simulate(targets, n=2, seed=5, keep=True, domains=labels).realizations

    tb.to_parquet(tmp_path / "tb.parquet")
    for again in (cs.TurningBands.from_parquet(tmp_path / "tb.parquet"), pickle.loads(pickle.dumps(tb))):
        np.testing.assert_array_equal(run(again), run(tb))
    with pytest.raises(cs.InvalidInput, match="simulate needs domains"):
        tb.simulate(targets, n=1)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        tb.simulate(targets, n=1, domains="QE")
    with pytest.raises(cs.InvalidInput, match="takes none"):
        banded().fit(xyz, grades).simulate(targets, n=1, domains="MS")
    with pytest.raises(cs.InvalidInput, match="needs domains at fit"):
        banded(soft).fit(xyz, grades)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        banded(soft).fit(xyz, grades, domains="MS")
    with pytest.raises(cs.InvalidInput, match="does not take domains"):
        cs.MultivariateSimulation(cs.PCA(), [banded(soft)])


@pytest.mark.parametrize(
    "model",
    [
        cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)),
        cs.TurningBands(gaussian, bands=60, search=cs.Search(radius=40, max_samples=12)),
    ],
)
def test_grades_follow_each_realization_of_simulated_domains(model):
    zone = np.where(coords[:, 0] < 50, "lean", "rich")
    grades = np.where(zone == "lean", values, 100 * values)
    model.fit(coords, grades, domains=zone)
    nodes = grid.centroids
    fixed = np.where(nodes[:, 0] < 50, "lean", "rich")

    def run(domains):
        return model.simulate(grid, n=3, seed=4, keep=True, domains=domains).realizations

    np.testing.assert_array_equal(run(np.tile(fixed, (3, 1))), run(fixed))
    simulated = np.array([np.where(nodes[:, 0] < edge, "lean", "rich") for edge in (20, 50, 80)])
    reals = run(simulated)
    split = (values.max() + 100 * values.min()) / 2
    for real, domains in zip(reals, simulated):
        assert (real[domains == "lean"] < split).all() and (real[domains == "rich"] > split).all()
    with pytest.raises(cs.InvalidInput, match="expected 3 realizations of domains, got 2"):
        run(simulated[:2])


@pytest.mark.parametrize("make", [lambda s: cs.SGS(gaussian, s), banded], ids=["SGS", "TurningBands"])
def test_simulators_take_column_names_and_domain_column(make):
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    soft = cs.Search(30.0, max_samples=12, soft=8.0)
    trend, at = xyz[:, 1] / 100, targets[:, 1] / 100
    labels = np.where(targets[:, 0] < 40, "MS", "SM")
    samples = cs.PointSet(xyz, {"zn": grades, "w": weights, "hole": holes, "t": trend, "zone": zone})
    nodes = cs.PointSet(targets, {"t": at, "zone": labels})
    arrays = make(soft).fit(xyz, grades, weights=weights, holes=holes, trend=trend, domains=zone)
    names = make(soft).fit(samples, "zn", weights="w", holes="hole", trend="t", domain_column="zone")
    want = arrays.simulate(targets, n=2, seed=5, keep=True, trend=at, domains=labels).realizations
    for model in (arrays, names):
        got = model.simulate(nodes, n=2, seed=5, keep=True, trend="t", domain_column="zone")
        np.testing.assert_array_equal(got.realizations, want)
    if isinstance(names, cs.SGS):
        passes = names.passes(nodes, domain_column="zone")
        np.testing.assert_array_equal(passes, arrays.passes(targets, domains=labels))
    with pytest.raises(cs.InvalidInput, match="one of domains or domain_column"):
        names.simulate(nodes, n=1, trend="t", domains=labels, domain_column="zone")
    with pytest.raises(cs.InvalidInput, match="one of domains or domain_column"):
        make(soft).fit(samples, "zn", domains=zone, domain_column="zone")
    with pytest.raises(cs.InvalidInput, match="simulate needs domains"):
        names.simulate(nodes, n=1, trend="t")
    with pytest.raises(cs.MissingColumn, match="columns: t, zone"):
        names.simulate(nodes, n=1, trend="t", domain_column="rock")
    with pytest.raises(TypeError):
        make(soft).fit(samples, "zn", "w")
    with pytest.raises(TypeError):
        names.simulate(nodes, 1)


def test_categorical_and_multivariate_simulators_take_column_names():
    rock = (values > 1).astype(int)
    samples = cs.PointSet(coords, {"rock": rock, "v": values, "root": values**0.5, "w": np.ones(60)})
    near = cs.Search(radius=40, max_samples=12)
    for model in (cs.SIS([gaussian] * 2, near), cs.Plurigaussian(gaussian, proportions=[0.5, 0.5])):
        want = model.fit(coords, rock).simulate(grid, n=2, seed=1, keep=True).realizations
        got = model.fit(samples, "rock").simulate(grid, n=2, seed=1, keep=True).realizations
        np.testing.assert_array_equal(got, want)
        with pytest.raises(TypeError):
            model.fit(coords, rock, None)
    sgs = cs.SGS(gaussian, near)
    mv = cs.MultivariateSimulation(cs.PCA(), [sgs, sgs])
    want = mv.fit(coords, np.column_stack([values, values**0.5]), weights=np.ones(60)).simulate(grid, n=2)
    got = mv.fit(samples, ["v", "root"], weights="w").simulate(grid, n=2)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a.mean, b.mean)
    with pytest.raises(cs.MissingColumn):
        mv.fit(samples, ["v", "missing"])


def test_cdf_bands_contain_the_data_cdf_of_the_distribution_drawn_from():
    points = cs.PointSet(coords, {"v": values, "w": rng.uniform(0.5, 2.0, 60)})
    p = points["w"] / points["w"].sum()
    drawn = rng.choice(values, size=(50, 400), p=p)
    check = cs.check_realizations(grid, drawn, points, "v", weights="w")
    inner = (check.probabilities >= 0.05) & (check.probabilities <= 0.95)
    for q, target in [
        (check.quantiles[0], check.data_quantiles[0]),
        (check.score_quantiles[0], cs.normal_ppf(check.probabilities)),
    ]:
        assert np.all((q.min(axis=0) <= target)[inner] & (target <= q.max(axis=0))[inner])
    shifted = cs.check_realizations(grid, drawn * 1.5, points, "v", weights="w").quantiles[0]
    assert np.mean(shifted.min(axis=0)[inner] > check.data_quantiles[0][inner]) > 0.5
    stats = check.statistics
    assert stats.num_rows == 51 and list(stats["realization"][:2]) == [0, 1]
    np.testing.assert_allclose(stats["mean"][0], np.average(values, weights=p))
    np.testing.assert_allclose(stats["mean"][1:], drawn.mean(axis=1))


def test_unconditional_sgs_reproduces_the_model_variogram_at_short_lags():
    model = cs.Variogram([("spherical", 1.0, 15.0)])
    data = cs.PointSet(rng.uniform(1000, 2000, (1000, 2)), {"v": rng.normal(size=1000)})
    nodes = cs.BlockModel((0.5, 0.5), (1.0, 1.0), (60, 60))
    sgs = cs.SGS(model, cs.Search(radius=30, max_samples=16)).fit(data, "v")
    reals = sgs.simulate(nodes, n=20, seed=4, keep=True)
    check = cs.check_realizations(nodes, reals, data, "v", variogram=model, lag=1.0, max_lag=8.0)
    assert check.directions == [(0.0, 0.0), (90.0, 0.0)]
    for d in range(2):
        mean = np.mean([r[d].gammas for r in check.variograms[0]], axis=0)
        np.testing.assert_allclose(mean, model.gamma(check.variograms[0][0][d].lags), atol=0.1)


def test_multivariate_and_categorical_checks():
    nodes = cs.BlockModel((0.5, 0.5), (1.0, 1.0), (20, 20))
    a = rng.normal(size=(8, 400))
    b = 0.8 * a + 0.6 * rng.normal(size=(8, 400))
    x = rng.normal(size=60)
    points = cs.PointSet(coords, {"a": x, "b": 0.8 * x + 0.6 * rng.normal(size=60)})
    check = cs.check_realizations(nodes, [a, b], points, ["a", "b"], lag=2.0, max_lag=8.0)
    assert check.names == ["a", "b"] and check.correlations.shape == (8, 2, 2)
    np.testing.assert_allclose(check.correlations[3, 0, 1], np.corrcoef(a[3], b[3])[0, 1])
    np.testing.assert_allclose(check.data_correlation, cs.correlation(points, columns=["a", "b"]))
    assert len(check.variograms[1]) == 8 and check.directions is None
    codes = (values > 1).astype(int) + (values > 2)
    reals = rng.integers(0, 3, (6, 400))
    cats = cs.check_realizations(nodes, reals, coords, codes, lag=2.0, max_lag=8.0)
    assert cats.categorical and cats.quantiles is None and cats.proportions.shape == (6, 3)
    np.testing.assert_allclose(cats.data_proportions, np.bincount(codes, minlength=3) / 60)
    np.testing.assert_allclose(cats.proportions[0], np.bincount(reals[0], minlength=3) / 400)
    indicator = cs.experimental_variogram(nodes, (reals[2] == 1).astype(float), 2.0, 8.0)
    np.testing.assert_array_equal(cats.variograms[1][2][0].gammas, indicator.gammas)
    no_reals = cs.SGS(gaussian, cs.Search(40.0)).fit(coords, values).simulate(nodes, n=2)
    for args, options in [
        ([a, b], {}),
        (a, {"lag": 2.0}),
        (no_reals, {}),
    ]:
        with pytest.raises(ValueError):
            cs.check_realizations(nodes, args, points, "a", **options)


def _cosimulation_case():
    model = cs.Variogram([("spherical", 1.0, 10.0)])
    nodes = cs.BlockModel((0.5, 0.5), (1.0, 1.0), (40, 40))
    far = cs.PointSet(rng.uniform(1000, 2000, (500, 2)), {"v": rng.normal(size=500)})
    field = cs.SGS(model, cs.Search(radius=30, max_samples=16)).fit(far, "v")
    secondary = field.simulate(nodes, n=2, seed=9, keep=True).realizations
    rows = rng.choice(1600, 100, replace=False)
    s = secondary[0]
    scores = (s - s.mean()) / s.std()
    grade = np.exp(0.7 * scores[rows] + np.sqrt(0.51) * rng.normal(size=100))
    points = cs.PointSet(nodes.centroids[rows], {"v": grade, "s": s[rows]})
    return model, nodes, secondary, rows, points


def test_collocated_cosimulation_reproduces_correlation_histogram_and_variogram():
    model, nodes, secondary, rows, points = _cosimulation_case()
    s = secondary[0]
    search = cs.Search(radius=30, max_samples=16)
    sgs = cs.SGS(model, search).fit(points, "v", secondary="s")
    assert sgs.correlation == pytest.approx(0.7, abs=0.15)
    reals = sgs.simulate(nodes, n=10, seed=1, secondary=s, keep=True).realizations
    np.testing.assert_array_equal(reals[:, rows], np.tile(points["v"], (10, 1)))
    logs = cs.PointSet(points.coords, {"v": np.log(points["v"]), "s": points["s"]})
    check = cs.check_realizations(
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
    np.testing.assert_allclose(median[inner], cs.normal_ppf(check.probabilities[inner]), atol=0.3)
    for d in range(2):
        mean = np.mean([r[d].gammas for r in check.variograms[0]], axis=0)
        np.testing.assert_allclose(mean, model.gamma(check.variograms[0][0][d].lags), atol=0.15)

    independent = cs.SGS(model, search).fit(points, "v").simulate(nodes, n=3, seed=1, keep=True)
    zero = cs.SGS(model, search).fit(points, "v", secondary="s", correlation=0.0)
    assert zero.correlation == 0.0
    same = zero.simulate(nodes, n=3, seed=1, secondary=s, keep=True).realizations
    np.testing.assert_array_equal(same, independent.realizations)


def test_cosimulation_takes_a_secondary_realization_per_realization(tmp_path):
    model, nodes, secondary, _, points = _cosimulation_case()
    sgs = cs.SGS(model, cs.Search(radius=30, max_samples=16)).fit(points, "v", secondary="s", correlation=0.8)
    both = sgs.simulate(nodes, n=2, seed=4, secondary=secondary, keep=True).realizations
    second = sgs.simulate(nodes, n=2, seed=4, secondary=secondary[[1, 1]], keep=True).realizations
    np.testing.assert_array_equal(both[1], second[1])
    assert not np.array_equal(both[0], second[0])
    sgs.to_parquet(tmp_path / "cosgs.parquet")
    loaded = cs.SGS.from_parquet(tmp_path / "cosgs.parquet")
    assert loaded.correlation == 0.8
    again = loaded.simulate(nodes, n=2, seed=4, secondary=secondary, keep=True).realizations
    np.testing.assert_array_equal(again, both)
    plain = cs.SGS(model, cs.Search(radius=30)).fit(points, "v")
    for call in [
        lambda: sgs.simulate(nodes, n=2),
        lambda: sgs.simulate(nodes, n=3, secondary=secondary),
        lambda: sgs.simulate(nodes, n=1, secondary=secondary[0][:10]),
        lambda: plain.simulate(nodes, n=1, secondary=secondary[0]),
        lambda: cs.SGS(model, cs.Search(radius=30)).fit(points, "v", correlation=0.5),
        lambda: cs.SGS(model, cs.Search(radius=30)).fit(points, "v", secondary="s", correlation=1.5),
    ]:
        with pytest.raises(ValueError):
            call()


def test_correct_distribution_matches_the_target_and_keeps_ranks():
    rng = np.random.default_rng(11)
    reals = rng.normal(size=(4, 250))
    target = rng.lognormal(0.0, 0.7, 250)
    out = cs.correct_distribution(reals, target)
    np.testing.assert_allclose(np.sort(out, axis=1), np.tile(np.sort(target), (4, 1)), atol=1e-9)
    np.testing.assert_array_equal(np.argsort(out, axis=1), np.argsort(reals, axis=1))
    np.testing.assert_array_equal(cs.correct_distribution(reals, target, strength=0.0), reals)
    half = cs.correct_distribution(reals, target, strength=0.5)
    np.testing.assert_allclose(half, 0.5 * (reals + out))
    only = cs.correct_distribution(reals, target, realizations=[2])
    np.testing.assert_array_equal(only[[0, 1, 3]], reals[[0, 1, 3]])
    np.testing.assert_array_equal(only[2], out[2])
    one = cs.correct_distribution(reals[0], target)
    assert one.shape == (250,)
    np.testing.assert_array_equal(one, out[0])
    kde = cs.KernelDensity(lower=0.0).fit(target)
    smooth = cs.correct_distribution(reals, kde)
    np.testing.assert_allclose(np.median(smooth, axis=1), kde.quantile([0.5])[0], rtol=0.02)
    with pytest.raises(ValueError):
        cs.correct_distribution(reals, target, strength=2.0)
    with pytest.raises(ValueError):
        cs.correct_distribution(reals, target, realizations=[9])


def test_correct_distribution_takes_a_summary():
    model = cs.Variogram([("spherical", 1.0, 30.0)])
    rng = np.random.default_rng(2)
    points = cs.PointSet(rng.uniform(0, 100, (40, 2)), {"v": rng.lognormal(0.0, 0.5, 40)})
    nodes = cs.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(20, 20))
    sgs = cs.SGS(model, cs.Search(radius=40)).fit(points, "v")
    summary = sgs.simulate(nodes, n=3, seed=1, keep=True)
    out = cs.correct_distribution(summary, points["v"])
    assert out.shape == summary.realizations.shape
    np.testing.assert_allclose(out.mean(axis=1), points["v"].mean(), rtol=0.01)
    with pytest.raises(ValueError):
        cs.correct_distribution(sgs.simulate(nodes, n=2), points["v"])


def test_keep_selects_realizations():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    everything = sgs.simulate(grid, n=4, seed=7, keep=True)
    some = sgs.simulate(grid, n=4, seed=7, keep=[3, 1])
    assert everything.kept == [0, 1, 2, 3]
    assert some.kept == [1, 3]
    np.testing.assert_array_equal(some.realizations, everything.realizations[[1, 3]])
    np.testing.assert_array_equal(some.mean, everything.mean)
    none = sgs.simulate(grid, n=4, seed=7)
    assert none.kept == [] and none.realizations is None


def test_keep_rejects_bad_indices():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    for keep in ([4], [1, 1], [-1], "all"):
        with pytest.raises(cs.InvalidInput):
            sgs.simulate(grid, n=4, seed=7, keep=keep)


def test_kept_realizations_survive_parquet(tmp_path):
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    summary = sgs.simulate(grid, n=4, seed=7, keep=[2])
    summary.to_parquet(tmp_path / "s.parquet")
    back = cs.SimulationSummary.from_parquet(tmp_path / "s.parquet")
    assert back.kept == [2]
    np.testing.assert_array_equal(back.realizations, summary.realizations)
