import pickle

import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(5)
coords = rng.uniform(0, 100, (60, 2))
values = rng.lognormal(0, 0.6, 60)
gaussian = cs.Variogram([("spherical", 1.0, 30.0)])
grid = cs.BlockModel(origin=(0, 0), size=(5, 5), count=(20, 20))


def test_sgs_is_reproducible_and_honours_data():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    a = sgs.simulate(grid, n=3, seed=7, realizations=True).realizations
    assert a.shape == (3, 400)
    np.testing.assert_array_equal(a, sgs.simulate(grid, n=3, seed=7, realizations=True).realizations)
    assert not np.array_equal(a[0], a[1])
    at_data = sgs.simulate(coords[:5], n=4, seed=1)
    np.testing.assert_allclose(at_data.mean, values[:5])
    np.testing.assert_allclose(at_data.variance, 0, atol=1e-12)


def test_sgs_summary_matches_its_realizations():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    cut = float(np.median(values))
    s = sgs.simulate(grid, n=20, seed=3, cutoffs=[cut], quantiles=[0.1, 0.5, 0.9], realizations=True)
    reals = s.realizations
    np.testing.assert_allclose(s.mean, reals.mean(axis=0))
    np.testing.assert_allclose(s.variance, reals.var(axis=0), atol=1e-9)
    np.testing.assert_allclose(s.probability_above[0], (reals > cut).mean(axis=0))
    np.testing.assert_allclose(s.quantile_values[1], np.median(reals, axis=0), rtol=1e-6)
    np.testing.assert_allclose(s.realization_above[0], (reals > cut).mean(axis=1))
    assert np.median(reals) == pytest.approx(np.median(values), rel=0.25)
    assert sgs.simulate(grid, n=2, seed=3).realizations is None


def test_block_support_averages_each_realization():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    nodes = sgs.simulate(grid, n=6, seed=2, realizations=True).realizations
    s = sgs.simulate(grid, n=6, seed=2, cutoffs=[1.0], realizations=True, blocks=blocks)
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
    c = sis.simulate(grid, n=3, seed=4, realizations=True, blocks=blocks)
    assert c.realizations.shape == (3, 25) and c.probabilities.shape == (2, 25)


def test_localize_realizations_within_panels():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    panels = cs.BlockModel(origin=(0, 0), size=(50, 50), count=(2, 2))
    smus = panels.discretize(5)
    s = sgs.simulate(grid, n=8, seed=2, realizations=True, blocks=smus)
    smus = smus.with_column("etype", s.mean)
    out = cs.localize(smus, "etype", s.realizations, panels)
    owner, local = smus["block"].astype(int), out["localized"]
    for p in range(4):
        mine = owner == p
        pooled = np.sort(s.realizations[:, mine].ravel())
        assert local[mine].mean() == pytest.approx(pooled.mean(), rel=1e-9)
        by_rank = local[mine][np.argsort(s.mean[mine], kind="stable")]
        np.testing.assert_allclose(by_rank, pooled.reshape(25, 8).mean(axis=1), rtol=1e-12)

    one = cs.localize(smus, "etype", s.realizations[:1], panels, name="one")["one"]
    for p in range(4):
        mine = owner == p
        by_rank = one[mine][np.argsort(s.mean[mine], kind="stable")]
        np.testing.assert_array_equal(by_rank, np.sort(s.realizations[0, mine]))
    with pytest.raises(cs.InvalidInput):
        cs.localize(smus, "etype", s.realizations[:, 1:], panels)
    with pytest.raises(KeyError):
        cs.localize(smus, "missing", s.realizations, panels)


def test_turning_bands_summary():
    tb = cs.TurningBands(gaussian, bands=100, step=1.0).fit(coords, values)
    s = tb.simulate(grid, n=2, seed=1, cutoffs=[1.0, 2.0])
    assert s.mean.shape == (400,) and s.probability_above.shape == (2, 400)


def test_sis_probabilities():
    cats = (values > np.median(values)).astype(int)
    sis = cs.SIS([gaussian, gaussian], cs.Search(radius=40, max_samples=12)).fit(coords, cats)
    s = sis.simulate(grid, n=4, seed=4, realizations=True)
    assert s.probabilities.shape == (2, 400) and s.proportions.shape == (4, 2)
    np.testing.assert_allclose(s.probabilities.sum(axis=0), 1.0)
    assert set(np.unique(s.realizations)) <= {0, 1}
    assert ((s.entropy >= 0) & (s.entropy <= 1)).all()


def test_plurigaussian_proportions():
    facies = rng.choice(3, 60, p=[0.2, 0.3, 0.5])
    pgs = cs.Plurigaussian(gaussian, proportions=[0.2, 0.3, 0.5]).fit(coords, facies)
    s = pgs.simulate(grid, n=2, seed=2)
    assert s.probabilities.shape == (3, 400) and set(np.unique(s.most_likely)) <= {0, 1, 2}


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


def test_simulation_with_a_trend_follows_it_and_honours_data():
    xy, z, trend = trended_samples()
    search = cs.Search(radius=30, max_samples=12)
    node_trend = grid.centroids[:, 0] / 100
    white = cs.Variogram([("spherical", 1.0, 4.0)])
    for simulator in (cs.SGS(white, search, classes=5), cs.TurningBands(white, bands=100, classes=5)):
        simulator.fit(xy, z, trend=trend)
        reals = simulator.simulate(grid, n=4, seed=2, realizations=True, trend=node_trend).realizations
        want = np.corrcoef(np.log(z), trend)[0, 1]
        for r in reals:
            assert np.corrcoef(np.log(r), node_trend)[0, 1] == pytest.approx(want, abs=0.12)
        at_data = simulator.simulate(xy[:5], n=3, seed=1, trend=trend[:5])
        np.testing.assert_allclose(at_data.mean, z[:5])
        np.testing.assert_allclose(at_data.variance, 0, atol=1e-9)

    sgs = cs.SGS(gaussian, search).fit(xy, z, trend=trend)
    model = grid.with_column("drift", node_trend)
    by_array = sgs.simulate(grid, n=3, seed=4, realizations=True, trend=node_trend).realizations
    np.testing.assert_array_equal(
        sgs.simulate(model, n=3, seed=4, realizations=True, trend="drift").realizations, by_array
    )
    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    by_block = sgs.simulate(
        grid, n=3, seed=4, realizations=True, blocks=blocks, trend=node_trend
    ).realizations
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
    with pytest.raises(cs.InvalidInput, match="BlockModel"):
        sgs.simulate(grid.centroids, n=1, trend="drift")
    with pytest.raises(ValueError):
        sgs.simulate(grid, n=1, trend=node_trend[1:])
    with pytest.raises(ValueError):
        cs.SGS(gaussian, search).fit(xy, z, trend=trend[1:])
    with pytest.raises(cs.InvalidInput, match="does not take a trend"):
        cs.TurningBands(gaussian).fit(xy, z, trend=trend).simulate_to_parquet("in.parquet", "out.parquet")


def test_simulators_with_a_trend_save_and_load_it(tmp_path):
    xy, z, trend = trended_samples()
    node_trend = grid.centroids[:, 0] / 100
    for simulator in (
        cs.SGS(gaussian, cs.Search(radius=30, max_samples=12), classes=4),
        cs.TurningBands(gaussian, bands=50, classes=4),
    ):
        simulator.fit(xy, z, trend=trend)
        want = simulator.simulate(grid, n=2, seed=6, realizations=True, trend=node_trend).realizations
        path = tmp_path / "simulator.parquet"
        simulator.to_parquet(path)
        for back in (type(simulator).from_parquet(path), pickle.loads(pickle.dumps(simulator))):
            got = back.simulate(grid, n=2, seed=6, realizations=True, trend=node_trend).realizations
            np.testing.assert_array_equal(got, want)


def test_multivariate_simulation_reproduces_correlation_and_honours_data():
    g = rng.standard_normal((60, 2))
    data = np.exp(np.column_stack([g[:, 0], 0.8 * g[:, 0] + 0.6 * g[:, 1]]))
    search = cs.Search(radius=40, max_samples=12)
    mv = cs.MultivariateSimulation(
        cs.PPMT(seed=3), [cs.SGS(gaussian, search), cs.TurningBands(gaussian, bands=100, step=1.0)]
    ).fit(coords, data)
    a, b = mv.simulate(grid, n=10, seed=4, realizations=True)
    assert a.realizations.shape == b.realizations.shape == (10, 400)
    r = np.mean([np.corrcoef(np.log(x), np.log(y))[0, 1] for x, y in zip(a.realizations, b.realizations)])
    assert r == pytest.approx(np.corrcoef(np.log(data.T))[0, 1], abs=0.15)
    np.testing.assert_array_equal(
        a.realizations, mv.simulate(grid, n=10, seed=4, realizations=True)[0].realizations
    )

    at_data = mv.simulate(coords[:5], n=3, seed=1)
    for v, s in enumerate(at_data):
        np.testing.assert_allclose(s.mean, data[:5, v], rtol=1e-6)

    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    xy = grid.centroids[:, :2] // 20
    rows = (xy[:, 0] + 5 * xy[:, 1]).astype(int)
    by_block = mv.simulate(grid, n=10, seed=4, realizations=True, blocks=blocks)
    for s, nodes in zip(by_block, (a, b)):
        expected = np.stack([np.bincount(rows, x) / np.bincount(rows) for x in nodes.realizations])
        np.testing.assert_allclose(s.realizations, expected)


def test_multivariate_simulation_drops_incomplete_samples_and_checks_inputs():
    data = np.column_stack([values, values**0.5])
    data[:4, 1] = np.nan
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12))
    with pytest.warns(UserWarning, match="4 samples miss a variable"):
        cs.MultivariateSimulation(cs.StepwiseConditional(), [sgs, sgs]).fit(coords, data)
    with pytest.raises(ValueError, match="transform must be"):
        cs.MultivariateSimulation(cs.NormalScore(), [sgs])
    with pytest.raises(ValueError, match="2 columns"):
        cs.MultivariateSimulation(cs.PCA(), [sgs, sgs]).fit(coords, data[:, :1])
