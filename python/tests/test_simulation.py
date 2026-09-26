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
