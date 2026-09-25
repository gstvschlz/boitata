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
    a = sgs.simulate(grid, n=3, seed=7)
    assert a.shape == (3, 400)
    np.testing.assert_array_equal(a, sgs.simulate(grid, n=3, seed=7))
    assert not np.array_equal(a[0], a[1])
    np.testing.assert_allclose(sgs.simulate(coords[:5], seed=1)[0], values[:5])


def test_sgs_reproduces_the_histogram():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    reals = sgs.simulate(grid, n=20, seed=3)
    assert np.median(reals) == pytest.approx(np.median(values), rel=0.25)
    assert reals.min() >= values.min() - 1e-6


def test_turning_bands_shapes():
    tb = cs.TurningBands(gaussian, bands=100, step=1.0).fit(coords, values)
    assert tb.simulate(grid, n=2, seed=1).shape == (2, 400)


def test_sis_returns_known_categories():
    cats = (values > np.median(values)).astype(int)
    sis = cs.SIS([gaussian, gaussian], cs.Search(radius=40, max_samples=12)).fit(coords, cats)
    out = sis.simulate(grid, n=2, seed=4)
    assert out.shape == (2, 400) and set(np.unique(out)) <= {0, 1}


def test_plurigaussian_proportions():
    facies = rng.choice(3, 60, p=[0.2, 0.3, 0.5])
    pgs = cs.Plurigaussian(gaussian, proportions=[0.2, 0.3, 0.5]).fit(coords, facies)
    out = pgs.simulate(grid, seed=2)
    assert out.shape == (400,) and set(np.unique(out)) <= {0, 1, 2}


def test_gibbs_respects_bounds():
    pts = rng.uniform(0, 50, (20, 2))
    bounds = np.column_stack([np.zeros(20), np.full(20, np.inf)])
    draw = cs.gibbs(pts, bounds, gaussian, seed=3)
    assert np.all(draw >= 0)


def test_postprocessing():
    reals = rng.normal(size=(200, 5))
    stats = cs.summarize_realizations(reals, quantiles=[0.5])
    np.testing.assert_allclose(stats["mean"], reals.mean(axis=0))
    assert cs.probability_above(reals, 0.0) == pytest.approx((reals > 0).mean(axis=0))
