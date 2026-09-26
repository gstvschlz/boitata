import ceres as cs
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)


@pytest.fixture(autouse=True)
def close():
    yield
    plt.close("all")


def test_histogram_and_probability():
    v = rng.lognormal(0, 1, 300)
    _, ax = cs.plot.histogram(v, weights=np.full(300, 2.0), log=True)
    assert ax.get_xscale() == "log"
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(1.0)
    _, ax = cs.plot.probability(np.log(v))
    x, y = ax.lines[0].get_data()
    assert np.corrcoef(x, y)[0, 1] > 0.98


def test_cdf_is_monotone_to_one_and_follows_weights():
    v = rng.lognormal(0, 1, 300)
    w = np.where(v > np.median(v), 3.0, 1.0)
    _, ax = cs.plot.cdf([v, v], weights=[None, w], labels=["naive", "declustered"], log=True)
    (x, naive), (_, weighted) = (line.get_data() for line in ax.lines)
    assert np.all(np.diff(x) >= 0) and np.all(np.diff(naive) >= 0)
    assert naive[-1] == pytest.approx(1.0) and weighted[-1] == pytest.approx(1.0)
    assert np.all(weighted <= naive + 1e-12) and weighted[150] < naive[150]
    _, ax = cs.plot.cdf(np.append(v, np.nan), weights=np.append(w, 1.0))
    assert len(ax.lines) == 1


def test_qq_of_a_sample_against_itself_is_on_the_diagonal():
    v = rng.lognormal(0, 1, 300)
    _, ax = cs.plot.qq(v, v)
    x, y = ax.lines[0].get_data()
    np.testing.assert_allclose(x, y)
    w = np.where(v > np.median(v), 3.0, 1.0)
    _, ax = cs.plot.qq(v, v, y_weights=w, quantiles=[0.25, 0.5, 0.75])
    x, y = ax.lines[0].get_data()
    np.testing.assert_allclose(y, cs.describe(v, w, quantiles=[0.25, 0.5, 0.75])["quantiles"])
    assert np.all(y > x)


def test_boxplot_quartiles_are_describe_quantiles():
    v = rng.lognormal(0, 1, 300)
    domain = np.where(np.arange(300) < 100, "b", "a")
    w = rng.uniform(0.5, 2.0, 300)
    _, ax = cs.plot.boxplot(v, domain, weights=w, sort=True)
    stats = [cs.describe(v[domain == d], w[domain == d]) for d in ("a", "b")]
    order = np.argsort([s["quantiles"][2] for s in stats])
    for box, i in zip(ax.patches, order, strict=True):
        y = box.get_path().vertices[:, 1]
        assert (y.min(), y.max()) == pytest.approx(tuple(stats[i]["quantiles"][[1, 3]]))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[order.tolist().index(0)] == "a\nn = 200"
    _, ax = cs.plot.boxplot(v, domain)
    assert ax.patches[0].get_path().vertices[:, 1].max() != pytest.approx(stats[0]["quantiles"][3])


def test_variogram_with_anisotropic_model():
    xy = rng.uniform(0, 100, (200, 2))
    exp = cs.experimental_variogram(xy, rng.normal(size=200), 10.0, 60.0, azimuth=30)
    model = cs.Variogram([("spherical", 1.0, 40.0)], rotation=(30, 0, 0), ratios=(0.5, 1.0))
    _, ax = cs.plot.variogram(exp, model, direction=(120, 0))
    h, g = ax.lines[0].get_data()
    assert g[np.searchsorted(h, 20.0)] == pytest.approx(1.0, abs=1e-9)


def test_scatter_reports_slope():
    x = rng.normal(size=100)
    _, ax = cs.plot.scatter(x, 2 * x + 1)
    assert ax.lines[1].get_label() == "slope 2.00"


def test_section_slices_a_masked_model():
    bm = cs.BlockModel(origin=(0, 0, 0), size=(2, 2, 1), count=(4, 3, 2))
    bm = bm.with_column("g", np.arange(24.0)).mask(np.arange(24) != 5)
    _, ax = cs.plot.section(bm, "g", axis="z", index=0)
    image = ax.images[0].get_array()
    assert image.shape == (3, 4) and np.isnan(np.ma.filled(image, np.nan)[1, 1])
    assert ax.images[0].get_extent() == [0, 8, 0, 6]
    _, ax = cs.plot.section(bm, np.ones(23), axis="x")
    assert ax.images[0].get_array().shape == (2, 3)


def test_swath_draws_each_result():
    xy = rng.uniform(0, 100, (300, 2))
    s = cs.swath(xy, xy[:, 0], 10.0, axis="x")
    _, ax = cs.plot.swath([s, s], labels=["a", "b"])
    assert len(ax.lines) == 2


def test_uncertain_fades_to_white():
    values = np.array([[0.0, 1.0], [1.0, np.nan]])
    uncertainty = np.array([[0.0, 1.0], [0.5, 0.0]])
    _, ax = cs.plot.uncertain(values, uncertainty)
    rgba = ax.images[0].get_array()
    np.testing.assert_allclose(rgba[0, 1, :3], 1.0)
    assert rgba[0, 0, :3].max() < 1 and rgba[1, 1, 3] == 0


def test_uncertain_slices_a_masked_model():
    bm = cs.BlockModel(origin=(0, 0, 0), size=(2, 2, 1), count=(4, 3, 2))
    bm = bm.with_column("g", np.arange(24.0)).mask(np.arange(24) != 5)
    uncertainty = np.where(bm.index == 6, 1.0, 0.0)
    _, ax = cs.plot.uncertain("g", uncertainty, block_model=bm, axis="z", index=0)
    rgba = ax.images[0].get_array()
    assert rgba.shape == (3, 4, 4) and ax.images[0].get_extent() == [0, 8, 0, 6]
    np.testing.assert_allclose(rgba[1, 2, :3], 1.0)
    assert rgba[1, 1, 3] == 0 and rgba[1, 3, :3].max() < 1


def test_probability_draws_the_cap():
    v = rng.lognormal(0, 1, 300)
    _, ax = cs.plot.probability(v, log=True, cap=5.0)
    assert ax.lines[-1].get_xdata()[0] == 5.0
