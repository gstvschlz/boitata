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
