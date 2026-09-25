import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(7)


def test_describe_skips_nan_and_matches_hazen_quantiles():
    v = rng.lognormal(0, 1, 200)
    s = cs.describe(np.r_[v, np.nan], quantiles=[0.1, 0.5, 0.9])
    assert s["n"] == 200
    assert s["mean"] == pytest.approx(v.mean())
    assert s["std"] == pytest.approx(v.std())
    np.testing.assert_allclose(s["quantiles"], np.quantile(v, [0.1, 0.5, 0.9], method="hazen"))
    w = rng.uniform(0.5, 2, 200)
    assert cs.describe(v, w)["mean"] == pytest.approx(np.average(v, weights=w))
    with pytest.raises(cs.InvalidInput):
        cs.describe(v, weights=-w)


def test_swath_along_azimuth_and_axis():
    xy = rng.uniform(0, 100, (500, 2))
    s = cs.swath(xy, xy[:, 0], 10.0, azimuth=90)
    np.testing.assert_allclose(s["mean"], s["centres"], atol=1.5)
    assert sum(s["count"]) == 500
    assert cs.swath(xy, xy[:, 0], 10.0, axis="x")["count"] == s["count"]
    with pytest.raises(cs.InvalidInput):
        cs.swath(xy, xy[:, 0], 10.0)


def test_contact_signs_distance_by_side():
    z = np.tile(np.arange(20.0), 2)
    coords = np.c_[np.zeros(40), np.zeros(40), z]
    domains = np.where(z < 10, "ore", "waste")
    values = np.where(z < 10, 3.0, 1.0)
    c = cs.contact(coords, values, domains, np.repeat(["A", "B"], 20), "ore", "waste", 50.0, 2.0)
    assert sum(c["count"]) == 40
    np.testing.assert_array_equal(c["mean"], np.where(c["distance"] < 0, 3.0, 1.0))
    with pytest.raises(cs.InvalidInput):
        cs.contact(coords, values, domains, np.zeros(40), "ore", "oxide", 50.0, 2.0)


def test_capping_at_maximum_removes_nothing():
    v = rng.lognormal(0, 1, 300)
    c = cs.capping(v, caps=[v.max(), np.median(v)])
    assert c["metal_removed"][0] == pytest.approx(0.0)
    assert c["fraction"][1] == pytest.approx(0.5)
    assert c["mean"][1] == pytest.approx(np.minimum(v, np.median(v)).mean())
    assert len(cs.capping(v)["cap"]) == 6


def test_h_scatter_correlation_decays():
    x = np.arange(300.0)
    coords = np.c_[x, np.zeros(300)]
    z = np.sin(x / 15)
    r = [cs.h_scatter(coords, z, lag, 0.5)[2] for lag in (1, 10, 20)]
    assert r[0] > 0.99 and r[0] > r[1] > r[2]
    head, tail, _ = cs.h_scatter(coords, z, 1.0, 0.1, azimuth=90, other=2 * z)
    np.testing.assert_allclose(tail, 2 * np.sin((x[:-1]) / 15))
    assert len(head) == 299


def test_correlation_pairwise():
    a = rng.normal(size=100)
    data = np.c_[a, 2 * a + 1, a**3]
    data[5, 1] = np.nan
    r = cs.correlation(data)
    np.testing.assert_allclose(r[0, 1], 1.0)
    np.testing.assert_allclose(r, r.T)
    assert cs.correlation(data, method="spearman")[0, 2] == pytest.approx(1.0)
    with pytest.raises(cs.InvalidInput):
        cs.correlation(data, method="kendall")
