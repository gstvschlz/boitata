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


def test_describe_by_ends_with_the_all_data_row():
    v = np.r_[rng.lognormal(0, 1, 90), np.nan]
    c = np.repeat(["b", "a", "c"], 31)[:91]
    w = rng.uniform(0.5, 2, 91)
    t = cs.describe_by(v, c, w, quantiles=[0.5, 0.975])
    assert t.column_names == ["category", "n", "mean", "variance", "std", "cv", "min", "max", "P50", "P97.5"]
    assert t["category"] == ["a", "b", "c", "all"]
    whole = cs.describe(v, w, quantiles=[0.5, 0.975])
    assert t["n"][-1] == whole["n"] == t["n"][:3].sum()
    assert t["mean"][-1] == pytest.approx(whole["mean"])
    np.testing.assert_allclose([t["P50"][-1], t["P97.5"][-1]], whole["quantiles"])
    a = c == "a"
    assert t["mean"][0] == pytest.approx(np.average(v[a], weights=w[a]))


def test_grade_tonnage_from_data():
    v = rng.lognormal(0, 1, 200)
    volume, density = rng.uniform(1, 2, 200), rng.uniform(2.5, 3, 200)
    gt = cs.grade_tonnage(v, [-np.inf, 0.5, 1, 2, 5, 1e9], volume, density)
    tonnes = volume * density
    assert gt["tonnage"][0] == pytest.approx(tonnes.sum())
    assert gt["mean_grade"][0] == pytest.approx(np.average(v, weights=tonnes))
    assert np.all(np.diff(gt["tonnage"]) <= 0)
    np.testing.assert_allclose(gt["metal"][:-1], gt["tonnage"][:-1] * gt["mean_grade"][:-1])
    assert gt["tonnage"][-1] == 0 and np.isnan(gt["mean_grade"][-1])
    assert cs.grade_tonnage(v, [2.0])["tonnage"][0] == (v >= 2).sum()


def test_capping_report_by_domain():
    v = rng.lognormal(0, 1, 300)
    d = np.repeat([1, 2, 3], 100)
    w = rng.uniform(0.5, 2, 300)
    caps = {1: 3.0, 3: 5.0}
    t = cs.capping_report(v, d, caps, weights=w)
    assert t["domain"] == ["1", "2", "3", "all"]
    cap = np.select([d == 1, d == 3], [3.0, 5.0], np.inf)
    removed = np.maximum(v - cap, 0) * w
    np.testing.assert_allclose(
        t["metal_removed"], [removed[d == k].sum() for k in (1, 2, 3)] + [removed.sum()]
    )
    np.testing.assert_allclose(t["cap"], [3.0, np.nan, 5.0, np.nan])
    assert t["n_capped"][-1] == (v > cap).sum()
    assert t["max_capped"][0] == 3.0 and t["max_capped"][1] == t["max"][1]
    assert t["mean_capped"][-1] == pytest.approx(np.average(np.minimum(v, cap), weights=w))
    with pytest.raises(cs.InvalidInput):
        cs.capping_report(v, d, {4: 1.0})
