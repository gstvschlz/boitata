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


def test_pairs_recover_twins_and_their_bias():
    a = np.c_[np.arange(0.0, 500.0, 25.0), np.zeros(20)]
    order = rng.permutation(20)
    b = cs.PointSet(a[order] + rng.uniform(-1, 1, (20, 2)))
    za = rng.uniform(1, 5, 20)
    zb = 1.1 * za[order]
    p = cs.pairs(a, b, 3.0, values=(za, zb))
    np.testing.assert_array_equal(order[p["b"].astype(int)], p["a"])
    np.testing.assert_allclose(p["value_b"], 1.1 * p["value_a"])
    bias = cs.paired_bias(p, bins=[0, 1, 2])
    assert bias["n"].sum() == 20
    np.testing.assert_allclose(bias["bias"], 0.1)
    assert len(cs.pairs(a, b, 3.0, holes=(np.arange(20), order))) == 0
    with pytest.raises(cs.InvalidInput):
        cs.pairs(a, b, 3.0, values=za)


def test_duplicates_report_group_ids_and_merge():
    coords = np.array([[0, 0, 0], [5, 0, 0], [0, 0, 0], [0.4, 0.3, 0], [9, 9, 9]], dtype=float)
    report, group = cs.duplicates(coords)
    np.testing.assert_array_equal(group, [0, -1, 0, -1, -1])
    assert report["n"].tolist() == [2] and report["spread"].tolist() == [0.0]
    report, group = cs.duplicates(coords, tolerance=0.5)
    np.testing.assert_array_equal(group, [0, -1, 0, 0, -1])
    assert report["spread"][0] == pytest.approx(0.5)

    zn = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    cu = np.arange(1.0, 6.0)
    rock = ["a", "b", "c", "d", "e"]
    points = cs.PointSet(coords, {"zn": zn, "cu": cu, "rock": rock}, crs="EPSG:32718")
    merged = cs.duplicates(points, 0.5, merge="mean", weights=[1, 1, 3, 1, 1])
    np.testing.assert_allclose(merged.coords, coords[[0, 1, 4]])
    np.testing.assert_allclose(merged["zn"], [2.5, 2.0, 5.0])
    assert merged["rock"] == ["a", "b", "e"] and merged["n"].tolist() == [3, 1, 1]
    assert merged.crs == "EPSG:32718"
    counted = cs.duplicates(points, 0.5, merge="mean")
    assert np.average(counted["cu"], weights=counted["n"]) == pytest.approx(cu.mean())
    assert cs.duplicates(points, 0.5, merge="max")["zn"][0] == 3.0
    assert cs.duplicates(points, 0.5, merge="first")["zn"][0] == 1.0
    for bad in (
        lambda: cs.duplicates(coords, merge="mean"),
        lambda: cs.duplicates(points, merge="median"),
        lambda: cs.duplicates(points, merge="max", weights=np.ones(5)),
        lambda: cs.duplicates(coords, tolerance=-1.0),
    ):
        with pytest.raises(cs.InvalidInput):
            bad()
