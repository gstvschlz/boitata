import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(7)


def test_describe_skips_nan_and_matches_hazen_quantiles():
    v = rng.lognormal(0, 1, 200)
    s = bt.describe(np.r_[v, np.nan], quantiles=[0.1, 0.5, 0.9])
    assert s["n"] == 200
    assert s["mean"] == pytest.approx(v.mean())
    assert s["std"] == pytest.approx(v.std())
    np.testing.assert_allclose(
        [s["P10"], s["P50"], s["P90"]], np.quantile(v, [0.1, 0.5, 0.9], method="hazen")
    )
    w = rng.uniform(0.5, 2, 200)
    assert bt.describe(v, weights=w)["mean"] == pytest.approx(np.average(v, weights=w))
    with pytest.raises(bt.InvalidInput):
        bt.describe(v, weights=-w)


def test_swath_along_azimuth_and_axis():
    xy = rng.uniform(0, 100, (500, 2))
    s = bt.swath(xy, xy[:, 0], 10.0, azimuth=90)
    assert s.column_names == ["center", "n", "mean", "tonnage", "metal"]
    np.testing.assert_allclose(s["mean"], s["center"], atol=1.5)
    assert s["n"].sum() == 500
    np.testing.assert_array_equal(bt.swath(xy, xy[:, 0], 10.0, axis="x")["n"], s["n"])
    with pytest.raises(bt.InvalidInput):
        bt.swath(xy, xy[:, 0], 10.0)
    with pytest.raises(TypeError):
        bt.swath(xy, xy[:, 0], 10.0, 90)


def test_swath_tonnage_and_metal_balance_grade_tonnage():
    xy, v, w = rng.uniform(0, 100, (300, 2)), rng.lognormal(0, 1, 300), rng.uniform(1, 2, 300)
    s = bt.swath(xy, v, 10.0, axis="y", weights=w, density=2.7)
    gt = bt.grade_tonnage(v, [-np.inf], weights=w, density=np.full(300, 2.7))
    assert s["tonnage"].sum() == pytest.approx(gt["tonnage"][0])
    assert s["metal"].sum() == pytest.approx(gt["metal"][0])
    blocks = bt.BlockModel((0, 0, 0), (5, 5, 5), (20, 15, 1), attributes={"v": v})
    s = bt.swath(blocks, "v", 10.0, axis="x", density=2.7)
    gt = bt.grade_tonnage("v", [-np.inf], density=2.7, data=blocks)
    assert s["tonnage"].sum() == pytest.approx(gt["tonnage"][0]) == pytest.approx(300 * 125 * 2.7)
    assert s["metal"].sum() == pytest.approx(gt["metal"][0])
    by_array = bt.swath(blocks.coords, v, 10.0, axis="x", weights=np.full(300, 125.0), density=2.7)
    for c in s.column_names:
        np.testing.assert_allclose(s[c], by_array[c])


def test_validate_model_equal_to_data():
    v = rng.lognormal(0, 1, 200)
    d = np.where(np.arange(200) < 80, "ox", "fr")
    blocks = bt.BlockModel((0, 0, 0), (10, 10, 10), (200, 1, 1), attributes={"g": v, "rock": d})
    samples = bt.PointSet(blocks.coords, {"v": v, "w": np.ones(200), "rock": d})
    t = bt.validate_model(blocks, "g", samples, "v", weights="w", domain_column="rock", reference="g")
    assert list(t["domain"])[::4] == ["fr", "ox", "all"]
    assert list(t["source"])[:4] == ["naive", "declustered", "model", "reference"]
    model = np.asarray(t["source"]) != "naive"
    np.testing.assert_allclose(np.asarray(t["mean_diff"])[model], 0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(t["variance_ratio"])[model], 1)
    assert t["tonnage"][-2] == pytest.approx(200 * 1000)
    arrays = bt.validate_model(
        blocks, v, samples, v, weights=np.ones(200), domain_column=("rock", "rock"), reference=v
    )
    for c in t.column_names[2:]:
        np.testing.assert_array_equal(arrays[c], t[c])
    plain = bt.validate_model(blocks, v * 1.1, samples, "v", density=2.0)
    assert list(plain["source"]) == ["naive", "model"]
    assert plain["mean_diff"][1] == pytest.approx(0.1)
    assert plain["tonnage"][1] == pytest.approx(200 * 1000 * 2)
    with pytest.raises(TypeError):
        bt.validate_model(blocks, "g", samples, "v", "w")
    with pytest.raises(bt.InvalidInput, match="BlockModel"):
        bt.validate_model(samples, "v", samples, "v")


def test_contact_signs_distance_by_side():
    z = np.tile(np.arange(20.0), 2)
    coords = np.c_[np.zeros(40), np.zeros(40), z]
    domains = np.where(z < 10, "ore", "waste")
    values = np.where(z < 10, 3.0, 1.0)
    holes = np.repeat(["A", "B"], 20)
    c = bt.contact(
        coords,
        values,
        domains=domains,
        holes=holes,
        inside="ore",
        outside="waste",
        max_distance=50.0,
        bin=2.0,
    )
    assert c.column_names == ["distance", "mean", "n"]
    assert c["n"].sum() == 40
    np.testing.assert_array_equal(c["mean"], np.where(c["distance"] < 0, 3.0, 1.0))
    edge = bt.contact(
        coords, values, domains=domains, holes=holes, inside="ore", outside="waste", max_distance=8.0, bin=1.0
    )
    assert edge["n"].sum() == 32 and np.abs(edge["distance"]).max() == 7.5
    with pytest.raises(bt.InvalidInput):
        bt.contact(
            coords,
            values,
            domains=domains,
            holes=holes,
            inside="ore",
            outside="oxide",
            max_distance=50.0,
            bin=2.0,
        )


def test_soft_boundary_folds_in_nearby_other_domain_samples():
    z = np.arange(20.0)
    coords = np.c_[np.zeros(20), np.zeros(20), z]
    domains = np.where(z < 10, "ore", "waste")
    added, t = bt.soft_boundary(coords, z, domains=domains, target="ore", buffer=1.5, quantiles=[0.5])
    assert added.dtype == np.bool_
    assert list(added) == [i == 0 for i in range(10)]
    assert t.column_names == ["kind", "n", "mean", "variance", "std", "cv", "min", "max", "P50"]
    assert list(t["kind"]) == ["hard", "soft"]
    assert t["n"][0] == 10
    assert t["n"][1] == 10 + added.sum()
    assert t["mean"][0] == pytest.approx(z[:10].mean())
    assert t["mean"][1] == pytest.approx(np.r_[z[:10], 10.0].mean())
    with pytest.raises(bt.InvalidInput):
        bt.soft_boundary(coords, z, domains=domains, target="ore", buffer=0.0)
    with pytest.raises(bt.InvalidInput):
        bt.soft_boundary(coords, z, domains=domains, target="oxide", buffer=1.5)


def test_capping_at_maximum_removes_nothing():
    v = rng.lognormal(0, 1, 300)
    c = bt.capping(v, caps=[v.max(), np.median(v)])
    assert c["metal_removed"][0] == pytest.approx(0.0)
    assert c["fraction"][1] == pytest.approx(0.5)
    assert c["mean"][1] == pytest.approx(np.minimum(v, np.median(v)).mean())
    assert len(bt.capping(v)) == 6


def test_h_scatter_correlation_decays():
    x = np.arange(300.0)
    coords = np.c_[x, np.zeros(300)]
    z = np.sin(x / 15)
    r = [bt.h_scatter(coords, z, lag, 0.5)[2] for lag in (1, 10, 20)]
    assert r[0] > 0.99 and r[0] > r[1] > r[2]
    head, tail, _ = bt.h_scatter(coords, z, 1.0, 0.1, azimuth=90, other=2 * z)
    np.testing.assert_allclose(tail, 2 * np.sin((x[:-1]) / 15))
    assert len(head) == 299


def test_correlation_pairwise():
    a = rng.normal(size=100)
    data = np.c_[a, 2 * a + 1, a**3]
    data[5, 1] = np.nan
    r = bt.correlation(data)
    np.testing.assert_allclose(r[0, 1], 1.0)
    np.testing.assert_allclose(r, r.T)
    assert bt.correlation(data, method="spearman")[0, 2] == pytest.approx(1.0)
    w = rng.uniform(0.5, 2.0, 100)
    c = bt.correlation(data[:, :2], weights=w, method="covariance")
    assert c[0, 0] == pytest.approx(bt.describe(a, weights=w)["variance"])
    ok = ~np.isnan(data[:, 1])
    assert c[1, 1] == pytest.approx(4 * bt.describe(a[ok], weights=w[ok])["variance"])
    with pytest.raises(bt.InvalidInput):
        bt.correlation(data, method="kendall")


def test_describe_by_ends_with_the_all_data_row():
    v = np.r_[rng.lognormal(0, 1, 90), np.nan]
    c = np.repeat(["b", "a", "c"], 31)[:91]
    w = rng.uniform(0.5, 2, 91)
    t = bt.describe_by(v, c, weights=w, quantiles=[0.5, 0.975])
    assert t.column_names == ["category", "n", "mean", "variance", "std", "cv", "min", "max", "P50", "P97.5"]
    assert list(t["category"]) == ["a", "b", "c", "all"]
    whole = bt.describe(v, weights=w, quantiles=[0.5, 0.975])
    assert t["n"][-1] == whole["n"] == t["n"][:3].sum()
    assert t["mean"][-1] == pytest.approx(whole["mean"])
    np.testing.assert_allclose([t["P50"][-1], t["P97.5"][-1]], [whole["P50"], whole["P97.5"]])
    a = c == "a"
    assert t["mean"][0] == pytest.approx(np.average(v[a], weights=w[a]))


def test_grade_tonnage_from_data():
    v = rng.lognormal(0, 1, 200)
    volume, density = rng.uniform(1, 2, 200), rng.uniform(2.5, 3, 200)
    cutoffs = [-np.inf, 0.5, 1, 2, 5, 1e9]
    gt = bt.grade_tonnage(v, cutoffs, weights=volume, density=density)
    tonnes = volume * density
    assert gt["tonnage"][0] == pytest.approx(tonnes.sum())
    assert gt["mean_grade"][0] == pytest.approx(np.average(v, weights=tonnes))
    assert np.all(np.diff(gt["tonnage"]) <= 0)
    np.testing.assert_allclose(gt["metal"][:-1], gt["tonnage"][:-1] * gt["mean_grade"][:-1])
    assert gt["tonnage"][-1] == 0 and np.isnan(gt["mean_grade"][-1])
    assert bt.grade_tonnage(v, [2.0])["tonnage"][0] == (v >= 2).sum()
    points = bt.PointSet(rng.uniform(0, 1, (200, 3)), {"v": v, "w": volume, "d": density})
    named = bt.grade_tonnage("v", cutoffs, weights="w", density="d", data=points)
    np.testing.assert_array_equal(named["metal"], gt["metal"])
    with pytest.raises(TypeError):
        bt.grade_tonnage(v, [2.0], volume)


def test_compare_models_reference_scaling_and_metal_balance():
    v = rng.lognormal(0, 1, 300)
    classes = np.repeat(["measured", "indicated", "inferred"], 100)
    cutoffs = [-np.inf, 1.0, 3.0]
    columns = {"kriged": v, "scaled": 1.25 * v, "same": v, "class": classes}
    blocks = bt.BlockModel((0, 0, 0), (5, 5, 5), (300, 1, 1), attributes=columns)
    t = bt.compare_models(blocks, ["kriged", "scaled", "same"], cutoffs, categories="class", density=3.0)
    assert t.column_names[:3] == ["category", "cutoff", "model"]
    assert t.num_rows == 4 * 3 * 3
    assert list(t["category"][::9]) == ["indicated", "inferred", "measured", "all"]
    model = np.asarray(t["model"])
    for name in ("kriged", "same"):
        for diff in ("tonnage_diff", "grade_diff", "metal_diff"):
            np.testing.assert_allclose(np.asarray(t[diff])[model == name], 0, atol=1e-12)
    first = (model == "scaled") & (np.asarray(t["cutoff"]) == -np.inf)
    np.testing.assert_allclose(np.asarray(t["metal_diff"])[first], 0.25)
    kriged = t.filter(model == "kriged")
    gt = bt.grade_tonnage(v, cutoffs, weights=np.full(300, 125.0), density=3.0, categories=classes)
    np.testing.assert_allclose(kriged["metal"], gt["metal"])
    by_name = bt.grade_tonnage("kriged", cutoffs, density=3.0, categories="class", data=blocks)
    np.testing.assert_array_equal(by_name["metal"], gt["metal"])
    metal = np.asarray(gt["metal"]).reshape(4, 3)
    np.testing.assert_allclose(metal[:3].sum(axis=0), metal[3])
    arrays = bt.compare_models(
        blocks, {"kriged": v, "scaled": "scaled", "same": v}, cutoffs, categories=classes
    )
    np.testing.assert_allclose(arrays["metal"], t["metal"] / 3.0)
    assert list(bt.compare_models(blocks, {"a": v, "b": "same"}, [0.0], reference="b")["model"]) == ["a", "b"]
    with pytest.raises(bt.InvalidInput):
        bt.compare_models(blocks, {"a": v}, [0.0], reference="b")
    with pytest.raises(bt.InvalidInput):
        bt.compare_models(blocks, {"a": v, "b": v[1:]}, [0.0])
    with pytest.raises(TypeError):
        bt.compare_models(blocks, ["kriged"], [0.0], classes)


def test_capping_report_by_domain():
    v = rng.lognormal(0, 1, 300)
    d = np.repeat([1, 2, 3], 100)
    w = rng.uniform(0.5, 2, 300)
    caps = {1: 3.0, 3: 5.0}
    t = bt.capping_report(v, caps, domains=d, weights=w)
    assert list(t["domain"]) == ["1", "2", "3", "all"]
    cap = np.select([d == 1, d == 3], [3.0, 5.0], np.inf)
    removed = np.maximum(v - cap, 0) * w
    np.testing.assert_allclose(
        t["metal_removed"], [removed[d == k].sum() for k in (1, 2, 3)] + [removed.sum()]
    )
    np.testing.assert_allclose(t["cap"], [3.0, np.nan, 5.0, np.nan])
    assert t["n_capped"][-1] == (v > cap).sum()
    assert t["max_capped"][0] == 3.0 and t["max_capped"][1] == t["max"][1]
    assert t["mean_capped"][-1] == pytest.approx(np.average(np.minimum(v, cap), weights=w))
    with pytest.raises(bt.InvalidInput):
        bt.capping_report(v, {4: 1.0}, domains=d)


def test_capping_transform_per_domain():
    d = np.repeat(["a", "b", "c"], 200)
    v = rng.lognormal(0, 1, 600) * np.repeat([1.0, 3.0, 10.0], 200)
    w = rng.uniform(0.5, 2, 600)
    capping = bt.Capping(quantile=0.95)
    capped = capping.fit_transform(v, domains=d, weights=w)
    caps = capping.caps_
    assert len(set(caps.values())) == 3
    for k in "abc":
        assert caps[k] == pytest.approx(bt.describe(v[d == k], weights=w[d == k], quantiles=[0.95])["P95"])
    cap = np.array([caps[k] for k in d])
    assert (capped <= cap).all() and (capped[v <= cap] == v[v <= cap]).all()
    t = bt.capping_report(v, caps, domains=d, weights=w)
    np.testing.assert_allclose(
        list(capping.metal_removed_.values()), 1 - t["mean_capped"][:3] / t["mean"][:3]
    )
    np.testing.assert_array_equal(capping.transform(v, domains=d), capped)
    points = bt.PointSet(rng.uniform(0, 100, (600, 3)), {"v": v, "rock": d, "w": w})
    by_name = bt.Capping(quantile=0.95).fit("v", domain_column="rock", weights="w", data=points)
    assert by_name.caps_ == caps
    assert np.isnan(capping.transform([np.nan], domains=["a"])[0])
    assert capping.transform([1e9], domains=["z"])[0] == 1e9
    with pytest.raises(bt.InvalidInput):
        capping.transform(v)


def test_capping_rules_and_pipeline():
    v = rng.lognormal(0, 1, 500)
    m = bt.Capping(metal_removed=0.05).fit(v)
    assert m.metal_removed_ == pytest.approx(0.05)
    c = bt.Capping(cv=0.8).fit(v)
    assert bt.capping(v, caps=[c.caps_])["cv"][0] == pytest.approx(0.8)
    assert bt.Capping(cap=2.0).fit(v).caps_ == 2.0
    capping, scores = bt.Capping(quantile=0.99), bt.NormalScore()
    y = scores.fit_transform(capping.fit_transform(v))
    below = v < capping.caps_
    np.testing.assert_allclose(scores.transform(capping.transform(v))[below], y[below])
    assert scores.inverse_transform(y).max() <= capping.caps_ + 1e-9
    for bad in ({}, {"cap": 1.0, "cv": 1.0}, {"cap": np.inf}):
        with pytest.raises(bt.InvalidInput):
            bt.Capping(**bad)
    with pytest.raises(bt.InvalidInput):
        bt.Capping(cap={"a": 1.0}).fit(v)


def test_pairs_recover_twins_and_their_bias():
    a = np.c_[np.arange(0.0, 500.0, 25.0), np.zeros(20)]
    order = rng.permutation(20)
    b = bt.PointSet(a[order] + rng.uniform(-1, 1, (20, 2)))
    za = rng.uniform(1, 5, 20)
    zb = 1.1 * za[order]
    p = bt.pairs(a, b, 3.0, values=(za, zb))
    np.testing.assert_array_equal(order[p["b"].astype(int)], p["a"])
    np.testing.assert_allclose(p["value_b"], 1.1 * p["value_a"])
    bias = bt.paired_bias(p, bins=[0, 1, 2])
    assert bias["n"].sum() == 20
    np.testing.assert_allclose(bias["bias"], 0.1)
    assert len(bt.pairs(a, b, 3.0, holes=(np.arange(20), order))) == 0
    with pytest.raises(bt.InvalidInput):
        bt.pairs(a, b, 3.0, values=za)


def test_duplicates_report_group_ids_and_merge():
    coords = np.array([[0, 0, 0], [5, 0, 0], [0, 0, 0], [0.4, 0.3, 0], [9, 9, 9]], dtype=float)
    report, group = bt.duplicates(coords)
    np.testing.assert_array_equal(group, [0, -1, 0, -1, -1])
    assert report["n"].tolist() == [2] and report["spread"].tolist() == [0.0]
    report, group = bt.duplicates(coords, tolerance=0.5)
    np.testing.assert_array_equal(group, [0, -1, 0, 0, -1])
    assert report["spread"][0] == pytest.approx(0.5)

    zn = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    cu = np.arange(1.0, 6.0)
    rock = ["a", "b", "c", "d", "e"]
    points = bt.PointSet(coords, {"zn": zn, "cu": cu, "rock": rock}, crs="EPSG:32718")
    merged = bt.duplicates(points, tolerance=0.5, merge="mean", weights=[1, 1, 3, 1, 1])
    np.testing.assert_allclose(merged.coords, coords[[0, 1, 4]])
    np.testing.assert_allclose(merged["zn"], [2.5, 2.0, 5.0])
    assert list(merged["rock"]) == ["a", "b", "e"] and merged["n"].tolist() == [3, 1, 1]
    assert merged.crs == "EPSG:32718"
    counted = bt.duplicates(points, tolerance=0.5, merge="mean")
    assert np.average(counted["cu"], weights=counted["n"]) == pytest.approx(cu.mean())
    assert bt.duplicates(points, tolerance=0.5, merge="max")["zn"][0] == 3.0
    assert bt.duplicates(points, tolerance=0.5, merge="first")["zn"][0] == 1.0
    for bad in (
        lambda: bt.duplicates(coords, merge="mean"),
        lambda: bt.duplicates(points, merge="median"),
        lambda: bt.duplicates(points, merge="max", weights=np.ones(5)),
        lambda: bt.duplicates(coords, tolerance=-1.0),
    ):
        with pytest.raises(bt.InvalidInput):
            bad()


def test_data_spacing_of_a_square_grid_is_its_spacing():
    g = np.arange(0.0, 500.0, 25.0)
    xy = np.array([(x, y) for x in g for y in g])
    hole = np.repeat(np.arange(len(xy)), 50)
    composites = np.c_[np.repeat(xy, 50, axis=0), np.tile(np.arange(1.0, 100.0, 2.0), len(xy))]
    targets = np.c_[rng.uniform(150, 350, (100, 2)), np.full(100, 50.0)]
    volume = bt.data_spacing(targets, composites, bt.Search(60.0), composite_length=2.0)
    assert np.mean(volume) == pytest.approx(25.0, rel=0.03)
    plan = bt.data_spacing(targets, composites, None, holes=hole)
    np.testing.assert_array_equal(plan, bt.data_spacing(targets, xy, None))
    assert np.mean(plan) == pytest.approx(25.0, rel=0.05)
    data = bt.PointSet(composites, {"hole": hole})
    np.testing.assert_array_equal(bt.data_spacing(targets, data, None, holes="hole"), plan)
    bm = bt.BlockModel(origin=(-50.0, 100.0), size=(50.0, 50.0), count=(2, 1))
    assert np.isnan(bt.data_spacing(bm, xy, None, hull=True)).tolist() == [True, False]
    assert np.isnan(bt.data_spacing(targets, xy[:5], None)).all()
    with pytest.warns(UserWarning, match="fewer than 4 composites"):
        bt.data_spacing(targets, composites, bt.Search(5.0), composite_length=2.0)
    with pytest.raises(bt.InvalidInput, match="composite_length"):
        bt.data_spacing(targets, composites, bt.Search(60.0))
    with pytest.raises(bt.InvalidInput):
        bt.data_spacing(targets, xy, None, n=0)


def test_data_spacing_takes_the_composite_length_of_drillholes():
    collar = {"HOLE_ID": ["a", "b"], "X": [0.0, 30.0], "Y": [0.0, 0.0], "Z": [0.0, 0.0]}
    survey = {"HOLE_ID": ["a", "b"], "DEPTH": [0.0, 0.0], "AZIMUTH": [0.0, 0.0], "DIP": [90.0, 90.0]}
    edges = np.arange(0.0, 41.0, 4.0)
    intervals = {
        "HOLE_ID": ["a"] * 10 + ["b"] * 10,
        "FROM": np.tile(edges[:-1], 2),
        "TO": np.tile(edges[1:], 2),
    }
    dh = bt.Drillholes(collar, survey, intervals)
    at = np.array([[0.0, 0.0, -20.0]])
    given = bt.data_spacing(at, dh.samples(), bt.Search(15.0), composite_length=4.0)
    assert bt.data_spacing(at, dh, bt.Search(15.0)) == pytest.approx(given)


def test_domain_change_margins_and_metal():
    before = rng.integers(0, 3, 400).astype(float)
    after = np.where(rng.random(400) < 0.25, rng.integers(0, 3, 400), before)
    volume, grade = rng.uniform(1, 2, 400), rng.uniform(0, 3, 400)
    scheme = bt.Categories(["a", "b", "c"])
    t = bt.domain_change(before, after, weights=volume, grades=grade, scheme=scheme)
    assert t.column_names == ["from", "to", "tonnage", "mean_grade", "metal"]
    assert list(t["from"])[:3] == ["a", "a", "a"] and list(t["to"])[:3] == ["a", "b", "c"]
    cells = np.asarray(t["tonnage"]).reshape(3, 3)
    for c in range(3):
        assert np.isclose(cells[c].sum(), volume[before == c].sum())
        assert np.isclose(cells[:, c].sum(), volume[after == c].sum())
    assert np.isclose(np.trace(cells), volume[before == after].sum())
    assert np.isclose(np.sum(t["metal"]), np.sum(volume * grade))
    labels = bt.domain_change(scheme.decode(before), scheme.decode(after), weights=volume)
    np.testing.assert_allclose(np.asarray(labels["tonnage"]), cells.ravel())
    points = bt.PointSet(rng.uniform(0, 1, (400, 3)), {"old": before, "new": after})
    assert np.isclose(np.sum(bt.domain_change("old", "new", scheme=scheme, data=points)["tonnage"]), 400)
    with pytest.raises(bt.InvalidInput):
        bt.domain_change(before + 3, after, scheme=scheme)


def test_transition_matrix_counts_and_frequency():
    depth = np.tile(np.arange(5, dtype=float), 2)
    categories = np.tile([0, 0, 1, 1, 2], 2)
    holes = np.repeat([0, 1], 5)
    scheme = bt.Categories(["a", "b", "c"])
    t = bt.transition_matrix(depth, categories, holes, lag=1.0, scheme=scheme)
    assert t.column_names == ["from", "to", "count", "frequency"]
    assert list(t["from"])[:3] == ["a", "a", "a"] and list(t["to"])[:3] == ["a", "b", "c"]
    counts = np.asarray(t["count"]).reshape(3, 3)
    np.testing.assert_array_equal(counts, [[2, 2, 0], [0, 2, 2], [0, 0, 0]])
    freq = np.asarray(t["frequency"]).reshape(3, 3)
    np.testing.assert_allclose(freq[:2], counts[:2] / counts[:2].sum(axis=1, keepdims=True))
    assert np.isnan(freq[2]).all()
    labels = bt.transition_matrix(depth, scheme.decode(categories), holes, lag=1.0)
    np.testing.assert_array_equal(np.asarray(labels["count"]), counts.ravel())
    data = {"depth": depth, "lith": categories, "hole": holes}
    same = bt.transition_matrix("depth", "lith", "hole", lag=1.0, scheme=scheme, data=data)
    np.testing.assert_array_equal(np.asarray(same["count"]), counts.ravel())
    with pytest.raises(bt.InvalidInput):
        bt.transition_matrix(depth, categories, holes, lag=0.0, scheme=scheme)
    with pytest.raises(bt.InvalidInput):
        bt.transition_matrix(depth, categories + 3, holes, lag=1.0, scheme=scheme)


def test_uncertainty_curve_bins_spacing_and_finds_the_required_spacing():
    spacing = rng.uniform(5, 30, 2000)
    error = 0.01 * spacing * rng.uniform(0.5, 1.5, 2000)
    t = bt.Table({"ds": np.r_[spacing, np.nan], "mee": np.r_[error, 0.1]})
    curve = bt.uncertainty_curve("ds", "mee", bins=np.arange(5, 31, 5.0), data=t)
    assert curve.column_names == ["spacing", "n", "P50", "P90", "share"]
    assert curve["n"].sum() == 2000
    for q in ("P50", "P90"):
        assert np.all(np.diff(curve[q]) >= 0)
    assert np.all(curve["P90"] >= curve["P50"])
    np.testing.assert_allclose(curve["share"][0], (error <= 0.15)[spacing < 10].mean())
    required = bt.required_spacing(curve, threshold=0.15)
    assert curve["spacing"][0] < required < curve["spacing"][-1]
    with pytest.warns(UserWarning, match="beyond the tested range"):
        assert bt.required_spacing(curve, column="P50", threshold=1.0) == curve["spacing"][-1]
    assert len(bt.uncertainty_curve(spacing, error)) > 1
    assert len(bt.uncertainty_curve(spacing, error, bins=3, min_count=10_000)) == 0
    with pytest.raises(bt.InvalidInput):
        bt.uncertainty_curve(spacing, error[:-1])


def test_required_spacing_interpolates_a_linear_curve():
    curve = bt.Table({"spacing": [10.0, 20.0, 30.0], "P90": [0.1, 0.2, 0.3]})
    assert bt.required_spacing(curve) == pytest.approx(15.0)
    assert bt.required_spacing(curve, threshold=0.1) == 10.0
    assert np.isnan(bt.required_spacing(curve, threshold=0.05))
    with pytest.warns(UserWarning):
        assert bt.required_spacing(curve, threshold=0.3) == 30.0
    with pytest.raises(bt.InvalidInput):
        bt.required_spacing(curve, column="P50")
