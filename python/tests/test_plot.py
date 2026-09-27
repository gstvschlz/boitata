import ceres as cs
import numpy as np
import pytest

pytest.importorskip("matplotlib").use("Agg")
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
    s = cs.describe(v, weights=w, quantiles=[0.25, 0.5, 0.75])
    np.testing.assert_allclose(y, [s["P25"], s["P50"], s["P75"]])
    assert np.all(y > x)


def test_boxplot_quartiles_are_describe_quantiles():
    v = rng.lognormal(0, 1, 300)
    domain = np.where(np.arange(300) < 100, "b", "a")
    w = rng.uniform(0.5, 2.0, 300)
    _, ax = cs.plot.boxplot(v, domain, weights=w, sort=True)
    stats = [cs.describe(v[domain == d], weights=w[domain == d]) for d in ("a", "b")]
    order = np.argsort([s["P50"] for s in stats])
    for box, i in zip(ax.patches, order, strict=True):
        y = box.get_path().vertices[:, 1]
        assert (y.min(), y.max()) == pytest.approx((stats[i]["P25"], stats[i]["P75"]))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[order.tolist().index(0)] == "a\nn = 200"
    _, ax = cs.plot.boxplot(v, domain)
    assert ax.patches[0].get_path().vertices[:, 1].max() != pytest.approx(stats[0]["P75"])


def test_variogram_with_anisotropic_model():
    xy = rng.uniform(0, 100, (200, 2))
    exp = cs.experimental_variogram(xy, rng.normal(size=200), 10.0, 60.0, azimuth=30)
    model = cs.Variogram([("spherical", 1.0, 40.0)], rotation=(30, 0, 0), ratios=(0.5, 1.0))
    _, ax = cs.plot.variogram(exp, variogram=model, direction=(120, 0))
    h, g = ax.lines[0].get_data()
    assert g[np.searchsorted(h, 20.0)] == pytest.approx(1.0, abs=1e-9)


def test_variogram_matrix():
    xy = rng.uniform(0, 100, (200, 2))
    a = rng.normal(size=200)
    vs = cs.experimental_variograms(
        xy, [a, a + rng.normal(size=200)], 10.0, 60.0, directions=[(0, 0), (90, 0)]
    )
    lmc = cs.Coregionalization.fit(vs)
    _, axes = cs.plot.variograms(vs, model=lmc, labels=["A", "B"])
    assert axes.shape == (2, 2) and not axes[1, 0].axison
    assert axes[0, 1].get_title() == "A × B" and len(axes[0, 1].lines) == 3


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


def test_row_at_finds_rotated_and_masked_blocks():
    bm = cs.BlockModel(origin=(10, 20, 0), size=(2, 2, 1), count=(4, 3, 2), rotation=(30, 0, 0))
    bm = bm.mask(np.arange(24) != 5)
    rows = bm.row_at(np.vstack([bm.centroids, [[0, 0, 0]]]))
    assert rows.dtype == np.int64 and (rows == [*range(23), -1]).all()


def test_section_on_a_plane_matches_the_axis_slice():
    bm = cs.BlockModel(origin=(0, 0, 0), size=(2, 2, 1), count=(4, 3, 2))
    bm = bm.with_column("g", np.arange(24.0)).mask(np.arange(24) != 5)
    _, ax = cs.plot.section(bm, "g", plane=((4, 3, 0.5), 90, 0), resolution=0.5)
    np.testing.assert_allclose(ax.images[0].get_extent(), [0, 8, 0, 6], atol=1e-9)
    assert ax.get_xlabel() == "Easting (m)" and ax.get_ylabel() == "Northing (m)"
    _, axis = cs.plot.section(bm, "g", axis="z", index=0)
    image = np.ma.filled(ax.images[0].get_array(), np.nan)
    np.testing.assert_array_equal(image[::4, ::4], np.ma.filled(axis.images[0].get_array(), np.nan))
    _, ax = cs.plot.uncertain("g", np.zeros(23), model=bm, plane=((4, 3, 0.5), 45, 90))
    assert ax.get_xlabel() == "Along strike (m)" and ax.get_ylabel() == "Elevation (m)"


def test_slab_keeps_points_traces_and_clipped_lines():
    points = cs.PointSet(np.array([[0.0, 4, 0], [5, -4, 1], [9, 1, -2], [2, 0.5, 3]]), {"g": [1.0, 2, 3, 4]})
    cube = cs.convex_hull(
        np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    )
    line = np.array([[3.0, -10, 5], [3, 10, 5], [3, 10, 6]])
    _, ax = cs.plot.slab(
        points, "g", plane=((0, 0, 0), 90, 90), thickness=2, meshes=cube, lines=[line], labels=list("aaba")
    )
    np.testing.assert_allclose(ax.collections[-1].get_offsets(), [[9, -2], [2, 3]])
    trace = np.concatenate(ax.collections[0].get_segments())
    np.testing.assert_allclose(np.abs(trace).max(axis=1), 1)
    np.testing.assert_allclose(ax.collections[1].get_segments()[0], [[3, 5], [3, 5]])
    assert [t.get_text() for t in ax.texts] == ["b", "a"]
    assert ax.get_xlabel() == "Easting (m)" and ax.get_ylabel() == "Elevation (m)"


def test_swath_draws_each_result():
    xy = rng.uniform(0, 100, (300, 2))
    s = cs.swath(xy, xy[:, 0], 10.0, axis="x")
    _, ax = cs.plot.swath([s, s], labels=["a", "b"])
    assert len(ax.lines) == 2
    _, ax = cs.plot.swath(s, y="metal")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), s["metal"])


def test_paired_bias_draws_bias_and_counts():
    a = rng.uniform(0, 100, (50, 2))
    p = cs.pairs(a, a + 0.5, 1.0, values=(np.ones(50), np.full(50, 1.2)))
    _, ax = cs.plot.paired_bias(cs.paired_bias(p, [0, 1]))
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), 20.0)


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
    _, ax = cs.plot.uncertain("g", uncertainty, model=bm, axis="z", index=0)
    rgba = ax.images[0].get_array()
    assert rgba.shape == (3, 4, 4) and ax.images[0].get_extent() == [0, 8, 0, 6]
    np.testing.assert_allclose(rgba[1, 2, :3], 1.0)
    assert rgba[1, 1, 3] == 0 and rgba[1, 3, :3].max() < 1


def test_uncertain_draws_category_codes_with_a_legend():
    scheme = cs.Categories(["a", "b", "c"], colors=["red", "green", "blue"])
    codes = np.array([[0.0, 1.0], [2.0, np.nan]])
    entropy = np.array([[0.0, 1.0], [0.0, 0.0]])
    _, ax = cs.plot.uncertain(codes, entropy, scheme=scheme)
    rgba = ax.images[0].get_array()
    np.testing.assert_allclose(rgba[0, 0], [1, 0, 0, 1])
    np.testing.assert_allclose(rgba[0, 1, :3], 1.0)
    np.testing.assert_allclose(rgba[1, 0], [0, 0, 1, 1])
    assert rgba[1, 1, 3] == 0
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b", "c"]


def test_probability_draws_the_cap():
    v = rng.lognormal(0, 1, 300)
    _, ax = cs.plot.probability(v, log=True, cap=5.0)
    assert ax.lines[-1].get_xdata()[0] == 5.0


def test_probability_fences_are_tukeys():
    v = rng.normal(10, 1, 2000)
    _, ax = cs.plot.probability(v, fences=1.5)
    s = cs.describe(v, quantiles=[0.25, 0.75])
    q1, q3 = s["P25"], s["P75"]
    low, high = (line.get_xdata()[0] for line in ax.lines[1:])
    assert (low, high) == pytest.approx((q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)))
    assert ax.lines[2].get_label() == f"fence {high:.3g}, {np.sum(v > high)} beyond"
    _, ax = cs.plot.probability(np.exp(v / 4), log=True, fences=3.0)
    assert len(ax.lines) == 1


def test_stats_box_writes_describe():
    v = rng.lognormal(0, 1, 300)
    w = rng.uniform(0.5, 2.0, 300)
    s = cs.describe(v, weights=w)
    _, ax = cs.plot.histogram(v, weights=w, stats=True)
    assert f"mean {s['mean']:.3g}" in " ".join(ax.texts[0].get_text().split())
    _, ax = cs.plot.cdf([v, v], weights=[None, w], labels=["naive", "declustered"], stats=True)
    header, n, mean = ax.texts[0].get_text().splitlines()[:3]
    assert header.split() == ["naive", "declustered"] and n.split() == ["n", "300", "300"]
    assert mean.split()[2] == f"{s['mean']:.3g}"


def test_correlation_heatmap_and_covariance():
    a = rng.normal(size=200)
    data = {"a": a, "b": 3 * a, "c": rng.normal(size=200)}
    _, ax = cs.plot.correlation(data, method="covariance")
    c = ax.images[0].get_array()
    assert c[1, 1] == pytest.approx(9 * cs.describe(a)["variance"]) and ax.images[0].norm.vmax == c[1, 1]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    _, ax = cs.plot.correlation(data, method="spearman", colorbar=False)
    assert ax.texts[1].get_text() == "1.00" and ax.images[0].norm.vmin == -1


def test_declustering_marks_the_chosen_size():
    xy = np.r_[rng.uniform(0, 100, (100, 2)), rng.uniform(0, 10, (100, 2))]
    v = np.r_[np.ones(100), np.full(100, 3.0)]
    d = cs.cell_declustering(xy, v, sizes=np.arange(5.0, 55.0, 5.0))
    _, ax = cs.plot.declustering(d, naive=v.mean())
    np.testing.assert_allclose(ax.lines[0].get_ydata(), d.means)
    assert ax.lines[-1].get_xdata()[0] == d.cell_size and d.mean < v.mean()
    with pytest.raises(cs.InvalidInput, match="scan"):
        cs.plot.declustering(cs.cell_declustering(xy, v, cell_size=10.0))


def test_conditional_mean_follows_a_linear_relation():
    x = rng.uniform(0, 10, 5000)
    y = 2 * x + rng.normal(0, 1, 5000)
    _, ax = cs.plot.conditional(x, y, bins=8)
    cx, mean = ax.lines[0].get_data()
    assert len(cx) == 8
    np.testing.assert_allclose(mean, 2 * cx, atol=0.1)


def test_completeness_counts_rows_by_variables_present():
    data = rng.normal(size=(50, 3))
    data[:5, 0] = np.nan
    data[:2, 1] = np.nan
    _, ax = cs.plot.completeness(data)
    assert [p.get_height() for p in ax.patches] == [0, 2, 3, 45]


def test_scatter_matrix_annotates_correlations_and_weights_histograms():
    x = rng.lognormal(0, 1, 300)
    y = x * rng.lognormal(0, 0.3, 300)
    z = rng.normal(0, 1, 300)
    x[5] = np.nan
    w = rng.uniform(0.5, 2.0, 300)
    fig, axes = cs.plot.scatter_matrix({"x": x, "y": y, "z": z}, weights=w, log=[True, True, False])
    assert axes.shape == (3, 3) and axes[2, 0].get_xlabel() == "x" and axes[1, 0].get_ylabel() == "y"
    assert axes[1, 0].get_xscale() == "log" and axes[1, 0].get_yscale() == "log"
    assert axes[2, 2].get_xscale() == "linear"
    columns = np.column_stack([x, y, z])
    r = cs.correlation(columns, weights=w)
    rank = cs.correlation(columns, weights=w, method="spearman")
    assert axes[1, 0].texts[0].get_text() == f"r {r[1, 0]:.2f}\nrank {rank[1, 0]:.2f}"
    bars = [a for a in fig.axes if not any(a is b for b in axes.flat)]
    np.testing.assert_allclose([sum(p.get_height() for p in b.patches) for b in bars], [1.0] * 3)
    _, grid = plt.subplots(2, 2)
    _, axes = cs.plot.scatter_matrix(columns[:, :2], labels=["a", "b"], axes=grid)
    assert axes[0, 1] is grid[0, 1] and axes[1, 1].get_xlabel() == "b"


def test_category_swath_stacks_to_one_and_follows_a_trend():
    xy = np.c_[np.arange(0.5, 100), np.zeros(100)]
    rock = np.where(xy[:, 0] < 30, "a", np.where(xy[:, 0] < 60, "b", "c"))
    _, ax = cs.plot.category_swath(xy, rock, 10.0, axis="x")
    heights = np.array([p.get_height() for p in ax.patches]).reshape(3, 10)
    np.testing.assert_allclose(heights.sum(axis=0), 1.0)
    np.testing.assert_allclose(heights[0], [1, 1, 1] + [0] * 7)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["c", "b", "a"]


def test_proportions_are_weighted_shares():
    rock = np.array(["a", "b", "b", "c"])
    w = np.array([2.0, 1.0, 1.0, 4.0])
    _, ax = cs.plot.proportions(rock, weights=w)
    np.testing.assert_allclose([p.get_width() for p in ax.patches], [0.25, 0.25, 0.5])
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.25, 0.5, 0.25])


def test_directions_project_the_major_axis():
    angles = [[90.0, 0.0, 0.0], [0.0, 60.0, 0.0]]
    la = cs.LocalAnisotropy([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], angles, [[1.0, 1.0]] * 2)
    _, ax = cs.plot.directions(la)
    q = ax.collections[0]
    np.testing.assert_allclose(np.c_[q.U, q.V], [[1, 0], [0, 0.5]], atol=1e-12)
    _, ax = cs.plot.directions(la, plane=((0, 0, 0), 0.0, 90.0), thickness=1.0)
    q = ax.collections[0]
    np.testing.assert_allclose(np.c_[q.U, q.V], [[0, 0]], atol=1e-12)
    assert ax.get_ylabel() == "Elevation (m)"


def _drawn(ax):
    parts = [line.get_xydata().ravel() for line in ax.lines]
    parts += [np.ravel(p.get_extents().bounds) for p in ax.patches]
    return np.concatenate(parts)


def test_names_resolve_against_data_like_arrays():
    v, w = rng.lognormal(0, 1, 100), rng.uniform(0.5, 2.0, 100)
    rock = np.where(np.arange(100) < 40, "a", "b")
    points = cs.PointSet(rng.uniform(0, 100, (100, 3)), {"v": v, "w": w, "rock": rock})
    for f, args, names in [
        (cs.plot.histogram, (v,), ("v",)),
        (cs.plot.probability, (v,), ("v",)),
        (cs.plot.cdf, (v,), ("v",)),
        (cs.plot.boxplot, (v, rock), ("v", "rock")),
        (cs.plot.proportions, (rock,), ("rock",)),
        (cs.plot.conditional, (v, w), ("v", "w")),
    ]:
        _, a = f(*args, weights=w)
        _, b = f(*names, weights="w", data=points)
        np.testing.assert_allclose(_drawn(a), _drawn(b), err_msg=f.__name__)
    _, a = cs.plot.qq(v, v**2, x_weights=w, y_weights=w)
    _, b = cs.plot.qq("v", v**2, x_weights="w", y_weights="w", data=points)
    np.testing.assert_allclose(a.lines[0].get_ydata(), b.lines[0].get_ydata())
    _, a = cs.plot.category_swath(points, "rock", 20.0, axis="x", weights="w")
    _, b = cs.plot.category_swath(points.coords, rock, 20.0, axis="x", weights=w)
    np.testing.assert_allclose(_drawn(a), _drawn(b))


def test_block_model_data_weights_by_volume():
    bm = cs.BlockModel.subblocked(
        origin=(0, 0, 0),
        size=(1, 1, 1),
        count=(2, 1, 1),
        parent=np.array([0, 1], dtype=np.uint64),
        extents=np.array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0.25, 1, 1.0]]),
        attributes={"g": [1.0, 2.0]},
    )
    _, ax = cs.plot.histogram("g", bins=[0.5, 1.5, 2.5], data=bm)
    np.testing.assert_allclose([p.get_height() for p in ax.patches], [0.8, 0.2])


def test_user_input_errors_are_invalid_input():
    with pytest.raises(cs.InvalidInput, match="container"):
        cs.plot.histogram("v")
    with pytest.raises(cs.MissingColumn, match="columns: v"):
        cs.plot.histogram("x", data={"v": np.ones(3)})
    with pytest.raises(cs.InvalidInput, match="codes"):
        cs.plot.proportions([0.0, 5.0], scheme=cs.Categories(["a", "b"]))
    bm = cs.BlockModel(origin=(0, 0, 0), size=(1, 1, 1), count=(2, 2, 2), attributes={"g": np.ones(8)})
    with pytest.raises(cs.InvalidInput, match="misses"):
        cs.plot.section(bm, "g", plane=((0, 0, 50), 90, 0))
    with pytest.raises(cs.InvalidInput, match="center"):
        cs.plot.slab(np.zeros((1, 3)), plane=((0, 0), 90, 90), thickness=1)
    with pytest.raises(TypeError):
        cs.plot.histogram(np.ones(3), np.ones(3))


def _twin(ax):
    return next(a for a in ax.figure.axes if a is not ax)


def test_grade_tonnage_draws_every_table_kind():
    v = rng.lognormal(0, 1, 400)
    cutoffs = np.linspace(0, 3, 7)
    gt = cs.grade_tonnage(v, cutoffs)
    _, ax = cs.plot.grade_tonnage(gt)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), gt["tonnage"])
    np.testing.assert_allclose(_twin(ax).lines[0].get_ydata(), gt["mean_grade"])

    _, ax = cs.plot.grade_tonnage(cs.grade_tonnage(v, cutoffs, categories=np.arange(400) % 2))
    assert len(ax.lines) == 3 and len(_twin(ax).get_legend().get_texts()) == 5

    anamorphosis = cs.HermiteAnamorphosis().fit(v)
    bm = cs.BlockModel((0, 0, 0), (10, 10, 10), (5, 5, 1))
    g = rng.lognormal(0, 0.5, 25)
    bm = bm.with_columns({"a": g, "b": 1.1 * g})
    uc = cs.UniformConditioning(anamorphosis, 0.7, r_panel=0.5)
    curves = {
        "samples": gt,
        "point": anamorphosis.grade_tonnage(cutoffs),
        "UC": uc.grade_tonnage(bm, "a", cutoffs),
    }
    _, ax = cs.plot.grade_tonnage(curves, relative=True)
    assert len(ax.lines) == 3 and all(line.get_ydata()[0] == 1.0 for line in ax.lines)

    compared = cs.compare_models(bm, ["a", "b"], cutoffs, categories=np.arange(25) % 2)
    _, ax = cs.plot.grade_tonnage(compared)
    labels = [t.get_text() for t in _twin(ax).get_legend().get_texts()]
    assert labels[2:] == ["a 0", "b 0", "a 1", "b 1", "a all", "b all"]
    b1 = (compared["model"] == "b") & (compared["category"] == "1")
    np.testing.assert_allclose(ax.lines[3].get_ydata(), compared["tonnage"][b1])


def _cv(n=2000, spread=1.0):
    estimate = rng.normal(0, 1, n)
    variance = np.full(n, 0.25)
    actual = estimate + rng.normal(0, 0.5 * spread, n)
    estimate[0] = np.nan
    return cs.CrossValidation(actual, estimate, variance)


def test_cross_validation_scatter_and_errors():
    cv = _cv()
    _, ax = cs.plot.cross_validation(cv)
    x, y = ax.lines[1].get_data()
    assert (y[1] - y[0]) / (x[1] - x[0]) == pytest.approx(cv.slope)
    assert "RMSE" in ax.texts[0].get_text()
    _, ax = cs.plot.cross_validation(cv, kind="errors")
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(1.0)
    coords = rng.uniform(0, 100, (2000, 2))
    _, ax = cs.plot.cross_validation(cv, kind="errors", coords=cs.PointSet(coords))
    assert len(ax.collections[0].get_offsets()) == 1999
    with pytest.raises(cs.InvalidInput, match="kind"):
        cs.plot.cross_validation(cv, kind="map")
    with pytest.raises(cs.InvalidInput, match="coords"):
        cs.plot.cross_validation(cv, kind="errors", coords=coords[:10])


def test_accuracy_is_diagonal_when_the_variance_is_calibrated():
    _, ax = cs.plot.cross_validation(_cv(), kind="accuracy")
    p, accuracy = ax.lines[1].get_data()
    np.testing.assert_allclose(accuracy, p, atol=0.03)
    _, ax = cs.plot.cross_validation(_cv(spread=2.0), kind="accuracy")
    p, accuracy = ax.lines[1].get_data()
    assert np.all(accuracy[1:-1] < p[1:-1])

    pit = rng.uniform(0, 1, 500)
    cv = cs.IndicatorCrossValidation(np.ones(500), np.ones(500), np.ones(500), [1.0], np.ones((1, 500)), pit)
    _, ax = cs.plot.cross_validation(cv, kind="accuracy")
    p, accuracy = ax.lines[1].get_data()
    np.testing.assert_allclose(accuracy, cv.accuracy(p))
    assert f"{cv.goodness:.2f}" in ax.texts[0].get_text()


def test_contact_draws_each_side_apart():
    z = np.tile(np.arange(20.0), 2)
    table = cs.contact(
        np.c_[np.zeros(40), np.zeros(40), z],
        np.where(z < 10, 3.0, 1.0) + rng.normal(0, 0.1, 40),
        domains=np.where(z < 10, "ore", "waste"),
        holes=np.repeat(["A", "B"], 20),
        inside="ore",
        outside="waste",
        max_distance=50.0,
        bin=2.0,
    )
    _, ax = cs.plot.contact(table, labels=("ore", "waste"))
    inside, outside = (line.get_xdata() for line in ax.lines[:2])
    assert np.all(inside < 0) and np.all(outside > 0)
    assert sum(p.get_height() for p in _twin(ax).patches) == 40
    assert [t.get_text() for t in ax.texts] == ["ore", "waste"]
