import boitata as bt
import numpy as np
import pytest
from boitata._columns import column, stack

rng = np.random.default_rng(0)
xyz = rng.uniform(0, 100, (30, 3))
points = bt.PointSet(xyz, {"v": rng.uniform(1, 2, 30), "rock": ["ox", None, "fr"] * 10})
model = bt.BlockModel((0, 0, 0), (10, 10, 10), (3, 2, 1), attributes={"d": np.full(6, 2.5)})


def test_missing_column_is_a_key_error_listing_the_columns():
    for data, listed in (
        (points, "v, rock"),
        (points.attributes, "v, rock"),
        (model, "d"),
        ({"v": [1.0]}, "v"),
    ):
        with pytest.raises(bt.MissingColumn, match=f'no column "w"; columns: {listed}$') as e:
            column(data, "w")
        assert isinstance(e.value, KeyError) and isinstance(e.value, bt.BoitataError)
    with pytest.raises(KeyError, match="columns: v, rock"):
        points["w"]
    with pytest.raises(bt.InvalidInput, match="needs a container"):
        column(None, "v")


def test_text_columns_are_object_arrays_with_none():
    rock = points["rock"]
    assert rock.dtype == object and list(rock[:3]) == ["ox", None, "fr"]


def test_column_takes_a_name_or_an_array():
    assert column(points, "v") is not None and np.array_equal(column(points, "v"), points["v"])
    assert column(points, [1.0]) == [1.0]
    data, labels = stack({"a": [1, 2], "b": [3, 4]})
    assert labels == ["a", "b"] and data.tolist() == [[1, 3], [2, 4]]


def test_containers_are_coordinates():
    assert np.array_equal(bt.data_spacing(points, points, None), bt.data_spacing(xyz, xyz, None))
    grade = np.arange(6.0)
    by_model = bt.swath(model, grade, 10.0, axis="x")
    assert np.array_equal(by_model["mean"], bt.swath(model.coords, grade, 10.0, axis="x")["mean"])


def test_per_row_takes_a_constant_an_array_or_a_name():
    grade = np.arange(6.0)
    tonnage = [
        bt.swath(model, grade, 10.0, axis="x", density=d)["tonnage"] for d in (2.5, np.full(6, 2.5), "d")
    ]
    assert np.array_equal(tonnage[0], tonnage[1]) and np.array_equal(tonnage[0], tonnage[2])
    with pytest.raises(bt.InvalidInput, match="density: expected 6"):
        bt.swath(model, grade, 10.0, axis="x", density=[1.0])


def test_pair_takes_one_name_or_two():
    a, b = points.filter(np.arange(30) < 15), points.filter(np.arange(30) >= 15)
    by_name = bt.pairs(a, b, 50.0, values="v")
    by_arrays = bt.pairs(a, b, 50.0, values=(a["v"], b["v"]))
    mixed = bt.pairs(a, b, 50.0, values=("v", b["v"]))
    for t in (by_arrays, mixed):
        assert np.array_equal(t["value_a"], by_name["value_a"]) and np.array_equal(
            t["value_b"], by_name["value_b"]
        )
    with pytest.raises(bt.InvalidInput, match="column name or a pair"):
        bt.pairs(a, b, 50.0, values=1.0)


def test_filter_and_with_columns_round_trip():
    keep = points["v"] > 1.5
    kept = points.filter(keep)
    assert len(kept) == keep.sum() and np.array_equal(kept.coords, xyz[keep])
    assert list(kept["rock"]) == list(points["rock"][keep])
    with pytest.raises(bt.InvalidInput):
        points.filter(keep[:3])
    added = points.with_columns({"w": points["v"] * 2, "rock": ["x"] * 30})
    assert np.array_equal(added["w"], points["v"] * 2) and set(added["rock"]) == {"x"}
    again = bt.PointSet(xyz).with_columns(points.attributes)
    assert again.attributes.column_names == ["v", "rock"] and list(again["rock"]) == list(points["rock"])
    graded = model.with_columns(bt.Table({"g": np.arange(6.0)}))
    assert graded.attributes.column_names == ["d", "g"]
    with pytest.raises(bt.InvalidInput):
        model.with_columns({"g": [1.0]})


def test_with_column_takes_text():
    rocks = np.array(["ox", "fr"] * 15)
    assert list(points.with_column("r", rocks)["r"]) == list(rocks)
    assert list(points.with_column("r", list(rocks))["r"]) == list(rocks)
    gaps = np.array(["a", None, "b", None, "c", "d"], dtype=object)
    assert list(model.with_column("r", gaps)["r"]) == list(gaps)
    with pytest.raises(bt.InvalidInput):
        points.with_column("r", ["ox"])


rock = np.array(["ox", "na", "fr"] * 10)
labeled = points.with_columns({"rock": rock, "w": np.linspace(1, 2, 30)})


def test_statistics_take_names_of_data():
    v, w = points["v"], labeled["w"]
    assert bt.describe("v", weights="w", data=labeled) == bt.describe(v, weights=w)
    by_name = bt.describe_by("v", "rock", weights="w", data=labeled)
    by_array = bt.describe_by(v, rock, weights=w)
    assert all(np.array_equal(by_name[c], by_array[c]) for c in ("n", "mean", "P50"))
    assert np.array_equal(bt.capping("v", data=points)["mean"], bt.capping(v)["mean"])
    assert bt.describe("d", data=model)["n"] == 6
    by_name = bt.correlation(labeled, columns=["v", "w"], weights="w")
    assert np.array_equal(by_name, bt.correlation(np.c_[v, w], weights=w))
    assert bt.correlation({"a": v, "b": 2 * v}).shape == (2, 2)
    assert np.array_equal(bt.h_scatter(points, "v", 20.0, 10.0)[0], bt.h_scatter(xyz, v, 20.0, 10.0)[0])
    with pytest.raises(bt.MissingColumn):
        bt.describe("x", data=points)


def test_domains_or_domain_column():
    caps = {"ox": 1.5}
    by_name = bt.capping_report("v", caps, domain_column="rock", data=labeled)
    by_array = bt.capping_report(points["v"], caps, domains=rock)
    assert np.array_equal(by_name["mean_capped"], by_array["mean_capped"])
    down = np.c_[np.zeros(30), np.zeros(30), np.tile(np.arange(10.0), 3)]
    side = np.where(np.arange(30) % 10 < 5, "a", "b")
    hole = np.repeat(np.arange(3), 10)
    holes = bt.PointSet(down, {"v": points["v"], "side": side, "hole": hole})
    kw = {"inside": "a", "outside": "b", "max_distance": 20.0, "bin": 2.0}
    c = bt.contact(holes, "v", domain_column="side", holes="hole", **kw)
    assert np.array_equal(c["mean"], bt.contact(down, points["v"], domains=side, holes=hole, **kw)["mean"])
    for bad in ({}, {"domains": rock, "domain_column": "rock"}):
        with pytest.raises(bt.InvalidInput, match="one of domains or domain_column"):
            bt.capping_report("v", caps, data=labeled, **bad)


def test_defaulted_options_are_keyword_only():
    v = points["v"]
    for call in (
        lambda: bt.describe(v, None),
        lambda: bt.describe_by(v, rock, None),
        lambda: bt.capping(v, None),
        lambda: bt.capping_report(v, {}, rock),
        lambda: bt.h_scatter(xyz, v, 1.0, 0.5, 90.0),
        lambda: bt.correlation(xyz, None),
        lambda: bt.duplicates(xyz, 0.5),
        lambda: bt.data_spacing(xyz, xyz, None, 2),
        lambda: bt.pairs(xyz, xyz, 1.0, "v"),
        lambda: bt.contact(xyz, v, rock, rock, "ox", "fr", 1.0, 1.0),
    ):
        with pytest.raises(TypeError):
            call()


def test_a_close_name_is_suggested():
    for data in (points, {"v": [1.0], "rock": ["a"]}):
        with pytest.raises(bt.MissingColumn, match='did you mean "rock"'):
            column(data, "Rock")


def _containers():
    lines = bt.Polylines([xyz[:4], xyz[4:7]], features=[0, 1], attributes={"name": ["a", "b"]})
    mesh = bt.Mesh(np.eye(3), [[0, 1, 2]]).with_vertex_column("h", [1.0, 2.0, 3.0])
    collar = {"HOLE_ID": ["a"], "X": [0.0], "Y": [0.0], "Z": [10.0]}
    survey = {"HOLE_ID": ["a", "a"], "DEPTH": [0.0, 10.0], "AZIMUTH": [0.0, 0.0], "DIP": [90.0, 90.0]}
    assays = {"HOLE_ID": ["a", "a"], "FROM": [0.0, 5.0], "TO": [5.0, 10.0], "AU": [1.0, 2.0]}
    holes = bt.Drillholes(collar, survey, assays)
    return points, lines, model, mesh, holes


def test_every_container_spells_coordinates_bounds_and_columns_the_same():
    for data in _containers():
        coords = data.coords
        assert coords.shape[1] == 3
        np.testing.assert_array_equal(np.column_stack([data.x, data.y, data.z]), coords)
        lo, hi = data.bounds
        assert (coords >= np.array(lo) - 1e-9).all() and (coords <= np.array(hi) + 1e-9).all()
        assert all(np.asarray(data[c]).shape[0] > 0 for c in data.column_names)
    _, lines, _, _, holes = _containers()
    np.testing.assert_array_equal(holes.coords, [[0, 0, 7.5], [0, 0, 2.5]])
    assert list(holes["AU"]) == [1.0, 2.0] and holes.column_names == ["HOLE_ID", "FROM", "TO", "AU"]
    assert lines.filter(np.array([False, True]))["name"].tolist() == ["b"]
    assert len(lines.with_columns({"n": [1.0, 2.0]}).column_names) == 2


def test_mesh_columns_come_from_vertices_then_faces():
    mesh = bt.Mesh(np.eye(3), [[0, 1, 2]]).with_vertex_column("h", [1.0, 2.0, 3.0])
    assert len(mesh) == 3 and mesh["h"].tolist() == [1.0, 2.0, 3.0]
    mesh = mesh.with_face_column("f", [7.0])
    assert mesh["f"].tolist() == [7.0] and mesh.column_names == ["h", "f"]
    with pytest.raises(bt.InvalidInput, match="both a vertex and a face"):
        mesh.with_face_column("h", [0.0])["h"]


def test_containers_convert_like_tables():
    pl = pytest.importorskip("polars")
    for data in (points, model):
        frame = data.to_polars()
        assert isinstance(frame, pl.DataFrame) and frame.columns[:3] == ["x", "y", "z"]
    table = bt.Table({"a": [1.0, 2.0]}).with_column("b", ["x", None]).with_columns({"a": [3.0, 4.0]})
    assert table.column_names == ["a", "b"] and table["a"].tolist() == [3.0, 4.0]


def test_sample_is_missing_outside_the_model():
    masked = model.with_columns({"rock": list("abcdef")}).filter(np.arange(6) != 1)
    at = np.array([[5.0, 5, 5], [15, 5, 5], [99, 99, 99], [25, 15, 5]])
    rows = masked.row_at(at)
    assert masked.contains(at).tolist() == [True, False, False, True]
    got = masked.sample(at, "d")
    np.testing.assert_array_equal(got[rows >= 0], masked["d"][rows[rows >= 0]])
    assert np.isnan(got[rows < 0]).all()
    assert masked.sample(at, "rock").tolist() == ["a", None, None, "f"]


def test_grid_places_rows_by_cell_and_leaves_absent_cells_missing():
    values = np.arange(6.0)
    full = bt.BlockModel((0, 0, 0), (1, 1, 1), (3, 2, 1), attributes={"v": values})
    np.testing.assert_array_equal(full.grid("v"), values.reshape(1, 2, 3))
    masked = full.filter(values != 4)
    grid = masked.grid("v")
    assert np.isnan(grid[0, 1, 1]) and grid[0, 1, 2] == 5.0


def test_drop_null_keeps_rows_with_every_value():
    data = bt.PointSet(xyz[:3], {"a": [1.0, np.nan, 3.0], "b": ["x", "y", None]})
    assert len(data.drop_null()) == 1 and len(data.drop_null("a")) == 2
    assert len(bt.Table({"a": [np.nan, 1.0]}).drop_null()) == 1


def test_drillholes_from_tables_reads_the_dataset_names():
    tables = {
        "collars": {"HOLE_ID": ["a"], "X": [0.0], "Y": [0.0], "Z": [10.0]},
        "surveys": {"HOLE_ID": ["a"], "DEPTH": [0.0], "AZIMUTH": [0.0], "DIP": [90.0]},
        "lithology": {"HOLE_ID": ["a"], "FROM": [0.0], "TO": [2.0], "ROCK": ["ox"]},
    }
    rocks = bt.Drillholes.from_tables(tables, intervals="lithology")
    assert rocks["ROCK"].tolist() == ["ox"]
    assert len(bt.Drillholes.from_tables(tables, intervals=None)) == 1


def test_standardized_variogram_keeps_shape_and_anisotropy():
    fitted = bt.Variogram([("spherical", 4.0, 100.0)], nugget=1.0, rotation=(30, 0, 0), ratios=(0.5, 0.25))
    unit = fitted.standardized()
    assert unit.sill == pytest.approx(1.0) and unit.nugget == pytest.approx(0.2)
    assert unit.rotation == fitted.rotation and unit.ratios == fitted.ratios
    np.testing.assert_allclose(unit.gamma([10.0, 60.0]), fitted.gamma([10.0, 60.0]) / 5.0)
    assert fitted.standardized(sill=2.0).sill == pytest.approx(2.0)


def test_compare_skips_missing_pairs():
    result = bt.compare("e", "t", data={"e": [1.0, 2.0, np.nan, 4.5], "t": [1.0, 2.0, 3.0, 4.0]})
    assert result["n"] == 3 and result["mean_error"] == pytest.approx(0.5 / 3)
    exact = bt.compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert (
        exact["rmse"] == 0.0
        and exact["correlation"] == pytest.approx(1.0)
        and exact["slope"] == pytest.approx(1.0)
    )
    with pytest.raises(bt.InvalidInput):
        bt.compare([1.0], [1.0, 2.0])
