import ceres as cs
import numpy as np
import pytest
from ceres._columns import column, stack

rng = np.random.default_rng(0)
xyz = rng.uniform(0, 100, (30, 3))
points = cs.PointSet(xyz, {"v": rng.uniform(1, 2, 30), "rock": ["ox", None, "fr"] * 10})
model = cs.BlockModel((0, 0, 0), (10, 10, 10), (3, 2, 1), attributes={"d": np.full(6, 2.5)})


def test_missing_column_is_a_key_error_listing_the_columns():
    for data, listed in (
        (points, "v, rock"),
        (points.attributes, "v, rock"),
        (model, "d"),
        ({"v": [1.0]}, "v"),
    ):
        with pytest.raises(cs.MissingColumn, match=f'no column "w"; columns: {listed}$') as e:
            column(data, "w")
        assert isinstance(e.value, KeyError) and isinstance(e.value, cs.CeresError)
    with pytest.raises(KeyError, match="columns: v, rock"):
        points["w"]
    with pytest.raises(cs.InvalidInput, match="needs a container"):
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
    assert np.array_equal(cs.data_spacing(points), cs.data_spacing(xyz))
    grade = np.arange(6.0)
    by_model = cs.swath(model, grade, 10.0, axis="x")
    assert np.array_equal(by_model["mean"], cs.swath(model.centroids, grade, 10.0, axis="x")["mean"])


def test_per_row_takes_a_constant_an_array_or_a_name():
    grade = np.arange(6.0)
    tonnage = [
        cs.swath(model, grade, 10.0, axis="x", density=d)["tonnage"] for d in (2.5, np.full(6, 2.5), "d")
    ]
    assert np.array_equal(tonnage[0], tonnage[1]) and np.array_equal(tonnage[0], tonnage[2])
    with pytest.raises(cs.InvalidInput, match="density: expected 6"):
        cs.swath(model, grade, 10.0, axis="x", density=[1.0])


def test_pair_takes_one_name_or_two():
    a, b = points.filter(np.arange(30) < 15), points.filter(np.arange(30) >= 15)
    by_name = cs.pairs(a, b, 50.0, values="v")
    by_arrays = cs.pairs(a, b, 50.0, values=(a["v"], b["v"]))
    mixed = cs.pairs(a, b, 50.0, values=("v", b["v"]))
    for t in (by_arrays, mixed):
        assert np.array_equal(t["value_a"], by_name["value_a"]) and np.array_equal(
            t["value_b"], by_name["value_b"]
        )
    with pytest.raises(cs.InvalidInput, match="column name or a pair"):
        cs.pairs(a, b, 50.0, values=1.0)


def test_filter_and_with_columns_round_trip():
    keep = points["v"] > 1.5
    kept = points.filter(keep)
    assert len(kept) == keep.sum() and np.array_equal(kept.coords, xyz[keep])
    assert list(kept["rock"]) == list(points["rock"][keep])
    with pytest.raises(cs.InvalidInput):
        points.filter(keep[:3])
    added = points.with_columns({"w": points["v"] * 2, "rock": ["x"] * 30})
    assert np.array_equal(added["w"], points["v"] * 2) and set(added["rock"]) == {"x"}
    again = cs.PointSet(xyz).with_columns(points.attributes)
    assert again.attributes.column_names == ["v", "rock"] and list(again["rock"]) == list(points["rock"])
    graded = model.with_columns(cs.Table({"g": np.arange(6.0)}))
    assert graded.attributes.column_names == ["d", "g"]
    with pytest.raises(cs.InvalidInput):
        model.with_columns({"g": [1.0]})


def test_with_column_takes_text():
    rocks = np.array(["ox", "fr"] * 15)
    assert list(points.with_column("r", rocks)["r"]) == list(rocks)
    assert list(points.with_column("r", list(rocks))["r"]) == list(rocks)
    gaps = np.array(["a", None, "b", None, "c", "d"], dtype=object)
    assert list(model.with_column("r", gaps)["r"]) == list(gaps)
    with pytest.raises(cs.InvalidInput):
        points.with_column("r", ["ox"])


rock = np.array(["ox", "na", "fr"] * 10)
labeled = points.with_columns({"rock": rock, "w": np.linspace(1, 2, 30)})


def test_statistics_take_names_of_data():
    v, w = points["v"], labeled["w"]
    assert cs.describe("v", weights="w", data=labeled) == cs.describe(v, weights=w)
    by_name = cs.describe_by("v", "rock", weights="w", data=labeled)
    by_array = cs.describe_by(v, rock, weights=w)
    assert all(np.array_equal(by_name[c], by_array[c]) for c in ("n", "mean", "P50"))
    assert np.array_equal(cs.capping("v", data=points)["mean"], cs.capping(v)["mean"])
    assert cs.describe("d", data=model)["n"] == 6
    by_name = cs.correlation(labeled, columns=["v", "w"], weights="w")
    assert np.array_equal(by_name, cs.correlation(np.c_[v, w], weights=w))
    assert cs.correlation({"a": v, "b": 2 * v}).shape == (2, 2)
    assert np.array_equal(cs.h_scatter(points, "v", 20.0, 10.0)[0], cs.h_scatter(xyz, v, 20.0, 10.0)[0])
    with pytest.raises(cs.MissingColumn):
        cs.describe("x", data=points)


def test_domains_or_domain_column():
    caps = {"ox": 1.5}
    by_name = cs.capping_report("v", caps, domain_column="rock", data=labeled)
    by_array = cs.capping_report(points["v"], caps, domains=rock)
    assert np.array_equal(by_name["mean_capped"], by_array["mean_capped"])
    down = np.c_[np.zeros(30), np.zeros(30), np.tile(np.arange(10.0), 3)]
    side = np.where(np.arange(30) % 10 < 5, "a", "b")
    hole = np.repeat(np.arange(3), 10)
    holes = cs.PointSet(down, {"v": points["v"], "side": side, "hole": hole})
    kw = {"inside": "a", "outside": "b", "max_distance": 20.0, "bin": 2.0}
    c = cs.contact(holes, "v", domain_column="side", holes="hole", **kw)
    assert np.array_equal(c["mean"], cs.contact(down, points["v"], domains=side, holes=hole, **kw)["mean"])
    for bad in ({}, {"domains": rock, "domain_column": "rock"}):
        with pytest.raises(cs.InvalidInput, match="one of domains or domain_column"):
            cs.capping_report("v", caps, data=labeled, **bad)


def test_defaulted_options_are_keyword_only():
    v = points["v"]
    for call in (
        lambda: cs.describe(v, None),
        lambda: cs.describe_by(v, rock, None),
        lambda: cs.capping(v, None),
        lambda: cs.capping_report(v, {}, rock),
        lambda: cs.h_scatter(xyz, v, 1.0, 0.5, 90.0),
        lambda: cs.correlation(xyz, None),
        lambda: cs.duplicates(xyz, 0.5),
        lambda: cs.data_spacing(xyz, 2),
        lambda: cs.pairs(xyz, xyz, 1.0, "v"),
        lambda: cs.contact(xyz, v, rock, rock, "ox", "fr", 1.0, 1.0),
    ):
        with pytest.raises(TypeError):
            call()
