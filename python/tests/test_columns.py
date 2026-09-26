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
