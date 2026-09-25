import math

import ceres as cs
import numpy as np
import pytest


@pytest.fixture
def samples(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("X,Y,au,rock\n1,3,0.5,ox\n2,4,-999,fr\n")
    return path


def test_read_csv_maps_nodata_to_null(samples):
    table = cs.read_csv(samples)
    assert table.column_names == ["X", "Y", "au", "rock"]
    au = table["au"]
    assert au[0] == 0.5 and math.isnan(au[1])
    assert table["rock"] == ["ox", "fr"]


def test_pointset_from_table_and_back(samples):
    points = cs.PointSet.from_table(cs.read_csv(samples), crs="EPSG:32611")
    np.testing.assert_array_equal(points.coords, [[1, 3, 0], [2, 4, 0]])
    assert len(points) == 2 and points.crs == "EPSG:32611"
    assert points.to_table().column_names == ["x", "y", "z", "au", "rock"]


def test_pointset_accepts_2d_coords_and_dict():
    points = cs.PointSet([[0, 0], [1, 1]], {"v": [1.0, np.nan]})
    assert points.coords.shape == (2, 3)
    assert math.isnan(points["v"][1])


def test_invalid_input_is_value_error_and_ceres_error():
    with pytest.raises(cs.InvalidInput) as info:
        cs.PointSet([[0, 0, 0]], {"v": [1.0, 2.0]})
    assert isinstance(info.value, ValueError) and isinstance(info.value, cs.CeresError)
    with pytest.raises(cs.FileError):
        cs.read_csv("missing.csv")


def test_blockmodel_rotation_mask_and_regular():
    model = cs.BlockModel(
        origin=(100, 200),
        size=(10, 5),
        count=(3, 2),
        rotation=(90, 0, 0),
        attributes={"v": np.arange(6.0)},
    )
    assert model.count == [3, 2, 1] and len(model) == 6
    np.testing.assert_allclose(model.centroids[0], [102.5, 195.0, 0.5])

    masked = model.mask(np.array([True, False, True, False, False, True]))
    np.testing.assert_array_equal(masked.index, [0, 2, 5])
    back = masked.to_regular()
    assert back.index is None
    np.testing.assert_array_equal(np.isnan(back["v"]), [False, True, False, True, True, False])


def test_gslib_round_trip(tmp_path):
    source = tmp_path / "grid.dat"
    source.write_text("grid\n2\nau\ncu\n1 -999\n0.25 3\n")
    table = cs.read_gslib(source)
    cs.write_gslib(tmp_path / "out.dat", table)
    back = cs.read_gslib(tmp_path / "out.dat")
    np.testing.assert_array_equal(back["au"], table["au"])


def test_arrow_interop():
    pa = pytest.importorskip("pyarrow")
    pl = pytest.importorskip("polars")
    points = cs.PointSet([[0, 0], [1, 2]], {"v": [1.0, 2.0]})
    assert pa.table(points).column_names == ["x", "y", "z", "v"]
    assert pl.DataFrame(points)["v"].to_list() == [1.0, 2.0]
    back = cs.Table(pa.table({"a": [1.5, None]}))
    assert back.num_rows == 2 and math.isnan(back["a"][1])
    assert points.attributes.to_pandas()["v"].tolist() == [1.0, 2.0]


def test_parquet_round_trips_containers(tmp_path):
    points = cs.PointSet([[0, 0, 1], [1, 2, 3]], {"v": [1.0, np.nan]}, crs="EPSG:32611")
    cs.write_parquet(tmp_path / "p.parquet", points)
    back = cs.read_parquet(tmp_path / "p.parquet")
    assert isinstance(back, cs.PointSet) and back.crs == "EPSG:32611"
    np.testing.assert_array_equal(back.coords, points.coords)
    assert math.isnan(back["v"][1])

    model = cs.BlockModel(
        origin=(0, 0), size=(10, 10), count=(4, 3), rotation=(30, 0, 0), attributes={"g": np.arange(12.0)}
    )
    masked = model.mask(np.arange(12) % 5 == 0)
    cs.write_parquet(tmp_path / "b.parquet", masked)
    back = cs.read_parquet(tmp_path / "b.parquet")
    assert isinstance(back, cs.BlockModel) and back.rotation == [30.0, 0.0, 0.0]
    np.testing.assert_array_equal(back.index, masked.index)
    np.testing.assert_array_equal(back["g"], masked["g"])


def test_parquet_is_readable_by_other_tools(tmp_path):
    pl = pytest.importorskip("polars")
    cs.write_parquet(
        tmp_path / "b.parquet",
        cs.BlockModel(origin=(0, 0), size=(1, 1), count=(2, 2), attributes={"g": [1.0, 2, 3, 4]}),
    )
    assert pl.read_parquet(tmp_path / "b.parquet")["g"].to_list() == [1, 2, 3, 4]
    assert isinstance(cs.read_parquet(tmp_path / "b.parquet"), cs.BlockModel)
