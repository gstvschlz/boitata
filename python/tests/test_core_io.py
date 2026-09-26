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


def test_discretize_keeps_each_node_in_its_block():
    model = cs.BlockModel(origin=(100, 200, 0), size=(10, 5, 2), count=(3, 2, 2), rotation=(30, 20, 10))
    masked = model.mask(np.arange(12) % 3 != 1)
    nodes = masked.discretize((2, 3, 2))
    assert len(nodes) == 12 * len(masked) and nodes.size == pytest.approx([5, 5 / 3, 1])
    block = nodes["block"].astype(int)
    np.testing.assert_allclose(np.bincount(block, nodes.volumes), masked.volumes)
    np.testing.assert_allclose(np.bincount(block, weights=nodes.centroids[:, 0]) / 12, masked.centroids[:, 0])
    flat = cs.BlockModel(origin=(0, 0), size=(10, 10), count=(2, 2)).discretize(4)
    assert flat.count == [8, 8, 1]
    parent = np.array([0, 0], dtype=np.uint64)
    sub = cs.BlockModel.subblocked(
        (0, 0), (10, 10), (2, 2), parent, [[0, 0, 0, 0.3, 1, 1], [0.3, 0, 0, 1, 1, 1]]
    )
    np.testing.assert_allclose(
        np.bincount(sub.discretize(3)["block"].astype(int), sub.discretize(3).volumes), [30, 70]
    )
    with pytest.raises(cs.InvalidInput):
        model.discretize(0)


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


def test_subblocked_model(tmp_path):
    parent = np.array([0, 0, 3], dtype=np.uint64)
    extents = [[0, 0, 0, 0.5, 1, 1], [0.5, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    model = cs.BlockModel.subblocked(
        (0, 0), (10, 10), (2, 2), parent, extents, subgrid=(2, 1, 1), attributes={"g": [1.0, 3.0, 5.0]}
    )
    np.testing.assert_allclose(model.volumes, [50, 50, 100])
    np.testing.assert_allclose(model.centroids[1], [7.5, 5, 0.5])
    regular = model.to_regular()
    np.testing.assert_allclose(regular["g"][[0, 3]], [2.0, 5.0])
    cs.write_parquet(tmp_path / "s.parquet", model)
    back = cs.read_parquet(tmp_path / "s.parquet")
    np.testing.assert_allclose(back.extents, model.extents)
    with pytest.raises(cs.InvalidInput):
        cs.BlockModel.subblocked((0, 0), (10, 10), (2, 2), parent, extents, subgrid=(3, 1, 1))


def test_mesh_files_round_trip(tmp_path):
    vertices = [[0, 0, 0], [1.5, 0, 0], [0, 1, 0], [0, 0, 1]]
    tetra = cs.Mesh(vertices, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tetra = tetra.with_face_column("layer", ["a", "a", "b", "b"])
    for name in ("m.obj", "m.stl", "m.dxf"):
        cs.write_mesh(tmp_path / name, tetra)
        back = cs.read_mesh(tmp_path / name)
        np.testing.assert_array_equal(back.vertices[back.triangles], tetra.vertices[tetra.triangles])
        assert back.is_closed and back.volume == pytest.approx(tetra.volume)
    assert back.face_attributes["layer"] == ["a", "a", "b", "b"]
    cs.write_mesh(tmp_path / "a.stl", tetra, ascii=True)
    assert (tmp_path / "a.stl").read_text().startswith("solid")
    np.testing.assert_array_equal(
        cs.read_mesh(tmp_path / "a.stl").vertices, cs.read_mesh(tmp_path / "m.stl").vertices
    )
    with pytest.raises(cs.InvalidInput):
        cs.write_mesh(tmp_path / "m.ply", tetra)
    with pytest.raises(cs.FileError):
        cs.read_mesh(tmp_path / "missing.obj")
