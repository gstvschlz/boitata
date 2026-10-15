import boitata as bt
import numpy as np
import pytest


def tetra():
    mesh = bt.Mesh(
        [[0, 0, 0], [1.5, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    )
    return mesh.with_vertex_column("v", [1.0, np.nan, 3.0, 4.0]).with_face_column("f", np.arange(4.0))


@pytest.mark.parametrize("ascii", [False, True])
def test_vtk_round_trip(tmp_path, ascii):
    mesh = tetra()
    for suffix in ("vtp", "vtu"):
        bt.write_vtk(tmp_path / f"m.{suffix}", mesh, ascii=ascii)
        back = bt.read_vtk(tmp_path / f"m.{suffix}")
        np.testing.assert_array_equal(back.coords, mesh.coords)
        np.testing.assert_array_equal(back.triangles, mesh.triangles)
        np.testing.assert_array_equal(back["v"], mesh["v"])
        np.testing.assert_array_equal(back["f"], mesh["f"])
    points = bt.PointSet([[1, 2, 3], [4, 5, 6]], attributes={"au": np.array([0.5, 2.0], dtype=np.float32)})
    bt.write_vtk(tmp_path / "p.vtp", points, ascii=ascii)
    back = bt.read_vtk(tmp_path / "p.vtp", length_unit="m")
    assert isinstance(back, bt.PointSet) and back.length_unit == "m"
    np.testing.assert_array_equal(back["au"], points["au"])
    lines = bt.Polylines(
        [[[0, 0], [1, 0]], [[0, 0], [1, 0], [1, 1]]], closed=[False, True], attributes={"id": [7.0, 8.0]}
    )
    bt.write_vtk(tmp_path / "l.vtu", lines, ascii=ascii)
    back = bt.read_vtk(tmp_path / "l.vtu")
    assert isinstance(back, bt.Polylines)
    np.testing.assert_array_equal(back.closed, lines.closed)
    np.testing.assert_array_equal(back["id"], lines["id"])
    with pytest.raises(bt.InvalidInput):
        bt.write_vtk(tmp_path / "x.vtk", points)


def test_vtk_files_open_in_pyvista(tmp_path):
    pv = pytest.importorskip("pyvista")
    mesh = tetra()
    for suffix in ("vtp", "vtu"):
        bt.write_vtk(tmp_path / f"m.{suffix}", mesh)
        read = pv.read(tmp_path / f"m.{suffix}")
        assert (read.n_points, read.n_cells) == (4, 4)
        np.testing.assert_array_equal(read.points, mesh.coords)
        np.testing.assert_array_equal(read.point_data["v_valid"], [1, 0, 1, 1])
        np.testing.assert_array_equal(read.cell_data["f"], mesh["f"])
        from vtkmodules import vtkIOXML

        writer = (
            vtkIOXML.vtkXMLPolyDataWriter if suffix == "vtp" else vtkIOXML.vtkXMLUnstructuredGridWriter
        )()
        writer.SetInputData(read)
        for mode, encode, header, zlib in [
            (writer.SetDataModeToBinary, False, writer.SetHeaderTypeToUInt32, True),
            (writer.SetDataModeToAppended, True, writer.SetHeaderTypeToUInt64, True),
            (writer.SetDataModeToAppended, False, writer.SetHeaderTypeToUInt32, False),
        ]:
            mode()
            header()
            writer.SetEncodeAppendedData(encode)
            writer.SetCompressorTypeToZLib() if zlib else writer.SetCompressorTypeToNone()
            writer.SetFileName(str(tmp_path / f"z.{suffix}"))
            writer.Write()
            back = bt.read_vtk(tmp_path / f"z.{suffix}")
            np.testing.assert_array_equal(back.coords, mesh.coords)
            np.testing.assert_array_equal(back["v"], mesh["v"])
            np.testing.assert_array_equal(back["f"], mesh["f"])


def grid(rotation=(30, 10, 0)):
    au = np.arange(24.0)
    au[5] = np.nan
    return bt.BlockModel(
        origin=(500, 7000, -40), size=(10, 5, 2.5), count=(4, 3, 2), rotation=rotation, attributes={"au": au}
    )


@pytest.mark.parametrize("suffix", ["vti", "vtu"])
def test_block_models_round_trip(tmp_path, suffix):
    model = grid()
    sparse = model.filter(np.arange(24) % 5 != 1)
    for m in (model, sparse):
        bt.write_vtk(tmp_path / f"b.{suffix}", m)
        back = bt.read_vtk(tmp_path / f"b.{suffix}")
        assert isinstance(back, bt.BlockModel) and back.count == m.count and len(back) == len(m)
        np.testing.assert_allclose(back.coords, m.coords)
        np.testing.assert_array_equal(back["au"], m["au"])
    nodes = bt.read_vtk(tmp_path / "b.vti", point_data=True) if suffix == "vti" else None
    assert nodes is None or len(nodes) == 5 * 4 * 3
    with pytest.raises(bt.InvalidInput):
        bt.write_vtk(tmp_path / "b.vtp", model)


def test_vtk_grids_match_pyvista(tmp_path):
    pv = pytest.importorskip("pyvista")
    model = grid().filter(np.arange(24) % 5 != 1)
    bt.write_vtk(tmp_path / "b.vti", model)
    image = pv.read(tmp_path / "b.vti")
    assert (image.cell_data["vtkGhostType"] == 32).sum() == 5
    np.testing.assert_allclose(image.cell_centers().points, model.coords, atol=1e-9)

    x, y, z = np.array([10.0, 12, 14]), np.array([0.0, 5]), np.array([0.0, 1, 2])
    rect = pv.RectilinearGrid(x, y, z)
    rect.cell_data["au"] = np.arange(4.0)
    rect.save(tmp_path / "r.vtr")
    back = bt.read_vtk(tmp_path / "r.vtr")
    assert isinstance(back, bt.BlockModel) and list(back.count) == [2, 1, 2]
    np.testing.assert_allclose(back.coords, rect.cell_centers().points)
    rect = pv.RectilinearGrid(np.array([10.0, 12, 15]), y, z)
    rect.cell_data["au"] = np.arange(4.0)
    rect.save(tmp_path / "i.vtr")
    back = bt.read_vtk(tmp_path / "i.vtr")
    assert isinstance(back, bt.PointSet)
    np.testing.assert_allclose(back.coords, rect.cell_centers().points)

    turned = grid(rotation=(25, 0, 0))
    bt.write_vtk(tmp_path / "t.vti", turned)
    pv.read(tmp_path / "t.vti").cast_to_structured_grid().save(tmp_path / "t.vts")
    back = bt.read_vtk(tmp_path / "t.vts")
    assert isinstance(back, bt.BlockModel)
    np.testing.assert_allclose(back.coords, turned.coords, atol=1e-9)
    np.testing.assert_array_equal(back["au"], turned["au"])
