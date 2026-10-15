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
