import subprocess
import sys

import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(5)


@pytest.fixture
def pv():
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    return pv


def test_importing_boitata_does_not_import_pyvista():
    code = "import sys, boitata; assert 'pyvista' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_pyvista_names_pip_and_conda(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(
        ImportError, match=r"pip install 'boitata\[3d\]' or conda install -c conda-forge pyvista"
    ):
        bt.plot3d.to_pyvista(bt.PointSet(np.zeros((1, 3))))


def test_point_set_keeps_coords_and_attributes(pv):
    xyz = rng.uniform(0, 100, (50, 3))
    points = bt.PointSet(xyz, {"grade": xyz[:, 0], "rock": ["a", "b"] * 25})
    mesh = bt.plot3d.to_pyvista(points)
    assert mesh.n_points == 50
    np.testing.assert_allclose(mesh.points, xyz)
    np.testing.assert_allclose(mesh.point_data["grade"], xyz[:, 0])
    assert list(mesh.point_data["rock"][:2]) == ["a", "b"]


GRID = {
    "origin": (100.0, 200.0, 50.0),
    "size": (10.0, 5.0, 2.0),
    "count": (4, 3, 2),
    "rotation": (30.0, 20.0, 10.0),
}


def test_regular_block_model_matches_centroids_and_values(pv):
    model = bt.BlockModel(**GRID, attributes={"v": np.arange(24.0)})
    grid = bt.plot3d.to_pyvista(model)
    assert isinstance(grid, pv.ImageData) and grid.n_cells == 24
    np.testing.assert_allclose(grid.cell_centers().points, model.centroids, atol=1e-9)
    np.testing.assert_array_equal(grid.cell_data["v"], model["v"])
    np.testing.assert_allclose(grid.compute_cell_sizes()["Volume"], 100.0)


def test_masked_block_model_keeps_only_its_cells(pv):
    model = bt.BlockModel(**GRID, attributes={"v": np.arange(24.0)})
    masked = model.mask(model["v"] % 3 == 0)
    grid = bt.plot3d.to_pyvista(masked)
    assert grid.n_cells == len(masked) == 8
    np.testing.assert_allclose(grid.cell_centers().points, masked.centroids, atol=1e-9)
    np.testing.assert_array_equal(grid.cell_data["v"], masked["v"])


def test_sub_blocks_match_centroids_and_volumes(pv):
    parent = np.array([0, 0, 5, 23], dtype=np.uint64)
    extents = np.array(
        [[0, 0, 0, 0.5, 1, 1], [0.5, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0.25, 0.5, 0, 0.75, 1, 0.5]]
    )
    model = bt.BlockModel.subblocked(**GRID, parent=parent, extents=extents, attributes={"v": [1.0, 2, 3, 4]})
    grid = bt.plot3d.to_pyvista(model)
    assert grid.n_cells == 4
    np.testing.assert_allclose(grid.cell_centers().points, model.centroids, atol=1e-9)
    np.testing.assert_allclose(grid.compute_cell_sizes()["Volume"], model.volumes)
    np.testing.assert_array_equal(grid.cell_data["v"], [1, 2, 3, 4])


def test_mesh_keeps_triangles_bounds_and_attributes(pv):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    mesh = (
        bt.Mesh(vertices, triangles)
        .with_face_column("f", [1.0, 2, 3, 4])
        .with_vertex_column("h", vertices[:, 2])
    )
    poly = bt.plot3d.to_pyvista(mesh)
    assert poly.n_cells == 4 and poly.n_points == 4
    np.testing.assert_allclose(poly.bounds, [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(poly.regular_faces, triangles)
    np.testing.assert_array_equal(poly.cell_data["f"], [1, 2, 3, 4])
    np.testing.assert_array_equal(poly.point_data["h"], vertices[:, 2])
    assert poly.volume == pytest.approx(mesh.volume)


def test_drillholes_become_one_polyline_per_hole(pv):
    collar = {"HOLE_ID": ["a", "b"], "X": [0.0, 50], "Y": [0.0, 0], "Z": [100.0, 100]}
    survey = {
        "HOLE_ID": ["a", "a", "b"],
        "DEPTH": [0.0, 60, 0],
        "AZIMUTH": [0.0, 10, 90],
        "DIP": [60.0, 70, 90],
    }
    intervals = {"HOLE_ID": ["a", "b"], "FROM": [0.0, 0], "TO": [100.0, 40]}
    dh = bt.Drillholes(collar, survey, intervals)
    traces = bt.plot3d.to_pyvista(dh)
    paths = dh.paths()
    assert traces.n_lines == 2 and traces.n_points == len(paths)
    np.testing.assert_allclose(traces.points[:, 2], paths["z"])
    np.testing.assert_allclose(traces.point_data["depth"], paths["depth"])


def test_plots_render_off_screen(pv):
    model = bt.BlockModel(**GRID, attributes={"v": np.arange(24.0)})
    plotter = bt.plot3d.slices(model, values="v")
    bt.plot3d.plot(bt.PointSet(model.centroids, {"v": model["v"]}), values="v", plotter=plotter)
    assert sum(isinstance(a, pv.Actor) for a in plotter.renderer.actors.values()) == 2
    image = plotter.screenshot(return_img=True)
    plotter.close()
    assert image.ndim == 3 and image.std() > 0
    with pytest.raises(TypeError):
        bt.plot3d.to_pyvista(np.zeros(3))


def layers(scene, pv):
    scene.render()
    return [a.mapper for a in scene.renderer.actors.values() if isinstance(a, pv.Actor)]


def drawn(mapper, pv):
    return pv.wrap(mapper.GetInputAlgorithm().GetInputDataObject(0, 0))


def test_scene_never_renders_nulls_in_any_style(pv):
    v = np.where(np.arange(24) % 4 == 0, np.nan, np.arange(24.0))
    model = bt.BlockModel(**GRID, attributes={"v": v})
    points = bt.PointSet(model.centroids, {"v": v, "rock": ["a", None, "b", "c"] * 6})
    for style in ("surface", "wireframe", "points"):
        scene = bt.plot3d.Scene(off_screen=True).add(model, "v", style=style).add(points, "rock", style=style)
        blocks, dots = (drawn(m, pv) for m in layers(scene, pv))
        assert blocks.n_cells == 18 and np.isfinite(blocks.cell_data["v"]).all()
        assert dots.n_points == 18
        scene.close()
    empty = bt.plot3d.Scene(off_screen=True).add(model.mask(np.isnan(v)), "v")
    assert layers(empty, pv) == []


def test_scene_shares_one_color_map_and_range_per_variable(pv):
    def points(**columns):
        n = len(next(iter(columns.values())))
        return bt.PointSet(rng.uniform(0, 1, (n, 3)), columns)

    scene = bt.plot3d.Scene(off_screen=True)
    scene.add(points(g=np.array([1.0, 2, 3, 4, np.nan])), "g", cmap="cividis")
    scene.add(points(g=np.array([0.0, 5, 6, 7, 8])), "g")
    scene.add(points(r=["b", "a", "b"]), "r")
    scene.add(points(r=["c", "a", None]), "r")
    mappers = layers(scene, pv)
    assert all(m.lookup_table is scene.colors["g"] for m in mappers[:2])
    assert all(m.lookup_table is scene.colors["r"] for m in mappers[2:])
    assert scene.colors["g"].scalar_range == (0, 8) and scene.colors["g"].cmap.name == "cividis"
    assert scene.colors["r"].annotations == {0: "a", 1: "b", 2: "c"} and len(scene.scalar_bars) == 2
    np.testing.assert_array_equal(drawn(mappers[3], pv).point_data["r"], [2, 0])
    bt.plot3d.plot(points(g=np.array([-1.0, 9])), "g", plotter=scene.plotter)
    assert scene.colors["g"].scalar_range == (-1, 9)
    scene.add(points(g=np.array([-5.0, 20])), "g", clim=(0, 10))
    scene.add(points(g=np.array([-5.0, 30])), "g")
    assert scene.colors["g"].scalar_range == (0, 10)
    image = scene.screenshot(return_img=True)
    scene.close()
    assert image.std() > 0


def test_scene_opens_a_page_in_the_browser(pv, monkeypatch):
    pytest.importorskip("trame_pyvista")
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    scene = bt.plot3d.Scene(off_screen=True).add(bt.PointSet(np.eye(3), {"g": [1.0, 2, 3]}), "g")
    path = scene.show(browser=True)
    scene.close()
    assert opened == [path.as_uri()] and path.stat().st_size > 0
