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


def test_scene_colors_meshes_by_vertex_or_face_on_the_range_of_points(pv):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    solid = (
        bt.Mesh(vertices, triangles)
        .with_vertex_column("g", [1.0, 2, 3, np.nan])
        .with_face_column("rock", ["a", "b", None, "a"])
    )
    points = bt.PointSet(vertices + 2, {"g": [0.0, 5, 6, 9]})
    scene = bt.plot3d.Scene(off_screen=True).add(solid, "g").add(solid, "rock", opacity=0.5).add(points, "g")
    by_vertex, by_face, dots = (drawn(m, pv) for m in layers(scene, pv))
    assert by_vertex.n_points == 3 and by_vertex.n_cells == 1
    assert by_face.n_cells == 3 and list(by_face.cell_data["rock"]) == [0, 1, 0]
    assert dots.n_points == 4 and scene.colors["g"].scalar_range == (0, 9)
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


def test_drillholes_render_as_one_tube_mesh_without_null_intervals(pv, monkeypatch):
    collar = {"HOLE_ID": ["a", "b", "c"], "X": [0.0, 50, 100], "Y": [0.0, 0, 0], "Z": [100.0, 100, 100]}
    survey = {"HOLE_ID": ["a", "b", "c"], "DEPTH": [0.0, 0, 0], "AZIMUTH": [0.0, 0, 0], "DIP": [90.0, 90, 90]}
    intervals = {
        "HOLE_ID": ["a", "a", "b", "b", "c"],
        "FROM": [0.0, 10, 0, 10, 0],
        "TO": [10.0, 20, 10, 20, 30],
        "CU": [1.0, np.nan, 2, 3, 4],
    }
    dh = bt.Drillholes(collar, survey, intervals)
    scene = bt.plot3d.Scene(off_screen=True)
    labels = []
    monkeypatch.setattr(scene.plotter, "add_point_labels", lambda *a, **k: labels.append(a))
    scene.add(dh, "CU", radius=1.0, labels=True)
    (mapper,) = layers(scene, pv)
    tubes = drawn(mapper, pv)
    assert sorted(np.unique(tubes.cell_data["CU"])) == [1, 2, 3, 4]
    np.testing.assert_allclose(tubes.bounds[4:], [70, 100], atol=0.1)
    assert tubes.points[np.abs(tubes.points[:, 0]) < 2, 2].min() == pytest.approx(90)
    ((collars, names),) = labels
    np.testing.assert_allclose(collars, [[0, 0, 100], [50, 0, 100], [100, 0, 100]])
    assert names == ["a", "b", "c"]
    scene.close()
    survey = {key: values * 2 for key, values in survey.items()} | {"DEPTH": [0.0] * 3 + [30.0] * 3}
    traces = bt.plot3d.Scene(off_screen=True).add(bt.Drillholes(collar, survey), radius=1.0)
    np.testing.assert_allclose(drawn(layers(traces, pv)[0], pv).bounds, [-1, 101, -1, 1, 70, 100], atol=0.01)
    traces.close()


def volumes(scene, pv):
    return [a for a in scene.renderer.actors.values() if isinstance(a, pv.Volume)]


def test_volume_fills_each_valid_block_and_hides_nulls(pv):
    v = np.where(np.arange(24) % 4 == 0, np.nan, np.arange(24.0))
    model = bt.BlockModel(**GRID, attributes={"v": v, "rock": ["a", None, "b"] * 8})
    scene = bt.plot3d.Scene(off_screen=True).add(model, "v", style="volume", opacity=1.0)
    scene.add(bt.PointSet(model.centroids, {"v": v - 10}), "v")
    (actor,) = volumes(scene, pv)
    grid = actor.mapper.dataset
    assert isinstance(grid, pv.ImageData) and grid.dimensions == (6, 5, 4)
    field = grid.point_data["v"]
    valid = field > field.min()
    np.testing.assert_allclose(grid.points[valid], model.centroids[np.isfinite(v)], atol=1e-4)
    shape = actor.prop.GetScalarOpacity()
    assert shape.GetValue(field.min()) == 0 and all(shape.GetValue(x) == 1 for x in field[valid])
    assert actor.mapper.lookup_table is scene.colors["v"] and scene.colors["v"].scalar_range == (-9, 23)
    assert actor.prop.GetRGBTransferFunction().GetRange() == pytest.approx((-9, 23), abs=1)
    scene.add(model, "rock", style="volume")
    rock = volumes(scene, pv)[1].mapper.dataset.point_data["rock"]
    assert sorted(set(rock[rock >= 0])) == [0, 1] and scene.colors["rock"].GetNumberOfAnnotatedValues() == 2
    assert scene.screenshot(return_img=True).std() > 0
    scene.close()
    with pytest.raises(ValueError, match="to_regular"):
        bt.plot3d.Scene(off_screen=True).add(model.mask(np.isfinite(v)), "v", style="volume")


def test_volume_above_gpu_memory_draws_coarse_while_moving(pv, monkeypatch):
    monkeypatch.setattr(pv.SmartVolumeMapper, "GetMaxMemoryInBytes", lambda self: 100)
    model = bt.BlockModel(**GRID, attributes={"v": np.arange(24.0)})
    scene = bt.plot3d.Scene(off_screen=True).add(model, "v", style="volume")
    mapper = volumes(scene, pv)[0].mapper
    window = scene.ren_win
    for rate, points, mode in ((30.0, 18, mapper.GPURenderMode), (0.0001, 120, mapper.RayCastRenderMode)):
        window.SetDesiredUpdateRate(rate)
        window.Render()
        assert pv.wrap(mapper.GetInput()).n_points == points and mapper.GetRequestedRenderMode() == mode
    scene.close()
