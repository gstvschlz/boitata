import csv
import subprocess
import sys
from pathlib import Path

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


def outer_faces(pv, model, values="v"):
    scene = bt.plot3d.Scene(off_screen=True).add(model, values)
    ((solid, *_, actor),) = scene._layers
    scene.close()
    return solid, drawn(actor.mapper, pv)


def test_solid_masked_cube_draws_its_outer_faces_only(pv):
    n = 4
    model = bt.BlockModel((0.0, 0, 0), (1.0, 2, 3), (n, n, n), rotation=(30.0, 20, 10))
    centre = 1 + n + n * n
    v = np.arange(n**3.0)
    solid, faces = outer_faces(pv, model.with_column("v", v).mask(np.ones(n**3, dtype=bool)))
    assert faces.n_cells == 6 * n**2 and faces.area == pytest.approx(22 * n**2)
    np.testing.assert_allclose(faces.bounds, solid.cells.bounds, atol=1e-9)
    np.testing.assert_allclose(solid.bounds, solid.cells.bounds, atol=1e-9)
    v[centre] = np.nan
    _, hollow = outer_faces(pv, model.with_column("v", v).mask(np.ones(n**3, dtype=bool)))
    assert hollow.n_cells == 6 * n**2 + 6 and np.isfinite(hollow.cell_data["v"]).all()
    pair, faces = outer_faces(pv, model.with_column("v", v).mask(np.arange(n**3) < 2))
    assert faces.n_cells == 10
    normals = faces.compute_normals(cell_normals=True, point_normals=False).cell_data["Normals"]
    outward = faces.cell_centers().points - pair.model.centroids.mean(axis=0)
    assert (np.einsum("ij,ij->i", normals, outward) > 0).all()


def test_sub_blocks_draw_faces_not_shared_whole(pv):
    n = 3
    parent = np.repeat(np.arange(n**3, dtype=np.uint64), 2)
    extents = np.tile([[0, 0, 0, 0.5, 1, 1], [0.5, 0, 0, 1, 1, 1]], (n**3, 1))
    rock = ["a", "b"] * n**3
    grid = {"origin": (0.0, 0, 0), "size": (1.0, 1, 1), "count": (n, n, n), "rotation": (30.0, 0, 0)}
    model = bt.BlockModel.subblocked(**grid, parent=parent, extents=extents, attributes={"rock": rock})
    _, faces = outer_faces(pv, model, "rock")
    assert faces.n_cells == 2 * n * n + 2 * 2 * (2 * n * n)
    assert set(faces.cell_data["rock"]) == {0, 1}
    free = bt.BlockModel.subblocked(
        (0.0, 0, 0),
        (1.0, 1, 1),
        (2, 1, 1),
        parent=np.array([0, 1, 1], dtype=np.uint64),
        extents=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0.3, 1], [0, 0.3, 0, 1, 1, 1]],
        attributes={"v": [1.0, 2, 3]},
    )
    _, faces = outer_faces(pv, free)
    assert faces.n_cells == 16 and faces.area == pytest.approx(10 + 2)


def test_masked_model_cuts_its_cells_and_draws_points_from_them(pv):
    model = bt.BlockModel((0.0, 0, 0), (1.0, 1, 1), (4, 3, 2), attributes={"v": np.arange(24.0)})
    model = model.mask(np.arange(24) != 5)
    scene = bt.plot3d.Scene(off_screen=True).add(model, "v").add(model, "v", style="points")
    assert drawn(scene._layers[1][-1].mapper, pv).n_points == 8 * 23
    (cut, _) = scene.section(model.centroids[0], azimuth=0, dip=0)._cuts
    assert drawn(cut.mapper, pv).n_cells == 11
    empty = model.with_column("v", np.full(23, np.nan))
    assert scene.add(empty, "v")._layers[-1][-1] is None
    scene.close()


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
    image = plotter.plotter.screenshot(return_img=True)
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
    image = scene.plotter.screenshot(return_img=True)
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
    image = scene.plotter.screenshot(return_img=True)
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
    assert scene.plotter.screenshot(return_img=True).std() > 0
    scene.close()
    with pytest.raises(ValueError, match="to_regular"):
        bt.plot3d.Scene(off_screen=True).add(model.mask(np.isfinite(v)), "v", style="volume")


def interact(scene, moving):
    style = scene.iren.interactor.GetInteractorStyle()
    style.StartRotate() if moving else style.EndRotate()
    scene.ren_win.Render()


def test_volume_above_gpu_memory_draws_coarse_while_moving(pv, monkeypatch):
    monkeypatch.setattr(pv.SmartVolumeMapper, "GetMaxMemoryInBytes", lambda self: 100)
    model = bt.BlockModel(**GRID, attributes={"v": np.arange(24.0)})
    scene = bt.plot3d.Scene(off_screen=True).add(model, "v", style="volume")
    (full,) = volumes(scene, pv)
    assert full.mapper.GetRequestedRenderMode() == full.mapper.RayCastRenderMode
    interact(scene, True)
    coarse = scene._fast[0]
    assert coarse.GetVisibility() and not full.GetVisibility() and len(volumes(scene, pv)) == 2
    assert coarse.mapper.dataset.n_points < full.mapper.dataset.n_points
    assert coarse.mapper.lookup_table is scene.colors["v"]
    interact(scene, False)
    assert full.GetVisibility() and not coarse.GetVisibility()
    scene.close()


def test_motion_quality_draws_a_cheaper_copy_of_every_layer_while_moving(pv):
    v = np.where(np.arange(512) % 7 == 0, np.nan, np.arange(512.0))
    model = bt.BlockModel(
        (0.0, 0, 0), (1.0, 1, 1), (8, 8, 8), attributes={"v": v, "rock": ["a", "b", None, "c"] * 128}
    )
    points = bt.PointSet(rng.uniform(0, 8, (400, 3)), {"v": rng.uniform(0, 600, 400)})
    collar = {"HOLE_ID": ["a", "b"], "X": [2.0, 6], "Y": [4.0, 4], "Z": [8.0, 8]}
    survey = {"HOLE_ID": ["a", "b"], "DEPTH": [0.0, 0], "AZIMUTH": [0.0, 0], "DIP": [90.0, 90]}
    intervals = {"HOLE_ID": ["a", "a", "b"], "FROM": [0.0, 2, 0], "TO": [2.0, 4, 6], "v": [1.0, np.nan, 3]}
    holes = bt.Drillholes(collar, survey, intervals)
    shell = pv.Sphere(radius=3, center=(4, 4, 4), theta_resolution=60, phi_resolution=60)
    shell.point_data["v"] = shell.points[:, 2] * 50
    scene = bt.plot3d.Scene(off_screen=True, motion_quality=0.25)
    scene.add(model, "v").add(model.mask(np.isfinite(v)), "rock", style="wireframe").add(points, "v")
    scene.add(holes, "v", radius=0.2).add(shell, "v", opacity=0.5)
    full = [actor for *_, actor in scene._layers]
    interact(scene, True)
    fast = [scene._fast[i] for i in range(5)]
    assert not any(a.GetVisibility() for a in full) and all(a.GetVisibility() for a in fast)
    for a, b in zip(full, fast, strict=True):
        assert drawn(b.mapper, pv).n_cells < 0.5 * drawn(a.mapper, pv).n_cells
    blocks, rocks, dots, lines, surface = (drawn(a.mapper, pv) for a in fast)
    assert blocks.n_cells <= 64 and np.isfinite(blocks.cell_data["v"]).all()
    assert set(rocks.cell_data["rock"]) <= {0, 1, 2} and fast[1].prop.style == "Wireframe"
    assert dots.n_points == 100 and not fast[2].prop.render_points_as_spheres
    assert lines.n_cells == 2 and sorted(lines.cell_data["v"]) == [1, 3]
    assert np.isfinite(surface.point_data["v"]).all() and fast[4].prop.opacity == 0.5
    assert all(a.mapper.lookup_table is scene.colors["v"] for i, a in enumerate(fast) if i != 1)
    assert fast[1].mapper.lookup_table is scene.colors["rock"] and len(scene.scalar_bars) == 2
    interact(scene, False)
    assert all(a.GetVisibility() for a in full) and not any(a.GetVisibility() for a in fast)
    assert scene.plotter.screenshot(return_img=True).std() > 0
    scene.close()


def test_motion_quality_auto_degrades_only_layers_above_the_budget(pv, monkeypatch):
    scene = bt.plot3d.Scene(off_screen=True).add(
        bt.PointSet(rng.uniform(0, 1, (100, 3)), {"g": np.arange(100.0)}), "g"
    )
    interact(scene, True)
    assert scene._fast == {0: None} and len(layers(scene, pv)) == 1
    monkeypatch.setattr(bt.plot3d, "_BUDGET", 50)
    scene.motion_quality = "auto"
    interact(scene, True)
    assert drawn(scene._fast[0].mapper, pv).n_points == 50
    scene.motion_quality = "full"
    interact(scene, True)
    assert scene._fast == {0: None} and len(layers(scene, pv)) == 1 and scene._layers[0][-1].GetVisibility()
    for bad in (0, 1.5, "fast", True):
        with pytest.raises(ValueError, match="motion_quality"):
            scene.motion_quality = bad
    scene.close()


def test_motion_quality_leaves_sections_alone(pv):
    scene = section_scene(pv)
    scene.motion_quality = 0.5
    interact(scene, True)
    scene.section((0.0, 1, 0), width=10)
    interact(scene, True)
    hidden = [a for *_, a in scene._layers] + [a for a in scene._fast.values() if a is not None]
    assert not any(a.GetVisibility() for a in hidden) and all(a.GetVisibility() for a in scene._cuts)
    scene.section(None)
    interact(scene, False)
    assert all(a.GetVisibility() for *_, a in scene._layers)
    scene.close()


def test_screenshot_scales_and_keeps_a_transparent_background(pv, tmp_path):
    from PIL import Image

    scene = bt.plot3d.Scene(off_screen=True, window_size=(120, 80))
    scene.add(bt.PointSet(np.eye(3), {"g": [1.0, 2, 3]}), "g")
    plain = scene.screenshot(tmp_path / "plain.png")
    clear = scene.screenshot(str(tmp_path / "clear.png"), scale=2, transparent=True)
    scene.close()
    assert plain.is_file() and Image.open(plain).size == (120, 80)
    rgba = np.asarray(Image.open(clear))
    assert rgba.shape == (160, 240, 4) and rgba[0, 0, 3] == rgba[-1, -1, 3] == 0 and rgba[..., 3].max() == 255


def section_scene(pv):
    collar = {"HOLE_ID": ["a", "b"], "X": [5.0, 5], "Y": [0.0, 30], "Z": [20.0, 20]}
    survey = {"HOLE_ID": ["a", "b"], "DEPTH": [0.0, 0], "AZIMUTH": [0.0, 0], "DIP": [90.0, 90]}
    holes = bt.Drillholes(collar, survey, {"HOLE_ID": ["a", "b"], "FROM": [0.0, 0], "TO": [30.0, 30]})
    model = bt.BlockModel((-20.0, -20, -20), (5.0, 5, 5), (8, 8, 8), attributes={"v": np.arange(512.0)})
    points = bt.PointSet([[0.0, 1, 0], [0, -3, 0], [0, 9, 0]], {"v": [1.0, 2, 3]})
    scene = bt.plot3d.Scene(off_screen=True)
    return (
        scene.add(model, "v")
        .add(holes, radius=0.5)
        .add(points, "v")
        .add(pv.Sphere(radius=10), style="wireframe")
    )


def test_section_cuts_every_layer_and_projects_holes_within_width(pv):
    scene = section_scene(pv).section((0.0, 1, 0), width=10)
    scene.render()
    blocks, holes, points, sphere = (drawn(a.mapper, pv) or pv.wrap(a.mapper.GetInput()) for a in scene._cuts)
    assert not any(a.GetVisibility() for *_, a in scene._layers)
    np.testing.assert_allclose(blocks.points[:, 1], 1, atol=1e-9)
    assert blocks.n_cells == 64 and np.isfinite(blocks.cell_data["v"]).all()
    np.testing.assert_allclose(holes.points[:, 1], 1, atol=1e-9)
    np.testing.assert_allclose(holes.bounds[:2], (4.5, 5.5), atol=0.01)
    np.testing.assert_allclose(holes.bounds[4:], (-10, 20), atol=0.01)
    np.testing.assert_allclose(points.points, [[0, 1, 0], [0, 1, 0]], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(sphere.points[:, ::2], axis=1), np.sqrt(99), rtol=0.02)
    assert scene.section(None)._cuts == [] and all(a.GetVisibility() for *_, a in scene._layers)
    scene.close()


def test_view_section_looks_normal_to_the_plane_in_parallel(pv):
    scene = section_scene(pv)
    with pytest.raises(RuntimeError):
        scene.view_section()
    scene.section((0.0, 0, 0), azimuth=30, dip=60).view_section()
    camera = scene.camera
    direction = np.subtract(camera.focal_point, camera.position)
    normal = bt.plot._frame(((0, 0, 0), 30, 60))[3]
    assert camera.parallel_projection
    np.testing.assert_allclose(direction / np.linalg.norm(direction), -normal, atol=1e-9)
    assert scene.plotter.screenshot(return_img=True).std() > 0
    scene.close()


def test_section_widget_cuts_through_the_center(pv):
    scene = section_scene(pv).section_widget(azimuth=0, dip=90)
    assert len(scene._cuts) == 3
    np.testing.assert_allclose(scene._plane[0], scene._extent()[0], atol=1e-9)
    assert abs(scene._plane[1]) % 180 == 0 and scene._plane[2] == 90
    scene.close()


def test_section_cuts_a_volume_as_a_surface(pv):
    model = bt.BlockModel(**GRID, attributes={"v": np.arange(24.0)})
    scene = bt.plot3d.Scene(off_screen=True).add(model, "v", style="volume", opacity="sigmoid")
    (cut,) = scene.section(model.centroids.mean(axis=0), azimuth=0, dip=0)._cuts
    assert isinstance(cut, pv.Actor) and cut.mapper.lookup_table is scene.colors["v"]
    scene.close()


@pytest.mark.slow
def test_viewer_benchmark_runs(pv, tmp_path):
    script = Path(__file__).parents[2] / "ci" / "bench_viewer.py"
    out = tmp_path / "bench.csv"
    cmd = [sys.executable, script, "--sizes", "1e3", "--holes", "10", "--frames", "3", "--csv", out]
    subprocess.run(cmd, check=True, timeout=300)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6 and all(float(row["fps"]) > 0 for row in rows)


def test_layers_get_names_and_kinds(pv):
    scene = section_scene(pv).add(pv.Sphere(), name="lens")
    assert scene._names == ["block model 1", "drill holes 1", "points 1", "surface 1", "lens"]
    assert scene._kinds == ["block models", "drill holes", "points", "surfaces", "surfaces"]
    scene.close()


def test_motion_quality_auto_degrades_a_layer_above_50k_cells(pv):
    model = bt.BlockModel((0.0, 0, 0), (1.0, 1, 1), (40, 40, 40), attributes={"v": np.arange(64000.0)})
    scene = bt.plot3d.Scene(off_screen=True).add(model, "v")
    interact(scene, True)
    assert drawn(scene._fast[0].mapper, pv).n_cells < 64000
    scene.close()


def test_section_along_a_polyline_cuts_each_segment_between_its_ends(pv):
    scene = section_scene(pv).section(points=[(-15, -15), (15, -15), (15, 15)], width=4)
    assert len(scene._pieces) >= 2
    for cut, center, u, at, *_ in scene._pieces:
        offset = cut.points[:, :2] - center[:2]
        along = offset @ u[:2]
        np.testing.assert_allclose(offset @ [u[1], -u[0]], 0, atol=1e-6)
        assert along.min() >= -1e-6 and along.max() <= 30 + 1e-6 and at in (0, 30)
    holes = [cut for cut, *_ in scene._pieces if cut.n_points and np.allclose(cut.points[:, 0], 5, atol=1)]
    assert not holes
    with pytest.raises(ValueError, match="points"):
        scene.section(points=[(0, 0)])
    scene.close()


def test_view_section_unfolds_a_polyline_onto_the_first_segment(pv):
    scene = section_scene(pv).section(points=[(-15, -15), (15, -15), (15, 15)]).view_section()
    assert scene._unfolded and scene.camera.parallel_projection
    for actor in scene._cuts:
        points = (drawn(actor.mapper, pv) or pv.wrap(actor.mapper.GetInput())).points
        np.testing.assert_allclose(points[:, 1], -15, atol=1e-6)
        assert points[:, 0].min() >= -15 - 1e-6 and points[:, 0].max() <= 45 + 1e-6
    scene.close()


def test_toggles_hide_a_layer_and_its_cuts(pv):
    scene = section_scene(pv).layer_toggles()
    scene._toggle(0, False)
    assert not scene._layers[0][-1].GetVisibility()
    interact(scene, False)
    assert not scene._layers[0][-1].GetVisibility()
    scene.section((0.0, 1, 0), width=10)
    assert len(scene._cuts) == 3 and all(cut.n_cells != 64 for cut, *_ in scene._pieces)
    scene._toggle(0, True)
    assert len(scene._cuts) == 4
    scene.section(None)
    assert all(a.GetVisibility() for *_, a in scene._layers)
    scene.close()


def test_drawer_clicks_snap_with_shift_and_enter_cuts(pv):
    scene = section_scene(pv).section_drawer(width=4)
    scene._draw_start()
    assert scene._drawer["on"] and scene.camera.parallel_projection
    interactor = scene.iren.interactor
    width, height = scene.window_size
    for x, y, shift in ((0.3, 0.3, 0), (0.7, 0.35, 1), (0.7, 0.7, 0)):
        interactor.SetEventPosition(int(x * width), int(y * height))
        interactor.SetShiftKey(shift)
        interactor.InvokeEvent("LeftButtonPressEvent")
    a, b, _ = scene._drawer["points"]
    np.testing.assert_allclose(b[1], a[1], atol=1e-9)
    scene._draw_finish()
    assert not scene._drawer["on"] and scene._path is not None and len(scene._path) == 3
    assert not scene.camera.parallel_projection and scene._cuts
    scene._clear()
    assert scene._plane is None
    scene.close()


def test_snap_keeps_45_degree_bearings():
    np.testing.assert_allclose(bt.plot3d._snap([0, 0], [10, 1]), [10, 0], atol=1e-9)
    np.testing.assert_allclose(bt.plot3d._snap([0, 0], [5, 6]), [5.5, 5.5], atol=1e-9)
    np.testing.assert_allclose(bt.plot3d._snap([1, 1], [1, -9]), [1, -9], atol=1e-9)
