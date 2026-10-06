import io
import re
import sys
from pathlib import Path

import boitata as bt
import numpy as np
import pytest
from boitata._scene import Scene

rng = np.random.default_rng(11)
ROOT = Path(__file__).resolve().parents[2]


def buffer(scene, key, dtype):
    return np.frombuffer(scene._buffers[key], dtype=dtype)


def column(scene, layer, name):
    spec = next(c for c in scene._spec()["layers"][layer]["columns"] if c["name"] == name)
    dtype = np.int32 if spec["type"] == "text" else np.float32
    return spec, buffer(scene, spec["buffer"], dtype)


def test_nulls_are_masked_and_never_enter_ranges():
    points = bt.PointSet(
        rng.uniform(0, 10, (4, 3)),
        {"grade": [1.0, np.nan, 3.0, -2.0], "rock": ["a", None, "b", "a"], "empty": [np.nan] * 4},
    )
    scene = Scene().add(points, "grade")
    spec, grade = column(scene, 0, "grade")
    assert np.isnan(grade[1]) and np.isfinite(grade[[0, 2, 3]]).all()
    assert (spec["min"], spec["max"]) == (-2.0, 3.0)
    spec, rock = column(scene, 0, "rock")
    assert spec["categories"] == ["a", "b"] and rock.tolist() == [0, -1, 1, 0]
    assert "empty" not in [c["name"] for c in scene._spec()["layers"][0]["columns"]]
    with pytest.raises(ValueError, match="no valid values"):
        Scene().add(points, "empty")


def test_coordinates_round_trip_through_the_local_origin():
    xyz = rng.uniform(0, 100, (50, 3)) + [654_321.0, 7_654_321.0, 1_200.0]
    scene = Scene().add(bt.PointSet(xyz))
    origin = np.array(scene._spec()["origin"])
    local = buffer(scene, "l0/positions", np.float32).reshape(-1, 3)
    assert np.abs(local).max() < 100
    np.testing.assert_allclose(local + origin, xyz, atol=1e-4)


def test_columns_limits_what_is_sent_and_keeps_values():
    points = bt.PointSet(rng.uniform(0, 1, (5, 3)), {"a": np.ones(5), "b": np.ones(5), "c": ["x"] * 5})
    names = [c["name"] for c in Scene().add(points, "b", columns=["c"])._spec()["layers"][0]["columns"]]
    assert sorted(names) == ["b", "c"]
    with pytest.raises(KeyError, match="nope"):
        Scene().add(points, columns=["nope"])


def corners_from_payload(scene, layer=0):
    spec = scene._spec()
    centers = buffer(scene, f"l{layer}/centers", np.float32).reshape(-1, 3) + np.array(spec["origin"])
    sizes = buffer(scene, f"l{layer}/sizes", np.float32).reshape(-1, 3)
    axes = np.array(spec["layers"][layer]["axes"])
    bits = np.array([[(c >> a) & 1 for a in range(3)] for c in range(8)]) - 0.5
    return centers[:, None] + np.einsum("ck,nk,kd->ncd", bits, sizes, axes)


def test_sub_block_sizes_and_rotated_axes_match_corners():
    parent = np.array([0, 0, 3, 5], dtype=np.uint64)
    extents = [
        [0, 0, 0, 0.5, 1, 1],
        [0.5, 0, 0, 1, 1, 0.25],
        [0, 0, 0, 1, 1, 1],
        [0.25, 0.5, 0, 0.75, 1, 0.5],
    ]
    model = bt.BlockModel.subblocked(
        (500.0, 800.0, 100.0), (10.0, 6.0, 4.0), (3, 2, 2), parent, extents, rotation=(30.0, 20.0, 10.0)
    )
    scene = Scene().add(model.with_column("v", np.arange(4.0)), "v")
    np.testing.assert_allclose(corners_from_payload(scene), model.corners, atol=1e-3)


def test_masked_and_regular_models_match_corners():
    model = bt.BlockModel((0.0, 0.0, 0.0), (5.0, 5.0, 2.0), (4, 3, 2), rotation=(45.0, 0.0, 0.0))
    np.testing.assert_allclose(corners_from_payload(Scene().add(model)), model.corners, atol=1e-4)
    masked = model.mask(np.arange(24) % 3 == 0)
    np.testing.assert_allclose(corners_from_payload(Scene().add(masked)), masked.corners, atol=1e-4)


def test_drill_hole_intervals_follow_the_trace_through_survey_stations():
    collar = {"HOLE_ID": ["A"], "X": [0.0], "Y": [0.0], "Z": [100.0]}
    survey = {
        "HOLE_ID": ["A"] * 3,
        "DEPTH": [0.0, 20.0, 40.0],
        "AZIMUTH": [0.0, 30.0, 60.0],
        "DIP": [60.0, 70.0, 80.0],
    }
    intervals = {"HOLE_ID": ["A", "A"], "FROM": [5.0, 30.0], "TO": [25.0, 38.0], "CU": [1.5, np.nan]}
    holes = bt.Drillholes(collar, survey, intervals)
    scene = Scene().add(holes, "CU")
    ends = buffer(scene, "l0/positions", np.float32).reshape(-1, 2, 3) + scene._spec()["origin"]
    rows = buffer(scene, "l0/rows", np.uint32)
    assert rows.tolist() == [0, 0, 1]
    expected = holes.at(["A"] * 4, [5.0, 20.0, 25.0, 38.0])
    np.testing.assert_allclose(ends[0], expected[[0, 1]], atol=1e-3)
    np.testing.assert_allclose(ends[1], expected[[1, 2]], atol=1e-3)
    np.testing.assert_allclose(ends[2, 1], expected[3], atol=1e-3)
    _, cu = column(scene, 0, "CU")
    assert cu[0] == 1.5 and np.isnan(cu[1])


def test_mesh_attributes_keep_their_association():
    mesh = bt.Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2], [1, 3, 2]])
    mesh = mesh.with_vertex_column("h", [0.0, 1.0, 2.0, np.nan]).with_face_column("zone", ["a", "b"])
    spec = Scene().add(mesh, "zone")._spec()["layers"][0]
    assert {c["name"]: c["on"] for c in spec["columns"]} == {"h": "vertex", "zone": "face"}
    assert spec["kind"] == "mesh" and spec["representation"] == "surface"


@pytest.mark.parametrize(
    ("theme", "error"),
    [
        ("sepia", ValueError),
        ({"bogus": "#fff"}, ValueError),
        ({"base": "sepia"}, ValueError),
        ({"background": 3}, TypeError),
        (3, TypeError),
    ],
)
def test_theme_validation(theme, error):
    with pytest.raises(error):
        Scene(theme=theme)


def test_theme_dict_and_view_reach_the_viewer():
    scene = Scene(theme={"base": "dark", "accent": "#00ff00"}).view(azimuth=-90, dip=90)
    spec = scene._spec()
    assert spec["theme"] == {"base": "dark", "accent": "#00ff00"}
    assert spec["view"] == {"azimuth": 270.0, "dip": 90.0}
    with pytest.raises(ValueError):
        scene.view(dip=120)


def test_axis_titles_follow_crs_and_unit():
    points = bt.PointSet(np.zeros((1, 3)), crs="EPSG:32722", length_unit="m")
    assert Scene().add(points)._spec()["axes"] == ["Easting (m)", "Northing (m)", "Elevation (m)"]
    assert Scene().add(bt.PointSet(np.zeros((1, 3))))._spec()["axes"] == ["X", "Y", "Z"]


def test_save_writes_a_self_contained_page(tmp_path):
    scene = Scene().add(bt.PointSet(rng.uniform(0, 1, (3, 3)), {"v": [1.0, 2.0, 3.0]}), "v")
    text = scene.save(tmp_path / "scene.html").read_text(encoding="utf-8")
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)=", text, re.IGNORECASE)
    assert not re.search(r"@import|url\(\s*['\"]?https?:", text, re.IGNORECASE)
    assert "scene-viewer" in text and "scene-data" in text


def test_large_payloads_warn(monkeypatch):
    monkeypatch.setattr("boitata._scene._WARN_BYTES", 100)
    with pytest.warns(UserWarning, match="MB of data"):
        Scene().add(bt.PointSet(rng.uniform(0, 1, (20, 3))))


def test_notebook_display_uses_the_widget_or_falls_back_to_an_iframe(monkeypatch):
    pytest.importorskip("anywidget")
    scene = Scene().add(bt.PointSet(np.zeros((1, 3))))
    data, _ = scene._repr_mimebundle_()
    assert "application/vnd.jupyter.widget-view+json" in data
    monkeypatch.setitem(sys.modules, "anywidget", None)
    monkeypatch.setattr("boitata._scene._widget_class", lambda: __import__("anywidget"))
    bundle = scene._repr_mimebundle_()
    assert bundle["text/html"].startswith("<iframe srcdoc=") and "boitata[3d]" in bundle["text/html"]


def lut_rgb(name, value, lo, hi):
    table = re.search(rf'{name}: "([0-9a-f]+)"', (ROOT / "js/viewer/src/colormaps.ts").read_text()).group(1)
    i = min(255, max(0, int((value - lo) / (hi - lo) * 256)))
    return tuple(int(table[6 * i + 2 * k : 6 * i + 2 * k + 2], 16) for k in range(3))


@pytest.mark.parametrize(
    ("data", "valid"),
    [
        (bt.PointSet(np.zeros((2, 3))), "points, spheres"),
        (bt.BlockModel((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 1, 1)), "cells, wireframe, points"),
        (bt.Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]), "surface, wireframe, points"),
    ],
)
def test_representation_is_validated_per_kind(data, valid):
    names = valid.split(", ")
    assert Scene().add(data)._spec()["layers"][0]["representation"] == names[0]
    assert Scene().add(data, representation=names[-1])._spec()["layers"][0]["representation"] == names[-1]
    with pytest.raises(ValueError, match=valid):
        Scene().add(data, representation="tubes")


def test_sizes_reach_the_viewer_and_apply_only_where_they_mean_something():
    points = bt.PointSet(np.zeros((2, 3)))
    layer = Scene().add(points, representation="spheres", radius=2, point_size=9)._spec()["layers"][0]
    assert (layer["radius"], layer["pointSize"], layer["lineWidth"]) == (2.0, 9.0, None)
    with pytest.raises(ValueError, match="line_width does not apply"):
        Scene().add(points, line_width=2)
    model = bt.BlockModel((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 1, 1))
    with pytest.raises(ValueError, match="radius does not apply"):
        Scene().add(model, radius=1)
    for bad in (0, -1, np.inf, True, "3"):
        with pytest.raises(ValueError, match="positive number"):
            Scene().add(points, point_size=bad)


@pytest.mark.parametrize(("value", "sent"), [("auto", "auto"), ("full", "full"), (0.3, 0.3), (1, 1.0)])
def test_motion_quality_reaches_the_viewer(value, sent):
    assert Scene(motion_quality=value)._spec()["motion"] == sent


@pytest.mark.parametrize("value", ["fast", 0, 1.5, -0.1, True, None])
def test_motion_quality_is_validated(value):
    with pytest.raises(ValueError, match="motion_quality"):
        Scene(motion_quality=value)


def test_theme_takes_a_halo_color():
    assert Scene(theme={"halo": "#808080"})._spec()["theme"] == {"halo": "#808080"}


def straight_hole(cu):
    """Hole A runs east, level, at y = 3, off the grid lines, with three 10 m intervals; B and C, all null, frame it in y."""
    collar = {"HOLE_ID": ["A", "B", "C"], "X": [0.0] * 3, "Y": [3.0, -17.0, 23.0], "Z": [100.0] * 3}
    survey = {
        "HOLE_ID": ["A", "A", "B", "B", "C", "C"],
        "DEPTH": [0.0, 30.0] * 3,
        "AZIMUTH": [90.0] * 6,
        "DIP": [0.0] * 6,
    }
    intervals = {
        "HOLE_ID": ["A"] * 3 + ["B", "C"],
        "FROM": [0.0, 10.0, 20.0, 0.0, 0.0],
        "TO": [10.0, 20.0, 30.0, 30.0, 30.0],
        "CU": [*cu, np.nan, np.nan],
    }
    return bt.Drillholes(collar, survey, intervals)


def test_drill_hole_points_sit_at_interval_midpoints():
    holes = straight_hole([0.2, np.nan, 0.8])
    scene = Scene().add(holes, "CU", representation="points")
    middles = buffer(scene, "l0/midpoints", np.float32).reshape(-1, 3) + scene._spec()["origin"]
    expected = holes.at(["A", "A", "A", "B", "C"], [5.0, 15.0, 25.0, 15.0, 15.0])
    np.testing.assert_allclose(middles, expected, atol=1e-3)


def rendered_pixels(scene, points, tmp_path):
    """Colors the saved page shows at world `points`, in a headless browser."""
    sync_api = pytest.importorskip("playwright.sync_api")
    image = pytest.importorskip("PIL.Image")
    page = scene.save(tmp_path / "scene.html")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except sync_api.Error as e:
            pytest.skip(f"no chromium for playwright: {e}")
        tab = browser.new_page(viewport={"width": 900, "height": 600})
        tab.goto(page.as_uri())
        tab.wait_for_function("window.scene !== undefined", timeout=60_000)
        tab.wait_for_timeout(300)
        at = tab.evaluate(
            """(points) => points.map(([x, y, z]) => {
                const s = window.scene, o = s.spec.origin;
                const v = s.camera.position.clone().set(x - o[0], y - o[1], z - o[2]).project(s.camera);
                return [Math.round((v.x + 1) / 2 * s.width), Math.round((1 - v.y) / 2 * s.height)];
            })""",
            points,
        )
        shot = image.open(io.BytesIO(tab.screenshot())).convert("RGB")
        browser.close()
    return [shot.getpixel(tuple(xy)) for xy in at]


@pytest.mark.parametrize(
    ("representation", "size"),
    [("lines", {"line_width": 12}), ("tubes", {"radius": 1.0}), ("points", {"point_size": 24})],
)
def test_rendered_interval_shows_its_exact_lut_color_and_a_null_paints_nothing(
    tmp_path, representation, size
):
    holes = straight_hole([0.2, np.nan, 0.8])
    scene = Scene(theme="light").add(holes, "CU", representation=representation, clim=(0.0, 1.0), **size)
    scene.view(azimuth=0, dip=90)
    first, null, last = rendered_pixels(scene, [[5, 3, 100], [13, 3, 100], [25, 3, 100]], tmp_path)
    shading = 2 if representation == "tubes" else 0
    for got, value in ((first, 0.2), (last, 0.8)):
        assert np.abs(np.subtract(got, lut_rgb("viridis", value, 0.0, 1.0))).max() <= shading
    assert null == (255, 255, 255)


def test_rendered_block_shows_its_exact_lut_color_and_a_null_paints_nothing(tmp_path):
    model = bt.BlockModel((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (3, 1, 1))
    model = model.with_column("v", [0.2, np.nan, 0.8])
    scene = Scene(theme="light").add(model, "v", clim=(0.0, 1.0)).view(dip=90)
    first, null, last = rendered_pixels(
        scene, [[3.5, 6.5, 10.0], [16.5, 6.5, 10.0], [23.5, 6.5, 10.0]], tmp_path
    )
    assert first == lut_rgb("viridis", 0.2, 0.0, 1.0)
    assert last == lut_rgb("viridis", 0.8, 0.0, 1.0)
    assert null == (255, 255, 255)
