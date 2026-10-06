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
    """Hole A runs east, level, at y = 3, off the grid lines, with three 10 m intervals; B and C frame it in y."""
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
        "CU": [*cu, 0.5, 0.5],
    }
    return bt.Drillholes(collar, survey, intervals)


def test_drill_hole_points_sit_at_interval_midpoints():
    holes = straight_hole([0.2, np.nan, 0.8])
    scene = Scene().add(holes, "CU", representation="points")
    middles = buffer(scene, "l0/midpoints", np.float32).reshape(-1, 3) + scene._spec()["origin"]
    expected = holes.at(["A", "A", "A", "B", "C"], [5.0, 15.0, 25.0, 15.0, 15.0])
    np.testing.assert_allclose(middles, expected, atol=1e-3)


def rendered_pixels(scene, points, tmp_path, bare=False):
    """Colors the saved page shows at world `points`, in a headless browser; `bare` hides the box, grid and panel."""
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
        if bare:
            tab.evaluate(
                "() => { const s = window.scene; s.axes.show = { box: false, grid: false, ticks: false };"
                " s.panel.hidden = s.bars.hidden = true;"
                " s.gizmo.el.style.display = 'none'; s.requestRender(); }"
            )
            tab.wait_for_timeout(100)
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


def framed(scene, z=0.0):
    """Adds two dots far out in plan, so filtering never changes the extent the camera frames."""
    return scene.add(bt.PointSet([[-15.0, -15.0, z], [45.0, 30.0, z]]), point_size=2)


def lensy_points():
    return bt.PointSet(
        rng.uniform(0, 1, (4, 3)),
        {"grade": [1.0, 5.0, np.nan, 9.0], "lens": ["a", "b", None, "a"], "other": [0.0, 1.0, 2.0, 3.0]},
    )


def test_filter_reaches_the_viewer_in_the_order_given():
    scene = Scene().add(lensy_points(), "grade", filter={"lens": ["a"], "grade": (2, None)})
    assert scene._spec()["layers"][0]["filter"] == {
        "lens": {"categories": ["a"]},
        "grade": {"range": [2.0, None]},
    }
    assert scene.filters == {"points 1": {"lens": ["a"], "grade": (2.0, None)}}
    assert Scene().add(lensy_points()).filters == {}


@pytest.mark.parametrize(
    ("conditions", "error", "match"),
    [
        ({"nope": (0, 1)}, ValueError, "columns: grade, lens, other"),
        ({"grade": [0, 1]}, TypeError, r"\(low, high\) tuple"),
        ({"grade": (0,)}, ValueError, "finite numbers or None"),
        ({"grade": (0, np.inf)}, ValueError, "finite numbers or None"),
        ({"grade": (True, 2)}, ValueError, "finite numbers or None"),
        ({"grade": (5, 1)}, ValueError, "low <= high"),
        ({"lens": ("a", "b")}, TypeError, "list of categories"),
        ({"lens": "a"}, TypeError, "list of categories"),
        ({"lens": ["a", "zz"]}, ValueError, r"no categories \['zz'\]"),
        (["grade"], TypeError, "dict of column name"),
    ],
)
def test_filter_is_validated(conditions, error, match):
    with pytest.raises(error, match=match):
        Scene().add(lensy_points(), filter=conditions)


def test_filter_columns_must_be_sent_and_are_capped():
    with pytest.raises(ValueError, match="columns= limits them"):
        Scene().add(lensy_points(), "grade", columns=["lens"], filter={"other": (0, 1)})
    table = {f"c{k}": np.arange(3.0) for k in range(5)}
    with pytest.raises(ValueError, match="4 columns at most"):
        Scene().add(bt.PointSet(np.zeros((3, 3)), table), filter={k: (0, 1) for k in table})


def test_filters_setter_validates_and_replaces_every_layer():
    scene = Scene().add(lensy_points(), name="a", filter={"grade": (1, 2)}).add(lensy_points(), name="b")
    scene.filters = {"b": {"lens": ["b"]}}
    assert scene.filters == {"b": {"lens": ["b"]}}
    with pytest.raises(ValueError, match="no layers"):
        scene.filters = {"c": {}}
    with pytest.raises(TypeError, match="list of categories"):
        scene.filters = {"a": {"lens": "b"}}
    assert scene.filters == {"b": {"lens": ["b"]}}
    twins = Scene().add(lensy_points(), name="x").add(lensy_points(), name="x")
    with pytest.raises(ValueError, match="distinct names"):
        twins.filters = {"x": {}}


def test_filters_follow_the_live_widgets_both_ways(monkeypatch):
    traitlets = pytest.importorskip("traitlets")

    class FakeWidget(traitlets.HasTraits):
        spec = traitlets.Dict()
        buffers = traitlets.Dict()
        filters = traitlets.Dict()

    monkeypatch.setattr("boitata._scene._widget_class", lambda: FakeWidget)
    scene = Scene().add(lensy_points(), name="a", filter={"grade": (1, None)}).add(lensy_points(), name="b")
    first, second = scene._widget(), scene._widget()
    assert first.filters == {"l0": {"grade": {"range": [1.0, None]}}}
    first.filters = {"l1": {"lens": {"categories": ["a"]}, "other": {"range": [None, 2]}}}
    assert scene.filters == {"b": {"lens": ["a"], "other": (None, 2.0)}}
    assert second.filters == first.filters
    scene.filters = {"a": {"lens": ["b"]}}
    assert first.filters == second.filters == {"l0": {"lens": {"categories": ["b"]}}}
    assert scene._spec()["layers"][0]["filter"] == {"lens": {"categories": ["b"]}}


def test_rendered_block_filtered_by_number_or_category_paints_nothing(tmp_path):
    model = bt.BlockModel((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (3, 1, 1))
    model = model.with_column("v", [0.2, 0.5, 0.8]).with_column("f", [1.0, 5.0, 9.0])
    model = model.with_column("zone", ["a", "b", "a"])
    centers = [[3.5, 6.5, 10.0], [16.5, 6.5, 10.0], [23.5, 6.5, 10.0]]
    colors = [lut_rgb("viridis", v, 0.0, 1.0) for v in (0.2, 0.5, 0.8)]
    by_number = Scene(theme="light").add(model, "v", clim=(0.0, 1.0), filter={"f": (4, None)})
    assert rendered_pixels(framed(by_number).view(dip=90), centers, tmp_path, bare=True) == [
        (255, 255, 255),
        colors[1],
        colors[2],
    ]
    by_zone = Scene(theme="light").add(model, "v", clim=(0.0, 1.0), filter={"zone": ["a"]})
    assert rendered_pixels(framed(by_zone).view(dip=90), centers, tmp_path, bare=True) == [
        colors[0],
        (255, 255, 255),
        colors[2],
    ]


@pytest.mark.parametrize(
    ("representation", "size"),
    [("lines", {"line_width": 12}), ("tubes", {"radius": 1.0}), ("points", {"point_size": 24})],
)
def test_rendered_interval_filtered_out_paints_nothing(tmp_path, representation, size):
    holes = straight_hole([0.2, 0.5, 0.8])
    scene = Scene(theme="light").add(
        holes, "CU", representation=representation, clim=(0.0, 1.0), filter={"CU": (None, 0.6)}, **size
    )
    framed(scene, 100.0).view(azimuth=0, dip=90)
    probes = [[5, 3, 100], [15, 3, 100], [25, 3, 100]]
    first, middle, last = rendered_pixels(scene, probes, tmp_path, bare=True)
    shading = 2 if representation == "tubes" else 0
    for got, value in ((first, 0.2), (middle, 0.5)):
        assert np.abs(np.subtract(got, lut_rgb("viridis", value, 0.0, 1.0))).max() <= shading
    assert last == (255, 255, 255)


@pytest.mark.parametrize("representation", ["points", "spheres"])
def test_rendered_point_filtered_out_paints_nothing(tmp_path, representation):
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    points = bt.PointSet(xyz, {"v": [0.2, 0.5, 0.8], "rock": ["x", "y", "x"]})
    size = {"point_size": 24} if representation == "points" else {"radius": 2.0}
    scene = Scene(theme="light").add(
        points, "v", representation=representation, clim=(0.0, 1.0), filter={"rock": ["x"]}, **size
    )
    got = rendered_pixels(framed(scene).view(dip=90), xyz.tolist(), tmp_path, bare=True)
    shading = 2 if representation == "spheres" else 0
    assert got[1] == (255, 255, 255)
    for i, value in ((0, 0.2), (2, 0.8)):
        assert np.abs(np.subtract(got[i], lut_rgb("viridis", value, 0.0, 1.0))).max() <= shading


@pytest.mark.parametrize("representation", ["surface", "points"])
def test_rendered_mesh_filtered_by_face_paints_nothing(tmp_path, representation):
    vertices = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [20, 0, 0], [20, 10, 0]]
    mesh = bt.Mesh(vertices, [[0, 1, 2], [1, 3, 4]]).with_face_column("zone", ["keep", "drop"])
    mesh = mesh.with_vertex_column("h", [0.2, 0.2, 0.2, 0.8, 0.8])
    scene = Scene(theme="light").add(
        mesh, "h", representation=representation, clim=(0.0, 1.0), point_size=24, filter={"zone": ["keep"]}
    )
    probes = [[3, 3, 0], [17, 4, 0]] if representation == "surface" else [[0, 0, 0], [20, 10, 0]]
    kept, dropped = rendered_pixels(framed(scene).view(dip=90), probes, tmp_path, bare=True)
    assert kept == lut_rgb("viridis", 0.2, 0.0, 1.0)
    assert dropped == (255, 255, 255)


WIDGET = """async () => {
    const text = (id) => document.getElementById(id).textContent;
    const code = Uint8Array.from(atob(text("scene-viewer")), (c) => c.charCodeAt(0));
    const viewer = await import(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
    const data = JSON.parse(text("scene-data"));
    const buffers = {};
    for (const [k, b64] of Object.entries(data.buffers))
        buffers[k] = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer;
    const state = { spec: data.spec, buffers, filters: {} };
    window.model = {
        state, saved: 0,
        get: (k) => state[k], set: (k, v) => { state[k] = v; },
        save_changes() { this.saved++; }, on() {}, off() {},
    };
    document.getElementById("scene").remove();
    const el = document.createElement("div");
    el.id = "widget";
    document.body.appendChild(el);
    viewer.default.render({ model: window.model, el });
}"""


def test_panel_edits_reach_the_widget_model(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    model = bt.BlockModel((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (3, 1, 1)).with_column("v", [0.2, 0.5, 0.8])
    page = Scene(theme="light").add(model, "v", name="blocks").save(tmp_path / "scene.html")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except sync_api.Error as e:
            pytest.skip(f"no chromium for playwright: {e}")
        tab = browser.new_page(viewport={"width": 900, "height": 600})
        tab.goto(page.as_uri())
        tab.wait_for_function("window.scene !== undefined", timeout=60_000)
        tab.evaluate(WIDGET)
        widget = tab.locator("#widget")
        widget.locator(".btv-row").filter(has_text="blocks").locator("button[title=Settings]").click()
        widget.locator(".btv-add").click()
        widget.locator(".btv-filter select").select_option("v")
        low = widget.locator(".btv-cond input[type=number]").first
        low.fill("0.4")
        low.dispatch_event("change")
        tab.wait_for_function("window.model.state.filters.l0?.v?.range?.[0] === 0.4", timeout=5_000)
        filters = tab.evaluate("window.model.state.filters")
        readout = widget.locator(".btv-readout").first.text_content()
        browser.close()
    assert filters == {"l0": {"v": {"range": [0.4, None]}}}
    assert readout == "2 of 3 shown"
