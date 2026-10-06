import io
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import boitata as bt
import numpy as np
import pytest
from boitata.plot3d import Scene

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
    masked = model.filter(np.arange(24) % 3 == 0)
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
    monkeypatch.setattr("boitata.plot3d._WARN_BYTES", 100)
    with pytest.warns(UserWarning, match="MB of data"):
        Scene().add(bt.PointSet(rng.uniform(0, 1, (20, 3))))


def displayed(scene, monkeypatch):
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda obj: shown.append(obj))
    scene._ipython_display_()
    return shown


def test_notebook_display_shows_the_widget_with_a_standby_iframe_or_only_the_iframe(monkeypatch):
    pytest.importorskip("anywidget")
    scene = Scene().add(bt.PointSet(np.zeros((1, 3))))
    widget, standby = displayed(scene, monkeypatch)
    assert "application/vnd.jupyter.widget-view+json" in widget._repr_mimebundle_()[0]
    assert len(widget.key) == 32 and widget.key in standby.data
    assert f'data-btv-fallback="{widget.key}" hidden><template><iframe srcdoc=' in standby.data
    assert "install anywidget where Jupyter runs" in standby.data
    monkeypatch.setitem(sys.modules, "anywidget", None)
    monkeypatch.setattr("boitata.plot3d._widget_class", lambda: __import__("anywidget"))
    (frame,) = displayed(scene, monkeypatch)
    assert frame.data.startswith("<iframe srcdoc=") and "boitata[3d]" in frame.data


FALLBACK_PAGE = """<!doctype html><body>{before}<div class="jp-OutputArea-child">{error}</div>
<div class="jp-OutputArea-child">{fallback}</div></body>"""


@pytest.mark.parametrize("rendered", [False, True])
def test_standby_iframe_shows_only_when_the_widget_never_renders(rendered):
    from boitata.plot3d import _FALLBACK

    sync_api = pytest.importorskip("playwright.sync_api")
    fallback = _FALLBACK.format(key="k1", frame="<p id='copy'>copy</p>", delay=200)
    error = '<div data-btv-key="k1"></div>' if rendered else "No version of module anywidget is registered"
    page = FALLBACK_PAGE.format(before="<p>other output</p>", error=error, fallback=fallback)
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except sync_api.Error as e:
            pytest.skip(f"no chromium for playwright: {e}")
        tab = browser.new_page()
        tab.set_content(page)
        tab.wait_for_timeout(800)
        copy = tab.locator("#copy").count()
        standby = tab.locator("[data-btv-fallback]").count()
        error_hidden = tab.evaluate("document.querySelector('.jp-OutputArea-child').hidden")
        browser.close()
    assert (copy, standby, error_hidden) == ((0, 0, False) if rendered else (1, 1, True))


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
    scene = Scene()
    scene.motion_quality = value
    assert scene.motion_quality == sent and scene._spec()["motion"] == sent


@pytest.mark.parametrize("value", ["fast", 0, 1.5, -0.1, True, None])
def test_motion_quality_is_validated(value):
    with pytest.raises(ValueError, match="motion_quality"):
        Scene(motion_quality=value)
    with pytest.raises(ValueError, match="motion_quality"):
        Scene().motion_quality = value


def test_plot_draws_one_container_in_a_new_scene():
    points = bt.PointSet(rng.uniform(0, 1, (4, 3)), {"v": [1.0, 2.0, 3.0, 4.0]})
    scene = bt.plot3d.plot(points, "v", theme="dark", height=300, representation="spheres", name="pts")
    spec = scene._spec()
    assert (spec["theme"], spec["height"]) == ("dark", 300)
    assert [(lay["name"], lay["values"], lay["representation"]) for lay in spec["layers"]] == [
        ("pts", "v", "spheres")
    ]
    with pytest.raises(TypeError):
        bt.plot3d.plot(points, plotter=None)


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
        tab.wait_for_function("typeof window.scene?.screenshot === 'function'", timeout=60_000)
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


def fake_widgets(monkeypatch):
    """Scene widgets that sync their traits like the real one, without a frontend."""
    traitlets = pytest.importorskip("traitlets")

    class FakeWidget(traitlets.HasTraits):
        spec = traitlets.Dict()
        buffers = traitlets.Dict()
        filters = traitlets.Dict()
        section = traitlets.Dict()
        picked = traitlets.Dict()
        view = traitlets.Dict()
        key = traitlets.Unicode()

    monkeypatch.setattr("boitata.plot3d._widget_class", lambda: FakeWidget)


def test_filters_follow_the_live_widgets_both_ways(monkeypatch):
    fake_widgets(monkeypatch)
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
    const filters = Object.fromEntries(
        data.spec.layers.filter((l) => Object.keys(l.filter).length).map((l) => [l.id, l.filter]));
    const state = { spec: data.spec, buffers, filters, section: data.spec.section, key: "widget-key" };
    window.model = {
        state, saved: 0,
        get: (k) => state[k], set: (k, v) => { state[k] = v; },
        save_changes() { this.saved++; }, off() {}, handlers: {}, sent: [],
        on(event, callback) { (this.handlers[event] ||= []).push(callback); },
        send(content, callbacks, buffers) { this.sent.push([content, buffers ?? []]); },
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
        tab.wait_for_function("typeof window.scene?.screenshot === 'function'", timeout=60_000)
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


def row_of_blocks():
    """Three rows of three 10 m blocks in plan, colored by row: 0.2 south, 0.5 middle, 0.8 north."""
    model = bt.BlockModel((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (3, 3, 1))
    return model.with_column("v", np.repeat([0.2, 0.5, 0.8], 3))


def test_section_reaches_the_viewer_in_real_world_coordinates():
    xyz = rng.uniform(0, 100, (20, 3)) + [500_000.0, 7_000_000.0, 300.0]
    scene = Scene().add(bt.PointSet(xyz))
    origin = scene._spec()["origin"]
    scene.section([(500_010.0, 7_000_020.0), (500_090.0, 7_000_020.0), (500_090.0, 7_000_090.0)], width=12)
    assert scene._spec()["section"] == {
        "points": [
            [500_010.0, 7_000_020.0, origin[2]],
            [500_090.0, 7_000_020.0, origin[2]],
            [500_090.0, 7_000_090.0, origin[2]],
        ],
        "width": 12.0,
        "dip": 90.0,
        "unfolded": False,
    }
    scene.section([(0, 0, 100), (0, 0, 120), (10, 0, 140)], dip=60)
    assert scene.sections == {
        "points": [(0.0, 0.0, 120.0), (10.0, 0.0, 120.0)],
        "width": None,
        "dip": 60.0,
        "unfolded": False,
    }
    assert scene.section(None).sections is None and scene._spec()["section"] == {}


@pytest.mark.parametrize(
    ("points", "kwargs", "match"),
    [
        ([(0, 0)], {}, "at least 2 distinct"),
        ([(0, 0, 1), (0, 0, 5)], {}, "at least 2 distinct"),
        ([(0, 0), (1, np.nan)], {}, "finite"),
        ([(0, 0), (np.inf, 1)], {}, "finite"),
        ([(0, 0, 0, 0), (1, 1, 1, 1)], {}, r"\(x, y\) or \(x, y, z\)"),
        ([(0, 0), (1,)], {}, r"\(x, y\) or \(x, y, z\)"),
        ([(0, 0), ("a", 1)], {}, r"\(x, y\) or \(x, y, z\)"),
        ([(k, k % 2) for k in range(18)], {}, "at most 17"),
        ([(0, 0), (1, 0)], {"width": 0}, "width"),
        ([(0, 0), (1, 0)], {"width": -2}, "width"),
        ([(0, 0), (1, 0)], {"width": np.inf}, "width"),
        ([(0, 0), (1, 0)], {"width": True}, "width"),
        ([(0, 0), (1, 0)], {"dip": 0}, "dip"),
        ([(0, 0), (1, 0)], {"dip": 91}, "dip"),
        ([(0, 0), (1, 0)], {"dip": "90"}, "dip"),
    ],
)
def test_section_is_validated(points, kwargs, match):
    scene = Scene().add(bt.PointSet(np.zeros((1, 3))))
    with pytest.raises(ValueError, match=match):
        scene.section(points, **kwargs)
    assert scene.sections is None


def test_sections_setter_validates_and_replaces_the_section():
    scene = Scene().add(bt.PointSet(np.zeros((1, 3))))
    scene.sections = {"points": [(0, 0, 5), (10, 10, 5)], "width": 3, "unfolded": True}
    assert scene.sections == {
        "points": [(0.0, 0.0, 5.0), (10.0, 10.0, 5.0)],
        "width": 3.0,
        "dip": 90.0,
        "unfolded": True,
    }
    with pytest.raises(ValueError, match="unknown section keys"):
        scene.sections = {"points": [(0, 0), (1, 1)], "azimuth": 3}
    with pytest.raises(ValueError, match="needs points"):
        scene.sections = {"width": 3}
    with pytest.raises(TypeError, match="bool"):
        scene.sections = {"points": [(0, 0), (1, 1)], "unfolded": 1}
    with pytest.raises(TypeError, match="dict or None"):
        scene.sections = [(0, 0), (1, 1)]
    assert scene.sections["width"] == 3.0
    scene.sections = None
    assert scene.sections is None


def test_sections_follow_the_live_widgets_both_ways(monkeypatch):
    fake_widgets(monkeypatch)
    scene = Scene().add(bt.PointSet(np.zeros((1, 3))))
    scene.section([(0, 0), (5, 0)], width=2)
    first, second = scene._widget(), scene._widget()
    assert first.section == {
        "points": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        "width": 2.0,
        "dip": 90.0,
        "unfolded": False,
    }
    first.section = {"points": [[1, 2, 3], [4, 5, 3], [7, 2, 3]], "width": 7.5, "dip": 80, "unfolded": True}
    assert scene.sections == {
        "points": [(1.0, 2.0, 3.0), (4.0, 5.0, 3.0), (7.0, 2.0, 3.0)],
        "width": 7.5,
        "dip": 80.0,
        "unfolded": True,
    }
    assert second.section == first.section
    first.section = {}
    assert scene.sections is None and second.section == {}
    scene.section([(0, 0), (0, 9)], dip=45)
    assert first.section == second.section == scene._spec()["section"]
    assert first.section["points"] == [[0.0, 0.0, 0.0], [0.0, 9.0, 0.0]]


def test_rendered_section_paints_only_the_blocks_in_its_slab(tmp_path):
    scene = Scene(theme="light").add(row_of_blocks(), "v", clim=(0.0, 1.0)).view(dip=90)
    scene.section([(-5, 15), (35, 15)], width=4)
    probes = [
        [5.5, 15.5, 10.0],
        [24.5, 14.5, 10.0],
        [15.5, 5.5, 10.0],
        [15.5, 24.5, 10.0],
        [15.5, 11.5, 10.0],
    ]
    inside, also, south, north, outside = rendered_pixels(scene, probes, tmp_path, bare=True)
    assert inside == also == lut_rgb("viridis", 0.5, 0.0, 1.0)
    assert south == north == outside == (255, 255, 255)


def test_rendered_cap_shows_the_exact_lut_color(tmp_path):
    scene = Scene(theme="light").add(row_of_blocks(), "v", clim=(0.0, 1.0)).view(azimuth=0, dip=0)
    scene.section([(-5, 15), (35, 15)], width=4)
    cap, edge = rendered_pixels(scene, [[15.0, 13.0, 5.0], [15.0, 13.0, 9.0]], tmp_path, bare=True)
    assert cap == edge == lut_rgb("viridis", 0.5, 0.0, 1.0)


def test_shift_drag_and_keys_reach_the_widget_model(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    page = (
        Scene(theme="light").add(row_of_blocks(), "v").view(azimuth=0, dip=90).save(tmp_path / "scene.html")
    )
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except sync_api.Error as e:
            pytest.skip(f"no chromium for playwright: {e}")
        tab = browser.new_page(viewport={"width": 900, "height": 600})
        tab.goto(page.as_uri())
        tab.wait_for_function("typeof window.scene?.screenshot === 'function'", timeout=60_000)
        tab.evaluate(WIDGET)
        tab.mouse.move(250, 300)
        tab.keyboard.down("Shift")
        tab.mouse.down()
        tab.mouse.move(450, 300, steps=5)
        tab.mouse.up()
        tab.keyboard.up("Shift")
        tab.wait_for_function("window.model.state.section?.points?.length === 2", timeout=5_000)
        cut = tab.evaluate("window.model.state.section")
        tab.keyboard.press("u")
        unfolded = tab.evaluate("window.model.state.section.unfolded")
        tab.keyboard.press("x")
        cleared = tab.evaluate("window.model.state.section")
        browser.close()
    (x0, y0, _), (x1, y1, _) = cut["points"]
    assert x0 < x1 and abs(y0 - y1) < 1e-6 and 0 < y0 < 30
    assert cut["dip"] == 90 and cut["width"] > 0 and cut["unfolded"] is False
    assert unfolded is True and cleared == {}


def test_picked_follows_the_live_widgets(monkeypatch):
    fake_widgets(monkeypatch)
    scene = Scene().add(lensy_points(), name="pts")
    assert scene.picked is None
    widget = scene._widget()
    values = {"grade": None, "lens": None, "other": 2.0}
    widget.picked = {"layer": "pts", "row": 2, "values": values, "position": [1, 2, 3]}
    assert scene.picked == {"layer": "pts", "row": 2, "values": values, "position": (1.0, 2.0, 3.0)}
    scene.picked["values"]["other"] = 5.0
    assert scene.picked["values"]["other"] == 2.0
    widget.picked = {}
    assert scene.picked is None
    with pytest.raises(AttributeError):
        scene.picked = None


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"path": "scene.jpg"}, ValueError, r"\.png"),
        ({"scale": 0}, ValueError, "scale"),
        ({"scale": 9}, ValueError, "scale"),
        ({"scale": np.nan}, ValueError, "scale"),
        ({"scale": True}, ValueError, "scale"),
        ({"scale": "2"}, ValueError, "scale"),
        ({"panel": 1}, TypeError, "panel"),
        ({"transparent": "yes"}, TypeError, "transparent"),
    ],
)
def test_screenshot_validates_its_arguments(tmp_path, kwargs, error, match):
    path = tmp_path / kwargs.pop("path", "scene.png")
    with pytest.raises(error, match=match):
        Scene().add(bt.PointSet(np.zeros((1, 3)))).screenshot(path, **kwargs)
    assert not path.exists()


def test_screenshot_renders_headless_at_the_size_and_view_the_widget_reported(monkeypatch, tmp_path):
    fake_widgets(monkeypatch)
    calls = []

    def headless(page, width, height, **options):
        calls.append((page, width, height, options))
        return b"\x89PNG"

    monkeypatch.setattr("boitata.plot3d._headless", headless)
    scene = Scene(height=300).add(bt.PointSet(np.zeros((1, 3))))
    assert scene.screenshot(tmp_path / "a.png").read_bytes() == b"\x89PNG"
    assert calls[-1][1:] == (480, 300, {"scale": 1.0, "panel": False, "transparent": False})
    view = {"size": [811.4, 433], "camera": {"position": [1, 2, 3]}, "theme": "auto", "dark": True}
    scene._widget().view = view
    scene.screenshot(tmp_path / "b.png", scale=2)
    page, width, height, _ = calls[-1]
    assert (width, height) == (811, 433)
    assert '"state":{"size":[811.4,433],"camera":{"position":[1,2,3]},"theme":"dark"' in page
    assert (
        scene.save(tmp_path / "scene.html").read_text(encoding="utf-8").count('"theme":"auto","dark":true')
        == 1
    )
    scene.view(azimuth=90)
    scene.screenshot(tmp_path / "c.png")
    assert '"camera"' not in calls[-1][0].split('"state":')[1] and calls[-1][1:3] == (811, 433)


def test_screenshot_without_playwright_says_how_to_install_it(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(ImportError, match=r"pip install 'boitata\[export\]'"):
        Scene().add(bt.PointSet(np.zeros((1, 3)))).screenshot(tmp_path / "scene.png")


def test_headless_screenshot_has_the_scale_and_exact_colors(tmp_path):
    pytest.importorskip("playwright.sync_api")
    image = pytest.importorskip("PIL.Image")
    model = bt.BlockModel((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (3, 1, 1)).with_column("v", [0.2, 0.5, 0.8])
    scene = Scene(theme="light", height=300).add(model, "v", clim=(0.0, 1.0)).view(dip=90)
    shot = image.open(scene.screenshot(tmp_path / "scene.png", scale=2))
    assert shot.size == (960, 600)
    assert shot.convert("RGB").getpixel((480, 300)) == lut_rgb("viridis", 0.5, 0.0, 1.0)
    clear = image.open(scene.screenshot(tmp_path / "clear.png", transparent=True)).convert("RGBA")
    assert (
        clear.size == (480, 300) and clear.getpixel((240, 150))[3] == 255 and clear.getpixel((2, 2))[3] == 0
    )


TO_SCREEN = """(points) => points.map(([x, y, z]) => {
    const s = window.scene, o = s.spec.origin;
    const v = s.camera.position.clone().set(x - o[0], y - o[1], z - o[2]).project(s.camera);
    return [(v.x + 1) / 2 * s.width, (1 - v.y) / 2 * s.height];
})"""


@contextmanager
def viewer_page(scene, tmp_path, points=(), widget=False):
    """A headless tab showing the saved scene, as a page or (`widget`) through the widget entry point, and where
    world `points` fall on it."""
    sync_api = pytest.importorskip("playwright.sync_api")
    page = scene.save(tmp_path / "scene.html")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except sync_api.Error as e:
            pytest.skip(f"no chromium for playwright: {e}")
        try:
            tab = browser.new_page(viewport={"width": 900, "height": 600})
            tab.goto(page.as_uri())
            tab.wait_for_function("typeof window.scene?.screenshot === 'function'", timeout=60_000)
            at = tab.evaluate(TO_SCREEN, list(points))
            if widget:
                tab.evaluate(WIDGET)
            tab.wait_for_timeout(200)
            yield tab, at
        finally:
            browser.close()


def live_edits(tab):
    """Filters the blocks in the panel, turns the camera, switches to the dark theme and hides the panel."""
    widget = tab.locator("#widget")
    widget.locator(".btv-row").filter(has_text="blocks").locator("button[title=Settings]").click()
    widget.locator(".btv-add").click()
    widget.locator(".btv-filter select").select_option("v")
    low = widget.locator(".btv-cond input[type=number]").first
    low.fill("0.4")
    low.dispatch_event("change")
    tab.mouse.move(300, 400)
    tab.mouse.down()
    tab.mouse.move(360, 360, steps=6)
    tab.mouse.up()
    tab.evaluate("document.getElementById('widget').shadowRoot.querySelector('.btv-root').focus()")
    tab.keyboard.press("t")
    tab.keyboard.press("h")
    tab.wait_for_function(
        "() => { const v = window.model.state.view; return v && v.dark && !v.panel; }", timeout=10_000
    )
    tab.wait_for_timeout(600)


def differing(a, b):
    """Share of pixels whose channels differ by more than 40 anywhere."""
    a, b = (np.asarray(x.convert("RGB"), dtype=np.int16) for x in (a, b))
    return float((np.abs(a - b).max(axis=2) > 40).mean())


def test_screenshot_reproduces_the_live_widget_after_view_edits(tmp_path):
    image = pytest.importorskip("PIL.Image")
    scene = Scene(theme="auto").add(row_of_blocks(), "v", name="blocks", clim=(0.0, 1.0))
    with viewer_page(scene, tmp_path, widget=True) as (tab, _):
        live_edits(tab)
        live = image.open(io.BytesIO(tab.screenshot()))
        state = tab.evaluate("window.model.state")
    assert state["view"]["size"] == [900, 600] and state["filters"] == {"l0": {"v": {"range": [0.4, None]}}}
    scene._view_edited({"new": state["view"]})
    scene._filters_edited({"new": state["filters"], "owner": None})
    synced = image.open(scene.screenshot(tmp_path / "synced.png"))
    scene._state = None
    scene.filters = {}
    unsynced = image.open(scene.screenshot(tmp_path / "unsynced.png"))
    assert synced.size == live.size == (900, 600) and unsynced.size == (960, 600)
    assert differing(live, synced) < 0.02
    assert differing(live, unsynced.crop((0, 0, 900, 600))) > 0.3


def test_click_picks_the_row_under_it_and_never_a_hidden_one(tmp_path):
    """Blocks 3, 4 and 5 lie in the section; the filter hides the middle column (4), the section rows 0 to 2."""
    model = row_of_blocks().with_column("w", [1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0])
    model = model.with_column("zone", ["a", "b", "a"] * 3)
    scene = (
        Scene(theme="light").add(model, "v", name="blocks", filter={"zone": ["a"]}).view(azimuth=0, dip=90)
    )
    scene.section([(-5, 15), (35, 15)], width=4)
    picked = "window.model.state.picked"
    with viewer_page(scene, tmp_path, [[5, 15, 10], [15, 15, 10], [5, 5, 10]], widget=True) as (tab, at):
        kept, filtered, outside = at
        tab.mouse.click(*kept)
        first = tab.evaluate(picked)
        card = tab.locator("#widget .btv-card:not(.btv-help)").text_content()
        tab.mouse.click(*filtered)
        on_filtered = tab.evaluate(picked)
        tab.mouse.click(*kept)
        tab.mouse.click(*outside)
        on_outside = tab.evaluate(picked)
        tab.mouse.move(*filtered)
        tab.mouse.down()
        tab.mouse.move(kept[0], kept[1] + 3, steps=4)
        tab.mouse.up()
        dragged = tab.evaluate(picked)
        tab.mouse.click(*kept)
        tab.keyboard.press("Escape")
        escaped = tab.evaluate(picked)
    assert first == {
        "layer": "blocks",
        "row": 3,
        "values": {"v": 0.5, "w": None, "zone": "a"},
        "position": [5.0, 15.0, 5.0],
    }
    assert "—" in card and "NaN" not in card and "10 × 10 × 10" in card
    assert on_filtered == on_outside == dragged == escaped == {}


def test_every_representation_picks_its_rows(tmp_path):
    model = bt.BlockModel((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (3, 1, 1)).with_column("v", [0.2, 0.5, 0.8])
    points = bt.PointSet([[45.0, 5.0, 5.0], [55.0, 5.0, 5.0], [65.0, 5.0, 5.0]], {"v": [0.2, 0.5, 0.8]})
    holes = bt.Drillholes(
        {"HOLE_ID": ["A"], "X": [0.0], "Y": [-25.0], "Z": [5.0]},
        {"HOLE_ID": ["A", "A"], "DEPTH": [0.0, 30.0], "AZIMUTH": [90.0] * 2, "DIP": [0.0] * 2},
        {"HOLE_ID": ["A"] * 3, "FROM": [0.0, 10.0, 20.0], "TO": [10.0, 20.0, 30.0], "CU": [0.2, 0.5, 0.8]},
    )
    vertices = [[40, -30, 5], [50, -30, 5], [40, -20, 5], [60, -30, 5], [60, -20, 5]]
    mesh = bt.Mesh(vertices, [[0, 1, 2], [1, 3, 4]])
    scene = Scene(theme="light").view(azimuth=0, dip=90)
    scene.add(model, "v", name="blocks", point_size=14)
    scene.add(points, "v", name="points", point_size=14, radius=3)
    scene.add(holes, "CU", name="holes", line_width=8, radius=1.5, point_size=14)
    scene.add(mesh, name="mesh", point_size=14, line_width=6)
    cases = [
        (0, "cells", [15, 5, 10]),
        (0, "wireframe", [15, 5, 10]),
        (0, "points", [15, 5, 5]),
        (1, "points", [55, 5, 5]),
        (1, "spheres", [55, 5, 5]),
        (2, "lines", [15, -25, 5]),
        (2, "tubes", [15, -25, 5]),
        (2, "points", [15, -25, 5]),
        (3, "surface", [57, -27, 5]),
        (3, "wireframe", [60, -25, 5]),
        (3, "points", [60, -20, 5]),
    ]
    got = []
    with viewer_page(scene, tmp_path) as (tab, _):
        tab.evaluate("() => window.scene.togglePanel()")
        for layer, representation, point in cases:
            tab.evaluate(
                "([i, name]) => { const s = window.scene; s.dismiss(); s.setRepresentation(s.layers[i], name); }",
                [layer, representation],
            )
            tab.wait_for_timeout(50)
            ((x, y),) = tab.evaluate(TO_SCREEN, [point])
            tab.mouse.click(x, y)
            pick = tab.evaluate("window.scene.pickState()")
            got.append((representation, pick and pick["layer"], pick and pick["row"]))
    names = ["blocks", "points", "holes", "mesh"]
    assert got == [(representation, names[layer], 1) for layer, representation, _ in cases]


def test_question_mark_opens_the_shortcut_card_and_esc_closes_it(tmp_path):
    scene = Scene(theme="light").add(row_of_blocks(), "v")
    with viewer_page(scene, tmp_path) as (tab, _):
        tab.mouse.click(450, 40)
        tab.keyboard.press("?")
        help_card = tab.locator(".btv-help")
        shown = help_card.is_visible()
        text = help_card.text_content()
        tab.keyboard.press("Escape")
        closed = help_card.is_hidden()
        tab.locator("button[title='Keyboard shortcuts (?)']").click()
        reopened = help_card.is_visible()
        tab.mouse.click(450, 40)
        clicked_away = help_card.is_hidden()
    assert shown and closed and reopened and clicked_away
    assert "Keyboard shortcuts" in text and "Draw a section" in text and "Clear the section" in text


def test_fullscreen_button_fills_the_screen_or_says_why_it_cannot(tmp_path):
    scene = Scene(theme="light").add(row_of_blocks(), "v")
    with viewer_page(scene, tmp_path) as (tab, _):
        button = tab.locator(".btv-head button[aria-label='Fullscreen']")
        button.click()
        tab.wait_for_function(
            "document.fullscreenElement !== null ||"
            " document.querySelector('#scene').shadowRoot.querySelector('[aria-disabled=true]') !== null",
            timeout=5_000,
        )
        entered = tab.evaluate("document.fullscreenElement !== null")
        if entered:
            title = tab.locator(".btv-head button[aria-label='Leave fullscreen (Esc)']").get_attribute(
                "title"
            )
            tab.locator(".btv-head button[aria-label='Leave fullscreen (Esc)']").click()
            tab.wait_for_function("document.fullscreenElement === null", timeout=5_000)
            back = tab.locator(".btv-head button[aria-label='Fullscreen']").count()
        else:
            title = tab.locator(".btv-head button[aria-disabled=true]").get_attribute("title")
    if entered:
        assert title == "Leave fullscreen (Esc)" and back == 1
    else:
        assert "unavailable" in title


def test_wheel_zooms_and_shift_wheel_sets_the_section_width(tmp_path):
    scene = Scene(theme="light").add(row_of_blocks(), "v").view(azimuth=0, dip=90)
    scene.section([(-5, 15), (35, 15)], width=4)
    measure = (
        "() => { const s = window.scene;"
        " return [s.camera.position.distanceTo(s.controls.target), s.sectionState().width]; }"
    )
    with viewer_page(scene, tmp_path) as (tab, _):
        tab.mouse.move(300, 300)
        before = tab.evaluate(measure)
        tab.mouse.wheel(0, -300)
        tab.wait_for_timeout(100)
        zoomed = tab.evaluate(measure)
        tab.keyboard.down("Shift")
        tab.mouse.wheel(0, -300)
        tab.keyboard.up("Shift")
        tab.wait_for_timeout(100)
        widened = tab.evaluate(measure)
    assert zoomed[0] < before[0] and zoomed[1] == before[1] == 4
    assert widened[0] == pytest.approx(zoomed[0]) and widened[1] > 4
