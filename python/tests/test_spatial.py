import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(5)
xy = rng.uniform(0, 100, (400, 2))
xy = xy[(xy[:, 0] < 40) | (xy[:, 1] < 40)]
lode = np.where(xy[:, 0] < 50, "west", "east")
points = bt.PointSet(np.c_[xy, rng.uniform(40, 60, len(xy))], {"lode": lode})


def test_every_point_lies_inside_its_outline():
    for method, options in (("convex", {}), ("concave", {"max_edge": 20.0})):
        lines = bt.outline(points, method=method, buffer=0.5, categories="lode", **options)
        assert lines.attributes["lode"].tolist() == ["east", "west"]
        for f, name in enumerate(["east", "west"]):
            assert lines.contains(points.filter(lode == name), feature=f).all()
    convex = bt.outline(points).area()[0]
    concave = bt.outline(points, method="concave", max_edge=15.0).area()[0]
    assert concave < 0.8 * convex


def test_outline_vertices_and_plane():
    ring = bt.outline(points).parts[0]
    np.testing.assert_allclose(ring[:, 2], points.coords[:, 2].mean())
    section = np.c_[np.full(len(xy), 7.0), xy]
    ring = bt.outline(section, plane=(0.0, 90.0)).parts[0]
    np.testing.assert_allclose(ring[:, 0], 7.0)
    assert bt.outline(xy, categories=lode).attributes["category"].tolist() == ["east", "west"]


def test_outline_refuses_bad_options():
    with pytest.raises(bt.InvalidInput, match="needs max_edge"):
        bt.outline(points, method="concave")
    with pytest.raises(bt.InvalidInput, match="needs max_edge"):
        bt.outline(points, max_edge=5.0)
    with pytest.raises(bt.InvalidInput, match="erases"):
        bt.outline(points, buffer=-500.0)
    with pytest.raises(bt.InvalidInput, match="three points"):
        bt.outline([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])


def terrain(seed=1, n=300):
    r = np.random.default_rng(seed)
    xy = r.uniform(0, 500, (n, 2))
    return np.c_[xy, 100 + 0.05 * xy[:, 0] + 10 * np.sin(xy[:, 1] / 80)]


def test_delaunay_topography_is_exact_at_the_points_and_on_planes():
    pts = terrain()
    t = bt.topography(pts, cell=10.0)
    np.testing.assert_allclose(
        t.mesh.vertical_distance(pts)[np.isfinite(t.mesh.vertical_distance(pts))], 0, atol=1e-9
    )
    plane = pts.copy()
    plane[:, 2] = 50 + 0.2 * pts[:, 0] - 0.1 * pts[:, 1]
    t = bt.topography(plane, cell=10.0, clip=False)
    c, z = t.grid.coords, t.grid["z"]
    ok = np.isfinite(z)
    assert ok.mean() > 0.8
    np.testing.assert_allclose(z[ok], 50 + 0.2 * c[ok, 0] - 0.1 * c[ok, 1], atol=1e-9)
    np.testing.assert_allclose(t.residuals, 0, atol=1e-6)
    assert not t.flagged.any()


def test_topography_flags_a_spike_but_not_its_neighbors_and_clips():
    pts = terrain()
    pts[7, 2] += 25.0
    t = bt.topography(pts, cell=10.0)
    assert t.flagged[7] and t.residuals[7] == pytest.approx(25, abs=1.0)
    near = np.linalg.norm(pts[:, :2] - pts[7, :2], axis=1) < 40
    assert t.flagged[near].sum() == 1
    inside = t.outline.contains(np.c_[t.grid.coords[:, :2], np.zeros(len(t.grid))])
    assert np.isnan(t.grid["z"][~inside]).all()
    full = bt.topography(pts, cell=10.0, clip=False, max_residual=100.0)
    assert full.outline is None and not full.flagged.any()


def test_topography_with_an_estimator():
    pts = terrain()
    t = bt.topography(
        pts, cell=25.0, estimator=bt.InverseDistance(bt.Search(100.0)), clip=False, extent=(0, 500, 0, 500)
    )
    assert t.grid.count[:2] == [20, 20]
    assert np.isfinite(t.grid["z"]).all() and np.isfinite(t.residuals).all()
