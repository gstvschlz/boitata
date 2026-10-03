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
