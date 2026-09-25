import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(7)
centre, radius = np.array([50.0, 50.0, 50.0]), 20.0
coords = rng.uniform(10, 90, (300, 3))
distance = np.linalg.norm(coords - centre, axis=1) - radius


def area(vertices, triangles):
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()


def test_rbf_is_exact_at_data():
    model = cs.ImplicitModel().fit(coords, distance)
    np.testing.assert_allclose(model.evaluate(coords), distance, atol=1e-6)
    _, gradients = model.evaluate(coords[:5], gradient=True)
    assert gradients.shape == (5, 3)
    assert model.report is None


def test_sphere_isosurface_area():
    model = cs.ImplicitModel().fit(coords, distance)
    blocks = cs.BlockModel(origin=(0, 0, 0), size=(2, 2, 2), count=(50, 50, 50))
    vertices, triangles = model.isosurface(blocks)
    assert triangles.dtype == np.int64
    assert area(vertices, triangles) == pytest.approx(4 * np.pi * radius**2, rel=0.02)
    np.testing.assert_allclose(np.linalg.norm(vertices - centre, axis=1), radius, atol=0.5)

    half = cs.BlockModel(origin=(0, 0, 50), size=(2, 2, 2), count=(50, 50, 25))
    _, open_triangles = model.isosurface(half)
    vertices, closed_triangles = model.isosurface(half, closed=True)
    assert shared_edges(closed_triangles).min() == 2 and shared_edges(open_triangles).min() == 1
    a, b, c = (vertices[closed_triangles[:, i]] for i in range(3))
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    hemisphere = 2 / 3 * np.pi * radius**3
    assert 100 * 100 * 50 - volume == pytest.approx(hemisphere, rel=0.02)


def shared_edges(triangles):
    edges = np.sort(np.r_[triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
    return np.unique(edges, axis=0, return_counts=True)[1]


def test_planes_with_boundaries():
    xy = rng.uniform(0, 100, (20, 2))
    on_plane = np.c_[xy, 0.5 * xy[:, 0]]
    dip = np.degrees(np.arctan(0.5))
    planes = np.c_[on_plane, np.full(20, dip), np.full(20, 270.0)]
    with pytest.raises(cs.InvalidInput, match="triharmonic"):
        cs.ImplicitModel().fit([[50, 50, 60]], [1.0], boundaries=on_plane, planes=planes)
    model = cs.ImplicitModel(kernel="triharmonic").fit(
        [[50, 50, 60]], [1.0], boundaries=on_plane, planes=planes
    )
    probe = np.array([[30, 70, 15], [30, 70, 25], [30, 70, 5]], float)
    value = model.evaluate(probe)
    assert abs(value[0]) < 0.05 * value[1] and value[1] > 0 > value[2]
    with pytest.raises(cs.InvalidInput):
        cs.ImplicitModel().fit(planes=planes)


def test_engines():
    with pytest.raises(cs.InvalidInput):
        cs.ImplicitModel(engine="kriging")
    variogram = cs.Variogram([("gaussian", 1.0, 60.0)])
    model = cs.ImplicitModel(engine="kriging", variogram=variogram, drift_degree=0).fit(
        coords[:50], distance[:50]
    )
    np.testing.assert_allclose(model.evaluate(coords[:50]), distance[:50], atol=1e-3)
    gp = cs.ImplicitModel(engine="gp").fit(coords, distance)
    assert gp.report["status"] and np.isfinite(gp.evaluate(coords)).all()
    with pytest.raises(cs.InvalidInput):
        cs.ImplicitModel().evaluate(coords)
