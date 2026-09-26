import ceres as cs
import numpy as np
import pytest

cube_vertices = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float) * 10
cube_triangles = np.array(
    [
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
        [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
    ]
)  # fmt: skip
cube = cs.Mesh(cube_vertices, cube_triangles)


def test_mesh_inside_and_distance():
    inside = cube.contains([[5, 5, 5], [15, 5, 5]])
    np.testing.assert_array_equal(inside, [True, False])
    d = cube.distance([[5, 5, 5], [15, 5, 5]], signed=True)
    assert d[0] == pytest.approx(-5) and d[1] == pytest.approx(5)


def test_vertical_distance_to_a_tilted_plane():
    x, y = np.meshgrid(np.arange(11.0), np.arange(11.0))
    plane = lambda x, y: 0.5 * x - 0.25 * y + 3
    i = np.arange(10)[None, :] + 11 * np.arange(10)[:, None]
    tris = np.vstack(
        [np.c_[i.ravel(), i.ravel() + 1, i.ravel() + 12], np.c_[i.ravel(), i.ravel() + 12, i.ravel() + 11]]
    )
    surface = cs.Mesh(np.c_[x.ravel(), y.ravel(), plane(x, y).ravel()], tris)
    pts = np.random.default_rng(0).uniform([0, 0, -5], [10, 10, 10], (1000, 3))
    np.testing.assert_allclose(
        surface.vertical_distance(pts), pts[:, 2] - plane(pts[:, 0], pts[:, 1]), atol=1e-9
    )
    assert np.isnan(surface.vertical_distance([[11, 5, 0]])).all()
    grid = cs.BlockModel(origin=(0, 0, 0), size=(5, 5, 5), count=(2, 2, 2))
    assert surface.vertical_distance(grid).shape == (8,)


def test_block_proportions():
    grid = cs.BlockModel(origin=(-5, 0, 0), size=(10, 10, 10), count=(2, 1, 1))
    np.testing.assert_allclose(cube.proportion(grid), [0.5, 0.5], atol=0.13)


def test_polygons():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    np.testing.assert_array_equal(cs.point_in_polygon([[5, 5], [15, 5]], square), [True, False])
    assert cs.polygon_distance([[5, 5]], square, signed=True)[0] == pytest.approx(-5)
    ring = np.c_[square, np.zeros(4)]
    selector = cs.PolygonSelector([ring], closed=True, z_min=-1, z_max=1)
    np.testing.assert_array_equal(selector.contains([[5, 5, 0], [5, 5, 3]]), [True, False])


def test_assign_domain_nearest_and_solid():
    labels, confidence = cs.assign_domain([[1, 1, 0]], [[0, 0, 0], [9, 9, 0]], ["ox", "fr"])
    assert 0 <= confidence[0] <= 1
    assert labels == ["ox"]
    labels, _ = cs.assign_domain([[5, 5, 5], [20, 5, 5]], method="solid", mesh=cube)
    assert labels[0] != labels[1]


def test_block_shell_of_a_single_block():
    model = cs.BlockModel(origin=(0, 0, 0), size=(1, 1, 1), count=(1, 1, 1), attributes={"v": [2.0]})
    shell = cs.block_shell(model, "v")
    assert shell.triangles.shape == (12, 3) and np.all(shell.face_attributes["value"] == 2.0)
    assert shell.area == pytest.approx(6)


def test_mesh_topology():
    assert cube.is_closed and cube.volume == pytest.approx(1000) and cube.area == pytest.approx(600)
    square = cs.Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]], crs="EPSG:32722")
    assert square.analysis["boundary_edges"] == 4 and not square.is_closed and square.crs == "EPSG:32722"
    fin = cs.Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], [[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    assert fin.analysis["non_manifold_edges"] == 1
    for call in (
        lambda: square.volume,
        lambda: square.contains([[0.5, 0.5, 0]]),
        lambda: square.proportion([[0, 0, 0]], size=(1, 1, 1)),
    ):
        with pytest.raises(cs.errors.InvalidInput):
            call()
    assert square.distance([[0.5, 0.5, 2]])[0] == pytest.approx(2)
    labelled = square.with_face_column("layer", ["a", "b"]).with_vertex_column("z", [1, 2, 3, 4])
    assert labelled.face_attributes["layer"] == ["a", "b"] and labelled.vertex_attributes.num_rows == 4
    with pytest.raises(cs.errors.InvalidInput):
        square.with_face_column("bad", [1.0])


def test_convex_hull():
    points = np.random.default_rng(0).normal(size=(300, 3))
    hull = cs.convex_hull(points)
    assert hull.is_closed and hull.contains(points * 0.999).all()
    assert not hull.contains([[10, 0, 0]])[0]
    with pytest.raises(ValueError):
        cs.convex_hull([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])


def test_subblocks_from_meshes_and_regularize():
    topo = cs.Mesh([[-10, -10, 7], [30, -10, 7], [30, 30, 7], [-10, 30, 7]], [[0, 1, 2], [0, 2, 3]])
    domains = [(cube, "inside", "ore"), (topo, "below", "rock")]
    grid = cs.BlockModel((-4, -4, -4), (4, 4, 4), (5, 5, 4))
    sub = grid.subblock(domains, 4, fill="air")
    same = cs.BlockModel.from_meshes((-4, -4, -4), (4, 4, 4), (5, 5, 4), domains, (4, 4, 4), fill="air")
    np.testing.assert_array_equal(sub.extents, same.extents)
    domain = np.array(sub["domain"])
    volume = lambda label: sub.volumes[domain == label].sum()
    assert volume("ore") == pytest.approx(1000) and volume("rock") == pytest.approx(20 * 20 * 11 - 700)
    assert sub.volumes.sum() == pytest.approx(20 * 20 * 16)
    assert len(grid.subblock(domains, 4)) < len(sub)
    with pytest.raises(cs.InvalidInput):
        grid.subblock([(cube, "beside", "ore")], 4)

    grade = sub.with_column("au", np.where(domain == "ore", 2.0, 0.5))
    coarse = cs.BlockModel((-4, -4, -4), (10, 10, 8), (2, 2, 2))
    out = grade.regularize(coarse, min_fraction=0.0)
    metal = (out.volumes * out["fraction"] * out["au"]).sum()
    assert metal == pytest.approx((grade.volumes * grade["au"]).sum())
    assert out["domain"][0] == "rock"
    back = grade.to_regular()
    assert "fraction" not in back.attributes.column_names
    with pytest.raises(cs.InvalidInput):
        grade.regularize(cs.BlockModel((0, 0, 0), (10, 10, 8), (2, 2, 2), rotation=(10, 0, 0)))


def test_rotated_proportions():
    grid = cs.BlockModel(origin=(0, 0, 0), size=(10, 10, 10), count=(2, 2, 1), rotation=(45, 0, 0))
    nodes = grid.discretize(20)
    expected = cube.contains(nodes.centroids).mean() * grid.volumes.sum()
    inside = (cube.proportion(grid, discretization=20) * grid.volumes).sum()
    assert inside == pytest.approx(expected, rel=0.01) and 0 < inside < 1000
