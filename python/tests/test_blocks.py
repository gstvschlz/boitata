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
