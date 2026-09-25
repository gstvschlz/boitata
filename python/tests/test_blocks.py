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
    _, triangles, values = cs.block_shell(model, "v")
    assert triangles.shape == (12, 3) and np.all(values == 2.0)
