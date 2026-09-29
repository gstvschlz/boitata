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
    d = surface.vertical_distance([[np.nan, 5, 0], [5, 5, np.nan], [5, 5, 4]])
    np.testing.assert_array_equal(np.isnan(d), [True, True, False])
    grid = cs.BlockModel(origin=(0, 0, 0), size=(5, 5, 5), count=(2, 2, 2))
    assert surface.vertical_distance(grid).shape == (8,)


def test_block_proportions():
    grid = cs.BlockModel(origin=(-5, 0, 0), size=(10, 10, 10), count=(2, 1, 1))
    np.testing.assert_allclose(cube.proportion(grid), [0.5, 0.5], atol=0.13)
    np.testing.assert_allclose(cube.proportion(grid, discretization=(8, 1, 1)), [0.5, 0.5])
    centroids = grid.centroids
    np.testing.assert_allclose(
        cube.proportion(centroids, size=(10, 10, 10), discretization=(8, 1, 1)), [0.5, 0.5]
    )
    with pytest.raises(TypeError):
        cube.proportion(centroids, (10, 10, 10))


def test_domains_and_classes_by_name():
    grid = cs.BlockModel(
        (0, 0, 0), (1, 1, 1), (4, 1, 1), attributes={"c": ["a", "b", "a", "a"], "d": [1, 1, 2, 2]}
    )
    by_name = cs.smooth_classes(grid, "c", window=(3, 1, 1), domain_column="d")
    np.testing.assert_array_equal(
        by_name, cs.smooth_classes(grid, grid["c"], window=(3, 1, 1), domains=[1, 1, 2, 2])
    )
    samples = cs.PointSet([[0, 0, 0], [9, 9, 0]], {"rock": ["ox", "fr"]})
    labels, _ = cs.assign_domain([[1, 1, 0]], coords=samples, domain_column="rock")
    assert labels == ["ox"]
    with pytest.raises(cs.InvalidInput):
        cs.assign_domain([[1, 1, 0]], coords=samples, domains=["ox", "fr"], domain_column="rock")
    with pytest.raises(TypeError):
        cs.smooth_classes(grid, "c", (3, 1, 1))


def test_smooth_classes_subblocked_votes_by_volume():
    extents = [[0, 0, 0, 0.6, 1, 1], [0.6, 0, 0, 0.8, 1, 1], [0.8, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    model = cs.BlockModel.subblocked((0, 0), (10, 10), (2, 1), np.array([0, 0, 0, 1], np.uint64), extents)
    np.testing.assert_array_equal(
        cs.smooth_classes(model, ["a", "b", "b", "b"], window=(1, 1, 1)), list("aaab")
    )
    np.testing.assert_array_equal(
        cs.smooth_classes(model, ["a", "b", "b", "b"], window=(3, 1, 1)), list("bbbb")
    )


def test_remove_small_units():
    rock = np.full(25, "ox")
    rock[12], rock[[3, 4, 8, 9]] = "fr", "tr"
    grid = cs.BlockModel((0, 0, 0), (2, 2, 1), (5, 5, 1), attributes={"rock": rock})
    out = cs.remove_small_units(grid, "rock", min_blocks=2)
    np.testing.assert_array_equal(out, np.where(np.arange(25) == 12, "ox", rock))
    np.testing.assert_array_equal(cs.remove_small_units(grid, "rock", min_volume=8.0), out)
    kept = cs.remove_small_units(grid, "rock", min_blocks=2, domains=np.arange(25) == 12)
    assert kept[12] == "fr"
    with pytest.raises(cs.InvalidInput):
        cs.remove_small_units(grid, "rock")
    with pytest.raises(cs.InvalidInput):
        cs.remove_small_units(grid, "rock", min_blocks=2, connectivity=18)


def test_contact_distance_and_buffers():
    grid = cs.BlockModel((0, 0, 0), (1, 1, 1), (10, 2, 1))
    rock = np.where(grid.centroids[:, 0] < 5, "a", "b")
    d = cs.contact_distance(grid, rock)
    np.testing.assert_allclose(d[:10], [5, 4, 3, 2, 1, 1, 2, 3, 4, 5])
    signed = cs.contact_distance(grid, rock, target="b")
    np.testing.assert_allclose(signed[:10], [5, 4, 3, 2, 1, -1, -2, -3, -4, -5])
    assert np.isnan(cs.contact_distance(grid, rock, target="c")).all()
    buffered = cs.buffer_domains(grid, rock, distance=2)
    np.testing.assert_array_equal(buffered[:10], list("aaa") + ["contact"] * 4 + list("bbb"))
    coded = cs.buffer_domains(grid, (rock == "b").astype(int), distance=1, label=-1, target=1)
    np.testing.assert_array_equal(coded[:10], [0, 0, 0, 0, -1, -1, 1, 1, 1, 1])


def test_polygons():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    np.testing.assert_array_equal(cs.point_in_polygon([[5, 5], [15, 5]], square), [True, False])
    assert cs.polygon_distance([[5, 5]], square, signed=True)[0] == pytest.approx(-5)
    ring = np.c_[square, np.zeros(4)]
    selector = cs.PolygonSelector([ring], closed=True, z_min=-1, z_max=1)
    np.testing.assert_array_equal(selector.contains([[5, 5, 0], [5, 5, 3]]), [True, False])


def test_polygon_with_a_hole():
    square = lambda o, s: [[o, o], [o + s, o], [o + s, o + s], [o, o + s]]
    lines = cs.Polylines([square(0, 10), square(4, 2), square(20, 1)], closed=True, features=[0, 0, 1])
    np.testing.assert_allclose(lines.area(), [96, 1])
    np.testing.assert_allclose(lines.length(), [48, 4])
    points = [[2, 2], [5, 5], [20.5, 20.5], [15, 5]]
    np.testing.assert_array_equal(lines.contains(points), [True, False, True, False])
    np.testing.assert_array_equal(lines.contains(points, feature=1), [False, False, True, False])
    np.testing.assert_array_equal(lines.locate(points), [0, -1, 1, -1])
    across = [[x, 5] for x in (-1, 1, 3.5, 4.5, 5)]
    np.testing.assert_allclose(lines.distance(across, signed=True), [1, -1, -0.5, 0.5, 1])
    np.testing.assert_allclose(lines.distance(across, feature=1)[0], np.hypot(21, 15))
    np.testing.assert_array_equal(cs.point_in_polygon(points, lines), lines.contains(points))
    np.testing.assert_array_equal(
        cs.polygon_distance(across, lines, signed=True), lines.distance(across, signed=True)
    )
    selector = cs.PolygonSelector(lines)
    np.testing.assert_array_equal(selector.contains(points), [True, False, True, False])
    raw = cs.PolygonSelector([square(0, 10), square(4, 2)], closed=True)
    np.testing.assert_array_equal(raw.contains(points), [True, False, False, False])
    with pytest.raises(cs.InvalidInput):
        lines.contains(points, feature=2)
    with pytest.raises(cs.InvalidInput):
        cs.Polylines([square(0, 1)]).distance(points)


def test_assign_domain_nearest_and_solid():
    labels, confidence = cs.assign_domain([[1, 1, 0]], coords=[[0, 0, 0], [9, 9, 0]], domains=["ox", "fr"])
    assert labels == ["ox"] and confidence[0] == 0.5
    coords = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0]]
    domains = ["z", "a", "a", "z", "c", "a"]
    labels, confidence = cs.assign_domain([[0, 0, 0]], coords=coords, domains=domains, method="majority")
    assert labels == ["z"] and confidence[0] == pytest.approx(0.4)
    labels, confidence = cs.assign_domain([[0, 0, 0]], coords=coords, domains=domains, method="majority", n=3)
    assert labels == ["a"] and confidence[0] == pytest.approx(2 / 3)
    with pytest.raises(cs.InvalidInput):
        cs.assign_domain([[0, 0, 0]], coords=coords, domains=domains, n=0)
    labels, _ = cs.assign_domain([[5, 5, 5], [20, 5, 5]], method="solid", mesh=cube)
    assert labels[0] != labels[1]


def test_block_shell_of_a_single_block():
    model = cs.BlockModel(origin=(0, 0, 0), size=(1, 1, 1), count=(1, 1, 1), attributes={"v": [2.0]})
    shell = cs.block_shell(model, column="v")
    assert shell.triangles.shape == (12, 3) and np.all(shell.face_attributes["value"] == 2.0)
    assert shell.area == pytest.approx(6)


def test_mesh_topology():
    assert cube.is_closed and cube.volume == pytest.approx(1000) and cube.area == pytest.approx(600)
    square = cs.Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]], crs="EPSG:32722")
    assert square.analysis["boundary_edges"] == 4 and not square.is_closed and square.crs == "EPSG:32722"
    assert "open, 4 boundary edges" in repr(square) and "closed" in repr(cube)
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
    labeled = square.with_face_column("layer", ["a", "b"]).with_vertex_column("z", [1, 2, 3, 4])
    assert list(labeled.face_attributes["layer"]) == ["a", "b"] and labeled.vertex_attributes.num_rows == 4
    with pytest.raises(cs.errors.InvalidInput):
        square.with_face_column("bad", [1.0])


def test_convex_hull():
    points = np.random.default_rng(0).normal(size=(300, 3))
    hull = cs.convex_hull(points)
    assert hull.is_closed and hull.contains(points * 0.999).all()
    assert not hull.contains([[10, 0, 0]])[0]
    with pytest.raises(ValueError):
        cs.convex_hull([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])


def test_grid_surface_of_a_plane_with_a_hole():
    grid = cs.BlockModel(origin=(0, 0, 0), size=(10, 5, 1), count=(6, 5, 1), crs="EPSG:32722")
    plane = lambda xy: 0.3 * xy[:, 0] - 0.2 * xy[:, 1] + 100
    z = plane(grid.centroids)
    surface = cs.grid_surface(grid.with_column("z", z), "z")
    assert surface.triangles.shape == (2 * 5 * 4, 3) and surface.crs == "EPSG:32722"
    assert surface.area == pytest.approx(50 * 20 * np.sqrt(1 + 0.3**2 + 0.2**2))
    pts = np.random.default_rng(0).uniform([5, 2.5], [55, 22.5], (300, 2))
    np.testing.assert_allclose(surface.vertical_distance(np.c_[pts, plane(pts)]), 0, atol=1e-9)
    z[8] = np.nan
    holed = cs.grid_surface(grid.with_column("z", z), "z")
    assert len(holed.triangles) == 40 - 4 and holed.area < surface.area
    assert np.isnan(holed.vertical_distance([[22, 9, 0]]))[0]
    with pytest.raises(ValueError):
        cs.grid_surface(cs.BlockModel((0, 0, 0), (1, 1, 1), (2, 2, 2), attributes={"z": np.zeros(8)}), "z")


def test_repair_rebuilds_a_broken_cube():
    rng = np.random.default_rng(1)
    soup = cube_vertices[cube_triangles] + rng.uniform(-1e-6, 1e-6, (12, 3, 3))
    triangles = np.arange(36).reshape(12, 3)
    triangles[::2] = triangles[::2, ::-1]
    triangles = np.r_[triangles, triangles[:1], [[0, 1, 1]]]
    broken = cs.Mesh(soup.reshape(-1, 3), triangles).with_face_column("face", np.arange(14.0))
    assert not broken.is_closed
    fixed = broken.repair(tolerance=1e-4)
    assert fixed.is_closed and fixed.volume == pytest.approx(1000)
    assert fixed.vertices.shape == (8, 3) and list(fixed.face_attributes["face"]) == list(range(12))
    again = fixed.repair(tolerance=1e-4)
    np.testing.assert_array_equal(again.triangles, fixed.triangles)
    np.testing.assert_array_equal(again.vertices, fixed.vertices)
    with pytest.raises(ValueError):
        cube.repair(tolerance=-1)


def test_repair_keeps_a_cavity_wound_inward():
    vertices = np.r_[cube_vertices * 3, cube_vertices + 10]
    flip = lambda t: t[:, [0, 2, 1]]
    outer, inner = cube_triangles, flip(cube_triangles) + 8
    hollow = cs.Mesh(vertices, np.r_[outer, inner])
    for start in (np.r_[outer, flip(inner)], np.r_[flip(outer), flip(inner)], np.r_[outer, inner]):
        fixed = cs.Mesh(vertices, start).repair()
        np.testing.assert_array_equal(fixed.triangles, hollow.triangles)
    assert fixed.volume == pytest.approx(27000 - 1000)
    np.testing.assert_array_equal(fixed.contains([[5, 5, 5], [15, 15, 15], [35, 5, 5]]), [True, False, False])
    blocks = [[5, 5, 5], [15, 15, 15]]
    np.testing.assert_allclose(fixed.proportion(blocks, size=(10, 10, 10)), [1, 0])


def test_subblocks_from_meshes_and_regularize():
    topo = cs.Mesh([[-10, -10, 7], [30, -10, 7], [30, 30, 7], [-10, 30, 7]], [[0, 1, 2], [0, 2, 3]])
    meshes = [(cube, "inside", "ore"), (topo, "below", "rock")]
    grid = cs.BlockModel((-4, -4, -4), (4, 4, 4), (5, 5, 4))
    sub = grid.subblock(meshes, 4, fill="air")
    same = cs.BlockModel.from_meshes((-4, -4, -4), (4, 4, 4), (5, 5, 4), meshes, (4, 4, 4), fill="air")
    np.testing.assert_array_equal(sub.extents, same.extents)
    domain = np.array(sub["domain"])
    volume = lambda label: sub.volumes[domain == label].sum()
    assert volume("ore") == pytest.approx(1000) and volume("rock") == pytest.approx(20 * 20 * 11 - 700)
    assert sub.volumes.sum() == pytest.approx(20 * 20 * 16)
    assert len(grid.subblock(meshes, 4)) < len(sub)
    with pytest.raises(cs.InvalidInput):
        grid.subblock([(cube, "beside", "ore")], 4)
    ring = lambda h: [[-h, -h], [h, -h], [h, h], [-h, h]]
    lease = cs.Polylines([ring(8), ring(4)], closed=True, features=[0, 0])
    prism = grid.subblock([(lease, "inside", "lease")], 2)
    assert prism.volumes.sum() == pytest.approx((12 * 12 - 8 * 8) * 16)
    with pytest.raises(cs.InvalidInput):
        grid.subblock([(lease, "below", "lease")], 2)

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


def test_from_extents_holds_every_object_with_the_buffer():
    rng = np.random.default_rng(7)
    points = cs.PointSet(rng.uniform((200, 300, 50), (600, 500, 150), (300, 3)))
    shifted = cs.Mesh(cube_vertices + (700, 450, 20), cube_triangles)
    model = cs.BlockModel.from_extents(points, shifted, size=(10, 10, 5), buffer=(20, 20, 5), crs="local")
    everything = np.vstack([points.coords, shifted.vertices])
    assert (model.row_at(everything) >= 0).all()
    np.testing.assert_allclose(model.origin, everything.min(0) - (20, 20, 5))
    top = np.array(model.origin) + np.array(model.count) * model.size
    assert (top >= everything.max(0) + (20, 20, 5)).all()
    assert (top - model.size < everything.max(0) + (20, 20, 5)).all()
    assert model.crs == "local" and len(model) == np.prod(model.count)


def test_from_extents_snaps_rotates_and_flattens():
    rng = np.random.default_rng(3)
    xyz = rng.uniform((13.3, 7.1, 2.2), (487.0, 233.0, 91.0), (200, 3))
    snapped = cs.BlockModel.from_extents(xyz, size=(25, 25, 10), snap=True)
    np.testing.assert_allclose(np.array(snapped.origin) % (25, 25, 10), 0)
    stepped = cs.BlockModel.from_extents(xyz, size=(5, 5, 5), snap=100)
    np.testing.assert_allclose(stepped.origin, (0, 0, 0))

    local = rng.uniform(0.5, (99.5, 49.5, 19.5), (300, 3))
    az = np.radians(30)
    world = np.c_[
        local[:, 0] * np.cos(az) + local[:, 1] * np.sin(az),
        -local[:, 0] * np.sin(az) + local[:, 1] * np.cos(az),
        local[:, 2],
    ]
    rotated = cs.BlockModel.from_extents(world, size=(10, 10, 10), rotation=(30, 0, 0))
    assert rotated.count == [10, 5, 2] and (rotated.row_at(world) >= 0).all()

    flat = cs.BlockModel.from_extents(xyz[:, :2], size=(10, 10))
    assert flat.count[2] == 1 and (flat.row_at(xyz[:, :2]) >= 0).all()
    deep = cs.BlockModel.from_extents(xyz, size=(10, 10, None))
    assert deep.count[2] == 1 and (deep.row_at(xyz) >= 0).all()
    with pytest.raises(ValueError):
        cs.BlockModel.from_extents(size=(10, 10))


def test_from_extents_covers_drill_holes_lines_and_grids():
    holes = cs.Drillholes(
        {"HOLE_ID": ["a", "b"], "X": [0.0, 100.0], "Y": [0.0, 0.0], "Z": [500.0, 500.0]},
        {"HOLE_ID": ["a", "b"], "DEPTH": [0.0, 0.0], "AZIMUTH": [0.0, 90.0], "DIP": [90.0, 45.0]},
        {"HOLE_ID": ["a", "b"], "FROM": [0.0, 0.0], "TO": [50.0, 80.0]},
    )
    line = cs.Polylines([np.array([[-40.0, 10, 480], [20, 60, 470]])])
    model = cs.BlockModel.from_extents(holes, line, size=(5, 5, 5))
    stations = holes.paths()
    ends = np.c_[stations["x"], stations["y"], stations["z"]]
    assert (model.row_at(np.vstack([ends, line.vertices])) >= 0).all()
    copy = cs.BlockModel.from_extents(model, size=model.size)
    assert copy.origin == model.origin and copy.count == model.count


def test_unfold_flat_and_folded_layers():
    grid = cs.BlockModel(origin=(0, 0, 0), size=(10, 10, 1), count=(11, 11, 1))
    x = grid.centroids[:, 0]
    base, top = (cs.grid_surface(grid.with_column("z", 0 * x + z), "z") for z in (0, 10))
    flat = cs.Unfold(base, top)
    pts = np.random.default_rng(0).uniform([5, 5, 0], [105, 105, 10], (200, 3))
    np.testing.assert_allclose(flat.transform(pts), pts / [1, 1, 10], atol=1e-12)
    np.testing.assert_allclose(flat.inverse(flat.fit_transform(pts)), pts, atol=1e-9)
    assert np.isnan(flat.transform([[50, 50, 11], [200, 50, 5]])).all()
    assert flat.transform(cs.PointSet(pts)).shape == (200, 3)
    with pytest.raises(ValueError):
        cs.Unfold(base, top, mode="up")

    fold = 20 * np.sin(x / 40)
    footwall = cs.grid_surface(grid.with_column("z", fold), "z")
    hangingwall = cs.grid_surface(grid.with_column("z", fold + 10), "z")
    unfold = cs.Unfold(footwall, hangingwall, reference="footwall")
    assert (unfold.transform(footwall.vertices)[:, 2] == 0).all()
    assert (unfold.transform(hangingwall.vertices)[:, 2] == 1).all()
    below = np.c_[footwall.vertices[:, :2], footwall.vertices[:, 2] - 2]
    assert np.isnan(unfold.transform(below)).all()
    extra = cs.Unfold(footwall, hangingwall, mode="footwall", extrapolate=True).transform(below)
    np.testing.assert_allclose(extra[:, 2], -2, atol=1e-9)


def test_kriging_on_unfolded_coordinates_follows_the_layer():
    grid = cs.BlockModel(origin=(0, 0, 0), size=(5, 5, 1), count=(81, 21, 1))
    x = grid.centroids[:, 0]
    fold = lambda x: 25 * np.sin(x / 40)
    footwall, hangingwall = (cs.grid_surface(grid.with_column("z", fold(x) + dz), "z") for dz in (0, 12))
    unfold = cs.Unfold(footwall, hangingwall, mode="footwall", reference="footwall")
    truth = lambda uvw: np.sin(uvw[:, 2] * np.pi / 4) + 0.3 * np.cos(uvw[:, 0] / 60)
    holes = [(hx, hy) for hx in np.arange(10, 400, 30.0) for hy in (30.0, 70.0)]
    samples = np.array([[hx, hy, fold(hx) + d] for hx, hy in holes for d in np.linspace(0.25, 11.75, 24)])
    targets = np.random.default_rng(7).uniform([10, 30, 0], [390, 70, 1], (400, 3))
    targets[:, 2] = fold(targets[:, 0]) + 0.5 + 11 * targets[:, 2]
    s, t = unfold.transform(samples), unfold.transform(targets)
    assert not np.isnan(s).any() and not np.isnan(t).any()
    model = cs.Variogram([("gaussian", 1.0, 150.0)], ratios=(1.0, 0.03))
    search = cs.Search(200.0, max_samples=24, ratios=(1.0, 0.05))
    error = {
        name: np.sqrt(
            np.mean((cs.OrdinaryKriging(model, search).fit(a, truth(s)).predict(b) - truth(t)) ** 2)
        )
        for name, a, b in [("real", samples, targets), ("unfolded", s, t)]
    }
    assert error["unfolded"] < 0.3 * error["real"], error


def _box_volume(corners):
    e0, e1, e2 = (corners[:, i] - corners[:, 0] for i in (1, 2, 4))
    return np.abs(np.sum(np.cross(e0, e1) * e2, axis=1))


def test_corners_hold_every_centroid_and_reconstruct_the_volume():
    bm = cs.BlockModel(origin=(10, 20, 0), size=(2, 3, 1), count=(2, 2, 1), rotation=(30, 0, 0))
    assert bm.corners.shape == (4, 8, 3)
    np.testing.assert_allclose(bm.corners.mean(axis=1), bm.centroids, atol=1e-9)
    np.testing.assert_allclose(_box_volume(bm.corners), bm.volumes, atol=1e-9)

    extents = [[0, 0, 0, 0.6, 1, 1], [0.6, 0, 0, 1, 1, 1]]
    sub = cs.BlockModel.subblocked(
        (0, 0, 0), (10, 10, 10), (1, 1, 1), np.array([0, 0], np.uint64), extents, rotation=(15, 20, 5)
    )
    np.testing.assert_allclose(_box_volume(sub.corners), sub.volumes, atol=1e-9)
