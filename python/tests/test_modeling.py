import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(7)
centre, radius = np.array([50.0, 50.0, 50.0]), 20.0
coords = rng.uniform(10, 90, (300, 3))
distance = np.linalg.norm(coords - centre, axis=1) - radius


def test_rbf_is_exact_at_data():
    model = cs.ImplicitModel().fit(coords, distance)
    np.testing.assert_allclose(model.evaluate(coords), distance, atol=1e-6)
    _, gradients = model.evaluate(coords[:5], gradient=True)
    assert gradients.shape == (5, 3)
    assert model.report is None


def test_cutoff_codes_grades():
    grade = 5 - distance
    coded = cs.ImplicitModel().fit(coords, np.where(grade >= 5, 1.0, -1.0))
    raw = cs.ImplicitModel().fit(coords, grade, cutoff=5)
    np.testing.assert_array_equal(raw.evaluate(coords[:20]), coded.evaluate(coords[:20]))
    with pytest.raises(cs.InvalidInput, match="cutoff"):
        cs.ImplicitModel().fit(coords, grade, cutoff=float("nan"))


def test_sphere_isosurface_area():
    model = cs.ImplicitModel().fit(coords, distance)
    blocks = cs.BlockModel(origin=(0, 0, 0), size=(2, 2, 2), count=(50, 50, 50))
    sphere = model.isosurface(blocks)
    assert sphere.triangles.dtype == np.int64 and sphere.is_closed
    assert sphere.area == pytest.approx(4 * np.pi * radius**2, rel=0.02)
    assert abs(sphere.volume) == pytest.approx(4 / 3 * np.pi * radius**3, rel=0.03)
    np.testing.assert_allclose(np.linalg.norm(sphere.vertices - centre, axis=1), radius, atol=0.5)

    half = cs.BlockModel(origin=(0, 0, 50), size=(2, 2, 2), count=(50, 50, 25))
    assert model.isosurface(half).analysis["boundary_edges"] > 0
    capped = model.isosurface(half, closed=True)
    assert capped.is_closed
    hemisphere = 2 / 3 * np.pi * radius**3
    assert 100 * 100 * 50 - capped.volume == pytest.approx(hemisphere, rel=0.02)


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


def test_anisotropy_variance_and_contact_positions():
    xyz = rng.uniform(0, 100, (60, 3))
    values = np.where(xyz[:, 2] > 50, 1.0, -1.0)
    flat = cs.ImplicitModel(drift_degree=0, rotation=(0, 0, 0), ratios=(1.0, 0.1)).fit(xyz, values)
    np.testing.assert_allclose(flat.evaluate(xyz), values, atol=1e-6)
    with pytest.raises(cs.InvalidInput, match="variogram"):
        cs.ImplicitModel("kriging", variogram=cs.Variogram([("spherical", 1.0, 50.0)]), rotation=(0, 0, 0))
    with pytest.raises(cs.InvalidInput, match="learns"):
        cs.ImplicitModel("gp", ratios=(1.0, 0.5))
    gp = cs.ImplicitModel("gp").fit(xyz, values)
    _, variance = gp.evaluate([[50, 50, 50], [500, 500, 500]], variance=True)
    assert variance[1] > variance[0] >= 0
    with pytest.raises(cs.InvalidInput, match="only"):
        flat.evaluate(xyz, variance=True)

    collar = {"HOLEID": np.array([1.0]), "X": np.array([10.0]), "Y": np.array([20.0]), "Z": np.array([100.0])}
    survey = {
        k: np.array(v) for k, v in {"HOLEID": [1.0], "DEPTH": [0.0], "AZIMUTH": [0.0], "DIP": [90.0]}.items()
    }
    dh = cs.Drillholes(
        collar, survey, {"HOLEID": np.array([1.0]), "FROM": np.array([0.0]), "TO": np.array([100.0])}
    )
    np.testing.assert_allclose(dh.at(["1.0"], [30.0]), [[10.0, 20.0, 70.0]], atol=1e-9)
    with pytest.raises(cs.InvalidInput):
        dh.at(["2.0"], [1.0])
