import boitata as bt
import numpy as np
import pytest
from boitata.errors import InvalidInput

truth = bt.LocalGrid(
    scale=0.9996, rotation=120.0, shift=(512_000.0, 7_400_000.0), z_shift=8.0, crs="EPSG:31982"
)
local = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 10.0], [0.0, 2000.0, -5.0], [750.0, 1300.0, 40.0]])


@pytest.mark.parametrize("rotation", [0.0, 35.0, 120.0, 180.0, 300.0])
def test_fit_recovers_the_similarity_with_zero_residuals(rotation):
    g = bt.LocalGrid(scale=truth.scale, rotation=rotation, shift=truth.shift, z_shift=truth.z_shift)
    projected = g.apply(local)
    fit = bt.LocalGrid.fit(local, projected, crs="EPSG:31982")
    assert fit.scale == pytest.approx(g.scale, abs=1e-12)
    assert (fit.rotation - rotation + 180) % 360 - 180 == pytest.approx(0, abs=1e-9)
    assert fit.z_shift == pytest.approx(8.0)
    np.testing.assert_allclose(fit.residuals(local, projected), 0, atol=1e-6)
    np.testing.assert_allclose(fit.inverse(fit.apply(local)), local, atol=1e-9)


def test_containers_round_trip_and_carry_the_crs():
    points = bt.PointSet(local, {"v": np.arange(4.0)})
    moved = truth.apply(points)
    assert moved.crs == "EPSG:31982"
    back = truth.inverse(moved)
    assert back.crs is None
    np.testing.assert_allclose(back.coords, local, atol=1e-9)
    with pytest.raises(InvalidInput):
        truth.apply(moved)
    lines = truth.apply(bt.Polylines([local]))
    np.testing.assert_allclose(lines.coords, moved.coords)
    mesh = truth.apply(bt.Mesh(local, [[0, 1, 2], [1, 2, 3]]))
    np.testing.assert_allclose(mesh.coords, moved.coords)


def test_block_model_keeps_its_cells_on_the_moved_points():
    model = bt.BlockModel((100.0, 200.0, 50.0), (10.0, 5.0, 2.0), (3, 4, 2), rotation=(30.0, 0.0, 0.0))
    moved = truth.apply(model)
    np.testing.assert_allclose(moved.coords, truth.apply(model.coords), atol=1e-6)
    np.testing.assert_allclose(moved.size, [10 * 0.9996, 5 * 0.9996, 2.0])
    assert moved.rotation[0] == pytest.approx(150.0)
    np.testing.assert_allclose(truth.inverse(moved).coords, model.coords, atol=1e-6)


def test_drillholes_move_with_their_samples():
    collar = {"HOLE_ID": [1.0], "X": [10.0], "Y": [20.0], "Z": [500.0]}
    survey = {"HOLE_ID": [1.0], "DEPTH": [0.0], "AZIMUTH": [45.0], "DIP": [60.0]}
    intervals = {"HOLE_ID": [1.0, 1.0], "FROM": [0.0, 5.0], "TO": [5.0, 10.0]}
    holes = bt.Drillholes(collar, survey, intervals)
    moved = truth.apply(holes)
    assert moved.crs == "EPSG:31982" and moved.samples().crs == "EPSG:31982"
    np.testing.assert_allclose(moved.samples().coords, truth.apply(holes.samples().coords), atol=1e-6)


def test_fit_rejects_bad_controls():
    with pytest.raises(InvalidInput):
        bt.LocalGrid.fit(local[:1], local[:1])
    with pytest.raises(InvalidInput):
        bt.LocalGrid.fit(local, local[:3])
