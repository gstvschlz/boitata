import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(8)


def layered(azimuth=30.0, n=40):
    grid = cs.BlockModel(origin=(0, 0), size=(1, 1), count=(n, n))
    xy = grid.centroids
    t = np.radians(azimuth)
    values = np.sin(0.3 * (xy[:, 0] * np.cos(t) - xy[:, 1] * np.sin(t)))
    return grid.with_column("v", values)


def test_from_grid_follows_the_layers():
    lva = cs.LocalAnisotropy.from_grid(layered(), "v", window=2)
    azimuth = lva.angles[820, 0] % 180
    assert min(abs(azimuth - 30), 180 - abs(azimuth - 30)) < 1
    assert np.all(lva.ratios <= 1)


def test_smooth_and_at():
    lva = cs.LocalAnisotropy(np.zeros((1, 3)), [[40.0, 0.0, 0.0]], [[0.5, 1.0]])
    moved = lva.at([[10.0, 10.0], [20.0, 5.0]])
    np.testing.assert_allclose(moved.angles[:, 0], 40.0)
    assert len(moved.smooth(5.0)) == 2


def test_uniform_field_equals_global_anisotropy():
    coords = rng.uniform(0, 100, (200, 2))
    values = np.sin(coords[:, 0] / 12) + coords[:, 1] / 40
    targets = rng.uniform(5, 95, (50, 2))
    search = cs.Search(radius=20, max_samples=500)
    base = cs.Variogram([("spherical", 1.0, 40.0)])
    field = cs.LocalAnisotropy(targets, np.tile([35.0, 0.0, 0.0], (50, 1)), np.tile([0.3, 1.0], (50, 1)))
    local = cs.OrdinaryKriging(base, search).fit(coords, values).predict(targets, anisotropy=field)
    rotated = base.with_anisotropy((35.0, 0.0, 0.0), (0.3, 1.0))
    reference = cs.OrdinaryKriging(rotated, search).fit(coords, values).predict(targets)
    np.testing.assert_allclose(local, reference, atol=1e-9)


def test_sgs_with_local_anisotropy():
    coords = rng.uniform(0, 40, (60, 2))
    values = rng.normal(size=60)
    grid = layered()
    lva = cs.LocalAnisotropy.from_grid(grid, "v", ratios=(0.3, 1.0))
    sgs = cs.SGS(cs.Variogram([("spherical", 1.0, 15.0)]), cs.Search(radius=15, max_samples=12)).fit(
        coords, values
    )
    reals = sgs.simulate(grid, n=2, seed=1, anisotropy=lva)
    assert reals.shape == (2, 1600) and np.all(np.isfinite(reals))


def test_from_points_and_mesh():
    line = np.c_[np.arange(30.0), np.arange(30.0), rng.normal(0, 0.05, 30)]
    lva = cs.LocalAnisotropy.from_points(line, k=8)
    assert abs(lva.angles[15, 0] % 180 - 45) < 2
    mesh = cs.Mesh([[0, 0, 0], [0, 10, 0], [10, 0, -10], [10, 10, -10]], [[0, 1, 2], [1, 3, 2]])
    dip = cs.LocalAnisotropy.from_mesh(mesh, [[5, 5, -5]], major="dip")
    assert dip.angles[0, 0] == pytest.approx(90) and dip.angles[0, 1] == pytest.approx(45)
