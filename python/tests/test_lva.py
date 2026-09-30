import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(8)


def layered(azimuth=30.0, n=40):
    grid = bt.BlockModel(origin=(0, 0), size=(1, 1), count=(n, n))
    xy = grid.centroids
    t = np.radians(azimuth)
    values = np.sin(0.3 * (xy[:, 0] * np.cos(t) - xy[:, 1] * np.sin(t)))
    return grid.with_column("v", values)


def test_from_grid_follows_the_layers():
    lva = bt.LocalAnisotropy.from_grid(layered(), "v", window=2)
    azimuth = lva.angles[820, 0] % 180
    assert min(abs(azimuth - 30), 180 - abs(azimuth - 30)) < 1
    assert np.all(lva.ratios <= 1)


def test_smooth_and_at():
    lva = bt.LocalAnisotropy(np.zeros((1, 3)), [[40.0, 0.0, 0.0]], [[0.5, 1.0]])
    moved = lva.at([[10.0, 10.0], [20.0, 5.0]])
    np.testing.assert_allclose(moved.angles[:, 0], 40.0)
    assert len(moved.smooth(5.0)) == 2


def test_uniform_field_equals_global_anisotropy():
    coords = rng.uniform(0, 100, (200, 2))
    values = np.sin(coords[:, 0] / 12) + coords[:, 1] / 40
    targets = rng.uniform(5, 95, (50, 2))
    search = bt.Search(radius=20, max_samples=500)
    base = bt.Variogram([("spherical", 1.0, 40.0)])
    field = bt.LocalAnisotropy(targets, np.tile([35.0, 0.0, 0.0], (50, 1)), np.tile([0.3, 1.0], (50, 1)))
    local = bt.OrdinaryKriging(base, search).fit(coords, values).predict(targets, anisotropy=field)
    rotated = base.with_anisotropy((35.0, 0.0, 0.0), (0.3, 1.0))
    reference = bt.OrdinaryKriging(rotated, search).fit(coords, values).predict(targets)
    np.testing.assert_allclose(local, reference, atol=1e-9)


def test_cokriging_with_a_uniform_field_equals_global_anisotropy():
    coords = rng.uniform(0, 100, (200, 2))
    values = np.sin(coords[:, 0] / 12) + coords[:, 1] / 40
    secondary = values + rng.normal(0, 0.3, 200)
    targets = rng.uniform(5, 95, (50, 2))
    field = bt.LocalAnisotropy(targets, np.tile([35.0, 0.0, 0.0], (50, 1)), np.tile([0.3, 1.0], (50, 1)))
    structures = [("spherical", 40.0, [[1.0, 0.7], [0.7, 1.0]])]
    search = bt.Search(radius=20, max_samples=16)

    def cokriging(**rotated):
        lmc = bt.Coregionalization([[0.0, 0.0], [0.0, 0.0]], structures=structures, **rotated)
        return bt.Cokriging(lmc, search).fit(
            np.vstack([coords, coords]), np.r_[values, secondary], [0] * 200 + [1] * 200
        )

    at = bt.PointSet(targets, {"s": np.sin(targets[:, 0] / 12)})
    local = cokriging().predict(at, anisotropy=field, collocated={1: "s"}, progress=False)
    reference = cokriging(rotation=(35.0, 0.0, 0.0), ratios=(0.3, 1.0)).predict(
        at, collocated={1: at["s"]}, progress=False
    )
    np.testing.assert_allclose(local, reference, atol=1e-9)


def test_sgs_with_local_anisotropy():
    coords = rng.uniform(0, 40, (60, 2))
    values = rng.normal(size=60)
    grid = layered()
    lva = bt.LocalAnisotropy.from_grid(grid, "v", ratios=(0.3, 1.0))
    sgs = bt.SGS(bt.Variogram([("spherical", 1.0, 15.0)]), bt.Search(radius=15, max_samples=12)).fit(
        coords, values
    )
    reals = sgs.simulate(grid, n=2, seed=1, keep=True, anisotropy=lva).realizations
    assert reals.shape == (2, 1600) and np.all(np.isfinite(reals))


def test_from_points_and_mesh():
    line = np.c_[np.arange(30.0), np.arange(30.0), rng.normal(0, 0.05, 30)]
    lva = bt.LocalAnisotropy.from_points(line, k=8)
    assert abs(lva.angles[15, 0] % 180 - 45) < 2
    mesh = bt.Mesh([[0, 0, 0], [0, 10, 0], [10, 0, -10], [10, 10, -10]], [[0, 1, 2], [1, 3, 2]])
    dip = bt.LocalAnisotropy.from_mesh(mesh, [[5, 5, -5]], major="dip")
    assert dip.angles[0, 0] == pytest.approx(90) and dip.angles[0, 1] == pytest.approx(45)


def waves(azimuth, major, ratio, seed, k=400):
    r = np.random.default_rng(seed)
    t = np.radians(azimuth)
    along, across = np.array([np.sin(t), np.cos(t)]), np.array([-np.cos(t), np.sin(t)])
    w = (
        r.normal(size=(k, 1)) * np.sqrt(6) / major * along
        + r.normal(size=(k, 1)) * np.sqrt(6) / (major * ratio) * across
    )
    phase = r.uniform(0, 2 * np.pi, k)
    return lambda xy: np.sqrt(2 / k) * np.cos(xy @ w.T + phase).sum(axis=1)


def two_regions(n=1500):
    xy = np.random.default_rng(3).uniform((0, 0), (200, 100), (n, 2))
    west, east = waves(30.0, 20.0, 0.25, 1), waves(120.0, 8.0, 0.4, 2)
    return xy, np.where(xy[:, 0] < 100, west(xy), east(xy))


def test_constant_scales_equal_a_longer_global_range():
    coords = rng.uniform(0, 100, (200, 2))
    values = np.sin(coords[:, 0] / 12) + coords[:, 1] / 40
    targets = rng.uniform(5, 95, (50, 2))
    field = bt.LocalAnisotropy(
        targets, np.tile([35.0, 0.0, 0.0], (50, 1)), np.tile([0.3, 1.0], (50, 1)), scales=1.5
    )
    base = bt.Variogram([("spherical", 1.0, 40.0)])
    local = bt.OrdinaryKriging(base, bt.Search(radius=20, max_samples=12)).fit(coords, values)
    longer = bt.Variogram([("spherical", 1.0, 60.0)], rotation=(35.0, 0.0, 0.0), ratios=(0.3, 1.0))
    reference = bt.OrdinaryKriging(longer, bt.Search(radius=30, max_samples=12)).fit(coords, values)
    np.testing.assert_allclose(
        local.predict(targets, anisotropy=field), reference.predict(targets), atol=1e-9
    )
    np.testing.assert_allclose(field.scales, 1.5)


def test_local_parameters_recover_each_region():
    xy, values = two_regions(6000)
    vg = bt.Variogram([("gaussian", 0.99, 12.0)], nugget=0.01)
    nodes = np.array([[x, y] for x in (25.0, 50.0, 75.0, 125.0, 150.0, 175.0) for y in (25.0, 50.0, 75.0)])
    west = nodes[:, 0] < 100
    local = bt.local_variogram_parameters(xy, values, nodes, variogram=vg, window=25, lag=2, max_lag=15)
    for side, (azimuth, ratio, scale) in [(west, (30, 0.25, 20 / 12)), (~west, (120, 0.4, 8 / 12))]:
        d = np.median(local.angles[side, 0] % 180) - azimuth
        assert abs(d) < 10
        assert abs(np.median(local.ratios[side, 0]) - ratio) < 0.15
        assert np.median(local.scales[side]) == pytest.approx(scale, rel=0.2)
    angles = np.where(west, 30.0, 120.0)
    field = bt.LocalAnisotropy(nodes, np.c_[angles, 0 * angles, 0 * angles], np.ones((len(nodes), 2)))
    along = bt.local_variogram_parameters(xy, values, nodes, variogram=vg, window=25, lag=2, anisotropy=field)
    np.testing.assert_allclose(along.angles[:, 0] % 180, angles, atol=1e-6)
    assert np.median(along.scales[west]) == pytest.approx(20 / 12, rel=0.2)
    assert np.median(along.ratios[~west, 0]) == pytest.approx(0.4, abs=0.1)


def test_experimental_variogram_in_local_frames():
    xy, values = two_regions(400)
    field = bt.LocalAnisotropy(xy, np.tile([30.0, 0.0, 0.0], (400, 1)), np.ones((400, 2)))
    local = bt.experimental_variogram(xy, values, 3, 30, azimuth=10.0, anisotropy=field)
    world = bt.experimental_variogram(xy, values, 3, 30, azimuth=40.0)
    np.testing.assert_array_equal(local.counts, world.counts)
    np.testing.assert_allclose(local.gammas, world.gammas, rtol=1e-12)
    with pytest.raises(ValueError):
        bt.experimental_variogram(xy, values, 3, 30, anisotropy=field, holes=np.zeros(400))
