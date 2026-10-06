import pickle

import boitata as bt
import numpy as np
import pytest

CHANNELS = {"shape": "channel", "code": 1, "proportion": 0.25, "width": 5.0, "azimuth": (80, 100)}
ti = bt.object_training_image(bt.BlockModel((0, 0), (1, 1), (80, 80)), [CHANNELS], seed=3)
grid = bt.BlockModel((0, 0), (1, 1), (40, 40))


def test_hard_data_are_reproduced_exactly():
    rows = np.arange(0, 1600, 37)
    coords = grid.centroids[rows]
    codes = ti["facies"][rows].astype(int)
    s = bt.SNESIM(ti, "facies", template_size=20, n_levels=2).fit(coords, codes)
    r = s.simulate(grid, n=3, seed=1, keep=True).realizations
    assert r.shape == (3, 1600)
    np.testing.assert_array_equal(r[:, rows], np.tile(codes, (3, 1)))


def test_unconditional_realizations_follow_the_image_and_the_targets():
    share = np.mean(ti["facies"] == 1)
    plain = bt.SNESIM(ti, "facies", template_size=20, n_levels=2).simulate(grid, n=6)
    assert abs(plain.proportions[:, 1].mean() - share) < 0.08
    steered = bt.SNESIM(ti, "facies", template_size=20, n_levels=2, target_proportions=[0.6, 0.4], servo=0.9)
    assert abs(steered.simulate(grid, n=6).proportions[:, 1].mean() - 0.4) < 0.05


def test_masked_targets_have_one_category_per_row():
    masked = grid.mask(grid.centroids[:, 0] < 20)
    s = bt.SNESIM(ti, "facies", template_size=12, n_levels=1).simulate(masked, n=2, keep=True)
    assert s.realizations.shape == (2, 800)


def test_bad_inputs_are_refused():
    with pytest.raises(bt.InvalidInput, match="target proportions"):
        bt.SNESIM(ti, "facies", target_proportions=[1.0])
    with pytest.raises(bt.InvalidInput, match="servo"):
        bt.SNESIM(ti, "facies", servo=1.0)
    with pytest.raises(bt.InvalidInput, match="category 5"):
        bt.SNESIM(ti, "facies").fit([[0.5, 0.5]], [5])
    with pytest.raises(bt.InvalidInput, match="BlockModel"):
        bt.SNESIM(ti, "facies").simulate(np.zeros((3, 3)), n=1)


def sand_trend():
    """Probability of code 1 rising from 0.05 in the west to 0.95 in the east."""
    east = 0.05 + 0.9 * grid.centroids[:, 0] / 40
    return np.column_stack([1 - east, east])


def test_realizations_follow_a_soft_trend_given_as_columns_or_array():
    soft = sand_trend()
    targets = grid.with_columns({"shale": soft[:, 0], "sand": soft[:, 1]})
    s = bt.SNESIM(ti, "facies", template_size=20, n_levels=2)
    by_name = s.simulate(targets, n=6, soft=["shale", "sand"], keep=True)
    by_array = s.simulate(grid, n=6, soft=soft, keep=True)
    np.testing.assert_array_equal(by_name.realizations, by_array.realizations)
    columns = by_array.probabilities[:, 1].reshape(40, 40).mean(axis=0)
    bands = columns.reshape(4, 10).mean(axis=1)
    assert np.all(np.diff(bands) > 0.05), bands


def test_certain_soft_probabilities_are_reproduced_and_nan_rows_are_free():
    soft = np.full((1600, 2), np.nan)
    soft[::4] = [0.0, 1.0]
    soft[1::4] = [2.0, 0.0]
    s = bt.SNESIM(ti, "facies", template_size=20, n_levels=2)
    r = s.simulate(grid, n=3, soft=soft, keep=True).realizations
    assert np.all(r[:, ::4] == 1)
    assert np.all(r[:, 1::4] == 0)
    assert 0 < np.mean(r[:, 2::4]) < 1


def test_bad_soft_probabilities_are_refused():
    s = bt.SNESIM(ti, "facies", template_size=12, n_levels=1)
    with pytest.raises(bt.InvalidInput, match=r"shape \(1600, 2\)"):
        s.simulate(grid, n=1, soft=np.ones((1600, 3)))
    with pytest.raises(bt.InvalidInput, match=r"shape \(1600, 2\)"):
        s.simulate(grid, n=1, soft=np.ones((10, 2)))
    with pytest.raises(bt.InvalidInput, match="non-negative"):
        s.simulate(grid, n=1, soft=np.tile([1.2, -0.2], (1600, 1)))
    with pytest.raises(bt.InvalidInput, match="some probabilities missing"):
        s.simulate(grid, n=1, soft=np.tile([0.5, np.nan], (1600, 1)))
    with pytest.raises(bt.InvalidInput, match="give 2"):
        s.simulate(grid.with_column("p", np.ones(1600)), n=1, soft=["p"])


def field(targets, azimuth, semi=1.0, scale=1.0):
    n = len(targets.centroids)
    angles = np.column_stack([np.broadcast_to(azimuth, n), np.zeros(n), np.zeros(n)])
    ratios = np.column_stack([np.broadcast_to(semi, n), np.ones(n)])
    return bt.LocalAnisotropy(targets.centroids, angles, ratios, scales=np.broadcast_to(scale, n).copy())


def runs(r, axis):
    """Mean length of the runs of code 1 along x (axis 1) or y (axis 0) of (ny, nx) images."""
    lines = np.moveaxis(r, axis + 1, -1).reshape(-1, r.shape[1 + axis])
    padded = np.pad(lines == 1, ((0, 0), (1, 1))).astype(int)
    steps = np.diff(padded, axis=1)
    return (lines == 1).sum() / max((steps == 1).sum(), 1)


def test_channels_follow_a_rotation_field():
    targets = bt.BlockModel((0, 0), (1, 1), (64, 64))
    azimuth = np.where(targets.centroids[:, 0] < 32, 0.0, 90.0)
    s = bt.SNESIM(ti, "facies", template_size=24, n_levels=2)
    r = s.simulate(targets, n=6, anisotropy=field(targets, azimuth), keep=True)
    images = r.realizations.reshape(6, 64, 64)
    west, east = images[:, :, :32], images[:, :, 32:]
    # The image's channels run east; azimuth 90 turns its north to the east,
    # so its channels run south there.
    assert runs(west, 1) > 1.4 * runs(west, 0)
    assert runs(east, 0) > 1.4 * runs(east, 1)
    assert s.n_classes == 2


def test_affinity_widens_the_channels():
    s = bt.SNESIM(ti, "facies", template_size=24, n_levels=2)

    def width(scale):
        a = field(grid, 0.0, semi=1 / scale, scale=scale)
        r = s.simulate(grid, n=6, anisotropy=a, keep=True).realizations
        return runs(r.reshape(6, 40, 40), 0)

    assert 1.4 < width(2.0) / width(1.0) < 2.6


def test_angle_step_sets_the_template_classes():
    a = field(grid, np.arange(1600) * 0.3)
    fine = bt.SNESIM(ti, "facies", template_size=12, n_levels=1)
    coarse = bt.SNESIM(ti, "facies", template_size=12, n_levels=1, angle_step=90)
    fine.simulate(grid, n=1, anisotropy=a)
    coarse.simulate(grid, n=1, anisotropy=a)
    assert (fine.n_classes, coarse.n_classes) == (36, 4)


other = bt.object_training_image(
    bt.BlockModel((0, 0), (1, 1), (80, 80)), [dict(CHANNELS, code=2, proportion=0.5)], seed=4
)


def zoned():
    return bt.SNESIM({"west": (ti, "facies"), "east": (other, "facies")}, template_size=20, n_levels=2)


def test_each_domain_draws_from_its_training_image():
    domains = np.where(grid.centroids[:, 0] < 20, "west", "east")
    r = zoned().simulate(grid, n=12, domains=domains, keep=True).realizations
    west, east = r[:, domains == "west"], r[:, domains == "east"]
    assert set(np.unique(west)) <= {0, 1} and set(np.unique(east)) <= {0, 2}
    assert abs(np.mean(west == 1) - np.mean(ti["facies"] == 1)) < 0.1
    assert abs(np.mean(east == 2) - np.mean(other["facies"] == 2)) < 0.1
    by_column = zoned().simulate(grid.with_column("zone", domains), n=12, domain_column="zone", keep=True)
    np.testing.assert_array_equal(by_column.realizations, r)


def test_zones_and_anisotropy_combine_and_persist(tmp_path):
    domains = np.where(grid.centroids[:, 1] < 20, "west", "east")
    a = field(grid, np.where(grid.centroids[:, 0] < 20, 0.0, 90.0))
    s = zoned().fit(grid.centroids[:3], [0, 2, 1])
    first = s.simulate(grid, n=2, domains=domains, anisotropy=a, keep=True)
    assert s.n_classes == 4
    path = tmp_path / "snesim.parquet"
    s.to_parquet(path)
    for back in (bt.SNESIM.from_parquet(path), pickle.loads(pickle.dumps(s))):
        again = back.simulate(grid, n=2, domains=domains, anisotropy=a, keep=True)
        np.testing.assert_array_equal(again.realizations, first.realizations)


def test_bad_domains_are_refused():
    domains = np.where(grid.centroids[:, 0] < 20, "west", "north")
    with pytest.raises(bt.InvalidInput, match='domain "north" has no training image'):
        zoned().simulate(grid, n=1, domains=domains)
    with pytest.raises(bt.InvalidInput, match="give domains or domain_column"):
        zoned().simulate(grid, n=1)
    with pytest.raises(bt.InvalidInput, match="give ti as a dict"):
        bt.SNESIM(ti, "facies").simulate(grid, n=1, domains="west")
    with pytest.raises(bt.InvalidInput, match="dict names each image's column"):
        bt.SNESIM({"a": (ti, "facies")}, "facies")
    with pytest.raises(bt.InvalidInput, match="pair"):
        bt.SNESIM({"a": ti})
    with pytest.raises(bt.InvalidInput, match="angle_step"):
        bt.SNESIM(ti, "facies", angle_step=0)


def grades():
    """A value peaking at the center of each channel of `ti`, over a background rising from 0 to 0.5 along x."""
    x, y = ti.centroids[:, 0], ti.centroids[:, 1]
    return ti.with_columns({"grade": np.where(ti["facies"] == 1, 1 + 2 * np.abs(np.sin(0.7 * y)), x / 160)})


def test_continuous_images_simulate_values_and_reproduce_the_data():
    image = grades()
    rows = np.arange(0, 1600, 37)
    values = np.linspace(0.0, 3.0, rows.size)
    s = bt.SNESIM(image, "grade", template_size=20, n_levels=2)
    assert not s.categorical
    summary = s.fit(grid.centroids[rows], values).simulate(grid, n=4, seed=1, keep=True)
    assert isinstance(summary, bt.SimulationSummary)
    np.testing.assert_allclose(summary.realizations[:, rows], np.tile(values, (4, 1)))
    free = np.delete(summary.realizations, rows, axis=1)
    assert np.isin(free.astype(np.float32), image["grade"].astype(np.float32)).all()
    with pytest.raises(bt.InvalidInput, match="categorical training image"):
        s.simulate(grid, n=1, soft=np.full((1600, 2), 0.5))
    with pytest.raises(bt.InvalidInput, match="strictly ascending"):
        bt.SNESIM(image, "grade", cutoffs=[2.0, 1.0])


def test_continuous_round_trip_simulates_bit_identically(tmp_path):
    path = tmp_path / "snesim.parquet"
    s = bt.SNESIM(grades(), "grade", template_size=12, n_levels=1, cutoffs=[0.2, 1.5, 2.5])
    s.fit(grid.centroids[:30], np.linspace(0, 3, 30))
    s.to_parquet(path)
    back = bt.SNESIM.from_parquet(path)
    np.testing.assert_array_equal(
        back.simulate(grid, n=2, seed=3, keep=True).realizations,
        s.simulate(grid, n=2, seed=3, keep=True).realizations,
    )


def test_continuous_zones_and_anisotropy_round_trip(tmp_path):
    low, high = grades(), grades()
    high = high.with_columns({"grade": high["grade"] + 10})
    s = bt.SNESIM({"west": (low, "grade"), "east": (high, "grade")}, template_size=12, n_levels=1)
    domains = np.where(grid.centroids[:, 0] < 20, "west", "east")
    field = bt.LocalAnisotropy(grid.centroids, np.tile([45.0, 0, 0], (1600, 1)), np.ones((1600, 2)))
    run = {"n": 2, "seed": 5, "keep": True, "domains": domains, "anisotropy": field}
    r = s.simulate(grid, **run).realizations
    np.testing.assert_array_equal(r >= 10, np.tile(domains == "east", (2, 1)))
    s.to_parquet(tmp_path / "zoned.parquet")
    back = bt.SNESIM.from_parquet(tmp_path / "zoned.parquet")
    np.testing.assert_array_equal(back.simulate(grid, **run).realizations, r)
