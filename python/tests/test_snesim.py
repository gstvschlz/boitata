import ceres as cs
import numpy as np
import pytest

CHANNELS = {"shape": "channel", "code": 1, "proportion": 0.25, "width": 5.0, "azimuth": (80, 100)}
ti = cs.object_training_image(cs.BlockModel((0, 0), (1, 1), (80, 80)), [CHANNELS], seed=3)
grid = cs.BlockModel((0, 0), (1, 1), (40, 40))


def test_hard_data_are_reproduced_exactly():
    rows = np.arange(0, 1600, 37)
    coords = grid.centroids[rows]
    codes = ti["facies"][rows].astype(int)
    s = cs.SNESIM(ti, "facies", template_size=20, n_levels=2).fit(coords, codes)
    r = s.simulate(grid, n=3, seed=1, keep=True, progress=False).realizations
    assert r.shape == (3, 1600)
    np.testing.assert_array_equal(r[:, rows], np.tile(codes, (3, 1)))


def test_unconditional_realizations_follow_the_image_and_the_targets():
    share = np.mean(ti["facies"] == 1)
    plain = cs.SNESIM(ti, "facies", template_size=20, n_levels=2).simulate(grid, n=6, progress=False)
    assert abs(plain.proportions[:, 1].mean() - share) < 0.08
    steered = cs.SNESIM(ti, "facies", template_size=20, n_levels=2, target_proportions=[0.6, 0.4], servo=0.9)
    assert abs(steered.simulate(grid, n=6, progress=False).proportions[:, 1].mean() - 0.4) < 0.05


def test_masked_targets_have_one_category_per_row():
    masked = grid.mask(grid.centroids[:, 0] < 20)
    s = cs.SNESIM(ti, "facies", template_size=12, n_levels=1).simulate(masked, n=2, keep=True, progress=False)
    assert s.realizations.shape == (2, 800)


def test_bad_inputs_are_refused():
    with pytest.raises(cs.InvalidInput, match="target proportions"):
        cs.SNESIM(ti, "facies", target_proportions=[1.0])
    with pytest.raises(cs.InvalidInput, match="servo"):
        cs.SNESIM(ti, "facies", servo=1.0)
    with pytest.raises(cs.InvalidInput, match="category 5"):
        cs.SNESIM(ti, "facies").fit([[0.5, 0.5]], [5])
    with pytest.raises(cs.InvalidInput, match="BlockModel"):
        cs.SNESIM(ti, "facies").simulate(np.zeros((3, 3)), n=1, progress=False)
