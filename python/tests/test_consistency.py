import boitata as bt
import numpy as np
import pytest

SIDE = 100
GRID = bt.BlockModel(origin=(0, 0), size=(1, 1), count=(SIDE, SIDE))


def channels(azimuth):
    """Sinuous channels on 30 % of a 100 x 100 image, running along `azimuth`."""
    band = {"shape": "channel", "code": 1, "proportion": 0.3, "width": (4, 10), "azimuth": azimuth}
    return bt.object_training_image(GRID, [{**band, "amplitude": 6, "wavelength": 60}], seed=7)


def holes(ti, n=25, seed=0):
    """`n` holes along y across the whole image at distinct random columns, every cell sampled."""
    columns = np.random.default_rng(seed).permutation(SIDE)[:n]
    i, j = np.repeat(columns, SIDE), np.tile(np.arange(SIDE), n)
    return bt.PointSet(np.column_stack([i + 0.5, j + 0.5]), {"facies": ti["facies"][i + SIDE * j]})


def test_data_from_the_image_are_consistent_and_from_a_rotated_one_are_not():
    east, north = channels(90), channels(0)
    data = holes(east)
    own = bt.training_image_consistency(east, "facies", data, "facies", seed=1)
    rotated = bt.training_image_consistency(north, "facies", data, "facies", seed=1)
    assert own["p_value"] > 0.05 and rotated["p_value"] < 0.025
    assert rotated["distance"] > 10 * own["distance"]
    assert (own["n_cells"], own["n_events"], len(own["reference"])) == (2500, 25 * 97, 200)
    assert np.all(np.diff(own["reference"]) >= 0)
    assert own["data_proportions"].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(own["proportions"], [0.7, 0.3], atol=0.01)


def test_seeded_and_continuous():
    east = channels(90)
    data = holes(east)
    xy = data.coords[:, :2]
    first = bt.training_image_consistency(east, "facies", xy, data["facies"], grid=GRID, axis="y", seed=3)
    again = bt.training_image_consistency(east, "facies", data, "facies", seed=3)
    np.testing.assert_array_equal(first["reference"], again["reference"])
    assert first["p_value"] == again["p_value"]
    continuous = bt.training_image_consistency(east, "facies", data, "facies", categorical=False, n_classes=2)
    assert continuous["proportions"] is None and continuous["data_proportions"] is None
    assert 0 <= continuous["distance"] <= 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"axis": "w"}, "axis must be"),
        ({"axis": "x", "pattern_length": 26}, "consecutive informed cells along x"),
        ({"pattern_length": 1}, "pattern_length must be at least 2"),
        ({"n_samples": 0}, "n_samples"),
    ],
)
def test_errors_say_what_to_change(kwargs, match):
    east = channels(90)
    with pytest.raises(bt.InvalidInput, match=match):
        bt.training_image_consistency(east, "facies", holes(east), "facies", **kwargs)


def test_codes_must_be_codes_of_the_image():
    east = channels(90)
    data = holes(east)
    with pytest.raises(bt.InvalidInput, match="not a code"):
        bt.training_image_consistency(east, "facies", data, data["facies"] + 2)
