import ceres as cs
import numpy as np
import pytest

collar = {
    "HOLEID": np.array([1.0, 2.0]),
    "X": np.array([0.0, 100.0]),
    "Y": np.zeros(2),
    "Z": np.full(2, 500.0),
}
survey = {
    "HOLEID": np.array([1.0, 2.0]),
    "DEPTH": np.zeros(2),
    "AZIMUTH": np.array([0.0, 90.0]),
    "DIP": np.array([90.0, 45.0]),
}
intervals = {
    "HOLEID": np.array([1.0, 1.0, 1.0, 2.0, 2.0]),
    "FROM": np.array([0.0, 2.0, 5.0, 0.0, 4.0]),
    "TO": np.array([2.0, 5.0, 6.0, 4.0, 8.0]),
    "AU": np.array([1.0, 2.0, np.nan, 3.0, 5.0]),
}


def test_desurvey_follows_dip_and_azimuth():
    dh = cs.Drillholes(collar, survey, intervals)
    assert dh.holes == ["1.0", "2.0"] and len(dh) == 2
    samples = dh.samples()
    np.testing.assert_allclose(samples.coords[0], [0, 0, 499])
    d = 6 / np.sqrt(2)
    np.testing.assert_allclose(samples.coords[4], [100 + d, 0, 500 - d], atol=1e-9)


def test_compositing_conserves_metal():
    comps = cs.Drillholes(collar, survey, intervals).composite(3.0, ["AU"])
    second = np.array(comps.attributes["hole"]) == "2.0"
    assert np.dot(comps["length"][second], comps["AU"][second]) == pytest.approx(4 * 3 + 4 * 5)
    assert np.all(comps["length"] <= 3.0 + 1e-9)
    first = comps["AU"][~second]
    np.testing.assert_allclose(first, [(2 * 1 + 1 * 2) / 3, 2.0])


def test_intervals_are_required_for_samples():
    with pytest.raises(cs.InvalidInput):
        cs.Drillholes(collar, survey).samples()


def test_merge_then_composite_by_domain():
    geology = {
        "HOLEID": np.array([1.0, 1.0, 2.0]),
        "FROM": np.array([0.0, 3.0, 0.0]),
        "TO": np.array([3.0, 6.0, 8.0]),
        "LITH": np.array([1.0, 2.0, 1.0]),
    }
    merged = cs.merge_intervals(intervals, geology)
    assert merged.column_names == ["HOLEID", "FROM", "TO", "AU", "LITH"]
    np.testing.assert_array_equal(merged["TO"][:4], [2, 3, 5, 6])
    comps = cs.Drillholes(collar, survey, merged).composite(2.0, ["AU"], domain="LITH")
    first = np.array(comps.attributes["hole"]) == "1.0"
    assert set(np.array(comps.attributes["LITH"])[first]) == {"1.0", "2.0"}
    crosses = (comps["from"][first] < 3.0 - 1e-9) & (comps["to"][first] > 3.0 + 1e-9)
    assert not crosses.any()


METHODS = ["minimum_curvature", "tangential", "balanced_tangential"]


def test_desurvey_methods_agree_on_straight_holes_and_differ_on_curves():
    straight = [cs.Drillholes(collar, survey, intervals, method=m).samples().coords for m in METHODS]
    for coords in straight[1:]:
        np.testing.assert_allclose(coords, straight[0], atol=1e-9)
    bend = {"HOLEID": [1.0, 1.0], "DEPTH": [0.0, 50.0], "AZIMUTH": [90.0, 90.0], "DIP": [90.0, 30.0]}
    east = [cs.Drillholes(collar, bend, method=m).at(["1.0"], [50.0])[0, 0] for m in METHODS]
    assert east[1] < east[2] < east[0]
    with pytest.raises(cs.InvalidInput):
        cs.Drillholes(collar, survey, method="spline")


full = dict(intervals, AU=np.array([1.0, 2.0, 4.0, 3.0, 5.0]))
total = 1 * 2 + 2 * 3 + 4 * 1 + 3 * 4 + 5 * 4


def metal(comps):
    return np.dot(comps["length"], comps["AU"])


def test_compositing_modes_conserve_metal():
    dh = cs.Drillholes(collar, survey, full)
    runs = dh.composite(None, ["AU"])
    assert len(runs) == 2 and metal(runs) == pytest.approx(total)
    benches = {"HOLEID": [1.0, 2.0, 2.0], "FROM": [0.0, 0.0, 5.0], "TO": [6.0, 5.0, 8.0]}
    to_benches = dh.composite(None, ["AU"], intervals=benches)
    np.testing.assert_allclose(to_benches["to"], [6, 5, 8])
    assert metal(to_benches) == pytest.approx(total)
    merged = dh.composite(3.0, ["AU"], residual="merge", min_fraction=0.9)
    np.testing.assert_allclose(merged["length"], [3, 3, 3, 5])
    assert metal(merged) == pytest.approx(total)
    dropped = dh.composite(3.0, ["AU"], residual="drop", min_fraction=0.9)
    np.testing.assert_allclose(dropped["length"], [3, 3, 3, 3])
    assert metal(dropped) == pytest.approx(total - 5 * 2)
    with pytest.raises(cs.InvalidInput):
        dh.composite(3.0, ["AU"], residual="split")


def test_categories_take_the_length_weighted_majority():
    rock = dict(full, ROCK=np.array(["ox", "fresh", "fresh", "ox", "ox"]))
    comps = cs.Drillholes(collar, survey, rock).composite(None, ["AU"], categories=["ROCK"])
    assert list(np.array(comps.attributes["ROCK"])) == ["fresh", "ox"]


def test_sampled_length_balances_metal_in_every_mode():
    dh = cs.Drillholes(collar, survey, intervals)
    total = 1 * 2 + 2 * 3 + 3 * 4 + 5 * 4
    benches = {"HOLEID": [1.0, 1.0, 2.0], "FROM": [0.0, 4.0, 0.0], "TO": [4.0, 6.0, 8.0]}
    for kwargs in [
        {"length": 4.0},
        {"length": None},
        {"length": None, "intervals": benches},
        {"length": 4.0, "residual": "merge"},
    ]:
        comps = dh.composite(grades=["AU"], **kwargs)
        assert np.nansum(comps["AU"] * comps["AU_length"]) == pytest.approx(total)
        assert np.any(comps["AU_length"] < comps["length"])


@pytest.mark.slow
def test_sampled_length_balances_metal_on_the_dataset():
    t = cs.datasets.drillhole_tables()
    merged = cs.merge_intervals(t["assay"], t["geology"])
    comps = cs.Drillholes(t["collar"], t["survey"], merged).composite(
        2.0, ["ZN"], domain="LITH", residual="merge"
    )
    metal = np.nansum(merged["ZN"] * (merged["TO"] - merged["FROM"]))
    assert np.nansum(comps["ZN"] * comps["ZN_length"]) == pytest.approx(metal, rel=1e-9)
