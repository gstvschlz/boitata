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
