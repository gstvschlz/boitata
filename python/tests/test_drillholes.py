import ceres as cs
import numpy as np
import pytest

collar = {
    "HOLE_ID": np.array([1.0, 2.0]),
    "X": np.array([0.0, 100.0]),
    "Y": np.zeros(2),
    "Z": np.full(2, 500.0),
}
survey = {
    "HOLE_ID": np.array([1.0, 2.0]),
    "DEPTH": np.zeros(2),
    "AZIMUTH": np.array([0.0, 90.0]),
    "DIP": np.array([90.0, 45.0]),
}
intervals = {
    "HOLE_ID": np.array([1.0, 1.0, 1.0, 2.0, 2.0]),
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
    second = np.array(comps.attributes["HOLE_ID"]) == "2.0"
    assert np.dot(comps["length"][second], comps["AU"][second]) == pytest.approx(4 * 3 + 4 * 5)
    assert np.all(comps["length"] <= 3.0 + 1e-9)
    first = comps["AU"][~second]
    np.testing.assert_allclose(first, [(2 * 1 + 1 * 2) / 3, 2.0])


def test_outputs_keep_the_hole_column_name():
    def rename(t):
        return {("BHID" if k == "HOLE_ID" else k): v for k, v in t.items()}

    dh = cs.Drillholes(rename(collar), rename(survey), rename(intervals), hole="BHID")
    assert dh.paths().column_names[0] == "BHID"
    assert "BHID" in dh.samples().attributes.column_names
    assert dh.composite(3.0, ["AU"]).attributes.column_names[:4] == ["BHID", "from", "to", "length"]


def test_intervals_are_required_for_samples():
    with pytest.raises(cs.InvalidInput):
        cs.Drillholes(collar, survey).samples()


def test_merge_then_composite_by_domain():
    geology = {
        "HOLE_ID": np.array([1.0, 1.0, 2.0]),
        "FROM": np.array([0.0, 3.0, 0.0]),
        "TO": np.array([3.0, 6.0, 8.0]),
        "LITH": np.array([1.0, 2.0, 1.0]),
    }
    merged = cs.merge_intervals(intervals, geology)
    assert merged.column_names == ["HOLE_ID", "FROM", "TO", "AU", "LITH"]
    np.testing.assert_array_equal(merged["TO"][:4], [2, 3, 5, 6])
    comps = cs.Drillholes(collar, survey, merged).composite(2.0, ["AU"], domain="LITH")
    first = np.array(comps.attributes["HOLE_ID"]) == "1.0"
    assert set(np.array(comps.attributes["LITH"])[first]) == {"1.0", "2.0"}
    crosses = (comps["from"][first] < 3.0 - 1e-9) & (comps["to"][first] > 3.0 + 1e-9)
    assert not crosses.any()


def test_merge_rejects_overlapping_intervals():
    geology = {"HOLE_ID": np.array([2.0]), "FROM": np.array([0.0]), "TO": np.array([8.0])}
    overlapping = {**intervals, "FROM": np.array([0.0, 2.0, 5.0, 0.0, 3.0])}
    with pytest.raises(ValueError, match=r"1 holes.*left 2.0: 0-4 and 3-8"):
        cs.merge_intervals(overlapping, geology)


METHODS = ["minimum_curvature", "tangential", "balanced_tangential"]


def test_desurvey_methods_agree_on_straight_holes_and_differ_on_curves():
    straight = [cs.Drillholes(collar, survey, intervals, method=m).samples().coords for m in METHODS]
    for coords in straight[1:]:
        np.testing.assert_allclose(coords, straight[0], atol=1e-9)
    bend = {"HOLE_ID": [1.0, 1.0], "DEPTH": [0.0, 50.0], "AZIMUTH": [90.0, 90.0], "DIP": [90.0, 30.0]}
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
    benches = {"HOLE_ID": [1.0, 2.0, 2.0], "FROM": [0.0, 0.0, 5.0], "TO": [6.0, 5.0, 8.0]}
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
    benches = {"HOLE_ID": [1.0, 1.0, 2.0], "FROM": [0.0, 4.0, 0.0], "TO": [4.0, 6.0, 8.0]}
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
    flags, _, _ = cs.check_drillholes(t["collar"], intervals={"assay": t["assay"]}, hole="HOLEID")
    assay = t["assay"].filter(~flags["assay"]["overlap"])
    merged = cs.merge_intervals(assay, t["geology"], hole="HOLEID")
    comps = cs.Drillholes(t["collar"], t["survey"], merged, hole="HOLEID").composite(
        2.0, ["ZN"], domain="LITH", residual="merge"
    )
    metal = np.nansum(merged["ZN"] * (merged["TO"] - merged["FROM"]))
    assert np.nansum(comps["ZN"] * comps["ZN_length"]) == pytest.approx(metal, rel=1e-9)


def _flagged(flags):
    return {
        (name, check, i)
        for name, t in flags.items()
        for check in t.column_names
        for i in np.flatnonzero(t[check])
    }


checked_collar = {
    "HOLE_ID": np.array(["A", "B", "B", None, "D"], dtype=object),
    "X": np.array([0.0, 1.0, 1.0, 2.0, -999.0]),
    "Y": np.zeros(5),
    "Z": np.array([9.0, 9.0, 9.0, 9.0, np.nan]),
    "LENGTH": np.array([20.0, 30.0, 30.0, 10.0, 10.0]),
}
checked_survey = {
    "HOLE_ID": np.array(["A", "A", "A", "A", "A", "B", "E"]),
    "DEPTH": np.array([0.0, 10.0, 10.0, 15.0, 25.0, 0.0, 0.0]),
    "AZIMUTH": np.array([90.0, 90.0, 90.0, 400.0, 90.0, 0.0, 0.0]),
    "DIP": np.array([60.0, -60.0, 60.0, 60.0, 60.0, 90.0, 90.0]),
}
checked_assay = {
    "HOLE_ID": np.array(["A", "A", "A", "A", "A", "B", "E"]),
    "FROM": np.array([0.0, 1.0, 0.5, 4.0, 6.0, 0.0, 0.0]),
    "TO": np.array([1.0, 3.0, 2.0, 4.0, 22.0, 5.0, 1.0]),
    "AU": np.array([1.0, 2.0, 3.0, 4.0, 5.0, -99.0, 1.0]),
}


def test_each_defect_is_flagged_once():
    flags, summary, _ = cs.check_drillholes(
        checked_collar, checked_survey, {"assay": checked_assay}, max_depth="LENGTH"
    )
    assert _flagged(flags) == {
        ("collar", "duplicate", 2),
        ("collar", "missing", 3),
        ("collar", "missing", 4),
        ("collar", "sentinel", 4),
        ("collar", "no_survey", 4),
        ("collar", "no_assay", 4),
        ("survey", "deviation", 1),
        ("survey", "deviation", 4),
        ("survey", "duplicate", 2),
        ("survey", "out_of_range", 3),
        ("survey", "past_depth", 4),
        ("survey", "no_collar", 6),
        ("assay", "overlap", 2),
        ("assay", "inverted", 3),
        ("assay", "gap", 4),
        ("assay", "past_depth", 4),
        ("assay", "sentinel", 5),
        ("assay", "no_collar", 6),
    }
    assert summary.num_rows == sum(len(t.column_names) for t in flags.values())
    flags, _, _ = cs.check_drillholes(checked_collar, checked_survey, {"assay": checked_assay}, nodata=[])
    assert not any(t["sentinel"].any() for t in flags.values())
    rows = dict(zip(zip(summary["table"], summary["check"]), summary["rows"]))
    assert rows[("assay", "overlap")] == 1 and rows[("collar", "out_of_range")] == 0


def test_clean_tables_give_no_flags():
    flags, summary, _ = cs.check_drillholes(collar, survey, intervals, inclination=None)
    assert list(flags) == ["collar", "survey", "intervals"]
    assert not _flagged(flags)
    assert (summary["rows"] == 0).all()


def test_inclination_is_checked_as_dip():
    upward = {**survey, "INC": np.array([0.0, 170.0])}
    flags, _, _ = cs.check_drillholes(collar, upward, inclination="INC")
    assert not flags["survey"]["out_of_range"].any()
    flags, _, _ = cs.check_drillholes(collar, {**upward, "INC": np.array([0.0, 190.0])}, inclination="INC")
    assert list(flags["survey"]["out_of_range"]) == [False, True]


def test_fix_applies_one_rule_per_check():
    tables = {"collar": checked_collar, "survey": checked_survey, "assay": checked_assay}
    flags, _, _ = cs.check_drillholes(
        checked_collar, checked_survey, {"assay": checked_assay}, max_depth="LENGTH"
    )
    clean, log = cs.fix_drillholes(flags, tables)
    assert list(clean["collar"]["HOLE_ID"]) == ["A", "B"]
    assert list(clean["survey"]["DEPTH"]) == [0.0, 0.0]
    assert list(clean["assay"]["FROM"]) == [0.0, 1.0, 6.0, 0.0]
    assert np.isnan(clean["assay"]["AU"][-1])
    actions = dict(zip(zip(log["table"], log["check"]), log["action"]))
    assert actions[("assay", "overlap")] == "keep_first" and actions[("assay", "sentinel")] == "null"
    assert ("assay", "past_depth") not in actions and ("assay", "gap") not in actions
    kept, _ = cs.fix_drillholes(flags, tables, deviation="keep", sentinels="drop", past_depth="drop")
    assert list(kept["survey"]["DEPTH"]) == [0.0, 10.0, 0.0]
    assert list(kept["assay"]["FROM"]) == [0.0, 1.0]
    with pytest.raises(ValueError, match="overlaps must be one of keep_first, keep"):
        cs.fix_drillholes(flags, tables, overlaps="drop")


entry_collar = {
    "HOLE_ID": np.array(["A", "B", "C "]),
    "X": np.zeros(3),
    "Y": np.zeros(3),
    "Z": np.zeros(3),
}
entry_survey = {
    "HOLE_ID": np.array(["A", "B", "B", "C"]),
    "DEPTH": np.array([0.0, 0.0, 10.0, 0.0]),
    "AZIMUTH": np.zeros(4),
    "DIP": np.array([60.0, -60.0, -58.0, 70.0]),
}
entry_assay = {
    "HOLE_ID": np.array(["A", "a", "B ", "D", "A"]),
    "FROM": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    "TO": np.array([1.0, 1.0, 1.0, 1.0, 2.0]),
    "AU": np.array(["1.5", "NS", "<0.02", "2.0", "-999"], dtype=object),
    "LITH": np.array(["MS", "MS", "SMS", "1", "MS"], dtype=object),
}


def test_entry_errors_are_flagged_with_their_values():
    flags, _, details = cs.check_drillholes(entry_collar, entry_survey, {"assay": entry_assay})
    assert _flagged(flags) == {
        ("collar", "no_assay", 2),
        ("survey", "dip_sign", 1),
        ("survey", "dip_sign", 2),
        ("survey", "id_mismatch", 3),
        ("assay", "id_mismatch", 1),
        ("assay", "id_mismatch", 2),
        ("assay", "no_collar", 3),
        ("assay", "text_values", 1),
        ("assay", "text_values", 2),
        ("assay", "sentinel", 4),
    }
    rows = zip(*(details[c] for c in ("table", "row", "check", "column", "value", "suggestion")))
    assert [(t, int(r), *rest) for t, r, *rest in rows] == [
        ("survey", 1, "dip_sign", "DIP", "-60.0", "60.0"),
        ("survey", 2, "dip_sign", "DIP", "-58.0", "58.0"),
        ("assay", 1, "text_values", "AU", "NS", None),
        ("assay", 2, "text_values", "AU", "<0.02", None),
        ("survey", 3, "id_mismatch", "HOLE_ID", "C", "C "),
        ("assay", 1, "id_mismatch", "HOLE_ID", "a", "A"),
        ("assay", 2, "id_mismatch", "HOLE_ID", "B ", "B"),
    ]
    _, _, lith = cs.check_drillholes(entry_collar, intervals={"assay": entry_assay}, grades=["LITH"])
    assert list(lith["value"])[:4] == ["MS", "MS", "SMS", "MS"] and list(lith["column"])[:4] == ["LITH"] * 4
    with pytest.raises(ValueError, match="column CU not found in any table"):
        cs.check_drillholes(entry_collar, grades=["CU"])


def test_entry_errors_are_fixed_only_as_asked():
    tables = {"collar": entry_collar, "survey": entry_survey, "assay": entry_assay}
    flags, _, _ = cs.check_drillholes(entry_collar, entry_survey, {"assay": entry_assay})
    fixed, log = cs.fix_drillholes(flags, tables)
    assert list(fixed["survey"]["DIP"]) == [60.0, -60.0, -58.0, 70.0]
    assert list(fixed["survey"]["HOLE_ID"]) == ["A", "B", "B", "C "]
    assert list(fixed["assay"]["HOLE_ID"]) == ["A", "A", "B", "A"]
    np.testing.assert_array_equal(fixed["assay"]["AU"], [1.5, np.nan, np.nan, np.nan])
    actions = dict(zip(zip(log["table"], log["check"]), log["action"]))
    assert actions[("assay", "text_values")] == "null" and ("survey", "dip_sign") not in actions
    assert actions[("assay", "sentinel")] == "null"
    fixed, _ = cs.fix_drillholes(flags, tables, dip_sign="negate", text_values="half", id_mismatch="keep")
    assert list(fixed["survey"]["DIP"]) == [60.0, 60.0, 58.0, 70.0]
    assert list(fixed["assay"]["HOLE_ID"]) == ["A", "a", "B ", "A"]
    np.testing.assert_array_equal(fixed["assay"]["AU"], [1.5, np.nan, 0.01, np.nan])
    fixed, _ = cs.fix_drillholes(flags, tables, text_values="limit")
    np.testing.assert_array_equal(fixed["assay"]["AU"], [1.5, np.nan, 0.02, np.nan])
    fixed, _ = cs.fix_drillholes(flags, tables, text_values="keep")
    assert list(fixed["assay"]["AU"]) == ["1.5", "NS", "<0.02", None]
    inclined = {**entry_survey, "INC": 90.0 - entry_survey["DIP"]}
    flags, _, _ = cs.check_drillholes(entry_collar, inclined, inclination="INC")
    fixed, _ = cs.fix_drillholes(flags, {"survey": inclined}, dip_sign="negate")
    np.testing.assert_allclose(fixed["survey"]["INC"], [30.0, 30.0, 32.0, 20.0])


@pytest.mark.slow
def test_planted_entry_errors_are_found_on_the_raw_dataset():
    data = cs.datasets.stacked_sulphide_lenses(raw=True)
    intervals = {"assays": data["assays"], "lithology": data["lithology"]}
    flags, _, details = cs.check_drillholes(data["collars"], data["surveys"], intervals)
    found = {}
    for check, hole, column, value, suggestion in zip(
        *(details[c] for c in ("check", "hole", "column", "value", "suggestion"))
    ):
        found.setdefault(check, set()).add(
            hole if check == "dip_sign" else (value, suggestion) if suggestion else (hole, column, value)
        )
    assert found == {
        "dip_sign": {"DD0116"},
        "id_mismatch": {("dd0062", "DD0062"), ("DD0132 ", "DD0132"), ("DD0104", "DD0104 ")},
        "text_values": {
            *(("DD0083", g, "NS") for g in ("ZN_PCT", "PB_PCT", "CU_PCT", "AG_GPT", "AU_GPT")),
            ("DD0083", "CU_PCT", "<0.01"),
            ("DD0083", "AU_GPT", "<0.01"),
        },
    }
    tables = {"collar": data["collars"], "survey": data["surveys"], **intervals}
    fixed, _ = cs.fix_drillholes(flags, tables, dip_sign="negate")
    flags, _, details = cs.check_drillholes(
        fixed["collar"], fixed["survey"], {"assays": fixed["assays"], "lithology": fixed["lithology"]}
    )
    assert details.num_rows == 0
    assert not any(flags[t]["no_collar"].any() for t in intervals)


def test_filter_keeps_masked_rows():
    t = cs.Table(intervals)
    assert list(t.filter(t["FROM"] > 1)["TO"]) == [5.0, 6.0, 8.0]
    with pytest.raises(ValueError, match="mask has 1 values"):
        t.filter(np.array([True]))


@pytest.mark.slow
def test_overlap_flags_keep_the_first_interval_on_the_dataset():
    t = cs.datasets.drillhole_tables()
    assay = t["assay"]
    hole, start, end = np.array(assay["HOLEID"]), assay["FROM"], assay["TO"]
    keep, reach = np.ones(assay.num_rows, bool), {}
    for i in np.lexsort((start, hole)):
        keep[i] = start[i] >= reach.get(hole[i], -np.inf)
        if keep[i]:
            reach[hole[i]] = end[i]
    flags, _, _ = cs.check_drillholes(
        t["collar"], t["survey"], {"assay": assay}, hole="HOLEID", max_depth="DEPTH"
    )
    assert (flags["assay"]["overlap"] == ~keep).all()
