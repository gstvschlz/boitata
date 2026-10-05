import boitata as bt
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
    dh = bt.Drillholes(collar, survey, intervals)
    assert dh.holes == ["1.0", "2.0"] and len(dh) == 2
    samples = dh.samples()
    np.testing.assert_allclose(samples.coords[0], [0, 0, 499])
    d = 6 / np.sqrt(2)
    np.testing.assert_allclose(samples.coords[4], [100 + d, 0, 500 - d], atol=1e-9)


def test_compositing_conserves_metal():
    comps = bt.Drillholes(collar, survey, intervals).composite(3.0, ["AU"])
    second = np.array(comps.attributes["HOLE_ID"]) == "2.0"
    assert np.dot(comps["length"][second], comps["AU"][second]) == pytest.approx(4 * 3 + 4 * 5)
    assert np.all(comps["length"] <= 3.0 + 1e-9)
    first = comps["AU"][~second]
    np.testing.assert_allclose(first, [(2 * 1 + 1 * 2) / 3, 2.0])


def test_outputs_keep_the_hole_column_name():
    def rename(t):
        return {("BHID" if k == "HOLE_ID" else k): v for k, v in t.items()}

    dh = bt.Drillholes(rename(collar), rename(survey), rename(intervals), hole="BHID")
    assert dh.paths().column_names[0] == "BHID"
    assert "BHID" in dh.samples().attributes.column_names
    assert dh.composite(3.0, ["AU"]).attributes.column_names[:4] == ["BHID", "from", "to", "length"]


def test_intervals_are_required_for_samples():
    with pytest.raises(bt.InvalidInput):
        bt.Drillholes(collar, survey).samples()


def test_merge_then_composite_by_domain():
    geology = {
        "HOLE_ID": np.array([1.0, 1.0, 2.0]),
        "FROM": np.array([0.0, 3.0, 0.0]),
        "TO": np.array([3.0, 6.0, 8.0]),
        "LITH": np.array([1.0, 2.0, 1.0]),
    }
    merged = bt.merge_intervals(intervals, geology)
    assert merged.column_names == ["HOLE_ID", "FROM", "TO", "AU", "LITH"]
    np.testing.assert_array_equal(merged["TO"][:4], [2, 3, 5, 6])
    comps = bt.Drillholes(collar, survey, merged).composite(2.0, ["AU"], domain="LITH")
    first = np.array(comps.attributes["HOLE_ID"]) == "1.0"
    assert set(np.array(comps.attributes["LITH"])[first]) == {"1.0", "2.0"}
    crosses = (comps["from"][first] < 3.0 - 1e-9) & (comps["to"][first] > 3.0 + 1e-9)
    assert not crosses.any()


def test_merge_rejects_overlapping_intervals():
    geology = {"HOLE_ID": np.array([2.0]), "FROM": np.array([0.0]), "TO": np.array([8.0])}
    overlapping = {**intervals, "FROM": np.array([0.0, 2.0, 5.0, 0.0, 3.0])}
    with pytest.raises(ValueError, match=r"1 holes.*left 2.0: 0-4 and 3-8"):
        bt.merge_intervals(overlapping, geology)


METHODS = ["minimum_curvature", "tangential", "balanced_tangential"]


def test_desurvey_methods_agree_on_straight_holes_and_differ_on_curves():
    straight = [bt.Drillholes(collar, survey, intervals, method=m).samples().coords for m in METHODS]
    for coords in straight[1:]:
        np.testing.assert_allclose(coords, straight[0], atol=1e-9)
    bend = {"HOLE_ID": [1.0, 1.0], "DEPTH": [0.0, 50.0], "AZIMUTH": [90.0, 90.0], "DIP": [90.0, 30.0]}
    east = [bt.Drillholes(collar, bend, method=m).at(["1.0"], [50.0])[0, 0] for m in METHODS]
    assert east[1] < east[2] < east[0]
    with pytest.raises(bt.InvalidInput):
        bt.Drillholes(collar, survey, method="spline")


full = dict(intervals, AU=np.array([1.0, 2.0, 4.0, 3.0, 5.0]))
total = 1 * 2 + 2 * 3 + 4 * 1 + 3 * 4 + 5 * 4


def metal(comps):
    return np.dot(comps["length"], comps["AU"])


def test_compositing_modes_conserve_metal():
    dh = bt.Drillholes(collar, survey, full)
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
    with pytest.raises(bt.InvalidInput):
        dh.composite(3.0, ["AU"], residual="split")


def test_categories_take_the_length_weighted_majority():
    rock = dict(full, ROCK=np.array(["ox", "fresh", "fresh", "ox", "ox"]))
    comps = bt.Drillholes(collar, survey, rock).composite(None, ["AU"], categories=["ROCK"])
    assert list(np.array(comps.attributes["ROCK"])) == ["fresh", "ox"]


def test_sampled_length_balances_metal_in_every_mode():
    dh = bt.Drillholes(collar, survey, intervals)
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


def planted_overlaps(holes=30, rows=40, seed=7):
    rng = np.random.default_rng(seed)
    assay, geology, depth = [], [], []
    for h in range(holes):
        end = np.cumsum(rng.uniform(0.5, 3.0, rows))
        start = np.r_[0.0, end[:-1]]
        shift = rng.random(rows) < 0.1
        shift[0] = False
        start[shift] = np.maximum(start[shift] - rng.uniform(0.1, 1.0, shift.sum()), 0.0)
        zn = np.where(rng.random(rows) < 0.1, np.nan, rng.lognormal(0.0, 1.0, rows))
        assay.append((np.full(rows, f"H{h:02}"), start, end, zn))
        cuts = np.r_[0.0, np.sort(rng.uniform(0, end[-1], 3)), end[-1]]
        geology.append((np.full(4, f"H{h:02}"), cuts[:-1], cuts[1:], np.array(["A", "B", "A", "C"])))
        depth.append(end[-1])
    order = rng.permutation(holes * rows)
    a = [np.concatenate(c)[order] for c in zip(*assay)]
    g = [np.concatenate(c) for c in zip(*geology)]
    ids = np.array([f"H{h:02}" for h in range(holes)], dtype=object)
    return {
        "collar": bt.Table(
            {
                "HOLE_ID": ids,
                "X": np.arange(holes) * 50.0,
                "Y": np.zeros(holes),
                "Z": np.full(holes, 100.0),
                "DEPTH": np.array(depth),
            }
        ),
        "survey": bt.Table(
            {
                "HOLE_ID": ids,
                "DEPTH": np.zeros(holes),
                "AZIMUTH": np.zeros(holes),
                "DIP": np.full(holes, 90.0),
            }
        ),
        "assay": bt.Table({"HOLE_ID": a[0].astype(object), "FROM": a[1], "TO": a[2], "ZN": a[3]}),
        "geology": bt.Table(
            {"HOLE_ID": g[0].astype(object), "FROM": g[1], "TO": g[2], "LITH": g[3].astype(object)}
        ),
    }


def test_sampled_length_balances_metal_after_dropping_overlaps():
    t = planted_overlaps()
    flags, _, _ = bt.check_drillholes(t["collar"], intervals={"assay": t["assay"]})
    assert flags["assay"]["overlap"].sum() > 0
    assay = t["assay"].filter(~flags["assay"]["overlap"])
    merged = bt.merge_intervals(assay, t["geology"])
    comps = bt.Drillholes(t["collar"], t["survey"], merged).composite(
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
    flags, summary, _ = bt.check_drillholes(
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
    flags, _, _ = bt.check_drillholes(checked_collar, checked_survey, {"assay": checked_assay}, nodata=[])
    assert not any(t["sentinel"].any() for t in flags.values())
    rows = dict(zip(zip(summary["table"], summary["check"]), summary["rows"]))
    assert rows[("assay", "overlap")] == 1 and rows[("collar", "out_of_range")] == 0


def test_clean_tables_give_no_flags():
    flags, summary, _ = bt.check_drillholes(collar, survey, intervals, inclination=None)
    assert list(flags) == ["collar", "survey", "intervals"]
    assert not _flagged(flags)
    assert (summary["rows"] == 0).all()


def test_inclination_is_checked_as_dip():
    upward = {**survey, "INC": np.array([0.0, 170.0])}
    flags, _, _ = bt.check_drillholes(collar, upward, inclination="INC")
    assert not flags["survey"]["out_of_range"].any()
    flags, _, _ = bt.check_drillholes(collar, {**upward, "INC": np.array([0.0, 190.0])}, inclination="INC")
    assert list(flags["survey"]["out_of_range"]) == [False, True]


def test_fix_applies_one_rule_per_check():
    tables = {"collar": checked_collar, "survey": checked_survey, "assay": checked_assay}
    flags, _, _ = bt.check_drillholes(
        checked_collar, checked_survey, {"assay": checked_assay}, max_depth="LENGTH"
    )
    clean, log = bt.fix_drillholes(flags, tables)
    assert list(clean["collar"]["HOLE_ID"]) == ["A", "B"]
    assert list(clean["survey"]["DEPTH"]) == [0.0, 0.0]
    assert list(clean["assay"]["FROM"]) == [0.0, 1.0, 6.0, 0.0]
    assert np.isnan(clean["assay"]["AU"][-1])
    actions = dict(zip(zip(log["table"], log["check"]), log["action"]))
    assert actions[("assay", "overlap")] == "keep_first" and actions[("assay", "sentinel")] == "null"
    assert ("assay", "past_depth") not in actions and ("assay", "gap") not in actions
    kept, _ = bt.fix_drillholes(flags, tables, deviation="keep", sentinels="drop", past_depth="drop")
    assert list(kept["survey"]["DEPTH"]) == [0.0, 10.0, 0.0]
    assert list(kept["assay"]["FROM"]) == [0.0, 1.0]
    with pytest.raises(ValueError, match="overlaps must be one of keep_first, keep"):
        bt.fix_drillholes(flags, tables, overlaps="drop")


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
    flags, _, details = bt.check_drillholes(entry_collar, entry_survey, {"assay": entry_assay})
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
    _, _, lith = bt.check_drillholes(entry_collar, intervals={"assay": entry_assay}, grades=["LITH"])
    assert list(lith["value"])[:4] == ["MS", "MS", "SMS", "MS"] and list(lith["column"])[:4] == ["LITH"] * 4
    with pytest.raises(ValueError, match="column CU not found in any table"):
        bt.check_drillholes(entry_collar, grades=["CU"])


def test_entry_errors_are_fixed_only_as_asked():
    tables = {"collar": entry_collar, "survey": entry_survey, "assay": entry_assay}
    flags, _, _ = bt.check_drillholes(entry_collar, entry_survey, {"assay": entry_assay})
    fixed, log = bt.fix_drillholes(flags, tables)
    assert list(fixed["survey"]["DIP"]) == [60.0, -60.0, -58.0, 70.0]
    assert list(fixed["survey"]["HOLE_ID"]) == ["A", "B", "B", "C "]
    assert list(fixed["assay"]["HOLE_ID"]) == ["A", "A", "B", "A"]
    np.testing.assert_array_equal(fixed["assay"]["AU"], [1.5, np.nan, np.nan, np.nan])
    actions = dict(zip(zip(log["table"], log["check"]), log["action"]))
    assert actions[("assay", "text_values")] == "null" and ("survey", "dip_sign") not in actions
    assert actions[("assay", "sentinel")] == "null"
    fixed, _ = bt.fix_drillholes(flags, tables, dip_sign="negate", text_values="half", id_mismatch="keep")
    assert list(fixed["survey"]["DIP"]) == [60.0, 60.0, 58.0, 70.0]
    assert list(fixed["assay"]["HOLE_ID"]) == ["A", "a", "B ", "A"]
    np.testing.assert_array_equal(fixed["assay"]["AU"], [1.5, np.nan, 0.01, np.nan])
    fixed, _ = bt.fix_drillholes(flags, tables, text_values="limit")
    np.testing.assert_array_equal(fixed["assay"]["AU"], [1.5, np.nan, 0.02, np.nan])
    fixed, _ = bt.fix_drillholes(flags, tables, text_values="keep")
    assert list(fixed["assay"]["AU"]) == ["1.5", "NS", "<0.02", None]
    inclined = {**entry_survey, "INC": 90.0 - entry_survey["DIP"]}
    flags, _, _ = bt.check_drillholes(entry_collar, inclined, inclination="INC")
    fixed, _ = bt.fix_drillholes(flags, {"survey": inclined}, dip_sign="negate")
    np.testing.assert_allclose(fixed["survey"]["INC"], [30.0, 30.0, 32.0, 20.0])


@pytest.mark.slow
def test_planted_entry_errors_are_found_on_the_raw_dataset():
    data = bt.datasets.stacked_sulphide_lenses(raw=True)
    intervals = {"assays": data["assays"], "lithology": data["lithology"]}
    flags, _, details = bt.check_drillholes(data["collars"], data["surveys"], intervals)
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
    fixed, _ = bt.fix_drillholes(flags, tables, dip_sign="negate")
    flags, _, details = bt.check_drillholes(
        fixed["collar"], fixed["survey"], {"assays": fixed["assays"], "lithology": fixed["lithology"]}
    )
    assert details.num_rows == 0
    assert not any(flags[t]["no_collar"].any() for t in intervals)


def test_filter_keeps_masked_rows():
    t = bt.Table(intervals)
    assert list(t.filter(t["FROM"] > 1)["TO"]) == [5.0, 6.0, 8.0]
    with pytest.raises(ValueError, match="mask has 1 values"):
        t.filter(np.array([True]))


def test_overlap_flags_keep_the_first_interval():
    t = planted_overlaps()
    assay = t["assay"]
    hole, start, end = np.array(assay["HOLE_ID"]), assay["FROM"], assay["TO"]
    keep, reach = np.ones(assay.num_rows, bool), {}
    for i in np.lexsort((start, hole)):
        keep[i] = start[i] >= reach.get(hole[i], -np.inf)
        if keep[i]:
            reach[hole[i]] = end[i]
    flags, _, _ = bt.check_drillholes(t["collar"], t["survey"], {"assay": assay}, max_depth="DEPTH")
    assert (~keep).sum() > 0
    assert (flags["assay"]["overlap"] == ~keep).all()


def one_hole(grades, lith=None):
    n = len(grades)
    table = {
        "HOLE_ID": ["A"] * n,
        "FROM": np.arange(n, dtype=float),
        "TO": np.arange(1.0, n + 1),
        "AU": grades,
    }
    if lith is not None:
        table["LITH"] = lith
    c = {"HOLE_ID": ["A"], "X": [0.0], "Y": [0.0], "Z": [0.0]}
    s = {"HOLE_ID": ["A"], "DEPTH": [0.0], "AZIMUTH": [0.0], "DIP": [90.0]}
    return bt.Drillholes(c, s, table)


def test_runs_balance_metal_and_honor_rules():
    rng = np.random.default_rng(11)
    grades = rng.lognormal(-0.5, 1.0, 400)
    dh = one_hole(grades)
    for rules in [{}, {"min_length": 4.0}, {"min_length": 4.0, "max_dilution": 3.0, "edge": 0.5}]:
        runs = dh.runs("AU", cutoff=1.0, **rules)
        assert np.dot(runs["AU"], runs["length"]) == pytest.approx(grades.sum())
        assert np.all(np.asarray(runs["to"]) - runs["from"] >= rules.get("min_length", 0.0) - 1e-9)
        assert np.all(np.diff(np.asarray(runs["ore"], dtype=int)) != 0)
        again = dh.runs("AU", cutoff=1.0, **rules)
        np.testing.assert_array_equal(runs["AU"], again["AU"])


def test_internal_dilution_by_hand():
    runs = one_hole([2.0, 2.0, 0.1, 3.0, 0.1]).runs("AU", cutoff=1.0, max_dilution=1.0)
    assert runs.column_names == ["HOLE_ID", "from", "to", "length", "AU", "ore"]
    np.testing.assert_allclose(runs["to"], [4.0, 5.0])
    np.testing.assert_allclose(runs["AU"], [7.1 / 4, 0.1])
    assert list(runs["ore"]) == [True, False]


box = bt.Mesh(
    np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float) * 10,
    [
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
        [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
    ],
)  # fmt: skip


def test_mesh_intervals_bracket_the_crossing():
    # Hole straight down through the box, entering its top face at depth 5 and leaving the bottom at depth 15.
    c = {"HOLE_ID": ["A"], "X": [5.0], "Y": [5.0], "Z": [15.0]}
    s = {"HOLE_ID": ["A", "A"], "DEPTH": [0.0, 25.0], "AZIMUTH": [0.0, 0.0], "DIP": [90.0, 90.0]}
    dh = bt.Drillholes(c, s)
    crossings = dh.mesh_intervals(box, step=0.5, tolerance=0.01)
    assert crossings.column_names == ["HOLE_ID", "FROM", "TO", "INSIDE"]
    assert list(crossings["INSIDE"]) == [False, True, False]
    np.testing.assert_allclose(crossings["FROM"], [0.0, 5.0, 15.0], atol=0.01)
    np.testing.assert_allclose(crossings["TO"], [5.0, 15.0, 25.0], atol=0.01)

    open_mesh = bt.Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 1, 2]])
    with pytest.raises(bt.InvalidInput, match="not closed"):
        dh.mesh_intervals(open_mesh)


def test_mesh_intervals_split_assays_with_merge_intervals():
    assays = {
        "HOLE_ID": ["A", "A", "A"],
        "FROM": [0.0, 8.0, 18.0],
        "TO": [8.0, 18.0, 25.0],
        "AU": [1.0, 2.0, 3.0],
    }
    c = {"HOLE_ID": ["A"], "X": [5.0], "Y": [5.0], "Z": [15.0]}
    s = {"HOLE_ID": ["A"], "DEPTH": [0.0], "AZIMUTH": [0.0], "DIP": [90.0]}
    dh = bt.Drillholes(c, s, assays)
    crossings = dh.mesh_intervals(box, step=0.5, tolerance=0.01)
    split = bt.merge_intervals(assays, crossings)
    np.testing.assert_allclose(sorted(split["TO"]), [5.0, 8.0, 15.0, 18.0, 25.0], atol=0.01)
    assert set(split.column_names) == {"HOLE_ID", "FROM", "TO", "AU", "INSIDE"}


def test_category_runs_and_bad_input():
    dh = one_hole([1.0] * 5, lith=["QV", "QV", None, "BX", "QV"])
    runs = dh.runs(None, category="LITH", ore=["QV"])
    assert list(runs["ore"]) == [True, False, True] and "AU" not in runs.column_names
    with pytest.raises(bt.InvalidInput):
        dh.runs("AU")
    with pytest.raises(bt.InvalidInput):
        dh.runs("AU", cutoff=1.0, min_length=-1.0)


def test_snap_to_surface_moves_collars_and_their_holes():
    collar = {
        "HOLE_ID": ["a", "b", "c"],
        "X": [5.0, 15.0, 500.0],
        "Y": [5.0, 12.0, 500.0],
        "Z": [90.0, 120.0, 0.0],
    }
    survey = {
        "HOLE_ID": ["a", "b", "c"],
        "DEPTH": [0.0, 0.0, 0.0],
        "AZIMUTH": [0.0, 45.0, 0.0],
        "DIP": [90.0, 60.0, 90.0],
    }
    holes = bt.Drillholes(collar, survey)
    ground = bt.BlockModel((0, 0), (1, 1), (30, 30))
    ground = ground.with_column("elevation", 100 + 0.5 * ground.centroids[:, 0])
    with pytest.warns(UserWarning, match="1 holes lie off the surface.*c"):
        moved, report = bt.snap_to_surface(holes, ground, column="elevation")
    assert report["hole"].tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(report["z_after"][:2], [102.5, 107.5])
    np.testing.assert_allclose(report["shift"][:2], [12.5, -12.5])
    assert np.isnan(report["shift"][2]) and report["z_after"][2] == 0.0
    before, after = holes.at(["b", "b"], [0.0, 40.0]), moved.at(["b", "b"], [0.0, 40.0])
    np.testing.assert_allclose(after - before, [[0, 0, -12.5]] * 2)
    mesh = bt.grid_surface(ground, "elevation")
    with pytest.warns(UserWarning):
        assert bt.snap_to_surface(holes, mesh)[1]["shift"][0] == pytest.approx(12.5)
    with pytest.raises(bt.InvalidInput, match="needs column"):
        bt.snap_to_surface(holes, ground)


def test_planned_drillholes_cover_the_model_and_composite():
    model = bt.BlockModel((0, 0, -50), (10, 10, 10), (10, 10, 4))
    ground = bt.BlockModel((-50, -50), (10, 10), (30, 30))
    ground = ground.with_column("Z", 5 + 0.1 * ground.centroids[:, 0])
    plan = bt.planned_drillholes(model, 25.0, topography=bt.grid_surface(ground, "Z"))
    assert len(plan) == 25 and plan.holes[0] == "P0001"
    collars = plan.at(plan.holes, np.zeros(len(plan)))
    np.testing.assert_allclose(collars[:, 2], 5 + 0.1 * collars[:, 0], atol=1e-9)
    assert sorted(set(np.round(collars[:, 0], 9))) == [5, 30, 55, 80, 105]
    lengths = plan.samples()["TO"]
    np.testing.assert_allclose(plan.at(plan.holes, lengths)[:, 2], -50, atol=1e-9)
    composites = plan.composite(5.0, [])
    assert composites["length"].max() == pytest.approx(5.0)
    assert composites["length"].sum() == pytest.approx(lengths.sum())


def test_planned_drillholes_rotated_inclined_and_bad_input():
    points = np.array([[0.0, 0.0, -10.0], [100.0, 0.0, -30.0]])
    plan = bt.planned_drillholes(points, (30.0, 10.0), rotation=90.0, azimuth=90.0, dip=60.0)
    collars = plan.at(plan.holes, np.zeros(len(plan)))
    np.testing.assert_allclose(collars, [[0, 0, -10], [90, 0, -10]], atol=1e-9)
    np.testing.assert_allclose(plan.at(plan.holes, plan.samples()["TO"])[:, 2], -30, atol=1e-9)
    mesh = bt.Mesh([[500, 500, 0], [600, 500, 0], [500, 600, 0]], [[0, 1, 2]])
    with pytest.warns(UserWarning, match="2 collars lie off"):
        bt.planned_drillholes(points, 30.0, topography=mesh)
    with pytest.raises(bt.InvalidInput):
        bt.planned_drillholes(points, -1.0)
    with pytest.raises(bt.InvalidInput):
        bt.planned_drillholes(points, 10.0, dip=0.0)
    with pytest.raises(bt.InvalidInput, match="no hole reaches"):
        bt.planned_drillholes(points[:, :2], 10.0)
