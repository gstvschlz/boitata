import boitata as bt
import numpy as np
import pytest

xyz = np.array([[0.0, 0, 0], [0, 0, 0], [5, 0, 0], [9, 0, 0]])
raw = bt.PointSet(xyz, {"Au PPM": [1.0, -99.0, 3.0, 400.0], "RockType": ["Ox ", "OX", None, "fr"]})


def test_rename_columns_maps_then_recases_and_inverts():
    step = bt.RenameColumns(mapping={"Au PPM": "au"}, case="snake")
    out = step.fit_transform(raw)
    assert out.attributes.column_names == ["au", "rock_type"]
    np.testing.assert_array_equal(out.coords, raw.coords)
    assert step.inverse_transform(out).attributes.column_names == ["Au PPM", "RockType"]
    assert bt.RenameColumns(case="upper", strip=True).fit_transform({" a ": [1]}) == {"A": [1]}
    with pytest.raises(bt.InvalidInput, match="repeated column names: a"):
        bt.RenameColumns(case="lower").fit({"a": [1], "A": [2]})
    with pytest.raises(bt.MissingColumn):
        bt.RenameColumns(mapping={"cu": "x"}).fit(raw)


def test_value_steps():
    au = bt.ToNull([-99, "N/A"]).transform(raw)["Au PPM"]
    assert np.isnan(au[1]) and au[0] == 1.0
    filled = bt.FillNull(0.0, columns="Au PPM").transform({"Au PPM": au})["Au PPM"]
    assert filled.tolist() == [1.0, 0.0, 3.0, 400.0]
    assert bt.Clip(columns="Au PPM", upper=100).transform(raw)["Au PPM"].tolist() == [1, -99, 3, 100]
    rock = bt.Replace({"fr": "fresh"}).transform(raw)["RockType"]
    assert rock.tolist() == ["Ox ", "OX", None, "fresh"]
    assert bt.Replace({-99.0: 0.0}).transform(raw)["Au PPM"].tolist() == [1, 0, 3, 400]


def test_to_number_parses_text_and_refuses_the_rest():
    text = {"au": np.array(["0.5", None, "<0.01"], object)}
    with pytest.raises(bt.InvalidInput, match="not numbers: '<0.01'"):
        bt.ToNumber("au").transform(text)
    out = bt.Pipeline([("dl", bt.Replace({"<0.01": "0.005"})), ("n", bt.ToNumber("au"))]).fit_transform(text)
    np.testing.assert_array_equal(out["au"], [0.5, np.nan, 0.005])


def test_normalize_text_merges_categories():
    rock = bt.NormalizeText(case="lower").transform(raw)["RockType"]
    assert rock.tolist() == ["ox", "ox", None, "fr"]
    np.testing.assert_array_equal(bt.NormalizeText().transform(raw)["Au PPM"], raw["Au PPM"])


def test_drop_steps():
    assert len(bt.DropNull(columns="RockType").transform(raw)) == 3
    assert len(bt.DropNull().transform(raw)) == 3
    kept = bt.DropDuplicates().transform(raw)
    assert kept["Au PPM"].tolist() == [1.0, 3.0, 400.0]
    assert len(bt.DropDuplicates(tolerance=4.5).transform(raw)) == 2
    model = bt.BlockModel((0, 0, 0), (1, 1, 1), (2, 1, 1), attributes={"d": [1.0, np.nan]})
    assert len(bt.DropNull().transform(model)) == 1


def test_scale_and_log_invert():
    au = {"au": np.array([0.5, 2.0, 0.0])}
    for step in (bt.Scale("au", 1000.0), bt.Log("au"), bt.Log10("au")):
        out = step.transform(au)
        back = step.inverse_transform(out)["au"]
        np.testing.assert_allclose(back[:2], au["au"][:2])
    assert np.isnan(bt.Log("au").transform(au)["au"][2])
    assert bt.Log10("au").transform(au)["au"][1] == pytest.approx(np.log10(2.0))


def test_steps_chain_in_a_saved_pipeline():
    pipe = bt.Pipeline(
        [
            ("names", bt.RenameColumns(mapping={"Au PPM": "au"}, case="snake")),
            ("nulls", bt.ToNull(-99)),
            ("drop", bt.DropNull(columns="au")),
            ("rock", bt.NormalizeText(case="lower")),
            ("ppb", bt.Scale("au", 1000.0)),
            ("ns", bt.NormalScore(), "au"),
        ]
    )
    out = pipe.fit_transform(raw)
    assert out.attributes.column_names == ["au", "rock_type"] and len(out) == 3
    again = bt.Pipeline.from_json(pipe.to_json())
    np.testing.assert_array_equal(again.transform(raw)["au"], pipe.transform(raw)["au"])
    back = again.inverse_transform(out)
    assert back.attributes.column_names == ["Au PPM", "RockType"]
    np.testing.assert_allclose(back["Au PPM"], [1.0, 3.0, 400.0])
    with pytest.raises(bt.MissingColumn, match="step 'ns'"):
        bt.Pipeline([("ns", bt.NormalScore(), "au")]).fit(raw)
