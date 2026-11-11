import boitata as bt
import numpy as np
import pytest
from boitata import _display

rng = np.random.default_rng(0)
xyz = rng.uniform(0, 50, (20, 3))
values = rng.normal(size=20)
variogram = bt.Variogram([("spherical", 1.0, 100.0)], nugget=0.1)
kriging = bt.OrdinaryKriging(variogram, bt.Search(radius=100.0))
model = bt.BlockModel((0, 0, 0), (1, 1, 1), (3, 2, 1), attributes={"au": np.arange(6.0)}).with_units(
    {"au": "g/t"}
)
EVAL = {"Variogram": bt.Variogram, "Structure": bt.Structure, "Search": bt.Search, "HighGrade": bt.HighGrade}


def shown():
    collar = {"HOLE_ID": ["a"], "X": [0.0], "Y": [0.0], "Z": [10.0]}
    survey = {"HOLE_ID": ["a"], "DEPTH": [0.0], "AZIMUTH": [0.0], "DIP": [90.0]}
    assays = {"HOLE_ID": ["a"], "FROM": [0.0], "TO": [5.0], "AU": [1.0]}
    search = bt.Search(radius=100.0)
    fitted = bt.OrdinaryKriging(variogram, search).fit(xyz, values)
    sgs = bt.SGS(variogram, search).fit(xyz, values)
    points = bt.PointSet(xyz, {"v": values, "w": values + 1})
    return (
        (bt.Table({"a": [1.0, 2.0], "b": ["x", None]}), "Table"),
        (points, "PointSet"),
        (bt.Polylines([xyz[:3]]), "Polylines"),
        (model, "BlockModel"),
        (bt.Mesh(np.eye(3), [[0, 1, 2]]), "Mesh"),
        (bt.Drillholes(collar, survey, assays), "Drillholes"),
        (bt.Structure("spherical", 1.0, 100.0), "Structure"),
        (variogram, "Variogram"),
        (bt.experimental_variogram(xyz, values, 10.0, 40.0), "ExperimentalVariogram"),
        (bt.experimental_variograms(points, ["v", "w"], 10.0, 40.0), "VariogramSet"),
        (bt.HighGrade(5.0, 30.0), "HighGrade"),
        (search, "Search"),
        (kriging, "OrdinaryKriging"),
        (fitted, "OrdinaryKriging"),
        (sgs, "SGS"),
        (bt.DSS(variogram, [search, search]), "DSS"),
        (bt.TurningBands(variogram), "TurningBands"),
        (fitted.cross_validate(), "CrossValidation"),
        (sgs.simulate(model, n=2), "SimulationSummary"),
    )


def test_every_object_has_a_line_a_card_and_html():
    for obj, title in shown():
        line, card, html = repr(obj), str(obj), obj._repr_html_()
        assert line.startswith(f"{title}(") and "\n" not in line and len(line) < 200
        assert card.splitlines()[0].split("  ")[0] == title
        assert f"<b>{title}</b>" in html
        assert f"{obj}" == line


def test_text_and_html_show_the_same_card():
    for obj, _ in shown():
        card = type(obj).__str__(obj)
        html = obj._repr_html_()
        for cell in card.replace("\n", "  ").split("  "):
            assert _display.escape(cell.strip()) in html


def test_a_dict_of_objects_prints_one_line_each():
    text = repr({"points": bt.PointSet(xyz, {"v": values}), "model": model})
    assert text == (
        "{'points': PointSet(20 points, 1 column), "
        "'model': BlockModel(regular, 6 blocks, 3 × 2 × 1 of 1 × 1 × 1, 1 column)}"
    )


def test_container_card_lists_columns_with_units_and_statistics():
    table = bt.Table({"au": [1.0, 3.0, np.nan], "rock": ["a", None, "b"], "ore": [True, False, True]})
    assert str(table.with_units({"au": "g/t"})) == (
        "Table  3 rows × 3 columns\n"
        "  column  type     unit  nulls  min  mean  max\n"
        "  au      float64  g/t       1    1     2    3\n"
        "  rock    str                1  2 unique: a, b\n"
        "  ore     bool               0  2 true"
    )
    html = model._repr_html_()
    assert "<td>au</td><td>float64</td><td>g/t</td>" in html and "6 blocks" in html


def test_statistics_are_skipped_on_large_tables(monkeypatch):
    monkeypatch.setattr(_display, "STATISTICS_ROWS", 5)
    assert str(model) == (
        "BlockModel  regular, 6 blocks\n"
        "  count   3 × 2 × 1\n"
        "  size    1 × 1 × 1\n"
        "  origin  0, 0, 0\n"
        "  bounds  x 0 to 3, y 0 to 2, z 0 to 1\n"
        "  column  type     unit\n"
        "  au      float64  g/t\n"
        "  statistics skipped above 5 rows"
    )


def test_long_tables_are_cut(monkeypatch):
    monkeypatch.setattr(_display, "SHOWN_ROWS", 2)
    card = str(bt.Table({name: [1.0] for name in "abcde"}))
    assert card.splitlines()[-1] == "  … 3 more columns" and len(card.splitlines()) == 5


@pytest.mark.parametrize(
    "obj",
    [
        bt.Structure("matern", 1.0, 50.0, order=2.5),
        bt.Variogram([("spherical", 0.5, 80.0), bt.Structure("power", 0.4, 250.0, exponent=1.5)], nugget=0.1),
        bt.Variogram([("gaussian", 1.0, 80.0)], rotation=(30, 10, 0), ratios=(0.6, 0.3), length_unit="m"),
        bt.HighGrade(5.0, (30.0, 20.0, 10.0), rotation=(10, 0, 0), mode="clamp"),
        bt.Search(150.0, min_samples=4, max_per_hole=3, length_unit="ft"),
        bt.Search(100.0, sectors=6, ratios=(0.5, 0.25), rotation=(10, 20, 30), min_sectors=2),
        bt.Search(ranges=(150.0, 90.0, 45.0), octant=True, soft=10.0, high_grade=bt.HighGrade(5.0, 30.0)),
        bt.Search(60.0, soft={("ore", "waste"): 5.0}, min_holes=2, target_slope=0.9),
    ],
)
def test_the_line_of_a_value_object_builds_it_back(obj):
    again = eval(repr(obj), EVAL)
    assert repr(again) == repr(obj) and again.to_json() == obj.to_json()


def test_search_line_leaves_defaults_out():
    assert repr(bt.Search(150.0, min_samples=4)) == "Search(150.0, min_samples=4)"


def test_variogram_card_gives_the_ranges_along_each_axis():
    v = bt.Variogram([("spherical", 0.5, 80.0)], nugget=0.1, rotation=(30, 0, 0), ratios=(0.5, 0.25))
    assert str(v) == (
        "Variogram  sill 0.6, 1 structure\n"
        "  rotation  azimuth 30°, dip 0°, rake 0°\n"
        "  ratios    semi/major 0.5, minor/major 0.25\n"
        "  structure  sill  major  semi  minor\n"
        "  nugget      0.1\n"
        "  spherical   0.5     80    40     20"
    )


def test_estimator_says_whether_it_is_fitted():
    fitted = bt.OrdinaryKriging(variogram, bt.Search(radius=100.0)).fit(xyz, values)
    for obj, state in ((kriging, "not fitted"), (fitted, "fitted on 20 samples")):
        assert state in repr(obj) and state in str(obj) and state in obj._repr_html_()
    passes = bt.OrdinaryKriging(variogram, bt.Search.passes(bt.Search(radius=10.0)))
    assert "3 search passes" in repr(passes) and "search 3   radius 30" in str(passes)


def test_cross_validation_shows_scores_not_arrays():
    cv = bt.OrdinaryKriging(variogram, bt.Search(radius=100.0)).fit(xyz, values).cross_validate()
    assert repr(cv).startswith("CrossValidation(20 samples, rmse ") and "array" not in repr(cv)
    assert f"  rmse                        {_display._number(cv.rmse)}" in str(cv)
    unestimated = bt.OrdinaryKriging(variogram, bt.Search(radius=1e-6)).fit(xyz, values).cross_validate()
    assert repr(unestimated) == "CrossValidation(20 samples)"
    assert str(unestimated) == "CrossValidation  20 samples, 0 estimated"


def test_html_escapes_text():
    assert "&lt;b&gt;" in bt.Table({"<b>": [1.0]})._repr_html_()


def test_ipython_shows_the_card_at_the_top_and_the_line_inside_a_container():
    pretty = pytest.importorskip("IPython.lib.pretty").pretty
    assert pretty(variogram) == str(variogram)
    assert pretty([variogram]) == f"[{variogram!r}]"
