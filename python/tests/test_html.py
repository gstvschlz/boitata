import boitata as bt
import numpy as np

rng = np.random.default_rng(0)
xyz = rng.uniform(0, 50, (20, 3))
values = rng.normal(size=20)
variogram = bt.Variogram([("spherical", 1.0, 100.0)], nugget=0.1)
kriging = bt.OrdinaryKriging(variogram, bt.Search(radius=100.0))


def test_every_container_model_and_result_has_html():
    model = bt.BlockModel((0, 0, 0), (1, 1, 1), (3, 2, 1), attributes={"au": np.arange(6.0)}).with_units(
        {"au": "g/t"}
    )
    collar = {"HOLE_ID": ["a"], "X": [0.0], "Y": [0.0], "Z": [10.0]}
    survey = {"HOLE_ID": ["a"], "DEPTH": [0.0], "AZIMUTH": [0.0], "DIP": [90.0]}
    assays = {"HOLE_ID": ["a"], "FROM": [0.0], "TO": [5.0], "AU": [1.0]}
    fitted = bt.OrdinaryKriging(variogram, bt.Search(radius=100.0)).fit(xyz, values)
    sgs = bt.SGS(variogram, bt.Search(radius=100.0)).fit(xyz, values)
    for obj, title in (
        (bt.Table({"a": [1.0, 2.0], "b": ["x", None]}), "Table: 2 rows"),
        (bt.PointSet(xyz, {"v": values}), "PointSet"),
        (bt.Polylines([xyz[:3]]), "Polylines"),
        (model, "BlockModel"),
        (bt.Mesh(np.eye(3), [[0, 1, 2]]), "Mesh"),
        (bt.Drillholes(collar, survey, assays), "Drillholes"),
        (variogram, "Variogram"),
        (kriging, "OrdinaryKriging"),
        (fitted.cross_validate(), "CrossValidation"),
        (sgs.simulate(model, n=2), "SimulationSummary"),
    ):
        html = obj._repr_html_()
        assert f"<b>{title}</b>" in html


def test_html_lists_columns_with_types_and_units():
    model = bt.BlockModel((0, 0, 0), (1, 1, 1), (3, 2, 1), attributes={"au": np.arange(6.0)}).with_units(
        {"au": "g/t"}
    )
    html = model._repr_html_()
    assert "<td>au</td><td>Float64</td><td>g/t</td>" in html and "6 of 6 cells" in html


def test_estimator_html_says_whether_it_is_fitted():
    assert "not fitted" in kriging._repr_html_()
    assert (
        "20 samples" in bt.OrdinaryKriging(variogram, bt.Search(radius=100.0)).fit(xyz, values)._repr_html_()
    )


def test_html_escapes_text():
    assert "&lt;b&gt;" in bt.Table({"<b>": [1.0]})._repr_html_()
