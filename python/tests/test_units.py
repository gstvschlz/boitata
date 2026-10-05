import boitata as bt
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

points = bt.PointSet(np.zeros((3, 3)), {"au": [1.0, np.nan, 34.285714], "d": [2.7, 2.8, 2.9]}).with_units(
    {"au": "g/t", "d": "t/m3"}
)


def test_units_convert_and_round_trip(tmp_path):
    assert points.units == {"au": "g/t", "d": "t/m3"}
    oz = points.convert_units("au", to="oz/t")
    np.testing.assert_allclose(oz["au"], [1 / 34.285714, np.nan, 1.0], rtol=1e-6)
    assert oz.units["au"] == "oz/t"
    back = oz.convert_units("au", to="ppb").convert_units("au", to="g/t")
    np.testing.assert_allclose(back["au"], points["au"])
    assert points.convert_units("d", to="g/cm3")["d"].tolist() == points["d"].tolist()
    path = tmp_path / "p.parquet"
    bt.write_parquet(path, points)
    assert bt.read_parquet(path).units == points.units
    assert points.attributes.units == points.units


def test_units_are_dropped_with_replaced_values_and_refused_across_kinds():
    assert points.with_column("au", [1.0, 2.0, 3.0]).units == {"d": "t/m3"}
    assert points.with_units({"d": None}).units == {"au": "g/t"}
    with pytest.raises(bt.InvalidInput, match="cannot convert grade"):
        points.convert_units("au", to="m")
    with pytest.raises(bt.InvalidInput, match=r"cannot convert grade \(%\) to ratio \(ratio%\)"):
        points.with_units({"au": "%"}).convert_units("au", to="ratio%")
    with pytest.raises(bt.InvalidInput, match="cannot convert metal"):
        bt.Table({"m": [1.0]}).with_units({"m": "kg metal"}).convert_units("m", to="t")
    with pytest.raises(bt.InvalidInput, match="cannot convert USD"):
        bt.Table({"c": [1.0]}).with_units({"c": "USD"}).convert_units("c", to="BRL")
    with pytest.raises(bt.InvalidInput, match="unknown unit `ppn`; did you mean `ppm`"):
        points.with_units({"au": "ppn"})
    with pytest.raises(bt.InvalidInput, match="has no unit"):
        points.with_units({"au": None}).convert_units("au", to="ppm")


def test_compound_units_convert():
    t = bt.Table({"d": [2700.0], "c": [2.0], "m": [1.0]}).with_units(
        {"d": "kg/m3", "c": "kUSD/m", "m": "Moz metal"}
    )
    assert t.convert_units("d", to="t/m3")["d"].tolist() == pytest.approx([2.7])
    assert t.convert_units("c", to="USD/ft")["c"].tolist() == pytest.approx([609.6])
    assert t.convert_units("m", to="t metal")["m"].tolist() == pytest.approx([31.1034768])


def test_csv_headers_carry_units_and_unknown_stored_units_stay_opaque(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("hole (DDH),au [g/t],cu (%),dens\nA,1,2,2.7\n")
    table = bt.read_csv(path, units={"dens": "t/m3", "cu": "ppm"}, progress=False)
    assert table.column_names == ["hole (DDH)", "au", "cu", "dens"]
    assert table.units == {"au": "g/t", "cu": "%", "dens": "t/m3"}
    with pytest.raises(bt.InvalidInput, match="unknown unit"):
        bt.read_csv(path, units={"dens": "tm3"}, progress=False)
    pq = pytest.importorskip("pyarrow.parquet")
    stored = tmp_path / "opaque.parquet"
    bt.write_parquet(stored, bt.Table({"au": [1.0]}))
    raw = pq.read_table(stored)
    field = raw.schema.field("au").with_metadata({b"unit": b"dwt"})
    pq.write_table(raw.cast(raw.schema.set(0, field)), stored, compression="none")
    opaque = bt.read_parquet(stored, progress=False)
    assert opaque.units == {"au": "dwt"}
    with pytest.raises(bt.InvalidInput, match="the unit `dwt` of column `au` is not understood"):
        opaque.convert_units("au", to="g/t")


def test_project_units_fill_missing_units_only(tmp_path):
    bt.write_parquet(tmp_path / "p.parquet", points)
    try:
        bt.set_units(columns={"au": "g/t", "cu": "%"})
        assert bt.Table({"au": [1.0], "x": [0.0]}).units == {"au": "g/t"}
        assert bt.PointSet(np.zeros((1, 3)), {"cu": [1.0]}).units == {"cu": "%"}
        with bt.units(columns={"au": "ppb", "cu": None}):
            assert bt.Table({"au": [1.0], "cu": [1.0]}).units == {"au": "ppb"}
        assert bt.Table({"au": [1.0], "cu": [1.0]}).units == {"au": "g/t", "cu": "%"}
        assert bt.read_parquet(tmp_path / "p.parquet", progress=False).units == points.units
        with pytest.raises(bt.InvalidInput, match="unknown unit"):
            bt.set_units(columns={"au": "gpt"})
    finally:
        bt.set_units(columns={"au": None, "cu": None})
    assert bt.Table({"au": [1.0]}).units == {}


def test_plots_label_axes_with_units():
    _, ax = bt.plot.histogram("au", data=points)
    assert ax.get_xlabel() == "au (g/t)"
    _, ax = bt.plot.scatter("au", "d", data=points)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("au (g/t)", "d (t/m3)")
    _, ax = bt.plot.histogram([1.0, 2.0])
    assert ax.get_xlabel() == ""


rng = np.random.default_rng(5)
samples = bt.PointSet(rng.uniform(0, 100, (60, 3)) * [1, 1, 0], {"au": rng.lognormal(0, 0.5, 60)}).with_units(
    {"au": "g/t"}
)
grid = bt.BlockModel((0, 0, 0), (20, 20, 1), (5, 5, 1))
variogram = bt.Variogram([("spherical", 0.25, 40.0)])
search = bt.Search(radius=60, max_samples=12)


def test_columns_carry_units_until_arithmetic():
    au = samples["au"]
    assert isinstance(au, bt.UnitArray) and au.unit == "g/t" and au[:3].unit == "g/t"
    assert (
        type(au * 2) is np.ndarray and type(np.log(au)) is np.ndarray and type(au.mean()) is not bt.UnitArray
    )
    assert samples.with_column("copy", au).units["copy"] == "g/t"
    assert "copy" not in samples.with_column("copy", au * 1.0).units
    assert samples.with_column("copy", au * 1000, unit="ppb").units["copy"] == "ppb"
    assert samples.with_columns({"a": au, "b": au * 1.0}).units == {"au": "g/t", "a": "g/t"}
    assert bt.Table({"a": au}).units == {"a": "g/t"}
    import pickle

    assert pickle.loads(pickle.dumps(au)).unit == "g/t"


def test_estimates_take_the_unit_of_the_values():
    ok = bt.OrdinaryKriging(variogram, search).fit(samples, "au")
    estimate, variance = ok.predict(grid, return_variance=True, progress=False)
    assert ok.unit == "g/t" and estimate.unit == "g/t" and variance.unit == "(g/t)^2"
    assert grid.with_column("au", estimate).units == {"au": "g/t"}
    diagnostics = ok.predict(grid, diagnostics=True, progress=False)
    assert diagnostics.units == {"value": "g/t", "variance": "(g/t)^2"}
    cv = ok.cross_validate()
    assert (cv.actual.unit, cv.estimate.unit) == ("g/t", "g/t")
    idw = bt.InverseDistance(search).fit(samples.coords, samples["au"])
    assert idw.predict(grid, progress=False).unit == "g/t"
    ik = bt.IndicatorKriging(variogram, search, threshold=1.0).fit(samples, "au")
    assert ik.predict(grid, progress=False).unit == "ratio"
    unknown = bt.OrdinaryKriging(variogram, search).fit(samples.coords, np.asarray(samples["au"]))
    assert not hasattr(unknown.predict(grid, progress=False), "unit")


def test_transforms_and_simulations_carry_units():
    ns = bt.NormalScore().fit(samples["au"])
    scores = ns.transform(samples["au"])
    assert not hasattr(scores, "unit")
    assert ns.inverse_transform(scores).unit == "g/t"
    assert bt.NormalScore.from_json(ns.to_json()).inverse_transform(scores).unit == "g/t"
    assert bt.Capping(cap=2.0).fit(samples["au"]).transform(samples["au"]).unit == "g/t"
    sgs = bt.SGS(variogram, search).fit(samples, "au")
    s = sgs.simulate(grid, n=3, seed=1, keep=True, cutoffs=[1.0], progress=False)
    assert (s.mean.unit, s.variance.unit, s.realizations.unit, s.mean_above.unit) == (
        "g/t",
        "(g/t)^2",
        "g/t",
        "g/t",
    )
    assert not hasattr(s.probability_above, "unit")


def test_regularize_and_composites_keep_units():
    fine = grid.with_column("au", np.linspace(0, 1, len(grid)), unit="g/t")
    coarse = bt.BlockModel((0, 0, 0), (50, 50, 1), (2, 2, 1))
    assert fine.regularize(coarse).units == {"au": "g/t", "fraction": "ratio"}
    collar = {"HOLE_ID": ["A"], "X": [0.0], "Y": [0.0], "Z": [0.0]}
    survey = {"HOLE_ID": ["A"], "DEPTH": [0.0], "AZIMUTH": [0.0], "DIP": [90.0]}
    intervals = bt.Table({"HOLE_ID": ["A", "A"], "FROM": [0.0, 1.0], "TO": [1.0, 2.0], "AU": [1.0, 3.0]})
    holes = bt.Drillholes(collar, survey, intervals.with_units({"AU": "g/t"}))
    assert holes.composite(2.0, ["AU"]).units == {"AU": "g/t"}
