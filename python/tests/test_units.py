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


FT = 0.3048


def test_containers_declare_convert_and_store_length_units(tmp_path):
    m = bt.PointSet([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], length_unit="m")
    ft = m.to_length_unit("ft")
    assert ft.length_unit == "ft"
    np.testing.assert_allclose(ft.coords[1], [10 / FT, 0, 0])
    bt.write_parquet(tmp_path / "p.parquet", ft)
    assert bt.read_parquet(tmp_path / "p.parquet", progress=False).length_unit == "ft"
    model = bt.BlockModel((0, 0, 0), (10, 10, 5), (2, 2, 1), length_unit="m", crs="EPSG:31982")
    with pytest.raises(bt.InvalidInput, match="give their CRS in ft"):
        model.to_length_unit("ft")
    moved = model.to_length_unit("ft", crs="local")
    np.testing.assert_allclose(moved.size, [10 / FT, 10 / FT, 5 / FT])
    bt.write_parquet(tmp_path / "b.parquet", moved)
    assert bt.read_parquet(tmp_path / "b.parquet", progress=False).length_unit == "ft"
    with pytest.raises(bt.InvalidInput, match="declare the length unit"):
        bt.PointSet([[0.0, 0.0, 0.0]]).to_length_unit("m")
    with pytest.raises(bt.InvalidInput, match="need a length unit"):
        bt.PointSet([[0.0, 0.0, 0.0]], length_unit="g/t")
    with bt.units(length="ft"):
        assert bt.PointSet([[0.0, 0.0, 0.0]]).length_unit == "ft"
        assert bt.read_parquet(tmp_path / "p.parquet", progress=False).length_unit == "ft"
    assert bt.PointSet([[0.0, 0.0, 0.0]]).length_unit is None


def test_kriging_with_parameters_in_feet_matches_kriging_in_metres():
    """Theory check: lengths in another unit change no estimate once converted."""
    data = bt.PointSet(samples.coords, samples.attributes, length_unit="m")
    targets = bt.BlockModel((0, 0, 0), (20, 20, 1), (5, 5, 1), length_unit="m")
    metres = bt.OrdinaryKriging(variogram, search).fit(data, "au").predict(targets, progress=False)
    in_feet = bt.Variogram([("spherical", 0.25, 40.0 / FT)], length_unit="ft")
    search_feet = bt.Search(radius=60 / FT, max_samples=12, length_unit="ft")
    feet = bt.OrdinaryKriging(in_feet, search_feet).fit(data, "au").predict(targets, progress=False)
    np.testing.assert_allclose(feet, metres, rtol=1e-10)
    with pytest.raises(bt.InvalidInput, match="samples are in m and targets in ft"):
        bt.OrdinaryKriging(variogram, search).fit(data, "au").predict(
            targets.to_length_unit("ft"), progress=False
        )
    with pytest.raises(bt.InvalidInput, match="give them one"):
        bt.OrdinaryKriging(in_feet, bt.Search(radius=60, length_unit="m"))
    # Coordinates without a unit are taken as they are, in the unit of the parameters.
    undeclared = bt.OrdinaryKriging(in_feet, search_feet).fit(samples.coords, samples["au"])
    assert np.isfinite(undeclared.predict(targets.centroids, progress=False)).any()


def test_variograms_carry_length_and_value_units():
    data = bt.PointSet(samples.coords, samples.attributes, length_unit="m")
    experimental = bt.experimental_variogram(data, "au", 10.0, 60.0)
    assert (experimental.length_unit, experimental.unit) == ("m", "g/t")
    fitted = bt.Variogram.fit(experimental)
    assert (fitted.length_unit, fitted.unit) == ("m", "g/t")
    back = bt.Variogram.from_json(fitted.to_json())
    assert (back.length_unit, back.unit) == ("m", "g/t")
    assert bt.Variogram.from_json(variogram.to_json()).length_unit is None
    _, ax = bt.plot.variogram(experimental, variogram=fitted)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Lag distance (m)", "γ(h) (g/t)²")


def test_containers_in_different_units_do_not_meet():
    data = bt.PointSet(samples.coords, samples.attributes, length_unit="m")
    sgs = bt.SGS(variogram, search).fit(data, "au")
    feet_grid = bt.BlockModel((0, 0, 0), (20, 20, 1), (5, 5, 1), length_unit="ft")
    with pytest.raises(bt.InvalidInput, match="samples are in m and targets in ft"):
        sgs.simulate(feet_grid, n=1, seed=1, progress=False)
    with pytest.raises(bt.InvalidInput, match="variogram and search lengths are in ft and samples in m"):
        bt.SGS(bt.Variogram([("spherical", 1.0, 30.0)], length_unit="ft"), search).fit(data, "au")
    fine = bt.BlockModel((0, 0, 0), (10, 10, 1), (4, 4, 1), length_unit="m").with_column("v", np.ones(16))
    with pytest.raises(bt.InvalidInput, match="source blocks are in m and target blocks in ft"):
        fine.regularize(feet_grid)
    cube = bt.Mesh(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        length_unit="ft",
    )
    with pytest.raises(bt.InvalidInput, match="mesh vertices are in ft and points in m"):
        cube.contains(data)


def test_drillholes_carry_their_length_unit():
    collar = {"HOLE_ID": ["A"], "X": [0.0], "Y": [0.0], "Z": [0.0]}
    survey = {"HOLE_ID": ["A"], "DEPTH": [0.0], "AZIMUTH": [0.0], "DIP": [90.0]}
    intervals = {"HOLE_ID": ["A", "A"], "FROM": [0.0, 1.0], "TO": [1.0, 2.0], "AU": [1.0, 3.0]}
    holes = bt.Drillholes(collar, survey, intervals, length_unit="ft")
    composites = holes.composite(2.0, ["AU"])
    assert (
        holes.length_unit == "ft" and composites.length_unit == "ft" and holes.samples().length_unit == "ft"
    )
    assert composites.units == {"from": "ft", "to": "ft", "length": "ft", "AU_length": "ft"}


def _model(grade_unit):
    model = bt.BlockModel((0, 0, 0), (10, 10, 5), (4, 4, 1), length_unit="m")
    au = np.linspace(0.5, 4.0, len(model))
    return model.with_column("au", au, unit=grade_unit).with_column(
        "dens", np.full(len(model), 2.7), unit="t/m3"
    )


def test_grade_tonnage_balances_metal_in_auto_scaled_units():
    """Theory check: Σ volume × density × grade equals the metal column converted back to grams."""
    model = _model("g/t")
    gt = bt.grade_tonnage("au", [0.0, 2.0], data=model, density="dens")
    assert gt.units == {"cutoff": "g/t", "tonnage": "kt", "mean_grade": "g/t", "metal": "kg metal"}
    metal = gt.convert_units("metal", to="g metal")["metal"]
    expected = np.sum(model.volumes * model["dens"] * model["au"])
    assert metal[0] == pytest.approx(expected, rel=1e-12)
    assert gt.convert_units("tonnage", to="t")["tonnage"][0] == pytest.approx(16 * 500 * 2.7)
    with pytest.raises(bt.InvalidInput, match="cannot convert metal"):
        gt.convert_units("metal", to="t")
    bare = bt.grade_tonnage("au", [0.0], data=model, density=2.7)
    assert bare.units == {"cutoff": "g/t", "mean_grade": "g/t"}
    assert bare["tonnage"][0] == pytest.approx(16 * 500 * 2.7)


def test_metal_reads_in_the_units_of_its_grade():
    gold = bt.grade_tonnage("au", [0.0], data=_model("oz/t"), density="dens")
    assert gold.units["metal"] == "koz metal"
    copper = bt.grade_tonnage("au", [0.0], data=_model("%"), density="dens")
    assert copper.units["metal"] == "t metal"
    model = _model("g/t").with_column("other", np.linspace(0.4, 3.0, 16), unit="g/t")
    compared = bt.compare_models(model, ["au", "other"], [0.0], density="dens")
    assert compared.units["tonnage_diff"] == "ratio" and compared.units["metal"] == "kg metal"


def test_recoveries_and_simulated_curves_carry_units():
    hermite = bt.HermiteAnamorphosis(degree=12).fit(samples["au"])
    recovery = hermite.grade_tonnage([0.5, 1.0])
    assert recovery.units == {
        "cutoff": "g/t",
        "tonnage": "ratio",
        "mean_grade": "g/t",
        "metal": "g/t",
        "benefit": "g/t",
    }
    assert hermite.inverse_transform([0.0]).unit == "g/t"
    data = bt.PointSet(samples.coords, samples.attributes, length_unit="m")
    blocks = bt.BlockModel((0, 0, 0), (20, 20, 5), (5, 5, 1), length_unit="m")
    summary = (
        bt.SGS(variogram, search)
        .fit(data, "au")
        .simulate(
            blocks.with_column("dens", np.full(25, 2.7), unit="t/m3"),
            n=4,
            seed=2,
            grade_tonnage_cutoffs=[0.0, 1.0],
            density="dens",
            progress=False,
        )
    )
    curves = summary.grade_tonnage()
    assert (curves.units["tonnage"], curves.units["metal"], curves.units["probability"]) == (
        "kt",
        "kg metal",
        "ratio",
    )
