import pickle

import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(11)
values = rng.lognormal(0.0, 0.8, 300)
coords = rng.uniform(0, 100, (300, 3))
table = np.column_stack([values, values * rng.uniform(0.5, 1.5, 300), rng.normal(size=300)])
grid = np.stack(np.meshgrid(np.arange(10.0), np.arange(10.0), [0.0]), -1).reshape(-1, 3)


def fitted():
    anam = cs.HermiteAnamorphosis(degree=20).fit(values)
    return [
        (
            cs.NormalScore(tails=(0.0, np.inf)).fit(values),
            lambda o: (o.transform(values), o.inverse_transform([-4.0, 0.0, 4.0])),
        ),
        (anam, lambda o: (o.transform(values), o.inverse_transform([-3.0, 0.0, 3.0]))),
        (cs.BoxCox().fit(values), lambda o: (o.transform(values), o.inverse_transform([0.1, 1.0]))),
        (
            cs.PPMT(iterations=5, seed=4).fit(table),
            lambda o: (o.transform(table), o.inverse_transform(table)),
        ),
        (
            cs.GaussianImputer(seed=2).fit(np.where(table > 1.5, np.nan, table)),
            lambda o: (o.transform(np.where(table > 1.5, np.nan, table)), o.correlation_),
        ),
        (cs.PCA(standardize=True).fit(table), lambda o: (o.transform(table), o.inverse_transform(table))),
        (
            cs.MAF(lag=1.0, tolerance=0.01).fit(rng.normal(size=(100, 2)), grid),
            lambda o: (o.transform(table[:, :2]),),
        ),
        (cs.StepwiseConditional(classes=5).fit(table[:, :2]), lambda o: (o.transform(table[:, :2]),)),
        (
            cs.UniformConditioning(anam, 0.8, 0.6),
            lambda o: (o.panel_recovery(1.2, [0.5, 1.0])["metal"], o.localized_grades(1.2, 8)),
        ),
        (cs.detrend(coords, values, degree=2)[0], lambda o: (o.predict(coords),)),
        (cs.cell_declustering(coords, values), lambda o: (o.weights, o.sizes, o.means, o.mean)),
        (
            cs.Variogram(
                [("spherical", 0.7, 50.0)], nugget=0.1, rotation=(30.0, 10.0, 0.0), ratios=(0.5, 0.2)
            ),
            lambda o: (o.gamma_between(coords[:10], coords[10:20]),),
        ),
    ]


def plain():
    return [
        cs.Structure("exponential", 1.0, 30.0),
        cs.Coregionalization([[0.1, 0.0], [0.0, 0.2]], [("spherical", 40.0, [[1.0, 0.7], [0.7, 1.0]])]),
        cs.Search(np.inf, high_grade=(np.inf, 5.0), rotation=(10.0, 0.0, 0.0), ratios=(0.5, 0.5)),
        cs.Search(30.0, soft={("MS", "SM"): 5.0, (1, True): np.inf}),
        cs.Search(30.0, soft=np.inf),
        cs.NormalScore(),
        cs.HermiteAnamorphosis(),
        cs.BoxCox(lambda_=0.3),
        cs.PPMT(),
        cs.GaussianImputer(),
        cs.PCA(),
        cs.MAF(lag=2.0),
        cs.StepwiseConditional(),
    ]


def same(a, b):
    for x, y in zip(a, b, strict=True):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


@pytest.mark.parametrize("obj, outputs", fitted(), ids=lambda o: type(o).__name__)
def test_round_trip_is_bit_identical(obj, outputs):
    for back in (type(obj).from_json(obj.to_json()), pickle.loads(pickle.dumps(obj))):
        assert type(back) is type(obj)
        same(outputs(back), outputs(obj))
        assert back.to_json() == obj.to_json()


@pytest.mark.parametrize("obj", plain(), ids=lambda o: type(o).__name__)
def test_parameters_and_unfitted_objects_round_trip(obj):
    for back in (type(obj).from_json(obj.to_json()), pickle.loads(pickle.dumps(obj))):
        assert back.to_json() == obj.to_json()


def test_search_keeps_infinite_radius():
    assert cs.Search.from_json(cs.Search(np.inf).to_json()).radius == np.inf


def test_envelope_is_checked():
    text = cs.BoxCox(lambda_=0.5).to_json()
    with pytest.raises(cs.InvalidInput, match="expected a PCA"):
        cs.PCA.from_json(text)
    with pytest.raises(cs.InvalidInput, match="format"):
        cs.BoxCox.from_json(text.replace('"format":1', '"format":2'))
    with pytest.raises(cs.InvalidInput):
        cs.BoxCox.from_json("[1, 2]")


model = cs.Variogram([("spherical", 0.8, 40.0)], nugget=0.2)
search = cs.Search(60.0, max_samples=12)
targets = rng.uniform(0, 100, (50, 3))
holes = np.repeat(np.arange(30), 10)


def estimators():
    passes = [cs.Search(20.0, max_samples=8, max_per_hole=2), cs.Search(np.inf)]
    anam = cs.HermiteAnamorphosis(degree=20).fit(values)
    lmc = cs.Coregionalization([[0.1, 0.0], [0.0, 0.1]], [("spherical", 40.0, [[1.0, 0.6], [0.6, 1.0]])])
    return [
        (cs.OrdinaryKriging(model, passes), {"holes": holes, "error_variance": np.full(300, 0.01)}),
        (cs.SimpleKriging(model, search, mean=1.2), {}),
        (cs.IndicatorKriging(model, search, threshold=1.0), {}),
        (cs.UniversalKriging(model, search), {}),
        (cs.FactorialKriging(model, search, [0], nugget=True), {}),
        (cs.BlockKriging(model, search, size=(5.0, 5.0, 5.0)), {}),
        (cs.BayesianKriging(model, search, [1.0], [0.5]), {}),
        (cs.InverseDistance(search, power=3.0, variogram=model), {}),
        (cs.NearestNeighbor(search), {}),
        (cs.MovingAverage(search), {}),
        (cs.MovingMedian(search), {}),
        (cs.LocalLeastSquares(search), {}),
        (cs.DualKriging(model, degree=1), {}),
        (cs.Cokriging(lmc, search), {"variables": np.arange(300) % 2}),
        (cs.DisjunctiveKriging(anam, model, search, order=10), {}),
        (
            cs.MultipleIndicatorKriging(
                [model, model], passes, [0.8, 1.5], tails=(0.0, 20.0), upper_tail=("power", 0.5)
            ),
            {"weights": np.linspace(1.0, 2.0, 300), "holes": holes},
        ),
    ]


def indicator_arrays(s):
    names = (
        "mean",
        "variance",
        "thresholds",
        "cdf",
        "correction",
        "cutoffs",
        "probability_above",
        "mean_above",
    )
    return [getattr(s, name) for name in (*names, "quantiles", "quantile_values")]


def prediction(estimator):
    if isinstance(estimator, cs.DisjunctiveKriging):
        return estimator.predict(targets), estimator.predict_tonnage(targets, 1.0)
    if isinstance(estimator, cs.MultipleIndicatorKriging):
        return indicator_arrays(estimator.predict(targets, cutoffs=[1.0], quantiles=[0.5]))
    if isinstance(estimator, cs.DualKriging):
        return (estimator.predict(targets),)
    if isinstance(estimator, cs.Cokriging):
        return estimator.predict(targets, variable=1, return_variance=True)
    return estimator.predict(targets, return_variance=True)


@pytest.mark.parametrize("estimator, extra", estimators(), ids=lambda o: type(o).__name__)
def test_estimator_round_trip_predicts_bit_identically(estimator, extra, tmp_path):
    path = tmp_path / "model.parquet"
    estimator.to_parquet(path)
    unfitted = type(estimator).from_parquet(path)
    estimator.fit(coords, values, **extra)
    same(prediction(unfitted.fit(coords, values, **extra)), prediction(estimator))
    estimator.to_parquet(path)
    for back in (type(estimator).from_parquet(path), pickle.loads(pickle.dumps(estimator))):
        assert type(back) is type(estimator)
        same(prediction(back), prediction(estimator))
    if hasattr(estimator, "cross_validate"):
        a, b = estimator.cross_validate(folds=5), back.cross_validate(folds=5)
        same((a.estimate, a.variance), (b.estimate, b.variance))


def test_estimator_with_domains_round_trips(tmp_path):
    path, zone = tmp_path / "ok.parquet", ["MS" if x < 50 else 7 for x in coords[:, 0]]
    passes = [cs.Search(20.0, soft={("MS", 7): 6.0}), cs.Search(60.0, soft=np.inf)]
    ok = cs.OrdinaryKriging(model, passes).fit(coords, values, domains=zone)
    ok.to_parquet(path)
    back = cs.OrdinaryKriging.from_parquet(path)
    at = ["MS" if x < 50 else 7 for x in targets[:, 0]]
    a, b = (e.predict(targets, domains=at, diagnostics=True) for e in (back, ok))
    same([a[c] for c in a.column_names], [b[c] for c in b.column_names])
    same((back.cross_validate().estimate,), (ok.cross_validate().estimate,))
    assert cs.read_parquet(path)["domain"].max() == 1


def test_estimator_file_is_a_table_of_samples(tmp_path):
    path = tmp_path / "ok.parquet"
    cs.OrdinaryKriging(model, search).fit(coords, values, holes=holes).to_parquet(path)
    table = cs.read_parquet(path)
    assert table.column_names == ["x", "y", "z", "value", "hole", "error_variance", "domain"]
    assert table.num_rows == 300
    with pytest.raises(cs.InvalidInput, match="expected a SimpleKriging"):
        cs.SimpleKriging.from_parquet(path)
    with pytest.raises(cs.FileError):
        cs.OrdinaryKriging.from_parquet(tmp_path / "missing.parquet")


gaussian = cs.Variogram([("spherical", 1.0, 30.0)])
nodes = rng.uniform(0, 100, (40, 3))
facies = (values > 0.8).astype(int) + (values > 1.5)


def simulators():
    near = cs.Search(40.0, max_samples=8)
    return [
        (
            cs.SGS(gaussian, near),
            {"values": values},
            {"cutoffs": [1.0, 2.0], "quantiles": [0.1, 0.9], "realizations": True},
        ),
        (cs.TurningBands(gaussian, bands=50, step=2.0), {"values": values}, {"cutoffs": [1.0]}),
        (cs.SIS([gaussian] * 3, near), {"categories": facies}, {"realizations": True}),
        (cs.Plurigaussian(gaussian, proportions=[0.4, 0.4, 0.2]), {"categories": facies}, {}),
        (
            cs.Plurigaussian([gaussian] * 2, proportions=[0.4, 0.4, 0.2], rule=(0, [0, (1, [1, 2])])),
            {"categories": facies, "proportions": west_to_east(coords)},
            {"proportions": west_to_east(nodes)},
        ),
    ]


def west_to_east(xyz):
    p0 = 0.2 + 0.6 * xyz[:, 0] / 100
    return np.column_stack([p0, (1 - p0) / 2, (1 - p0) / 2])


def summary_arrays(summary):
    if isinstance(summary, cs.CategoricalSummary):
        names = ("n", "probabilities", "most_likely", "entropy", "proportions", "realizations")
    else:
        names = ("n", "mean", "variance", "cutoffs", "probability_above", "mean_above", "quantiles")
        names += ("quantile_values", "realization_mean", "realization_above", "realizations")
    return [getattr(summary, name) for name in names]


@pytest.mark.parametrize("simulator, data, options", simulators(), ids=lambda o: type(o).__name__)
def test_simulator_round_trip_simulates_bit_identically(simulator, data, options, tmp_path):
    path = tmp_path / "simulator.parquet"
    simulator.to_parquet(path)
    unfitted = type(simulator).from_parquet(path).fit(coords, **data)
    simulator.fit(coords, **data)
    summary = simulator.simulate(nodes, n=4, seed=9, **options)
    same(summary_arrays(unfitted.simulate(nodes, n=4, seed=9, **options)), summary_arrays(summary))
    simulator.to_parquet(path)
    for back in (type(simulator).from_parquet(path), pickle.loads(pickle.dumps(simulator))):
        same(summary_arrays(back.simulate(nodes, n=4, seed=9, **options)), summary_arrays(summary))
    summary.to_parquet(path)
    for back in (type(summary).from_parquet(path), pickle.loads(pickle.dumps(summary))):
        assert type(back) is type(summary)
        same(summary_arrays(back), summary_arrays(summary))


def test_indicator_summary_round_trip(tmp_path):
    path = tmp_path / "summary.parquet"
    mik = cs.MultipleIndicatorKriging(model, cs.Search(15.0), [0.8, 1.5]).fit(coords, values)
    summary = mik.predict(targets, cutoffs=[1.0, 2.0], quantiles=[0.1, 0.9], diagnostics=True)
    assert np.isnan(summary.mean).any()
    summary.to_parquet(path)
    for back in (cs.IndicatorSummary.from_parquet(path), pickle.loads(pickle.dumps(summary))):
        same(indicator_arrays(back), indicator_arrays(summary))
        names = summary.diagnostics.column_names
        assert back.diagnostics.column_names == names
        same([back.diagnostics[n] for n in names], [summary.diagnostics[n] for n in names])


xy = rng.uniform(0, 100, (20, 2))
on_plane = np.c_[xy, 0.5 * xy[:, 0]]
structure = {
    "coords": [[50, 50, 60]],
    "values": [1.0],
    "boundaries": on_plane,
    "planes": np.c_[on_plane, np.full(20, 26.6), np.full(20, 270.0)],
    "lineations": np.c_[on_plane[:3], np.zeros(3), np.zeros(3)],
}
shell = np.linalg.norm(coords - 50.0, axis=1) - 20.0


def implicit_models():
    return [
        (
            cs.ImplicitModel(rotation=(20.0, 0.0, 0.0), ratios=(1.0, 0.5)),
            {"coords": coords, "values": values, "cutoff": 1.0},
        ),
        (cs.ImplicitModel(kernel="triharmonic"), structure),
        (
            cs.ImplicitModel(engine="kriging", variogram=gaussian, drift_degree=0),
            {"coords": coords[:50], "values": shell[:50]},
        ),
        (cs.ImplicitModel(engine="gp", rotation=(10.0, 0.0, 0.0)), {"coords": coords, "values": shell}),
    ]


def field(implicit):
    if implicit.report is not None:
        return (*implicit.evaluate(targets, variance=True), *implicit.report.values())
    return implicit.evaluate(targets, gradient=True)


@pytest.mark.parametrize("implicit, inputs", implicit_models(), ids=["rbf", "structure", "kriging", "gp"])
def test_implicit_model_round_trip_evaluates_bit_identically(implicit, inputs, tmp_path):
    path = tmp_path / "implicit.parquet"
    implicit.to_parquet(path)
    unfitted = pickle.loads(pickle.dumps(cs.ImplicitModel.from_parquet(path)))
    with pytest.raises(cs.InvalidInput, match="not fitted"):
        unfitted.evaluate(targets)
    implicit.fit(**inputs)
    same(field(unfitted.fit(**inputs)), field(implicit))
    implicit.to_parquet(path)
    for back in (cs.ImplicitModel.from_parquet(path), pickle.loads(pickle.dumps(implicit))):
        same(field(back), field(implicit))


def test_implicit_model_file_is_a_table_of_constraints(tmp_path):
    path = tmp_path / "implicit.parquet"
    cs.ImplicitModel(kernel="triharmonic").fit(**structure).to_parquet(path)
    table = cs.read_parquet(path)
    assert table.column_names == ["x", "y", "z", "value", "dip", "dip_direction", "plunge", "trend"]
    assert table.num_rows == 1 + 20 + 20 + 3
    with pytest.raises(cs.InvalidInput, match="expected a LocalAnisotropy"):
        cs.LocalAnisotropy.from_parquet(path)


def test_local_anisotropy_round_trip(tmp_path):
    path = tmp_path / "lva.parquet"
    lva = cs.LocalAnisotropy.from_points(coords, k=15)
    lva.to_parquet(path)
    assert cs.read_parquet(path).column_names[3:] == ["azimuth", "dip", "rake", "semi_ratio", "minor_ratio"]
    estimator = cs.OrdinaryKriging(model, search).fit(coords, values)
    expected = estimator.predict(targets, return_variance=True, anisotropy=lva)
    for back in (cs.LocalAnisotropy.from_parquet(path), pickle.loads(pickle.dumps(lva))):
        same((back.coords, back.angles, back.ratios), (lva.coords, lva.angles, lva.ratios))
        same(estimator.predict(targets, return_variance=True, anisotropy=back), expected)
