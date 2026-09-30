import pickle

import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(11)
values = rng.lognormal(0.0, 0.8, 300)
coords = rng.uniform(0, 100, (300, 3))
table = np.column_stack([values, values * rng.uniform(0.5, 1.5, 300), rng.normal(size=300)])
grid = np.stack(np.meshgrid(np.arange(10.0), np.arange(10.0), [0.0]), -1).reshape(-1, 3)


def fitted():
    anam = bt.HermiteAnamorphosis(degree=20).fit(values)
    return [
        (
            bt.NormalScore(tails=(0.0, np.inf)).fit(values),
            lambda o: (o.transform(values), o.inverse_transform([-4.0, 0.0, 4.0])),
        ),
        (anam, lambda o: (o.transform(values), o.inverse_transform([-3.0, 0.0, 3.0]))),
        (
            bt.Capping(cv=0.9).fit(values, domains=np.arange(300) % 3),
            lambda o: (o.transform(values, domains=np.arange(300) % 3), list(o.metal_removed_.values())),
        ),
        (
            bt.Capping(cap={"x": 2.0}).fit(values, domains=np.where(values > 1, "x", "y")),
            lambda o: (o.transform(values, domains=["x"] * 300), list(o.caps_.values())),
        ),
        (bt.BoxCox().fit(values), lambda o: (o.transform(values), o.inverse_transform([0.1, 1.0]))),
        (
            bt.PPMT(iterations=5, seed=4).fit(table),
            lambda o: (o.transform(table), o.inverse_transform(table)),
        ),
        (
            bt.GaussianImputer(seed=2).fit(np.where(table > 1.5, np.nan, table)),
            lambda o: (o.transform(np.where(table > 1.5, np.nan, table)), o.correlation_),
        ),
        (
            bt.GaussianImputer(components=2, seed=2).fit(np.where(table > 1.5, np.nan, table)),
            lambda o: (o.transform(np.where(table > 1.5, np.nan, table)), o.correlation_),
        ),
        (
            bt.KernelDensity(log=True, upper=50.0).fit(values),
            lambda o: (o.pdf(values), o.quantile([0.01, 0.99]), o.sample(20, seed=1)),
        ),
        (
            bt.GaussianMixture(max_components=3, seed=1).fit(table[:, :2]),
            lambda o: (o.pdf(table[:, :2]), o.sample(20), o.means_, o.covariances_, list(o.bic_.values())),
        ),
        (
            bt.NormalScore(reference=bt.KernelDensity(lower=0.0).fit(values)).fit(values),
            lambda o: (o.transform(values), o.inverse_transform([-6.0, 0.0, 6.0])),
        ),
        (bt.PCA(standardize=True).fit(table), lambda o: (o.transform(table), o.inverse_transform(table))),
        (
            bt.MAF(lag=1.0, tolerance=0.01).fit(rng.normal(size=(100, 2)), grid),
            lambda o: (o.transform(table[:, :2]),),
        ),
        (bt.StepwiseConditional(classes=5).fit(table[:, :2]), lambda o: (o.transform(table[:, :2]),)),
        (
            bt.UniformConditioning(anam, 0.8, r_panel=0.6),
            lambda o: (o.panel_recovery(1.2, [0.5, 1.0])["metal"], o.localized_grades(1.2, 8)),
        ),
        (bt.detrend(coords, values, degree=2)[0], lambda o: (o.predict(coords),)),
        (
            bt.detrend(coords, values, bandwidth=[5.0, 20.0], rotation=(30.0, 0.0, 0.0), ratios=(0.5, 1))[0],
            lambda o: (o.predict(coords), o.scores),
        ),
        (bt.cell_declustering(coords, values), lambda o: (o.weights, o.sizes, o.means, o.mean)),
        (
            bt.Variogram(
                [("spherical", 0.7, 50.0)], nugget=0.1, rotation=(30.0, 10.0, 0.0), ratios=(0.5, 0.2)
            ),
            lambda o: (o.gamma_between(coords[:10], coords[10:20]),),
        ),
    ]


def plain():
    return [
        bt.Structure("exponential", 1.0, 30.0),
        bt.Coregionalization(
            [[0.1, 0.0], [0.0, 0.2]], structures=[("spherical", 40.0, [[1.0, 0.7], [0.7, 1.0]])]
        ),
        bt.Search(np.inf, high_grade=(np.inf, 5.0), rotation=(10.0, 0.0, 0.0), ratios=(0.5, 0.5)),
        bt.Search(30.0, soft={("MS", "SM"): 5.0, (1, True): np.inf}),
        bt.Search(30.0, soft=np.inf),
        bt.NormalScore(),
        bt.Capping(quantile=0.99),
        bt.HermiteAnamorphosis(),
        bt.BoxCox(lambda_=0.3),
        bt.PPMT(),
        bt.GaussianImputer(),
        bt.KernelDensity(bandwidth=0.3, lower=0.0),
        bt.GaussianMixture(),
        bt.PCA(),
        bt.MAF(lag=2.0),
        bt.StepwiseConditional(),
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
    assert bt.Search.from_json(bt.Search(np.inf).to_json()).radius == np.inf


def test_envelope_is_checked():
    text = bt.BoxCox(lambda_=0.5).to_json()
    with pytest.raises(bt.InvalidInput, match="expected a PCA"):
        bt.PCA.from_json(text)
    with pytest.raises(bt.InvalidInput, match="format"):
        bt.BoxCox.from_json(text.replace('"format":1', '"format":2'))
    with pytest.raises(bt.InvalidInput):
        bt.BoxCox.from_json("[1, 2]")


model = bt.Variogram([("spherical", 0.8, 40.0)], nugget=0.2)
search = bt.Search(60.0, max_samples=12)
targets = rng.uniform(0, 100, (50, 3))
holes = np.repeat(np.arange(30), 10)


def estimators():
    passes = [bt.Search(20.0, max_samples=8, max_per_hole=2), bt.Search(np.inf)]
    anam = bt.HermiteAnamorphosis(degree=20).fit(values)
    lmc = bt.Coregionalization(
        [[0.1, 0.0], [0.0, 0.1]], structures=[("spherical", 40.0, [[1.0, 0.6], [0.6, 1.0]])]
    )
    return [
        (bt.OrdinaryKriging(model, passes), {"holes": holes, "error_variance": np.full(300, 0.01)}),
        (bt.SimpleKriging(model, search, mean=1.2), {}),
        (bt.IndicatorKriging(model, search, threshold=1.0), {}),
        (bt.UniversalKriging(model, search), {}),
        (bt.FactorialKriging(model, search, [0], nugget=True), {}),
        (bt.BlockKriging(model, search, size=(5.0, 5.0, 5.0)), {}),
        (bt.BayesianKriging(model, search, [1.0], [0.5]), {}),
        (bt.InverseDistance(search, power=3.0, variogram=model), {}),
        (bt.NearestNeighbor(search), {}),
        (bt.MovingAverage(search), {}),
        (bt.MovingMedian(search), {}),
        (bt.LocalLeastSquares(search), {}),
        (bt.DualKriging(model, degree=1), {}),
        (bt.Cokriging(lmc, search), {"variables": np.arange(300) % 2}),
        (bt.DisjunctiveKriging(anam, model, search, order=10), {}),
        (
            bt.MultipleIndicatorKriging(
                [model, model], passes, [0.8, 1.5], tails=(0.0, 20.0), upper_tail=("power", 0.5)
            ),
            {"weights": np.linspace(1.0, 2.0, 300), "holes": holes},
        ),
        (
            bt.MultigaussianKriging(model, passes, tails=(0.0, 20.0)),
            {"weights": np.linspace(1.0, 2.0, 300), "holes": holes, "despike": True},
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
    if isinstance(estimator, bt.DisjunctiveKriging):
        return estimator.predict(targets), estimator.predict_tonnage(targets, 1.0)
    if isinstance(estimator, (bt.MultipleIndicatorKriging, bt.MultigaussianKriging)):
        return indicator_arrays(estimator.predict(targets, cutoffs=[1.0], quantiles=[0.5]))
    if isinstance(estimator, bt.DualKriging):
        return (estimator.predict(targets),)
    if isinstance(estimator, bt.Cokriging):
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
    passes = [bt.Search(20.0, soft={("MS", 7): 6.0}), bt.Search(60.0, soft=np.inf)]
    ok = bt.OrdinaryKriging(model, passes).fit(coords, values, domains=zone)
    ok.to_parquet(path)
    back = bt.OrdinaryKriging.from_parquet(path)
    at = ["MS" if x < 50 else 7 for x in targets[:, 0]]
    a, b = (e.predict(targets, domains=at, diagnostics=True) for e in (back, ok))
    same([a[c] for c in a.column_names], [b[c] for c in b.column_names])
    same((back.cross_validate().estimate,), (ok.cross_validate().estimate,))
    assert bt.read_parquet(path)["domain"].max() == 1


def test_estimator_file_is_a_table_of_samples(tmp_path):
    path = tmp_path / "ok.parquet"
    bt.OrdinaryKriging(model, search).fit(coords, values, holes=holes).to_parquet(path)
    table = bt.read_parquet(path)
    assert table.column_names == ["x", "y", "z", "value", "hole", "error_variance", "domain"]
    assert table.num_rows == 300
    with pytest.raises(bt.InvalidInput, match="expected a SimpleKriging"):
        bt.SimpleKriging.from_parquet(path)
    with pytest.raises(bt.FileError):
        bt.OrdinaryKriging.from_parquet(tmp_path / "missing.parquet")


gaussian = bt.Variogram([("spherical", 1.0, 30.0)])
nodes = rng.uniform(0, 100, (40, 3))
facies = (values > 0.8).astype(int) + (values > 1.5)


def simulators():
    near = bt.Search(40.0, max_samples=8)
    return [
        (
            bt.SGS(gaussian, near),
            {"values": values},
            {"cutoffs": [1.0, 2.0], "quantiles": [0.1, 0.9], "keep": True},
        ),
        (bt.TurningBands(gaussian, bands=50, step=2.0), {"values": values}, {"cutoffs": [1.0]}),
        (bt.SIS([gaussian] * 3, near), {"categories": facies}, {"keep": True}),
        (bt.Plurigaussian(gaussian, proportions=[0.4, 0.4, 0.2]), {"categories": facies}, {}),
        (
            bt.Plurigaussian([gaussian] * 2, proportions=[0.4, 0.4, 0.2], rule=(0, [0, (1, [1, 2])])),
            {"categories": facies, "proportions": west_to_east(coords)},
            {"proportions": west_to_east(nodes)},
        ),
    ]


def west_to_east(xyz):
    p0 = 0.2 + 0.6 * xyz[:, 0] / 100
    return np.column_stack([p0, (1 - p0) / 2, (1 - p0) / 2])


def summary_arrays(summary):
    if isinstance(summary, bt.CategoricalSummary):
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
    mik = bt.MultipleIndicatorKriging(model, bt.Search(15.0), [0.8, 1.5]).fit(coords, values)
    summary = mik.predict(targets, cutoffs=[1.0, 2.0], quantiles=[0.1, 0.9], diagnostics=True)
    assert np.isnan(summary.mean).any()
    summary.to_parquet(path)
    for back in (bt.IndicatorSummary.from_parquet(path), pickle.loads(pickle.dumps(summary))):
        same(indicator_arrays(back), indicator_arrays(summary))
        names = summary.diagnostics.column_names
        assert back.diagnostics.column_names == names
        same([back.diagnostics[n] for n in names], [summary.diagnostics[n] for n in names])


def test_categorical_indicator_kriging_round_trip(tmp_path):
    path = tmp_path / "cik.parquet"
    scheme = bt.Categories(["low", "mid", "high"], colors=["C0", "C1", "C2"])
    rock = np.array(["low", "mid", "high"])[np.digitize(values, [0.8, 1.5])]
    zone = np.where(coords[:, 0] < 50, "w", "e")
    passes = [bt.Search(20.0, max_samples=8, soft={("w", "e"): 5.0}), bt.Search(np.inf)]
    cik = bt.CategoricalIndicatorKriging([model] * 3, passes, simple=True, scheme=scheme)
    cik.fit(coords, rock, weights=np.linspace(1.0, 2.0, 300), holes=holes, domains=zone)
    at = np.where(targets[:, 0] < 50, "w", "e")
    summary = cik.predict(targets, domains=at, diagnostics=True)
    cik.to_parquet(path)
    for back in (bt.CategoricalIndicatorKriging.from_parquet(path), pickle.loads(pickle.dumps(cik))):
        assert back.scheme == scheme
        same((back.predict(targets, domains=at).probabilities,), (summary.probabilities,))
        same((back.cross_validate(folds=5).probabilities,), (cik.cross_validate(folds=5).probabilities,))
    summary.to_parquet(path)
    for back in (bt.CategoricalIndicatorSummary.from_parquet(path), pickle.loads(pickle.dumps(summary))):
        names = ("probabilities", "most_likely", "entropy", "correction", "proportions")
        same([getattr(back, n) for n in names], [getattr(summary, n) for n in names])
        assert back.scheme == scheme and back.diagnostics.column_names == summary.diagnostics.column_names


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
            bt.ImplicitModel(rotation=(20.0, 0.0, 0.0), ratios=(1.0, 0.5)),
            {"coords": coords, "values": values, "cutoff": 1.0},
        ),
        (bt.ImplicitModel(kernel="triharmonic"), structure),
        (
            bt.ImplicitModel(engine="kriging", variogram=gaussian, degree=0),
            {"coords": coords[:50], "values": shell[:50]},
        ),
        (bt.ImplicitModel(engine="gp", rotation=(10.0, 0.0, 0.0)), {"coords": coords, "values": shell}),
    ]


def field(implicit):
    if implicit.report is not None:
        return (*implicit.predict(targets, variance=True), *implicit.report.values())
    return implicit.predict(targets, gradient=True)


@pytest.mark.parametrize("implicit, inputs", implicit_models(), ids=["rbf", "structure", "kriging", "gp"])
def test_implicit_model_round_trip_evaluates_bit_identically(implicit, inputs, tmp_path):
    path = tmp_path / "implicit.parquet"
    implicit.to_parquet(path)
    unfitted = pickle.loads(pickle.dumps(bt.ImplicitModel.from_parquet(path)))
    with pytest.raises(bt.InvalidInput, match="not fitted"):
        unfitted.predict(targets)
    implicit.fit(**inputs)
    same(field(unfitted.fit(**inputs)), field(implicit))
    implicit.to_parquet(path)
    for back in (bt.ImplicitModel.from_parquet(path), pickle.loads(pickle.dumps(implicit))):
        same(field(back), field(implicit))


def test_implicit_model_file_is_a_table_of_constraints(tmp_path):
    path = tmp_path / "implicit.parquet"
    bt.ImplicitModel(kernel="triharmonic").fit(**structure).to_parquet(path)
    table = bt.read_parquet(path)
    assert table.column_names == ["x", "y", "z", "value", "dip", "dip_direction", "plunge", "trend"]
    assert table.num_rows == 1 + 20 + 20 + 3
    with pytest.raises(bt.InvalidInput, match="expected a LocalAnisotropy"):
        bt.LocalAnisotropy.from_parquet(path)


def test_local_anisotropy_round_trip(tmp_path):
    path = tmp_path / "lva.parquet"
    lva = bt.LocalAnisotropy.from_points(coords, k=15)
    lva.to_parquet(path)
    assert bt.read_parquet(path).column_names[3:] == [
        "azimuth",
        "dip",
        "rake",
        "semi_ratio",
        "minor_ratio",
        "scale",
    ]
    estimator = bt.OrdinaryKriging(model, search).fit(coords, values)
    expected = estimator.predict(targets, return_variance=True, anisotropy=lva)
    for back in (bt.LocalAnisotropy.from_parquet(path), pickle.loads(pickle.dumps(lva))):
        same(
            (back.coords, back.angles, back.ratios, back.scales),
            (lva.coords, lva.angles, lva.ratios, lva.scales),
        )
        same(estimator.predict(targets, return_variance=True, anisotropy=back), expected)


def test_snesim_round_trip_simulates_bit_identically(tmp_path):
    path = tmp_path / "snesim.parquet"
    grid = bt.BlockModel((0, 0), (1, 1), (30, 30))
    channels = {"shape": "channel", "code": 1, "proportion": 0.25, "width": 5.0, "azimuth": (80, 100)}
    ti = bt.object_training_image(bt.BlockModel((0, 0), (1, 1), (60, 60)), [channels], seed=3)
    snesim = bt.SNESIM(ti, "facies", template_size=12, n_levels=1, target_proportions=[0.7, 0.3])
    snesim.to_parquet(path)
    unfitted = bt.SNESIM.from_parquet(path)
    same(
        summary_arrays(unfitted.simulate(grid, n=3, seed=4)),
        summary_arrays(snesim.simulate(grid, n=3, seed=4)),
    )
    snesim.fit(grid.centroids[:50], np.arange(50) % 2)
    snesim.to_parquet(path)
    summary = snesim.simulate(grid, n=3, seed=4, keep=True)
    for back in (bt.SNESIM.from_parquet(path), pickle.loads(pickle.dumps(snesim))):
        same(summary_arrays(back.simulate(grid, n=3, seed=4, keep=True)), summary_arrays(summary))


# Image quilting.


@pytest.mark.parametrize("categorical", [True, False])
def test_image_quilting_round_trip_simulates_bit_identically(categorical, tmp_path):
    x, y = np.meshgrid(np.arange(40), np.arange(40))
    image = ((y - 4 * np.sin(x / 5)) % 10 < 4) * 1.0 if categorical else np.sin(x / 5) * np.cos(y / 7)
    ti = bt.BlockModel(
        (0, 0), (1, 1), (40, 40), attributes={"v": image.ravel(), "s": (image + x / 40).ravel()}
    )
    targets = bt.BlockModel((0, 0), (4, 4), (25, 25))
    data = {"values": facies % 2 if categorical else values}
    simulator = bt.ImageQuilting(
        ti, "v", patch_size=8, n_best=4, secondary="s", secondary_weight=0.5, soft_weight=2.0
    )
    p1 = np.linspace(0, 1, 625)
    given = {
        "secondary": np.linspace(0, 1.5, 625),
        "soft": np.column_stack([1 - p1, p1]) if categorical else None,
    }
    path = tmp_path / "simulator.parquet"
    simulator.to_parquet(path)
    unfitted = type(simulator).from_parquet(path).fit(coords[:, :2], **data)
    simulator.fit(coords[:, :2], **data)
    summary = simulator.simulate(targets, n=3, seed=9, keep=True, progress=False, **given)
    same(
        summary_arrays(unfitted.simulate(targets, n=3, seed=9, keep=True, progress=False, **given)),
        summary_arrays(summary),
    )
    simulator.to_parquet(path)
    for back in (type(simulator).from_parquet(path), pickle.loads(pickle.dumps(simulator))):
        same(
            summary_arrays(back.simulate(targets, n=3, seed=9, keep=True, progress=False, **given)),
            summary_arrays(summary),
        )
