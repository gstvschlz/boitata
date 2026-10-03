import inspect
import re

import boitata as bt
import numpy as np
import pytest

BRITISH = re.compile(
    r"neighbour|colour|centre|modell|grey|honour|metre|labell|favour|isation"
    r"|(normal|standard|local|discret|regular|optim|maxim|minim|summar|random|categor|initial|visual|priorit|real)"
    r"is(e|es|ed|ing|er|ers)(?![a-z])"
)

# Defaulted parameters that may stay positional: optional data, and the one choice that defines the call.
POSITIONAL_DEFAULTS = {
    ("PointSet", "attributes"),
    ("Drillholes", "intervals"),
    ("check_drillholes", "survey"),
    ("check_drillholes", "intervals"),
    ("ImplicitModel", "engine"),
    ("SNESIM", "column"),
    ("ImplicitModel.fit", "coords"),
    ("ImplicitModel.fit", "values"),
    ("Variogram.fit", "model"),
    ("Variogram.fit_directional", "model"),
    ("ExperimentalVariogram.fit", "model"),
    ("Coregionalization.fit", "model"),
    ("plot.slab", "values"),
    ("plot3d.plot", "values"),
    ("plot3d.slices", "values"),
}
# `domains` without `domain_column`: none.
DOMAINS_EXEMPT = set()
# Not plots: color and legend helpers, and the conversion to pyvista.
NOT_PLOTS = {"plot.category_colors", "plot.category_legend", "plot3d.to_pyvista"}
# `fit` building a new model from experimental variograms, not fitting the object in place.
MODEL_FITS = {"Variogram", "ExperimentalVariogram", "Coregionalization"}


def public_callables():
    out = {}
    for prefix, module in [("", bt), ("plot.", bt.plot), ("plot3d.", bt.plot3d)]:
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.ismodule(obj) or (inspect.isclass(obj) and issubclass(obj, Exception)):
                continue
            out[prefix + name] = obj
            if not inspect.isclass(obj):
                continue
            for attr in dir(obj):
                static = inspect.getattr_static(obj, attr)
                if (
                    attr.startswith("_")
                    or isinstance(static, property)
                    or type(static).__name__ == "getset_descriptor"
                ):
                    continue
                if callable(getattr(obj, attr)):
                    out[f"{prefix}{name}.{attr}"] = getattr(obj, attr)
    return out


CALLABLES = public_callables()


def parameters(obj):
    params = list(inspect.signature(obj).parameters.values())
    return params[1:] if params and params[0].name == "self" else params


def test_every_public_callable_has_a_signature():
    missing = []
    for name, obj in CALLABLES.items():
        try:
            inspect.signature(obj)
        except (TypeError, ValueError):
            missing.append(name)
    assert missing == []


def test_names_and_parameters_are_american():
    found = [name for name in CALLABLES if BRITISH.search(name.lower())]
    found += [
        f"{name}({p.name})"
        for name, obj in CALLABLES.items()
        for p in parameters(obj)
        if BRITISH.search(p.name)
    ]
    assert found == []


def test_defaulted_parameters_are_keyword_only():
    found = [
        f"{name}({p.name})"
        for name, obj in CALLABLES.items()
        for p in parameters(obj)
        if p.default is not p.empty
        and p.kind is not p.KEYWORD_ONLY
        and (name, p.name) not in POSITIONAL_DEFAULTS
        and (name.split(".")[-1], p.name) not in POSITIONAL_DEFAULTS
    ]
    assert found == []


def test_domains_come_with_domain_column():
    found = [
        name
        for name, obj in CALLABLES.items()
        if "domains" in (names := {p.name for p in parameters(obj)})
        and "domain_column" not in names
        and name not in DOMAINS_EXEMPT
    ]
    assert found == []


def test_plots_take_keyword_only_axes():
    plots = {
        name: obj for name, obj in CALLABLES.items() if name.startswith("plot") and name not in NOT_PLOTS
    }
    for name, obj in plots.items():
        target = {"plot.scatter_matrix": "axes", "plot.variograms": "axes", "plot.fence": "axes"}.get(
            name, "plotter" if name.startswith("plot3d.") else "ax"
        )
        p = inspect.signature(obj).parameters.get(target)
        assert p is not None and p.kind is p.KEYWORD_ONLY, name


def test_transforms_have_fit_transform():
    classes = {name for name in CALLABLES if name.endswith(".transform")}
    assert classes
    assert [c for c in classes if c.replace(".transform", ".fit_transform") not in CALLABLES] == []


rng = np.random.default_rng(3)
values = rng.lognormal(0.0, 0.5, 60)
coords = rng.uniform(0, 100, (60, 3))
table = np.column_stack([values, values * rng.uniform(0.5, 1.5, 60)])
grid = np.stack(np.meshgrid(np.arange(8.0), np.arange(8.0), [0.0]), -1).reshape(-1, 3)
categories = (values > 0.8).astype(int) + (values > 1.5)
model = bt.Variogram([("spherical", 1.0, 40.0)])
search = bt.Search(60.0, max_samples=12)


def fits():
    anam = bt.HermiteAnamorphosis(degree=10).fit(values)
    square = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]])
    walls = [bt.Mesh(square + [0, 0, z], [[0, 1, 2], [0, 2, 3]]) for z in (0, 100)]
    lmc = bt.Coregionalization(
        [[0.0, 0.0], [0.0, 0.0]], structures=[("spherical", 40.0, [[1.0, 0.5], [0.5, 1.0]])]
    )
    return [
        (bt.OrdinaryKriging(model, search), (coords, values)),
        (bt.SimpleKriging(model, search), (coords, values)),
        (bt.IndicatorKriging(model, search, threshold=1.0), (coords, values)),
        (bt.UniversalKriging(model, search), (coords, values)),
        (
            bt.ExternalDriftKriging(model, search, "drift"),
            (bt.PointSet(coords, {"drift": coords[:, 0]}), values),
        ),
        (bt.FactorialKriging(model, search, [0]), (coords, values)),
        (bt.BlockKriging(model, search, (5.0, 5.0, 5.0)), (coords, values)),
        (bt.BayesianKriging(model, search, [1.0], [0.5]), (coords, values)),
        (bt.InverseDistance(search), (coords, values)),
        (bt.NearestNeighbor(search), (coords, values)),
        (bt.MovingAverage(search), (coords, values)),
        (bt.MovingMedian(search), (coords, values)),
        (bt.LocalLeastSquares(search), (coords, values)),
        (bt.DualKriging(model), (coords, values)),
        (bt.Cokriging(lmc, search), (coords, values, np.arange(60) % 2)),
        (bt.DisjunctiveKriging(anam, model, search), (coords, values)),
        (bt.MultipleIndicatorKriging(model, search, [0.8, 1.5]), (coords, values)),
        (bt.MultigaussianKriging(model, search), (coords, values)),
        (bt.CategoricalIndicatorKriging(model, search), (coords, categories)),
        (bt.SGS(model, search), (coords, values)),
        (bt.TurningBands(model, bands=20), (coords, values)),
        (bt.SIS([model] * 3, search), (coords, categories)),
        (bt.Plurigaussian(model, proportions=[0.4, 0.4, 0.2]), (coords, categories)),
        (
            bt.SNESIM(bt.object_training_image(bt.BlockModel((0, 0), (1, 1), (20, 20)), []), "facies"),
            (coords, categories * 0),
        ),
        (
            bt.ImageQuilting(
                bt.BlockModel((0, 0), (1, 1), (8, 8), attributes={"f": np.arange(64.0) % 3}), "f"
            ),
            (coords, categories),
        ),
        (bt.MultivariateSimulation(bt.PCA(), [bt.SGS(model, search)] * 2), (coords, table)),
        (bt.ImplicitModel(), (coords, values - values.mean())),
        (bt.NormalScore(), (values,)),
        (bt.Capping(quantile=0.99), (values,)),
        (bt.HermiteAnamorphosis(degree=10), (values,)),
        (bt.BoxCox(), (values,)),
        (bt.PPMT(iterations=2), (table,)),
        (bt.PCA(), (table,)),
        (bt.MAF(lag=1.0, tolerance=0.01), (rng.normal(size=(64, 2)), grid)),
        (bt.StepwiseConditional(classes=3), (table,)),
        (bt.GaussianImputer(), (table,)),
        (bt.KernelDensity(), (values,)),
        (bt.GaussianMixture(components=2), (table,)),
        (bt.Unfold(*walls), (coords,)),
        (bt.Pipeline([("ns", bt.NormalScore(), "v")]), ({"v": values},)),
        *(
            (step, ({"v": values},))
            for step in (
                bt.RenameColumns(case="upper"),
                bt.Clip(upper=1.0),
                bt.FillNull(0.0),
                bt.Replace({1.0: 2.0}),
                bt.ToNull(-99),
                bt.NormalizeText(),
                bt.DropNull(),
                bt.Scale("v", 2.0),
                bt.Log("v"),
                bt.Log10("v"),
                bt.ToNumber("v"),
            )
        ),
        (bt.DropDuplicates(), (bt.PointSet(coords),)),
        *((cls(), (bt.closure(table),)) for cls in (bt.ALR, bt.CLR, bt.ILR)),
    ]


def test_fit_returns_self():
    cases = fits()
    covered = {type(o).__name__ for o, _ in cases}
    fitted = {name.split(".")[0] for name in CALLABLES if name.endswith(".fit")}
    assert fitted - MODEL_FITS == covered
    for o, args in cases:
        assert o.fit(*args) is o, type(o).__name__


def columns(result):
    return list(result.column_names if hasattr(result, "column_names") else result.keys())


def test_returned_columns_are_american():
    points = bt.PointSet(coords, {"v": values, "rock": np.where(categories > 0, "a", "b")})
    anam = bt.HermiteAnamorphosis(degree=10).fit(values)
    ok = bt.OrdinaryKriging(model, search).fit(coords, values)
    results = [
        bt.swath(coords, values, 20.0, axis="x"),
        bt.contact(
            points,
            "v",
            domain_column="rock",
            holes=np.arange(60) // 10,
            inside="a",
            outside="b",
            max_distance=50.0,
            bin=10.0,
        ),
        bt.capping(values),
        bt.capping_report("v", {"a": 2.0, "b": 2.0}, domain_column="rock", data=points),
        bt.grade_tonnage(values, [0.5, 1.0]),
        bt.describe(values),
        bt.describe_by("v", "rock", data=points),
        bt.neighborhood_stats(grid, coords, values),
        ok.predict(grid, diagnostics=True),
        bt.global_bias(ok.predict(grid), values),
        anam.grade_tonnage([0.5, 1.0]),
        bt.UniformConditioning(anam, 0.8, r_panel=0.6).panel_recovery(1.2, [0.5, 1.0]),
    ]
    found = [c for r in results for c in columns(r) if BRITISH.search(c.lower())]
    assert found == []


@pytest.mark.parametrize("name", ["correlation", "scatter_matrix", "completeness"])
def test_matrix_plots_draw_the_named_columns(name):
    pytest.importorskip("matplotlib").use("Agg")
    import matplotlib.pyplot as plt

    w = rng.uniform(0.5, 2.0, 60)
    points = bt.PointSet(coords, {"a": values, "w": w, "b": table[:, 1], "rock": ["x"] * 60})
    f = getattr(bt.plot, name)
    options = {} if name == "completeness" else {"weights": "w"}
    _, ax = f(points, columns=["a", "b"], **options)
    if name == "correlation":
        np.testing.assert_allclose(ax.images[0].get_array(), bt.correlation(table, weights=w))
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    elif name == "scatter_matrix":
        assert ax.shape == (2, 2) and ax[1, 0].get_xlabel() == "a" and ax[1, 1].get_xlabel() == "b"
        r = bt.correlation(table, weights=w)
        assert ax[1, 0].texts[0].get_text().startswith(f"r {r[1, 0]:.2f}")
    else:
        assert len(ax.patches) == 3
    with pytest.raises(bt.MissingColumn):
        f(points, columns=["a", "c"])
    plt.close("all")


def test_categorical_probabilities_have_one_row_per_target():
    summaries = [
        bt.SIS([model] * 3, search).fit(coords, categories).simulate(grid, n=2),
        bt.Plurigaussian(model, proportions=[0.4, 0.4, 0.2]).fit(coords, categories).simulate(grid, n=2),
        bt.CategoricalIndicatorKriging(model, search).fit(coords, categories).predict(grid),
    ]
    assert [s.probabilities.shape for s in summaries] == [(len(grid), 3)] * 3


def test_continuous_summaries_have_one_row_per_target():
    grades = values - values.min() + 0.1
    options = {"cutoffs": [0.5, 1.0], "quantiles": [0.1, 0.5, 0.9]}
    thresholds = list(np.quantile(grades, [0.25, 0.5, 0.75, 0.9]))
    kriged = [
        bt.MultipleIndicatorKriging(model, search, thresholds).fit(coords, grades).predict(grid, **options),
        bt.MultigaussianKriging(model, search).fit(coords, grades).predict(grid, **options),
    ]
    for s in kriged:
        assert s.probability_above.shape == s.mean_above.shape == (len(grid), 2)
        assert s.quantile_values.shape == (len(grid), 3)
    assert kriged[0].cdf.shape == (len(grid), 4) and kriged[1].cdf.shape == (len(grid), 0)
    simulated = [
        bt.SGS(model, search).fit(coords, grades).simulate(grid, n=3, keep=True, **options),
        bt.TurningBands(model, bands=50).fit(coords, grades).simulate(grid, n=3, keep=True, **options),
    ]
    for s in simulated:
        assert s.probability_above.shape == s.mean_above.shape == (len(grid), 2)
        assert s.quantile_values.shape == (len(grid), 3)
        assert s.realization_above.shape == (3, 2) and s.realizations.shape == (3, len(grid))
