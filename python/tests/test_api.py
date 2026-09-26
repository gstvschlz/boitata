import inspect
import re

import ceres as cs
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
    for prefix, module in [("", cs), ("plot.", cs.plot), ("plot3d.", cs.plot3d)]:
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
        target = {"plot.scatter_matrix": "axes"}.get(name, "plotter" if name.startswith("plot3d.") else "ax")
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
model = cs.Variogram([("spherical", 1.0, 40.0)])
search = cs.Search(60.0, max_samples=12)


def fits():
    anam = cs.HermiteAnamorphosis(degree=10).fit(values)
    lmc = cs.Coregionalization(
        [[0.0, 0.0], [0.0, 0.0]], structures=[("spherical", 40.0, [[1.0, 0.5], [0.5, 1.0]])]
    )
    return [
        (cs.OrdinaryKriging(model, search), (coords, values)),
        (cs.SimpleKriging(model, search), (coords, values)),
        (cs.IndicatorKriging(model, search, threshold=1.0), (coords, values)),
        (cs.UniversalKriging(model, search), (coords, values)),
        (cs.FactorialKriging(model, search, [0]), (coords, values)),
        (cs.BlockKriging(model, search, (5.0, 5.0, 5.0)), (coords, values)),
        (cs.BayesianKriging(model, search, [1.0], [0.5]), (coords, values)),
        (cs.InverseDistance(search), (coords, values)),
        (cs.NearestNeighbor(search), (coords, values)),
        (cs.MovingAverage(search), (coords, values)),
        (cs.MovingMedian(search), (coords, values)),
        (cs.LocalLeastSquares(search), (coords, values)),
        (cs.DualKriging(model), (coords, values)),
        (cs.Cokriging(lmc, search), (coords, values, np.arange(60) % 2)),
        (cs.DisjunctiveKriging(anam, model, search), (coords, values)),
        (cs.MultipleIndicatorKriging(model, search, [0.8, 1.5]), (coords, values)),
        (cs.SGS(model, search), (coords, values)),
        (cs.TurningBands(model, bands=20), (coords, values)),
        (cs.SIS([model] * 3, search), (coords, categories)),
        (cs.Plurigaussian(model, proportions=[0.4, 0.4, 0.2]), (coords, categories)),
        (cs.MultivariateSimulation(cs.PCA(), [cs.SGS(model, search)] * 2), (coords, table)),
        (cs.ImplicitModel(), (coords, values - values.mean())),
        (cs.NormalScore(), (values,)),
        (cs.HermiteAnamorphosis(degree=10), (values,)),
        (cs.BoxCox(), (values,)),
        (cs.PPMT(iterations=2), (table,)),
        (cs.PCA(), (table,)),
        (cs.MAF(lag=1.0, tolerance=0.01), (rng.normal(size=(64, 2)), grid)),
        (cs.StepwiseConditional(classes=3), (table,)),
        (cs.GaussianImputer(), (table,)),
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
    points = cs.PointSet(coords, {"v": values, "rock": np.where(categories > 0, "a", "b")})
    anam = cs.HermiteAnamorphosis(degree=10).fit(values)
    ok = cs.OrdinaryKriging(model, search).fit(coords, values)
    results = [
        cs.swath(coords, values, 20.0, axis="x"),
        cs.contact(
            points,
            "v",
            domain_column="rock",
            holes=np.arange(60) // 10,
            inside="a",
            outside="b",
            max_distance=50.0,
            bin=10.0,
        ),
        cs.capping(values),
        cs.capping_report("v", {"a": 2.0, "b": 2.0}, domain_column="rock", data=points),
        cs.grade_tonnage(values, [0.5, 1.0]),
        cs.describe(values),
        cs.describe_by("v", "rock", data=points),
        cs.neighborhood_stats(grid, coords, values),
        ok.predict(grid, diagnostics=True),
        cs.global_bias(ok.predict(grid), values),
        anam.grade_tonnage([0.5, 1.0]),
        cs.UniformConditioning(anam, 0.8, r_panel=0.6).panel_recovery(1.2, [0.5, 1.0]),
    ]
    found = [c for r in results for c in columns(r) if BRITISH.search(c.lower())]
    assert found == []


@pytest.mark.parametrize("name", ["correlation", "scatter_matrix", "completeness"])
def test_matrix_plots_draw_the_named_columns(name):
    pytest.importorskip("matplotlib").use("Agg")
    import matplotlib.pyplot as plt

    w = rng.uniform(0.5, 2.0, 60)
    points = cs.PointSet(coords, {"a": values, "w": w, "b": table[:, 1], "rock": ["x"] * 60})
    f = getattr(cs.plot, name)
    options = {} if name == "completeness" else {"weights": "w"}
    _, ax = f(points, columns=["a", "b"], **options)
    if name == "correlation":
        np.testing.assert_allclose(ax.images[0].get_array(), cs.correlation(table, weights=w))
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    elif name == "scatter_matrix":
        assert ax.shape == (2, 2) and ax[1, 0].get_xlabel() == "a" and ax[1, 1].get_xlabel() == "b"
        r = cs.correlation(table, weights=w)
        assert ax[1, 0].texts[0].get_text().startswith(f"r {r[1, 0]:.2f}")
    else:
        assert len(ax.patches) == 3
    with pytest.raises(cs.MissingColumn):
        f(points, columns=["a", "c"])
    plt.close("all")
