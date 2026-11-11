"""Lines and cards of transforms, the remaining estimators and simulators, and result objects.

Estimators and transforms give their parameters through ``_meta``, which leaves the fitted arrays out, so
showing a fitted object costs the same whatever its size.
"""

import inspect
import json

import numpy as np

from boitata import _boitata as bt
from boitata._display import (
    Card,
    Grid,
    _angles,
    _capped,
    _count,
    _number,
    _plain,
    _ratios,
    _show,
    _spread,
    _triple,
)
from boitata.checks import RealizationCheck
from boitata.pipeline import Pipeline
from boitata.spatial import Topography
from boitata.steps import _Step

BRIEF = 8


def _brief(value):
    """A scalar as it is, a few values in full, and a long array as its length and range."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_brief(v)}" for k, v in list(value.items())[:BRIEF])
    values = np.asarray(value)
    if values.ndim == 0:
        return _number(values.item())
    if values.dtype.kind not in "fiub":
        return ", ".join(str(v) for v in values.ravel()[:BRIEF]) + (", …" if values.size > BRIEF else "")
    if values.size <= BRIEF:
        return ", ".join(_number(v) for v in values.ravel().tolist())
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    return f"{_count(values.size, 'value')}, {_number(lo)} to {_number(hi)}"


def _shape(values):
    return " × ".join(str(n) for n in np.shape(values))


def _json_variogram(v):
    nugget = [f"nugget {_number(v['nugget'])}"] if v["nugget"] else []
    structures = []
    for s in v["structures"]:
        model = s["model"] if isinstance(s["model"], str) else next(iter(s["model"]))
        structures.append(f"{model.lower()}({_number(s['sill'])}, {_number(s['range'])})")
    return " + ".join(nugget + structures)


def _json_anisotropy(v):
    a = v.get("anisotropy")
    if a is None:
        return None
    angles = _angles((a["azimuth"], a["dip"], a["rake"]))
    return f"{angles}; {_ratios((a['semi'], a['minor']))}"


def _json_search(s):
    low, high = s["min_samples"], s["max_samples"]
    samples = _plain(high, "sample") if low == high else f"{low} to {high} samples"
    return f"radius {_number(s['radius'])}, {samples}"


def _passes(search):
    return [] if search is None else search if isinstance(search, list) else [search]


def _parameters(obj):
    meta = json.loads(obj._meta())
    return {**meta, **meta.pop("model", {})}


def _state(rows, count):
    return "not fitted" if rows is None else f"fitted on {count(rows, 'sample')}"


def _options(meta, names):
    """The named parameters that are set, as (name, text)."""
    out = []
    for name in names:
        value = meta.get(name)
        if value is None or value is False:
            continue
        text = "yes" if value is True else value.lower() if isinstance(value, str) else _brief(value)
        out.append((name.replace("_", " "), text))
    return out


def _model(*names):
    """Line and card of an estimator or simulator from its ``_meta``; `names` are the options to show."""

    def variograms(meta):
        return [meta["variogram"]] if "variogram" in meta else meta.get("variograms", [])

    def line(obj):
        meta = _parameters(obj)
        models, passes = variograms(meta), _passes(meta.get("search"))
        parts = [_state(meta["rows"], _plain)]
        if models:
            parts.append(_json_variogram(models[0]) if len(models) == 1 else f"{len(models)} variograms")
        if passes:
            parts.append(_json_search(passes[0]) if len(passes) == 1 else f"{len(passes)} search passes")
        parts += [f"{name} {text}" if text != "yes" else name for name, text in _options(meta, names)]
        return f"{type(obj).__name__}({', '.join(parts)})"

    def card(obj):
        meta = _parameters(obj)
        models, passes = variograms(meta), _passes(meta.get("search"))
        facts = []
        for i, model in enumerate(models[:BRIEF], 1):
            label = "variogram" if len(models) == 1 else f"variogram {i}"
            facts.append((label, _json_variogram(model)))
        if models and all(_json_anisotropy(m) == _json_anisotropy(models[0]) for m in models):
            facts.append(("anisotropy", _json_anisotropy(models[0])))
        for i, search in enumerate(passes, 1):
            facts.append(("search" if len(passes) == 1 else f"search {i}", _json_search(search)))
        facts += _options(meta, names)
        if len(models) > BRIEF:
            facts.append(("more variograms", len(models) - BRIEF))
        return Card(type(obj).__name__, _state(meta["rows"], _count), facts)

    return line, card


_DEFAULTS = {"GaussianImputer": {"components": 1}, "KernelDensity": {"bandwidth": "silverman"}}


def _transform(**fitted):
    """Line and card of a transform from its ``_meta``; `fitted` names what to show once it is fitted."""

    def given(obj):
        meta = json.loads(obj._meta())
        defaults = {name: p.default for name, p in inspect.signature(type(obj)).parameters.items()}
        defaults |= _DEFAULTS.get(type(obj).__name__, {})
        state = meta.pop("fitted")
        return state, [
            (k, v) for k, v in meta.items() if v is not None and v is not False and v != defaults.get(k)
        ]

    def line(obj):
        state, parameters = given(obj)
        parts = ["fitted" if state else "not fitted"] + [f"{k}={v!r}" for k, v in parameters]
        return f"{type(obj).__name__}({', '.join(parts)})"

    def card(obj):
        state, parameters = given(obj)
        facts = [(k.strip("_").replace("_", " "), _brief(v)) for k, v in parameters]
        if state:
            facts += [(label, _brief(read(obj))) for label, read in fitted.items()]
        return Card(type(obj).__name__, "fitted" if state else "not fitted", facts)

    return line, card


def declustering_line(d):
    weights = _plain(len(d.weights), "weight")
    return f"Declustering({weights}, mean {_number(d.mean)}, cell size {_number(d.cell_size)})"


def declustering_card(d):
    facts = [
        ("declustered mean", d.mean),
        ("cell size", d.cell_size),
        ("weights", _brief(d.weights)),
        ("sizes tried", len(d.sizes) or None),
    ]
    return Card("Declustering", _count(len(d.weights), "weight"), facts)


def categories_line(c):
    options = [
        f"colors={list(c.colors)!r}" if c.colors else None,
        f"mapping={c.mapping!r}" if c.mapping else None,
        f"other={c.other!r}" if c.other is not None else None,
    ]
    return f"Categories({list(c.names)!r}{''.join(f', {o}' for o in options if o)})"


def categories_card(c):
    names, colors = list(c.names), c.colors
    merged = {}
    for source, target in (c.mapping or {}).items():
        merged.setdefault(target, []).append(str(source))
    header = ("code", "name") + (("color",) if colors else ()) + (("from",) if merged else ())
    rows = []
    for code, name in enumerate(names):
        row = (code, name) + ((colors[code],) if colors else ())
        rows.append(row + ((", ".join(merged.get(name, [])),) if merged else ()))
    rows, note = _capped(rows, "category")
    return Card(
        "Categories",
        _count(len(names), "category", "categories"),
        [("other", c.other)],
        [Grid(header, rows, note=note)],
    )


def local_grid_line(g):
    options = [
        f"scale={g.scale!r}" if g.scale != 1.0 else None,
        f"rotation={g.rotation!r}" if g.rotation else None,
        f"shift={tuple(g.shift)!r}" if any(g.shift) else None,
        f"z_shift={g.z_shift!r}" if g.z_shift else None,
        f"crs={g.crs!r}" if g.crs is not None else None,
    ]
    return f"LocalGrid({', '.join(o for o in options if o)})"


def local_grid_card(g):
    facts = [
        ("scale", g.scale),
        ("rotation", f"{_number(g.rotation)}°"),
        ("shift", ", ".join(_number(v) for v in g.shift)),
        ("z shift", g.z_shift),
        ("crs", g.crs),
    ]
    return Card("LocalGrid", "mine grid to projected coordinates", facts)


def _coregionalization_summary(c):
    return f"{_plain(c.nvar, 'variable')}, {_plain(len(c.structures), 'structure')}"


def coregionalization_line(c):
    return f"Coregionalization({_coregionalization_summary(c)})"


def coregionalization_card(c):
    isotropic = tuple(c.ratios) == (1.0, 1.0)
    facts = [
        ("rotation", None if isotropic and not any(c.rotation) else _angles(c.rotation)),
        ("ratios", None if isotropic else _ratios(c.ratios)),
    ]
    rows = [("nugget", None, _brief(np.diag(c.nugget)))]
    rows += [(model, range_, _brief(np.diag(sills))) for model, range_, sills in c.structures]
    grid = Grid(("structure", "range", "sills of each variable"), rows)
    return Card("Coregionalization", _coregionalization_summary(c), facts, [grid])


def transiogram_line(t):
    meta = json.loads(t.to_json())
    return f"Transiogram({meta['proportions']!r}, {meta['range']!r})"


def transiogram_card(t):
    meta = json.loads(t.to_json())
    summary = _count(len(meta["proportions"]), "category", "categories")
    return Card(
        "Transiogram", summary, [("proportions", _brief(meta["proportions"])), ("range", meta["range"])]
    )


def _uniform_conditioning(u):
    meta = json.loads(u.to_json())
    return [("r smu", meta["r_smu"]), ("panel", "given" if meta["panel"] is not None else None)]


def uniform_conditioning_line(u):
    meta = json.loads(u.to_json())
    return (
        f"UniformConditioning(r_smu={meta['r_smu']!r}{', panel given' if meta['panel'] is not None else ''})"
    )


def uniform_conditioning_card(u):
    return Card("UniformConditioning", "", _uniform_conditioning(u))


def _trend_kind(t):
    if t.categories is not None:
        return f"{_plain(len(t.categories), 'category', 'categories')}"
    if t.degree is not None:
        return f"polynomial of degree {t.degree}"
    return "moving window"


def trend_line(t):
    return f"Trend({_trend_kind(t)})"


def trend_card(t):
    facts = [
        ("degree", t.degree),
        ("bandwidth", t.bandwidth),
        ("bandwidths", None if t.bandwidths is None or len(t.bandwidths) < 2 else _brief(t.bandwidths)),
        ("categories", None if t.categories is None else _brief(t.categories)),
        ("coefficients", None if t.coefficients is None else _brief(t.coefficients)),
    ]
    return Card("Trend", _trend_kind(t), facts)


def _domains_summary(d, count):
    names = d.scheme.names
    return f"{count(len(names), 'domain')}, {count(len(d.labels), 'sample')}"


def domains_line(d):
    return f"Domains({_domains_summary(d, _plain)})"


def domains_card(d):
    labels = d.labels
    names = list(d.scheme.names)
    facts = [("wcss", d.wcss), ("spatial entropy", d.spatial_entropy)]
    rows = [(name, int(np.sum(labels == code))) for code, name in enumerate(names)]
    rows, note = _capped(rows, "domain")
    return Card("Domains", _domains_summary(d, _count), facts, [Grid(("domain", "samples"), rows, note=note)])


def _mesh_report_summary(r, count):
    summary = r.summary
    closed = "closed" if summary["is_closed"] else "open"
    return f"{count(len(r.problems), 'problem')}, {count(summary['shells'], 'shell')}, {closed}"


def mesh_report_line(r):
    return f"MeshReport({_mesh_report_summary(r, _plain)})"


def mesh_report_card(r):
    facts = [(k.replace("_", " "), v) for k, v in r.summary.items() if k not in ("shells", "is_closed")]
    return Card("MeshReport", _mesh_report_summary(r, _count), facts)


def block_model_file_line(f):
    return f"BlockModelFile({_plain(len(f), 'block')}, {_triple(f.count)} of {_triple(f.size)}, {_plain(len(f.column_names), 'column')})"


def block_model_file_card(f):
    facts = [
        ("count", _triple(f.count)),
        ("size", _triple(f.size)),
        ("origin", ", ".join(_number(v) for v in f.origin)),
        ("rotation", _angles(f.rotation) if any(f.rotation) else None),
        ("crs", f.crs),
        ("columns", _brief(f.column_names)),
    ]
    return Card("BlockModelFile", _count(len(f), "block"), facts)


def local_anisotropy_line(a):
    return f"LocalAnisotropy({_plain(len(a), 'location')})"


def local_anisotropy_card(a):
    angles, ratios = np.asarray(a.angles), np.asarray(a.ratios)
    rows = [(name, *_spread(angles[:, i])) for i, name in enumerate(("azimuth", "dip", "rake"))]
    rows += [(name, *_spread(ratios[:, i])) for i, name in enumerate(("semi/major", "minor/major"))]
    facts = [("scales", None if a.scales is None else "given")]
    grid = Grid(("over the locations", "min", "mean", "max"), [r for r in rows if len(r) > 1])
    return Card("LocalAnisotropy", _count(len(a), "location"), facts, [grid])


def _values(numbers):
    return ", ".join(_number(v) for v in numbers) or None


def _indicator_summary(s, count):
    return f"{count(len(s.mean), 'row')}, {count(len(s.thresholds), 'threshold')}"


def indicator_summary_line(s):
    return f"IndicatorSummary({_indicator_summary(s, _plain)})"


def indicator_summary_card(s):
    facts = [
        ("thresholds", _values(s.thresholds)),
        ("cutoffs", _values(s.cutoffs)),
        ("quantiles", _values(s.quantiles)),
    ]
    rows = [("mean", *_spread(s.mean)), ("std", *_spread(s.std))]
    grid = Grid(("over the rows", "min", "mean", "max"), [r for r in rows if len(r) > 1])
    return Card("IndicatorSummary", _indicator_summary(s, _count), facts, [grid] if grid.rows else [])


def _categorical_indicator_summary(s, count):
    return f"{count(len(s.correction), 'row')}, {count(len(s.names), 'category', 'categories')}"


def categorical_indicator_summary_line(s):
    return f"CategoricalIndicatorSummary({_categorical_indicator_summary(s, _plain)})"


def categorical_indicator_summary_card(s):
    rows, note = _capped(list(zip(s.names, map(float, s.proportions))), "category")
    grid = Grid(("category", "proportion"), rows, note=note)
    return Card("CategoricalIndicatorSummary", _categorical_indicator_summary(s, _count), grids=[grid])


def _categorical_summary(s, count):
    rows, categories = len(s.most_likely), np.shape(s.proportions)[1]
    return f"{count(s.n, 'realization')}, {count(rows, 'row')}, {count(categories, 'category', 'categories')}"


def categorical_summary_line(s):
    return f"CategoricalSummary({_categorical_summary(s, _plain)})"


def categorical_summary_card(s):
    proportions = np.asarray(s.proportions)
    rows = list(
        zip(
            range(proportions.shape[1]),
            *(map(float, f(proportions, axis=0)) for f in (np.min, np.mean, np.max)),
        )
    )
    rows, note = _capped(rows, "category")
    grid = Grid(("code", "min", "mean", "max"), rows, "proportion over the realizations", note)
    return Card("CategoricalSummary", _categorical_summary(s, _count), grids=[grid])


def variogram_map_line(m):
    return f"VariogramMap({_plain(len(m.angles), 'angle')} × {_plain(len(m.lags), 'lag')})"


def variogram_map_card(m):
    facts = [
        ("angles", _brief(m.angles)),
        ("lags", _brief(m.lags)),
        ("pairs", f"{int(np.nansum(m.counts)):,}"),
        ("ranges", _brief(m.ranges)),
    ]
    return Card("VariogramMap", f"{_count(len(m.angles), 'angle')} × {_count(len(m.lags), 'lag')}", facts)


def _variogram_volume_summary(v, count):
    return f"{count(len(v.directions), 'direction')} × {count(len(v.lags), 'lag')}"


def variogram_volume_line(v):
    return f"VariogramVolume({_variogram_volume_summary(v, _plain)})"


def variogram_volume_card(v):
    facts = [
        ("rotation", _angles(v.rotation)),
        ("ratios", _ratios(v.ratios)),
        ("ranges", _brief(v.ranges)),
        ("lags", _brief(v.lags)),
        ("pairs", f"{int(np.nansum(v.counts)):,}"),
    ]
    return Card("VariogramVolume", _variogram_volume_summary(v, _count), facts)


def _step_options(s):
    """The options of a step that differ from the defaults of its constructor."""
    defaults = {name: p.default for name, p in inspect.signature(type(s)).parameters.items()}
    empty = (None, {}, [], ())
    return [(k, v) for k, v in vars(s).items() if v not in empty and v != defaults.get(k)]


def step_line(s):
    return f"{type(s).__name__}({', '.join(f'{k}={v!r}' for k, v in _step_options(s))})"


def step_card(s):
    return Card(type(s).__name__, "", [(k.replace("_", " "), _brief(v)) for k, v in _step_options(s)])


def _pipeline_state(p):
    return "fitted" if p.fitted_ else "not fitted"


def pipeline_line(p):
    return f"Pipeline({', '.join([_pipeline_state(p)] + [name for name, _, _ in p.steps])})"


def pipeline_card(p):
    rows = []
    for name, step, columns in p.steps:
        on = "" if columns is None else columns if isinstance(columns, str) else ", ".join(columns)
        rows.append((name, repr(step), on))
    summary = f"{_count(len(p.steps), 'step')}, {_pipeline_state(p)}"
    return Card("Pipeline", summary, grids=[Grid(("step", "transform", "columns"), rows)])


def realization_check_line(c):
    kind = "categorical" if c.categorical else "continuous"
    return f"RealizationCheck({kind}, {_brief(c.names)})"


def realization_check_card(c):
    kind = "categorical" if c.categorical else "continuous"
    facts = [
        ("names", _brief(c.names)),
        ("statistics", repr(c.statistics)),
        ("directions", None if c.directions is None else _count(len(c.directions), "direction")),
    ]
    return Card("RealizationCheck", kind, facts)


def topography_line(t):
    flagged = int(np.sum(t.flagged))
    return f"Topography({_plain(len(t.grid), 'block')}, {_plain(flagged, 'flagged point')})"


def topography_card(t):
    residuals = np.abs(np.asarray(t.residuals, dtype=float))
    facts = [
        ("grid", repr(t.grid)),
        ("mesh", repr(t.mesh)),
        ("outline", None if t.outline is None else repr(t.outline)),
        ("largest residual", float(np.nanmax(residuals)) if residuals.size else None),
        ("residual limit", t.max_residual),
    ]
    return Card("Topography", _count(int(np.sum(t.flagged)), "flagged point"), facts)


for _cls, (_line, _card) in (
    (bt.DualKriging, _model("degree")),
    (bt.MultigaussianKriging, _model("tails")),
    (bt.MultipleIndicatorKriging, _model("thresholds", "simple", "interpolation")),
    (bt.CategoricalIndicatorKriging, _model("categories", "simple")),
    (bt.Cokriging, _model("kind", "means")),
    (bt.DisjunctiveKriging, _model("order")),
    (bt.SIS, _model()),
    (bt.Plurigaussian, _model()),
    (bt.NormalScore, _transform()),
    (bt.Capping, _transform(caps=lambda c: c.caps_, **{"metal removed": lambda c: c.metal_removed_})),
    (bt.BoxCox, _transform(**{"lambda": lambda b: b.lambda_})),
    (bt.HermiteAnamorphosis, _transform(mean=lambda a: a.mean_, variance=lambda a: a.variance_)),
    (bt.KernelDensity, _transform(**{"fitted bandwidth": lambda k: k.bandwidth_})),
    (bt.GaussianMixture, _transform(proportions=lambda g: g.proportions_)),
    (bt.PCA, _transform(**{"explained variance ratio": lambda p: p.explained_variance_ratio_})),
    (bt.MAF, _transform()),
    (bt.PPMT, _transform()),
    (bt.StepwiseConditional, _transform()),
    (bt.GaussianImputer, _transform()),
    (bt.Declustering, (declustering_line, declustering_card)),
    (bt.Categories, (categories_line, categories_card)),
    (bt.LocalGrid, (local_grid_line, local_grid_card)),
    (bt.Coregionalization, (coregionalization_line, coregionalization_card)),
    (bt.Transiogram, (transiogram_line, transiogram_card)),
    (bt.UniformConditioning, (uniform_conditioning_line, uniform_conditioning_card)),
    (bt.Trend, (trend_line, trend_card)),
    (bt.Domains, (domains_line, domains_card)),
    (bt.MeshReport, (mesh_report_line, mesh_report_card)),
    (bt.BlockModelFile, (block_model_file_line, block_model_file_card)),
    (bt.LocalAnisotropy, (local_anisotropy_line, local_anisotropy_card)),
    (bt.IndicatorSummary, (indicator_summary_line, indicator_summary_card)),
    (
        bt.CategoricalIndicatorSummary,
        (categorical_indicator_summary_line, categorical_indicator_summary_card),
    ),
    (bt.CategoricalSummary, (categorical_summary_line, categorical_summary_card)),
    (bt.VariogramMap, (variogram_map_line, variogram_map_card)),
    (bt.VariogramVolume, (variogram_volume_line, variogram_volume_card)),
    (_Step, (step_line, step_card)),
    (Pipeline, (pipeline_line, pipeline_card)),
    (RealizationCheck, (realization_check_line, realization_check_card)),
    (Topography, (topography_line, topography_card)),
):
    _show(_cls, _line, _card)
