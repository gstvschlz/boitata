"""Notebook HTML for boitata objects: a title, key facts and the columns."""

from html import escape

import numpy as np

from boitata import _boitata as bt
from boitata.errors import BoitataError
from boitata.estimation import CategoricalCrossValidation, CrossValidation, _Base

_STYLE = (
    "<style>.bt-repr{font-family:var(--jp-code-font-family,monospace);font-size:12px;color:inherit}"
    ".bt-repr table{border-collapse:collapse;margin:4px 0}"
    ".bt-repr th,.bt-repr td{padding:1px 10px 1px 0;text-align:left;vertical-align:top;"
    "border-bottom:1px solid rgba(127,127,127,.25)}"
    ".bt-repr th{font-weight:600}.bt-repr b{font-size:13px}</style>"
)


def _text(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_text(v) for v in value) + ")"
    return str(value)


def _facts(pairs):
    rows = "".join(
        f"<tr><th>{escape(k)}</th><td>{escape(_text(v))}</td></tr>" for k, v in pairs if v is not None
    )
    return f"<table>{rows}</table>"


def _grid(header, rows):
    head = "".join(f"<th>{escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{escape(_text(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _columns(table, title="columns"):
    if not table.column_names:
        return ""
    units = table.units
    rows = [(n, t, units.get(n, "")) for n, t in zip(table.column_names, table._dtypes())]
    return f"<div>{title}</div>" + _grid(("name", "type", "unit"), rows)


def _page(title, *parts):
    return f'{_STYLE}<div class="bt-repr"><b>{escape(title)}</b>{"".join(parts)}</div>'


def _bounds(obj):
    b = obj.bounds
    return None if b is None else f"{_text(tuple(b[0]))} to {_text(tuple(b[1]))}"


def table(t):
    head = t.filter(np.arange(len(t)) < 5)
    rows = list(zip(*(head[c].tolist() for c in t.column_names)))
    preview = _grid(t.column_names, rows) if t.column_names else ""
    return _page(f"Table: {len(t):,} rows", _columns(t), "<div>first rows</div>" + preview if rows else "")


def point_set(p):
    facts = _facts(
        [("points", f"{len(p):,}"), ("crs", p.crs), ("length unit", p.length_unit), ("bounds", _bounds(p))]
    )
    return _page("PointSet", facts, _columns(p.attributes))


def polylines(p):
    facts = _facts(
        [
            ("features", f"{len(p):,}"),
            ("parts", f"{len(p.parts):,}"),
            ("crs", p.crs),
            ("length unit", p.length_unit),
            ("bounds", _bounds(p)),
        ]
    )
    return _page("Polylines", facts, _columns(p.attributes))


def block_model(m):
    layout = "sub-blocked" if m.extents is not None else "regular" if m.index is None else "masked"
    facts = _facts(
        [
            ("layout", layout),
            ("rows", f"{len(m):,} of {int(np.prod(m.count)):,} cells"),
            ("origin", m.origin),
            ("size", m.size),
            ("count", m.count),
            ("rotation", m.rotation),
            ("crs", m.crs),
            ("length unit", m.length_unit),
            ("bounds", _bounds(m)),
        ]
    )
    return _page("BlockModel", facts, _columns(m.attributes))


def mesh(m):
    facts = _facts(
        [
            ("vertices", f"{len(m):,}"),
            ("triangles", f"{len(m.triangles):,}"),
            ("closed", m.is_closed),
            ("crs", m.crs),
            ("length unit", m.length_unit),
            ("bounds", _bounds(m)),
        ]
    )
    return _page(
        "Mesh",
        facts,
        _columns(m.vertex_attributes, "vertex columns"),
        _columns(m.face_attributes, "face columns"),
    )


def drillholes(d):
    samples = d.samples() if d.interval_columns else None
    facts = _facts(
        [
            ("holes", f"{len(d):,}"),
            ("intervals", None if samples is None else f"{len(samples):,}"),
            ("length unit", d.length_unit),
            ("bounds", _bounds(d)),
        ]
    )
    return _page(
        "Drillholes", facts, "" if samples is None else _columns(samples.attributes, "interval columns")
    )


def variogram(v):
    facts = _facts(
        [
            ("nugget", v.nugget),
            ("sill", v.sill),
            ("rotation", v.rotation),
            ("ratios", v.ratios),
            ("length unit", v.length_unit),
        ]
    )
    structures = _grid(("model", "sill", "range"), [(s.model, s.sill, s.range) for s in v.structures])
    return _page("Variogram", facts, structures)


def estimator(e):
    try:
        fitted = f"{len(e._engine.values):,} samples"
    except BoitataError:
        fitted = "not fitted"
    variogram = e._engine.variogram
    facts = _facts(
        [
            ("fitted", fitted),
            ("unit", e.unit),
            ("search", None if getattr(e, "_search", None) is None else repr(e._search)),
            ("variogram", None if variogram is None else repr(variogram)),
        ]
    )
    return _page(type(e).__name__, facts)


def cross_validation(cv):
    facts = _facts(
        [
            ("samples", f"{len(cv.actual):,}"),
            ("estimated", f"{int(np.sum(~np.isnan(cv.estimate))):,}"),
            ("mean error", cv.mean_error),
            ("rmse", cv.rmse),
            ("correlation", cv.correlation),
            ("slope", cv.slope),
            ("standardized squared error", cv.standardized_squared_error),
        ]
    )
    return _page(type(cv).__name__, facts)


def categorical_cross_validation(cv):
    brier = dict(zip(cv.names, cv.brier.tolist()))
    return _page(
        type(cv).__name__,
        _facts([("samples", f"{len(cv.actual):,}"), ("categories", cv.names)]),
        _grid(("category", "brier"), brier.items()),
    )


def simulation_summary(s):
    facts = _facts(
        [
            ("realizations", s.n),
            ("rows", f"{len(s.mean):,}"),
            ("groups", s.groups),
            ("cutoffs", s.cutoffs or None),
            ("quantiles", s.quantiles or None),
            ("tolerances", s.tolerances or None),
        ]
    )
    return _page("SimulationSummary", facts)


for _cls, _render in (
    (bt.Table, table),
    (bt.PointSet, point_set),
    (bt.Polylines, polylines),
    (bt.BlockModel, block_model),
    (bt.Mesh, mesh),
    (bt.Drillholes, drillholes),
    (bt.Variogram, variogram),
    (_Base, estimator),
    (CrossValidation, cross_validation),
    (CategoricalCrossValidation, categorical_cross_validation),
    (bt.SimulationSummary, simulation_summary),
):
    _cls._repr_html_ = _render
