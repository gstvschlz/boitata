"""How boitata objects show themselves.

Each class gives a one-line summary and a card (title, key facts, tables). ``repr`` is the line, ``print`` the
card as text and a notebook the same card as HTML.
"""

import inspect
from dataclasses import dataclass, field
from html import escape

import numpy as np

from boitata import _boitata as bt
from boitata.errors import BoitataError
from boitata.estimation import (
    CategoricalCrossValidation,
    CrossValidation,
    IndicatorCrossValidation,
    _Base,
)

STATISTICS_ROWS = 1_000_000
SHOWN_ROWS = 30

_STYLE = (
    "<style>.bt-repr{font-family:var(--jp-code-font-family,monospace);font-size:12px;color:inherit}"
    ".bt-repr table{border-collapse:collapse;margin:4px 0}"
    ".bt-repr th,.bt-repr td{padding:1px 10px 1px 0;text-align:left;vertical-align:top;"
    "border-bottom:1px solid rgba(127,127,127,.25)}"
    ".bt-repr th{font-weight:600}.bt-repr b{font-size:13px}.bt-repr .bt-n{text-align:right}"
    ".bt-repr .bt-note{opacity:.7}</style>"
)
_TYPES = {"Float64": "float64", "Utf8": "str", "LargeUtf8": "str", "Utf8View": "str", "Boolean": "bool"}


@dataclass
class Grid:
    """A table of a card. A row shorter than the header ends in a cell spanning the remaining columns."""

    header: tuple
    rows: list
    caption: str | None = None
    note: str | None = None


@dataclass
class Card:
    title: str
    summary: str = ""
    facts: list = field(default_factory=list)
    grids: list = field(default_factory=list)


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    value = float(value)
    size = abs(value)
    if np.isnan(value):
        return "nan"
    if size == 0:
        return "0"
    if size >= 1e15 or size < 1e-4:
        return f"{value:.3g}"
    if size >= 1000:
        return f"{value:.1f}".removesuffix(".0")
    return f"{value:.4g}"


def _is_number(cell):
    return isinstance(cell, (int, float, np.integer, np.floating)) and not isinstance(cell, bool)


def _cell(cell):
    if cell is None:
        return ""
    return _number(cell) if _is_number(cell) or isinstance(cell, (bool, np.bool_)) else str(cell)


def _count(n, noun, plural=None):
    return f"{n:,} {noun if n == 1 else plural or noun + 's'}"


def _plain(n, noun, plural=None):
    return f"{n} {noun if n == 1 else plural or noun + 's'}"


def _with_unit(text, unit):
    return f"{text} {unit}" if unit else text


def _numeric_columns(grid):
    width = len(grid.header)
    numeric = []
    for i in range(width):
        cells = [
            row[i] for row in grid.rows if i < len(row) - (len(row) < width) and row[i] not in (None, "")
        ]
        numeric.append(bool(cells) and all(_is_number(c) for c in cells))
    return numeric


def _grid_text(grid, indent):
    width = len(grid.header)
    numeric = _numeric_columns(grid)
    rows = [[_cell(c) for c in row] for row in grid.rows]
    widths = [
        max(
            [len(h)]
            + [len(row[i]) for row in rows if i < len(row) and (len(row) == width or i < len(row) - 1)]
        )
        for i, h in enumerate(grid.header)
    ]

    def line(cells):
        out = []
        for i, cell in enumerate(cells):
            spans = len(cells) < width and i == len(cells) - 1
            out.append(cell if spans else cell.rjust(widths[i]) if numeric[i] else cell.ljust(widths[i]))
        return (indent + "  ".join(out)).rstrip()

    lines = [f"{indent}{grid.caption}"] if grid.caption else []
    lines += [line(list(grid.header))] + [line(row) for row in rows]
    if grid.note:
        lines.append(f"{indent}{grid.note}")
    return lines


def _text(card):
    lines = [f"{card.title}  {card.summary}".rstrip()]
    facts = [(k, v) for k, v in card.facts if v is not None]
    width = max((len(k) for k, _ in facts), default=0)
    lines += [f"  {k:<{width}}  {_cell(v)}" for k, v in facts]
    for grid in card.grids:
        lines += _grid_text(grid, "  ")
    return "\n".join(lines)


def _grid_html(grid):
    width = len(grid.header)
    numeric = _numeric_columns(grid)
    head = "".join(f"<th>{escape(h)}</th>" for h in grid.header)
    body = ""
    for row in grid.rows:
        cells = ""
        for i, cell in enumerate(row):
            if len(row) < width and i == len(row) - 1:
                cells += f'<td colspan="{width - i}">{escape(_cell(cell))}</td>'
            else:
                kind = ' class="bt-n"' if numeric[i] else ""
                cells += f"<td{kind}>{escape(_cell(cell))}</td>"
        body += f"<tr>{cells}</tr>"
    caption = f"<div>{escape(grid.caption)}</div>" if grid.caption else ""
    note = f'<div class="bt-note">{escape(grid.note)}</div>' if grid.note else ""
    return f"{caption}<table><tr>{head}</tr>{body}</table>{note}"


def _html(card):
    facts = "".join(
        f"<tr><th>{escape(k)}</th><td>{escape(_cell(v))}</td></tr>" for k, v in card.facts if v is not None
    )
    return (
        f'{_STYLE}<div class="bt-repr"><b>{escape(card.title)}</b> {escape(card.summary)}'
        f"{f'<table>{facts}</table>' if facts else ''}{''.join(_grid_html(g) for g in card.grids)}</div>"
    )


def _capped(rows, noun):
    """The first rows and a note counting the ones left out."""
    hidden = len(rows) - SHOWN_ROWS
    return (rows, None) if hidden <= 0 else (rows[:SHOWN_ROWS], f"… {_count(hidden, f'more {noun}')}")


def _column_row(values, statistics):
    if not statistics:
        return ()
    if values.dtype.kind == "b":
        return (0, _count(int(values.sum()), "true", "true"))
    if values.dtype.kind in "fiu":
        values = values.astype(float, copy=False)
        nulls = int(np.isnan(values).sum())
        if nulls == len(values):
            return (nulls, None, None, None)
        return (nulls, float(np.nanmin(values)), float(np.nanmean(values)), float(np.nanmax(values)))
    present = [v for v in values.tolist() if v is not None]
    unique = list(dict.fromkeys(present))
    shown = ", ".join(str(v) if len(str(v)) <= 16 else str(v)[:15] + "…" for v in unique[:3])
    more = ", …" if len(unique) > 3 else ""
    return (len(values) - len(present), f"{_count(len(unique), 'unique', 'unique')}: {shown}{more}")


def _columns(table, caption=None):
    names = table.column_names
    if not names:
        return []
    statistics = 0 < len(table) <= STATISTICS_ROWS
    units = table.units
    types = [_TYPES.get(t, t.lower()) for t in table._dtypes()]
    rows = [
        (name, kind, units.get(name, ""), *_column_row(table[name], statistics))
        for name, kind in list(zip(names, types))[:SHOWN_ROWS]
    ]
    notes = []
    if len(names) > SHOWN_ROWS:
        notes.append(f"… {_count(len(names) - SHOWN_ROWS, 'more column')}")
    if len(table) > STATISTICS_ROWS:
        notes.append(f"statistics skipped above {STATISTICS_ROWS:,} rows")
    numbers = any(len(row) == 7 for row in rows)
    described = ("nulls", "min", "mean", "max") if numbers else ("nulls", "values")
    header = ("column", "type", "unit") + (described if statistics else ())
    return [Grid(header, rows, caption, "; ".join(notes) or None)]


def _bounds(obj):
    bounds = obj.bounds
    if bounds is None:
        return None
    return ", ".join(f"{axis} {_number(lo)} to {_number(hi)}" for axis, lo, hi in zip("xyz", *bounds))


def _triple(values, unit=None):
    return _with_unit(" × ".join(_number(v) for v in values), unit)


def _angles(rotation):
    return ", ".join(f"{name} {_number(a)}°" for name, a in zip(("azimuth", "dip", "rake"), rotation))


def _ratios(ratios):
    return f"semi/major {_number(ratios[0])}, minor/major {_number(ratios[1])}"


def _located(obj):
    return [("crs", obj.crs), ("length unit", obj.length_unit), ("bounds", _bounds(obj))]


def table_line(t):
    return f"Table({_plain(len(t), 'row')} × {_plain(len(t.column_names), 'column')})"


def table_card(t):
    summary = f"{_count(len(t), 'row')} × {_count(len(t.column_names), 'column')}"
    return Card("Table", summary, grids=_columns(t))


def point_set_line(p):
    return f"PointSet({_plain(len(p), 'point')}, {_plain(len(p.column_names), 'column')})"


def point_set_card(p):
    return Card("PointSet", _count(len(p), "point"), _located(p), _columns(p.attributes))


def polylines_line(p):
    features, parts = _plain(len(p), "feature"), _plain(len(p.parts), "part")
    return f"Polylines({features}, {parts}, {_plain(len(p.column_names), 'column')})"


def polylines_card(p):
    summary = f"{_count(len(p), 'feature')}, {_count(len(p.parts), 'part')}"
    closed = int(np.sum(p.closed))
    facts = [("closed", f"{closed:,} of {_count(len(p.parts), 'part')}"), *_located(p)]
    return Card("Polylines", summary, facts, _columns(p.attributes))


def _layout(m):
    return "sub-blocked" if m.extents is not None else "regular" if m.index is None else "masked"


def _block_rows(m, count):
    cells = int(np.prod(m.count))
    if m.extents is not None:
        return f"{count(len(m), 'sub-block')} in {count(cells, 'block')}"
    if m.index is None:
        return count(cells, "block")
    return f"{len(m) if count is _plain else f'{len(m):,}'} of {count(cells, 'block')}"


def block_model_line(m):
    grid = f"{_triple(m.count)} of {_triple(m.size, m.length_unit)}"
    columns = _plain(len(m.column_names), "column")
    return f"BlockModel({_layout(m)}, {_block_rows(m, _plain)}, {grid}, {columns})"


def block_model_card(m):
    facts = [
        ("count", _triple(m.count)),
        ("size", _triple(m.size, m.length_unit)),
        ("origin", ", ".join(_number(v) for v in m.origin)),
        ("rotation", _angles(m.rotation) if any(m.rotation) else None),
        *_located(m),
    ]
    return Card("BlockModel", f"{_layout(m)}, {_block_rows(m, _count)}", facts, _columns(m.attributes))


def _mesh_summary(m, count):
    closed = "closed" if m.is_closed else f"open, {count(m._boundary_edges, 'boundary edge')}"
    return f"{count(len(m), 'vertex', 'vertices')}, {count(len(m.triangles), 'triangle')}, {closed}"


def mesh_line(m):
    return f"Mesh({_mesh_summary(m, _plain)})"


def mesh_card(m):
    area = [("area", _with_unit(_number(m.area), m.length_unit and f"{m.length_unit}²"))]
    if m.is_closed:
        area.append(("volume", _with_unit(_number(m.volume), m.length_unit and f"{m.length_unit}³")))
    grids = _columns(m.vertex_attributes, "vertex columns") + _columns(m.face_attributes, "face columns")
    return Card("Mesh", _mesh_summary(m, _count), [*area, *_located(m)], grids)


def _drillholes_summary(d, count):
    intervals = d._intervals
    holes = count(len(d), "hole")
    return holes if intervals is None else f"{holes}, {count(len(intervals), 'interval')}"


def drillholes_line(d):
    return f"Drillholes({_drillholes_summary(d, _plain)})"


def drillholes_card(d):
    intervals = d._intervals
    grids = [] if intervals is None else _columns(intervals, "interval columns")
    return Card("Drillholes", _drillholes_summary(d, _count), _located(d), grids)


def _structure_model(s):
    if s.order is not None:
        return f"{s.model} (order {_number(s.order)})"
    if s.exponent is not None:
        return f"{s.model} (exponent {_number(s.exponent)})"
    return s.model


def _structure_shape(s):
    if s.order not in (None, 1.5):
        return f", order={s.order!r}"
    if s.exponent not in (None, 1.0):
        return f", exponent={s.exponent!r}"
    return ""


def structure_line(s):
    return f"Structure({s.model!r}, {s.sill!r}, {s.range!r}{_structure_shape(s)})"


def structure_card(s):
    return Card("Structure", _structure_model(s), [("sill", s.sill), ("range", s.range)])


def _isotropic(v):
    return tuple(v.ratios) == (1.0, 1.0)


def _variogram_short(v):
    nugget = [f"nugget {_number(v.nugget)}"] if v.nugget else []
    return " + ".join(nugget + [f"{s.model}({_number(s.sill)}, {_number(s.range)})" for s in v.structures])


def variogram_line(v):
    structures = ", ".join(
        structure_line(s) if _structure_shape(s) else repr((s.model, s.sill, s.range)) for s in v.structures
    )
    options = [
        f"nugget={v.nugget!r}" if v.nugget else None,
        f"rotation={tuple(v.rotation)!r}" if any(v.rotation) else None,
        None if _isotropic(v) else f"ratios={tuple(v.ratios)!r}",
        f"length_unit={v.length_unit!r}" if v.length_unit else None,
    ]
    return f"Variogram([{structures}]{''.join(f', {o}' for o in options if o)})"


def variogram_card(v):
    structures = v.structures
    facts = [
        ("rotation", None if _isotropic(v) and not any(v.rotation) else _angles(v.rotation)),
        ("ratios", None if _isotropic(v) else _ratios(v.ratios)),
        ("length unit", v.length_unit),
        ("unit", v.unit),
    ]
    if _isotropic(v):
        header = ("structure", "sill", "range")
        rows = [("nugget", v.nugget, None)] + [(_structure_model(s), s.sill, s.range) for s in structures]
    else:
        header = ("structure", "sill", "major", "semi", "minor")
        semi, minor = v.ratios
        rows = [("nugget", v.nugget, None, None, None)] + [
            (_structure_model(s), s.sill, s.range, s.range * semi, s.range * minor) for s in structures
        ]
    summary = f"sill {_number(v.sill)}, {_count(len(structures), 'structure')}"
    return Card("Variogram", summary, facts, [Grid(header, rows)])


def _experimental_summary(e, count):
    lags, pairs = e.lags, int(np.sum(e.counts))
    reach = f" to {_with_unit(_number(float(lags[-1])), e.length_unit)}" if len(lags) else ""
    return f"{count(len(lags), 'lag')}{reach}, {count(pairs, 'pair')}"


def experimental_line(e):
    return f"ExperimentalVariogram({_experimental_summary(e, _plain)})"


def experimental_card(e):
    rows = [(float(h), float(g), int(n)) for h, g, n in zip(e.lags, e.gammas, e.counts)]
    rows, note = _capped(rows, "lag")
    facts = [("length unit", e.length_unit), ("unit", e.unit)]
    grid = Grid(("lag", "gamma", "pairs"), rows, note=note)
    return Card("ExperimentalVariogram", _experimental_summary(e, _count), facts, [grid])


def _variogram_set_summary(s):
    directions = s.directions
    spread = "omnidirectional" if directions is None else _plain(len(directions), "direction")
    return f"{_plain(s.nvar, 'variable')}, {spread}"


def variogram_set_line(s):
    return f"VariogramSet({_variogram_set_summary(s)})"


def variogram_set_card(s):
    names = [n for n in s.names if n is not None]
    facts = [("variables", ", ".join(names) or None)]
    grids = []
    if s.directions is not None:
        rows, note = _capped([(float(a), float(d)) for a, d in s.directions], "direction")
        grids.append(Grid(("azimuth", "dip"), rows, note=note))
    return Card("VariogramSet", _variogram_set_summary(s), facts, grids)


def _extent(obj):
    if obj.ranges is not None:
        return repr(tuple(obj.ranges))
    return repr(obj.radius)


def high_grade_line(h):
    options = [
        f"rotation={tuple(h.rotation)!r}" if h.rotation is not None else None,
        f"mode={h.mode!r}" if h.mode != "drop" else None,
    ]
    return f"HighGrade({h.threshold!r}, {_extent(h)}{''.join(f', {o}' for o in options if o)})"


def _high_grade_short(h):
    reach = _triple(h.ranges) if h.ranges is not None else _number(h.radius)
    verb = "dropped" if h.mode == "drop" else "clamped"
    return f"above {_number(h.threshold)} {verb} beyond {reach}"


def high_grade_card(h):
    facts = [
        ("threshold", h.threshold),
        ("radius", h.radius if h.ranges is None else None),
        ("ranges", None if h.ranges is None else _triple(h.ranges)),
        ("rotation", None if h.rotation is None else _angles(h.rotation)),
        ("mode", h.mode),
    ]
    return Card("HighGrade", _high_grade_short(h), facts)


def search_line(s):
    hidden = {"radius", "ratios"} if s.ranges is not None else set()
    hidden |= {"sectors"} if s.octant else set()
    parts = []
    for name, parameter in inspect.signature(type(s)).parameters.items():
        value = getattr(s, name)
        if name in hidden or value is None or value == parameter.default:
            continue
        parts.append(repr(value) if name == "radius" else f"{name}={value!r}")
    return f"Search({', '.join(parts)})"


def _samples(s):
    low, high = s.min_samples, s.max_samples
    return _plain(high, "sample") if low == high else f"{low} to {high} samples"


def _reach(s):
    unit = s.length_unit
    if s.ranges is not None:
        return f"ranges {_triple(s.ranges, unit)}"
    return f"radius {_with_unit(_number(s.radius), unit)}"


def _search_short(search):
    (only,) = [search] if isinstance(search, bt.Search) else search
    return f"{_reach(only)}, {_samples(only)}"


def _soft(soft):
    if soft is None or not isinstance(soft, dict):
        return soft
    return ", ".join(f"{target} from {sample} within {_number(d)}" for (target, sample), d in soft.items())


def search_card(s):
    sectors = "octants" if s.octant else None if s.sectors is None else _plain(s.sectors, "sector")
    facts = [
        ("radius", None if s.ranges is not None else _with_unit(_number(s.radius), s.length_unit)),
        ("ranges", None if s.ranges is None else _triple(s.ranges, s.length_unit)),
        ("rotation", None if s.rotation is None else _angles(s.rotation)),
        (
            "ratios",
            None if s.ratios is None or s.ranges is not None else _ratios(s.ratios),
        ),
        ("samples", f"{s.min_samples} to {s.max_samples}"),
        ("sectors", sectors),
        ("max per sector", s.max_per_sector),
        ("min sectors", s.min_sectors),
        ("max empty sectors", s.max_empty_sectors),
        ("max per hole", s.max_per_hole),
        ("min holes", s.min_holes),
        ("high grade", None if s.high_grade is None else _high_grade_short(s.high_grade)),
        ("soft", _soft(s.soft)),
        ("target slope", s.target_slope),
        ("target efficiency", s.target_efficiency),
    ]
    return Card("Search", f"{_reach(s)}, {_samples(s)}", facts)


def _search_facts(search):
    if search is None:
        return []
    passes = [search] if isinstance(search, bt.Search) else list(search)
    if len(passes) == 1:
        return [("search", _search_short(passes[0]))]
    return [(f"search {i}", _search_short(s)) for i, s in enumerate(passes, 1)]


def _model_parts(variogram, search, samples):
    passes = 1 if search is None or isinstance(search, bt.Search) else len(search)
    return [
        "not fitted" if samples is None else f"fitted on {_plain(samples, 'sample')}",
        None if variogram is None else _variogram_short(variogram),
        None if search is None else _search_short(search) if passes == 1 else f"{passes} search passes",
    ]


def _model_card(name, variogram, search, samples, unit, extra=()):
    summary = "not fitted" if samples is None else f"fitted on {_count(samples, 'sample')}"
    facts = [
        ("unit", unit),
        ("variogram", None if variogram is None else _variogram_short(variogram)),
        *_search_facts(search),
        *extra,
    ]
    return Card(name, summary, facts)


def _estimator_state(e):
    engine = getattr(e, "_engine", None)
    try:
        samples = None if engine is None else len(engine.values)
    except BoitataError:
        samples = None
    variogram = None if engine is None else engine.variogram
    unit = getattr(e, "_variogram_unit", None)
    if variogram is not None and unit is not None:
        variogram = bt.Variogram(
            variogram.structures,
            nugget=variogram.nugget,
            rotation=variogram.rotation,
            ratios=variogram.ratios,
            length_unit=unit,
        )
    return variogram, getattr(e, "_search", None), samples


def estimator_line(e):
    parts = _model_parts(*_estimator_state(e))
    return f"{type(e).__name__}({', '.join(p for p in parts if p)})"


def estimator_card(e):
    return _model_card(type(e).__name__, *_estimator_state(e), e.unit)


def simulator_line(s):
    parts = _model_parts(s.variogram, s.search or None, s.n_samples)
    return f"{type(s).__name__}({', '.join(p for p in parts if p)})"


def simulator_card(s):
    extra = [("bands", getattr(s, "bands", None)), ("correlation", getattr(s, "correlation", None))]
    return _model_card(type(s).__name__, s.variogram, s.search or None, s.n_samples, None, extra)


def _estimated(cv):
    return int(np.sum(~np.isnan(cv.estimate)))


def cross_validation_line(cv):
    parts = [_plain(len(cv.actual), "sample")]
    if _estimated(cv) > 1:
        parts += [f"rmse {_number(cv.rmse)}", f"slope {_number(cv.slope)}"]
    return f"{type(cv).__name__}({', '.join(parts)})"


def cross_validation_card(cv):
    estimated = _estimated(cv)
    summary = f"{_count(len(cv.actual), 'sample')}, {estimated:,} estimated"
    if estimated < 2:
        return Card(type(cv).__name__, summary)
    facts = [
        ("mean error", cv.mean_error),
        ("rmse", cv.rmse),
        ("correlation", cv.correlation),
        ("slope", cv.slope),
        ("standardized squared error", cv.standardized_squared_error),
    ]
    grids = []
    if isinstance(cv, IndicatorCrossValidation):
        facts.append(("goodness", cv.goodness))
        rows, note = _capped(list(zip(map(float, cv.thresholds), map(float, cv.brier))), "threshold")
        grids.append(Grid(("threshold", "brier"), rows, note=note))
    return Card(type(cv).__name__, summary, facts, grids)


def categorical_cross_validation_line(cv):
    samples, categories = _plain(len(cv.actual), "sample"), _plain(len(cv.names), "category", "categories")
    return f"{type(cv).__name__}({samples}, {categories})"


def categorical_cross_validation_card(cv):
    summary = f"{_count(len(cv.actual), 'sample')}, {_count(len(cv.names), 'category', 'categories')}"
    rows, note = _capped(list(zip(cv.names, map(float, cv.brier))), "category")
    return Card(type(cv).__name__, summary, grids=[Grid(("category", "brier"), rows, note=note)])


def _spread(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return ()
    return (float(np.nanmin(values)), float(np.nanmean(values)), float(np.nanmax(values)))


def _summary_summary(s, count):
    return f"{count(s.n, 'realization')}, {count(len(s.mean), 'row')}"


def simulation_summary_line(s):
    return f"SimulationSummary({_summary_summary(s, _plain)})"


def simulation_summary_card(s):
    groups = s.groups
    facts = [
        ("groups", None if groups is None else _count(len(groups), "group")),
        ("cutoffs", ", ".join(_number(c) for c in s.cutoffs) or None),
        ("quantiles", ", ".join(_number(q) for q in s.quantiles) or None),
        ("tolerances", ", ".join(_number(t) for t in s.tolerances) or None),
    ]
    rows = [("mean", *_spread(s.mean)), ("std", *_spread(s.std))]
    grid = Grid(("over the rows", "min", "mean", "max"), [r for r in rows if len(r) > 1])
    return Card("SimulationSummary", _summary_summary(s, _count), facts, [grid] if grid.rows else [])


def _pretty(self, printer, cycle):
    """IPython shows the card for the object asked for and the line for one inside a container."""
    printer.text(repr(self) if cycle or len(getattr(printer, "stack", ())) > 1 else str(self))


def _show(cls, line, card):
    cls.__repr__ = lambda self: line(self)
    cls.__str__ = lambda self: _text(card(self))
    # An f-string embeds the line; without this it would embed the card of __str__.
    cls.__format__ = lambda self, spec: format(line(self), spec)
    cls._repr_html_ = lambda self: _html(card(self))
    cls._repr_pretty_ = _pretty


for _cls, _line, _card in (
    (bt.Table, table_line, table_card),
    (bt.PointSet, point_set_line, point_set_card),
    (bt.Polylines, polylines_line, polylines_card),
    (bt.BlockModel, block_model_line, block_model_card),
    (bt.Mesh, mesh_line, mesh_card),
    (bt.Drillholes, drillholes_line, drillholes_card),
    (bt.Structure, structure_line, structure_card),
    (bt.Variogram, variogram_line, variogram_card),
    (bt.ExperimentalVariogram, experimental_line, experimental_card),
    (bt.VariogramSet, variogram_set_line, variogram_set_card),
    (bt.HighGrade, high_grade_line, high_grade_card),
    (bt.Search, search_line, search_card),
    (_Base, estimator_line, estimator_card),
    (bt.SGS, simulator_line, simulator_card),
    (bt.DSS, simulator_line, simulator_card),
    (bt.TurningBands, simulator_line, simulator_card),
    (CrossValidation, cross_validation_line, cross_validation_card),
    (IndicatorCrossValidation, cross_validation_line, cross_validation_card),
    (CategoricalCrossValidation, categorical_cross_validation_line, categorical_cross_validation_card),
    (bt.SimulationSummary, simulation_summary_line, simulation_summary_card),
):
    _show(_cls, _line, _card)
