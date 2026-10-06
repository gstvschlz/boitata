"""3D scenes drawn in the browser by the bundled three.js viewer.

A `Scene` turns containers into float32 buffers around a local float64 origin plus a JSON description, and hands
both to the viewer: as an anywidget widget in Jupyter, or inlined in a self-contained HTML page.
"""

import base64
import html
import json
import math
import tempfile
import warnings
import webbrowser
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import numpy as np

from boitata._boitata import BlockModel, Drillholes, Mesh, PointSet

_BUNDLE = Path(__file__).with_name("_viewer.js")
_COLORMAPS = ("viridis", "magma", "inferno", "plasma", "cividis", "turbo")
_THEMES = ("auto", "light", "dark")
_THEME_KEYS = ("background", "panel", "raised", "border", "text", "muted", "grid", "box", "accent", "layer")
_WARN_BYTES = 200 * 2**20
_NAMES = {"drillholes": "drill holes", "points": "points", "mesh": "mesh", "blocks": "block model"}
_REPRESENTATIONS = {"drillholes": "lines", "points": "points", "mesh": "surface", "blocks": "cells"}


def _bundle():
    if not _BUNDLE.exists():
        raise RuntimeError(
            f"the 3D viewer bundle {_BUNDLE.name} is missing; build it with `mise run js:build`"
        )
    return _BUNDLE


def _theme(theme):
    if isinstance(theme, str):
        if theme not in _THEMES:
            raise ValueError(f"theme must be one of {', '.join(_THEMES)} or a dict, got {theme!r}")
        return theme
    if not isinstance(theme, Mapping):
        raise TypeError(f"theme must be a str or a dict, got {type(theme).__name__}")
    unknown = set(theme) - {"base", *_THEME_KEYS}
    if unknown:
        raise ValueError(f"unknown theme keys {sorted(unknown)}; valid keys: base, {', '.join(_THEME_KEYS)}")
    if theme.get("base", "auto") not in _THEMES:
        raise ValueError(f"theme base must be one of {', '.join(_THEMES)}, got {theme['base']!r}")
    if not all(isinstance(v, str) for v in theme.values()):
        raise TypeError("theme values must be CSS color strings")
    return dict(theme)


def _valid_text(values):
    def valid(v):
        return v is not None and v != "" and not (isinstance(v, float) and math.isnan(v))

    return np.array([valid(v) for v in values.tolist()], dtype=bool)


def _encode(values):
    """Float32 values with NaN for nulls, or int32 category codes with -1 for nulls; None when all null."""
    a = np.asarray(values)
    if a.dtype == object or a.dtype.kind in "USb":
        a = a.astype(object)
        valid = _valid_text(a)
        if not valid.any():
            return None
        categories, codes = np.unique(a[valid].astype(str), return_inverse=True)
        out = np.full(len(a), -1, dtype=np.int32)
        out[valid] = codes
        return out, {"type": "text", "categories": categories.tolist()}
    if a.dtype.kind not in "iuf":
        return None
    out = a.astype(np.float32)
    valid = np.isfinite(out)
    if not valid.any():
        return None
    out[~valid] = np.nan
    return out, {"type": "number", "min": float(out[valid].min()), "max": float(out[valid].max())}


def _drillholes(data):
    """Per-interval segments split at survey stations, or per-hole traces without intervals."""
    paths = data.paths()
    hole_column = paths.column_names[0]
    station_holes = np.asarray(paths[hole_column], dtype=object)
    station_depths = np.asarray(paths["depth"], dtype=np.float64)
    if data.interval_columns is None:
        xyz = np.c_[paths["x"], paths["y"], paths["z"]]
        same = station_holes[1:] == station_holes[:-1]
        starts = np.flatnonzero(same)
        ends = np.c_[xyz[starts], xyz[starts + 1]].reshape(-1, 3)
        table = {hole_column: station_holes[starts]}
        return ends, np.arange(len(starts), dtype=np.uint32), table
    table = data.samples().attributes
    hole, start, end = data.interval_columns
    holes = np.asarray(table[hole], dtype=object)
    lo, hi = np.asarray(table[start], dtype=np.float64), np.asarray(table[end], dtype=np.float64)
    names, codes = np.unique(np.r_[station_holes, holes].astype(str), return_inverse=True)
    station_code, interval_code = codes[: len(station_holes)], codes[len(station_holes) :]
    span = max(float(np.nanmax(np.abs(np.r_[station_depths, lo, hi]))), 1.0) * 2 + 1
    key = station_code * span + station_depths
    order = np.argsort(key, kind="stable")
    key, depths = key[order], station_depths[order]
    first = np.searchsorted(key, interval_code * span + lo, side="right")
    last = np.searchsorted(key, interval_code * span + hi, side="left")
    inner = np.maximum(last - first, 0)
    count = inner + 2
    at = np.cumsum(count) - count
    along = np.empty(int(count.sum()))
    along[at] = lo
    along[at + count - 1] = hi
    before = np.cumsum(inner) - inner
    k = np.arange(int(inner.sum()))
    along[np.repeat(at + 1 - before, inner) + k] = depths[np.repeat(first - before, inner) + k]
    xyz = data.at(names[np.repeat(interval_code, count)].tolist(), along)
    a = np.delete(np.arange(len(along)), at + count - 1)
    ends = np.c_[xyz[a], xyz[a + 1]].reshape(-1, 3)
    rows = np.repeat(np.arange(len(holes), dtype=np.uint32), count - 1)
    return ends, rows, table


def _block_axes(model):
    """Unit world directions of the model's x, y and z axes, from the corners boitata gives a one-cell model."""
    corners = BlockModel((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), rotation=tuple(model.rotation)).corners[
        0
    ]
    return np.array([corners[1] - corners[0], corners[2] - corners[0], corners[4] - corners[0]])


def _block_sizes(model):
    size = np.asarray(model.size, dtype=np.float64)
    extents = model.extents
    if extents is None:
        return np.tile(size, (len(model), 1))
    return (np.asarray(extents)[:, 3:] - np.asarray(extents)[:, :3]) * size


def _layer(data):
    """Kind, world-coordinate geometry (name to array) and attribute tables by association."""
    if isinstance(data, PointSet):
        return "points", {"positions": np.asarray(data.coords, dtype=np.float64)}, {"row": data.attributes}
    if isinstance(data, Drillholes):
        ends, rows, table = _drillholes(data)
        return "drillholes", {"positions": ends, "rows": rows}, {"row": table}
    if isinstance(data, BlockModel):
        geometry = {"centers": np.asarray(data.centroids, dtype=np.float64), "sizes": _block_sizes(data)}
        return "blocks", geometry, {"row": data.attributes}
    if isinstance(data, Mesh):
        geometry = {
            "positions": np.asarray(data.vertices, dtype=np.float64),
            "triangles": np.asarray(data.triangles, dtype=np.uint32),
        }
        return "mesh", geometry, {"vertex": data.vertex_attributes, "face": data.face_attributes}
    raise TypeError(f"cannot draw {type(data).__name__}; pass a PointSet, Drillholes, BlockModel or Mesh")


def _names(table):
    return list(table) if isinstance(table, Mapping) else table.column_names


def _bounds(points):
    points = points[np.isfinite(points).all(axis=1)]
    return (points.min(axis=0), points.max(axis=0)) if len(points) else None


def _in_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and getattr(shell, "kernel", None) is not None


@cache
def _widget_class():
    import anywidget
    import traitlets

    class SceneWidget(anywidget.AnyWidget):
        _esm = _bundle()
        spec = traitlets.Dict().tag(sync=True)
        buffers = traitlets.Dict().tag(sync=True)

    return SceneWidget


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>html, body {{ margin: 0; height: 100%; }} #scene {{ height: 100%; }}</style>
</head>
<body>
<div id="scene"></div>
<script id="scene-viewer" type="text/plain">{bundle}</script>
<script id="scene-data" type="application/json">{data}</script>
<script type="module">
const text = (id) => document.getElementById(id).textContent;
const code = Uint8Array.from(atob(text("scene-viewer")), (c) => c.charCodeAt(0));
const viewer = await import(URL.createObjectURL(new Blob([code], {{ type: "text/javascript" }})));
window.scene = await viewer.default.mount(document.getElementById("scene"), JSON.parse(text("scene-data")), {{ fill: true }});
</script>
</body>
</html>
"""


class Scene:
    """3D scene of drill holes, points, meshes and block models, drawn by a three.js viewer.

    Layers colored by the same variable share its color map, range and color bar. Null values are never drawn:
    a row null in the column that colors its layer is left out, and nulls never enter a color range.

    Parameters
    ----------
    theme : {"auto", "light", "dark"} or dict, default "auto"
        Colors of the background, box, grid, labels, panel and color bars, and of layers drawn in no color of
        their own. ``"auto"`` follows the page hosting the viewer (JupyterLab, VS Code, the documentation) or else
        the system preference, and switches with it. A dict overrides keys of a built-in theme: ``base``
        (``"auto"``, ``"light"`` or ``"dark"``, default ``"auto"``) and CSS colors for ``background``, ``panel``,
        ``raised``, ``border``, ``text``, ``muted``, ``grid``, ``box``, ``accent`` and ``layer``.
    height : int, default 600
        Height of the viewer in a notebook, in pixels; a saved page fills the browser window.
    """

    def __init__(self, *, theme="auto", height=600):
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError(f"height must be a positive int, got {height!r}")
        self._theme = _theme(theme)
        self._height = height
        self._view = {"azimuth": 45.0, "dip": 30.0}
        self._layers = []
        self._buffers = {}
        self._variables = {}
        self._origin = None
        self._crs = None
        self._unit = None
        self._warned = False

    def add(
        self,
        data,
        values=None,
        *,
        name=None,
        color=None,
        opacity=1.0,
        cmap=None,
        clim=None,
        label=None,
        visible=True,
        columns=None,
    ):
        """Adds a layer.

        Drill holes draw as lines, one segment per interval split at the survey stations so it follows the
        trace; block models as boxes, sub-blocks at their own size, along the model's rotated axes; meshes as
        surfaces; point sets as round points of a fixed screen size.

        Parameters
        ----------
        data : PointSet, Drillholes, BlockModel or Mesh
            What to draw. Drill holes take their interval columns, or the hole name without intervals; a mesh
            its vertex and face attributes.
        values : str, optional
            Column that colors the layer; its null rows are not drawn. Numbers map through `cmap`, text takes one
            color per category. Without it the layer is drawn in `color`.
        name : str, optional
            Name in the layer panel; default the kind and a number, e.g. ``"block model 1"``.
        color : str, optional
            CSS color of the layer when not colored by a column; default the theme's ``layer`` color.
        opacity : float, default 1.0
            From 0 (invisible) to 1 (opaque).
        cmap : str, optional
            Color map of `values` for every layer it colors: ``"viridis"`` (default), ``"magma"``,
            ``"inferno"``, ``"plasma"``, ``"cividis"`` or ``"turbo"``.
        clim : tuple of float, optional
            ``(low, high)`` range of `values` for every layer it colors; default the range of their valid values.
        label : str, optional
            Title of the color bar of `values`; default the column name.
        visible : bool, default True
            Whether the layer starts shown; the panel toggles it.
        columns : sequence of str, optional
            Columns sent to the viewer, which offers them to color by; default every numeric and text column.
            `values` is always sent.

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        kind, geometry, tables = _layer(data)
        if not 0.0 <= float(opacity) <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {opacity!r}")
        if color is not None and not isinstance(color, str):
            raise TypeError(f"color must be a CSS color string, got {color!r}")
        if cmap is not None and cmap not in _COLORMAPS:
            raise ValueError(f"cmap must be one of {', '.join(_COLORMAPS)}, got {cmap!r}")
        if clim is not None:
            clim = tuple(float(c) for c in clim)
            if len(clim) != 2 or not all(math.isfinite(c) for c in clim) or clim[0] >= clim[1]:
                raise ValueError(f"clim must be two finite numbers, low < high, got {clim!r}")
        if values is None and (cmap is not None or clim is not None or label is not None):
            raise ValueError("cmap, clim and label describe `values`; pass values too")
        available = [n for table in tables.values() for n in _names(table)]
        if values is not None and values not in available:
            raise KeyError(f"no column {values!r}; columns: {', '.join(available) or 'none'}")
        wanted = None if columns is None else {*columns, *([values] if values else [])}
        if wanted is not None and wanted - set(available):
            raise KeyError(f"no columns {sorted(wanted - set(available))}; columns: {', '.join(available)}")

        layer_id = f"l{len(self._layers)}"
        points = geometry.get("positions", geometry.get("centers"))
        if self._origin is None:
            bounds = _bounds(points)
            self._origin = (bounds[0] + bounds[1]) / 2 if bounds else np.zeros(3)
        buffers = {}
        refs = {}
        for role, array in geometry.items():
            if role in ("positions", "centers"):
                array = (array - self._origin).astype(np.float32)
            elif role == "sizes":
                array = array.astype(np.float32)
            buffers[f"{layer_id}/{role}"] = array
            refs[role] = f"{layer_id}/{role}"
        specs = []
        for on, table in tables.items():
            for column in _names(table):
                if wanted is not None and column not in wanted:
                    continue
                encoded = _encode(table[column])
                if encoded is None:
                    if column == values:
                        raise ValueError(f"column {values!r} has no valid values to color by")
                    continue
                array, spec = encoded
                key = f"{layer_id}/c{len(specs)}"
                buffers[key] = array
                specs.append({"name": column, "buffer": key, "on": on, **spec})

        layer = {
            "id": layer_id,
            "name": name or f"{_NAMES[kind]} {sum(item['kind'] == kind for item in self._layers) + 1}",
            "kind": kind,
            "representation": _REPRESENTATIONS[kind],
            "color": color,
            "opacity": float(opacity),
            "visible": bool(visible),
            "values": values,
            "geometry": refs,
            "columns": specs,
        }
        if kind == "blocks":
            layer["axes"] = _block_axes(data).tolist()
        if values is not None:
            variable = self._variables.setdefault(values, {})
            variable.update(
                {k: v for k, v in (("cmap", cmap), ("clim", clim), ("label", label)) if v is not None}
            )
        self._layers.append(layer)
        self._buffers.update({k: np.ascontiguousarray(v).tobytes() for k, v in buffers.items()})
        self._crs = self._crs or getattr(data, "crs", None)
        self._unit = self._unit or getattr(data, "length_unit", None)
        size = sum(len(b) for b in self._buffers.values())
        if size > _WARN_BYTES and not self._warned:
            self._warned = True
            warnings.warn(
                f"the scene holds {size / 2**20:.0f} MB of data; pass columns= to send fewer columns",
                stacklevel=2,
            )
        return self

    def view(self, *, azimuth=45.0, dip=30.0):
        """Sets the direction the camera looks along; the view still frames every shown layer.

        Parameters
        ----------
        azimuth : float, default 45.0
            Bearing of the line of sight, in degrees clockwise from north.
        dip : float, default 30.0
            Angle of the line of sight below horizontal, in degrees: 0 looks level, 90 straight down (a plan
            with north up).

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        azimuth, dip = float(azimuth), float(dip)
        if not math.isfinite(azimuth) or not -90.0 <= dip <= 90.0:
            raise ValueError(f"azimuth must be finite and dip in [-90, 90], got {azimuth!r}, {dip!r}")
        self._view = {"azimuth": azimuth % 360.0, "dip": dip}
        return self

    def _spec(self):
        origin = self._origin if self._origin is not None else np.zeros(3)
        titles = ("Easting", "Northing", "Elevation") if self._crs else ("X", "Y", "Z")
        return {
            "origin": [float(v) for v in origin],
            "axes": [f"{t} ({self._unit})" if self._unit else t for t in titles],
            "theme": self._theme,
            "height": self._height,
            "view": dict(self._view),
            "variables": {
                k: dict(v, clim=list(v["clim"])) if "clim" in v else dict(v)
                for k, v in self._variables.items()
            },
            "layers": self._layers,
        }

    def _html(self):
        data = {
            "spec": self._spec(),
            "buffers": {k: base64.b64encode(v).decode("ascii") for k, v in self._buffers.items()},
        }
        names = ", ".join(layer["name"] for layer in self._layers) or "empty scene"
        return _PAGE.format(
            title=html.escape(f"Scene: {names}"),
            bundle=base64.b64encode(_bundle().read_bytes()).decode("ascii"),
            data=json.dumps(data, separators=(",", ":")).replace("</", "<\\/"),
        )

    def save(self, path):
        """Writes the scene to a self-contained HTML page: the viewer and the data inlined, nothing fetched.

        Parameters
        ----------
        path : str or pathlib.Path
            HTML file to write.

        Returns
        -------
        pathlib.Path
        """
        path = Path(path)
        path.write_text(self._html(), encoding="utf-8")
        return path

    def show(self):
        """Displays the scene in Jupyter, or else saves it to a temporary page and opens it in the web browser.

        Returns
        -------
        pathlib.Path or None
            The page opened in the browser; None in Jupyter.
        """
        if _in_notebook():
            from IPython.display import display

            display(self)
            return None
        path = self.save(Path(tempfile.mkdtemp()) / "scene.html")
        webbrowser.open(path.as_uri())
        return path

    def _widget(self):
        return _widget_class()(spec=self._spec(), buffers=dict(self._buffers))

    def _repr_mimebundle_(self, include=None, exclude=None):
        try:
            widget = self._widget()
        except ImportError:
            page = html.escape(self._html(), quote=True)
            frame = (
                f'<iframe srcdoc="{page}" style="width:100%;height:{self._height}px;border:0"></iframe>'
                '<div style="font:12px system-ui,sans-serif;opacity:.7">'
                "For the notebook widget: pip install 'boitata[3d]'</div>"
            )
            return {"text/html": frame, "text/plain": repr(self)}
        return widget._repr_mimebundle_(include=include, exclude=exclude)

    def __repr__(self):
        return f"Scene({len(self._layers)} layers)"
