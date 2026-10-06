"""3D views on pyvista (the ``3d`` extra).

`to_pyvista` converts a container to a pyvista dataset with its attributes as point or cell data. A `Scene` holds
layers that share one color map and range per variable; `plot` and `slices` add a layer to the scene of `plotter`
when given, else to a new one, and return the scene.
"""

import sys
import tempfile
import weakref
import webbrowser
from functools import cached_property, singledispatch
from itertools import pairwise
from pathlib import Path

import numpy as np

from boitata._boitata import BlockModel, Drillholes, Mesh, PointSet, _outer_faces
from boitata.plot import _angles_for_normal, _frame

__all__ = ["Scene", "plot", "slices", "to_pyvista"]

_HEX = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])


def _pyvista():
    try:
        import pyvista
    except ImportError as e:
        raise ImportError(
            "boitata.plot3d needs pyvista: pip install 'boitata[3d]' or conda install -c conda-forge pyvista"
        ) from e
    return pyvista


def _axes(rotation):
    """Rows are the world directions of the block model's x, y and z axes."""
    a, d, r = np.radians(rotation)
    sa, ca, sd, cd, sr, cr = np.sin(a), np.cos(a), np.sin(d), np.cos(d), np.sin(r), np.cos(r)
    r1 = np.array([[sa, ca, 0], [-ca, sa, 0], [0, 0, 1]])
    r2 = np.array([[cd, 0, -sd], [0, 1, 0], [sd, 0, cd]])
    r3 = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    m = r3 @ r2 @ r1
    return np.array([-m[1], m[0], m[2]])


def _column(values):
    return np.where(values == None, "", values).astype(str) if values.dtype == object else values


def _columns(table):
    for name in table.column_names:
        yield name, _column(table[name])


def _blocks(pv, model):
    origin, size, count = (np.asarray(v, dtype=float) for v in (model.origin, model.size, model.count))
    axes = _axes(model.rotation)
    if model.index is None:
        return pv.ImageData(
            dimensions=(count + 1).astype(int), spacing=size, origin=origin, direction_matrix=axes.T
        )
    parent = model.index.astype(np.int64)
    nx, ny = int(count[0]), int(count[1])
    ijk = np.c_[parent % nx, parent // nx % ny, parent // (nx * ny)]
    extent = model.extents
    if extent is None:
        extent = np.tile([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], (len(parent), 1))
    fraction = np.where(_HEX[None], extent[:, None, 3:], extent[:, None, :3])
    points = origin + ((ijk[:, None] + fraction) * size).reshape(-1, 3) @ axes
    cells = np.arange(len(points)).reshape(-1, 8)
    return pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: cells}, points)


class _Solid:
    """Masked or sub-blocked model drawn as its outer faces; `cells`, its hexahedra, are built on the first cut."""

    def __init__(self, model):
        self.model = model
        self.n_cells = len(model)

    @cached_property
    def cells(self):
        return to_pyvista(self.model)

    @property
    def bounds(self):
        return self.cells.bounds

    def slice(self, **kwargs):
        return self.cells.slice(**kwargs)

    def faces(self, values):
        """Faces between a block and an empty cell or the grid's edge, a null of `values` counting as empty."""
        table = self.model.attributes
        name = values if values is not None else next(iter(table.column_names), None)
        field = None if name is None else _column(table[name])
        points, quads, rows = _outer_faces(self.model, None if values is None else _valid(field))
        out = _pyvista().PolyData.from_regular_faces(points, quads)
        if name is not None:
            out.cell_data[name] = field[rows]
        return out


def to_pyvista(data):
    """pyvista dataset of a container, attributes attached.

    Parameters
    ----------
    data : PointSet, Drillholes, BlockModel or Mesh
        A `PointSet` becomes points (``PolyData``) with point data; `Drillholes` one polyline per hole through
        its desurveyed stations, with ``depth`` as point data; a regular `BlockModel` an ``ImageData`` oriented
        by its rotation, a masked or sub-blocked one hexahedra (``UnstructuredGrid``), one cell per row with
        cell data; a `Mesh` triangles (``PolyData``) with vertex and face attributes.

    Returns
    -------
    pyvista.DataSet
    """
    pv = _pyvista()
    if isinstance(data, PointSet):
        out = pv.PolyData(data.coords)
        fields = [(out.point_data, data.attributes)]
    elif isinstance(data, Drillholes):
        paths = data.paths()
        hole = np.array(paths[paths.column_names[0]])
        starts = np.flatnonzero(np.r_[True, hole[1:] != hole[:-1], True])
        lines = [v for a, b in pairwise(starts) if b - a > 1 for v in (b - a, *range(a, b))]
        out = pv.PolyData(np.c_[paths["x"], paths["y"], paths["z"]], lines=lines or None)
        out.point_data["depth"] = paths["depth"]
        fields = []
    elif isinstance(data, BlockModel):
        out = _blocks(pv, data)
        fields = [(out.cell_data, data.attributes)]
    elif isinstance(data, Mesh):
        out = pv.PolyData.from_regular_faces(data.vertices, data.triangles)
        fields = [(out.point_data, data.vertex_attributes), (out.cell_data, data.face_attributes)]
    else:
        raise TypeError(f"cannot convert {type(data).__name__} to pyvista")
    for target, table in fields:
        for name, values in _columns(table):
            target[name] = values
    return out


@singledispatch
def _layer(data):
    """pyvista dataset of `data` and its ``add_mesh`` defaults; each data type registers its own."""
    return (data, {}) if isinstance(data, _pyvista().DataObject) else (to_pyvista(data), {})


@_layer.register
def _(data: BlockModel):
    return (to_pyvista(data) if data.index is None else _Solid(data)), {}


@_layer.register
def _(data: PointSet):
    return to_pyvista(data), {"render_points_as_spheres": True, "point_size": 6}


def _traces(data):
    if data.interval_columns is None:
        return to_pyvista(data)
    table = data.samples().attributes
    hole, start, end = data.interval_columns
    holes = list(table[hole])
    ends = np.hstack([data.at(holes, table[start]), data.at(holes, table[end])]).reshape(-1, 3)
    segments = np.arange(len(ends)).reshape(-1, 2)
    lines = _pyvista().PolyData(ends, lines=np.c_[np.full(len(segments), 2), segments].ravel())
    for name, values in _columns(table):
        lines.cell_data[name] = values
    return lines


@_layer.register
def _(data: Drillholes, *, radius=None):
    lines = _traces(data)
    return lines.tube(radius=lines.length / 400 if radius is None else radius), {}


def _collars(data):
    paths = data.paths()
    hole = np.asarray(paths[paths.column_names[0]])
    first = np.r_[True, hole[1:] != hole[:-1]]
    return np.c_[paths["x"], paths["y"], paths["z"]][first], hole[first].tolist()


def _text(field):
    return field.dtype.kind in "OUS"


def _on_cells(mesh, values):
    return mesh.get_array_association(values).name == "CELL"


def _valid(field):
    return field != "" if _text(field) else np.isfinite(field)


def _drop_nulls(mesh, values):
    keep = _valid(mesh.get_array(values))
    if keep.all():
        return mesh
    if _on_cells(mesh, values):
        return mesh.extract_cells(keep)
    return mesh.extract_points(keep, adjacent_cells=False)


def _image(pv, cube, origin, spacing, axes, values):
    out = pv.ImageData(dimensions=cube.shape, spacing=spacing, origin=origin, direction_matrix=axes)
    out.point_data[values] = cube.ravel(order="F")
    return out


def _volume(pv, mesh, values):
    """Block centers as points padded by one null layer, so nearest interpolation fills each block to its faces."""
    if not isinstance(mesh, pv.ImageData) or not _on_cells(mesh, values):
        raise ValueError("style='volume' needs a regular block model; BlockModel.to_regular makes one")
    field = mesh.cell_data[values]
    keep = np.isfinite(field)
    lo, hi = float(field[keep].min()), float(field[keep].max())
    null = lo - max(hi - lo, 1.0)
    cube = np.where(keep, field, null).astype(np.float32).reshape(np.subtract(mesh.dimensions, 1), order="F")
    axes = np.asarray(mesh.direction_matrix)
    origin = mesh.origin - axes @ np.multiply(mesh.spacing, 0.5)
    grid = _image(pv, np.pad(cube, 1, constant_values=null), origin, mesh.spacing, axes, values)
    return grid, (null, lo, hi)


def _memory(mapper, values):
    """Share of the GPU memory budget a volume takes."""
    budget = mapper.GetMaxMemoryInBytes() * mapper.GetMaxMemoryFraction()
    return mapper.dataset.point_data[values].nbytes / budget


def _points_only(mesh):
    return isinstance(mesh, _pyvista().PolyData) and mesh.n_verts == mesh.n_cells


def _cheap(pv, data, mesh, values, fraction):
    """About `fraction` of a layer's geometry, and the ``add_mesh`` options that draw it, or None to keep it."""
    if isinstance(data, Drillholes):
        return _traces(data), {}
    if isinstance(mesh, pv.ImageData) and (values is None or _on_cells(mesh, values)):
        shape = np.subtract(mesh.dimensions, 1)
        step = np.where(shape > 1, int(np.ceil(fraction ** (-1 / max((shape > 1).sum(), 1)))), 1)
        out = pv.ImageData(
            dimensions=-(-shape // step) + 1,
            spacing=np.multiply(mesh.spacing, step),
            origin=mesh.origin,
            direction_matrix=mesh.direction_matrix,
        )
        every = tuple(slice(None, None, s) for s in step)
        for name, field in mesh.cell_data.items():
            out.cell_data[name] = field.reshape(shape, order="F")[every].ravel(order="F")
        return out, {}
    if not isinstance(mesh, pv.PolyData) or _points_only(mesh):
        n = mesh.n_points if _points_only(mesh) else mesh.n_cells
        keep = np.sort(np.random.default_rng(0).choice(n, max(round(fraction * n), 1), replace=False))
        if not _points_only(mesh):
            return mesh.extract_cells(keep), {}
        out = pv.PolyData(mesh.points[keep])
        for name, field in mesh.point_data.items():
            out.point_data[name] = field[keep]
        return out, {"render_points_as_spheres": False}
    from vtkmodules.vtkFiltersCore import vtkQuadricClustering

    alg = vtkQuadricClustering()
    alg.SetInputData(mesh)
    alg.CopyCellDataOn()
    alg.UseInputPointsOn()
    alg.SetNumberOfDivisions(*[max(int(np.sqrt(fraction * mesh.n_cells / 2)), 2)] * 3)
    alg.Update()
    out = pv.wrap(alg.GetOutput())
    if out.n_cells and values in mesh.point_data and not _text(mesh.point_data[values]):
        out = out.sample(mesh)
    return (out, {}) if out.n_cells and (values is None or values in out.array_names) else (None, {})


_BUDGET = 50_000
_KINDS = {
    "block models": "block model",
    "drill holes": "drill holes",
    "surfaces": "surface",
    "points": "points",
}


def _kind(data, mesh):
    """Group of a layer in `Scene.layer_toggles`."""
    pv = _pyvista()
    if isinstance(data, BlockModel) or isinstance(mesh, _Solid | pv.ImageData | pv.UnstructuredGrid):
        return "block models"
    if isinstance(data, Drillholes):
        return "drill holes"
    if isinstance(data, PointSet) or _points_only(mesh):
        return "points"
    return "surfaces"


def _segments(points):
    """Vertical plane of each segment of an xy polyline: start, along-segment unit vector, length, chainage."""
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or len(xy) < 2 or xy.shape[1] not in (2, 3):
        raise ValueError(f"points must be (n, 2) or (n, 3) with n >= 2, got shape {xy.shape}")
    xy = xy[:, :2]
    steps = np.diff(xy, axis=0)
    lengths = np.hypot(*steps.T)
    keep = lengths > 0
    if not keep.any():
        raise ValueError("points must not all coincide")
    chainage = np.r_[0.0, np.cumsum(lengths[keep])[:-1]]
    return [
        (np.r_[start, 0.0], np.r_[step / length, 0.0], length, at)
        for start, step, length, at in zip(xy[:-1][keep], steps[keep], lengths[keep], chainage, strict=True)
    ]


def _snap(start, end):
    """`end` moved onto the nearest bearing from `start` that is a multiple of 45°, keeping its projection."""
    step = np.subtract(end, start)
    bearing = np.radians(np.round(np.degrees(np.arctan2(step[0], step[1])) / 45) * 45)
    direction = np.array([np.sin(bearing), np.cos(bearing)])
    return np.asarray(start, dtype=float) + max(step @ direction, 0.0) * direction


def _colab_iframe(viewer, src, **kwargs):
    """Embeds a trame view through Colab's proxy of the kernel's ports."""
    from urllib.parse import urlsplit

    from google.colab import output

    url = urlsplit(src)
    height = str(kwargs.get("height", "600px"))
    height = int(height[:-2]) if height.endswith("px") else 600
    output.serve_kernel_port_as_iframe(url.port, path=f"{url.path}?{url.query}", height=height)


_SCENES = weakref.WeakKeyDictionary()


class Scene:
    """Live 3D scene: layers colored by a variable share its color map and range.

    Parameters
    ----------
    plotter : pyvista.Plotter, optional
        Plotter to draw into; a new one by default.
    motion_quality : {"auto", "full"} or float, default "auto"
        What draws while the camera moves (dragged or zoomed): ``"full"`` every layer as it is; ``"auto"`` a cheaper
        copy of about 50 000 cells or points of each larger layer (a volume also when it exceeds GPU memory); a
        number in (0, 1] a cheaper copy of every layer with about that fraction of its geometry. Once the camera
        stops, the layers draw in full again. Each copy is built the first time the camera moves and kept: a
        regular block model takes every n-th block along each axis, points a fixed random subset drawn flat,
        other cells and the faces of masked or sub-blocked models a fixed random subset, drill holes lines in place of tubes,
        surfaces a decimated copy; each keeps its layer's colors and leaves nulls out. Takes effect in the native window and in trame with server
        rendering; client rendering (vtk.js, ``show(browser=True)``) draws the full layers.
    **kwargs
        Passed to ``pyvista.Plotter`` when `plotter` is not given (e.g. ``window_size``, ``off_screen``).

    Attributes
    ----------
    plotter : pyvista.Plotter
        Attributes the scene does not have are looked up on it (``view_vector``, ``close``, ...).
    colors : dict of str to pyvista.LookupTable
        Color map and range of each variable, shared by every layer it colors.
    motion_quality : {"auto", "full"} or float
        As the parameter; setting it drops the cheaper copies built so far.
    """

    def __init__(self, *, plotter=None, motion_quality="auto", **kwargs):
        self.plotter = _pyvista().Plotter(**kwargs) if plotter is None else plotter
        self.colors = {}
        self._categories = {}
        self._ranges = {}
        self._pinned = set()
        self._bars = {}
        self._volumes = {}
        self._layers = []
        self._cuts = []
        self._plane = None
        self._sources = []
        self._fast = {}
        self._names = []
        self._kinds = []
        self._hidden = set()
        self._path = None
        self._pieces = []
        self._width = None
        self._unfolded = False
        self._drawer = None
        self.motion_quality = motion_quality
        self.plotter.renderer.AddObserver("StartEvent", self._move)
        _SCENES[self.plotter] = self

    def __getattr__(self, name):
        if name == "plotter":
            raise AttributeError(name)
        return getattr(self.plotter, name)

    @property
    def motion_quality(self):
        return self._quality

    @motion_quality.setter
    def motion_quality(self, value):
        number = isinstance(value, int | float) and not isinstance(value, bool)
        if value not in ("auto", "full") and not (number and 0 < value <= 1):
            raise ValueError(f"motion_quality must be 'auto', 'full' or a number in (0, 1], got {value!r}")
        for fast in self._fast.values():
            if fast is not None:
                self.plotter.remove_actor(fast, render=False)
                for props in self._volumes.values():
                    props[:] = [v for v in props if v[0] is not fast.prop]
        for i, (*_, actor) in enumerate(self._layers):
            if actor is not None:
                actor.SetVisibility(self._plane is None and i not in self._hidden)
        self._fast = {}
        self._quality = value

    def _move(self, renderer, _):
        moving = self._plane is None and renderer.GetRenderWindow().GetDesiredUpdateRate() >= 1
        for i, layer in enumerate(self._layers):
            if moving and i not in self._fast:
                self._fast[i] = self._degrade(self._sources[i], *layer)
            fast = self._fast.get(i)
            if fast is not None:
                shown = i not in self._hidden
                fast.SetVisibility(moving and shown)
                if self._plane is None:
                    layer[-1].SetVisibility(not moving and shown)

    def _degrade(self, data, mesh, flat, values, style, kwargs, actor):
        if actor is None or self.motion_quality == "full":
            return None
        if isinstance(mesh, _Solid):
            faces = actor.mapper.GetInputAlgorithm().GetInputDataObject(0, 0)
            mesh = _pyvista().wrap(faces).cast_to_unstructured_grid()
        fraction = self.motion_quality
        if fraction == "auto":
            size = mesh.n_points if _points_only(mesh) else mesh.n_cells
            share = _memory(actor.mapper, values) if style == "volume" else 0
            fraction = min(1, _BUDGET / max(size, 1), 1 / share if share else 1)
        if fraction >= 1:
            return None
        cheap, options = _cheap(_pyvista(), data, mesh, values, fraction)
        if cheap is None:
            return None
        options = {**kwargs, **options, "render": False, "reset_camera": False, "show_scalar_bar": False}
        options.pop("name", None)
        return self._draw(cheap, values, style, options)

    def add(self, data, values=None, *, name=None, style=None, radius=None, labels=False, **kwargs):
        """Adds a layer.

        Parameters
        ----------
        data : PointSet, Drillholes, BlockModel, Mesh or pyvista.DataObject
            What to draw; containers go through `to_pyvista`.
        values : str, optional
            Attribute that colors it. Its null rows never render: null points, cells, and the cells of null points
            are left out. Numbers share one range over every layer the variable colors, text one category list.
        name : str, optional
            Label of the layer in `layer_toggles`; default its kind and number, e.g. ``"surface 2"``.
        style : {"surface", "wireframe", "points", "points_gaussian", "volume"}, optional
            How cells are drawn; a `PointSet` draws its points as spheres. A masked or sub-blocked `BlockModel`
            draws as surface or wireframe only the faces between its blocks and empty cells, so a translucent one
            shows its outer shell; a section still cuts its blocks. ``"volume"`` renders a regular block
            model colored by `values` on the GPU, each block a uniform cube; a model too large for GPU memory draws
            on the CPU once the camera stops (see `motion_quality` for while it moves).
        radius : float, optional
            `Drillholes` only: radius of their tubes, 1/400 of the diagonal of their bounds by default. All holes form
            one tube mesh: a tube per interval with the interval columns as cell data, or per hole through its
            desurveyed stations when there are no intervals.
        labels : bool, default False
            `Drillholes` only: writes each hole's name at its collar.
        **kwargs
            Passed to ``plotter.add_mesh``, or ``plotter.add_volume`` for a volume. ``cmap`` sets the variable's
            color map and ``clim`` fixes its range for every layer; the first ``scalar_bar_args`` given for a
            variable style its one color bar. A volume's ``opacity`` is a number or a named ramp over the range
            (``"linear"`` by default, ``"sigmoid"``, ``"geom_r"``, ...).

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        pv = _pyvista()
        mesh, defaults = _layer(data, **({} if radius is None else {"radius": radius}))
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        if style == "volume" and values is None:
            raise ValueError("style='volume' needs values")
        flat = isinstance(data, PointSet | Drillholes) or (
            isinstance(mesh, pv.PolyData) and mesh.n_faces == 0
        )
        kwargs = {**defaults, **kwargs}
        actor = self._draw(mesh, values, style, dict(kwargs))
        kind = _kind(data, mesh)
        self._layers.append((mesh, flat, values, style, kwargs, actor))
        self._sources.append(data)
        self._kinds.append(kind)
        self._names.append(name or f"{_KINDS[kind]} {self._kinds.count(kind)}")
        if labels:
            self.plotter.add_point_labels(*_collars(data), shape=None, show_points=False, always_visible=True)
        return self

    def _draw(self, mesh, values, style, kwargs):
        if isinstance(mesh, _Solid):
            mesh = mesh.faces(values) if style in (None, "surface", "wireframe") else mesh.cells
        volume = style == "volume"
        if values is not None:
            if not volume:
                mesh = _drop_nulls(mesh, values)
            if mesh.n_points == 0 or not _valid(mesh.get_array(values)).any():
                return None
            mesh, kwargs["cmap"] = self._color(
                mesh, values, kwargs.pop("cmap", None), kwargs.pop("clim", None)
            )
            ticks = {"n_labels": 0} if values in self._categories else {}
            bar = {"title": values, **ticks, **kwargs.get("scalar_bar_args", {})}
            kwargs["scalar_bar_args"] = dict(self._bars.setdefault(values, bar))
        if volume:
            return self._volume(mesh, values, **kwargs)
        actor = self.plotter.add_mesh(mesh, scalars=values, style=style, **kwargs)
        if values is not None:
            actor.mapper.SetUseLookupTableScalarRange(True)
            self._paint(values)
        return actor

    def _volume(self, mesh, values, *, opacity="linear", **kwargs):
        pv = _pyvista()
        grid, span = _volume(pv, mesh, values)
        actor = self.plotter.add_volume(
            grid, scalars=values, clim=self._ranges[values], mapper="smart", **kwargs
        )
        actor.prop.interpolation_type = "nearest"
        actor.mapper.SetInterpolationModeToNearestNeighbor()
        if _memory(actor.mapper, values) > 1:
            actor.mapper.SetRequestedRenderMode(actor.mapper.RayCastRenderMode)
        self._volumes.setdefault(values, []).append((actor.prop, opacity, *span))
        self._paint(values)
        return actor

    def _paint(self, values):
        from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
        from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

        pv = _pyvista()
        lut = self.colors[values]
        lut.scalar_range = lo, hi = self._ranges[values]
        n = lut.n_values
        ramp = np.linspace(lo, hi, 256)
        for prop, opacity, null, low, high in self._volumes.get(values, ()):
            color = vtkColorTransferFunction()
            for x in lo + (np.arange(n) + 0.5) * (hi - lo) / n:
                color.AddRGBPoint(x, *lut.map_value(x, opacity=False))
            alpha = (
                pv.opacity_transfer_function(opacity, 256) / 255
                if isinstance(opacity, str)
                else [opacity] * 256
            )
            shape = vtkPiecewiseFunction()
            shape.AddPoint(null, 0.0)
            shape.AddPoint((null + low) / 2, 0.0)
            for x in np.linspace(low, high, 64):
                shape.AddPoint(x, float(np.interp(x, ramp, alpha)))
            prop.SetColor(color)
            prop.SetScalarOpacity(shape)

    def section(self, origin=None, *, points=None, azimuth=90.0, dip=90.0, width=None):
        """Cuts every shown layer with a plane or a vertical stepped curtain, in place of the layers and of the
        previous section.

        Surfaces and solids show their intersection with the plane; points and drill holes within `width` of it
        are clipped to that slab and projected onto it. Each piece keeps its layer's variable and style. Layers
        hidden with `layer_toggles` are left out.

        Parameters
        ----------
        origin : array_like or None
            ``(x, y, z)`` point on the plane; None, without `points`, removes the section and shows the layers
            again.
        points : array_like, optional
            ``(n, 2)`` or ``(n, 3)`` vertices of a polyline in plan, in place of `origin`, `azimuth` and `dip`:
            each segment cuts on the vertical plane through it, between its two ends.
        azimuth, dip : float
            Bearing of the section line and dip of the plane, in degrees (90 and 90: a vertical east-west
            section, dip 0: a plan).
        width : float, optional
            Full width of the slab that keeps points and drill holes; default a twentieth of the scene's diagonal.

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        segments = None if points is None else _segments(points)
        for actor in self._cuts:
            self.plotter.remove_actor(actor, render=False)
        self._cuts = []
        self._pieces = []
        self._unfolded = False
        self._width = width
        self._path = None if segments is None else np.asarray(points, dtype=float)[:, :2]
        self._plane = None if origin is None and segments is None else (origin, azimuth, dip)
        for i, (*_, actor) in enumerate(self._layers):
            if actor is not None:
                actor.SetVisibility(self._plane is None and i not in self._hidden)
        if self._plane is None:
            return self
        half = (self._extent()[1] / 20 if width is None else width) / 2
        if segments is None:
            center, _, _, n = _frame(self._plane)
            planes = [(center, n, None, None, 0.0)]
        else:
            planes = [
                (start, np.array([u[1], -u[0], 0.0]), u, length, at) for start, u, length, at in segments
            ]
        for i, (mesh, flat, values, style, kwargs, _) in enumerate(self._layers):
            if i in self._hidden:
                continue
            if style == "volume":
                style, kwargs = None, {k: v for k, v in kwargs.items() if k != "opacity"}
            for center, n, u, length, at in planes:
                if flat:
                    cut = mesh.clip(normal=n, origin=center + half * n).clip(
                        normal=-n, origin=center - half * n
                    )
                    cut.points = cut.points - np.outer((cut.points - center) @ n, n)
                else:
                    cut = mesh.slice(normal=n, origin=center)
                if u is not None and cut.n_points:
                    cut = cut.clip(normal=-u, origin=center).clip(normal=u, origin=center + length * u)
                if cut.n_points:
                    self._pieces.append((cut, center, u, at, values, style, kwargs))
        self._show_pieces(self._pieces)
        return self

    def _show_pieces(self, pieces):
        for actor in self._cuts:
            self.plotter.remove_actor(actor, render=False)
        self._cuts = []
        for cut, *_, values, style, kwargs in pieces:
            actor = self._draw(cut, values, style, dict(kwargs))
            self._cuts += [actor] if actor is not None else []

    def section_widget(self, *, origin=None, azimuth=90.0, dip=90.0, width=None, **kwargs):
        """Interactive plane that calls `section` each time it is moved.

        Parameters
        ----------
        origin : array_like, optional
            Starting point on the plane; default the center of the layers.
        azimuth, dip, width : float
            As in `section`.
        **kwargs
            Passed to ``plotter.add_plane_widget``.

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        center, _, _, n = _frame((self._extent()[0] if origin is None else origin, azimuth, dip))

        def move(normal, point):
            a, d = _angles_for_normal(np.asarray(normal, dtype=float))
            self.section(point, azimuth=a, dip=d, width=width)

        self.plotter.add_plane_widget(move, normal=n, origin=center, **kwargs)
        return self

    def view_section(self):
        """Orthographic view normal to the section plane, strike to the right and up dip upward. A stepped curtain
        is first unfolded: each segment's pieces are laid along the first segment's plane at their distance along
        the polyline, so the whole curtain reads as one true-scale section.

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        if self._plane is None:
            raise RuntimeError("no section: call Scene.section first")
        if self._path is None:
            center, _, v, n = _frame(self._plane)
        else:
            first = _segments(self._path)[0]
            start, u0 = first[0], first[1]
            pieces = []
            for cut, center, u, at, *rest in self._pieces:
                flat = cut.copy()
                along = at + (flat.points[:, :2] - center[:2]) @ u[:2]
                flat.points = np.c_[start[:2] + np.outer(along, u0[:2]), flat.points[:, 2]]
                pieces.append((flat, center, u, at, *rest))
            self._show_pieces(pieces)
            self._unfolded = True
            total = sum(length for _, _, length, _ in _segments(self._path))
            z = self._extent()[0][2]
            center, v, n = (
                start + total / 2 * u0 + [0, 0, z],
                np.array([0.0, 0.0, 1.0]),
                np.array([u0[1], -u0[0], 0]),
            )
        self.plotter.enable_parallel_projection()
        self.plotter.camera_position = [center + n, center, v]
        self.plotter.reset_camera()
        return self

    def section_drawer(self, *, width=None):
        """Draws stepped sections with the mouse.

        ``d`` (or the *draw* button) switches to a plan view; each left click adds a vertex, held Shift locks the
        new segment to a multiple of 45° from north, and Enter, a double click or the button again cuts the scene
        along the polyline with `section` and returns to the 3D view. Esc cancels, ``c`` (or *clear*) removes the
        section and ``u`` (or *unfold*) toggles the unfolded view of `view_section`. The buttons sit at the bottom
        left, for viewers that do not pass keys on, such as trame in a browser.

        Parameters
        ----------
        width : float, optional
            As in `section`.

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        self._drawer = {"width": width, "points": [], "on": False, "camera": None, "line": None}
        plotter = self.plotter
        for key, action in (
            ("d", self._draw_toggle),
            ("Return", self._draw_finish),
            ("Escape", self._draw_cancel),
        ):
            plotter.add_key_event(key, action)
        plotter.add_key_event("c", lambda: self._clear())
        plotter.add_key_event("u", self._unfold_toggle)
        if plotter.iren is not None:
            interactor = plotter.iren.interactor
            self._click = interactor.AddObserver("LeftButtonPressEvent", self._draw_click, 10.0)
        buttons = (("draw", self._draw_toggle), ("unfold", self._unfold_toggle), ("clear", self._clear))
        for k, (label, action) in enumerate(buttons):
            self._button(label, action, (10 + 90 * k, 10))
        return self

    def _button(self, label, action, position, size=20):
        def press(_):
            action()
            widget.GetRepresentation().SetState(int(label == "draw" and self._drawer["on"]))

        widget = self.plotter.add_checkbox_button_widget(press, value=False, position=position, size=size)
        self.plotter.add_text(label, position=(position[0] + size + 6, position[1] + 2), font_size=9)

    def _clear(self):
        self.section(None)
        self.plotter.render()

    def _unfold_toggle(self):
        if self._plane is None:
            return
        if self._unfolded:
            self._show_pieces(self._pieces)
            self._unfolded = False
        else:
            self.view_section()
        self.plotter.render()

    def _draw_toggle(self):
        (self._draw_finish if self._drawer["on"] else self._draw_start)()

    def _draw_start(self):
        plotter, drawer = self.plotter, self._drawer
        drawer.update(on=True, points=[], camera=(plotter.camera_position, plotter.parallel_projection))
        center, diagonal = self._extent()
        plotter.enable_parallel_projection()
        plotter.camera_position = [center + [0, 0, diagonal], center, (0, 1, 0)]
        plotter.reset_camera()
        plotter.render()

    def _draw_end(self):
        plotter, drawer = self.plotter, self._drawer
        if drawer["line"] is not None:
            plotter.remove_actor(drawer["line"], render=False)
        position, parallel = drawer["camera"]
        plotter.camera_position = position
        (plotter.enable_parallel_projection if parallel else plotter.disable_parallel_projection)()
        drawer.update(on=False, line=None)
        points, drawer["points"] = drawer["points"], []
        return points

    def _draw_finish(self):
        if not self._drawer["on"]:
            return
        points = self._draw_end()
        if len(points) > 1:
            self.section(points=points, width=self._drawer["width"])
        self.plotter.render()

    def _draw_cancel(self):
        if self._drawer["on"]:
            self._draw_end()
            self.plotter.render()

    def _draw_click(self, interactor, _):
        drawer = self._drawer
        if not drawer["on"]:
            return
        interactor.GetCommand(self._click).SetAbortFlag(1)
        if interactor.GetRepeatCount():
            self._draw_finish()
            return
        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(*interactor.GetEventPosition(), 0)
        renderer.DisplayToWorld()
        x, y, _, w = renderer.GetWorldPoint()
        point = np.array([x, y]) / w
        points = drawer["points"]
        if points and interactor.GetShiftKey():
            point = _snap(points[-1], point)
        points.append(point)
        if drawer["line"] is not None:
            self.plotter.remove_actor(drawer["line"], render=False)
        z = self._extent()[0][2] + self._extent()[1]
        vertices = np.c_[np.array(points), np.full(len(points), z)]
        line = _pyvista().lines_from_points(vertices) if len(points) > 1 else _pyvista().PolyData(vertices)
        drawer["line"] = self.plotter.add_mesh(
            line,
            color="#c05a28",
            line_width=3,
            point_size=8,
            render_points_as_spheres=True,
            reset_camera=False,
        )
        self.plotter.render()

    def layer_toggles(self, *, size=18):
        """A checkbox per layer at the top left, under the name of its kind, that shows or hides it and its cuts.

        Parameters
        ----------
        size : int
            Checkbox size in pixels.

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        plotter = self.plotter
        y = plotter.window_size[1] - 30
        for kind in _KINDS:
            rows = [i for i, k in enumerate(self._kinds) if k == kind]
            if not rows:
                continue
            plotter.add_text(kind, position=(10, y), font_size=9)
            y -= size + 8
            for i in rows:
                plotter.add_checkbox_button_widget(
                    lambda on, i=i: self._toggle(i, on),
                    value=i not in self._hidden,
                    position=(10, y),
                    size=size,
                )
                plotter.add_text(self._names[i], position=(16 + size, y + 2), font_size=8)
                y -= size + 6
            y -= 6
        return self

    def _toggle(self, i, on):
        (self._hidden.discard if on else self._hidden.add)(i)
        actor, fast = self._layers[i][-1], self._fast.get(i)
        if fast is not None:
            fast.SetVisibility(False)
        if self._plane is None:
            if actor is not None:
                actor.SetVisibility(on)
        else:
            unfolded = self._unfolded
            origin, azimuth, dip = self._plane
            self.section(origin, points=self._path, azimuth=azimuth, dip=dip, width=self._width)
            if unfolded:
                self.view_section()
        self.plotter.render()

    def _extent(self):
        """Center and diagonal of the bounds of every layer."""
        bounds = np.array([mesh.bounds for mesh, *_ in self._layers]).reshape(-1, 3, 2)
        lo, hi = bounds[..., 0].min(axis=0), bounds[..., 1].max(axis=0)
        return (lo + hi) / 2, float(np.linalg.norm(hi - lo))

    def _color(self, mesh, values, cmap, clim):
        pv = _pyvista()
        field = mesh.get_array(values)
        lut = self.colors.get(values)
        if lut is None:
            lut = self.colors[values] = pv.LookupTable(cmap or pv.global_theme.cmap)
        elif cmap is not None:
            lut.cmap = cmap
        if _text(field):
            names = self._categories.setdefault(values, [])
            names += sorted(set(field.tolist()) - set(names) - {""})
            code = {name: i for i, name in enumerate(names)}
            codes = np.array([code.get(v, np.nan) for v in field.tolist()], dtype=float)
            mesh = mesh.copy(deep=False)
            (mesh.cell_data if _on_cells(mesh, values) else mesh.point_data)[values] = codes
            lut.apply_cmap(lut.cmap, len(names))
            lut.annotations = dict(enumerate(names))
            self._ranges[values] = (-0.5, len(names) - 0.5)
        elif clim is not None:
            self._pinned.add(values)
            self._ranges[values] = tuple(clim)
        elif values not in self._pinned:
            lo, hi = self._ranges.get(values, (np.inf, -np.inf))
            self._ranges[values] = (min(lo, float(np.nanmin(field))), max(hi, float(np.nanmax(field))))
        return mesh, lut

    def show(self, *, browser=False, **kwargs):
        """Opens the scene: trame in Jupyter, a native window otherwise, or the browser.

        In Google Colab the scene renders on the kernel with trame and streams through Colab's port proxy, so
        `section_drawer`, `layer_toggles` and `motion_quality` work there too; without trame it falls back to a
        static view that only turns and zooms.

        Parameters
        ----------
        browser : bool, default False
            Writes the scene to an interactive HTML page with trame and opens it in the web browser.
        **kwargs
            Passed to ``plotter.show``; ``auto_close`` defaults to False, so `screenshot` works after ``q``.

        Returns
        -------
        pathlib.Path or None
            The HTML page when `browser`, else what ``plotter.show`` returns.
        """
        if not browser and "google.colab" in sys.modules and "jupyter_backend" not in kwargs:
            try:
                return self.plotter.show(
                    jupyter_backend="server",
                    jupyter_kwargs={"handler": _colab_iframe},
                    auto_close=False,
                    **kwargs,
                )
            except ImportError:
                print("the 3D tools need trame (pip install 'boitata[3d]'); showing a static view")
                return self.plotter.show(jupyter_backend="html", auto_close=False, **kwargs)
        if not browser:
            return self.plotter.show(**{"auto_close": False, **kwargs})
        path = Path(tempfile.mkdtemp()) / "scene.html"
        try:
            getattr(self.plotter, "trame", self.plotter).export_html(path)
        except ImportError as e:
            raise ImportError("Scene.show(browser=True) needs trame: pip install 'pyvista[jupyter]'") from e
        webbrowser.open(path.as_uri())
        return path

    def screenshot(self, path, *, scale=1, transparent=False):
        """Writes the scene's current view to an image.

        Works off-screen, while a trame view is open, and after the native window of `show` is closed with ``q``;
        its close button destroys the window, so take screenshots before.

        Parameters
        ----------
        path : str or pathlib.Path
            Image file; ``.png`` keeps transparency.
        scale : int, default 1
            Resolution multiplier on the window size.
        transparent : bool, default False
            Transparent background.

        Returns
        -------
        pathlib.Path
        """
        path = Path(path)
        self.plotter.screenshot(path, transparent_background=transparent, return_img=False, scale=scale)
        return path


def _scene(plotter):
    if isinstance(plotter, Scene):
        return plotter
    return (_SCENES.get(plotter) if plotter is not None else None) or Scene(plotter=plotter)


def plot(data, values=None, *, plotter=None, **kwargs):
    """Adds a container to a 3D scene.

    Parameters
    ----------
    data : PointSet, Drillholes, BlockModel, Mesh or pyvista.DataObject
        What to draw; containers go through `to_pyvista`.
    values : str, optional
        Attribute that colors it; its nulls never render.
    plotter : Scene or pyvista.Plotter, optional
        Scene to add to; a new one by default.
    **kwargs
        Passed to `Scene.add`.

    Returns
    -------
    Scene
    """
    return _scene(plotter).add(data, values, **kwargs)


def slices(model, values=None, *, x=None, y=None, z=None, plotter=None, **kwargs):
    """Three orthogonal slices through a block model.

    Parameters
    ----------
    model : BlockModel
        Block model to cut.
    values : str, optional
        Attribute that colors the slices; its nulls never render.
    x, y, z : float, optional
        World coordinates the slices pass through; the model's center by default.
    plotter : Scene or pyvista.Plotter, optional
        Scene to add to; a new one by default.
    **kwargs
        Passed to `Scene.add`.

    Returns
    -------
    Scene
    """
    mesh = to_pyvista(model)
    center = mesh.center
    at = [center[a] if v is None else v for a, v in enumerate((x, y, z))]
    return plot(mesh.slice_orthogonal(x=at[0], y=at[1], z=at[2]), values, plotter=plotter, **kwargs)
