"""3D views on pyvista (the ``3d`` extra).

`to_pyvista` converts a container to a pyvista dataset with its attributes as point or cell data. A `Scene` holds
layers that share one color map and range per variable; `plot` and `slices` add a layer to the scene of `plotter`
when given, else to a new one, and return the scene.
"""

import tempfile
import weakref
import webbrowser
from functools import singledispatch
from itertools import pairwise
from pathlib import Path

import numpy as np

from boitata._boitata import BlockModel, Drillholes, Mesh, PointSet
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


def _columns(table):
    for name in table.column_names:
        values = table[name]
        if values.dtype == object:
            values = np.where(values == None, "", values).astype(str)
        yield name, values


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
def _(data: PointSet):
    return to_pyvista(data), {"render_points_as_spheres": True, "point_size": 6}


@_layer.register
def _(data: Drillholes, *, radius=None):
    if data.interval_columns is None:
        lines = to_pyvista(data)
    else:
        table = data.samples().attributes
        hole, start, end = data.interval_columns
        holes = list(table[hole])
        ends = np.hstack([data.at(holes, table[start]), data.at(holes, table[end])]).reshape(-1, 3)
        segments = np.arange(len(ends)).reshape(-1, 2)
        lines = _pyvista().PolyData(ends, lines=np.c_[np.full(len(segments), 2), segments].ravel())
        for name, values in _columns(table):
            lines.cell_data[name] = values
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


def _detail(pv, renderer, mapper, full, values):
    """Above the GPU memory budget, a coarse copy of `full` draws while the camera moves, `full` once it stops."""
    budget = mapper.GetMaxMemoryInBytes() * mapper.GetMaxMemoryFraction()
    step = int(np.ceil((full.point_data[values].nbytes / budget) ** (1 / 3)))
    if step <= 1:
        return
    cube = full.point_data[values].reshape(full.dimensions, order="F")[::step, ::step, ::step]
    coarse = _image(pv, cube, full.origin, np.multiply(full.spacing, step), full.direction_matrix, values)
    window = renderer.GetRenderWindow

    def switch(*_):
        moving = window().GetDesiredUpdateRate() >= mapper.GetInteractiveUpdateRate()
        mapper.SetInputData(coarse if moving else full)
        mapper.SetRequestedRenderMode(mapper.GPURenderMode if moving else mapper.RayCastRenderMode)

    switch()
    renderer.AddObserver("StartEvent", switch)


_SCENES = weakref.WeakKeyDictionary()


class Scene:
    """Live 3D scene: layers colored by a variable share its color map and range.

    Parameters
    ----------
    plotter : pyvista.Plotter, optional
        Plotter to draw into; a new one by default.
    **kwargs
        Passed to ``pyvista.Plotter`` when `plotter` is not given (e.g. ``window_size``, ``off_screen``).

    Attributes
    ----------
    plotter : pyvista.Plotter
        Attributes the scene does not have are looked up on it (``view_vector``, ``close``, ...).
    colors : dict of str to pyvista.LookupTable
        Color map and range of each variable, shared by every layer it colors.
    """

    def __init__(self, *, plotter=None, **kwargs):
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
        _SCENES[self.plotter] = self

    def __getattr__(self, name):
        if name == "plotter":
            raise AttributeError(name)
        return getattr(self.plotter, name)

    def add(self, data, values=None, *, style=None, radius=None, labels=False, **kwargs):
        """Adds a layer.

        Parameters
        ----------
        data : PointSet, Drillholes, BlockModel, Mesh or pyvista.DataObject
            What to draw; containers go through `to_pyvista`.
        values : str, optional
            Attribute that colors it. Its null rows never render: null points, cells, and the cells of null points
            are left out. Numbers share one range over every layer the variable colors, text one category list.
        style : {"surface", "wireframe", "points", "points_gaussian", "volume"}, optional
            How cells are drawn; a `PointSet` draws its points as spheres. ``"volume"`` renders a regular block
            model colored by `values` on the GPU, each block a uniform cube; a model too large for GPU memory draws
            a coarser copy while the camera moves and the full one once it stops.
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
        self._layers.append((mesh, flat, values, style, kwargs, actor))
        if labels:
            self.plotter.add_point_labels(*_collars(data), shape=None, show_points=False, always_visible=True)
        return self

    def _draw(self, mesh, values, style, kwargs):
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
        _detail(pv, self.plotter.renderer, actor.mapper, actor.mapper.dataset, values)
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

    def section(self, origin, *, azimuth=90.0, dip=90.0, width=None):
        """Cuts every layer with a plane, in place of the layers and of the previous section.

        Surfaces and solids show their intersection with the plane; points and drill holes within `width` of it
        are clipped to that slab and projected onto it. Each piece keeps its layer's variable and style.

        Parameters
        ----------
        origin : array_like or None
            ``(x, y, z)`` point on the plane; None removes the section and shows the layers again.
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
        for actor in self._cuts:
            self.plotter.remove_actor(actor, render=False)
        self._cuts = []
        self._plane = None if origin is None else (origin, azimuth, dip)
        for *_, actor in self._layers:
            if actor is not None:
                actor.SetVisibility(origin is None)
        if origin is None:
            return self
        center, _, _, n = _frame(self._plane)
        half = (self._extent()[1] / 20 if width is None else width) / 2
        for mesh, flat, values, style, kwargs, _ in self._layers:
            if flat:
                cut = mesh.clip(normal=n, origin=center + half * n).clip(normal=-n, origin=center - half * n)
                cut.points = cut.points - np.outer((cut.points - center) @ n, n)
            else:
                cut = mesh.slice(normal=n, origin=center)
            if style == "volume":
                style, kwargs = None, {k: v for k, v in kwargs.items() if k != "opacity"}
            if cut.n_points:
                actor = self._draw(cut, values, style, dict(kwargs))
                self._cuts += [actor] if actor is not None else []
        return self

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
        """Orthographic view normal to the section plane, strike to the right and up dip upward.

        Returns
        -------
        Scene
            This scene, to chain calls.
        """
        if self._plane is None:
            raise RuntimeError("no section: call Scene.section first")
        center, _, v, n = _frame(self._plane)
        self.plotter.enable_parallel_projection()
        self.plotter.camera_position = [center + n, center, v]
        self.plotter.reset_camera()
        return self

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
