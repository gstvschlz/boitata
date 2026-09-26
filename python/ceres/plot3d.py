"""3D views on pyvista (``pip install ceres[3d]``).

`to_pyvista` converts a container to a pyvista dataset with its attributes as point or cell data; the plotting
functions add it to `plotter` when given, else to a new one, and return the plotter.
"""

from itertools import pairwise

import numpy as np

from ceres._ceres import BlockModel, Drillholes, Mesh, PointSet

__all__ = ["plot", "slices", "to_pyvista"]

_HEX = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])


def _pyvista():
    try:
        import pyvista
    except ImportError as e:
        raise ImportError("ceres.plot3d needs pyvista: pip install ceres[3d]") from e
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
        if isinstance(values, list):
            values = np.array(["" if v is None else v for v in values])
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
        hole = np.array(paths["hole"])
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


def _plotter(plotter):
    return _pyvista().Plotter() if plotter is None else plotter


def plot(data, scalars=None, plotter=None, **kwargs):
    """Adds a container to a 3D scene.

    Parameters
    ----------
    data : PointSet, Drillholes, BlockModel, Mesh or pyvista.DataObject
        What to draw; containers go through `to_pyvista`.
    scalars : str, optional
        Attribute that colours it.
    plotter : pyvista.Plotter, optional
        Scene to add to; a new one by default.
    **kwargs
        Passed to ``plotter.add_mesh``; points are drawn as spheres unless set otherwise.

    Returns
    -------
    pyvista.Plotter
    """
    plotter = _plotter(plotter)
    mesh = data if isinstance(data, _pyvista().DataObject) else to_pyvista(data)
    if isinstance(data, PointSet):
        kwargs.setdefault("render_points_as_spheres", True)
        kwargs.setdefault("point_size", 6)
    plotter.add_mesh(mesh, scalars=scalars, **kwargs)
    return plotter


def slices(model, scalars=None, x=None, y=None, z=None, plotter=None, **kwargs):
    """Three orthogonal slices through a block model.

    Parameters
    ----------
    model : BlockModel
        Block model to cut.
    scalars : str, optional
        Attribute that colours the slices.
    x, y, z : float, optional
        World coordinates the slices pass through; the model's centre by default.
    plotter : pyvista.Plotter, optional
        Scene to add to; a new one by default.
    **kwargs
        Passed to ``plotter.add_mesh``.

    Returns
    -------
    pyvista.Plotter
    """
    mesh = to_pyvista(model)
    centre = mesh.center
    at = [centre[a] if v is None else v for a, v in enumerate((x, y, z))]
    return plot(mesh.slice_orthogonal(x=at[0], y=at[1], z=at[2]), scalars=scalars, plotter=plotter, **kwargs)
