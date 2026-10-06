"""Accessors every container shares, derived from its Rust methods."""

import numpy as np

from boitata._boitata import BlockModel, Drillholes, Mesh, PointSet, Polylines, Table
from boitata._columns import column as _column
from boitata.errors import InvalidInput


def _axis(a):
    return property(lambda self: self.coords[:, a], doc=f"{'xyz'[a]} of each row's coordinates.")


def _bounds(self):
    paths = self.paths()
    if not len(paths):
        return None
    xyz = np.column_stack([paths["x"], paths["y"], paths["z"]])
    return xyz.min(0).tolist(), xyz.max(0).tolist()


for _cls in (PointSet, Polylines, BlockModel, Mesh, Drillholes):
    _cls.x, _cls.y, _cls.z = (_axis(a) for a in range(3))

for _cls in (PointSet, Polylines, BlockModel):
    _cls.column_names = property(
        lambda self: self.attributes.column_names, doc="Names of the attribute columns."
    )

Mesh.column_names = property(
    lambda self: self.vertex_attributes.column_names + self.face_attributes.column_names,
    doc="Names of the vertex columns, then the face columns.",
)

Drillholes.coords = property(lambda self: self.samples().coords, doc="``(n, 3)`` interval midpoints.")
Drillholes.bounds = property(
    _bounds, doc="``(min, max)`` corners of the desurveyed holes, None without stations."
)
Drillholes.column_names = property(
    lambda self: self.samples().column_names, doc="Names of the interval columns."
)
Drillholes.to_table = lambda self: self.samples().to_table()
Drillholes.to_table.__doc__ = "Interval midpoints ``x``, ``y``, ``z`` and the interval columns."
Drillholes.__getitem__ = lambda self, name: self.samples()[name]

for _cls in (PointSet, Polylines, BlockModel, Drillholes):
    _cls.to_polars = lambda self: self.to_table().to_polars()
    _cls.to_pandas = lambda self: self.to_table().to_pandas()
    _cls.to_pyarrow = lambda self: self.to_table().to_pyarrow()


def _sample(self, points, column):
    """Value of `column` (a name or one value per row) at each point; NaN, or None
    for text, outside the model or in a missing block."""
    rows = self.row_at(points)
    values = np.asarray(_column(self, column, "column"))
    out = values[np.maximum(rows, 0)]
    out = out.astype(object if values.dtype == object else float)
    out[rows < 0] = None if values.dtype == object else np.nan
    return out


def _grid(self, column):
    """`column` (a name or one value per row) on the full grid as an ``(nz, ny, nx)``
    array, NaN (None for text) in absent cells; for regular and masked models."""
    if self.extents is not None:
        raise InvalidInput("grid needs a regular or masked model; see to_regular")
    values = np.asarray(_column(self, column, "column"))
    nx, ny, nz = self.count
    text = values.dtype == object
    out = np.full(nx * ny * nz, None if text else np.nan, dtype=object if text else float)
    out[np.arange(len(self)) if self.index is None else self.index] = values
    return out.reshape(nz, ny, nx)


def _drop_null(self, *columns):
    """Rows with a value in every one of `columns`, all columns by default."""
    keep = np.ones(len(self), dtype=bool)
    for name in columns or self.column_names:
        values = _column(self, name, "columns")
        keep &= values != None if values.dtype == object else ~np.isnan(values.astype(float))
    return self.filter(keep)


def _from_tables(tables, *, intervals="assays", **options):
    """Drillholes from a mapping with ``collars`` and ``surveys`` tables, plus the
    `intervals` table (None for none), as the datasets return them; `options` go
    to the constructor."""
    return Drillholes(
        tables["collars"], tables["surveys"], None if intervals is None else tables[intervals], **options
    )


BlockModel.sample = _sample
BlockModel.contains = lambda self, points: self.row_at(points) >= 0
BlockModel.contains.__doc__ = "Whether each point falls in a block of the model."
BlockModel.grid = _grid
for _cls in (Table, PointSet, Polylines, BlockModel):
    _cls.drop_null = _drop_null
Drillholes.from_tables = staticmethod(_from_tables)
