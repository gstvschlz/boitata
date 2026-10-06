"""Accessors every container shares, derived from its Rust methods."""

import numpy as np

from boitata._boitata import BlockModel, Drillholes, Mesh, PointSet, Polylines


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
