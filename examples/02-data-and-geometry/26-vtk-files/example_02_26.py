"""
# VTK files

`write_vtk` writes a `PointSet`, `Polylines` or `Mesh` as a VTK XML `.vtp` (polydata) or `.vtu` (unstructured grid)
file, which 3D viewers and VTK-based tools open with their columns. `read_vtk` reads either back to the same
container, whether the file is ASCII, binary or appended, compressed or not. you write a vein of the grade-control
dataset with a column per vertex and per triangle, and points around it with their distance to the vein.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import tempfile

import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import INK, map_axes, save

vein = bt.read_mesh(bt.datasets.fetch("mining/3d/vein-gold-grade-control/vein_V1.stl"))
elevation = vein.z.copy()
elevation[::50] = np.nan
a, b, c = vein.coords[vein.triangles].transpose(1, 0, 2)
normal = np.cross(b - a, c - a)
dip = np.degrees(np.arccos(np.abs(normal[:, 2]) / np.linalg.norm(normal, axis=1)))
vein = vein.with_vertex_column("elevation", elevation).with_face_column("dip", dip)
print(vein)

rng = np.random.default_rng(7)
(x0, y0, z0), (x1, y1, z1) = vein.bounds
coords = rng.uniform([x0, y0, z0], [x1, y1, z1], size=(300, 3))
points = bt.PointSet(coords, attributes={"distance": vein.distance(coords)})

# %% [markdown]
# point and cell data become columns. VTK has no nulls, so a column with nulls is written with NaN in their place
# and a `<name>_valid` array of ones and zeros, which `read_vtk` turns back into nulls. binary files append the
# arrays raw after the XML; `ascii=True` writes them as text.

# %%
folder = Path(tempfile.mkdtemp())
for file, data, options in [
    ("vein.vtp", vein, {}),
    ("vein.vtu", vein, {"ascii": True}),
    ("points.vtp", points, {}),
]:
    path = folder / file
    bt.write_vtk(path, data, **options)
    back = bt.read_vtk(path)
    same = all(np.array_equal(back[c], data[c], equal_nan=True) for c in data.column_names)
    print(f"{file:>11}: {path.stat().st_size / 1e3:6.0f} kB, {type(back).__name__}, same columns: {same}")
back = bt.read_vtk(folder / "vein.vtp")
print("nulls in elevation:", np.isnan(back["elevation"]).sum())

# %%
fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
image = ax.tripcolor(back.x, back.y, back.triangles, facecolors=back["dip"], cmap="cividis")
near = points["distance"] < 10
ax.scatter(points.x[near], points.y[near], s=8, color=INK, label="points within 10 m")
fig.colorbar(image, ax=ax, shrink=0.8, label="triangle dip, degrees (cell data)")
ax.legend(loc="upper right")
map_axes(ax, "Vein V1 read back from vein.vtp")
save(fig, "vtk")
