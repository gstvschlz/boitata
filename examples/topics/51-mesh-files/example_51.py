"""
# 51. Mesh files

`read_mesh` and `write_mesh` handle OBJ, STL (binary or ASCII) and DXF, chosen by the file extension.
`Mesh.analysis` reports what keeps a mesh from being a solid, and `Mesh.repair` fixes what it can. Here the four
gold veins of the grade-control dataset are read from their STL files, written back in every format, and one is
broken into loose triangles and repaired.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import tempfile

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import save

# %% [markdown]
# `cs.datasets.fetch` gives the local path of a dataset file. Reading welds corners repeated between triangles, as
# STL stores each triangle with its own three corners; the veins come back closed, with a volume and an area.

# %%
veins = {}
for name in ("V1", "V2", "V3", "V4"):
    path = cs.datasets.fetch(f"mining/3d/vein-gold-grade-control/vein_{name}.stl")
    veins[name] = cs.read_mesh(path)
    mesh = veins[name]
    print(f"{name}: {path.stat().st_size / 1e6:.1f} MB, {mesh}, {mesh.volume:,.0f} m3, {mesh.area:,.0f} m2")

# %% [markdown]
# The veins in plan at 700 m and on an east–west section at northing 15 000 m:

# %%
planes = {
    "Plan at 700 m": ((0, 0, 700), 90, 0),
    "Section at northing 15 000 m": ((0, 15000, 0), 90, 90),
}
fig, axes = plt.subplots(1, 2, figsize=(8, 6), layout="constrained")
for ax, (title, plane) in zip(axes, planes.items()):
    cs.plot.slab(np.empty((0, 3)), plane=plane, thickness=1, meshes=list(veins.values()), ax=ax)
    normal = np.array([0, 0, 1]) if plane[2] == 0 else np.array([0, 1, 0])
    along = [0, 1] if plane[2] == 0 else [0, 2]
    for name, mesh in veins.items():
        near = np.abs((mesh.vertices - plane[0]) @ normal) < 5
        if near.any():
            top = mesh.vertices[near][:, along]
            ax.annotate(name, top[top[:, 1].argmax()], xytext=(0, 3), textcoords="offset points", ha="center")
    ax.set_title(title)
save(fig, "veins")

# %% [markdown]
# Each format round-trips the vein. Binary STL stores single precision, so vertices can move by a fraction of a
# millimeter at mine coordinates; these files were written in single precision and come back unchanged. DXF
# writes 3D faces and reads them back with a `layer` column per triangle.

# %%
v1 = veins["V1"]
with tempfile.TemporaryDirectory() as folder:
    for file, options in [("v1.stl", {}), ("v1_ascii.stl", {"ascii": True}), ("v1.obj", {}), ("v1.dxf", {})]:
        path = Path(folder) / file
        cs.write_mesh(path, v1, **options)
        back = cs.read_mesh(path)
        shift = np.abs(back.vertices[back.triangles] - v1.vertices[v1.triangles]).max()
        print(
            f"{file:>12}: {path.stat().st_size / 1e6:5.1f} MB, {len(back.triangles)} triangles,"
            f" {back.volume:,.0f} m3, largest shift {shift * 1000:.3f} mm, columns {back.face_attributes.column_names}"
        )

# %% [markdown]
# ## Repair
#
# A solid from elsewhere can arrive as loose triangles: each with its own copy of its corners, rounded
# differently, and wound either way. It shares no edges, so it is not closed and has no volume. Here V1 is broken
# that way, with corners moved by about 0.01 mm. `repair` welds corners within `tolerance`, drops degenerate and
# repeated triangles, and winds each piece consistently, outward where it is closed. The tolerance has to exceed
# the rounding and stay below the shortest edge, or welding collapses triangles and opens new holes:

# %%
rng = np.random.default_rng(7)
corners = v1.vertices[v1.triangles] + rng.normal(0, 1e-5, (len(v1.triangles), 3, 3))
loose = np.arange(3 * len(v1.triangles)).reshape(-1, 3)
flip = rng.random(len(loose)) < 0.5
loose[flip] = loose[flip, ::-1]
broken = cs.Mesh(corners.reshape(-1, 3), loose)
edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
print(broken, broken.analysis)
print(f"shortest edge {edges.min() * 1000:.1f} mm")
for tolerance in (1e-6, 1e-4, 5e-3):
    attempt = broken.repair(tolerance=tolerance)
    print(f"tolerance {tolerance:g} m: {attempt}, {attempt.analysis['boundary_edges']} boundary edges")
repaired = broken.repair(tolerance=1e-4)
print(f"repaired at 0.1 mm: {repaired.volume:,.0f} m3, original {v1.volume:,.0f} m3")
