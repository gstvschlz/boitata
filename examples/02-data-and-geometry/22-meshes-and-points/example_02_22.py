"""
# meshes and points in a scene

a `Mesh` adds to a `bt.plot3d.Scene` as a surface, a wireframe or its vertices, and a `PointSet` as points of a fixed
screen size or as shaded spheres. a mesh is colored by a vertex attribute, interpolated across each triangle, or by
a face attribute, one color per triangle; either way its null rows never draw. here a topography surface colored by
elevation sits above three stacked sulphide lenses and the zinc composites that cut them.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import boitata as bt
import numpy as np
from common import INK, show

# %% [markdown]
# `topography` triangulates the collar elevations ([topography](../../02-data-and-geometry/19-topography/README.md)) into a mesh; its
# elevation goes in as a vertex column. the lenses are closed solids, drawn half transparent. the composites keep
# only those inside a lens, drawn as spheres 12 m across and colored by zinc on a second color bar.

# %%
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes.from_tables(data)
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
collars = bt.PointSet.from_table(data["collars"], z="Z")
ground = bt.topography(collars, cell=20.0).mesh
ground = ground.with_vertex_column("elevation", ground.z)
composites = holes.composite(2.0, ["ZN_PCT"])
ore = composites.filter(np.any([lens.contains(composites.coords) for lens in lenses], axis=0))
print(f"topography: {len(ground.coords):,} vertices, {len(ground.triangles):,} triangles")
print(f"{len(ore):,} composites inside a lens, {np.isnan(ore['ZN_PCT']).sum()} of them null")

scene = bt.plot3d.Scene()
scene.add(ground, "elevation", name="topography", cmap="cividis", opacity=0.6, label="z (m)")
for i, lens in enumerate(lenses, 1):
    scene.add(lens, name=f"lens {i}", color=INK, opacity=0.2)
scene.add(
    ore, "ZN_PCT", name="ore composites", representation="spheres", radius=6, clim=(0, 10), label="Zn (%)"
)
scene.view(azimuth=305, dip=25)
show(scene, "scene", "Topography by elevation, the three lenses and their Zn composites")
