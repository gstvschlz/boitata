"""
# interactive sections

`Scene.section` cuts every layer of a scene to a slab around a section: blocks are cut and capped in their own
colors, lens surfaces are cut and outlined where the slab's faces cross them, and drill-hole intervals show whole
when their middle lies in the slab. the section runs along a polyline in plan and may dip; `U` in the viewer, or
`unfolded` in `sections`, lays it out flat as one true-scale section. while viewing, Shift-drag cuts a straight
section and `S` draws a polyline. here each view renders to an image.
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
# zinc composites of the stacked sulphide lenses inform a grid rotated with them. the scene holds the grid, the drill
# holes and the three lens surfaces; one section across the lenses, running at azimuth 112.5 and dipping 80 to its
# right, cuts them all. whatever lies more than 15 m from it drops out.

# %%
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes.from_tables(data)
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
composites = holes.composite(2.0, ["ZN_PCT"])
grid = bt.BlockModel.from_extents(*lenses, size=(20, 20, 10), buffer=20, rotation=(22.5, 0.0, 55.0))
search = bt.Search(radius=60, min_samples=1, max_samples=12)
idw = bt.InverseDistance(search, power=2).fit(composites, "ZN_PCT")
grid = grid.with_column("ZN_PCT", idw.predict(grid))

scene = bt.plot3d.Scene()
scene.add(grid, "ZN_PCT", name="grid", clim=(0, 10), label="Zn (%)")
scene.add(holes, name="holes", representation="tubes", color=INK, radius=2)
for i, lens in enumerate(lenses, 1):
    scene.add(lens, name=f"lens {i}", color="white")
center = grid.coords.mean(axis=0)
run = np.radians(112.5)
reach = 400 * np.array([np.sin(run), np.cos(run), 0])
scene.section([center - reach, center + reach], width=30, dip=80)
scene.view(azimuth=200, dip=35)
print(f"section through {np.round(center).tolist()}: {scene.sections['dip']:.0f} degrees, 30 m wide")
show(scene, "cut", "One section cuts the grid, the lenses and the holes near it")

# %% [markdown]
# unfolded, the section lies flat in front of the camera: the distance along it runs on the bottom axis and the
# elevation up, at true scale.

# %%
scene.sections = {**scene.sections, "unfolded": True}
show(scene, "section", "The same section, unfolded")

# %% [markdown]
# a polyline in plan cuts a section along each segment, between its ends, so the section steps around the lenses
# like a curtain. here it runs across the lenses, turns north along strike, then crosses them again. in the viewer
# `S` (or the draw button) switches to a plan view where clicks add the vertices, Shift locks a segment to a multiple
# of 45 degrees, Backspace removes the last vertex and Enter cuts; Shift+wheel sets the width and `X` clears the
# section.

# %%
x, y, z = center
path = [(x - 250, y - 150, z), (x + 100, y - 150, z), (x + 100, y + 150, z), (x - 250, y + 150, z)]
scene.section(path, width=30)
scene.view(azimuth=330, dip=40)
show(scene, "curtain", "A stepped curtain along a polyline")

# %% [markdown]
# unfolding the curtain lays its three panels end to end, one true-scale section along the polyline.

# %%
scene.sections = {**scene.sections, "unfolded": True}
print(f"{len(scene.sections['points'])} vertices, unfolded: {scene.sections['unfolded']}")
show(scene, "unfolded", "The curtain unfolded into one section")
