"""
# live scene

`bt.plot3d.Scene` stacks layers in one 3D view. layers colored by the same variable share its color map, range and
color bar, so a composite and the blocks around it show one grade in one color, and null values never draw, in any
representation. in Jupyter, Colab or VS Code the scene is a widget whose panel, filters, sections and clicks sync
back to python; from a script `show()` opens it as a self-contained page in the web browser and `save` writes that
page. `screenshot` renders the scene to a PNG in headless Chromium, as the widget last showed it.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import boitata as bt
import numpy as np
from common import GRAY, LIGHT, show

# %% [markdown]
# zinc composites of the stacked sulphide lenses inform a grid rotated with them, by inverse distance within 60 m.
# both layers carry `ZN_PCT`, so they share one color bar. without `clim` its range would run from the lowest to the
# highest value of any layer; here the first layer fixes it at 0 to 10 % and the later ones follow. the blocks with
# no composite in reach are null and never draw; a `filter` keeps the blocks above 2 % from the start, and the
# layer's filter panel edits it while viewing.

# %%
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes(data["collars"], data["surveys"], data["assays"])
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
composites = holes.composite(2.0, ["ZN_PCT"])
grid = bt.BlockModel.from_extents(*lenses, size=(20, 20, 10), buffer=20, rotation=(22.5, 0.0, 55.0))
search = bt.Search(radius=60, min_samples=1, max_samples=12)
idw = bt.InverseDistance(search, power=2).fit(composites.coords, composites["ZN_PCT"])
grid = grid.with_column("ZN_PCT", idw.predict(grid))
near = np.all(
    (composites.coords > grid.coords.min(axis=0)) & (composites.coords < grid.coords.max(axis=0)),
    axis=1,
)

scene = bt.plot3d.Scene()
scene.add(composites.filter(near), "ZN_PCT", name="composites", point_size=4, clim=(0, 10), label="Zn (%)")
scene.add(grid, "ZN_PCT", name="grid", opacity=0.6, filter={"ZN_PCT": (2, None)})
for i, lens in enumerate(lenses, 1):
    scene.add(lens, name=f"lens {i}", representation="wireframe", color=LIGHT, opacity=0.3)
scene.view(azimuth=305, dip=30)
print(f"{np.isnan(grid['ZN_PCT']).mean():.0%} of {len(grid):,} blocks null")
print(f"filters: {scene.filters}")
show(scene, "grade", "Zn composites and the grid blocks above 2 %, on one scale")

# %% [markdown]
# text columns share a category list the same way. each composite inside a lens takes its name, the others stay null
# and do not draw; the sub-blocks of the lenses carry the same names in `domain`, here renamed `lens` so both layers
# color by one variable and every lens keeps its color across them.

# %%
inside = [lens.contains(composites.coords) for lens in lenses]
names = np.select(inside, [f"lens {i}" for i in (1, 2, 3)], "")
composites = composites.with_column("lens", [n or None for n in names])
blocks = bt.BlockModel.from_meshes(
    grid.origin,
    grid.size,
    grid.count,
    [(lens, "inside", f"lens {i}") for i, lens in enumerate(lenses, 1)],
    subgrid=2,
    fill="host",
    rotation=tuple(grid.rotation),
)
blocks = blocks.filter(np.asarray(blocks["domain"], dtype=object) != "host")
blocks = blocks.with_column("lens", blocks["domain"])

scene = bt.plot3d.Scene()
scene.add(blocks, "lens", name="sub-blocks", representation="wireframe", opacity=0.3)
scene.add(composites, "lens", name="composites", point_size=5)
scene.view(azimuth=305, dip=30)
print(f"{len(blocks):,} sub-blocks, {(names != '').sum():,} of {len(composites):,} composites in a lens")
show(scene, "lenses", "Lens of each composite and sub-block, one color per lens")

# %% [markdown]
# drill holes with intervals draw one segment per interval, split at the survey stations so it follows the trace;
# unassayed ground has no interval and leaves a gap, as would a null `ZN_PCT`. holes without intervals draw their
# traces instead. `"tubes"` turns the segments into shaded cylinders, `radius` meters wide. here the holes collared
# within 30 m of a north-south line through the middle: thin gray traces, thick assays on top.

# %%
x = np.asarray(data["collars"]["X"])
fence = data["collars"].filter(np.abs(x - np.median(x)) < 30)
fence_holes = bt.Drillholes(fence, data["surveys"], data["assays"])
scene = bt.plot3d.Scene()
scene.add(bt.Drillholes(fence, data["surveys"]), name="traces", color=GRAY, line_width=1)
scene.add(fence_holes, "ZN_PCT", name="assays", representation="tubes", radius=4, clim=(0, 5), label="Zn (%)")
for i, lens in enumerate(lenses, 1):
    scene.add(lens, name=f"lens {i}", representation="wireframe", color=LIGHT, opacity=0.2)
scene.view(azimuth=300, dip=20)
print(f"{len(fence_holes.holes)} of {len(holes.holes)} holes, {len(fence_holes.samples()):,} intervals")
show(scene, "holes", "Zn assays down a fence of holes, as tubes")

# %% [markdown]
# ## inspecting and reading back
#
# a click on a block, point, interval or triangle opens a card with every column sent for its row, its coordinates,
# and for a block its size, and outlines it; Esc or a click on empty space closes it. in a notebook `picked` reads
# the last click back, while `filters` and `sections` follow the panel's edits and the cuts drawn in the view, both
# ways: setting them updates the widgets shown. `columns=` limits what a layer sends, and so what the card lists.
# from a script nothing is clicked, so `picked` is None.

# %%
scene = bt.plot3d.Scene()
scene.add(grid, "ZN_PCT", name="grid", columns=["ZN_PCT"], clim=(0, 10), label="Zn (%)")
scene.filters = {"grid": {"ZN_PCT": (4, None)}}
scene.view(azimuth=305, dip=30)
print(f"picked: {scene.picked}")
print(f"filters: {scene.filters}")
show(scene, "rich", "Blocks above 4 % Zn, set through `filters`")
