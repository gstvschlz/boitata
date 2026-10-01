"""
# block models

a `BlockModel` is a grid of boxes, the blocks, with one attribute row per block. three numbers per axis define
the grid (origin, block size and block count), plus three angles when it is rotated. the blocks come in three
layouts: a regular model holds every cell of the grid, a masked model a subset of them, and a sub-blocked model
splits some cells into smaller boxes. you build each one here on grids small enough to count by eye, then read a
model back from centroids and from a file.
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
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, map_axes, save
from matplotlib.collections import PolyCollection


def draw(ax, model, color=LIGHT, edge="white", **style):
    outlines = model.corners[:, [0, 1, 3, 2], :2]
    ax.add_collection(PolyCollection(outlines, facecolors=color, edgecolors=edge, **style))
    ax.autoscale_view()


# %% [markdown]
# ## a regular grid
#
# the grid has 5 × 4 × 3 blocks of 10 × 10 × 5 m. its origin is the outer corner of the first block (the one with
# the smallest x, y and z), not that block's center. the first centroid sits half a block from the origin along
# each axis.

# %%
model = bt.BlockModel(origin=(1000, 2000, 300), size=(10, 10, 5), count=(5, 4, 3))
print(model)
print("first corner:  ", model.corners[0, 0])
print("first centroid:", model.centroids[0])

# %% [markdown]
# ## cell index, (i, j, k) and centroid
#
# each cell has a linear index that runs along x first, then y, then z: n = i + nx (j + ny k). a regular model
# stores its rows in that order and stores no coordinates, so row n is cell n. integer division takes an index
# back to (i, j, k), and the centroid is the origin plus (i + ½, j + ½, k + ½) blocks.

# %%
nx, ny, nz = model.count
n = np.arange(len(model))
i, j, k = n % nx, n // nx % ny, n // (nx * ny)
centroids = np.array(model.origin) + (np.column_stack([i, j, k]) + 0.5) * model.size
print("index 27 is (i, j, k) =", (int(i[27]), int(j[27]), int(k[27])), "at", centroids[27])
print("same centroids as the model:", np.allclose(centroids, model.centroids))
print("back to the index:", np.array_equal(i + nx * (j + ny * k), n))

# %% [markdown]
# to go from a point to its cell, reverse the last step: subtract the origin, divide by the block size and round
# down. `row_at` does this and returns -1 for a point outside the grid.

# %%
points = np.array([[1027.0, 2013.0, 308.0], [1049.9, 2039.9, 314.9], [1055.0, 2010.0, 305.0]])
ijk = np.floor((points - model.origin) / model.size).astype(int)
print("(i, j, k):", ijk.tolist())
print("index:    ", (ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])).tolist())
print("row_at:   ", model.row_at(points).tolist())

# %% [markdown]
# the third point lies east of the last column, so its i of 5 is out of range and `row_at` gives -1. the plan shows
# the middle level (k = 1), each cell labeled with its index and (i, j), and the first point:

# %%
level = model.mask(k == 1)
fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
draw(ax, level)
for c, index in zip(level.centroids, level.index):
    ax.text(
        *c[:2],
        f"{index}\n({index % nx}, {index // nx % ny})",
        ha="center",
        va="center",
        fontsize=7,
        color=INK,
    )
ax.plot(*model.origin[:2], "s", color=ACCENT, ms=5)
ax.annotate("origin", model.origin[:2], xytext=(4, -10), textcoords="offset points", color=ACCENT)
ax.plot(*points[0, :2], "o", color=HIGHLIGHT, ms=5)
map_axes(ax, "Level k = 1: cell index and (i, j)")
save(fig, "indices")

# %% [markdown]
# ## a rotated grid
#
# `rotation` takes azimuth, dip and rake in degrees, and the grid turns about its origin. with an azimuth of 30°
# the y axis of the grid points N30°E and the x axis N120°E. the rows of `corners` give those axes: vertices 1, 2
# and 4 of a block sit one block length from vertex 0 along x, y and z. written as the columns of a matrix, the axes
# carry a position in the grid (i + ½, j + ½, k + ½ blocks) to the world, and the transposed matrix carries a world
# point back into the grid.

# %%
rotated = bt.BlockModel(origin=(1000, 2000, 300), size=(10, 10, 5), count=(5, 4, 3), rotation=(30, 0, 0))
corner = rotated.corners[0]
axes = ((corner[[1, 2, 4]] - corner[0]) / np.array(rotated.size)[:, None]).T
print("x axis", axes[:, 0].round(3), " y axis", axes[:, 1].round(3))
local = (np.column_stack([i, j, k]) + 0.5) * rotated.size
print("same centroids:", np.allclose(np.array(rotated.origin) + local @ axes.T, rotated.centroids))

point = np.array([1032.0, 2012.0, 308.0])
ijk = np.floor((point - rotated.origin) @ axes / rotated.size).astype(int)
print(
    "point in cell",
    ijk.tolist(),
    "index",
    ijk[0] + nx * (ijk[1] + ny * ijk[2]),
    "row_at",
    rotated.row_at([point]),
)

# %% [markdown]
# the same level of the rotated grid, with the grid axes drawn from the origin. indices keep their order in the
# grid frame, so cell 20 stays at the origin corner whatever the rotation:

# %%
level = rotated.mask(k == 1)
fig, ax = plt.subplots(figsize=(5.2, 5.2), layout="constrained")
draw(ax, level)
for c, index in zip(level.centroids, level.index):
    ax.text(*c[:2], index, ha="center", va="center", fontsize=7, color=INK)
for axis, name in zip(axes.T[:2], "xy"):
    ax.annotate(
        "",
        rotated.origin[:2] + 15 * axis[:2],
        rotated.origin[:2],
        arrowprops={"arrowstyle": "->", "color": ACCENT},
    )
    ax.text(*(rotated.origin[:2] + 18 * axis[:2]), name, color=ACCENT, ha="center", va="center")
ax.plot(*point[:2], "o", color=HIGHLIGHT, ms=5)
map_axes(ax, "Rotated 30°: level k = 1")
save(fig, "rotated")

# %% [markdown]
# to size a rotated grid on data instead of choosing its origin and count by hand, see
# [block model from extents](../../02-data-and-geometry/12-block-model-from-extents/README.md).
#
# ## masked layout
#
# a masked model keeps some cells of the grid and drops the rest. it stores the geometry once and the sorted
# indices of the cells present, so row and cell index part ways: `index[row]` is the cell. here a 2D grid of
# 20 × 16 blocks of 10 m keeps the cells whose centroid falls inside a domain outline. a 2D grid has one level
# (`nz = 1`), and its blocks take a unit height.

# %%
grid = bt.BlockModel(origin=(0, 0), size=(10, 10), count=(20, 16))
outline = bt.Polylines([[[25, 30], [150, 12], [190, 90], [120, 150], [40, 125]]], closed=True)
domain = grid.mask(outline.contains(grid))
print(domain)
print("first rows hold cells", domain.index[:5].tolist())

points = np.array([[62.0, 44.0], [5.0, 5.0]])
row = domain.row_at(points)
print(
    "rows", row.tolist(), "-> cell", domain.index[row[0]], "| in the full grid:", grid.row_at(points).tolist()
)

# %% [markdown]
# the second point falls in cell 0, which the mask dropped, so the masked model returns -1 where the full grid
# returns row 0. `to_regular` puts every cell back and leaves the dropped ones null.

# %%
domain = domain.with_column("cu", np.random.default_rng(7).gamma(4, 0.25, len(domain)))
full = domain.to_regular()
print(full, "| null cells:", np.isnan(full["cu"]).sum())

fig, ax = plt.subplots(figsize=(5.6, 4.6), layout="constrained")
draw(ax, grid)
draw(ax, domain, color=ACCENT)
ring = np.vstack([outline.parts[0], outline.parts[0][:1]])
ax.plot(ring[:, 0], ring[:, 1], color=HIGHLIGHT, lw=1.4)
ax.plot(*points.T, "o", color=HIGHLIGHT, ms=5)
map_axes(ax, f"Masked: {len(domain)} of {len(grid)} cells inside the outline")
save(fig, "masked")

# %% [markdown]
# masking suits models that fill a fraction of their grid, such as one domain in a large box. the
# [models larger than memory](../../02-data-and-geometry/10-large-models/README.md) page builds one level by level.
#
# ## sub-blocked layout
#
# a sub-blocked model keeps a parent grid and gives each row a parent cell and an extent inside it, as fractions of
# the parent from 0 to 1: minimum u, v, w, then maximum u, v, w. a parent can hold one row, whole or partial, or
# several smaller boxes. with `subgrid=(4, 4, 1)` each fraction must fall on quarters of a parent. here cell 0
# stays whole, cell 1 splits into four quarters and cell 4 keeps only its southern quarter.

# %%
parents = np.array([0, 1, 1, 1, 1, 4], dtype=np.uint64)
extents = [
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0.5, 0.5, 1],
    [0.5, 0, 0, 1, 0.5, 1],
    [0, 0.5, 0, 0.5, 1, 1],
    [0.5, 0.5, 0, 1, 1, 1],
    [0, 0, 0, 1, 0.25, 1],
]
cu = [0.4, 1.2, 0.9, 0.3, 0.5, 2.0]
blocks = bt.BlockModel.subblocked(
    (0, 0), (10, 10), (3, 2), parents, extents, subgrid=(4, 4, 1), attributes={"cu": cu}
)
print(blocks)
print("areas:", blocks.volumes.tolist())
merged = blocks.to_regular()
print("per parent:", merged["cu"].round(2).tolist())

fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
draw(ax, blocks, color=ACCENT)
draw(ax, bt.BlockModel(origin=(0, 0), size=(10, 10), count=(3, 2)), color="none", edge=GRAY, lw=1.2)
for c, parent, value in zip(blocks.centroids, blocks.index, cu):
    ax.text(*c[:2], f"{parent}: {value}", ha="center", va="center", fontsize=7, color="white")
map_axes(ax, "Sub-blocks in a 3 × 2 parent grid")
save(fig, "subblocks")

# %% [markdown]
# `to_regular` merged each parent back into one row. floats become area-weighted means (1.2, 0.9, 0.3 and 0.5
# average to 0.72 in cell 1), and cell 4 keeps the 2.0 of its only sub-block. cells 2, 3 and 5 hold no sub-block
# and stay null. in practice sub-blocks come from solids and surfaces; the
# [sub-blocks](../../02-data-and-geometry/06-sub-blocks/README.md) page builds them from meshes and regularizes them.
#
# ## a model from centroids
#
# block models often arrive as a table of centroids and attributes, one row per block, with the block size known
# from elsewhere. the outer centroids lie half a block inside the grid, so `from_extents` with a buffer of half a
# block recovers it, and `row_at` gives each centroid its cell. the table here is the masked domain from above,
# written as CSV.

# %%
folder = Path(tempfile.mkdtemp())
bt.write_csv(folder / "domain.csv", domain, progress=False)
table = bt.read_csv(folder / "domain.csv", progress=False)
xyz = np.column_stack([table["x"], table["y"], table["z"]])
size = (10.0, 10.0, 1.0)
box = bt.BlockModel.from_extents(xyz, size=size, buffer=np.array(size) / 2)
cells = box.row_at(xyz).astype(np.uint64)
order = np.argsort(cells)
rebuilt = bt.BlockModel(
    box.origin, size, box.count, index=cells[order], attributes={"cu": table["cu"][order]}
)
print(rebuilt)
print("origin", rebuilt.origin, "count", rebuilt.count)

# %% [markdown]
# the domain spans fewer cells than the grid it came from, so the rebuilt grid is smaller, starts elsewhere
# (origin 30, 10 instead of 0, 0) and numbers its cells differently. the blocks coincide: same centroids, same values.

# %%
print("same centroids:", np.allclose(rebuilt.centroids, domain.centroids))
print("same values:   ", np.allclose(rebuilt["cu"], domain["cu"]))

# %% [markdown]
# without a known block size, take the smallest gap between distinct centroid coordinates along each axis, as long
# as at least two neighboring blocks share a row.
#
# ## round trip to a file
#
# CSV kept only centroids and attributes. parquet keeps the whole model: geometry, rotation, layout, index and
# CRS travel in the file metadata. the rotated grid, masked to its middle level, goes out and comes back unchanged.

# %%
middle = rotated.with_column("cu", np.linspace(0.1, 3.0, len(rotated))).mask(k == 1)
bt.write_parquet(folder / "middle.parquet", middle, progress=False)
back = bt.read_parquet(folder / "middle.parquet", progress=False)
print(back)
print("origin", back.origin, "rotation", back.rotation)
print(
    "same cells:",
    np.array_equal(back.index, middle.index),
    "| same cu:",
    np.array_equal(back["cu"], middle["cu"]),
)

# %% [markdown]
# [storing containers in parquet](../../01-first-steps/04-parquet/README.md) covers the file format, reading a model
# with polars, and models of other layouts.
