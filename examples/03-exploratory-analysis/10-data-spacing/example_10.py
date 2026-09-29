"""
# Data spacing

A coal seam drilled on a regular mesh, then infilled where the seam is thick. How far apart are the holes, and how far
is each part of the lease from the drilling? Spacing measured at the holes describes the drilling; distance measured
from every cell of a grid is the usual basis for resource classification ([classification](../../10-checking-models/05-classification/README.md)).
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[2]))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, INK, map_axes, save

data = cs.datasets.coal_seam_thickness()
holes, lease, grid = data["boreholes"], data["boundary"], data["grid"]
print(f"{len(holes)} holes, {grid['INSIDE'].sum():.0f} cells of 100 m inside the lease")

# %% [markdown]
# ## Spacing between holes
#
# `data_spacing` gives the distance from each point to its nearest neighbor, or the mean over its `n` nearest; with
# `targets=` it measures from other locations, and `horizontal=True` measures in plan, for 3D data. The spread is wide:
# a tenth of the holes have a neighbor within 90 m, in the infill clusters, and a tenth none within 526 m, on the mesh
# and at the edges.

# %%
spacing = cs.data_spacing(holes)
print(f"nearest hole: median {np.median(spacing):.0f} m, P10 {np.percentile(spacing, 10):.0f} m, ", end="")
print(f"P90 {np.percentile(spacing, 90):.0f} m")
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", width_ratios=[1.5, 1])
drawn = a.scatter(*holes.coords[:, :2].T, c=spacing, s=14, edgecolors=INK, linewidths=0.3)
a.plot(*lease.vertices[:, :2].T, color=GRAY, lw=0.8)
fig.colorbar(drawn, ax=a, shrink=0.8, label="Spacing (m)")
map_axes(a, "Distance to the nearest hole")
cs.plot.histogram(spacing, bins=np.arange(0, spacing.max() + 25, 25), stats=True, ax=b, color=ACCENT)
b.set(title="Spacing of the holes", xlabel="Spacing (m)")
save(fig, "spacing")

# %% [markdown]
# ## Distance from the lease to the holes
#
# `hole_distance` measures from any targets, here the grid cells, the mean distance to the `n` nearest holes, one
# column per `n`. It counts each hole once, at its nearest sample, so a hole with many samples down its length does
# not pass for several holes; here each hole is one point. The distance to the nearest hole says whether a cell is
# drilled at all; the mean over three holes also asks whether it is surrounded by drilling, which is what a
# classification by spacing reads.

# %%
distance = cs.hole_distance(grid, holes, "ID", [1, 3])
inside = grid["INSIDE"] == 1
for k, n in enumerate([1, 3]):
    d = distance[inside, k]
    print(f"{n} nearest: median {np.median(d):.0f} m, P90 {np.percentile(d, 90):.0f} m, max {d.max():.0f} m")
top = np.percentile(distance[inside], 99)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
for ax, k, n in zip(axes, [0, 1], [1, 3], strict=True):
    shown = np.where(inside, distance[:, k], np.nan)
    cs.plot.section(grid, shown, axis="z", index=0, colorbar=False, vmin=0, vmax=top, ax=ax)
    ax.plot(*lease.vertices[:, :2].T, color=GRAY, lw=0.8)
    ax.scatter(*holes.coords[:, :2].T, s=2, color=INK)
    map_axes(ax, "Distance to the nearest hole" if n == 1 else f"Mean distance to the {n} nearest holes")
fig.colorbar(ax.collections[0], ax=axes, shrink=0.8, label="Distance (m)")
axes[1].set_ylabel("")
save(fig, "distance")

# %% [markdown]
# Half the lease lies within 233 m of a hole, but the mean distance to three holes has a median of 376 m: a cell next
# to one isolated hole looks well drilled by the first measure and not by the second. The gaps between the mesh lines
# and the edges of the lease, up to 923 m from three holes, stand out on the right. [Classification](../../10-checking-models/05-classification/README.md) turns these distances
# into measured, indicated and inferred classes.
