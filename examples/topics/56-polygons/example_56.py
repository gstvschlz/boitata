"""
# 56. Polygons

A coal lease: 295 boreholes, each tagged with the mining method its seam suits, a 48-vertex lease boundary and a
100 m grid. Polygons select what lies in the lease, measure how far each point is from its edge, and domains spread
from the holes to the grid.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, map_axes, save

# %% [markdown]
# ## Inside the lease
#
# The boundary is a closed `Polylines`; its single part is the `(n, 2)` ring that `point_in_polygon` tests points
# against, with the even-odd rule.

# %%
data = cs.datasets.coal_seam_thickness()
holes, grid, boundary = data["boreholes"], data["grid"], data["boundary"]
print(boundary)
lease = boundary.parts[0][:, :2]
xy, centers = holes.coords[:, :2], grid.centroids[:, :2]
inside = cs.point_in_polygon(centers, lease)
print(f"{inside.sum()} of {len(grid)} cells inside, {inside.sum() * 100 * 100 / 1e6:.1f} km²")
print(f"agreement with the INSIDE flag: {np.mean(inside == (grid['INSIDE'] == 1)):.1%}")
print(f"{cs.point_in_polygon(xy, lease).sum()} of {len(xy)} holes inside")


def outline(ax, ring, **kwargs):
    ax.plot(*np.vstack([ring, ring[:1]]).T, **kwargs)


# %% [markdown]
# ## Distance to the boundary
#
# `polygon_distance` is the plan distance to the nearest edge; with `signed=True` points inside are negative. A
# 200 m standoff along the boundary, where no mining is allowed, is the cells within 200 m inside it.

# %%
distance = cs.polygon_distance(centers, lease, signed=True)
standoff = (distance > -200) & (distance < 0)
print(f"standoff: {standoff.sum()} cells, {standoff.sum() * 100 * 100 / 1e6:.1f} km²")
to_edge = -cs.polygon_distance(xy, lease, signed=True)
print(
    f"holes from the boundary: {to_edge.min():.1f} to {to_edge.max():.0f} m, {np.sum(to_edge < 200)} within 200 m"
)

fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
cs.plot.section(grid, np.where(inside, -distance, np.nan), ax=ax, colorbar=False, cmap="Greys")
fig.colorbar(ax.images[0], ax=ax, shrink=0.8, label="Distance inside the boundary (m)")
ax.contour(
    centers[:, 0].reshape(90, 120),
    centers[:, 1].reshape(90, 120),
    distance.reshape(90, 120),
    levels=[-200],
    colors=HIGHLIGHT,
    linewidths=1,
)
outline(ax, lease, color=INK, lw=1)
ax.scatter(*xy.T, s=5, color=ACCENT)
map_axes(ax, "Distance to the lease boundary; 200 m standoff in orange")
save(fig, "distance")

# %% [markdown]
# ## Domains from the holes
#
# `assign_domain` gives each target the domain of its nearest sample, read from the `domain_column` of the holes.
# Here the targets are the cells in the lease and the domain is the mining method; labels come back as a list, with
# a confidence per target (always 1 for the nearest sample).

# %%
cells = grid.mask(inside)
labels, confidence = cs.assign_domain(cells, coords=holes, domain_column="CATEGORY")
methods = cs.Categories(["MECHANIZED", "SELECTIVE", "UNECONOMIC"], colors=[ACCENT, "#9ebad6", LIGHT])
codes = methods.encode(labels)
logged = methods.encode(holes["CATEGORY"])
for name, cell_share, hole_share in zip(
    methods.names, methods.shares(codes), methods.shares(logged), strict=True
):
    print(f"{name:>10}: {cell_share:6.1%} of the lease, {hole_share:6.1%} of the holes")

fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
cs.plot.section(cells, codes, scheme=methods, colorbar=False, ax=ax)
outline(ax, lease, color=GRAY, lw=1)
cmap, norm = cs.plot.category_colors(methods)
ax.scatter(*xy.T, c=logged, cmap=cmap, norm=norm, s=10, edgecolors=INK, linewidths=0.4)
cs.plot.category_legend(methods, ax, loc="upper left", bbox_to_anchor=(1, 1))
map_axes(ax, "Mining method of the nearest hole")
save(fig, "domains")

# %% [markdown]
# `MECHANIZED` holes are 81.0 % of the holes but their domain covers 69.7 % of the lease: the infill was drilled
# where the seam is thick. Counting cells, not holes, declusters the shares.
#
# ## A polygon with a hole
#
# A lease often excludes an area inside it; take a 1.5 km circle around a point in the middle. The even-odd rule
# handles it in a single ring: the outer ring, closed, then the inner ring, closed. The two bridge edges between
# them run along the same segment, so they cancel, and points in the inner ring cross the boundary an even number
# of times.
#
# `PolygonSelector` takes several rings, with an optional elevation window, but a point is selected when it is
# inside *any* ring: the exclusion zone counts as inside. It cannot express a hole yet.

# %%
center = lease.mean(axis=0)
angle = np.linspace(0, 2 * np.pi, 60, endpoint=False)
exclusion = center + 1500 * np.column_stack([np.cos(angle), np.sin(angle)])
with_hole = np.vstack([lease, lease[:1], exclusion, exclusion[:1]])
even_odd = cs.point_in_polygon(centers, with_hole)
selector = cs.PolygonSelector([np.c_[ring, np.zeros(len(ring))] for ring in (lease, exclusion)], closed=True)
selected = selector.contains(np.c_[centers, np.zeros(len(centers))])
in_exclusion = cs.point_in_polygon(centers, exclusion)
print(f"cells in the exclusion zone: {in_exclusion.sum()}")
print(f"point_in_polygon, one ring with a hole: {even_odd.sum()} cells")
print(
    f"PolygonSelector, outer and inner rings: {selected.sum()} cells, {np.sum(selected & in_exclusion)} of them in the hole"
)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, layout="constrained")
for ax, keep, title in (
    (a, even_odd, "point_in_polygon, even-odd"),
    (b, selected, "PolygonSelector, any ring"),
):
    cs.plot.section(grid, keep.astype(float), ax=ax, colorbar=False, cmap="Greys", vmin=0, vmax=2)
    outline(ax, lease, color=INK, lw=1)
    outline(ax, exclusion, color=HIGHLIGHT, lw=1)
    map_axes(ax, title)
b.set_ylabel("")
save(fig, "hole")
