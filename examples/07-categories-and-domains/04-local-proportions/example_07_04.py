"""
# local proportions

category proportions can vary in space: a vertical proportion curve, areal proportion maps, and their product as 3D
proportions that sum to 1 in every cell. passed to `SIS` as `proportions=`, they keep each lithology where the data
put it; `check_realizations` then compares realizations with and without them.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import INK, map_axes, save

# %% [markdown]
# the tailings of [categories](../../07-categories-and-domains/01-categories/README.md), as a first campaign would see them: every third of the 107 sonic holes, composited to
# 1 m, each composite taking the lithology that covers most of it. clay forms the foundation under the tailings, sand
# settled in the north and slimes in the south. the facility sits on rolling ground, so the height of a composite
# above the original ground orders the layers better than its elevation.

# %%
data = bt.datasets.tailings_reprocessing()
holes = bt.Drillholes(data["collars"], data["surveys"], data["lithology"])
composites = holes.composite(1.0, [], categories=["LITH"])
ids = np.asarray(composites["HOLE_ID"], dtype=object)
composites = composites.filter(np.isin(ids, sorted(set(ids))[::3]))
ground, top = data["original_ground"], data["surface"]
xyz = composites.coords
height = xyz[:, 2] - ground["Z"][ground.row_at(xyz[:, :2])]
scheme = bt.Categories(["CAP", "SAND", "SLIME", "CLAY"], colors=["#b8b8b8", "#d9b56c", "#5b7a99", "#7a4f35"])
codes = scheme.encode(composites["LITH"]).astype(int)
weights = bt.cell_declustering(xyz[:, :2], height, cell_size=100.0).weights
print(f"{len(composites)} composites, {height.min():.1f} to {height.max():.1f} m above the original ground")

# %% [markdown]
# ## vertical proportion curve
#
# `vertical_proportions` takes the weighted share of each category in slices of height, here 1 m slices of the
# height above the original ground, passed as `elevation`. the declustering weights keep densely drilled parts from
# dominating, and `scheme` fixes the order and names. the result is a `Table` with one row per slice: its center, its
# weight and one column per category.

# %%
curve = bt.vertical_proportions(
    composites, "LITH", size=1.0, elevation=height, weights=weights, scheme=scheme
)
levels = np.asarray(curve["elevation"])
print(curve.column_names)

fig, ax = plt.subplots(figsize=(4.2, 4.2))
left = np.zeros(len(levels))
for name, color in zip(scheme.names, scheme.colors, strict=True):
    right = left + np.asarray(curve[name])
    ax.fill_betweenx(levels, left, right, color=color, step="mid", lw=0)
    left = right
ax.set(xlim=(0, 1), xlabel="Proportion", ylabel="Height above original ground (m)", title="Vertical curve")
bt.plot.category_legend(scheme, ax, loc="upper left", bbox_to_anchor=(1.01, 1))
save(fig, "vertical")

# %% [markdown]
# clay fills the meters under the original ground; sand and slimes share the tailings, and the cap appears at the
# heights the facility's top reaches. the cap is a cover tied to the surface, so a depth below the surface would suit
# it better than a height above the base.
#
# ## areal proportion maps
#
# `detrend` with `categorical=True` in plan gives the areal proportions: the declustered average of each category's
# indicator under a gaussian kernel, with the bandwidth chosen by leave-one-out error. with `scheme`, its columns
# follow the curve's.

# %%
areal, _ = bt.detrend(
    xyz[:, :2],
    composites["LITH"],
    bandwidth=[60, 90, 130, 200],
    weights=weights,
    categorical=True,
    scheme=scheme,
)
print(f"bandwidth {areal.bandwidth:.0f} m")

nx, ny, _ = top.count
x0, y0, _ = top.origin
extent = (x0, x0 + nx * top.size[0], y0, y0 + ny * top.size[1])
plan = areal.predict(top.centroids[:, :2])

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True, layout="constrained")
for ax, name in zip(axes, ["SAND", "SLIME", "CLAY"], strict=True):
    shown = ax.imshow(
        np.asarray(plan[name]).reshape(ny, nx), origin="lower", extent=extent, vmin=0, vmax=1, cmap="Greys"
    )
    ax.scatter(*xyz[:, :2].T, s=1, color=INK)
    map_axes(ax, name.capitalize())
for ax in axes[1:]:
    ax.set_ylabel("")
fig.colorbar(shown, ax=axes, label="Areal proportion", shrink=0.8)
save(fig, "areal")

# %% [markdown]
# sand dominates the north and slimes the south; clay, under the whole facility, is even.
#
# ## proportions in 3D
#
# the model is a `BlockModel` of 20 × 20 × 1 m cells from 3 m under the original ground to the tailings surface.
# in each cell, `combine_proportions` multiplies the curve at the cell's height by the areal proportions at its
# position, divides by the global proportions (the curve's weighted mean) and rescales to sum 1. where the areal map
# equals the global proportions, the curve comes back unchanged; where it has more sand, each height above the clay
# gets more sand. the same call gives the proportions at the composites.

# %%
plan20 = bt.BlockModel(origin=(4945.0, 1945.0), size=(20.0, 20.0), count=(41, 28))
base_at = ground["Z"][ground.row_at(plan20.centroids[:, :2])] - 3.0
top_at = top["Z"][top.row_at(plan20.centroids[:, :2])]
z0 = np.floor(base_at.min())
full = bt.BlockModel(
    origin=(4945.0, 1945.0, z0), size=(20.0, 20.0, 1.0), count=(41, 28, int(np.ceil(top_at.max() - z0)))
)
column = plan20.row_at(full.centroids[:, :2])
z = full.centroids[:, 2]
model = full.mask((z > base_at[column]) & (z < top_at[column]))
cells = model.centroids
cell_height = cells[:, 2] - ground["Z"][ground.row_at(cells[:, :2])]

at_cells = bt.combine_proportions(cells, curve, areal.predict(cells[:, :2]), elevation=cell_height)
at_data = bt.combine_proportions(xyz, curve, areal.predict(xyz[:, :2]), elevation=height)
rows = np.column_stack([at_cells[name] for name in scheme.names])
print(
    f"{len(model)} cells, proportions {rows.min():.2f} to {rows.max():.2f}, sums {rows.sum(axis=1).min():.12f}"
)

# %% [markdown]
# ## SIS with and without local proportions
#
# each category gets one spherical indicator variogram with a sill of p(1 − p), 150 m across and 4 m vertical. plain
# `SIS` krigs each indicator by ordinary kriging from its neighbors. given `proportions=` at `fit` and `simulate`, as
# a `Table` or an (n, k) array, it krigs the indicator minus its local proportion by simple kriging instead, which
# pulls each node towards the proportions of its own cell.

# %%
shares = scheme.shares(codes, weights=weights)
variograms = [bt.Variogram([("spherical", p * (1 - p), 150.0)], ratios=(1.0, 4.0 / 150.0)) for p in shares]
search = bt.Search(radius=150.0, max_samples=16, ratios=(1.0, 0.05))
plain = bt.SIS(variograms, search).fit(xyz, codes)
local = bt.SIS(variograms, search).fit(xyz, codes, proportions=at_data)
runs = {
    "Global": plain.simulate(model, n=20, seed=7, keep=True),
    "Local proportions": local.simulate(model, n=20, seed=7, keep=True, proportions=at_cells),
}

# %% [markdown]
# a north-south section through the middle of the facility shows the difference. far from the holes, plain SIS
# falls back on the global proportions and scatters clay through the tailings; with local proportions the clay stays
# at the base, where the curve puts it.

# %%
cmap, norm = bt.plot.category_colors(scheme)
section = np.abs(cells[:, 0] - 5355.0) < 10.0
fig, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True, layout="constrained")
for ax, (title, summary) in zip(axes, runs.items(), strict=True):
    y, z = cells[section, 1], cells[section, 2]
    ax.scatter(y, z, c=summary.realizations[0][section], cmap=cmap, norm=norm, marker="s", s=14, linewidths=0)
    ax.set(title=f"{title}, realization 1", ylabel="Elevation (m)")
axes[1].set_xlabel("Northing (m)")
bt.plot.category_legend(scheme, fig, loc="outside right upper")
save(fig, "sections")

# %% [markdown]
# ## checks
#
# `check_realizations` ([realization checks](../../10-checking-models/04-realization-checks/README.md)) compares the category proportions of each realization with the declustered
# data's. both runs come within five points of them, the local one with more cap, which the curve spreads over each
# height the facility's top reaches. the global shares hide where each category went. the share of clay in each meter
# of height shows it, and only the run with local proportions follows the curve.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
for ax, (title, summary) in zip(axes[:2], runs.items(), strict=True):
    check = bt.check_realizations(model, summary, xyz, codes, weights=weights)
    bt.plot.histogram_reproduction(check, ax=ax)
    ax.set_xticks(range(len(scheme)), scheme.names)
    ax.set(title=title, xlabel="")
    print(
        f"{title:18}",
        np.round(check.proportions.mean(axis=0), 3),
        "data",
        np.round(check.data_proportions, 3),
    )
slices = np.floor(cell_height)
for (title, summary), style in zip(runs.items(), ["--", "-"], strict=True):
    clay = [(summary.realizations[:, slices == s] == 3).mean() for s in np.unique(slices)]
    axes[2].plot(clay, np.unique(slices) + 0.5, style, color=scheme.colors[3], label=title)
axes[2].plot(curve["CLAY"], levels, "o", color=INK, ms=3, label="Vertical curve")
axes[2].set(xlabel="Clay proportion", ylabel="Height above original ground (m)", title="Clay by height")
axes[2].legend()
save(fig, "checks")
