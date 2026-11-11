"""
# Declustering

A coal seam drilled on a regular mesh, then infilled where the seam is thick. The infill holes over-represent thick
coal, so the plain mean of the boreholes overstates the thickness of the seam. Declustering weights each hole by the
area it stands for: cell declustering by the number of holes sharing its cell, polygonal declustering by the area
nearer to it than to any other hole. Weight declustering instead weighs each hole by what it contributes to
estimating the whole lease.
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
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, map_axes, save

data = bt.datasets.coal_seam_thickness()
holes, lease, grid = data["boreholes"], data["boundary"], data["grid"]
thickness = holes["THICKNESS_M"]
x, y = holes.x, holes.y
print(f"{len(holes)} holes, mean thickness {thickness.mean():.2f} m")

fig, ax = plt.subplots(figsize=(7, 4.4), layout="constrained")
drawn = ax.scatter(x, y, c=thickness, s=14, edgecolors=INK, linewidths=0.3)
ax.plot(*lease.coords[:, :2].T, color=GRAY, lw=0.8)
fig.colorbar(drawn, ax=ax, shrink=0.8, label="Thickness (m)")
map_axes(ax, "Boreholes: infill where the seam is thick")
save(fig, "holes")

# %% [markdown]
# ## Cell declustering
#
# Each hole gets the inverse of the number of holes in its cell. Too small a cell holds one hole and changes nothing;
# too large a cell holds the whole lease. In between, when the infill targets high values, the declustered mean dips.
# `cell_declustering` scans cell sizes, averages each over 25 grid origins, and keeps the size with the lowest mean. The
# weights average 1.

# %%
cell = bt.cell_declustering(holes, "THICKNESS_M", sizes=np.arange(100.0, 3100.0, 100.0))
print(
    f"cell size {cell.cell_size:.0f} m, declustered mean {np.average(thickness, weights=cell.weights):.2f} m"
)
print(f"scanned mean at that size {cell.means[cell.sizes == cell.cell_size][0]:.2f} m")

fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
bt.plot.declustering(cell, naive=thickness.mean(), ax=ax, color=ACCENT, lw=1.6)
ax.set(title="Declustered mean against cell size", xlabel="Cell size (m)", ylabel="Mean thickness (m)")
save(fig, "cell_sizes")

# %% [markdown]
# The mean falls from 100 m cells, which hold about one hole each, to a flat minimum between 700 and 900 m. It then
# rises slowly as the cells grow to hold mesh and infill holes alike. Choosing the cell size is a judgment, and the
# minimum guides it only when you know the infill targets high values, as here. The curve averages 25 grid origins,
# while the returned weights come from one origin at the corner of the data, so their mean (the dot) sits a little above
# the curve.
#
# ## Polygonal declustering
#
# `polygon_declustering` weights each hole by the area of the grid nodes nearest to it, over the bounding box of the
# holes. It needs no cell size, but the holes on the edge take all the area out to the box, and the box differs from the
# lease.

# %%
polygon = bt.polygon_declustering(holes, "THICKNESS_M")
print(f"polygonal declustered mean {np.average(thickness, weights=polygon.weights):.2f} m")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained", sharey=True)
for ax, result, name in zip(axes, [cell, polygon], ["Cell", "Polygonal"], strict=True):
    w = result.weights
    drawn = ax.scatter(x, y, c=w, s=14, cmap="cividis", vmin=0, vmax=np.percentile(w, 98))
    ax.plot(*lease.coords[:, :2].T, color=GRAY, lw=0.8)
    map_axes(ax, f"{name} weights")
    fig.colorbar(drawn, ax=ax, shrink=0.8)
axes[1].set_ylabel("")
save(fig, "weights")

# %% [markdown]
# Both methods give the infill holes small weights and the sparse mesh large ones, and they agree on the declustered
# mean. The largest polygonal weights sit on the edge of the drilling, where the box reaches past the lease. The
# declustered histograms shift toward thin coal.

# %%
bins = np.linspace(0, np.ceil(thickness.max()), 25)
fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
ax.hist(thickness, bins, weights=np.full(len(holes), 1 / len(holes)), color="#e6e6e6", label="Equal weights")
for result, color, label in [(cell, ACCENT, "Cell"), (polygon, INK, "Polygonal")]:
    ax.hist(
        thickness,
        bins,
        weights=result.weights / len(holes),
        histtype="step",
        color=color,
        lw=1.4,
        label=label,
    )
ax.set(title="Thickness, declustered", xlabel="Thickness (m)", ylabel="Proportion")
ax.legend()
save(fig, "histograms")

# %% [markdown]
# These weights describe the data: histograms, statistics per domain, top cuts and the target of a normal-score
# transform.
#
# ## Weights from estimation
#
# `weight_declustering` estimates every cell of the lease and adds up the weight each hole receives. A hole alone in a
# sparse area informs many cells, while holes in the infill share the cells around them. The weights sum to the number
# of holes, as in `cell_declustering`. The targets are the 100 m cells inside the lease. The cell weights in the
# comparison are a second scan, the default cell sizes, again keeping the size with the lowest mean.

# %%
cells = grid.coords[grid["INSIDE"] == 1]
xy = holes.coords[:, :2]
print(f"{len(holes)} holes, {len(cells)} cells, naive mean {thickness.mean():.2f} m")
reference = bt.cell_declustering(holes, "THICKNESS_M")
print(f"cell {reference.cell_size:.0f} m: mean {reference.mean:.2f} m")

# %% [markdown]
# Inverse distance needs only a search; kriging also needs a variogram, fitted here on the holes.

# %%
experimental = bt.experimental_variogram(holes, "THICKNESS_M", 300.0, 4000.0)
model = experimental.fit("spherical")
search = bt.Search(radius=5000, max_samples=16)
methods = {
    "nearest neighbor": bt.NearestNeighbor(search),
    "inverse distance": bt.InverseDistance(search, power=2),
    "ordinary kriging": bt.OrdinaryKriging(model, search),
}
weights = {
    name: bt.weight_declustering(holes, "THICKNESS_M", cells, estimator=m) for name, m in methods.items()
}
for name, w in weights.items():
    print(f"{name}: mean {w.mean:.2f} m, weights {w.weights.min():.2f} to {w.weights.max():.2f}")

# %% [markdown]
# Nearest neighbor gives each hole the cells closest to it: polygon declustering at the resolution of the grid. Inverse
# distance and kriging spread each cell over several holes and even out the weights. Kriging also lets a hole screen
# those behind it, so its weights follow the spacing more closely.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (title, w) in zip(
    axes, [("Cell", reference.weights), ("Ordinary kriging", weights["ordinary kriging"].weights)]
):
    ax.scatter(cells[:, 0], cells[:, 1], s=1, color=LIGHT)
    points = ax.scatter(
        xy[:, 0], xy[:, 1], c=w, s=12, cmap="Greys", vmin=0, vmax=3, edgecolors=GRAY, linewidths=0.3
    )
    map_axes(ax, f"{title} weights")
axes[1].set_ylabel("")
fig.colorbar(points, ax=axes, label="Weight", shrink=0.8)
save(fig, "estimation-weights")

# %% [markdown]
# The two agree on which holes matter. Cell weights are flat within a cell, and kriging weights vary smoothly with the
# spacing.

# %%
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.scatter(reference.weights, weights["ordinary kriging"].weights, s=10, color=ACCENT)
ax.axline((0, 0), slope=1, color=GRAY, lw=0.8)
ax.set_xlabel("Cell weight")
ax.set_ylabel("Ordinary kriging weight")
ax.set_title("Weight per hole")
save(fig, "compare")

# %% [markdown]
# ## Declustered means
#
# All the declustered means sit well below the naive one. The ordinary kriging weights of each cell sum to 1, so their
# declustered mean equals the mean of the kriged cells (the dashed line).

# %%
kriged = methods["ordinary kriging"].fit(holes, "THICKNESS_M").predict(cells)
means = {"naive": thickness.mean(), "cell": reference.mean} | {name: w.mean for name, w in weights.items()}
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.barh(
    [name.capitalize() for name in means],
    list(means.values()),
    color=[HIGHLIGHT] + [ACCENT] * (len(means) - 1),
)
ax.axvline(np.nanmean(kriged), color=GRAY, ls="--", lw=0.8, label="Kriged cells")
ax.invert_yaxis()
ax.set_xlim(0, 2.7)
ax.set_xlabel("Mean thickness (m)")
ax.set_title("Declustered means")
ax.legend(loc="lower right")
save(fig, "means")
