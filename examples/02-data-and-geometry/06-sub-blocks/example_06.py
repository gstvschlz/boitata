"""
# Sub-blocks

Whole blocks misstate the volume of a thin solid ([solids](../../02-data-and-geometry/05-solids/README.md)). A sub-blocked model splits the blocks a mesh cuts
into smaller cells and keeps the others whole. `subblock` does this on an existing grid, `BlockModel.from_meshes`
builds the model from meshes in one call, and `regularize` moves columns from sub-blocks to any other grid of
the same rotation. Here the three stacked sulphide lenses and the topography are the meshes.
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
from common import ACCENT, HIGHLIGHT, LIGHT, save

# %% [markdown]
# The parent grid, 40 × 40 × 20 m around the lenses, comes from `BlockModel.from_extents` ([block model from extents](../../02-data-and-geometry/12-block-model-from-extents/README.md)).

# %%
data = cs.datasets.stacked_sulphide_lenses()
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
names = ["lens 1", "lens 2", "lens 3"]
parents = cs.BlockModel.from_extents(*lenses, size=(40, 40, 20), buffer=10, snap=True)
print(parents)

# %% [markdown]
# `subblock` takes `(mesh, rule, label)` domains in priority order and a sub-grid of `n` cells per parent edge.
# Each sub-cell of a block a mesh cuts takes the label of the first domain holding its center, and the sub-cells
# of a block merge along x, then y. Blocks no mesh cuts stay whole; without `fill`, cells outside every domain are
# dropped. Counting centers gets each lens's volume nearly right at any sub-grid, as errors on either side cancel.
# What a finer sub-grid shrinks is the volume in the wrong place: sub-block volume outside its lens plus lens
# volume outside its sub-blocks, measured with `Mesh.proportion`.

# %%
domains = [(lens, "inside", name) for lens, name in zip(lenses, names)]


def misplaced(model):
    labels = np.array(model["domain"])
    total = 0.0
    for lens, name in zip(lenses, names):
        part = model.mask(labels == name)
        inside = lens.proportion(part, discretization=2) * part.volumes
        total += (part.volumes - inside).sum() + lens.volume - inside.sum()
    return total


print(f"lenses: {sum(lens.volume for lens in lenses):,.0f} m3")
for n in (1, 2, 4):
    model = parents.subblock(domains, n)
    print(
        f"sub-grid {n}: {len(model):>5} sub-blocks, {model.volumes.sum():,.0f} m3, {misplaced(model):,.0f} m3 misplaced"
    )
subblocked = model
print(subblocked)
print(f"whole parents: {(subblocked.extents == [0, 0, 0, 1, 1, 1]).all(axis=1).sum()}")

# %% [markdown]
# Each sub-block stores its parent cell and its extent as fractions of that cell (`extents`: min x, y, z, then
# max x, y, z). The lenses are thinner than a 40 m parent, so the surface cuts every parent they reach and no
# lens block stays whole. On one bench, the sub-blocks in the parent grid, with the lens outlines at mid-bench:

# %%
size = np.array(parents.size)
z = subblocked.centroids[:, 2]
level = parents.origin[2] + size[2] * (np.round((np.median(z) - parents.origin[2]) / size[2]) + 0.5) + 0.1
half = size[2] * (subblocked.extents[:, 5] - subblocked.extents[:, 2]) / 2
cut = (z - half < level) & (z + half > level)
labels = np.array(subblocked["domain"])
colors = dict(zip(names, (ACCENT, "#6f9fc9", HIGHLIGHT)))
fig, ax = plt.subplots(figsize=(6.4, 7), layout="constrained")
for c, e, name in zip(subblocked.centroids[cut], subblocked.extents[cut], labels[cut]):
    dx, dy = (e[3] - e[0]) * size[0], (e[4] - e[1]) * size[1]
    ax.add_patch(
        plt.Rectangle(
            (c[0] - dx / 2, c[1] - dy / 2), dx, dy, facecolor=colors[name], edgecolor="white", lw=0.3
        )
    )
cs.plot.slab(np.empty((0, 3)), plane=((0, 0, level), 90, 0), thickness=1, meshes=lenses, ax=ax)
x0, y0, _ = parents.origin
nx, ny, _ = parents.count
ax.vlines(x0 + size[0] * np.arange(nx + 1), y0, y0 + ny * size[1], color=LIGHT, lw=0.5, zorder=0)
ax.hlines(y0 + size[1] * np.arange(ny + 1), x0, x0 + nx * size[0], color=LIGHT, lw=0.5, zorder=0)
ax.set(title=f"Bench at {level:.0f} m: sub-blocks of lens 1 (dark) and 2 (light), 40 m parents")
save(fig, "subblocks")

# %% [markdown]
# ## A domain model from meshes
#
# `BlockModel.from_meshes` builds the grid and sub-blocks it in one call. Besides `"inside"` a solid, a domain
# can be `"below"` or `"above"` a surface such as topography; `fill` labels what no domain holds. Topography
# usually comes as a grid of elevations, a 2D `BlockModel` with an elevation column; `grid_surface` triangulates
# it through the cell centers. The lenses come first, so they win over the host rock that also lies below the
# surface. The grid is the parent grid raised to the highest point of the topography.

# %%
topography = cs.grid_surface(data["topography"], "Z")
origin, count = parents.origin, np.array(parents.count)
count[2] = np.ceil((np.nanmax(data["topography"]["Z"]) - origin[2]) / size[2])
domain_model = cs.BlockModel.from_meshes(
    origin,
    size,
    count,
    [*domains, (topography, "below", "host rock")],
    4,
    fill="air",
)
print(domain_model)
labels = np.array(domain_model["domain"])
for name in [*names, "host rock", "air"]:
    print(f"{name:>9}: {domain_model.volumes[labels == name].sum():>13,.0f} m3")
print(f"    total: {domain_model.volumes.sum():>13,.0f} m3 = grid {np.prod(size) * count.prod():,.0f} m3")

# %% [markdown]
# A vertical section across strike (the lenses strike N22.5°E) through the domain model, with the lens outlines:

# %%
scheme = cs.Categories(["air", "host rock", *names], colors=["#f4f7fa", LIGHT, ACCENT, "#6f9fc9", HIGHLIGHT])
center = np.mean([lens.vertices.mean(axis=0) for lens in lenses], axis=0)
plane = (center, 112.5, 90)
fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
cs.plot.section(domain_model, scheme.encode(domain_model["domain"]), plane=plane, scheme=scheme, ax=ax)
cs.plot.slab(np.empty((0, 3)), plane=plane, thickness=1, meshes=lenses, ax=ax)
ax.set(title="Domain model, section across strike", xlabel="Across strike (m)")
save(fig, "domains")

# %% [markdown]
# ## Regularizing
#
# `regularize` moves columns between any two models of the same rotation by the volume each pair of blocks
# shares: floats as volume-weighted means, labels by the value filling the most volume. Here the lens sub-blocks
# take an inverse-distance Zn grade from the 2 m composites inside their own lens, then go back to the 40 m
# parents. `fraction` is how much of each parent the sub-blocks fill, so volume × fraction × grade keeps the
# metal; `min_fraction` drops the thin edges and the metal in them.

# %%
holes = cs.Drillholes(data["collars"], data["surveys"], data["assays"])
composites = holes.composite(2.0, ["ZN_PCT"])
xyz, zn = composites.coords, composites["ZN_PCT"]
search = cs.Search(100, max_samples=12)
labels = np.array(subblocked["domain"])
grade = np.full(len(subblocked), np.nan)
for lens, name in zip(lenses, names):
    inside = lens.contains(xyz) & ~np.isnan(zn)
    on = labels == name
    grade[on] = cs.InverseDistance(search).fit(xyz[inside], zn[inside]).predict(subblocked.centroids[on])
subblocked = subblocked.with_column("zn", grade)
metal = (subblocked.volumes * grade).sum()
print(f"sub-blocks: {subblocked.volumes.sum():,.0f} m3 at {metal / subblocked.volumes.sum():.2f}% Zn")
for minimum in (0.0, 0.5):
    out = subblocked.regularize(parents, min_fraction=minimum)
    kept = ~np.isnan(out["zn"])
    volume = out.volumes[kept] * out["fraction"][kept]
    print(
        f"min_fraction {minimum}: {kept.sum()} parents, {volume.sum():,.0f} m3 at"
        f" {np.average(out['zn'][kept], weights=volume):.2f}% Zn, {(volume * out['zn'][kept]).sum() / metal:.1%} of the metal"
    )

# %% [markdown]
# A label keeps only the value that fills most of a block. The lenses fill few 40 m blocks by more than half,
# so the regularized label keeps a fraction of their volume; an indicator column averages to a proportion per
# block instead and keeps all of it.

# %%
is_ore = np.isin(np.array(domain_model["domain"]), names)
domain_model = domain_model.with_column("ore", is_ore.astype(float))
coarse = domain_model.regularize(cs.BlockModel(origin=origin, size=size, count=count))
labeled = coarse.volumes[np.isin(np.array(coarse["domain"]), names)].sum()
proportion = (coarse.volumes * coarse["fraction"] * coarse["ore"]).sum()
print(f"lens sub-blocks {domain_model.volumes[is_ore].sum():,.0f} m3")
print(f"40 m blocks: labeled as a lens {labeled:,.0f} m3, lens proportion {proportion:,.0f} m3")
