"""
# Sub-block a model from solids

The previous tutorial kept vein proportions on 5 m blocks. Many models instead need one domain per cell, for
example to estimate grade inside each vein with its own samples. With 10 m parent blocks and veins one to two
meters thick, a label per parent block puts most of the vein volume in the wrong place. Sub-blocking splits the
parents a vein surface crosses into smaller cells, so the labels follow the contacts. You compare the volume the
parents capture and the volume sub-blocks of several sizes capture with the volume of the solids.
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
from common import ACCENT, GRAY, HIGHLIGHT, save

# %% [markdown]
# !!! learn "What you'll learn"
#     - How `BlockModel.subblock` splits parent blocks along solid contacts.
#     - Why a total volume close to the solid's can hide volume in the wrong place, and how to measure it.
#     - How to choose the sub-grid from the thickness of the solids.
#
#     Prerequisites: [flag solid proportions in a block model](../../14-workflows/01-solid-proportions/README.md) and
#     [sub-blocks](../../02-data-and-geometry/06-sub-blocks/README.md), which introduces `subblock`,
#     `from_meshes` and `regularize` on the stacked lenses.
#
# ## The data
#
# The four veins of the vein-gold grade-control dataset and a parent grid of 10 m blocks around them.

# %%
data = bt.datasets.vein_gold_grade_control()
names = ["V1", "V2", "V3", "V4"]
veins = [data[f"vein_{name}"] for name in names]
parents = bt.BlockModel.from_extents(*veins, size=(10, 10, 10), buffer=10, snap=True)
solid_volume = sum(vein.volume for vein in veins)
print(parents)
print(f"veins: {solid_volume:,.0f} m3")

# %% [markdown]
# !!! step "Step 1: Sub-block against the solids"
#     `subblock` takes `(mesh, "inside", label)` domains in priority order and `n`, the sub-cells per parent edge.
#     Every parent a vein surface crosses is split into n × n × n sub-cells; each sub-cell takes the label of the
#     first domain holding its center, and neighboring sub-cells with one label merge into larger sub-blocks.
#     Parents no surface crosses stay whole. Listing V4 last gives the space it shares with the other veins to
#     them, the same priority as in the previous tutorial. With `n = 1` there is no splitting: each parent takes
#     the label at its center.
#
# <figure class="bt-figure">
# --8<-- "svg/w1-subblocks.svg"
# <figcaption><b>Figure 1.</b> A parent block the contact crosses is split into sub-cells (here 4 × 4); each takes
# the domain at its center, and sub-cells of one domain merge along a row. Parents the contact misses stay
# whole.</figcaption>
# </figure>

# %%
domains = [(vein, "inside", name) for vein, name in zip(veins, names)]
models = {n: parents.subblock(domains, n) for n in (1, 2, 4, 8, 16)}
for n, model in models.items():
    print(f"n = {n:<2}: {len(model):>7,} blocks, smallest {model.volumes.min():g} m3")

# %% [markdown]
# !!! step "Step 2: Measure captured and misplaced volume"
#     The volume labeled as a vein can match the solid while sitting in the wrong place: a block labeled V1 that is
#     mostly outside V1 is offset by vein volume in blocks left unlabeled. The misplaced volume adds the two:
#     labeled volume outside the vein, plus vein volume outside its labeled blocks. `Mesh.proportion` on the
#     labeled blocks measures both.


# %%
def volumes(model):
    labels = model["domain"]
    captured = misplaced = 0.0
    for vein, name in zip(veins, names):
        part = model.filter(labels == name)
        inside = (vein.proportion(part) * part.volumes).sum()
        captured += part.volumes.sum()
        misplaced += part.volumes.sum() - inside + vein.volume - inside
    return captured, misplaced


print(f"{'':7}{'labeled (m3)':>14}{'vs solids':>11}{'misplaced (m3)':>16}{'share':>7}")
for n, model in models.items():
    captured, misplaced = volumes(model)
    print(
        f"n = {n:<2}{captured:14,.0f}{captured / solid_volume - 1:+11.1%}{misplaced:16,.0f}"
        f"{misplaced / solid_volume:7.0%}"
    )

# %% [markdown]
# Every sub-grid labels within about 1 % of the solids' total volume, the whole parents included. The misplaced volume
# tells them apart. Labeled at their centers, 10 m parents misplace 157 % of the vein volume: more than the veins
# hold. Each halving of the sub-cell cuts the misplaced volume, to 40 % at `n = 8` (1.25 m sub-cells) and 17 % at
# `n = 16` (0.625 m). What remains is the sliver along each contact thinner than a sub-cell, and a vein 1 m thick
# is mostly such slivers.
#
# !!! pitfall "Pitfall"
#     A total that matches the solid does not show that the labels sit in the right place. Centered labels err on both sides
#     of a contact, and the errors cancel in the total. Measure the misplaced volume, or compare a section with
#     the solid outlines.
#
# !!! step "Step 3: Look at a section"
#     A vertical east–west section across V1 and V4 at northing 15 000.3 m, with the vein outlines. The northing
#     falls between sub-cell faces, so the section cuts each cell once.

# %%
plane = ((0, 15000.3, 0), 90, 90)
scheme = bt.Categories(names, colors=[ACCENT, "#6f9fc9", HIGHLIGHT, GRAY])
fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True, layout="constrained")
for ax, n in zip(axes, (1, 4, 16)):
    model = models[n]
    codes = scheme.encode(model["domain"])
    edges = {"edgecolor": "white", "linewidths": 0.3 if n < 16 else 0}
    bt.plot.section(model, codes, plane=plane, scheme=scheme, colorbar=False, ax=ax, **edges)
    bt.plot.slab(np.empty((0, 3)), plane=plane, thickness=1, meshes=veins, ax=ax)
    title = "10 m parents, label at center" if n == 1 else f"n = {n}: {10 / n:g} m sub-cells"
    ax.set(title=title, xlim=(8040, 8090), ylim=(660, 720))
    if ax is not axes[0]:
        ax.set_ylabel("")
bt.plot.category_legend(scheme, ax=axes[2], loc="lower right")
save(fig, "section")

# %% [markdown]
# !!! check "Check before you move on"
#     Sub-blocks must tile their parents: no parent holds more sub-block volume than its own. Each vein's
#     labeled volume at `n = 16` should match its solid to within a percent.

# %%
fine = models[16]
per_parent = np.bincount(fine.index.astype(np.int64), weights=fine.volumes)
print(f"largest volume in one parent: {per_parent.max():g} m3 of {np.prod(parents.size):g} m3")
assert per_parent.max() <= np.prod(parents.size) + 1e-6
labels = fine["domain"]
for vein, name in zip(veins, names):
    labeled = fine.volumes[labels == name].sum()
    print(
        f"{name}: sub-blocks {labeled:9,.0f} m3, solid {vein.volume:9,.0f} m3 ({labeled / vein.volume - 1:+.2%})"
    )

# %% [markdown]
# The largest parent holds 648 m³ of vein sub-blocks, under its 1000 m³. V1 to V3 match their solids to within
# 0.02 %. V4 falls 3.79 % short, the space it gave to the veins it crosses, as in the previous tutorial.
#
# ## The decision
#
# Size the sub-cell from the thinnest solid. The veins are 0.7 to 2 m thick, so `n = 16` gives 0.625 m sub-cells,
# about half the thickness of V2 and V3. It needs 679,610 blocks against 152,906 at `n = 8`, and it misplaces 17 %
# of the vein volume against 40 %. The labels then follow the contacts close enough to select samples and
# estimate each vein apart. For tonnage, also store each sub-block's proportion inside its vein, as in the
# [previous tutorial](../../14-workflows/01-solid-proportions/README.md), so the volume balances without the slivers.
#
# !!! seealso "See also"
#     - [Sub-blocks](../../02-data-and-geometry/06-sub-blocks/README.md): `from_meshes`, surfaces as domains and
#       `regularize` back to parents.
#     - [Find and fix degenerate solids](../../14-workflows/03-degenerate-solids/README.md): the solids must be valid before any of this.
