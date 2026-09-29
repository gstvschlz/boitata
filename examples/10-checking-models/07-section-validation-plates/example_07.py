"""
# Section validation plates

A block model earns trust one section at a time: true block edges (not a resampled raster, so a sub-blocked
model shows its real geometry), the estimate on those blocks, the lens it was built from and the drillhole
composites it was built on, all on one plate and one color scale. `section` draws true edges whenever the cut is
normal to one of the model's own axes; `slab` overlays the lens trace and the composites on the same plane.
Whether the model honors the composites is a `row_at` lookup away from `scatter`.
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
from common import save

# %% [markdown]
# Sub-block the parent grid to one lens ([sub-blocks](../../02-data-and-geometry/06-sub-blocks/README.md)), then estimate Zn on the sub-blocks from the composites inside
# the lens.

# %%
data = cs.datasets.stacked_sulphide_lenses()
lens = data["lens_2"]
parents = cs.BlockModel.from_extents(lens, size=(40, 40, 20), buffer=10, snap=True)
model = parents.subblock([(lens, "inside", "ore")], 4, fill="waste")

holes = cs.Drillholes(data["collars"], data["surveys"], data["assays"])
composites = holes.composite(2.0, ["ZN_PCT"])
xyz, zn = composites.coords, composites["ZN_PCT"]
inside = lens.contains(xyz) & ~np.isnan(zn)
ore = np.array(model["domain"]) == "ore"
grade = np.full(len(model), np.nan)
estimator = cs.InverseDistance(cs.Search(100, max_samples=12)).fit(xyz[inside], zn[inside])
grade[ore] = estimator.predict(model.centroids[ore])
model = model.with_column("zn", grade)

# %% [markdown]
# `section` draws true block edges whenever the cut is normal to one of the model's own axes, here an east-west
# plane. `slab` overlays the lens trace and the composites within 25 m of that same `plane`, on the same color
# scale.

# %%
plane = (lens.vertices.mean(axis=0), 90, 90)
fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
cs.plot.section(model, "zn", plane=plane, cmap="viridis", vmin=0, vmax=8, ax=ax)
cs.plot.slab(
    composites,
    "ZN_PCT",
    plane=plane,
    thickness=25,
    meshes=lens,
    cmap="viridis",
    vmin=0,
    vmax=8,
    colorbar=False,
    ax=ax,
)
ax.set(title="East-west section: sub-block edges, lens trace, composites")
save(fig, "section")

# %% [markdown]
# ## Adherence
#
# Adherence is a resubstitution test, not cross-validation: does the model honor the holes it was built from?
# `row_at` finds the sub-block holding each composite and `scatter` gives the 1:1 line and the regression slope.

# %%
rows = model.row_at(xyz[inside])
paired = rows >= 0
fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
cs.plot.scatter(zn[inside][paired], model["zn"][rows[paired]], ax=ax)
ax.set(xlabel="composite Zn (%)", ylabel="block Zn (%)", title=f"Adherence, {paired.sum()} composites")
save(fig, "adherence")
