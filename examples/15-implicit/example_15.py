"""
# 15. Implicit modeling

An implicit model fits a scalar field to the data and takes a surface as one of its level sets, instead of
digitizing outlines section by section. Here a Zn > 5 % shell is modeled from the composites of the cluster seen
in [chapter 7](../07-solids/README.md).
"""

# %% [hidden]
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore", ".*locations hold several samples")

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, LIGHT, save
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# %% [markdown]
# With `cutoff=5` each composite is coded +1 at or above the cutoff and −1 below, so the shell is the zero level of
# the field. Two engines fit it: a radial basis function (RBF) interpolates the codes exactly by solving one dense
# system, whose cost grows with the cube of the sample count, so 10 m composites keep it to about 3,000 samples; a
# sparse Gaussian process (GP) smooths through them, learns anisotropic ranges and returns to the mean code away from
# the data.

# %%
dh = cs.datasets.drillholes()
composites = dh.composite(10.0, ["ZN"])
xyz, zn = composites.coords, composites["ZN"]
window = (xyz[:, 0] > 4550) & (xyz[:, 0] < 4950) & (xyz[:, 1] > 7400) & (xyz[:, 1] < 7700) & ~np.isnan(zn)
xyz, zn = xyz[window], zn[window]
models = {
    "RBF": cs.ImplicitModel("rbf", degree=0).fit(xyz, zn, cutoff=5),
    "GP": cs.ImplicitModel("gp", degree=0).fit(xyz, zn, cutoff=5),
}
indicator = np.where(zn >= 5, 1.0, -1.0)
print(f"{len(xyz)} composites, {(indicator > 0).sum()} above 5 % Zn")
for name, model in models.items():
    agree = np.mean(np.sign(model.predict(xyz)) == indicator)
    print(f"{name}: {agree:.1%} of composites on their side of the shell")
report = models["GP"].report
print(
    f"GP {report['status']} in {report['iterations']} iterations, ranges {np.round(report['lengthscales'])} m"
)

# %% [markdown]
# `isosurface` samples the field at the block centroids and triangulates the zero level. With `closed=True` the shell
# is capped where it leaves the block model, so it bounds a volume, which should match the count of blocks whose
# centroid lies inside.

# %%
size = 5.0
lo, hi = xyz.min(axis=0), xyz.max(axis=0)
count = np.ceil((hi - lo) / size).astype(int)
blocks = cs.BlockModel(origin=lo, size=(size, size, size), count=count)
fields, shells = {}, {}
for name, model in models.items():
    shell = shells[name] = model.isosurface(blocks, closed=True)
    fields[name] = model.predict(blocks).reshape(count[::-1])
    count_volume = (fields[name] > 0).sum() * size**3
    print(
        f"{name}: shell of {len(shell.triangles):,} triangles, {shell.volume:,.0f} m3; blocks inside {count_volume:,.0f} m3"
    )

# %% [markdown]
# An east–west section through the high-grade composites. The RBF honors every code but bulges into undrilled
# ground; the GP draws flat lenses along its learned ranges and leaves some isolated codes outside.

# %%
j = int((np.median(xyz[indicator > 0, 1]) - lo[1]) // size)
northing = lo[1] + (j + 0.5) * size
x = lo[0] + (np.arange(count[0]) + 0.5) * size
z = lo[2] + (np.arange(count[2]) + 0.5) * size
plane = ((0, northing, 0), 90, 90)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
for ax, (name, field) in zip(axes, fields.items()):
    ax.contourf(x, z, field[:, j, :], levels=[0, np.inf], colors=[LIGHT])
    ax.contour(x, z, field[:, j, :], levels=[0], colors=[ACCENT], linewidths=1.2)
    for mask, color, label in ((indicator < 0, GRAY, "Zn ≤ 5 %"), (indicator > 0, HIGHLIGHT, "Zn > 5 %")):
        cs.plot.slab(xyz[mask], plane=plane, thickness=20, s=8, color=color, label=label, ax=ax)
    ax.set_title(f"{name}, northing {northing:.0f} m")
axes[1].set_ylabel("")
axes[0].set_ylim(np.percentile(xyz[:, 2], 1) - 50, hi[2])
axes[0].legend(loc="lower right", title="composites within 10 m")
save(fig, "section")

# %% [markdown]
# The GP shell:

# %%
shell = shells["GP"]
fig = plt.figure(figsize=(7, 5.5), layout="constrained")
ax = fig.add_subplot(projection="3d")
ax.add_collection3d(
    Poly3DCollection(shell.vertices[shell.triangles], facecolor=ACCENT, edgecolor="none", alpha=0.25)
)
ax.scatter(*xyz[indicator > 0].T, s=2, color=HIGHLIGHT, depthshade=False)
ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
ax.set_box_aspect(hi - lo)
ax.set_title("GP Zn > 5 % shell and the composites above the cutoff")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
ax.set_zlabel("Elevation")
ax.tick_params(labelsize=6)
save(fig, "shell")
