"""
# solids

a closed triangle mesh bounds a domain. `Mesh.contains` flags the samples inside it, `Mesh.proportion` measures
how much of each block it fills, `BlockModel.mask` keeps the blocks that count as inside and `block_shell` draws
them. here the three stacked sulphide lenses are the solids.
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
from common import ACCENT, GRAY, HIGHLIGHT, LIGHT, save
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# %% [markdown]
# a solid needs a closed mesh: no boundary edges, each edge shared by two triangles. run `Mesh.validate` before you
# rely on an inside. [mesh files](../../02-data-and-geometry/07-mesh-files/README.md) repairs meshes that fail it.

# %%
data = bt.datasets.stacked_sulphide_lenses()
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
for i, lens in enumerate(lenses, 1):
    print(
        f"lens {i}: {lens}, {lens.validate().summary['boundary_edges']} boundary edges, {lens.volume:,.0f} m3"
    )

# %% [markdown]
# `contains` tests points by generalized winding number. of the 2 m zinc composites, those inside a lens carry
# the ore:

# %%
holes = bt.Drillholes(data["collars"], data["surveys"], data["assays"])
composites = holes.composite(2.0, ["ZN_PCT"])
xyz, zn = composites.coords, composites["ZN_PCT"]
inside = np.array([lens.contains(xyz) for lens in lenses])
for i, flags in enumerate(inside, 1):
    print(f"lens {i}: {flags.sum():4} composites, mean Zn {np.nanmean(zn[flags]):.2f}%")
outside = ~inside.any(axis=0)
print(f"outside: {outside.sum()} composites, mean Zn {np.nanmean(zn[outside]):.2f}%")

# %% [markdown]
# `BlockModel.from_extents` builds a 20 × 20 × 10 m model around the lenses ([block model from extents](../../02-data-and-geometry/12-block-model-from-extents/README.md)). `proportion` settles
# blocks that no triangle passes through with one centroid test and samples `discretization`³ points in the others,
# here 8. summed over the blocks, the proportions give back each lens's volume to within 1 %.

# %%
size = (20, 20, 10)
blocks = bt.BlockModel.from_extents(*lenses, size=size, buffer=10, snap=True)
proportions = [lens.proportion(blocks, discretization=2) for lens in lenses]
for i, (lens, p) in enumerate(zip(lenses, proportions), 1):
    print(f"lens {i}: mesh {lens.volume:,.0f} m3, blocks {p.sum() * np.prod(size):,.0f} m3")
blocks = blocks.with_column("proportion", np.sum(proportions, axis=0))

# %% [markdown]
# `mask` keeps the blocks more than half inside a lens. the lenses are thin next to the blocks, so many blocks
# they cross are less than half filled, and the kept blocks hold under two thirds of the lens volume. [sub-blocks](../../02-data-and-geometry/06-sub-blocks/README.md)
# sub-blocks the edges instead.

# %%
ore = blocks.mask(blocks["proportion"] > 0.5)
print(f"{len(blocks):,} blocks, {len(ore):,} more than half inside: {ore.volumes.sum():,.0f} m3")
print(f"lenses: {sum(lens.volume for lens in lenses):,.0f} m3")

# %% [markdown]
# a vertical section across strike (the lenses strike N22.5°E) shows the block proportions with the lens
# outlines and the composites within 10 m of the section. `block_shell` turns the masked blocks into the mesh of
# their outer faces, drawn in 3D over the lens surfaces.

# %%
center = np.mean([lens.vertices.mean(axis=0) for lens in lenses], axis=0)
plane = (center, 112.5, 90)
fig = plt.figure(figsize=(12, 5.5), layout="constrained")
a = fig.add_subplot(1, 2, 1)
bt.plot.section(blocks, "proportion", plane=plane, colorbar=False, vmin=0, vmax=1, cmap="Greys", ax=a)
bt.plot.slab(xyz[outside], plane=plane, thickness=20, s=4, color=GRAY, label="composite outside", ax=a)
bt.plot.slab(
    xyz[~outside],
    plane=plane,
    thickness=20,
    meshes=lenses,
    s=6,
    color=HIGHLIGHT,
    label="composite inside",
    ax=a,
)
a.set(title="Block proportion inside a lens, section across strike", xlabel="Across strike (m)")
a.legend(loc="lower left", frameon=True, framealpha=0.9)
fig.colorbar(a.images[0], ax=a, shrink=0.7, label="proportion of block inside")

b = fig.add_subplot(1, 2, 2, projection="3d")
shell = bt.block_shell(ore)
b.add_collection3d(
    Poly3DCollection(shell.vertices[shell.triangles], facecolor=ACCENT, edgecolor="none", alpha=0.35)
)
for lens in lenses:
    b.plot_trisurf(*lens.vertices.T, triangles=lens.triangles, color=LIGHT, linewidth=0, alpha=0.2)
lo, hi = np.array(blocks.origin), np.array(blocks.origin) + np.array(size) * blocks.count
b.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
b.set_box_aspect(hi - lo)
b.set_title(f"{len(ore):,} blocks more than half inside (shell)")
b.set_xlabel("Easting")
b.set_ylabel("Northing")
b.set_zlabel("Elevation")
b.tick_params(labelsize=6)
save(fig, "solids")
