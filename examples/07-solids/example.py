# %% [markdown]
# # 7. Solids and block models
#
# A wireframe bounds a domain. Here an ellipsoid is fitted to the Zn > 5 % composites of the cluster seen in
# [chapter 6](../06-drillholes/README.md).

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GREY, HIGHLIGHT, LIGHT, fetch, save
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# %% [markdown]
# A small helper builds a closed triangle mesh of an ellipsoid:


# %%
def ellipsoid(center, axes, rotation, rings=24, segments=48):
    """Closed triangle mesh of an ellipsoid with semi-axes `axes` along the columns of `rotation`."""
    theta = np.linspace(0, np.pi, rings + 1)[1:-1]
    phi = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    unit = np.c_[(np.sin(t) * np.cos(p)).ravel(), (np.sin(t) * np.sin(p)).ravel(), np.cos(t).ravel()]
    unit = np.vstack([[0, 0, 1], unit, [0, 0, -1]])
    vertices = center + (unit * axes) @ rotation.T
    ring = lambda i: 1 + i * segments + np.arange(segments)
    tris = [[0, *e] for e in zip(ring(0), np.roll(ring(0), -1))]
    for i in range(rings - 2):
        a, b = ring(i), ring(i + 1)
        tris += [[a[j], b[j], a[(j + 1) % segments]] for j in range(segments)]
        tris += [[a[(j + 1) % segments], b[j], b[(j + 1) % segments]] for j in range(segments)]
    last = len(vertices) - 1
    tris += [[last, *e[::-1]] for e in zip(ring(rings - 2), np.roll(ring(rings - 2), -1))]
    return vertices, np.array(tris)


# %% [markdown]
# The ellipsoid's axes come from the covariance of the high-grade composites (two standard deviations):

# %%
dh = cs.Drillholes(
    cs.read_csv(fetch("drillholes/collar.csv")),
    cs.read_csv(fetch("drillholes/survey.csv")),
    cs.read_csv(fetch("drillholes/assay.csv")),
)
composites = dh.composite(2.0, ["ZN"])
xyz, zn = composites.coords, composites["ZN"]
window = (xyz[:, 0] > 4550) & (xyz[:, 0] < 4950) & (xyz[:, 1] > 7400) & (xyz[:, 1] < 7700)
high = xyz[window & (zn > 5)]
center = high.mean(axis=0)
eigen, vectors = np.linalg.eigh(np.cov((high - center).T))
vertices, triangles = ellipsoid(center, 2 * np.sqrt(eigen), vectors)
solid = cs.Mesh(vertices, triangles)
lo, hi = solid.bounds
print(solid, "semi-axes", np.round(2 * np.sqrt(eigen), 1))


# %% [markdown]
# `Mesh.proportion` samples 4 × 4 × 4 points in each block; `Mesh.contains` tests points by generalized winding
# number. Block proportions should add up to the ellipsoid's volume.

# %%
size = 10.0
count = np.ceil((np.array(hi) - lo) / size).astype(int)
blocks = cs.BlockModel(origin=lo, size=(size, size, size), count=count)
proportion = solid.proportion(blocks, discretization=4)
blocks = blocks.with_column("inside", proportion)
ore = blocks.mask(proportion > 0.5)
local = window & ~np.isnan(zn)
inside = solid.contains(xyz[local])
print(f"{len(blocks)} blocks, {len(ore)} more than half inside; volume {proportion.sum() * size**3:,.0f} m3")
print(f"exact ellipsoid volume {4 / 3 * np.pi * np.prod(2 * np.sqrt(eigen)):,.0f} m3")
print(
    f"composites inside: {inside.sum()}, mean Zn {np.nanmean(zn[local][inside]):.2f}% vs outside {np.nanmean(zn[local][~inside]):.2f}%"
)


# %% [markdown]
# One bench of block proportions, and the blocks more than half inside as a masked `BlockModel` drawn from its
# visible faces with `block_shell`:

# %%
k = count[2] // 2
level = lo[2] + (k + 0.5) * size
layer = blocks.centroids[:, 2] == level
fig = plt.figure(figsize=(12, 5), layout="constrained")
a = fig.add_subplot(1, 2, 1)
image = a.imshow(
    proportion[layer].reshape(count[1], count[0]),
    origin="lower",
    extent=(lo[0], lo[0] + count[0] * size, lo[1], lo[1] + count[1] * size),
    cmap="Greys",
    vmin=0,
    vmax=1,
)
slab = local.copy()
slab[local] = np.abs(xyz[local, 2] - level) < size / 2
near_inside = solid.contains(xyz[slab])
a.scatter(*xyz[slab][~near_inside, :2].T, s=6, color=GREY, label="composite outside")
a.scatter(*xyz[slab][near_inside, :2].T, s=6, color=HIGHLIGHT, label="composite inside")
a.set_aspect("equal")
a.set(
    title=f"Block proportion inside the solid, bench {level:.0f} m",
    xlabel="Easting (m)",
    ylabel="Northing (m)",
)
a.legend(loc="lower right")
fig.colorbar(image, ax=a, shrink=0.8, label="proportion of block inside")

b = fig.add_subplot(1, 2, 2, projection="3d")
shell_vertices, shell_triangles, _ = cs.block_shell(ore)
b.add_collection3d(
    Poly3DCollection(shell_vertices[shell_triangles], facecolor=ACCENT, edgecolor="none", alpha=0.35)
)
b.plot_trisurf(*vertices.T, triangles=triangles, color=LIGHT, edgecolor=GREY, linewidth=0.1, alpha=0.15)
b.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
b.set_box_aspect(np.array(hi) - lo)
b.set_title(f"{len(ore)} blocks more than half inside (shell)")
b.set_xlabel("Easting")
b.set_ylabel("Northing")
b.set_zlabel("Elevation")
b.tick_params(labelsize=6)
save(fig, "solid")

# %% [markdown]
# Whole blocks misstate the volume near the wireframe. Sub-blocking keeps whole blocks inside the solid and splits
# the partial ones on a 4 × 4 × 4 sub-grid, keeping the sub-cells whose centres are inside. Each sub-block stores its
# parent cell and its extent as fractions of that cell.

# %%
n = 4
partial = np.flatnonzero((proportion > 0) & (proportion < 1))
full = np.flatnonzero(proportion >= 1)
steps = (np.arange(n) + 0.5) / n
local = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
centres = (blocks.centroids[partial, None, :] - size / 2) + local[None, :, :] * size
inside_sub = solid.contains(centres.reshape(-1, 3)).reshape(len(partial), -1)

parent = np.concatenate([full, np.repeat(partial, inside_sub.sum(axis=1))])
full_extent = np.tile([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], (len(full), 1))
low = np.concatenate([local[mask] - 0.5 / n for mask in inside_sub])
sub_extent = np.hstack([low, low + 1 / n])
extents = np.vstack([full_extent, sub_extent])
order = np.argsort(parent, kind="stable")
subblocked = cs.BlockModel.subblocked(
    origin=lo,
    size=(size, size, size),
    count=count,
    parent=parent[order].astype(np.uint64),
    extents=extents[order],
    subgrid=(n, n, n),
)
exact = 4 / 3 * np.pi * np.prod(2 * np.sqrt(eigen))
for name, volume in (
    ("blocks more than half inside", len(ore) * size**3),
    ("sub-blocked model", subblocked.volumes.sum()),
    ("exact ellipsoid", exact),
):
    print(f"{name:>28}: {volume:,.0f} m3 ({volume / exact - 1:+.1%})")
print(subblocked)

# %%
cut = subblocked.centroids[:, 2]
level_rows = np.abs(cut - level) < size / 2
fig, ax = plt.subplots(figsize=(6.4, 5), layout="constrained")
for (x0, y0), e in zip(
    (
        subblocked.centroids[level_rows, :2]
        - size / 2 * (subblocked.extents[level_rows, 3:5] - subblocked.extents[level_rows, :2])
    ),
    subblocked.extents[level_rows],
):
    w, h = (e[3] - e[0]) * size, (e[4] - e[1]) * size
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, facecolor=ACCENT if w == size else HIGHLIGHT, edgecolor="white", lw=0.3)
    )
ax.autoscale()
ax.set_aspect("equal")
ax.set(
    title=f"Sub-blocked bench {level:.0f} m: whole blocks and sub-blocks",
    xlabel="Easting (m)",
    ylabel="Northing (m)",
)
save(fig, "subblocks")
