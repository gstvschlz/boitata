import sys
from pathlib import Path

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from common import ACCENT, GREY, HIGHLIGHT, LIGHT, fetch, save


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
save(fig, HERE, "solid")
