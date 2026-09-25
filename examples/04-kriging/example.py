import sys
from pathlib import Path

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from common import ACCENT, GREY, HIGHLIGHT, INK, fetch, map_axes, save

samples = cs.PointSet.from_table(cs.read_csv(fetch("walker-lake/sample.csv")))
truth = cs.read_csv(fetch("walker-lake/exhaustive.csv"))["V"].reshape(300, 260)
model = cs.Variogram.from_json((HERE.parent / "03-variography" / "model.json").read_text())

grid = cs.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
search = cs.Search(radius=100, max_samples=24, min_samples=4)
ok = cs.OrdinaryKriging(model, search).fit(samples.coords, samples["V"])
estimate, variance = ok.predict(grid, return_variance=True)
grid = grid.with_column("estimate", estimate).with_column("variance", variance)

nodes = grid.centroids.astype(int)
true_at_nodes = truth[nodes[:, 1] - 1, nodes[:, 0] - 1]
cv = ok.cross_validate()
print(grid)
print(f"grid: mean estimate {estimate.mean():.1f}, true {true_at_nodes.mean():.1f}")
print(f"variance of estimates {estimate.var():.0f} vs true {true_at_nodes.var():.0f}")
print(
    f"cross-validation: ME {cv.mean_error:.1f}  RMSE {cv.rmse:.1f}  r {cv.correlation:.2f}  "
    f"SSE {cv.standardized_squared_error:.2f}"
)

shape = (60, 52)
extent = (0.5, 260.5, 0.5, 300.5)
norm = PowerNorm(0.5, vmin=0, vmax=1500)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
for ax, image, title in (
    (axes[0], true_at_nodes, "True V at grid nodes"),
    (axes[1], estimate, "Ordinary kriging"),
):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, norm=norm)
    map_axes(ax, title)
axes[1].scatter(samples.coords[:, 0], samples.coords[:, 1], s=2, color=INK, linewidths=0)
fig.colorbar(im, ax=axes[:2], shrink=0.8, label="V (ppm)")
sd = axes[2].imshow(np.sqrt(variance).reshape(shape), origin="lower", extent=extent, cmap="Greys")
map_axes(axes[2], "Kriging standard deviation")
axes[2].scatter(samples.coords[:, 0], samples.coords[:, 1], s=2, color=HIGHLIGHT, linewidths=0)
fig.colorbar(sd, ax=axes[2], shrink=0.8, label="ppm")
save(fig, HERE, "maps")

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for ax, x, y, title in (
    (a, true_at_nodes, estimate, "Estimates against the truth (3 120 nodes)"),
    (b, cv.actual, cv.estimate, "Cross-validation (470 samples)"),
):
    ax.scatter(x, y, s=4, color=ACCENT, alpha=0.4, linewidths=0)
    ax.plot([0, 1600], [0, 1600], color=GREY, lw=1, ls="--", label="1:1")
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(
        [0, 1600],
        [intercept, intercept + 1600 * slope],
        color=HIGHLIGHT,
        lw=1.4,
        label=f"regression, slope {slope:.2f}",
    )
    ax.set(xlim=(0, 1600), ylim=(0, 1600), xlabel="True V (ppm)", ylabel="Estimated V (ppm)")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper left")
save(fig, HERE, "validation")
