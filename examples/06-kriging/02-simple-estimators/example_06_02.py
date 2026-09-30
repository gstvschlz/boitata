"""
# Simple estimators

Nearest neighbor, inverse distance and moving average estimate Walker Lake `V` without a variogram. From the same
470 samples and neighborhood as ordinary kriging, they are checked against the exhaustive values, and
`compare_models` sets their grade-tonnage curves against the truth.
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
from common import ACCENT, GRAY, HIGHLIGHT, INK, map_axes, save
from matplotlib.colors import PowerNorm

samples = bt.datasets.walker_lake()
truth = bt.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
grid = bt.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
nodes = grid.centroids.astype(int)
true_at_nodes = truth[nodes[:, 1] - 1, nodes[:, 0] - 1]

azimuths = np.arange(0, 180, 22.5)
directional = [bt.experimental_variogram(samples, "V", 10.0, 120.0, azimuth=a) for a in azimuths]
model = bt.Variogram.fit_directional(
    directional, [(a, 0) for a in azimuths], ["spherical", "spherical"], weighting="count/gamma"
)

# %% [markdown]
# All four share one search: up to 24 samples in an ellipse along the direction of greatest continuity. Nearest
# neighbor takes the closest sample, inverse distance weights by 1/d², moving average weights all neighbors equally.
# Kriging, for reference, uses the variogram above.

# %%
search = bt.Search(radius=80, max_samples=24, min_samples=4, rotation=model.rotation, ratios=(0.5, 1.0))
methods = {
    "nearest neighbor": bt.NearestNeighbor(search),
    "inverse distance": bt.InverseDistance(search, power=2),
    "moving average": bt.MovingAverage(search),
    "ordinary kriging": bt.OrdinaryKriging(model, search),
}
estimates = {name: m.fit(samples, "V").predict(grid) for name, m in methods.items()}
print(f"{'method':>17}  RMSE   corr   variance ratio")
for name, e in estimates.items():
    ok = ~np.isnan(e)
    rmse = np.sqrt(np.mean((e[ok] - true_at_nodes[ok]) ** 2))
    corr = np.corrcoef(e[ok], true_at_nodes[ok])[0, 1]
    print(f"{name:>17}  {rmse:5.1f}  {corr:.3f}  {e[ok].var() / true_at_nodes[ok].var():.2f}")

# %% [markdown]
# Nearest neighbor keeps almost all the variance but places it poorly: a patchwork of polygons around the samples.
# Moving average smooths the most and is the least accurate, since distant samples weigh as much as close ones.
# Inverse distance sits between them and comes close to kriging, which is the most accurate here; kriging also gives
# a variance per estimate and accounts for clustered samples, which inverse distance does not.

# %%
shape = (60, 52)
extent = (0.5, 260.5, 0.5, 300.5)
norm = PowerNorm(0.5, vmin=0, vmax=1500)
fig, axes = plt.subplots(1, 5, figsize=(16, 4), layout="constrained")
for ax, (title, image) in zip(axes, [("truth", true_at_nodes), *estimates.items()]):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, norm=norm)
    map_axes(ax, title.capitalize())
    ax.set_ylabel("")
fig.colorbar(im, ax=axes, shrink=0.8, label="V (ppm)")
save(fig, "methods")

# %% [markdown]
# ## Grade-tonnage
#
# Smoothing shows in selection. `compare_models` computes, for each cutoff, the tonnage, mean grade and metal above it
# for every model of the same blocks, and their differences from a reference, here the truth at the nodes.

# %%
cutoffs = np.arange(0, 1001, 50)
table = bt.compare_models(grid, {"truth": true_at_nodes, **estimates}, cutoffs, reference="truth")
for cutoff in (300, 600):
    rows = np.asarray(table["cutoff"]) == cutoff
    print(f"cutoff {cutoff} ppm")
    for name, t, g in zip(
        np.asarray(table["model"])[rows],
        np.asarray(table["tonnage_diff"])[rows],
        np.asarray(table["grade_diff"])[rows],
    ):
        print(f"  {name:>17}: tonnage {t:+.0%}, grade {g:+.0%}")

colors = [INK, GRAY, ACCENT, "#8fb3d9", HIGHLIGHT]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for name, color in zip(["truth", *estimates], colors):
    rows = np.asarray(table["model"]) == name
    tonnage = np.asarray(table["tonnage"])[rows]
    a.plot(cutoffs, tonnage / tonnage[0], color=color, lw=2 if name == "truth" else 1.2, label=name)
    b.plot(cutoffs, np.asarray(table["mean_grade"])[rows], color=color, lw=2 if name == "truth" else 1.2)
a.set(xlabel="Cutoff V (ppm)", ylabel="Fraction of the area above cutoff", ylim=(0, 1.02))
a.set_title("Tonnage above cutoff")
a.legend()
b.set(xlabel="Cutoff V (ppm)", ylabel="Mean V above cutoff (ppm)")
b.set_title("Grade above cutoff")
save(fig, "grade_tonnage")

# %% [markdown]
# Smooth estimates put too much of the area above a low cutoff and too little above a high one: at 300 ppm moving
# average nearly doubles the true tonnage, at 600 ppm it misses 45 % of it. Nearest neighbor follows the true tonnage
# closely because it keeps the variance of the data, yet its estimates are the wrong ones locally; the reliable
# grade-tonnage of selected blocks is a change-of-support question (the [discrete Gaussian model](../../09-recoverable-resources/01-discrete-gaussian-model/README.md) and the pages after it).
