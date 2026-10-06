"""
# sequential gaussian simulation

kriging gives one smooth map. simulation draws many maps, each honoring the samples, the declustered histogram and
the variogram; together they measure uncertainty.
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
from matplotlib.colors import PowerNorm

samples = bt.datasets.walker_lake()
truth = bt.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
azimuth = 170.0


# %% [markdown]
# normal scores with declustering weights, tails bounded to 0 and the largest sample, and their variogram along
# N170°, the direction of greatest continuity ([variogram fitting](../../05-spatial-continuity/02-variogram-fitting/README.md)), and across it, scaled to a unit sill:

# %%
weights = bt.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights
y = bt.NormalScore(tails=(0.0, v.max())).fit_transform(v, weights=weights)

lag, max_lag = 10.0, 120.0
major = bt.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth).fit("spherical")
minor = bt.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth + 90).fit("spherical")
total = major.sill
a_major = major.structures[0].range
ratio = min(minor.structures[0].range / a_major, 1.0)
gaussian = bt.Variogram(
    [("spherical", major.structures[0].sill / total, a_major)],
    nugget=major.nugget / total,
    rotation=(azimuth, 0, 0),
    ratios=(ratio, 1.0),
)
print(gaussian)


# %% [markdown]
# SGS normal-scores the data, simulates along a random path and back-transforms. the 50 realizations come out the
# same on any number of threads. `simulate` returns a summary accumulated while it runs: the mean, variance and
# quantiles at each node, the probability and mean above each cutoff, and each realization's global mean and share
# above the cutoffs. `cv` and `relative_error` derive from these: the spread over the mean, and half the central 90 %
# interval over the mean, read from the 0.05 and 0.95 quantiles. it keeps realizations only on request: here the first
# 20, to check them.

# %%
grid = bt.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
sgs = bt.SGS(gaussian, bt.Search(radius=100, max_samples=24)).fit(samples, "V", weights=weights)
summary = sgs.simulate(grid, n=50, seed=42, cutoffs=[500.0], quantiles=[0.05, 0.95], keep=range(20))
reals = summary.realizations
etype = summary.mean
p500 = summary.probability_above[:, 0]

nodes = grid.coords.astype(int)
true_at_nodes = truth[nodes[:, 1] - 1, nodes[:, 0] - 1]
means = summary.realization_mean
print(f"realization means {means.min():.0f}-{means.max():.0f}, true {true_at_nodes.mean():.0f}")
print(f"realization variance {reals.var(axis=1).mean():.0f}, true {true_at_nodes.var():.0f}")
low, high = np.quantile(summary.realization_above[:, 0], [0.1, 0.9])
print(f"area above 500 ppm: P10 {low:.1%}, P90 {high:.1%}, true {np.mean(true_at_nodes > 500):.1%}")
error = summary.relative_error(confidence=0.9)
print(
    f"median cv {np.nanmedian(summary.cv):.2f}; nodes within ±50 % at 90 % confidence: {np.mean(error < 0.5):.0%}"
)


# %% [markdown]
# each realization looks like the truth; their mean is smooth like kriging, and their spread measures uncertainty.

# %%
shape = (60, 52)
extent = (0.5, 260.5, 0.5, 300.5)
norm = PowerNorm(0.5, vmin=0, vmax=1500)
fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), layout="constrained")
for ax, image, title in (
    (axes[0, 0], true_at_nodes, "True V at grid nodes"),
    (axes[0, 1], reals[0], "Realization 1"),
    (axes[0, 2], reals[1], "Realization 2"),
    (axes[1, 0], etype, "Mean of 50 realizations"),
):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, norm=norm)
    map_axes(ax, title)
fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="V (ppm)")
fig.colorbar(im, ax=axes[1, 0], shrink=0.8, label="V (ppm)")
spread = axes[1, 1].imshow(summary.std.reshape(shape), origin="lower", extent=extent, cmap="cividis")
map_axes(axes[1, 1], "Spread across realizations")
fig.colorbar(spread, ax=axes[1, 1], shrink=0.8, label="standard deviation (ppm)")
prob = axes[1, 2].imshow(p500.reshape(shape), origin="lower", extent=extent, vmin=0, vmax=1)
map_axes(axes[1, 2], "Probability V > 500 ppm")
axes[1, 2].contour(
    (true_at_nodes > 500).reshape(shape).astype(float),
    levels=[0.5],
    origin="lower",
    extent=extent,
    colors=HIGHLIGHT,
    linewidths=0.8,
)
fig.colorbar(prob, ax=axes[1, 2], shrink=0.8, label="probability; true V > 500 outlined")
save(fig, "maps")


# %% [markdown]
# each realization reproduces the declustered histogram and the model variogram:

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
grid_v = np.sort(true_at_nodes)
for r in reals:
    a.plot(np.sort(r), np.linspace(0, 1, r.size), color=LIGHT, lw=0.8)
a.plot(grid_v, np.linspace(0, 1, grid_v.size), color=INK, lw=1.4, label="truth")
order = np.argsort(v)
a.step(
    v[order],
    np.cumsum(weights[order]) / weights.sum(),
    color=ACCENT,
    lw=1.4,
    where="post",
    label="declustered samples",
)
a.plot([], [], color=LIGHT, label="20 realizations")
a.set(xlim=(0, 1600), xlabel="V (ppm)", ylabel="Cumulative probability", title="Histogram reproduction")
a.legend(loc="lower right")

xyz = grid.coords
for r in reals:
    scores = bt.NormalScore().fit_transform(r)
    exp = bt.experimental_variogram(xyz, scores, lag, max_lag, azimuth=azimuth)
    b.plot(exp.lags, exp.gammas, color=LIGHT, lw=0.8)
data_exp = bt.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth)
b.plot(data_exp.lags, data_exp.gammas, "o", color=ACCENT, ms=4, label="normal scores of samples")
h = np.linspace(0, max_lag, 200)
b.plot(h, gaussian.gamma(h), color=HIGHLIGHT, lw=1.4, label="model")
b.plot([], [], color=LIGHT, label="20 realizations")
b.axhline(1.0, color=GRAY, lw=0.8, ls="--")
b.set(xlim=(0, max_lag), ylim=(0, 1.4), xlabel="Lag distance (m)", ylabel="γ(h) of normal scores")
b.set_title(f"Variogram reproduction, N{azimuth:.0f}°")
b.legend(loc="lower right")
save(fig, "reproduction")

# %% [markdown]
# [simulation at block support](../../08-stochastic-simulation/03-simulation-at-block-support/README.md) targets blocks, [turning bands](../../08-stochastic-simulation/04-turning-bands/README.md) draws realizations another way, and [simulation with a trend](../../08-stochastic-simulation/05-simulation-with-trend/README.md) steers them with a
# trend.
