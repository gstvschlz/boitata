"""
# direct sequential simulation

SGS simulates normal scores and back-transforms them, so it reproduces the variogram of the scores. DSS kriges the
grades themselves with their own variogram and draws each node from the declustered histogram, so it needs no
gaussian model of the field. this page runs both on walker lake and compares what each reproduces.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import warnings

import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, map_axes, save
from matplotlib.colors import PowerNorm

samples = bt.datasets.walker_lake()
truth = bt.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
weights = bt.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights
mean = np.average(v, weights=weights)
variance = np.average((v - mean) ** 2, weights=weights)
print(f"declustered mean {mean:.0f} ppm, variance {variance:.0f} ppm²")


# %% [markdown]
# each method takes the variogram of the variable it kriges: SGS the variogram of the normal scores, DSS the variogram
# of V. both are fitted along N170°, the direction of greatest continuity, and across it, and scaled to a unit sill.
# DSS rescales its nugget and sill to the declustered variance of each domain, so a unit sill is enough.

# %%
azimuth, lag, max_lag = 170.0, 10.0, 120.0
y = bt.NormalScore(tails=(0.0, v.max())).fit_transform(v, weights=weights)


def unit_sill(values):
    major = bt.experimental_variogram(xy, values, lag, max_lag, azimuth=azimuth).fit("spherical")
    minor = bt.experimental_variogram(xy, values, lag, max_lag, azimuth=azimuth + 90).fit("spherical")
    a_major = major.structures[0].range
    return bt.Variogram(
        [("spherical", major.structures[0].sill / major.sill, a_major)],
        nugget=major.nugget / major.sill,
        rotation=(azimuth, 0, 0),
        ratios=(min(minor.structures[0].range / a_major, 1.0), 1.0),
    )


gaussian, raw = unit_sill(y), unit_sill(v)
print(f"normal scores: {gaussian}\ngrades:        {raw}")


# %% [markdown]
# DSS simple kriges each node with the declustered mean and draws it from the declustered histogram, as the
# back-transform of a gaussian whose transformed mean and variance equal the kriged ones. near the tails, or where
# kriging extrapolates beyond the data, no such draw exists; DSS takes the nearest reachable pair and warns with the
# share of nodes it moved. both simulators run 30 realizations from the same seed.

# %%
grid = bt.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
search = bt.Search(radius=100, max_samples=24)
options = {"n": 30, "seed": 42, "keep": True, "progress": False}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    dss = bt.DSS(raw, search).fit(samples, "V", weights=weights).simulate(grid, **options)
for warning in caught:
    print(warning.message)
sgs = bt.SGS(gaussian, search).fit(samples, "V", weights=weights).simulate(grid, **options)

nodes = grid.centroids.astype(int)
true_at_nodes = truth[nodes[:, 1] - 1, nodes[:, 0] - 1]
for name, s in (("DSS", dss), ("SGS", sgs)):
    reals = s.realizations
    print(
        f"{name}: mean {reals.mean():.0f}, variance {reals.var(axis=1).mean():.0f}, "
        f"area above 500 ppm {np.mean(reals > 500):.1%}"
    )
print(
    f"true: mean {true_at_nodes.mean():.0f}, variance {true_at_nodes.var():.0f}, "
    f"area above 500 ppm {np.mean(true_at_nodes > 500):.1%}"
)


# %% [markdown]
# the DSS realization puts the high grades where the SGS one does. its low-grade areas form wider patches at the
# lowest grades: where the kriged mean falls below anything the histogram reaches, DSS takes the bottom of the
# histogram. the means of 30 realizations agree.

# %%
shape = (60, 52)
extent = (0.5, 260.5, 0.5, 300.5)
norm = PowerNorm(0.5, vmin=0, vmax=1500)
fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), layout="constrained")
axes[1, 0].scatter(*xy[:, :2].T, c=v, s=6, norm=norm)
map_axes(axes[1, 0], "Samples")
axes[1, 0].set(xlim=extent[:2], ylim=extent[2:])
for ax, image, title in (
    (axes[0, 0], true_at_nodes, "True V at grid nodes"),
    (axes[0, 1], dss.realizations[0], "DSS realization 1"),
    (axes[0, 2], sgs.realizations[0], "SGS realization 1"),
    (axes[1, 1], dss.mean, "Mean of 30 DSS realizations"),
    (axes[1, 2], sgs.mean, "Mean of 30 SGS realizations"),
):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, norm=norm)
    map_axes(ax, title)
fig.colorbar(im, ax=axes, shrink=0.6, label="V (ppm)")
save(fig, "maps")


# %% [markdown]
# SGS reproduces the declustered histogram. DSS draws too many of the lowest grades: these are the nodes the warning
# counted. on V, the DSS realizations follow their model, rescaled to the declustered variance (dashed). the SGS
# realizations come as close, because on walker lake the variograms of the scores and of V have about the same shape.
# the samples sit above both curves since they cluster in the high grades.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
order = np.argsort(v)
h = np.linspace(0, max_lag, 200)
data_exp = bt.experimental_variogram(xy, v, lag, max_lag, azimuth=azimuth)
for ax, name, s, color in ((axes[1], "DSS", dss, ACCENT), (axes[2], "SGS", sgs, HIGHLIGHT)):
    for r in s.realizations:
        axes[0].plot(np.sort(r), np.linspace(0, 1, r.size), color=color, lw=0.5, alpha=0.25)
        exp = bt.experimental_variogram(grid.centroids, r, lag, max_lag, azimuth=azimuth)
        ax.plot(exp.lags, exp.gammas / 1e3, color=LIGHT, lw=0.8)
    axes[0].plot([], [], color=color, label=f"30 {name} realizations")
    ax.plot([], [], color=LIGHT, label=f"30 {name} realizations")
    ax.plot(data_exp.lags, data_exp.gammas / 1e3, "o", color=INK, ms=4, label="samples")
    ax.plot(h, variance * raw.gamma(h) / 1e3, color=color, lw=1.4, label="model of V")
    ax.axhline(variance / 1e3, color=GRAY, lw=0.8, ls="--")
    ax.set(xlim=(0, max_lag), ylim=(0, 1.6 * variance / 1e3), xlabel="Lag distance (m)")
    ax.set(ylabel="γ(h) of V (10³ ppm²)", title=f"{name} variogram of V, N{azimuth:.0f}°")
    ax.legend(loc="lower right")
axes[0].step(
    v[order],
    np.cumsum(weights[order]) / weights.sum(),
    color=INK,
    lw=1.4,
    where="post",
    label="declustered samples",
)
axes[0].set(xlim=(0, 1600), xlabel="V (ppm)", ylabel="Cumulative probability", title="Histogram reproduction")
axes[0].legend(loc="lower right")
save(fig, "reproduction")

# %% [markdown]
# [sequential gaussian simulation](../../08-stochastic-simulation/01-sgs/README.md) covers the summary that `simulate`
# returns, and [simulation at block support](../../08-stochastic-simulation/03-simulation-at-block-support/README.md)
# averages realizations to blocks.
