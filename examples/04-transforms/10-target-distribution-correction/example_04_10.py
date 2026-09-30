"""
# Target distribution correction

Each realization of a simulation reproduces the declustered histogram of its data only on average: over a small
domain, or with a variogram range large against it, single realizations drift high or low. `correct_distribution`
maps each realization rank for rank onto a target distribution, the declustered data or a fitted reference such as a
`KernelDensity`, so every realization takes the target histogram exactly while its values keep their order, and so
their spatial pattern. A `strength` below 1 moves the values only part of the way; `realizations=` limits the
correction to those a `check_realizations` shows outside a tolerance.
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
from common import map_axes, save

# %% [markdown]
# ## Realizations that drift
#
# Walker Lake V simulated as in [realization checks](../../10-checking-models/04-realization-checks/README.md): declustered normal scores, their variogram along N170° and across it, and
# 20 SGS realizations on a 5 m grid. The field is only a few variogram ranges across, so each realization samples
# the histogram of the data unevenly: their means stray up to 12 % from the declustered mean.

# %%
samples = bt.datasets.walker_lake()
xy, v = samples.coords, samples["V"]
weights = bt.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights
y = bt.NormalScore(tails=(0.0, v.max())).fit_transform(v, weights=weights)

azimuth, lag, max_lag = 170.0, 10.0, 120.0
major = bt.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth).fit("spherical")
minor = bt.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth + 90).fit("spherical")
total, a_major = major.sill, major.structures[0].range
gaussian = bt.Variogram(
    [("spherical", major.structures[0].sill / total, a_major)],
    nugget=major.nugget / total,
    rotation=(azimuth, 0, 0),
    ratios=(min(minor.structures[0].range / a_major, 1.0), 1.0),
)
grid = bt.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
sgs = bt.SGS(gaussian, bt.Search(radius=100, max_samples=24)).fit(samples, "V", weights=weights)
summary = sgs.simulate(grid, n=20, seed=42, keep=True)
reals = summary.realizations

check = bt.check_realizations(grid, reals, samples, "V", weights=weights)
means = check.statistics["mean"]
print(
    f"declustered mean {means[0]:.0f} ppm; realization means {means[1:].min():.0f} to {means[1:].max():.0f} ppm"
)

# %% [markdown]
# ## Correcting to the declustered data
#
# With the data as the reference, the value ranked `i` of the `m` values of a realization goes to the declustered
# quantile at `(i + 0.5) / m`. Every corrected realization then has the declustered distribution of the data, and the
# band of realization CDFs collapses onto it. `strength=0.5` moves each value half way: the band narrows around the
# target but keeps some of the spread between realizations.

# %%
corrected = bt.correct_distribution(reals, v, weights=weights)
half = bt.correct_distribution(reals, v, weights=weights, strength=0.5)
checks = {
    "SGS realizations": check,
    "strength 0.5": bt.check_realizations(grid, half, samples, "V", weights=weights),
    "strength 1": bt.check_realizations(grid, corrected, samples, "V", weights=weights),
}
for name, c in checks.items():
    m = c.statistics["mean"][1:]
    print(f"{name:17} means {m.min():6.0f} to {m.max():6.0f} ppm")
kept = all(np.all(np.diff(c[np.argsort(r, kind="stable")]) >= 0) for r, c in zip(reals, corrected))
print(f"order kept in every realization: {kept}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained", sharey=True)
for ax, (name, c) in zip(axes, checks.items()):
    bt.plot.histogram_reproduction(c, ax=ax)
    ax.set(xlim=(0, 1200), xlabel="V (ppm)", title=name)
for ax in axes[1:]:
    ax.set_ylabel("")
save(fig, "histograms")

# %% [markdown]
# The correction changes values, not their order: a corrected realization is the same map with its grades
# restretched. The realization furthest from the declustered mean, below, shifts as a whole; its highs and lows stay
# where they were.

# %%
low = int(np.argmax(np.abs(means[1:] - means[0])))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained", sharey=True)
for ax, values, title in zip(axes, [reals[low], corrected[low]], ["Realization", "Corrected"]):
    bt.plot.section(grid, values, vmin=0, vmax=1200, colorbar=False, ax=ax)
    map_axes(ax, f"{title} {low + 1}, mean {values.mean():.0f} ppm")
axes[1].set_ylabel("")
fig.colorbar(axes[1].collections[0], ax=axes, label="V (ppm)", shrink=0.8)
save(fig, "maps")

# %% [markdown]
# ## Only the realizations out of tolerance
#
# A realization whose mean lies within 5 % of the declustered mean is fine as it stands. `realizations=` takes the
# indices to correct, here those `check_realizations` puts outside that band; the others come back unchanged, so the
# set keeps its spread where the spread is plausible.

# %%
outside = np.flatnonzero(np.abs(means[1:] / means[0] - 1) > 0.05)
selected = bt.correct_distribution(reals, v, weights=weights, realizations=outside)
m = bt.check_realizations(grid, selected, samples, "V", weights=weights).statistics["mean"][1:]
print(f"{len(outside)} of {len(reals)} realizations corrected; means now {m.min():.0f} to {m.max():.0f} ppm")

# %% [markdown]
# ## A smooth reference
#
# The declustered data stop at their highest value, and a corrected realization with it. A `KernelDensity` fitted
# with the declustering weights and reflected at 0 ([reference distributions](../../03-exploratory-analysis/13-reference-distributions/README.md)) has a tail beyond the data; as the reference, it lets
# the top values of each realization run past the highest sample.

# %%
kde = bt.KernelDensity(lower=0.0).fit(v, weights=weights)
smooth = bt.correct_distribution(reals, kde)
print(f"highest datum {v.max():.0f} ppm; highest corrected value {corrected.max():.0f} ppm")
print(f"highest value corrected to the kernel density {smooth.max():.0f} ppm")
