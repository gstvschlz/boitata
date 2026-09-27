"""
# 37. Simulation at block support

Simulation reaches block support by averaging: each realization on fine nodes is averaged over the nodes of every
selective block, so no change-of-support model is needed. The realizations give block grade-tonnage curves with
their uncertainty, and, pooled inside panels, localized block grades. The exhaustive Walker Lake grid gives the true
10 m blocks.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, INK, save

samples = cs.datasets.walker_lake()
truth = cs.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
weights = cs.cell_declustering(samples, "V", sizes=np.arange(2.5, 102.5, 2.5)).weights
size = 10


# %% [markdown]
# SGS needs the variogram of normal scores with a unit sill: fitted along and across N170° (topic 19), then divided
# by its sill.

# %%
y = cs.NormalScore().fit(v, weights=weights).transform(v)
azimuths = (170, 260)
experimental = [cs.experimental_variogram(xy, y, 10, 120, azimuth=a) for a in azimuths]
fitted = cs.Variogram.fit_directional(experimental, [(a, 0) for a in azimuths], rotation=[170, 0, 0])
gaussian = cs.Variogram(
    [(s.model, s.sill / fitted.sill, s.range) for s in fitted.structures],
    nugget=fitted.nugget / fitted.sill,
    rotation=fitted.rotation,
    ratios=fitted.ratios,
)
print(gaussian)


# %% [markdown]
# Thirty realizations on 2.5 m nodes. `blocks=` averages each over the 16 nodes of every 10 × 10 m block before
# summarizing, so `cutoffs=` gives one block tonnage curve per realization:

# %%
cutoffs = np.linspace(0, 1000, 41)


def empirical(values):
    tonnage = np.array([(values > c).mean() for c in cutoffs])
    grade = np.array([values[values > c].mean() if (values > c).any() else np.nan for c in cutoffs])
    return tonnage, grade


nodes = cs.BlockModel(origin=(0.5, 0.5), size=(2.5, 2.5), count=(104, 120))
blocks = cs.BlockModel(origin=(0.5, 0.5), size=(size, size), count=(26, 30))
sgs = cs.SGS(gaussian, cs.Search(radius=100, max_samples=24)).fit(xy, v, weights=weights)
summary = sgs.simulate(nodes, n=30, seed=7, cutoffs=list(cutoffs), blocks=blocks)
low, high = np.quantile(summary.realization_above, [0.1, 0.9], axis=1)
true_block = empirical(truth.reshape(30, size, 26, size).mean(axis=(1, 3)).ravel())
for c in (300, 500, 800):
    k = np.searchsorted(cutoffs, c)
    print(f"above {c} ppm: P10 {low[k]:.1%}, P90 {high[k]:.1%}, true {true_block[0][k]:.1%}")

fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
ax.fill_between(cutoffs, low, high, color=ACCENT, alpha=0.25, lw=0, label="30 simulations, P10–P90")
ax.plot(cutoffs, true_block[0], color=INK, lw=1.2, label="true block averages")
ax.set(
    title=f"Proportion of {size} × {size} m blocks above cutoff",
    xlabel="Cutoff V (ppm)",
    ylabel="Proportion of blocks",
)
ax.legend()
save(fig, "simulated-blocks")


# %% [markdown]
# From 500 ppm up the band holds the true curve; at 300 ppm every realization puts a few per cent more blocks above
# cutoff than the truth has.


# %% [markdown]
# Localization pools the realizations panel by panel. Over the western 250 m, each 50 × 50 m panel holds 25 blocks;
# 25 blocks × 30 realizations give 750 values, sorted and cut into 25 chunks of 30, and the block ranked i by
# ordinary block kriging receives the mean of chunk i. The kriging uses the grade variogram, fitted like the one above.

# %%
grades = [cs.experimental_variogram(xy, v, 10, 120, azimuth=a) for a in azimuths]
grade = cs.Variogram.fit_directional(
    grades, [(a, 0) for a in azimuths], ["spherical", "spherical"], rotation=[170, 0, 0]
)
panels = cs.BlockModel(origin=(0.5, 0.5), size=(50, 50), count=(5, 6))
smus = panels.discretize(5)
search = cs.Search(radius=100, max_samples=24, min_samples=4)
kriged = cs.BlockKriging(grade, search, size=(size, size), discretization=(5, 5, 1)).fit(xy, v).predict(smus)
smus = smus.with_column("kriged", kriged)

west = cs.BlockModel(origin=(0.5, 0.5), size=(2.5, 2.5), count=(100, 120))
ensemble = sgs.simulate(west, n=30, seed=11, blocks=smus, realizations=True)
localized = cs.localize(smus, "kriged", panels, ensemble.realizations)["localized"]
true_smu = truth[:, :250].reshape(30, size, 25, size).mean(axis=(1, 3))
for label, values in (("kriged", kriged), ("localized", localized)):
    print(
        f"{label}: variance {values.var():.0f}, correlation with truth {np.corrcoef(values, true_smu.ravel())[0, 1]:.2f}"
    )
print(f"true blocks: variance {true_smu.var():.0f}")
true_curve, kriged_curve, sim_curve = empirical(true_smu.ravel()), empirical(kriged), empirical(localized)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained", width_ratios=(1, 1.3))
extent = (0.5, 250.5, 0.5, 300.5)
image = a.imshow(np.reshape(localized, (30, 25)), origin="lower", extent=extent, vmin=0, vmax=1000)
a.vlines(np.arange(50.5, 250, 50), 0.5, 300.5, color="white", lw=0.6, alpha=0.7)
a.hlines(np.arange(50.5, 300, 50), 0.5, 250.5, color="white", lw=0.6, alpha=0.7)
a.set_aspect("equal")
a.set(title="Localized simulation", xlabel="Easting (m)", ylabel="Northing (m)")
fig.colorbar(image, ax=a, shrink=0.8, label="V (ppm); white lines bound the 50 m panels")
for (tonnage, _), color, style, label in (
    (true_curve, INK, {"lw": 3, "alpha": 0.35}, "true blocks"),
    (sim_curve, ACCENT, {"lw": 1.4, "ls": "--"}, "localized simulation"),
    (kriged_curve, GRAY, {"lw": 1.2, "ls": ":"}, "kriged blocks"),
):
    b.plot(cutoffs, tonnage, color=color, label=label, **style)
b.set(
    title=f"Proportion of {size} × {size} m blocks above cutoff",
    xlabel="Cutoff V (ppm)",
    ylabel="Proportion of blocks",
)
b.legend(fontsize=8)
save(fig, "localized-simulation")


# %% [markdown]
# Each panel keeps the mean of its realizations, and its blocks the spread the simulation gives them: variance 40 836,
# between kriging's 35 064 and the true 47 350, with a correlation to the truth of 0.85 against kriging's 0.89. No
# change-of-support model is involved; what the pooling returns is only as good as the realizations. Topics 33 and 34
# localize the same panels from a change-of-support model.
