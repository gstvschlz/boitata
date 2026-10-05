"""
# simulation at block support

simulation reaches block support by averaging each realization on fine nodes over the nodes of each selective block,
with no change-of-support model. the realizations give block grade-tonnage curves with their uncertainty and, pooled
inside panels, localized block grades. the exhaustive walker lake grid gives the true 10 m blocks.
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
from common import ACCENT, GRAY, INK, save

samples = bt.datasets.walker_lake()
truth = bt.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
weights = bt.cell_declustering(samples, "V", sizes=np.arange(2.5, 102.5, 2.5)).weights
size = 10


# %% [markdown]
# SGS needs the variogram of normal scores with a unit sill: fitted along and across N170° ([variogram fitting](../../05-spatial-continuity/02-variogram-fitting/README.md)), then divided
# by its sill.

# %%
y = bt.NormalScore().fit(v, weights=weights).transform(v)
azimuths = (170, 260)
experimental = [bt.experimental_variogram(xy, y, 10, 120, azimuth=a) for a in azimuths]
fitted = bt.Variogram.fit_directional(experimental, [(a, 0) for a in azimuths], rotation=[170, 0, 0])
gaussian = bt.Variogram(
    [(s.model, s.sill / fitted.sill, s.range) for s in fitted.structures],
    nugget=fitted.nugget / fitted.sill,
    rotation=fitted.rotation,
    ratios=fitted.ratios,
)
print(gaussian)


# %% [markdown]
# thirty realizations on 2.5 m nodes. `blocks=` averages each over the 16 nodes of each 10 × 10 m block before
# summarizing, and `grade_tonnage_cutoffs=` accumulates each realization's block grade-tonnage curve as it runs, with
# tonnes from `density` times block area here. `grade_tonnage` then reads the P10, P50 and P90 of the tonnage and the
# mean grade above each cutoff, each quantity on its own.

# %%
cutoffs = np.linspace(0, 1000, 41)


def empirical(values):
    tonnage = np.array([(values >= c).mean() for c in cutoffs])
    grade = np.array([values[values >= c].mean() if (values >= c).any() else np.nan for c in cutoffs])
    return tonnage, grade


nodes = bt.BlockModel(origin=(0.5, 0.5), size=(2.5, 2.5), count=(104, 120))
blocks = bt.BlockModel(origin=(0.5, 0.5), size=(size, size), count=(26, 30))
sgs = bt.SGS(gaussian, bt.Search(radius=100, max_samples=24)).fit(xy, v, weights=weights)
summary = sgs.simulate(nodes, n=30, seed=7, blocks=blocks, grade_tonnage_cutoffs=list(cutoffs), density=1.0)
curves = summary.grade_tonnage(probabilities=[0.1, 0.5, 0.9])
true_block = empirical(truth.reshape(30, size, 26, size).mean(axis=(1, 3)).ravel())
share = np.asarray(curves["tonnage"]) / curves["tonnage"][0]
for c in (300, 500, 800):
    k = np.searchsorted(cutoffs, c)
    p10, p90 = share[(curves["cutoff"] == cutoffs[k]) & np.isin(curves["probability"], [0.1, 0.9])]
    print(f"above {c} ppm: P10 {p10:.1%}, P90 {p90:.1%}, true {true_block[0][k]:.1%}")

fig, ax = bt.plot.grade_tonnage(curves, relative=True)
fig.set_size_inches(6.4, 3.8)
ax.plot(cutoffs, true_block[0], color=INK, lw=1.2)
fig.axes[-1].plot(cutoffs, true_block[1], color=INK, lw=1.2, ls="--")
ax.set(
    title=f"{size} × {size} m blocks: 30 simulations against the true blocks (black)", xlabel="Cutoff V (ppm)"
)
save(fig, "simulated-blocks")


# %% [markdown]
# the band holds the true curve at each cutoff; at 300 ppm the truth sits at its lower edge.


# %% [markdown]
# a mine plan reads uncertainty over production volumes. `groups=` labels each block with its volume, here six 50 m
# strips of northing standing for periods, and summarizes each realization's mean per period, one row per label.
# `window=` gives each block the mean of the 50 × 50 m box centred on it, the boxes overlapping. both stream like
# `blocks=` and feed the same summaries, here the relative error at 90 % confidence: half the P5–P95 interval over the
# mean.

# %%
periods = blocks.centroids[:, 1] // 50
for label, volumes in (
    ("blocks", {}),
    ("periods", {"groups": periods}),
    ("50 m windows", {"window": (50, 50)}),
):
    error = sgs.simulate(
        nodes, n=30, seed=7, blocks=blocks, quantiles=[0.05, 0.95], **volumes
    ).relative_error()
    print(f"{label}: {len(error)} rows, median relative error {np.median(error):.1%}")


# %% [markdown]
# the median error falls from 59 % on single blocks to 24 % on 50 m windows and 11 % on periods of 130 blocks.


# %% [markdown]
# localization pools the realizations panel by panel. over the western 250 m, each 50 × 50 m panel holds 25 blocks.
# 25 blocks × 30 realizations give 750 values, sorted and cut into 25 chunks of 30, and the block ranked i by
# ordinary block kriging receives the mean of chunk i. the kriging uses the grade variogram, fitted like the one above.

# %%
grades = [bt.experimental_variogram(xy, v, 10, 120, azimuth=a) for a in azimuths]
grade = bt.Variogram.fit_directional(
    grades, [(a, 0) for a in azimuths], ["spherical", "spherical"], rotation=[170, 0, 0]
)
panels = bt.BlockModel(origin=(0.5, 0.5), size=(50, 50), count=(5, 6))
smus = panels.discretize(5)
search = bt.Search(radius=100, max_samples=24, min_samples=4)
kriged = bt.BlockKriging(grade, search, size=(size, size), discretization=(5, 5, 1)).fit(xy, v).predict(smus)
smus = smus.with_column("kriged", kriged)

west = bt.BlockModel(origin=(0.5, 0.5), size=(2.5, 2.5), count=(100, 120))
ensemble = sgs.simulate(west, n=30, seed=11, blocks=smus, keep=True)
localized = bt.localize(smus, "kriged", panels, ensemble.realizations)["localized"]
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
# each panel keeps the mean of its realizations, and its blocks the spread the simulation gives them: variance 41 411,
# between kriging's 35 064 and the true 47 350, with a correlation to the truth of 0.86 against kriging's 0.89. the
# pooling uses no change-of-support model, so its result depends on the realizations alone. [uniform conditioning](../../09-recoverable-resources/02-uniform-conditioning/README.md) and [MIK localization](../../09-recoverable-resources/03-mik-localization/README.md)
# localize the same panels from a change-of-support model.
