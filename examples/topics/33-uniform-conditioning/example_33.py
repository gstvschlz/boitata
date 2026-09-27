"""
# 33. Uniform conditioning

Kriged panels are too large to mine selectively, and kriged selective blocks are too smooth. Uniform conditioning
takes each panel's kriged grade and returns the grade-tonnage curve of the selective blocks inside it; localization
then places those blocks. The exhaustive Walker Lake grid gives the true 10 m blocks to check against.
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
from common import GRAY, HIGHLIGHT, INK, save

samples = cs.datasets.walker_lake()
truth = cs.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
weights = cs.cell_declustering(samples, "V", sizes=np.arange(2.5, 102.5, 2.5)).weights
size = 10
true_smu = truth[:, :250].reshape(30, size, 25, size).mean(axis=(1, 3))


# %% [markdown]
# Two models, both fitted along and across N170°, the major axis found in topic 19: the variogram of the Gaussian
# scores of a Hermite anamorphosis sets the change-of-support coefficient r of 10 m blocks (topic 32), and the
# variogram of the grades, rescaled to the anamorphosis variance, kriges the panels.

# %%
anam = cs.HermiteAnamorphosis(degree=40).fit(v, weights=weights)
azimuths = (170, 260)
directions = [(a, 0) for a in azimuths]
scores = [cs.experimental_variogram(xy, anam.transform(v), 10, 120, azimuth=a) for a in azimuths]
gaussian = cs.Variogram.fit_directional(scores, directions, rotation=[170, 0, 0])
r, _ = cs.change_of_support(anam, gaussian, size=(size, size), discretization=(5, 5, 1))

grades = [cs.experimental_variogram(xy, v, 10, 120, azimuth=a) for a in azimuths]
fitted = cs.Variogram.fit_directional(grades, directions, ["spherical", "spherical"], rotation=[170, 0, 0])
scale = anam.variance_ / fitted.sill
raw = cs.Variogram(
    [(s.model, s.sill * scale, s.range) for s in fitted.structures],
    nugget=fitted.nugget * scale,
    rotation=fitted.rotation,
    ratios=fitted.ratios,
)
print(f"r = {r:.3f}")
print(raw)


# %% [markdown]
# Ordinary block kriging of 50 × 50 m panels over the western 250 m. Its diagnostics give each panel the variance of
# its estimate, which sets the panel's own change-of-support coefficient: a panel estimated from few or distant
# samples, which kriging smooths more, spreads its selective blocks wider.

# %%
search = cs.Search(radius=100, max_samples=24, min_samples=4)
panels = cs.BlockModel(origin=(0.5, 0.5), size=(50, 50), count=(5, 6))
kriged = cs.BlockKriging(raw, search, size=(50, 50), discretization=(5, 5, 1)).fit(xy, v)
d = kriged.predict(panels, diagnostics=True)
panels = panels.with_columns({"V": d["value"], "estimate_variance": d["estimate_variance"]})
uc = cs.UniformConditioning(anam, r_smu=r)
cutoffs = np.linspace(0, 1000, 41)
curves = uc.grade_tonnage(panels, "V", cutoffs, estimate_variance="estimate_variance")
uc_tonnage, uc_grade = curves["tonnage"] / panels.volumes.sum(), curves["mean_grade"]


def empirical(values):
    tonnage = np.array([(values > c).mean() for c in cutoffs])
    grade = np.array([values[values > c].mean() if (values > c).any() else np.nan for c in cutoffs])
    return tonnage, grade


smus = panels.discretize(5)
direct = cs.BlockKriging(raw, search, size=(size, size), discretization=(5, 5, 1)).fit(xy, v).predict(smus)
true_curve, direct_curve = empirical(true_smu.ravel()), empirical(direct)
for c in (300, 500, 800):
    k = np.searchsorted(cutoffs, c)
    print(
        f"above {c} ppm: uniform conditioning {uc_tonnage[k]:.1%}, kriged blocks {direct_curve[0][k]:.1%}, "
        f"true {true_curve[0][k]:.1%}"
    )

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for (tonnage, grade), color, style, label in (
    (true_curve, INK, {"lw": 3, "alpha": 0.35}, "true blocks"),
    ((uc_tonnage, uc_grade), HIGHLIGHT, {"lw": 1.6}, "uniform conditioning"),
    (direct_curve, GRAY, {"lw": 1.2, "ls": ":"}, "kriged blocks"),
):
    a.plot(cutoffs, tonnage, color=color, label=label, **style)
    b.plot(cutoffs, grade, color=color, **style)
a.set(
    title=f"Proportion of {size} × {size} m blocks above cutoff",
    xlabel="Cutoff V (ppm)",
    ylabel="Proportion of blocks",
)
a.legend(fontsize=8)
b.set(title="Mean grade above cutoff", xlabel="Cutoff V (ppm)", ylabel="Mean V above cutoff (ppm)")
save(fig, "uniform-conditioning")


# %% [markdown]
# At 500 ppm uniform conditioning keeps 16.0% of the blocks against a true 16.8%, where the smoothed kriged blocks
# keep 13.6%. At 300 ppm it overstates the tonnage by a few per cent, as kriging does, and at 800 ppm, where few
# blocks remain, it thins the rich tail to 1.1% against a true 2.1%.


# %% [markdown]
# Uniform conditioning says how much of each panel is ore, not where. Localization places it: inside each panel the
# 25 blocks of `panels.discretize(5)` are ranked by their direct kriging, and the block ranked i receives the mean
# of the i-th of 25 equal-probability bands of the panel's block distribution. Every panel keeps its grade, and its
# blocks reproduce its grade-tonnage curve:

# %%
smus = smus.with_column("kriged", direct)
localized = uc.localize(smus, "kriged", panels, "V", estimate_variance="estimate_variance")["localized"]
for label, values in (("kriged", direct), ("localized", localized)):
    print(
        f"{label}: variance {values.var():.0f}, correlation with truth {np.corrcoef(values, true_smu.ravel())[0, 1]:.2f}"
    )
print(f"true blocks: variance {true_smu.var():.0f}")

fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), layout="constrained", sharey=True)
extent = (0.5, 250.5, 0.5, 300.5)
for ax, values, title in (
    (axes[0], direct, "Kriged 10 m blocks"),
    (axes[1], localized, "Localized uniform conditioning"),
    (axes[2], true_smu, "True 10 m blocks"),
):
    image = ax.imshow(np.reshape(values, (30, 25)), origin="lower", extent=extent, vmin=0, vmax=1000)
    ax.vlines(np.arange(50.5, 250, 50), 0.5, 300.5, color="white", lw=0.6, alpha=0.7)
    ax.hlines(np.arange(50.5, 300, 50), 0.5, 250.5, color="white", lw=0.6, alpha=0.7)
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="Easting (m)")
axes[0].set_ylabel("Northing (m)")
fig.colorbar(image, ax=axes, shrink=0.7, label="V (ppm); white lines bound the 50 m panels")
save(fig, "localized")


# %% [markdown]
# The localized blocks spread as the model says blocks should, and it says too little here: variance 36 264 against
# the true 47 350. Block by block they match the truth less well than kriging does (correlation 0.80 against 0.89),
# since the ranking inside a panel is only as good as the kriging that sets it; what localization keeps is each
# panel's grade and its tonnage above every cutoff.
