"""
# censored normal-score transform

in the tailings reprocessing dataset, 46 % of the cyanide (WAD) samples are below the detection limit: those cells
hold the limit itself, flagged by `CN_WAD_BDL`. a plain normal-score transform sorts the data by value, so the ties at
the limit break in the order of the assay sheet. `fit(..., censored=...)` shuffles each such tie with a seed instead.
the below-detection scores then carry no order, and every other value, including ties at a different limit, keeps its
exact rank.
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
from common import ACCENT, GRAY, HIGHLIGHT, LIGHT, map_axes, save

assays = bt.datasets.tailings_reprocessing()["assays"]
ok = ~np.isnan(assays["CN_WAD_PPM"])
cn, bdl = assays["CN_WAD_PPM"][ok], assays["CN_WAD_BDL"][ok].astype(bool)
print(f"{len(cn)} composites; {bdl.mean():.0%} below the {cn[bdl][0]:g} ppm detection limit")

fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
bins = np.linspace(0, np.percentile(cn, 99), 60)
ax.hist(cn, bins=bins, color=LIGHT, edgecolor=GRAY, lw=0.4, label="all composites")
ax.hist(cn[bdl], bins=bins, color=HIGHLIGHT, label="below detection")
ax.set(xlabel="CN WAD (ppm)", ylabel="Composites", title="A spike no transform can spread on its own")
ax.legend()
save(fig, "histogram")

# %% [markdown]
# ## a tie broken by row order is still a tie
#
# `NormalScore().fit(cn)` ignores the flag and sorts the below-detection composites by value. they are all equal, so the
# stable sort leaves them in the order of the assay table. compare the score of each composite below detection with the
# one before it in that table to see the artifact.

# %%
naive = bt.NormalScore().fit_transform(cn)


def rising_fraction(scores):
    tied = scores[bdl]
    return np.mean(tied[1:] > tied[:-1])


print(f"naive: {rising_fraction(naive):.0%} of consecutive below-detection rows have a rising score")

# %% [markdown]
# almost every consecutive pair rises: the transform asserts that cyanide increases down the assay sheet within the
# censored group, and the data say nothing about that. `censored=` shuffles the tie with a seed instead.

# %%
ns = bt.NormalScore()
aware = ns.fit_transform(cn, censored=bdl, seed=0)
print(
    f"censored=: {rising_fraction(aware):.0%} rising — a coin flip, as it should be for indistinguishable values"
)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, layout="constrained")
row = np.arange(bdl.sum())
for ax, scores, title in ((axes[0], naive, "Ignoring the flag"), (axes[1], aware, "censored=bdl, seed=0")):
    ax.scatter(row, scores[bdl], s=3, color=ACCENT, alpha=0.5, linewidths=0)
    ax.set(xlabel="Row among below-detection composites", title=title)
axes[0].set_ylabel("Normal score")
save(fig, "ties")

# %% [markdown]
# ## order is otherwise untouched
#
# values above and below the limit, and ties at any other value, keep the ranks that `NormalScore` gives them without
# `censored`. the scores of a censored fit back-transform to the reported data, limit values included, since the
# transform invents nothing below the limit. inventing values there needs a `reference` distribution, which you cannot
# combine with `censored`.

# %%
above = cn > cn[bdl][0]
print(
    f"lowest score above the limit exceeds every below-detection score: {aware[above].min() > aware[bdl].max()}"
)
back = ns.inverse_transform(aware)
print(f"round-trips to the reported data: {np.allclose(back, cn, atol=1e-6)}")

# %% [markdown]
# ## composing with external drift kriging
#
# the soil geochemistry survey ([external drift kriging](../../06-kriging/04-external-drift-kriging/README.md)) censors
# gold and arsenic too. normal-score `AU_PPB` and `AS_PPM` with their own `_BDL` flags, then krige the gold score with
# magnetics and elevation as external drift. the map below never reads an order into the samples at a detection limit.

# %%
soil = bt.datasets.soil_geochemistry_survey()
samples, covariates = soil["samples"], soil["covariates"]
rows = covariates.row_at(samples.coords[:, :2])
samples = samples.with_column("MAG_NT", covariates["MAG_NT"][rows]).with_column(
    "ELEVATION_M", covariates["ELEVATION_M"][rows]
)
au_bdl, as_bdl = samples["AU_BDL"].astype(bool), samples["AS_BDL"].astype(bool)
au_ns, as_ns = bt.NormalScore(), bt.NormalScore()
au_scores = au_ns.fit_transform(samples["AU_PPB"], censored=au_bdl, seed=0)
as_ns.fit_transform(samples["AS_PPM"], censored=as_bdl, seed=0)
print(
    f"AU_PPB {au_bdl.mean():.0%} censored, AS_PPM {as_bdl.mean():.0%} censored; both normal-scored the same way"
)

model = bt.experimental_variogram(samples.coords, au_scores, 200.0, 2500.0).fit(["spherical"])
search = bt.Search(radius=2500.0, max_samples=24, min_samples=4)
edk = (
    bt.ExternalDriftKriging(model, search, ["MAG_NT", "ELEVATION_M"])
    .fit(samples, au_scores)
    .predict(covariates)
)
grade = au_ns.inverse_transform(np.nan_to_num(edk))
grade[np.isnan(edk)] = np.nan
inside = covariates["INSIDE"] == 1
print(f"kriged Au inside the survey: mean {np.nanmean(np.where(inside, grade, np.nan)):.1f} ppb")

nx, ny = covariates.count[0], covariates.count[1]
ox, oy, sx, sy = covariates.origin[0], covariates.origin[1], covariates.size[0], covariates.size[1]
fig, ax = plt.subplots(figsize=(5.5, 4.2), layout="constrained")
im = ax.imshow(
    grade.reshape(ny, nx),
    origin="lower",
    extent=(ox, ox + sx * nx, oy, oy + sy * ny),
    vmax=np.nanpercentile(grade, 98),
)
ax.scatter(samples.coords[:, 0], samples.coords[:, 1], s=2, color="white", linewidths=0)
map_axes(ax, "Au (ppb), external drift on censored scores")
fig.colorbar(im, ax=ax, shrink=0.8, label="Au (ppb)")
save(fig, "map")
