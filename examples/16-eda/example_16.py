"""
# 16. Exploratory data analysis

Statistics per domain, top cuts, contacts, swaths, h-scatterplots and correlations on the 2 m composites of the
drillhole dataset. Every function skips missing values, so raw columns go in as they are.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GREY, INK, save

tables = cs.datasets.drillhole_tables()
intervals = cs.merge_intervals(tables["assay"], tables["geology"])
dh = cs.Drillholes(tables["collar"], tables["survey"], intervals)
grades = ["ZN", "PB", "CU", "AG", "AU"]
composites = dh.composite(2.0, grades, domain="LITH")
xyz = composites.coords
zn = composites["ZN"]
lith = np.array(composites.attributes["LITH"])
hole = np.array(composites.attributes["hole"])
print(f"{len(composites)} composites")

# %% [markdown]
# ## Declustered statistics
#
# Drilling concentrates where grades are high. Cell declustering (50 m cells) weights each composite by the inverse
# of the number of composites in its cell; `describe` gives weighted moments and quantiles.

# %%
print(f"{'LITH':<6}{'n':>7}{'mean':>8}{'declust.':>10}{'CV':>6}{'P50':>7}{'P90':>7}")
for name in ["MS", "SM", "QE", "EX", "RH"]:
    keep = (lith == name) & ~np.isnan(zn)
    w = cs.cell_declustering(xyz[keep], zn[keep], cell_size=50.0).weights
    naive, s = cs.describe(zn[keep]), cs.describe(zn[keep], w, quantiles=[0.5, 0.9])
    p50, p90 = s["quantiles"]
    print(f"{name:<6}{s['n']:>7}{naive['mean']:>8.2f}{s['mean']:>10.2f}{s['cv']:>6.2f}{p50:>7.2f}{p90:>7.2f}")

# %% [markdown]
# ## Top cuts
#
# `capping` reports, for caps at high quantiles, the share of composites cut and of metal removed. A cap that removes
# a few percent of the metal from a fraction of a percent of the samples tames the tail without flattening it.

# %%
ms = (lith == "MS") & ~np.isnan(zn)
caps = cs.capping(zn[ms])
print(f"{'cap':>7}{'cut (%)':>9}{'metal (%)':>11}{'mean':>7}{'CV':>6}")
for cap, frac, metal, mean, cv in zip(*caps.values(), strict=True):
    print(f"{cap:>7.2f}{100 * frac:>9.1f}{100 * metal:>11.2f}{mean:>7.2f}{cv:>6.2f}")

# %% [markdown]
# ## Contact analysis
#
# Zn against distance to the MS/RH contact, measured down each hole to the nearest composite of the other domain:
# negative inside MS, positive in RH. A sharp step supports a hard boundary in estimation.

# %%
c = cs.contact(xyz, zn, lith, hole, "MS", "RH", max_distance=30.0, bin=2.0)
fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
ax.axvline(0, color=GREY, lw=0.8, ls="--")
ax.plot(c["distance"], c["mean"], color=ACCENT, lw=1)
ax.scatter(c["distance"], c["mean"], s=np.sqrt(c["count"]), color=ACCENT)
ax.text(-15, 2, "inside MS", color=INK, ha="center")
ax.text(15, 5, "in RH", color=INK, ha="center")
ax.set(
    title="Zn across the MS/RH contact (points sized by count)",
    xlabel="Distance to contact (m)",
    ylabel="Zn (%)",
)
save(fig, "contact")

# %% [markdown]
# ## Swath
#
# Mean Zn in 100 m slices along easting, with the counts of the first swath as bars. The same call on block
# centroids and estimates gives the model swath to check for local bias.

# %%
swaths = [cs.swath(xyz[lith == name], zn[lith == name], 100.0, axis="x") for name in ["MS", "SM"]]
fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
cs.plot.swath(swaths, labels=["MS", "SM"], ax=ax)
ax.set(title="Zn swath along easting", xlabel="Easting (m)", ylabel="Zn (%)")
save(fig, "swath")

# %% [markdown]
# ## h-scatterplots
#
# Pairs of composites a lag apart: tail value against head value. Correlation drops as the lag grows, the mirror image
# of the variogram rising.

# %%
log_zn = np.log10(np.where(zn > 0, zn, np.nan))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained", sharey=True)
for ax, lag in zip(axes, [2.0, 10.0, 50.0], strict=True):
    head, tail, r = cs.h_scatter(xyz, log_zn, lag, 0.1 * lag)
    ax.hexbin(tail, head, gridsize=40, bins="log", cmap="ceres", linewidths=0)
    ax.set(title=f"h = {lag:g} m, ρ = {r:.2f}", xlabel="log₁₀ Zn at x", aspect="equal")
axes[0].set_ylabel("log₁₀ Zn at x + h")
save(fig, "h_scatter")

# %% [markdown]
# ## Correlation matrix
#
# Spearman correlation of the grades, each pair over the composites where both are assayed.

# %%
r = cs.correlation(np.column_stack([composites[g] for g in grades]), method="spearman")
fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
im = ax.imshow(r, cmap="ceres", vmin=0, vmax=1)
for i in range(len(grades)):
    for j in range(len(grades)):
        ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", color="white" if r[i, j] > 0.6 else INK)
ax.set_xticks(range(len(grades)), grades)
ax.set_yticks(range(len(grades)), grades)
ax.spines[:].set_visible(False)
ax.set_title("Spearman correlation")
fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, "correlation")
