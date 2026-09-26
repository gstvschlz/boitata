"""
# 16. Exploratory data analysis

Duplicates, paired data of two drilling types, statistics and distributions per domain, top cuts, grade-tonnage, contacts, swaths, h-scatterplots and
correlations on the 2 m composites of the drillhole dataset. Every function skips missing values, so raw columns go in as they are.
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
from common import ACCENT, GREY, HIGHLIGHT, INK, map_axes, save

# %% [markdown]
# The tables are checked and fixed with the default rules first, as in [chapter 6](../06-drillholes/README.md):
# overlapping assays keep the one that starts first and abruptly deviating survey stations are dropped.

# %%
tables = cs.datasets.drillhole_tables()
flags, _ = cs.check_drillholes(
    tables["collar"], tables["survey"], {"assay": tables["assay"], "geology": tables["geology"]}
)
tables, _ = cs.fix_drillholes(flags, tables)
intervals = cs.merge_intervals(tables["assay"], tables["geology"])
dh = cs.Drillholes(tables["collar"], tables["survey"], intervals)
grades = ["ZN", "PB", "CU", "AG", "AU"]
composites = dh.composite(2.0, grades, domain="LITH")
print(f"{len(composites)} composites")

# %% [markdown]
# ## Duplicates
#
# Twin holes, re-entries and collars entered twice put several composites at one place, and kriging cannot weight
# two samples at the same location. `duplicates` groups the composites closer than a tolerance, transitively, and
# reports each group with its first sample and its spread.

# %%
hole = np.array(composites.attributes["hole"])


def holes_per_group(tolerance):
    report, group = cs.duplicates(composites, tolerance)
    return report, [tuple(sorted(set(hole[group == k]))) for k in report["group"]]


report, groups = holes_per_group(0.1)
across = [h for h in groups if len(h) > 1]
print(f"{len(report)} groups within 0.1 m, at most {report['spread'].max():.2f} m from their first composite")
print(f"{len(across)} across {len(set(across))} sets of holes, {len(groups) - len(across)} inside one hole")
exact, groups = holes_per_group(0.0)
print(f"{len(exact)} groups at exactly the same location, from {len(set(groups))} pairs of holes")

# %% [markdown]
# The groups inside one hole are short intervals on both sides of a contact, in different lithologies: they stay.
# The exact duplicates are the top composites of pairs of holes collared at the same point. One pair puts an MS
# composite on one of unknown lithology (UNK), which a length-weighted `merge="mean"` would blend, so each pair keeps the composite
# of its first hole instead; the `n` column counts the composites behind each row.

# %%
composites = cs.duplicates(composites, merge="first")
print(f"{len(composites)} composites, {int((composites['n'] > 1).sum())} of them merged pairs")
xyz = composites.coords
zn = composites["ZN"]
lith = np.array(composites.attributes["LITH"])
hole = np.array(composites.attributes["hole"])

# %% [markdown]
# ## Paired data
#
# Most holes are diamond drill holes (DD); many others are sludge holes, and some percussion (PC). Are their grades
# alike? `pairs` finds, for each DD composite, the nearest composite of the other type within 10 m, each composite in
# at most one pair, the closest pairs first. `paired_bias` compares the paired means per bin of pairing distance: the
# bias read at short distances is the sampling bias, not a change of grade over space.

# %%
collar = tables["collar"]
kind = dict(zip(collar["HOLEID"], collar["TYPE"], strict=True))
drilling = np.array([kind[h] for h in hole])
dd = drilling == "DD"
edges = np.arange(0.0, 11.0, 2.0)
paired, bias = {}, {}
for name in ["SLUDGE", "PC"]:
    other = drilling == name
    paired[name] = cs.pairs(xyz[dd], xyz[other], 10.0, values=(zn[dd], zn[other]))
    bias[name] = cs.paired_bias(paired[name], edges)
    p = paired[name]
    print(f"DD-{name}: {len(p)} pairs, Zn {np.mean(p['value_a']):.2f} vs {np.mean(p['value_b']):.2f} %")

# %% [markdown]
# Sludge composites average about 5 % less Zn than their DD pairs, but the bins swing between -10 and +10 %: no
# consistent bias. Percussion composites read two to four times the DD grade in pairs closer than 4 m, and still
# more than 1.5 times at 10 m. A bias that is largest where the pairs are closest is a sampling problem, not a change
# of grade over space, and a reason to leave PC out of the estimate.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
for ax, name in zip(axes, ["SLUDGE", "PC"], strict=True):
    cs.plot.paired_bias(bias[name], ax=ax, color=ACCENT)
    ax.set(title=f"{name} against DD", xlabel="Pairing distance (m)", ylabel=f"Bias of {name} over DD (%)")
save(fig, "paired_bias")

# %% [markdown]
# The Q-Q plot and the scatter plot of the DD-PC pairs, and the pairs on a plan with the five largest differences
# listed and marked. The Q-Q plot sits above the 1:1 line at every quantile.

# %%
p = paired["PC"]
a, b = p["a"].astype(int), p["b"].astype(int)
pc_rows, dd_rows = np.flatnonzero(drilling == "PC")[b], np.flatnonzero(dd)[a]
worst = np.argsort(-np.abs(p["value_b"] - p["value_a"]))[:5]
print(f"{'DD hole':<9}{'PC hole':<9}{'distance':>9}{'DD Zn':>7}{'PC Zn':>7}")
for k in worst:
    print(
        f"{hole[dd_rows[k]]:<9}{hole[pc_rows[k]]:<9}{p['distance'][k]:>9.1f}"
        f"{p['value_a'][k]:>7.2f}{p['value_b'][k]:>7.2f}"
    )
fig, (q, s, m) = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
cs.plot.qq(p["value_a"], p["value_b"], ax=q, color=ACCENT)
q.set(title="Q-Q, DD-PC pairs", xlabel="DD Zn (%)", ylabel="PC Zn (%)")
cs.plot.scatter(p["value_a"], p["value_b"], ax=s, color=ACCENT)
s.set(title="Scatter, DD-PC pairs", xlabel="DD Zn (%)", ylabel="PC Zn (%)")
s.legend(loc="upper left")
m.scatter(*xyz[dd_rows, :2].T, s=4, color=GREY, label="paired DD composites")
m.scatter(
    *xyz[dd_rows[worst], :2].T, s=30, facecolors="none", edgecolors=HIGHLIGHT, label="5 largest differences"
)
map_axes(m, "DD-PC pairs")
m.legend(loc="lower left")
save(fig, "paired")

# %% [markdown]
# ## Declustered statistics
#
# Drilling concentrates where grades are high. Cell declustering (50 m cells) weights each composite by the inverse
# of the number of composites in its cell, domain by domain. `describe_by` gives weighted moments and quantiles per
# lithology and, on the last row, over all five together.

# %%
domains = ["MS", "SM", "QE", "EX", "RH"]
weights = np.zeros(len(zn))
for name in domains:
    keep = (lith == name) & ~np.isnan(zn)
    weights[keep] = cs.cell_declustering(xyz[keep], zn[keep], cell_size=50.0).weights
five = np.isin(lith, domains)
naive = cs.describe_by(zn[five], lith[five])
stats = cs.describe_by(zn[five], lith[five], weights[five], quantiles=[0.5, 0.9, 0.995])
print(f"{'LITH':<6}{'n':>7}{'mean':>8}{'declust.':>10}{'CV':>6}{'P50':>7}{'P90':>7}")
for name, n, raw, mean, cv, p50, p90 in zip(
    *(stats[c] for c in ["category", "n"]),
    naive["mean"],
    *(stats[c] for c in ["mean", "cv", "P50", "P90"]),
    strict=True,
):
    print(f"{name:<6}{n:>7.0f}{raw:>8.2f}{mean:>10.2f}{cv:>6.2f}{p50:>7.2f}{p90:>7.2f}")

# %% [markdown]
# The same declustered quantiles as box plots, sorted by median: the box spans P25 to P75, the whiskers P10 to P90,
# the dot is the declustered mean.

# %%
fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
cs.plot.boxplot(zn[five], lith[five], weights=weights[five], sort=True, log=True, ax=ax)
ax.set(title="Declustered Zn by lithology", ylabel="Zn (%)")
save(fig, "boxplot")

# %% [markdown]
# Cumulative distributions show that declustering shifts MS only slightly towards low grades; `stats=True` lists the
# count, weighted mean, CV and deciles of each curve in the corner. The Q-Q plot compares
# MS with SM quantile by quantile: above about 1 % Zn they follow the 1:1 line, below it MS is richer, so the two
# domains differ in their low tail rather than in their high grades.

# %%
ms, sm = lith == "MS", lith == "SM"
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
cs.plot.cdf(
    [zn[ms], zn[ms], zn[sm]],
    weights=[None, weights[ms], weights[sm]],
    labels=["MS naive", "MS declustered", "SM declustered"],
    log=True,
    stats=True,
    ax=a,
)
a.set(title="Cumulative distribution of Zn", xlabel="Zn (%)")
cs.plot.qq(zn[sm], zn[ms], weights[sm], weights[ms], log=True, ax=b)
b.set(title="Q-Q, declustered: P1 to P99", xlabel="SM Zn (%)", ylabel="MS Zn (%)")
save(fig, "distributions")

# %% [markdown]
# ## Top cuts
#
# `capping` reports, for caps at high quantiles, the share of composites cut and of metal removed. A cap that removes
# a few percent of the metal from a fraction of a percent of the samples tames the tail without flattening it.

# %%
ms = (lith == "MS") & ~np.isnan(zn)
caps = cs.capping(zn[ms], weights[ms])
print(f"{'cap':>7}{'cut (%)':>9}{'metal (%)':>11}{'mean':>7}{'CV':>6}")
for cap, frac, metal, mean, cv in zip(*caps.values(), strict=True):
    print(f"{cap:>7.2f}{100 * frac:>9.1f}{100 * metal:>11.2f}{mean:>7.2f}{cv:>6.2f}")

# %% [markdown]
# Zn is bounded by the zinc content of sphalerite: on a log-probability plot the upper tails bend towards a ceiling
# near 40 % instead of trailing off into isolated outliers. A cap at the declustered P99.5 of each domain, dashed,
# only trims the last half percent of that tail. The dotted Tukey fences, 1.5 interquartile ranges beyond the
# quartiles of log Zn, agree: no upper fence falls inside the data. The only fence drawn is a lower one in MS: its
# outliers are the few composites below 0.05 % Zn, a low tail that a cap does not touch.

# %%
cap = dict(zip(stats["category"], stats["P99.5"], strict=True))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", sharey=True)
for ax, name in zip(axes, ["MS", "RH"], strict=True):
    keep = (lith == name) & (zn > 0)
    cs.plot.probability(
        zn[keep], weights[keep], log=True, cap=cap[name], fences=1.5, ax=ax, color=ACCENT, ms=2
    )
    ax.set(title=f"{name}, declustered", xlabel="Zn (%)")
    ax.legend(loc="lower right")
axes[1].set_ylabel("")
save(fig, "probability")

# %% [markdown]
# `capping_report` applies one cap per domain and compares the declustered statistics before and after, with the
# metal removed; the last row pools the domains. Only RH, the low-grade host rock, loses more than a fraction of a
# percent of its metal.

# %%
report = cs.capping_report(zn[five], lith[five], {k: cap[k] for k in domains}, weights[five])
columns = ["domain", "cap", "n_capped", "mean", "mean_capped", "cv", "cv_capped"]
print(f"{'LITH':<6}{'cap':>7}{'cut':>5}{'mean':>7}{'capped':>8}{'CV':>6}{'capped':>8}{'metal (%)':>11}")
for name, c, n, mean, capped, cv, cv_capped in zip(*(report[k] for k in columns), strict=True):
    print(
        f"{name:<6}{'' if np.isnan(c) else f'{c:.2f}':>7}{n:>5.0f}{mean:>7.2f}{capped:>8.2f}{cv:>6.2f}"
        f"{cv_capped:>8.2f}{100 * (1 - capped / mean):>11.2f}"
    )

# %% [markdown]
# ## Grade-tonnage of the data
#
# `grade_tonnage` sums the weight of the composites at or above each cutoff and their mean grade: a first look at
# selectivity on composite support, here as a proportion of the total. Declustering moves the MS curves only a
# little, as it did the mean: slightly less material above low cutoffs, slightly richer above most of them. Blocks are
# less selective than composites; [chapter 9](../09-change-of-support/README.md) models that change of support.

# %%
cutoffs = np.linspace(0, 30, 61)
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
for w, color, label in ((None, GREY, "naive"), (weights[ms], ACCENT, "declustered")):
    gt = cs.grade_tonnage(zn[ms], cutoffs, w)
    a.plot(cutoffs, gt["tonnage"] / gt["tonnage"][0], color=color, label=label)
    b.plot(cutoffs, gt["mean_grade"], color=color)
a.set(title="MS proportion above cutoff", xlabel="Cutoff Zn (%)", ylabel="Proportion of weight")
a.legend()
b.set(title="MS mean grade above cutoff", xlabel="Cutoff Zn (%)", ylabel="Mean Zn above cutoff (%)")
save(fig, "grade_tonnage")

# %% [markdown]
# ## Contact analysis
#
# Zn against distance to a contact of MS, measured down each hole to the nearest composite of the other domain:
# negative inside MS, positive outside. Into the RH host rock, Zn drops from about 9 % to under 2 % within a composite:
# a sharp step that supports a hard boundary in estimation. Into the semi-massive sulphide SM it steps down by only
# about 2 %, and SM keeps 6 to 10 % out to 30 m: near the contact the samples of one domain say much about the other,
# the case for a soft boundary ([chapter 20](../20-workflow/README.md)).

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained", sharey=True)
for ax, other in zip(axes, ["RH", "SM"], strict=True):
    c = cs.contact(xyz, zn, lith, hole, "MS", other, max_distance=30.0, bin=2.0)
    ax.axvline(0, color=GREY, lw=0.8, ls="--")
    ax.plot(c["distance"], c["mean"], color=ACCENT, lw=1)
    ax.scatter(c["distance"], c["mean"], s=np.sqrt(c["count"]), color=ACCENT)
    ax.text(-15, 13, "inside MS", color=INK, ha="center")
    ax.text(15, 13, f"in {other}", color=INK, ha="center")
    ax.set(title=f"Zn across the MS/{other} contact", xlabel="Distance to contact (m)", ylim=(0, 14))
axes[0].set_ylabel("Zn (%), points sized by count")
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
    ax.hexbin(tail, head, gridsize=40, bins="log", linewidths=0)
    ax.set(title=f"h = {lag:g} m, ρ = {r:.2f}", xlabel="log₁₀ Zn at x", aspect="equal")
axes[0].set_ylabel("log₁₀ Zn at x + h")
save(fig, "h_scatter")

# %% [markdown]
# ## Correlations
#
# The scatter-plot matrix of the MS grades on log axes, with declustered histograms on the diagonal and, in each
# panel, the declustered Pearson (r) and rank correlation of the pair. Pearson's r, on the raw grades, falls well
# below the rank correlation wherever a few high values dominate a pair; on skewed grades the rank correlation is the
# one to read. Zn, Pb and Ag move together most closely. The rows of points at Ag 1 g/t and Au 0.01 g/t are
# detection limits.

# %%
ms = lith == "MS"
fig, axes = cs.plot.scatter_matrix({g: composites[g][ms] for g in grades}, weights=weights[ms], log=True)
fig.suptitle("MS grades, declustered", x=0.02, ha="left", fontweight="bold", fontsize=10)
save(fig, "scatter_matrix")

# %% [markdown]
# Not every composite is assayed for every grade, and each correlation only uses the composites where both grades
# are. `plot.completeness` counts the composites by the number of grades present, the complete ones in colour: Au is
# assayed in only half of them, so the correlations with Au rest on half the data.

# %%
assays = {g: composites[g] for g in grades}
print("  ".join(f"{g} {np.mean(~np.isnan(v)):.0%}" for g, v in assays.items()), "assayed")
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", width_ratios=[1, 1.2])
cs.plot.completeness(assays, ax=a)
a.set_title("Composites by grades assayed")
cs.plot.correlation(assays, method="spearman", ax=b)
b.set_title("Spearman correlation, all composites")
save(fig, "correlation")

# %% [markdown]
# A scatter plot hides how many points sit on top of one another. The mean of Pb and its P10 to P90 in ten bins of
# Zn, each holding a tenth of the MS composites, show the relation itself: Pb rises steadily with Zn, and its spread
# narrows from two orders of magnitude in the lowest bins to less than one in the richest.

# %%
fig, ax = plt.subplots(figsize=(5, 3.6), layout="constrained")
cs.plot.conditional(composites["ZN"][ms], composites["PB"][ms], weights=weights[ms], log=True, ax=ax)
ax.set(title="Pb given Zn, MS, declustered", xlabel="Zn (%)", ylabel="Pb (%)")
ax.legend(loc="lower right")
save(fig, "conditional")
