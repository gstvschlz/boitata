"""
# 69. Capping transform

`Capping` makes the top cut a pipeline step: `fit` chooses a cap per domain from the data, `transform` clips values
to it, and the fitted caps travel with the object to new data, to JSON and to pickle. The cap is either given or
chosen by a rule: a weighted quantile, a target fraction of metal removed, or a target coefficient of variation.
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
from common import ACCENT, GRAY, HIGHLIGHT, save

# %% [markdown]
# Four steep gold veins, drilled by diamond holes and sampled by channels in the drives. Assays are composited to
# 1 m inside each vein, and cell declustering evens out the channels, which crowd the levels.

# %%
data = cs.datasets.vein_gold_grade_control()
intervals = cs.merge_intervals(data["assays"], data["lithology"])
drillholes = cs.Drillholes(data["collars"], data["surveys"], intervals)
composites = drillholes.composite(1.0, ["AU_GPT"], domain="VEIN")
composites = composites.filter(np.asarray(composites["VEIN"]) != "")
composites = composites.filter(~np.isnan(composites["AU_GPT"]))
weights = cs.cell_declustering(composites, "AU_GPT", cell_size=20.0).weights
composites = composites.with_column("w", weights)
veins = sorted(set(composites["VEIN"]))
print(f"{len(composites)} composites in {len(veins)} veins")

# %% [markdown]
# ## One cap per vein
#
# Each rule is fitted per vein on the declustered composites. The quantile rule caps at the declustered P99; the
# metal rule finds the cap that removes 5 % of each vein's metal; the CV rule the largest cap whose capped
# coefficient of variation is 1.5. `caps_` and `metal_removed_` are dicts by vein.

# %%
rules = {
    "P99": cs.Capping(quantile=0.99),
    "5 % metal": cs.Capping(metal_removed=0.05),
    "CV 1.5": cs.Capping(cv=1.5),
}
for rule in rules.values():
    rule.fit("AU_GPT", domain_column="VEIN", weights="w", data=composites)
print(f"{'vein':<6}" + "".join(f"{name:>11}{'metal (%)':>11}" for name in rules))
for vein in veins:
    row = "".join(f"{r.caps_[vein]:>11.1f}{100 * r.metal_removed_[vein]:>11.2f}" for r in rules.values())
    print(f"{vein:<6}{row}")

# %% [markdown]
# The rules disagree most where the tail is longest. V1 and V2, sampled by thousands of channels with a lognormal
# tail up to 500 g/t, lose 12 to 15 % of their metal at the P99; holding the loss to 5 % puts their caps near
# 95 g/t. V4's P99 cuts a single composite and removes almost nothing. A CV target caps harder than either:
# it pulls every cap down to 11 to 13 g/t, and V4, whose grades are spread rather than skewed, loses half its metal.
#
# ## The same numbers as the report
#
# `capping_report`, used in [topic 13](../13-correlations/README.md) to compare statistics before and after a
# given cap, reads the fitted caps directly. Its fraction of metal removed, `1 - mean_capped / mean`, is the
# `metal_removed_` of the transform.

# %%
capping = rules["P99"]
report = cs.capping_report("AU_GPT", capping.caps_, domain_column="VEIN", weights="w", data=composites)
columns = ["domain", "cap", "n_capped", "mean", "mean_capped", "cv", "cv_capped"]
print(f"{'vein':<6}{'cap':>7}{'cut':>5}{'mean':>7}{'capped':>8}{'CV':>6}{'capped':>8}{'metal (%)':>11}")
for name, c, n, mean, capped, cv, cv_capped in zip(*(report[k] for k in columns), strict=True):
    print(
        f"{name:<6}{'' if np.isnan(c) else f'{c:.1f}':>7}{n:>5.0f}{mean:>7.2f}{capped:>8.2f}{cv:>6.2f}"
        f"{cv_capped:>8.2f}{100 * (1 - capped / mean):>11.2f}"
    )

# %% [markdown]
# On a log-probability plot the dashed caps cut the last percent of each vein's tail.

# %%
fig, axes = plt.subplots(1, len(veins), figsize=(11, 3.4), layout="constrained", sharey=True)
for ax, vein in zip(axes, veins, strict=True):
    keep = np.asarray(composites["VEIN"]) == vein
    cs.plot.probability(
        composites["AU_GPT"][keep],
        weights=weights[keep],
        log=True,
        cap=capping.caps_[vein],
        ax=ax,
        color=ACCENT,
        ms=2,
    )
    ax.set(title=vein, xlabel="Au (g/t)")
    ax.legend(loc="lower right")
for ax in axes[1:]:
    ax.set_ylabel("")
save(fig, "probability")

# %% [markdown]
# ## Capped kriging
#
# The capped grades feed the estimate as any other column. Blocks of 5 m inside the V1 solid are kriged from the
# raw and from the capped composites of V1 with one variogram and search. The extreme channels no longer spread
# their grade over their neighborhood: the blocks lose about as much metal as the declustered composites, 11 %,
# almost all of it from blocks above 3 g/t, while the low-grade blocks stay on the diagonal.

# %%
v1 = composites.filter(np.asarray(composites["VEIN"]) == "V1")
v1 = v1.with_column("AU_CAPPED", capping.transform("AU_GPT", domain_column="VEIN", data=v1))
blocks = cs.BlockModel.from_extents(data["vein_V1"], size=(5.0, 5.0, 5.0))
targets = blocks.centroids[data["vein_V1"].contains(blocks.centroids)]
model = cs.experimental_variogram(v1, "AU_CAPPED", 10.0, 150.0).fit("spherical")
search = cs.Search(80.0, max_samples=24)
kriged = {
    column: cs.OrdinaryKriging(model, search).fit(v1, column).predict(targets)
    for column in ("AU_GPT", "AU_CAPPED")
}
for column, grades in kriged.items():
    print(f"{column}: {np.isfinite(grades).sum()} blocks, mean {np.nanmean(grades):.2f} g/t")
print(
    f"composites: {v1['AU_GPT'] @ v1['w'] / v1['w'].sum():.2f} and {v1['AU_CAPPED'] @ v1['w'] / v1['w'].sum():.2f} g/t"
)

# %%
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.scatter(kriged["AU_GPT"], kriged["AU_CAPPED"], s=2, color=ACCENT)
ax.axline((0, 0), slope=1, color=GRAY, lw=0.8)
ax.axhline(capping.caps_["V1"], color=HIGHLIGHT, ls="--", lw=0.8, label="cap")
ax.set(xscale="log", yscale="log", xlabel="Kriged from raw Au (g/t)", ylabel="Kriged from capped Au (g/t)")
ax.set_title("V1 blocks")
ax.legend(loc="lower right")
save(fig, "kriged")

# %% [markdown]
# ## In a pipeline
#
# `Capping` has `fit`, `transform` and `fit_transform` like `NormalScore`, so the two chain: cap, then score, the
# usual preparation for a Gaussian simulation. Fitted once, both apply to new samples: 50 and 500 g/t are both
# capped, so they get the same score. The fitted caps round-trip through JSON and pickle.

# %%
capped = capping.fit_transform("AU_GPT", domain_column="VEIN", weights="w", data=v1)
scores = cs.NormalScore().fit(capped, weights=v1["w"])
new = np.array([0.5, 5.0, 50.0, 500.0])
print(scores.transform(capping.transform(new, domains=["V1"] * 4)).round(3))
restored = cs.Capping.from_json(capping.to_json())
print(restored.caps_ == capping.caps_)
