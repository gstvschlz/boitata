"""
# 64. Categorical indicator kriging

Logged rock types are categories, not grades. Categorical indicator kriging estimates, at each target, the
probability of every category: each category's indicator (1 inside it, 0 outside) is kriged with its own
variogram, and the probabilities are then clipped to [0, 1] and rescaled to sum 1. The most likely category is a
rock-type model, and the entropy of the probabilities says where that model is uncertain.
"""

# %% [hidden]
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))
warnings.filterwarnings("ignore", ".*locations hold several samples")

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import save

# %% [markdown]
# ## Rock types
#
# The stacked sulphide lenses are drilled towards the west-northwest across a stratigraphy that strikes 023° and
# dips about 55° to the east-southeast: footwall volcanics (`FWV`), volcaniclastics (`VCL`) and hanging-wall
# sediments (`HWS`) under overburden (`OB`). The three sulphide codes are lumped into `SUL`, and dykes are kept apart.
# A `Categories` scheme holds the names, the lumping and the colors; it encodes the logged codes for the estimator
# and colors every plot below. Logged intervals are composited to 5 m, each taking the rock covering most of it.

# %%
data = cs.datasets.stacked_sulphide_lenses()
drillholes = cs.Drillholes(data["collars"], data["surveys"], data["lithology"])
composites = drillholes.composite(5.0, [], categories=["LITH"])
scheme = cs.Categories(
    ["OB", "HWS", "VCL", "SUL", "FWV"],
    mapping={"MS": "SUL", "SMS": "SUL", "STR": "SUL"},
    other="DYK",
    colors=["#d9d9d9", "#a9bfd3", "#8c8c8c", "#c05a28", "#1f4e79", "#e0c080"],
)
codes = scheme.encode(composites["LITH"])
weights = cs.cell_declustering(composites, codes, cell_size=50.0).weights
naive, declustered = scheme.shares(codes), scheme.shares(codes, weights=weights)
print(f"{len(codes)} composites")
print(f"{'':<5}{'naive':>7}{'declustered':>13}")
for name, a, b in zip(scheme.names, naive, declustered):
    print(f"{name:<5}{a:7.3f}{b:13.3f}")

# %% [markdown]
# ## Variograms and the estimate
#
# Each category gets an indicator variogram, here chosen from the geology rather than fitted (topic 19 fits them):
# the layers are continuous along strike and down dip and short across, the sulphides form smaller lenses, the
# overburden is a flat blanket and the dykes are short in every direction. Only the shape of each variogram
# matters, since each indicator is kriged on its own. The search follows the layering in two passes. Ordinary
# kriging takes each indicator's mean from the samples found, so away from the holes the probabilities follow the
# nearest layers; `simple=True` would pull them towards the declustered proportions instead, which the weights set.

# %%
layers = (23.0, 55.0, 0.0)
variograms = [
    cs.Variogram([("spherical", 0.9, 400.0)], nugget=0.1, rotation=(0.0, 0.0, 0.0), ratios=(1.0, 0.05)),
    cs.Variogram([("spherical", 0.9, 500.0)], nugget=0.1, rotation=layers, ratios=(0.8, 0.2)),
    cs.Variogram([("spherical", 0.85, 400.0)], nugget=0.15, rotation=layers, ratios=(0.8, 0.2)),
    cs.Variogram([("spherical", 0.8, 150.0)], nugget=0.2, rotation=layers, ratios=(0.8, 0.25)),
    cs.Variogram([("spherical", 0.9, 500.0)], nugget=0.1, rotation=layers, ratios=(0.8, 0.2)),
    cs.Variogram([("spherical", 0.7, 40.0)], nugget=0.3),
]
passes = [
    cs.Search(150.0, max_samples=24, max_per_hole=6, rotation=layers, ratios=(1.0, 0.4)),
    cs.Search(300.0, max_samples=24, rotation=layers, ratios=(1.0, 0.4)),
]
cik = cs.CategoricalIndicatorKriging(variograms, passes, scheme=scheme)
cik.fit(composites, "LITH", weights=weights, holes="hole")

# %% [markdown]
# The targets are the cells of a vertical section across strike, one cell thick, through the middle of the drilling;
# cells above the topography are dropped. `predict` returns a summary: `probabilities` has one row per target and
# one column per category, `most_likely` holds category codes (NaN where the search found too few samples), and
# `diagnostics` is a Table of the search and of the correction.

# %%
center = np.array([12350.0, 29900.0, 0.0])
across = np.array([np.sin(np.radians(113.0)), np.cos(np.radians(113.0)), 0.0])
along = np.array([np.sin(np.radians(23.0)), np.cos(np.radians(23.0)), 0.0])
origin = center - 600.0 * across - 5.0 * along + [0.0, 0.0, -400.0]
section = cs.BlockModel(origin, (10.0, 10.0, 10.0), (120, 1, 80), rotation=(23.0, 0.0, 0.0))
topography = data["topography"]
rows = topography.row_at(section.centroids[:, :2])
section = section.mask((rows >= 0) & (section.centroids[:, 2] < topography["Z"][rows]))

summary = cik.predict(section, diagnostics=True)
p = summary.probabilities
done = ~np.isnan(summary.most_likely)
d = summary.diagnostics
print(f"{done.sum()} of {len(p)} cells estimated, {np.sum(d['pass'] == 2)} in the second pass")
print(f"rows sum to 1: {np.allclose(p[done].sum(axis=1), 1.0)}")
print(
    f"correction: {np.mean(d['n_order_violations'][done] > 0):.0%} of the cells had a category kriged outside "
    f"[0, 1]; mean size {np.mean(summary.correction[done]):.3f}"
)
columns = {f"p_{name}": p[:, c] for c, name in enumerate(scheme.names)}
section = section.with_columns({**columns, "most_likely": summary.most_likely, "entropy": summary.entropy})

# %% [markdown]
# ## Probability maps
#
# Each category's probability is high where its samples are and fades across its contacts; the sulphides, rare
# and short-ranged, stand out only near the holes that cut them.

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout="constrained", sharex=True, sharey=True)
for ax, name in zip(axes.flat, ["HWS", "VCL", "SUL", "FWV"]):
    cs.plot.section(section, f"p_{name}", axis="y", index=0, vmin=0.0, vmax=1.0, colorbar=False, ax=ax)
    ax.set_title(f"P({name})")
    ax.set_xlabel("Along the section (m)" if ax in axes[1] else "")
    ax.set_ylabel("Elevation (m)")
fig.colorbar(ax.images[0], ax=axes, shrink=0.6, label="probability")
save(fig, "probabilities")

# %% [markdown]
# ## Most likely rock and its uncertainty
#
# The most likely category is drawn in the scheme's colors. Entropy, scaled to [0, 1], is 0 where one category is
# certain and 1 where all are equally likely: it is highest along the contacts and far from the holes.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", sharey=True)
cs.plot.section(section, "most_likely", axis="y", index=0, scheme=scheme, colorbar=False, ax=axes[0])
axes[0].set_title("Most likely rock")
cs.plot.category_legend(scheme, axes[0], loc="lower left")
cs.plot.section(section, "entropy", axis="y", index=0, vmin=0.0, vmax=1.0, cmap="Greys", ax=axes[1])
axes[1].set_title("Entropy")
for ax, y in zip(axes, ["Elevation (m)", ""]):
    ax.set_xlabel("Along the section (m)")
    ax.set_ylabel(y)
save(fig, "most-likely")

# %% [markdown]
# ## Checks
#
# Splitting the holes into 5 folds and re-estimating each fold's composites from the others scores the probabilities
# with the Brier score, the mean squared difference between a category's probability and its indicator: 0 is
# perfect, and always forecasting the declustered proportion `p` scores `p (1 − p)`. The layers beat that baseline
# by far, the thin sulphides barely do, and the dykes, cutting across everything, do not.

# %%
cv = cik.cross_validate(folds=5)
print(
    f"the most likely rock is the logged one at {np.mean(cv.most_likely == cv.actual):.0%} of the composites"
)
print(f"{'':<5}{'Brier':>7}{'p(1-p)':>8}")
for c, name in enumerate(scheme.names):
    print(f"{name:<5}{cv.brier[c]:7.3f}{declustered[c] * (1 - declustered[c]):8.3f}")
