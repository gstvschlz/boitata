"""
# 13. Simulation methods

Sequential Gaussian simulation (SGS) and turning bands simulate a continuous variable; sequential indicator
simulation (SIS) and plurigaussian simulation (PGS) simulate categories. Several correlated grades are simulated
through independent factors in [chapter 17](../17-multivariate/README.md).
"""

# %% [hidden]
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import map_axes, save
from matplotlib.colors import ListedColormap, PowerNorm

samples = cs.datasets.walker_lake()
xy, v = samples.coords, samples["V"]
weights = cs.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights
gaussian = cs.Variogram([("spherical", 0.68, 82.0)], nugget=0.32, rotation=(170, 0, 0), ratios=(0.43, 1.0))
grid = cs.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))

# %% [markdown]
# Both use the normal-score variogram of [chapter 5](../05-simulation/README.md). SGS visits nodes along a random
# path, kriging each from data and nodes already simulated. Turning bands sums many one-dimensional processes along
# random lines into an unconditional Gaussian field, then conditions it to the data by kriging the residuals.

# %%
sgs = cs.SGS(gaussian, cs.Search(radius=100, max_samples=24)).fit(xy, v, weights=weights)
tb = cs.TurningBands(gaussian, bands=500).fit(xy, v, weights=weights)
start = time.perf_counter()
by_sgs = sgs.simulate(grid, n=20, seed=5, realizations=True).realizations
sgs_seconds = time.perf_counter() - start
start = time.perf_counter()
by_tb = tb.simulate(grid, n=20, seed=5, realizations=True).realizations
tb_seconds = time.perf_counter() - start
for name, reals, seconds in (("SGS", by_sgs, sgs_seconds), ("turning bands", by_tb, tb_seconds)):
    print(
        f"{name:>13}: 20 realizations in {seconds:.2f} s, mean {reals.mean():.0f} ppm, variance {reals.var():.0f} ppm²"
    )

# %%
shape = (60, 52)
extent = (0.5, 260.5, 0.5, 300.5)
norm = PowerNorm(0.5, vmin=0, vmax=1500)
fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), layout="constrained")
panels = [
    (by_sgs[0], "SGS, realization 1"),
    (by_sgs[1], "SGS, realization 2"),
    (by_tb[0], "Turning bands, realization 1"),
    (by_tb[1], "Turning bands, realization 2"),
]
for ax, (image, title) in zip(axes, panels):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, norm=norm)
    map_axes(ax, title)
fig.colorbar(im, ax=axes, shrink=0.8, label="V (ppm)")
save(fig, "continuous")

# %% [markdown]
# ## With a trend
#
# Far from the data, SGS draws from the global histogram, wherever it is. A trend known everywhere can steer it
# instead. Here the trend is a moving-window average of V within 40 m, built with an existing estimator; any model
# or estimate would do. `trend=` gives it at the data to `fit` and at the nodes to `simulate`, as an array or the
# name of a BlockModel column. The data are normal-scored within 8 equal-probability classes of the trend, the
# stepwise conditional transform of [chapter 17](../17-multivariate/README.md) on (trend, V), which leaves scores
# independent of the trend. Those scores are simulated with their own variogram, and every node is back-transformed
# with the histogram of its trend class, before any averaging to `blocks`.

# %%
window = cs.MovingAverage(cs.Search(radius=40, max_samples=200)).fit(xy, v)
trend = window.predict(xy)
trended = grid.with_column("trend", window.predict(grid))
pair = np.column_stack([trend, v])
scores = cs.StepwiseConditional(classes=8).fit(pair, weights=weights).transform(pair)[:, 1]
score_variogram = cs.experimental_variogram(xy, scores, 10.0, 150.0).fit("spherical")
with_trend = cs.SGS(score_variogram, cs.Search(radius=100, max_samples=24), classes=8)
with_trend.fit(xy, v, weights=weights, trend=trend)
by_trend = with_trend.simulate(trended, n=20, seed=5, realizations=True, trend="trend").realizations

node_trend = trended["trend"]
cov = np.cov(v, trend, aweights=weights)
print(f"correlation with the trend: data {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.2f}", end="")
for name, reals in (("SGS", by_sgs), ("SGS with trend", by_trend)):
    print(f", {name} {np.mean([np.corrcoef(r, node_trend)[0, 1] for r in reals]):.2f}", end="")
edges = np.quantile(node_trend, [0.25, 0.5, 0.75])
at_data, at_nodes = np.digitize(trend, edges), np.digitize(node_trend, edges)
print(f"\n{'mean V (ppm) by trend quartile':<32}{'data':>6}{'SGS':>6}{'SGS with trend':>16}")
for k in range(4):
    data_mean = np.average(v[at_data == k], weights=weights[at_data == k])
    plain_mean, trend_mean = (reals[:, at_nodes == k].mean() for reals in (by_sgs, by_trend))
    print(f"{f'quartile {k + 1}':<32}{data_mean:6.0f}{plain_mean:6.0f}{trend_mean:16.0f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), layout="constrained")
panels = [
    (node_trend, "Moving-window trend, 40 m"),
    (by_sgs[0], "SGS, realization 1"),
    (by_trend[0], "SGS with the trend, realization 1"),
]
for ax, (image, title) in zip(axes, panels):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, norm=norm)
    map_axes(ax, title)
fig.colorbar(im, ax=axes, shrink=0.8, label="V (ppm)")
save(fig, "trend")

# %% [markdown]
# Plain SGS already follows the trend where data are dense, but pulls the low-trend quarter up towards the global
# mean; with the trend, each quarter keeps the declustered mean of its data, and the realizations correlate with the
# trend about as much as the data do.

# %% [markdown]
# ## Categories
#
# Jura's rock types are known everywhere on the prediction grid, so simulated categories can be compared with the
# real geology. SIS krigs, at each node, the probability of every rock type from indicator variograms; PGS truncates
# a Gaussian field at thresholds set by the proportions, which orders the types.

# %%
train = cs.datasets.jura()["prediction"]
jura_grid = cs.datasets.jura()["grid"]
names = ["Argovian", "Kimmeridgian", "Sequanian", "Portlandian", "Quaternary"]
code = {name: i for i, name in enumerate(names)}
rock = np.array([code[r] for r in train["Rock"]])
true_rock = np.array([code[r] for r in jura_grid["Rock"]])
proportions = np.bincount(rock, minlength=5) / len(rock)

indicator_models = []
for k in range(5):
    indicator = (rock == k).astype(float)
    if indicator.sum() >= 5:
        fitted = cs.experimental_variogram(train.coords, indicator, 0.1, 1.5).fit("spherical")
    else:
        fitted = cs.Variogram([("spherical", indicator.var(), 0.5)])
    indicator_models.append(fitted)

sis = cs.SIS(indicator_models, cs.Search(radius=1.5, max_samples=16)).fit(train.coords, rock)
sis_summary = sis.simulate(jura_grid, n=10, seed=3, realizations=True)
by_sis = sis_summary.realizations
pgs = cs.Plurigaussian(cs.Variogram([("spherical", 1.0, 0.8)]), proportions=proportions).fit(
    train.coords, rock
)
by_pgs = pgs.simulate(jura_grid, n=1, seed=3, realizations=True).realizations[0]

print(f"{'':>13}" + "".join(f"{n[:5]:>8}" for n in names))
for label, cats in (("samples", rock), ("true grid", true_rock), ("SIS", by_sis[0]), ("PGS", by_pgs)):
    shares = np.bincount(cats, minlength=5) / len(cats)
    print(f"{label:>13}" + "".join(f"{s:8.2f}" for s in shares))
for label, cats in (("SIS", by_sis[0]), ("PGS", by_pgs)):
    print(f"{label}: {np.mean(cats == true_rock):.0%} of nodes match the true rock type")
matches = np.mean(sis_summary.most_likely == true_rock)
print(
    f"SIS most likely type over 10 realizations: {matches:.0%} match, mean entropy {sis_summary.entropy.mean():.2f}"
)

# %%
colors = ListedColormap(["#1f4e79", "#6f9fc9", "#c9d9ea", "#c05a28", "#8c8c8c"])
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
for ax, (cats, title) in zip(
    axes, [(true_rock, "True rock types"), (by_sis[0], "SIS realization"), (by_pgs, "PGS realization")]
):
    ax.scatter(
        *jura_grid.coords[:, :2].T, c=cats, cmap=colors, vmin=-0.5, vmax=4.5, s=7, marker="s", linewidths=0
    )
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="X (km)", ylabel="Y (km)")
handles = [plt.Line2D([], [], marker="s", ls="", color=colors(i), label=n) for i, n in enumerate(names)]
fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
save(fig, "categories")

# %% [markdown]
# SIS gives each type its own indicator variogram and matches the true type at half the nodes. PGS reproduces the
# proportions closely, but its ordered rule only allows contacts between neighbours in the order, so Portlandian appears
# as specks along every Sequanian-Quaternary contact: suited to sequences like stratigraphy, not these rocks. Neither
# recovers Portlandian's 5 % of the area from 3 of 259 samples.
