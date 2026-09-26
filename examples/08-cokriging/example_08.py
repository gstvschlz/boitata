"""
# 8. Cokriging and indicator kriging

Jura: 259 soil samples of heavy metals (mg/kg, coordinates in km) and 100 validation samples withheld from estimation.
Cd correlates with Zn, and Zn is also known at the validation points, which suits collocated cokriging.
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
from common import ACCENT, GREY, HIGHLIGHT, INK, save

train = cs.datasets.jura()["prediction"]
test = cs.datasets.jura()["validation"]
grid = cs.datasets.jura()["grid"]
xy = train.coords
cd, zn = train["Cd"], train["Zn"]
rho = np.corrcoef(cd, zn)[0, 1]
lag, max_lag = 0.1, 1.5


# %% [markdown]
# The linear model of coregionalization is fitted to the Cd and Zn variograms and their cross-variogram, the half mean
# product of the Cd and Zn increments between co-located samples, all together. The variables share the structures, a
# nugget and two spherical ranges, and each structure's sill matrix stays positive semi-definite: its smallest
# eigenvalue is never negative. The short structure takes the place of the nugget, which fits to zero.

# %%
experimentals = [
    [
        cs.experimental_variogram(xy, cd, lag, max_lag),
        cs.experimental_variogram(xy, cd, lag, max_lag, other=zn),
    ],
    [None, cs.experimental_variogram(xy, zn, lag, max_lag)],
]
lmc = cs.Coregionalization.fit(experimentals, ["spherical", "spherical"])
print(f"corr(Cd, Zn) {rho:.2f}")
for name, matrix in [("nugget", lmc.nugget)] + [(f"{m} {a:.2f} km", s) for m, a, s in lmc.structures]:
    print(f"{name}: {np.round(matrix, 3).tolist()}, smallest eigenvalue {np.linalg.eigvalsh(matrix)[0]:.3g}")

cd_model = experimentals[0][0].fit("spherical")
search = cs.Search(radius=1.5, max_samples=24, min_samples=4)
ok = cs.OrdinaryKriging(cd_model, search).fit(xy, cd)
ck = cs.Cokriging(lmc, search, means=[cd.mean(), zn.mean()])
ck.fit(np.vstack([xy, xy]), np.r_[cd, zn], [0] * len(cd) + [1] * len(zn))


# %% [markdown]
# Each curve is the fitted LMC's C(0) − C(h), and the three experimental variograms follow the shared short and long
# structures in their own proportions:

# %%
h = np.linspace(0, max_lag, 101)[1:]
origin = np.zeros((h.size, 3))
away = np.c_[h, np.zeros((h.size, 2))]
panels = (
    (0, 0, "Cd", experimentals[0][0]),
    (1, 1, "Zn", experimentals[1][1]),
    (0, 1, "Cd × Zn", experimentals[0][1]),
)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
for ax, (i, j, name, experimental) in zip(axes, panels):
    cs.plot.variogram(experimental, ax=ax, color=ACCENT)
    model = lmc.cross_covariance(i, j, origin, origin) - lmc.cross_covariance(i, j, origin, away)
    ax.plot(h, model, color=INK, lw=1)
    ax.set(title=name, xlabel="Lag (km)", ylabel="γ(h)" if i == j else "γ₁₂(h)")
save(fig, "variograms")


# %% [markdown]
# Both estimators at the validation points:

# %%
truth = test["Cd"]
by_ok = ok.predict(test)
by_ck = ck.predict(test, collocated={1: test["Zn"]})


def rmse(e):
    return float(np.sqrt(np.mean((e - truth) ** 2)))


print(f"validation RMSE: ordinary kriging {rmse(by_ok):.3f}, collocated cokriging {rmse(by_ck):.3f} mg/kg")


# %% [markdown]
# Indicator kriging estimates the probability that Cd exceeds 0.8 mg/kg, the Swiss guide value, from the indicator
# variogram. IK estimates P(Cd ≤ threshold), so the exceedance is its complement.

# %%
limit = 0.8
indicator_model = cs.experimental_variogram(xy, (cd > limit).astype(float), lag, max_lag).fit("spherical")
ik = cs.IndicatorKriging(indicator_model, search, threshold=limit).fit(xy, cd)
p_exceed = 1 - ik.predict(grid)
p_test = 1 - ik.predict(test)
exceeds = truth > limit
print(
    f"validation: mean P(Cd > {limit}) {p_test[exceeds].mean():.2f} where true exceedance, {p_test[~exceeds].mean():.2f} elsewhere"
)


# %% [markdown]
# Using Zn at the target lowers the error and removes the smoothing that flattens ordinary kriging:

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained", sharey=True)
for ax, estimate, title in (
    (axes[0], by_ok, "Ordinary kriging of Cd"),
    (axes[1], by_ck, "Collocated cokriging with Zn"),
):
    ax.scatter(truth, estimate, s=12, color=ACCENT, alpha=0.7, linewidths=0)
    ax.plot([0, 5], [0, 5], color=GREY, ls="--", lw=1)
    ax.set(xlim=(0, 5), ylim=(0, 5), xlabel="True Cd at validation points (mg/kg)", title=title)
    ax.set_aspect("equal")
    ax.text(0.2, 4.6, f"RMSE {rmse(estimate):.2f} mg/kg", color=INK)
axes[0].set_ylabel("Estimated Cd (mg/kg)")
save(fig, "validation")


# %% [markdown]
# Most of the area exceeds 0.8 mg/kg, so the map separates clean zones rather than hot spots:

# %%
fig, ax = plt.subplots(figsize=(6.2, 5), layout="constrained")
image = ax.scatter(*grid.coords[:, :2].T, c=p_exceed, s=7, marker="s", vmin=0, vmax=1, linewidths=0)
ax.scatter(
    *test.coords[exceeds, :2].T,
    s=14,
    facecolors="none",
    edgecolors=HIGHLIGHT,
    linewidths=0.9,
    label=f"validation point with Cd > {limit}",
)
ax.scatter(*test.coords[~exceeds, :2].T, s=6, color=GREY, label="validation point below")
ax.set_aspect("equal")
ax.set(title=f"Indicator kriging: P(Cd > {limit} mg/kg)", xlabel="X (km)", ylabel="Y (km)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
fig.colorbar(image, ax=ax, shrink=0.8, label="probability")
save(fig, "probability")


# %% [markdown]
# Multiple indicator kriging repeats this at the deciles of Cd and assembles the conditional distribution at each
# target: kriged probabilities are corrected to rise from 0 to 1, and between thresholds the distribution follows the
# declustered data. One indicator variogram per threshold lets low and high values have their own continuity; a single
# variogram at the median solves one system per target instead.

# %%
weights = cs.cell_declustering(xy, cd).weights
deciles = np.quantile(cd, np.linspace(0.1, 0.9, 9))
indicator_models = [
    cs.experimental_variogram(xy, (cd <= t).astype(float), lag, max_lag).fit("spherical") for t in deciles
]
median_model = indicator_models[4]
summaries = {"cutoffs": [limit], "quantiles": [0.1, 0.5, 0.9]}
mik = cs.MultipleIndicatorKriging(indicator_models, search, deciles, tails=(0.0, cd.max()))
by_mik = mik.fit(xy, cd, weights=weights).predict(test, **summaries)
median = cs.MultipleIndicatorKriging(median_model, search, deciles, tails=(0.0, cd.max()))
by_median = median.fit(xy, cd, weights=weights).predict(test, **summaries)
for name, s in (("per-threshold variograms", by_mik), ("median indicator", by_median)):
    p = s.probability_above[0]
    print(
        f"{name}: E-type RMSE {rmse(s.mean):.3f} mg/kg, mean P(Cd > {limit}) {p[exceeds].mean():.2f} where true"
        f" exceedance, {p[~exceeds].mean():.2f} elsewhere, mean correction {s.correction.mean():.3f}"
    )
inside = np.mean((truth >= by_mik.quantile_values[0]) & (truth <= by_mik.quantile_values[2]))
print(f"validation points inside their 10-90% interval: {inside:.0%}")


# %% [markdown]
# The E-type estimate, the mean of each distribution, has a lower error than ordinary kriging, and the median-indicator
# shortcut gives most of that gain back. The distributions also carry the uncertainty: those at the lowest and highest
# validation estimates sit on either side of the declustered global one:

# %%
order = np.argsort(by_mik.mean)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
sorted_cd = np.sort(cd)
cumulative = np.cumsum(weights[np.argsort(cd)]) / weights.sum()
axes[0].step(sorted_cd, cumulative, where="post", color=GREY, lw=1, label="declustered global")
for i, color, label in ((order[0], ACCENT, "lowest E-type"), (order[-1], HIGHLIGHT, "highest E-type")):
    axes[0].plot(
        deciles, by_mik.cdf[:, i], "o-", color=color, ms=3, lw=1, label=f"{label}, true {truth[i]:.2f}"
    )
axes[0].axvline(limit, color=INK, lw=0.6, ls=":")
axes[0].set(xlabel="Cd (mg/kg)", ylabel="P(Cd ≤ z)", title="Conditional distributions", xlim=(0, 4))
axes[0].legend(fontsize=8, loc="lower right")
axes[1].scatter(truth, by_mik.mean, s=12, color=ACCENT, alpha=0.7, linewidths=0)
axes[1].plot([0, 5], [0, 5], color=GREY, ls="--", lw=1)
axes[1].set(
    xlim=(0, 5), ylim=(0, 5), xlabel="True Cd (mg/kg)", ylabel="E-type Cd (mg/kg)", title="E-type estimate"
)
axes[1].set_aspect("equal")
axes[1].text(0.2, 4.6, f"RMSE {rmse(by_mik.mean):.2f} mg/kg", color=INK)
save(fig, "distributions")
