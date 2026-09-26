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
# The coregionalization uses the intrinsic model: Cd and Zn share Cd's spherical structure and nugget share,
# scaled by their covariance matrix.

# %%
cd_model = cs.experimental_variogram(xy, cd, lag, max_lag).fit("spherical")
nugget_share = cd_model.nugget / cd_model.sill
a = cd_model.structures[0].range
print(f"Cd: nugget share {nugget_share:.2f}, range {a:.2f} km, corr(Cd, Zn) {rho:.2f}")

cov = np.cov(cd, zn)
lmc = cs.Coregionalization(
    (nugget_share * cov).tolist(),
    [("spherical", a, ((1 - nugget_share) * cov).tolist())],
)
search = cs.Search(radius=1.5, max_samples=24, min_samples=4)
ok = cs.OrdinaryKriging(cd_model, search).fit(xy, cd)
ck = cs.Cokriging(lmc, search, means=[cd.mean(), zn.mean()])
ck.fit(np.vstack([xy, xy]), np.r_[cd, zn], [0] * len(cd) + [1] * len(zn))


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
