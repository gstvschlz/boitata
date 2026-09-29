"""
# Madogram

The madogram is half the mean absolute difference between values a lag apart, M(h) = Σ |z(x) − z(x + h)| / 2N,
where the variogram averages squared differences. It is in the units of the data and a single extreme pair moves it
far less, so its shape stays readable on skewed grades whose variogram the outliers make erratic. For Gaussian
increments M(h) = √(γ(h)/π), and `dissemination` measures how far the data are from that: √π · M(h) / √γ(h) is 1 for
Gaussian increments and drops below 1 when a few large differences carry γ, the signature of high grades scattered in
isolated samples.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[2]))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, INK, save

# %% [markdown]
# The 1 m quartz-vein composites of [top cuts](../../03-exploratory-analysis/05-top-cuts/README.md), gold in g/t, in vein V1:

# %%
data = cs.datasets.vein_gold_grade_control()
intervals = cs.merge_intervals(data["assays"], data["lithology"])
holes = cs.Drillholes(data["collars"], data["surveys"], intervals)
composites = holes.composite(1.0, ["AU_GPT"], domain="LITH", categories=["VEIN"])
vein = composites.filter(
    (composites["LITH"] == "QV") & (composites["VEIN"] == "V1") & ~np.isnan(composites["AU_GPT"])
)
au = vein["AU_GPT"]
print(
    f"{len(au)} composites, mean {au.mean():.2f} g/t, CV {au.std() / au.mean():.2f}, max {au.max():.0f} g/t"
)

# %% [markdown]
# ## Three estimators
#
# The classical variogram, the madogram and, for comparison, the robust `"cressie-hawkins"` estimator, all
# omnidirectional with 5 m lags. The three are in different units, so each is divided by its mean beyond 40 m, its
# plateau. All three level off near 25 m. Beyond, the classical variogram zigzags with the few pairs that hold the
# highest grades, while the madogram stays flat. The madogram starts higher because it scales like the square root of
# γ: for Gaussian increments, a madogram at 0.73 of its plateau matches a variogram at 0.73² = 0.53 of its sill.

# %%
lag, max_lag = 5.0, 80.0
estimators = {"matheron": INK, "cressie-hawkins": GRAY, "madogram": ACCENT}
fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
for e, color in estimators.items():
    curve = cs.experimental_variogram(vein, "AU_GPT", lag, max_lag, estimator=e)
    plateau = curve.gammas[curve.lags > 40].mean()
    ax.plot(curve.lags, curve.gammas / plateau, "o-", color=color, ms=3, lw=1, label=e)
ax.set(xlabel="Lag (m)", ylabel="estimate / plateau", ylim=(0, 1.2))
ax.legend()
save(fig, "estimators")

# %% [markdown]
# ## Outliers
#
# Drop the five highest composites, 0.14 % of the data, and recompute. The variogram loses 40 % at every lag, the
# madogram 10 %:

# %%
keep = au < np.sort(au)[-5]
trimmed = vein.filter(keep)
for e in ("matheron", "madogram"):
    full = cs.experimental_variogram(vein, "AU_GPT", lag, max_lag, estimator=e)
    cut = cs.experimental_variogram(trimmed, "AU_GPT", lag, max_lag, estimator=e)
    change = cut.gammas / full.gammas - 1
    print(
        f"{e:>9}: change without the top five, median {100 * np.median(change):.0f} %, "
        f"range {100 * change.min():.0f} to {100 * change.max():.0f} %"
    )

# %% [markdown]
# ## Dissemination
#
# `dissemination` takes a madogram and a classical variogram computed with the same arguments. On the gold it sits
# well below 1 at every lag. The logarithm of the grades, closer to Gaussian, brings it near 1:

# %%
fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
both = vein.with_column("log_au", np.log(au))
for name, column, color in [("Au", "AU_GPT", HIGHLIGHT), ("log Au", "log_au", ACCENT)]:
    madogram = cs.experimental_variogram(both, column, lag, max_lag, estimator="madogram")
    variogram = cs.experimental_variogram(both, column, lag, max_lag)
    ax.plot(madogram.lags, cs.dissemination(madogram, variogram), "o-", color=color, ms=3, lw=1, label=name)
ax.axhline(1.0, color=GRAY, lw=0.8, ls="--")
ax.set(xlabel="Lag (m)", ylabel="√π · M(h) / √γ(h)", ylim=(0, 1.2))
ax.legend()
save(fig, "dissemination")
