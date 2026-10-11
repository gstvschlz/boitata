"""
# universal kriging

the coal seam thins to the east. `detrend` fits that drift as a `Trend`. universal kriging estimates the thickness with
the drift re-fitted in every neighborhood, using the variogram of the residuals. inside the drilled lease it agrees with
ordinary kriging; beyond the last holes it keeps following the drift.
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
from common import ACCENT, GRAY, HIGHLIGHT, INK, map_axes, save

data = bt.datasets.coal_seam_thickness()
holes, grid, lease = data["boreholes"], data["grid"], data["boundary"]
inside = grid["INSIDE"] == 1

# %% [markdown]
# ## trend
#
# a plane fitted by least squares: `coefficients` are the constant, then the x, y and z slopes.

# %%
trend, residuals = bt.detrend(holes, "THICKNESS_M", degree=1)
constant, per_x, per_y, _ = trend.coefficients
print(f"thickness = {constant:.2f} {per_x * 1000:+.3f} m/km east {per_y * 1000:+.3f} m/km north")
print(f"variance {holes['THICKNESS_M'].var():.2f} m², of the residuals {residuals.var():.2f} m²")

# %% [markdown]
# the drift inflates the variogram of the thickness at long lags, and the residuals level off lower. universal kriging
# takes the residual variogram, since the weights estimate the drift.

# %%
raw = bt.experimental_variogram(holes, "THICKNESS_M", 250.0, 4000.0)
detrended = bt.experimental_variogram(holes.coords, residuals, 250.0, 4000.0)
model = raw.fit("spherical")
residual_model = detrended.fit("spherical")
print(model, residual_model, sep="\n")

fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
bt.plot.variogram(raw, variogram=model, ax=ax, color=GRAY, label="thickness")
bt.plot.variogram(detrended, variogram=residual_model, ax=ax, color=ACCENT, label="residuals")
ax.set_title("Variograms with and without the drift")
ax.set_xlabel("Lag (m)")
ax.legend()
save(fig, "variograms")

# %% [markdown]
# ## estimates
#
# universal kriging with a linear drift against ordinary kriging, same search:

# %%
search = bt.Search(radius=6000, max_samples=32, min_samples=12)
ordinary = bt.OrdinaryKriging(model, search).fit(holes, "THICKNESS_M")
universal = bt.UniversalKriging(residual_model, search, degree=1).fit(holes, "THICKNESS_M")
ok, uk = ordinary.predict(grid), universal.predict(grid)
difference = uk - ok
print(f"inside the lease: mean |UK - OK| {np.nanmean(np.abs(difference[inside])):.3f} m")
for name, estimator in (("ordinary", ordinary), ("universal", universal)):
    cv = estimator.cross_validate()
    print(f"{name:>9} cross-validation: ME {cv.mean_error:+.3f}  RMSE {cv.rmse:.3f}  slope {cv.slope:.2f}")

# %%
low, high = grid.bounds
extent = (low[0], high[0], low[1], high[1])


def image(ax, values, **kwargs):
    values = grid.grid(np.where(inside, values, np.nan))[0]
    shown = ax.imshow(values, origin="lower", extent=extent, **kwargs)
    ax.plot(*lease.coords[:, :2].T, color=INK, lw=0.6)
    return shown


fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
norm = plt.Normalize(0, 5)
image(axes[0], trend.predict(grid), norm=norm)
map_axes(axes[0], "Trend")
shown = image(axes[1], uk, norm=norm)
axes[1].scatter(*holes.coords[:, :2].T, s=2, color=INK)
map_axes(axes[1], "Universal kriging")
fig.colorbar(shown, ax=axes[:2], label="Thickness (m)", shrink=0.8)
limit = 0.3
shown = image(axes[2], difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
axes[2].scatter(*holes.coords[:, :2].T, s=2, color=INK)
map_axes(axes[2], "Universal minus ordinary")
fig.colorbar(shown, ax=axes[2], label="m", shrink=0.8)
for ax in axes[1:]:
    ax.set_ylabel("")
save(fig, "maps")

# %% [markdown]
# where holes surround a cell, the local drift and the local mean give nearly the same estimate, and the two
# cross-validations are equally good. the differences sit at the edges of the lease, where the neighbors lie on one
# side: universal kriging is thicker in the thick west corner and thinner in the thin northeast.
#
# ## beyond the data
#
# along an east-west line through the middle of the lease, extended past the last holes:

# %%
x = np.arange(20000.0, 36001.0, 100.0)
line = np.column_stack([x, np.full_like(x, 54000.0)])
near = np.abs(holes.y - 54000.0) < 500
print(f"easternmost hole at {holes.x.max():.0f} m")
profile = {
    "ordinary": ordinary.predict(line),
    "universal": universal.predict(line),
    "trend": trend.predict(line),
}
for at in (32000, 34000, 36000):
    i = np.searchsorted(x, at)
    print(f"x = {at}: " + ", ".join(f"{k} {v[i]:.2f} m" for k, v in profile.items()))

fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
ax.scatter(holes.x[near], holes["THICKNESS_M"][near], s=8, color=GRAY, label="holes within 500 m")
ax.plot(x, profile["trend"], color=INK, ls="--", lw=1, label="trend")
ax.plot(x, profile["ordinary"], color=ACCENT, label="ordinary kriging")
ax.plot(x, profile["universal"], color=HIGHLIGHT, label="universal kriging")
ax.axvline(holes.x.max(), color=GRAY, lw=0.8, ls=":")
ax.set(xlabel="Easting (m)", ylabel="Thickness (m)")
ax.set_title("Profile at northing 54 000 m")
ax.legend(ncols=2)
save(fig, "profile")

# %% [markdown]
# past the last hole ordinary kriging levels off at the mean of its neighbors, about 1.4 m, while universal kriging
# carries the thinning on, down to 0.66 m at 36 km against 0.56 m for the trend. whether that is right is a geological
# call. universal kriging fits the local drift from the neighbors, so it needs a wider search than ordinary kriging:
# with a few one-sided samples the drift is poorly determined and its extrapolation erratic.
