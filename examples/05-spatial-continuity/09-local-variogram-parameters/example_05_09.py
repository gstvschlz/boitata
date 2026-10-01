"""
# local variogram parameters

[locally varying anisotropy](../../06-kriging/14-local-anisotropy/README.md) bent the variogram along walker lake's
high-`V` bodies but kept one set of ranges everywhere. here the local frames of
[locally varying anisotropy](../../06-kriging/14-local-anisotropy/README.md) measure experimental variograms that follow
the bodies. moving-window fits of those variograms give a range scale and a ratio per region, and kriging takes them the
same way it takes the local angles.
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
from common import ACCENT, GRAY, HIGHLIGHT, map_axes, save
from matplotlib.colors import PowerNorm

samples = bt.datasets.walker_lake()
truth = bt.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
grid = bt.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
nodes = grid.centroids.astype(int)
true_at_nodes = truth[nodes[:, 1] - 1, nodes[:, 0] - 1]

# %% [markdown]
# the global model and the orientation field come from
# [locally varying anisotropy](../../06-kriging/14-local-anisotropy/README.md): two nested structures fitted along N170°
# and across it, and local directions from the gradient of an isotropic guide estimate.

# %%
azimuth, lag, max_lag = 170.0, 10.0, 120.0
along = bt.experimental_variogram(xy, v, lag, max_lag, azimuth=azimuth).fit(
    ["spherical", "spherical"], weighting="count/gamma"
)
across = bt.experimental_variogram(xy, v, lag, max_lag, azimuth=azimuth + 90).fit(
    ["spherical", "spherical"],
    weighting="count/gamma",
    nugget=along.nugget,
    sills=[s.sill for s in along.structures],
)
model = along.with_anisotropy(
    (azimuth, 0, 0), (across.structures[-1].range / along.structures[-1].range, 1.0)
)
isotropic = bt.Variogram([("spherical", model.sill - model.nugget, 30.0)], nugget=model.nugget)
guide = bt.OrdinaryKriging(isotropic, bt.Search(radius=60, max_samples=16)).fit(xy, v).predict(grid)
lva = bt.LocalAnisotropy.from_grid(
    grid.with_column("guide", guide), "guide", window=3, ratios=(0.3, 1.0)
).smooth(25.0)
print(f"global ratio {model.ratios[0]:.2f}, ranges {[round(s.range) for s in model.structures]} m")

# %% [markdown]
# ## variograms along the local directions
#
# with `anisotropy=`, `experimental_variogram` reads each pair in the local frame of its tail: azimuth 0 is the local
# major axis wherever the pair sits, azimuth 90 the local semi-major. compared with the fixed N170° and N80° directions,
# the local major direction keeps pairs inside the bodies and rises more slowly; the local cross direction crosses them
# everywhere and rises faster.

# %%
fixed = [bt.experimental_variogram(xy, v, lag, max_lag, azimuth=a) for a in (azimuth, azimuth + 90)]
local = [bt.experimental_variogram(xy, v, lag, max_lag, azimuth=a, anisotropy=lva) for a in (0.0, 90.0)]
fig, ax = plt.subplots(figsize=(6.4, 4), layout="constrained")
for exp, color, style, label in (
    (fixed[0], GRAY, "-", "N170°, fixed"),
    (fixed[1], GRAY, "--", "N80°, fixed"),
    (local[0], ACCENT, "-", "local major"),
    (local[1], HIGHLIGHT, "--", "local semi-major"),
):
    ax.plot(exp.lags, exp.gammas, style, marker="o", ms=3, color=color, label=label)
ax.set(xlabel="Lag (m)", ylabel="γ (ppm²)", title="Fixed and local directions")
ax.legend(frameon=False)
save(fig, "variograms")

# %% [markdown]
# `local_variogram_parameters` fits the shape of the global model (nugget and structures, rescaled to the variance of
# the samples in a window) to such variograms. with one window over the whole deposit it returns one fit: in the global
# frame it gives back the global model (scale 1), in the local frames a longer and narrower one.

# %%
center = [[130.0, 150.0]]
for name, field in (("global frame", None), ("local frames", lva)):
    whole = bt.local_variogram_parameters(
        xy, v, center, variogram=model, window=400.0, lag=lag, anisotropy=field
    )
    print(f"{name}: ratio {whole.ratios[0, 0]:.2f}, scale {whole.scales[0]:.2f}")

# %% [markdown]
# ## moving-window fits
#
# the same fit runs on a window of 100 m around each node of a coarse 20 m grid, in the local frames, and the result is
# smoothed over 60 m. the angles stay those of the field; each node gets its semi-major ratio and a scale that
# multiplies every range of the model. the coarse grid stores the result like any other attribute.

# %%
coarse = bt.BlockModel(origin=(0, 0), size=(20, 20), count=(13, 15))
fitted = bt.local_variogram_parameters(
    xy, v, coarse, variogram=model, window=100.0, lag=lag, max_lag=60.0, anisotropy=lva
).smooth(60.0)
coarse = coarse.with_column("scale", fitted.scales).with_column("ratio", fitted.ratios[:, 0])
print("scale percentiles (10, 50, 90):", np.percentile(fitted.scales, [10, 50, 90]).round(2))
print("ratio percentiles (10, 50, 90):", np.percentile(fitted.ratios[:, 0], [10, 50, 90]).round(2))

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), layout="constrained")
for ax, name, title, cmap in (
    (axes[0], "scale", "Range scale", "cividis"),
    (axes[1], "ratio", "Semi-major / major ratio", "Greys"),
):
    im = ax.imshow(coarse[name].reshape(15, 13), origin="lower", extent=(0, 260, 0, 300), cmap=cmap)
    ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.5, color=HIGHLIGHT)
    map_axes(ax, title)
    fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, "parameters")

# %% [markdown]
# ## kriging with local ranges
#
# four runs share the model and search: global anisotropy, local angles
# ([locally varying anisotropy](../../06-kriging/14-local-anisotropy/README.md)), local angles with the fitted scales
# only, and local angles with the fitted scales and ratios. kriging takes the field from the nearest coarse cell. errors
# are against the exhaustive values at the nodes.

# %%
search = bt.Search(radius=100, max_samples=24, min_samples=1)
ok = bt.OrdinaryKriging(model, search).fit(xy, v)
scaled = bt.LocalAnisotropy(lva.coords, lva.angles, lva.ratios, scales=fitted.at(lva.coords).scales)
estimates = {
    f"global N{model.rotation[0]:.0f}°": ok.predict(grid),
    "local angles": ok.predict(grid, anisotropy=lva),
    "+ scales": ok.predict(grid, anisotropy=scaled),
    "+ scales and ratios": ok.predict(grid, anisotropy=fitted),
}
for name, estimate in estimates.items():
    error = estimate - true_at_nodes
    print(
        f"{name:>20}: RMSE {np.sqrt(np.mean(error**2)):.1f} ppm, "
        f"correlation {np.corrcoef(estimate, true_at_nodes)[0, 1]:.3f}"
    )

# %%
norm = PowerNorm(0.5, vmin=0, vmax=1500)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
for ax, (image, title) in zip(
    axes,
    (
        (true_at_nodes, "True V at grid nodes"),
        (estimates["local angles"], "Local angles"),
        (estimates["+ scales and ratios"], "Local angles, scales and ratios"),
    ),
):
    im = ax.imshow(image.reshape(60, 52), origin="lower", extent=(0.5, 260.5, 0.5, 300.5), norm=norm)
    map_axes(ax, title)
fig.colorbar(im, ax=axes, shrink=0.8, label="V (ppm)")
save(fig, "kriging")

# %% [markdown]
# the scales alone change the estimate little: RMSE 148.1 against 148.3 ppm. longer ranges with the same nugget and
# ratios leave the kriging weights close. the fitted ratios change them: the windows see bodies about 0.14 as wide as
# they are long, against 0.33 in the global model, and kriging with those needle-thin ellipses streaks along the field's
# directions. wherever the guide's directions are off, the estimate follows them across the bodies, and the RMSE rises
# to 161.1 ppm, worse than the global model. narrow local ratios pay off only as far as you can trust the orientation
# field. with a field this rough, keep the global ratios and take the local angles and scales. SGS, indicator and
# categorical kriging take the same `anisotropy=` field, scales included.
