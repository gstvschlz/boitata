"""
# 87. External drift kriging

Copper grades from the soil geochemistry survey (topic 61's dataset) correlate weakly with the magnetics and
elevation of the covariates grid. `ExternalDriftKriging` drapes those covariates onto the samples and krige's
copper with them as an extra drift, in one system with the usual constant term; compared here with plain
ordinary kriging on the same grid.
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
from common import ACCENT, INK, map_axes, save

data = cs.datasets.soil_geochemistry_survey()
samples, covariates = data["samples"], data["covariates"]

# %% [markdown]
# `BlockModel.row_at` finds the cell under each sample; fancy-indexing the grid's columns at those rows drapes
# `MAG_NT` and `ELEVATION_M` onto the samples as two new columns.

# %%
rows = covariates.row_at(samples.coords[:, :2])
samples = samples.with_column("MAG_NT", covariates["MAG_NT"][rows]).with_column(
    "ELEVATION_M", covariates["ELEVATION_M"][rows]
)
print(
    f"correlation with Cu: magnetics {np.corrcoef(samples['CU_PPM'], samples['MAG_NT'])[0, 1]:.2f}, "
    f"elevation {np.corrcoef(samples['CU_PPM'], samples['ELEVATION_M'])[0, 1]:.2f}"
)

# %% [markdown]
# A single spherical structure fitted to the omnidirectional experimental variogram of copper.

# %%
xy, cu = samples.coords, samples["CU_PPM"]
model = cs.experimental_variogram(xy, cu, 200.0, 2500.0).fit(["spherical"], weighting="count/gamma")
search = cs.Search(radius=2000.0, max_samples=24, min_samples=4)
print(model)

# %% [markdown]
# `ExternalDriftKriging` takes the drift column names at `fit` (read off `samples`) and again at `predict` (read
# off `covariates`, which already carries them); `OrdinaryKriging` ignores the covariates entirely.

# %%
inside = covariates["INSIDE"] == 1
ok = cs.OrdinaryKriging(model, search).fit(samples, "CU_PPM").predict(covariates)
edk = (
    cs.ExternalDriftKriging(model, search, ["MAG_NT", "ELEVATION_M"])
    .fit(samples, "CU_PPM")
    .predict(covariates)
)
ok, edk = (np.where(inside, e, np.nan) for e in (ok, edk))
print(f"mean: ordinary {np.nanmean(ok):.1f}, external drift {np.nanmean(edk):.1f} ppm")
print(f"correlation between the two maps: {np.corrcoef(ok[inside], edk[inside])[0, 1]:.3f}")

# %% [markdown]
# The two maps agree where covariates are flat, and pull apart where magnetics or elevation depart from the
# neighborhood mean: the drift nudges the estimate up in high-magnetics, high-elevation ground even far from
# any sample, something ordinary kriging's constant mean cannot do.

# %%
nx, ny = covariates.count[0], covariates.count[1]
ox, oy = covariates.origin[0], covariates.origin[1]
sx, sy = covariates.size[0], covariates.size[1]
shape, extent = (ny, nx), (ox, ox + sx * nx, oy, oy + sy * ny)
vmax = np.nanpercentile(np.r_[ok, edk], 98)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
for ax, image, title in ((axes[0], ok, "Ordinary kriging"), (axes[1], edk, "External drift kriging")):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, vmin=0, vmax=vmax)
    ax.scatter(samples.coords[:, 0], samples.coords[:, 1], s=2, color=INK, linewidths=0)
    map_axes(ax, title)
fig.colorbar(im, ax=axes[:2], shrink=0.8, label="Cu (ppm)")
diff = ax = axes[2]
d = ax.imshow((edk - ok).reshape(shape), origin="lower", extent=extent, cmap="RdBu_r", vmin=-15, vmax=15)
ax.scatter(samples.coords[:, 0], samples.coords[:, 1], s=2, color=ACCENT, linewidths=0)
map_axes(ax, "External drift − ordinary")
fig.colorbar(d, ax=diff, shrink=0.8, label="Δ Cu (ppm)")
save(fig, "maps")

# %% [markdown]
# `cross_validate` and weight declustering are not wired up for external-drift kriging yet: each held-out point or
# target would need its own covariate row, which those two do not carry through.
