"""
# data spacing

the equivalent data spacing (cabral pinto and deutsch, 2017) is the spacing of a square grid of holes that would put
as much data around a location as the real drilling does. it reads in meters whatever the layout: a cell inside a
square grid of 700 m scores about 700 m, and a cell among irregular holes scores the square grid they are worth. it is
the usual basis for resource classification ([classification](../../10-checking-models/05-classification/README.md)).
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
from common import ACCENT, GRAY, INK, map_axes, save

data = bt.datasets.coal_seam_thickness()
holes, lease, grid = data["boreholes"], data["boundary"], data["grid"]
print(f"{len(holes)} holes, {grid['INSIDE'].sum():.0f} cells of 100 m inside the lease")

# %% [markdown]
# ## in plan
#
# a coal seam drilled on a regular mesh, then infilled where the seam is thick. with `search=None`, `data_spacing`
# takes the plan distances `d` from a target to its nearest holes. for each `n` it averages the `n`-th and
# `n + 1`-th, `r = (d_n + d_n+1) / 2`, and turns the circle of radius `r`, which holds `n` holes, into the area per
# hole: `sqrt(pi r² / n)`. the result is the mean over `n` from 4 to 10. `hull=True` leaves cells outside the convex
# hull of the holes blank, where any spacing would be an extrapolation.

# %%
inside = grid["INSIDE"] == 1
spacing = bt.data_spacing(grid, holes, None, hull=True)
shown = np.where(inside, spacing, np.nan)
p10, p50, p90 = np.nanpercentile(shown, [10, 50, 90])
print(f"spacing in the lease: median {p50:.0f} m, P10 {p10:.0f} m, P90 {p90:.0f} m")
print(f"{np.mean(np.isnan(spacing[inside])):.1%} of the lease lies outside the hull of the holes")
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", width_ratios=[1.5, 1])
bt.plot.section(grid, shown, axis="z", index=0, colorbar=False, ax=a)
a.plot(*lease.coords[:, :2].T, color=GRAY, lw=0.8)
a.scatter(*holes.coords[:, :2].T, s=2, color=INK)
fig.colorbar(a.collections[0], ax=a, shrink=0.8, label="Spacing (m)")
map_axes(a, "Equivalent spacing in plan")
bt.plot.histogram(shown[~np.isnan(shown)], bins=np.arange(0, 1100, 50), stats=True, ax=b, color=ACCENT)
b.set(title="Cells of the lease", xlabel="Spacing (m)")
save(fig, "plan")

# %% [markdown]
# the infill clusters read 200 to 400 m and the regional mesh 550 to 700 m. the measure
# averages over the 4 to 11 nearest holes, so it changes smoothly across the edge of the infill rather than jumping
# at it.
#
# ## in 3D
#
# an iron formation cut by 187 diamond holes, about 100 m apart. with a `search`, the spacing comes from the samples
# inside its ellipsoid: `sqrt(V / (c n))`, with `V` the volume of the ellipsoid, `n` the samples inside it around the
# block and `c` their length. `n c` is the length of hole inside `V`, so `V / (n c)` is the plan area per hole for
# vertical holes. a `Drillholes` stands for its interval midpoints, and `c` defaults to the median interval length;
# for composites in a `PointSet`, pass `composite_length=`. the ellipsoid here reaches 150 m in plan and 30 m
# vertically, so it follows the flat formation.

# %%
plateau = bt.datasets.iron_formation_plateau()
drillholes = bt.Drillholes.from_tables(plateau)
model = plateau["block_model"]
formation = model.filter(np.isin(model["LITH"], ["IC", "HC", "HF", "IF"]))
search = bt.Search(150.0, ratios=(1.0, 0.2))
volume = bt.data_spacing(formation, drillholes, search, hull=True)
plan = bt.data_spacing(formation, drillholes, None, holes="HOLE_ID", hull=True)
for name, values in (("3D", volume), ("plan", plan)):
    p10, p50, p90 = np.nanpercentile(values, [10, 50, 90])
    print(
        f"{name:>4}: median {p50:.0f} m, P10 {p10:.0f} m, P90 {p90:.0f} m, blank {np.mean(np.isnan(values)):.1%}"
    )

# %% [markdown]
# the plan form also works on drill holes: `holes=` counts each hole once, at its nearest sample, so a hole with a
# hundred samples down its length is one hole, and every block in a column gets the same value. the 3D form sees
# depth. both agree near 100 m where the holes cross the formation, but blocks below the ends of the holes have fewer
# samples around them and read wider, up to blank where no sample is in reach. when most blocks have fewer than 4
# samples inside the ellipsoid, `data_spacing` warns: counts that small are unstable, and the ellipsoid should grow.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3), layout="constrained", sharey=True)
for ax, values, title in (
    (axes[0], plan, "Plan, holes counted once"),
    (axes[1], volume, "3D, ellipsoid 150 × 30 m"),
):
    bt.plot.section(
        formation, values, axis="y", index=40, colorbar=False, vmin=50, vmax=250, ax=ax, holes=drillholes
    )
    ax.set(title=title, xlabel="Easting (m)")
axes[1].set_ylabel("")
fig.colorbar(axes[1].collections[0], ax=axes, shrink=0.9, extend="max", label="Spacing (m)")
save(fig, "section")
