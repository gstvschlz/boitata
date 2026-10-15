"""
# local mine grid

many mines survey in a local grid: a convenient origin, grid north along the orebody and round numbers. `LocalGrid`
is the similarity transform from that grid to a projected CRS (a scale, a rotation, a shift and an elevation offset),
fitted by least squares to control points known in both systems. it moves points, drill holes, polylines, meshes
and block models either way.
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

# %% [markdown]
# ## control points
#
# six survey monuments carry local coordinates and UTM coordinates from GNSS, with a few centimeters of survey error.
# the true grid here turns grid north 128° clockwise from UTM north and has a scale factor of 0.9996.

# %%
rng = np.random.default_rng(7)
truth = bt.LocalGrid(scale=0.9996, rotation=128.0, shift=(612_350.0, 7_731_820.0), z_shift=-412.0)
local = np.array(
    [
        [1000.0, 1000.0, 1450.0],
        [2600.0, 1100.0, 1462.0],
        [2500.0, 3200.0, 1488.0],
        [1100.0, 3000.0, 1471.0],
        [1800.0, 2100.0, 1430.0],
        [3100.0, 2400.0, 1502.0],
    ]
)
utm = truth.apply(local) + rng.normal(0.0, 0.03, local.shape)
grid = bt.LocalGrid.fit(local, utm, crs="EPSG:32723")
print(grid)
residuals = grid.residuals(local, utm)
print("residuals (m):")
print(np.round(residuals, 3))
print(f"horizontal rms: {np.sqrt(np.mean(np.sum(residuals[:, :2] ** 2, axis=1))):.3f} m")

# %% [markdown]
# the fit recovers the rotation within a few thousandths of a degree. the residuals are what the survey error leaves
# after the best similarity. a monument with a residual far above the others is a misidentified or disturbed one.
#
# ## moving a block model
#
# a block model built in the local grid moves with its origin. its azimuth turns by the grid rotation and its block
# sizes scale by the grid scale, so every block lands where its local centroid maps. the result carries the grid's
# CRS, and `inverse` brings it back exactly.

# %%
model = bt.BlockModel((1200.0, 1300.0, 1300.0), (25.0, 25.0, 10.0), (64, 64, 1))
holes = bt.Drillholes(
    {
        "HOLE_ID": ["A", "B", "C"],
        "X": [1500.0, 2200.0, 2800.0],
        "Y": [1800.0, 2500.0, 2000.0],
        "Z": [1460.0] * 3,
    },
    {"HOLE_ID": ["A", "B", "C"], "DEPTH": [0.0] * 3, "AZIMUTH": [90.0, 45.0, 0.0], "DIP": [60.0] * 3},
    {"HOLE_ID": ["A", "B", "C"], "FROM": [0.0] * 3, "TO": [400.0] * 3},
)
projected = grid.apply(model)
print(projected)
print(f"rotation {projected.rotation[0]:.3f}, size {np.round(projected.size, 4)}, crs {projected.crs}")
gap = np.abs(projected.coords - grid.apply(model.coords)).max()
print(f"largest centroid gap: {gap:.1e} m")
back = grid.inverse(projected)
print(f"round trip: {np.abs(back.coords - model.coords).max():.1e} m, crs {back.crs}")
moved_holes = grid.apply(holes)

footprint = np.array(
    [[1200.0, 1300.0, 1300.0], [2800.0, 1300.0, 1300.0], [2800.0, 2900.0, 1300.0], [1200.0, 2900.0, 1300.0]]
)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
for ax, ring, monuments, traces, title in (
    (a, footprint, local, holes, "Local mine grid"),
    (b, grid.apply(footprint), utm, moved_holes, "UTM zone 23S (EPSG:32723)"),
):
    ax.fill(*ring[:, :2].T, facecolor="#eef2f6", edgecolor=GRAY, lw=1)
    paths = traces.paths()
    for hole in traces.holes:
        rows = np.array(paths["HOLE_ID"]) == hole
        ax.plot(paths["x"][rows], paths["y"][rows], color=ACCENT, lw=1.5)
    ax.scatter(
        *monuments[:, :2].T, marker="^", s=40, color=HIGHLIGHT, edgecolors=INK, linewidths=0.5, zorder=3
    )
    map_axes(ax, title)
a.set_xlabel("Local east (m)")
a.set_ylabel("Local north (m)")
b.ticklabel_format(useOffset=False, style="plain")
b.tick_params(axis="x", labelrotation=30)
save(fig, "grids")

# %% [markdown]
# a plain `(n, 3)` array moves too, here the footprint of the block model. the footprint, the hole traces and the
# monuments keep their shapes; only their frame changes.
