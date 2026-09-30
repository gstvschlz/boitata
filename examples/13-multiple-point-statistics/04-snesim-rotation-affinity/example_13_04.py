"""
# Rotation and affinity

A training image has one direction and one scale; a deposit rarely keeps either. `anisotropy` turns and stretches the
patterns cell by cell without touching the image: each cell reads its neighbors through its own angles and
affinity, so one image can serve a whole field of orientations.
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
from common import HIGHLIGHT, INK, save
from matplotlib.colors import ListedColormap

ti = bt.datasets.strebelle()
n = ti.count[0]
grid = bt.BlockModel((0, 0), (1, 1), (n, n))
xy = grid.centroids
x, y = xy[:, 0], xy[:, 1]


# %% [markdown]
# `bt.LocalAnisotropy` holds, at each cell, the azimuth that turns the image's north, the ratios that shrink its X
# (semi-major) and Z (minor) axes against its Y axis, and a scale that grows all three. At azimuth 0 and scale 1 the
# cells read the image as it is.


# %%
def anisotropy(azimuth, semi=1.0, scale=1.0):
    azimuth, semi, scale = (np.broadcast_to(np.asarray(v, float), x.shape) for v in (azimuth, semi, scale))
    angles = np.column_stack([azimuth, np.zeros_like(x), np.zeros_like(x)])
    return bt.LocalAnisotropy(xy, angles, np.column_stack([semi, np.ones_like(x)]), scales=scale)


zones = (y > 170 - 0.3 * x).astype(int) + (y > 90 - 0.3 * x)
fields = {
    "As the image": (0.0, 1.0, 1.0),
    "Turned 45°": (45.0, 1.0, 1.0),
    "Twice as long": (0.0, 0.5, 2.0),
    "Fan, -40° to 40°": (-40 + 80 * x / n, 1.0, 1.0),
    "Three zones": (
        np.choose(zones, [0, 60, -30]),
        np.choose(zones, [1.0, 1.0, 0.5]),
        np.choose(zones, [1.5, 1, 1]),
    ),
}


# %% [markdown]
# "Twice as long" doubles the scale and halves the X ratio, so the channels double in length and keep their width.
# The three zones, split by two lines dipping east: the south enlarged 1.5 times, the middle turned 60°, the north
# turned -30° with channels half as wide. Angles are rounded to `angle_step` (10° by default), and each distinct
# rounded transform gets search trees of its own.


# %%
def runs_along_y(img):
    lengths = [len(r) for col in img.T for r in "".join(map(str, col.astype(int))).split("0") if r]
    return np.mean(lengths)


snesim = bt.SNESIM(ti, "facies")
realizations = {}
for title, (azimuth, semi, scale) in fields.items():
    summary = snesim.simulate(
        grid, n=1, seed=3, keep=True, anisotropy=anisotropy(azimuth, semi, scale), progress=False
    )
    realizations[title] = summary.realizations[0].reshape(n, n)
    print(
        f"{title}: template classes {snesim.n_classes}, sand runs {runs_along_y(realizations[title]):.1f} cells along Y"
    )

# %%
codes = ListedColormap(["white", "black"])
fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), layout="constrained")
for ax, (title, (azimuth, semi, scale)) in zip(axes[0], fields.items()):
    im = ax.imshow(
        np.broadcast_to(azimuth, x.shape).reshape(n, n), origin="lower", cmap="gray", vmin=-90, vmax=90
    )
    ax.set_title(title)
for ax, (title, img) in zip(axes[1], realizations.items()):
    ax.imshow(img, origin="lower", cmap=codes, vmin=0, vmax=1, interpolation="nearest")
for ax in (axes[0, 4], axes[1, 4]):
    ax.contour(zones.reshape(n, n), levels=[0.5, 1.5], colors=HIGHLIGHT, linewidths=1.2)
for ax in axes.flat:
    ax.set(xticks=[], yticks=[])
    for side in ax.spines.values():
        side.set(visible=True, color=INK, lw=0.6)
fig.colorbar(im, ax=axes[0], shrink=0.8, label="azimuth (°)", ticks=[-90, -45, 0, 45, 90])
save(fig, "rotation")


# %% [markdown]
# Turned 45°, a channel crosses the Y runs diagonally, so they shorten. Twice as long lengthens the runs from 18 to 22
# cells rather than doubling them: a doubled channel reaches past the grid template, and the coarse levels, which
# keep their spacing, see only half as far into the stretched image.
#
# The channels cross the zone boundaries without a seam: a cell near a boundary sees neighbors simulated under the
# other zone's transform and continues them.
#
# Continuous images turn the same way. The F3 sections of [continuous SNESIM](../../13-multiple-point-statistics/03-snesim-continuous/README.md)
# have nearly flat reflectors; an azimuth rising from -25° in the west to 25° in the east and back bends them into a
# fold. Here the cells are one trace by one sample, so an angle is measured in those units.

# %%
seismic = bt.datasets.f3_seismic()
nx, ny, nz = seismic.count
cube = seismic["amplitude"].astype(float).reshape(nz, ny, nx)[::-1]
image = np.hstack([cube[:, j, :] for j in range(0, 45, 3)])
sections = bt.BlockModel((0, 0), (1, 1), (image.shape[1], nz)).with_columns({"amplitude": image.ravel()})
width = 150
section = bt.BlockModel((0, 0), (1, 1), (width, nz))
along = section.centroids[:, 0]
fold = -25 * np.cos(np.pi * along / width)
field = bt.LocalAnisotropy(
    section.centroids, np.column_stack([fold, 0 * fold, 0 * fold]), np.ones((along.size, 2))
)
continuous = bt.SNESIM(sections, "amplitude")
flat = continuous.simulate(section, n=1, seed=2, keep=True, progress=False).realizations[0]
folded = continuous.simulate(section, n=1, seed=2, keep=True, anisotropy=field, progress=False).realizations[
    0
]

# %%
style = {"cmap": "gray", "vmin": -8000, "vmax": 8000, "aspect": "auto", "interpolation": "nearest"}
fig, axes = plt.subplots(3, 1, figsize=(10, 7), layout="constrained", sharex=True)
for ax, img, title in (
    (axes[0], image[:, :width], "Training image (first sections)"),
    (axes[1], flat.reshape(nz, width), "Realization as the image"),
    (axes[2], folded.reshape(nz, width), "Realization with the fold field"),
):
    ax.imshow(img, **style)
    ax.set(title=title, ylabel="sample")
axes[2].set_xlabel("trace")
save(fig, "fold")

# %% [markdown]
# [Several training images](../../13-multiple-point-statistics/05-snesim-training-images-by-zone/README.md) change the patterns themselves from zone
# to zone.
