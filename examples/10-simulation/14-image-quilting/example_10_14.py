"""
# Image quilting

Image quilting copies whole patches of the training image instead of one cell at a time. Patches overlap along a
raster path. Quilting draws each patch among the `n_best` positions of the image that best match what the grid
already holds under it, and joins it to its neighbors along the seam where the two differ least. It is fast, and a
patch keeps the image's patterns intact up to its own size. It copies values the same way it copies codes: a mismatch
is then the squared difference over the squared value range of the training image, with no classes.
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

ti = bt.datasets.channel_training_image()
nx, ny = ti.count[:2]
image = ti.grid("facies")[0]
grid = bt.BlockModel((0, 0), (1, 1), (nx, ny))


def runs(img, axis):
    """Mean length of the runs of code 1 along `axis` of a (y, x) image."""
    lines = img if axis == 1 else img.T
    lengths = [len(r) for line in lines for r in "".join(map(str, line.astype(int))).split("0") if r]
    return np.mean(lengths)


print(
    f"training image: sand {image.mean():.1%}, runs {runs(image, 0):.1f} cells along Y, {runs(image, 1):.1f} across"
)


# %% [markdown]
# `patch_size` sets how much of the image travels in one piece, and the overlap is a sixth of it by default. Twenty
# realizations for each of three sizes:

# %%
realizations = {}
for size in (10, 30, 60):
    summary = bt.ImageQuilting(ti, "facies", patch_size=size).simulate(grid, n=20, seed=7, keep=True)
    reals = summary.realizations.reshape(-1, ny, nx)
    realizations[size] = reals[0]
    along = np.mean([runs(r, 0) for r in reals])
    print(
        f"patch_size={size}: sand {reals.mean():.1%}, runs {along:.1f} along Y, {np.mean([runs(r, 1) for r in reals]):.1f} across"
    )


# %% [markdown]
# Small patches cut the channels at their seams. At 30 cells the runs along Y match the image's; at 60 the
# realizations are the image cut and reassembled, with fewer seams and less variety between them.

# %%
codes = ListedColormap(["white", "black"])


def frame(ax, title):
    ax.set(title=title, xticks=[], yticks=[])
    for side in ax.spines.values():
        side.set(visible=True, color=INK, lw=0.6)


fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), layout="constrained")
panels = [(image, "Training image")] + [(r, f"patch_size={k}") for k, r in realizations.items()]
for ax, (img, title) in zip(axes, panels):
    ax.imshow(img, origin="lower", cmap=codes, vmin=0, vmax=1, interpolation="nearest")
    frame(ax, title)
save(fig, "patches")


# %% [markdown]
# When quilting chooses a patch, each hard datum weighs `data_weight` times a mismatched overlap cell, and no patch
# pastes over it, so hard data hold in all realizations. Soft probabilities add `soft_weight` times the mean of
# `1 - P(c)` over the patch, `c` the code the patch puts in each cell. A patch can come from anywhere in the image, so
# the truth here is the image mirrored east to west: the same patterns, none of them in the same place. 100 of its
# cells serve as hard data, as in [conditioning SNESIM](../../10-simulation/11-snesim-basics/README.md), and smoothing
# turns it into soft data as in [soft data in SNESIM](../../10-simulation/12-non-stationary-snesim/README.md):

# %%
truth = image[:, ::-1]
rows = np.random.default_rng(0).choice(nx * ny, size=100, replace=False)
wells = grid.coords[rows]
facies = truth.ravel()[rows].astype(int)
hard = bt.ImageQuilting(ti, "facies", patch_size=30).fit(wells, facies)
with_data = hard.simulate(grid, n=20, seed=3, keep=True)
print(f"hard data reproduced: {(with_data.realizations[:, rows] == facies).all()}")

kernel = np.ones(11) / 11
p_soft = np.apply_along_axis(np.convolve, 1, truth, kernel, "same")
p_soft = np.clip(np.apply_along_axis(np.convolve, 0, p_soft, kernel, "same"), 0.02, 0.98).ravel()
with_soft = bt.ImageQuilting(ti, "facies", patch_size=30, soft_weight=5.0).simulate(
    grid, n=20, seed=3, keep=[0], soft=np.column_stack([1 - p_soft, p_soft])
)
for name, summary in (("hard data", with_data), ("soft data", with_soft)):
    hits = (summary.realizations[0] == truth.ravel()).mean()
    print(f"{name}: realization 1 matches the truth in {hits:.0%} of cells")


# %% [markdown]
# A hundred scattered cells barely move a patch cut to fit its overlap, so they change little beyond their own cells.
# The soft map covers each cell, and the patches follow it. A map this sharp with `soft_weight=5` leaves the
# realizations little room to differ: P(sand) is near 0 or 1, gray only at a few seams. Lower the weight when the
# soft data are less certain than the patterns.

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
panels = (
    (with_data.realizations[0], "Realization 1, hard data", codes),
    (with_data.probabilities[:, 1], "P(sand), hard data", "gray_r"),
    (with_soft.realizations[0], "Realization 1, soft data", codes),
    (with_soft.probabilities[:, 1], "P(sand), soft data", "gray_r"),
)
for ax, (values, title, cmap) in zip(axes, panels):
    im = ax.imshow(grid.grid(values)[0], origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    frame(ax, title)
for ax in axes[:2]:
    ax.scatter(
        *wells[:, :2].T, s=10, c=np.where(facies == 1, "black", "white"), edgecolors=HIGHLIGHT, linewidths=0.9
    )
fig.colorbar(im, ax=axes[3], shrink=0.8, label="Probability")
save(fig, "conditioning")

# %% [markdown]
# ## Continuous values
#
# Quilting copies values as it copies codes. Here it fills the seismic section of
# [SNESIM basics](../../10-simulation/11-snesim-basics/README.md) between the same five traces.

# %%
seismic = bt.datasets.seismic_cube()
nx, ny, nz = seismic.count
cube = seismic.grid("amplitude")[::-1]  # time down: row 0 at 1600 ms
image = np.hstack([cube[:, j, :] for j in range(0, 30, 3)])
section_truth = cube[:, 40, :]
trained = bt.BlockModel((0, 1600), (25, 4), (image.shape[1], nz)).with_columns({"amplitude": image.ravel()})
section = bt.BlockModel((0, 1600), (25, 4), (nx, nz))
traces = [4, 13, 22, 31, 40]
at_traces = np.zeros((nz, nx), bool)
at_traces[:, traces] = True
wells = section.coords[at_traces.ravel()]
values = section_truth[at_traces]


# %% [markdown]
# Fifty realizations with patches 9 and 25 cells square, against SNESIM on the same data. The traces are 9 apart, so
# each 9-cell patch holds one, and the choice of each patch weighs the trace in it five times as much as its overlap.


# %%
def correlation(a, b):
    return bt.compare(np.ravel(a), np.ravel(b))["correlation"]


summaries = {
    f"quilting, patch {size}": bt.ImageQuilting(trained, "amplitude", patch_size=size)
    .fit(wells, values)
    .simulate(section, n=50, seed=11, keep=[0])
    for size in (9, 25)
}
summaries["SNESIM"] = (
    bt.SNESIM(trained, "amplitude").fit(wells, values).simulate(section, n=50, seed=11, keep=[0])
)
for name, summary in summaries.items():
    print(
        f"{name}: hard data reproduced {np.allclose(summary.realizations[:, at_traces.ravel()], values)}, "
        f"correlation with the true section: realization 1 {correlation(summary.realizations[0], section_truth):.2f}, "
        f"mean of 50 {correlation(summary.mean, section_truth):.2f}"
    )


# %% [markdown]
# Small patches follow the traces, and a single realization is nearer the truth than SNESIM's, since each value
# between two traces arrives with the reflector around it instead of cell by cell. A 25-cell patch spans three traces
# and cannot match them all: the traces stand out as stripes the patch around them fails to continue. SNESIM, cell by
# cell, is rougher than either. The seams of the small patches show as steps in the reflectors. The means of 50 are
# about as good for the small patches and SNESIM, since both keep what the traces fix.

# %%
extent = (0, nx * 25, 1600 + nz * 4, 1600)
style = {
    "cmap": "gray",
    "vmin": -8000,
    "vmax": 8000,
    "extent": extent,
    "aspect": "auto",
    "interpolation": "nearest",
}
fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained", sharey=True)
panels = [(section_truth, "True section and traces")] + [
    (section.grid(s.realizations[0])[0], f"Realization 1, {name}") for name, s in summaries.items()
]
for ax, (img, title) in zip(axes, panels):
    im = ax.imshow(img, **style)
    ax.set(title=title, xlabel="Distance (m)")
    for t in traces:
        ax.axvline((t + 0.5) * 25, color=HIGHLIGHT, lw=0.8)
axes[0].set_ylabel("Two-way time (ms)")
fig.colorbar(im, ax=axes, shrink=0.8, label="Amplitude")
save(fig, "sections")

# %% [markdown]
# [Seismic volumes](../../10-simulation/15-seismic-volumes/README.md) reads and writes SEG-Y and uses seismic as
# secondary data for quilting.
