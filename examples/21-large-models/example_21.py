"""
# 21. Models larger than memory

A block model can stay in a Parquet file and be processed a chunk at a time, so its size is limited by the disk, not
by memory. Here the zinc of the drill-hole dataset is modelled on 2 m blocks over the massive sulphide lens and the
kilometre of rock the deepest holes cross — almost 20 million blocks — kriged and simulated without ever holding
more than a million of them.
"""

# %% [hidden]
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore", ".*locations hold several samples")

# %%
import shutil
import tempfile
import time

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import HIGHLIGHT, save
from matplotlib.colors import PowerNorm

composites = cs.datasets.drillholes().composite(2.0, ["ZN"])
xyz, zn = composites.coords, composites["ZN"]
window = ~np.isnan(zn) & (xyz[:, 0] > 5250) & (xyz[:, 0] < 5550) & (xyz[:, 1] > 8000) & (xyz[:, 1] < 8500)
xyz, zn = xyz[window], zn[window]
holes = np.array(composites["hole"], dtype=object)[window]
weights = cs.cell_declustering(xyz, zn, sizes=np.arange(10, 100, 10)).weights
print(f"{len(zn):,} composites of 2 m from {len(set(holes))} holes")

# %% [markdown]
# The model is written once. A regular model stores no coordinates, only its geometry; with no columns yet the file
# holds just the cell index. `BlockModelFile` reads the geometry and hands out `BlockModel` pieces of at most `rows`
# blocks.

# %%
folder = Path(tempfile.mkdtemp())
origin = np.floor(xyz.min(axis=0) / 2) * 2
count = tuple(int(c) for c in np.ceil((xyz.max(axis=0) - origin) / 2))
cs.write_parquet(folder / "grid.parquet", cs.BlockModel(origin=tuple(origin), size=(2, 2, 2), count=count))
file = cs.BlockModelFile(folder / "grid.parquet")
size = (folder / "grid.parquet").stat().st_size / 1e6
print(f"{len(file):,} blocks ({count[0]} × {count[1]} × {count[2]}), {size:.0f} MB")
print(f"{sum(1 for _ in file.chunks(rows=1_000_000))} chunks of at most 1,000,000 blocks")

# %% [markdown]
# `map_blocks` streams the file through any function of a chunk and writes the columns it returns next to the
# input's. Ordinary kriging is independent block by block, so chunked kriging equals kriging the whole model; blocks
# with fewer than four composites within 60 m are left unestimated.

# %%
grades = cs.experimental_variogram(xyz, zn, 10.0, 150.0).fit("spherical")
search = cs.Search(radius=60, max_samples=16, min_samples=4, max_per_hole=4)
kriging = cs.OrdinaryKriging(grades, search).fit(xyz, zn, holes=holes)
start = time.perf_counter()
cs.map_blocks(
    folder / "grid.parquet", folder / "kriged.parquet", lambda chunk: {"zn": kriging.predict(chunk)}
)
print(f"kriged in {time.perf_counter() - start:.1f} s")

# %% [markdown]
# Turning bands simulates every realization's bands once over the model's extent, then evaluates and conditions
# them chunk by chunk. Conditioning only reaches nodes within the variogram range of a composite, so beyond it the
# search is skipped. `simulate_to_parquet` writes the same summary `simulate` would return for the whole model,
# plus each realization's global statistics.

# %%
scores = cs.NormalScore().fit_transform(zn, weights=weights)
fitted = cs.experimental_variogram(xyz, scores, 10.0, 150.0).fit("spherical")
sill = fitted.nugget + fitted.structures[0].sill
reach = fitted.structures[0].range
gaussian = cs.Variogram([("spherical", fitted.structures[0].sill / sill, reach)], nugget=fitted.nugget / sill)
bands = cs.TurningBands(gaussian, bands=100, search=cs.Search(radius=reach, max_samples=16))
bands.fit(xyz, zn, weights=weights, holes=holes)
start = time.perf_counter()
result = bands.simulate_to_parquet(
    folder / "kriged.parquet", folder / "simulated.parquet", n=10, seed=1, cutoffs=[10.0]
)
seconds = time.perf_counter() - start
low, high = np.quantile(result["realization_above"][0], [0.1, 0.9])
print(f"10 realizations in {seconds:.0f} s; blocks above 10 % Zn: P10 {low:.2%}, P90 {high:.2%} of the model")
print(f"output {(folder / 'simulated.parquet').stat().st_size / 1e6:.0f} MB")

# %% [markdown]
# The output is too big to want in memory, so the east–west section with the most composites is collected from the
# chunks, reading only the columns it needs. Near the holes the simulations follow the data; beyond the variogram
# range kriging leaves blocks unestimated while each realization draws from the declustered histogram, so there the
# probability of exceeding 10 % Zn is just the global proportion — what is known, not detail.

# %%
nx, ny, nz = count
row = int(np.bincount(((xyz[:, 1] - origin[1]) // 2).astype(int), minlength=ny).argmax())
section = {name: np.full((nz, nx), np.nan) for name in ("zn", "mean", "p_above_10")}
for chunk in cs.BlockModelFile(folder / "simulated.parquet").chunks(columns=list(section)):
    index = chunk.index
    on = (index // nx) % ny == row
    k, i = index[on] // (nx * ny), index[on] % nx
    for name, image in section.items():
        image[k, i] = chunk[name][on]

north = origin[1] + (row + 0.5) * 2
near = np.abs(xyz[:, 1] - north) < 2
extent = (origin[0], origin[0] + 2 * nx, origin[2], origin[2] + 2 * nz)
fig, axes = plt.subplots(1, 3, figsize=(8, 7), layout="constrained", sharey=True)
for ax, (name, image), title, style in zip(
    axes,
    section.items(),
    ("Kriged Zn (%)", "Mean of 10 simulations (%)", "P(Zn > 10 %)"),
    (
        {"norm": PowerNorm(0.5, vmin=0, vmax=30)},
        {"norm": PowerNorm(0.5, vmin=0, vmax=30)},
        {"vmin": 0, "vmax": 1},
    ),
    strict=True,
):
    im = ax.imshow(image, origin="lower", extent=extent, **style)
    ax.scatter(xyz[near, 0], xyz[near, 2], s=2, color=HIGHLIGHT, linewidths=0)
    ax.set(title=title, xlabel="Easting (m)")
    fig.colorbar(im, ax=ax, shrink=0.5, location="bottom")
axes[0].set_ylabel("Elevation (m)")
save(fig, "section")

# %% [hidden]
shutil.rmtree(folder, ignore_errors=True)
