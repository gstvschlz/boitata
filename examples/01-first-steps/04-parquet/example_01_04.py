"""
# Storing containers in Parquet

`write_parquet` saves a `PointSet`, `BlockModel` or `Polylines` with its geometry, layout and CRS in the file metadata;
`read_parquet` returns the same container. The file is ordinary Parquet, so polars, pandas or DuckDB read it as a
table. Fitted estimators and categorical estimates are saved the same way.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import tempfile

import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import ACCENT, HIGHLIGHT, LIGHT, map_axes, save

# %% [markdown]
# Walker Lake `V` is kriged on a 1 m grid with a variogram fitted in eight directions ([variogram fitting](../../05-spatial-continuity/02-variogram-fitting/README.md) explains the fit), and
# the cells above 500 ppm are kept in a masked model.

# %%
samples = bt.datasets.walker_lake()
azimuths = np.arange(0, 180, 22.5)
directional = [bt.experimental_variogram(samples, "V", 10.0, 120.0, azimuth=a) for a in azimuths]
model = bt.Variogram.fit_directional(
    directional, [(a, 0) for a in azimuths], ["spherical", "spherical"], weighting="count/gamma"
)
grid = bt.BlockModel(origin=(0.5, 0.5), size=(1, 1), count=(260, 300), crs="local grid")
kriging = bt.OrdinaryKriging(model, bt.Search(radius=100, max_samples=24, min_samples=4)).fit(samples, "V")
estimate, variance = kriging.predict(grid, return_variance=True)
grid = grid.with_column("estimate", estimate).with_column("variance", variance)
rich = grid.mask(grid["estimate"] > 500)
print(grid)
print(rich)

# %% [markdown]
# The masked model keeps 10273 of the 78000 cells; its cell index is stored with the attributes.

# %%
folder = Path(tempfile.mkdtemp())
bt.write_parquet(folder / "grid.parquet", grid)
bt.write_parquet(folder / "rich.parquet", rich)
bt.write_csv(folder / "grid.csv", grid)
for name in ("grid.parquet", "rich.parquet", "grid.csv"):
    print(f"{name:>13}: {(folder / name).stat().st_size / 1e6:.2f} MB")

back = bt.read_parquet(folder / "rich.parquet")
print(back)
print("same cells:", np.array_equal(back.index, rich.index), "| crs:", back.crs)

# %% [markdown]
# The full grid takes 1.47 MB in Parquet against 4.04 MB as CSV. Grouping and summaries belong to polars, which reads
# the same file. A regular model stores no coordinates: its geometry is implicit and the row number is the cell index,
# x fastest, so a 1 m row of 260 cells is one northing.

# %%
table = pl.read_parquet(folder / "grid.parquet")
bands = (
    table.with_columns((pl.int_range(pl.len()) // 260 // 50 * 50).alias("northing band"))
    .group_by("northing band")
    .agg(
        pl.col("estimate").mean().round(0).alias("mean V"),
        (pl.col("estimate") > 500).mean().round(3).alias("share > 500"),
    )
    .sort("northing band")
)
print(bands)

# %% [markdown]
# A fitted estimator is saved the same way: its samples become columns and its variogram, search and options JSON
# in the file metadata. The estimator read back predicts exactly the same values. `ImplicitModel` and
# `LocalAnisotropy` save the same way.

# %%
kriging.to_parquet(folder / "kriging.parquet")
print(pl.read_parquet(folder / "kriging.parquet").head(3))
again = bt.OrdinaryKriging.from_parquet(folder / "kriging.parquet")
print("same estimates:", np.array_equal(again.predict(grid), estimate, equal_nan=True))

# %% [markdown]
# A categorical estimate keeps its `Categories` scheme, the names and colors of the codes. Walker Lake's type `T`
# (1 and 2) is kriged as indicators on a 5 m grid ([categorical indicator kriging](../../07-categories-and-domains/05-categorical-indicator-kriging/README.md)); the probabilities are saved as columns and the scheme
# as JSON in the metadata.

# %%
scheme = bt.Categories(["T1", "T2"], mapping={1: "T1", 2: "T2"}, colors=[LIGHT, ACCENT])
types = bt.CategoricalIndicatorKriging(
    bt.Variogram([("spherical", 0.25, 60.0)]), bt.Search(radius=80, max_samples=16), scheme=scheme
).fit(samples, "T")
summary = types.predict(bt.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(52, 60)))
summary.to_parquet(folder / "types.parquet")
print(pl.read_parquet(folder / "types.parquet").head(3))
kept = bt.CategoricalIndicatorSummary.from_parquet(folder / "types.parquet")
print(
    "same scheme:",
    kept.scheme == scheme,
    "| same probabilities:",
    np.array_equal(kept.probabilities, summary.probabilities),
)

# %% [markdown]
# Lines and polygons are `Polylines`: one attribute row per feature, each feature made of parts. A closed part is a
# ring whose last vertex joins the first; a ring inside another ring of the same feature is a hole. In Parquet a
# feature is one row whose `geometry` lists its parts, each a list of vertices; `Polylines.from_table` builds features
# from a long table with one row per vertex.

# %%
pit = [[60, 80], [60, 240], [200, 240], [200, 80]]
core = [[110, 140], [150, 140], [150, 190], [110, 190]]
pits = bt.Polylines([pit, core], closed=True, features=[0, 0], attributes={"name": ["pit"]}, crs="local grid")
bt.write_parquet(folder / "pit.parquet", pits)
print(pl.read_parquet(folder / "pit.parquet"))
pits = bt.read_parquet(folder / "pit.parquet")
print(pits)

vertices = pl.DataFrame({"ID": ["A-A'", "A-A'"], "X": [20.0, 240.0], "Y": [160.0, 160.0]})
lines = bt.Polylines.from_table(vertices)
print(lines)

fig, ax = plt.subplots(figsize=(5, 5.2), layout="constrained")
ax.scatter(*samples.coords[:, :2].T, s=6, color=LIGHT)
for part, closed in zip(pits.parts, pits.closed):
    ring = np.vstack([part, part[:1]]) if closed else part
    ax.plot(ring[:, 0], ring[:, 1], color=ACCENT, lw=1.6)
for part in lines.parts:
    ax.plot(part[:, 0], part[:, 1], color=HIGHLIGHT, lw=1.6, ls="--")
map_axes(ax, "Pit outline with its core, and section A-A'")
save(fig, "polylines")
