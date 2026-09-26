"""
# 14. Storing containers in Parquet

`write_parquet` saves a `PointSet` or `BlockModel` with its geometry, layout and CRS in the file metadata;
`read_parquet` returns the same container. The file is ordinary Parquet, so polars, pandas or DuckDB read it as a
table.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

# %%
import tempfile

import ceres as cs
import numpy as np
import polars as pl

samples = cs.PointSet.from_table(cs.read_csv(cs.datasets.fetch("walker-lake/sample.csv")), crs="local grid")
model = cs.Variogram.from_json((HERE.parent / "03-variography" / "model.json").read_text())
grid = cs.BlockModel(origin=(0, 0), size=(1, 1), count=(260, 300), rotation=(0, 0, 0), crs="local grid")
kriging = cs.OrdinaryKriging(model, cs.Search(radius=100, max_samples=24, min_samples=4)).fit(
    samples.coords, samples["V"]
)
estimate, variance = kriging.predict(grid, return_variance=True)
grid = grid.with_column("estimate", estimate).with_column("variance", variance)
rich = grid.mask(grid["estimate"] > 500)
print(grid)
print(rich)

# %% [markdown]
# Only the cells above 500 ppm are kept in the masked model; its cell index is stored with the attributes.

# %%
folder = Path(tempfile.mkdtemp())
cs.write_parquet(folder / "grid.parquet", grid)
cs.write_parquet(folder / "rich.parquet", rich)
cs.write_csv(folder / "grid.csv", grid)
for name in ("grid.parquet", "rich.parquet", "grid.csv"):
    print(f"{name:>13}: {(folder / name).stat().st_size / 1e6:.2f} MB")

back = cs.read_parquet(folder / "rich.parquet")
print(back)
print("same cells:", np.array_equal(back.index, rich.index), "| crs:", back.crs)

# %% [markdown]
# Grouping and summaries belong to polars, which reads the same file. A regular model stores no coordinates: its
# geometry is implicit and the row number is the cell index, x fastest, so a 1 m row of 260 cells is one northing.

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
again = cs.OrdinaryKriging.from_parquet(folder / "kriging.parquet")
print("same estimates:", np.array_equal(again.predict(grid), estimate, equal_nan=True))

# %% [markdown]
# GIS software exchanges points as shapefiles. `write_shapefile` stores the coordinates, the attributes (names of at
# most 10 characters) and the CRS in a `.prj`; `read_shapefile` returns the same PointSet.

# %%
cs.write_shapefile(folder / "samples.shp", samples)
print(sorted(p.name for p in folder.glob("samples.*")))
again = cs.read_shapefile(folder / "samples.shp")
print("same points:", np.array_equal(again.coords, samples.coords), "| crs:", again.crs)

# %% [markdown]
# Rasters travel as GeoTIFF. `write_geotiff` writes each column of a 2D grid as a band, nulls as the `nodata` value
# and the CRS in the GeoKeys, so GIS software opens it as a georeferenced raster; `read_geotiff` returns the same
# grid. Rotated grids are supported, and a masked model is written with nodata in its absent cells.

# %%
cs.write_geotiff(folder / "grid.tif", grid)
raster = cs.read_geotiff(folder / "grid.tif")
print(raster)
print(f"{(folder / 'grid.tif').stat().st_size / 1e6:.2f} MB")
print(
    "same grid:", raster.origin == grid.origin, np.array_equal(raster["estimate"], estimate, equal_nan=True)
)
