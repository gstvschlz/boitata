# 57. Storing containers in Parquet

`write_parquet` saves a `PointSet`, `BlockModel` or `Polylines` with its geometry, layout and CRS in the file metadata;
`read_parquet` returns the same container. The file is ordinary Parquet, so polars, pandas or DuckDB read it as a
table.

<details><summary>Python</summary>

```python
import tempfile

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import ACCENT, HIGHLIGHT, LIGHT, map_axes, save

samples = cs.PointSet.from_table(
    cs.read_csv(cs.datasets.fetch("mining/2d/walker-lake/sample.csv")), crs="local grid"
)
model = cs.Variogram.from_json((HERE.parent / "model.json").read_text())
grid = cs.BlockModel(origin=(0, 0), size=(1, 1), count=(260, 300), rotation=(0, 0, 0), crs="local grid")
kriging = cs.OrdinaryKriging(model, cs.Search(radius=100, max_samples=24, min_samples=4)).fit(
    samples.coords, samples["V"]
)
estimate, variance = kriging.predict(grid, return_variance=True)
grid = grid.with_column("estimate", estimate).with_column("variance", variance)
rich = grid.mask(grid["estimate"] > 500)
print(grid)
print(rich)
```

</details>

```text
BlockModel(regular, 78000 of 78000 cells, count [260, 300, 1], size [1.0, 1.0, 1.0], rotation [0.0, 0.0, 0.0])
  estimate: Float64
  variance: Float64
BlockModel(masked, 10400 of 78000 cells, count [260, 300, 1], size [1.0, 1.0, 1.0], rotation [0.0, 0.0, 0.0])
  estimate: Float64
  variance: Float64
```

Only the cells above 500 ppm are kept in the masked model; its cell index is stored with the attributes.

<details><summary>Python</summary>

```python
folder = Path(tempfile.mkdtemp())
cs.write_parquet(folder / "grid.parquet", grid)
cs.write_parquet(folder / "rich.parquet", rich)
cs.write_csv(folder / "grid.csv", grid)
for name in ("grid.parquet", "rich.parquet", "grid.csv"):
    print(f"{name:>13}: {(folder / name).stat().st_size / 1e6:.2f} MB")

back = cs.read_parquet(folder / "rich.parquet")
print(back)
print("same cells:", np.array_equal(back.index, rich.index), "| crs:", back.crs)
```

</details>

```text
 grid.parquet: 1.47 MB
 rich.parquet: 0.22 MB
     grid.csv: 4.03 MB
BlockModel(masked, 10400 of 78000 cells, count [260, 300, 1], size [1.0, 1.0, 1.0], rotation [0.0, 0.0, 0.0])
  estimate: Float64
  variance: Float64
same cells: True | crs: local grid
```

Grouping and summaries belong to polars, which reads the same file. A regular model stores no coordinates: its
geometry is implicit and the row number is the cell index, x fastest, so a 1 m row of 260 cells is one northing.

<details><summary>Python</summary>

```python
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
```

</details>

```text
shape: (6, 3)
┌───────────────┬────────┬─────────────┐
│ northing band ┆ mean V ┆ share > 500 │
│ ---           ┆ ---    ┆ ---         │
│ i64           ┆ f64    ┆ f64         │
╞═══════════════╪════════╪═════════════╡
│ 0             ┆ 319.0  ┆ 0.1         │
│ 50            ┆ 366.0  ┆ 0.171       │
│ 100           ┆ 345.0  ┆ 0.184       │
│ 150           ┆ 307.0  ┆ 0.168       │
│ 200           ┆ 263.0  ┆ 0.119       │
│ 250           ┆ 171.0  ┆ 0.058       │
└───────────────┴────────┴─────────────┘
```

A fitted estimator is saved the same way: its samples become columns and its variogram, search and options JSON
in the file metadata. The estimator read back predicts exactly the same values. `ImplicitModel` and
`LocalAnisotropy` save the same way.

<details><summary>Python</summary>

```python
kriging.to_parquet(folder / "kriging.parquet")
print(pl.read_parquet(folder / "kriging.parquet").head(3))
again = cs.OrdinaryKriging.from_parquet(folder / "kriging.parquet")
print("same estimates:", np.array_equal(again.predict(grid), estimate, equal_nan=True))
```

</details>

```text
shape: (3, 7)
┌──────┬──────┬─────┬───────┬──────┬────────────────┬────────┐
│ x    ┆ y    ┆ z   ┆ value ┆ hole ┆ error_variance ┆ domain │
│ ---  ┆ ---  ┆ --- ┆ ---   ┆ ---  ┆ ---            ┆ ---    │
│ f64  ┆ f64  ┆ f64 ┆ f64   ┆ f64  ┆ f64            ┆ f64    │
╞══════╪══════╪═════╪═══════╪══════╪════════════════╪════════╡
│ 11.0 ┆ 8.0  ┆ 0.0 ┆ 0.0   ┆ null ┆ 0.0            ┆ null   │
│ 8.0  ┆ 30.0 ┆ 0.0 ┆ 0.0   ┆ null ┆ 0.0            ┆ null   │
│ 9.0  ┆ 48.0 ┆ 0.0 ┆ 224.4 ┆ null ┆ 0.0            ┆ null   │
└──────┴──────┴─────┴───────┴──────┴────────────────┴────────┘
same estimates: True
```

GIS software exchanges points as shapefiles. `write_shapefile` stores the coordinates, the attributes (names of at
most 10 characters) and the CRS in a `.prj`; `read_shapefile` returns the same PointSet.

<details><summary>Python</summary>

```python
cs.write_shapefile(folder / "samples.shp", samples)
print(sorted(p.name for p in folder.glob("samples.*")))
again = cs.read_shapefile(folder / "samples.shp")
print("same points:", np.array_equal(again.coords, samples.coords), "| crs:", again.crs)
```

</details>

```text
['samples.cpg', 'samples.dbf', 'samples.prj', 'samples.shp', 'samples.shx']
same points: True | crs: local grid
```

Lines and polygons are `Polylines`: one attribute row per feature, each feature made of parts. A closed part is a
ring whose last vertex joins the first; a ring inside another ring of the same feature is a hole. A pit outline with
an unmined core and a section line go to separate files, polygons and lines, and come back the same. In Parquet a
feature is one row whose `geometry` lists its parts, each a list of vertices; `Polylines.from_table` builds features
from a long table with one row per vertex.

<details><summary>Python</summary>

```python
pit = [[60, 80], [60, 240], [200, 240], [200, 80]]
core = [[110, 140], [150, 140], [150, 190], [110, 190]]
pits = cs.Polylines([pit, core], closed=True, features=[0, 0], attributes={"name": ["pit"]}, crs="local grid")
lines = cs.Polylines([[[20, 160], [240, 160]]], attributes={"name": ["A-A'"]}, crs="local grid")
cs.write_shapefile(folder / "pit.shp", pits)
cs.write_shapefile(folder / "section.shp", lines)
pits, lines = cs.read_shapefile(folder / "pit.shp"), cs.read_shapefile(folder / "section.shp")
print(pits)
print(lines)

cs.write_parquet(folder / "pit.parquet", pits)
print(pl.read_parquet(folder / "pit.parquet"))
vertices = pl.DataFrame({"ID": ["A-A'", "A-A'"], "X": [20.0, 240.0], "Y": [160.0, 160.0]})
print("same section:", np.array_equal(cs.Polylines.from_table(vertices).vertices, lines.vertices))

fig, ax = plt.subplots(figsize=(5, 5.2), layout="constrained")
ax.scatter(*samples.coords[:, :2].T, s=6, color=LIGHT)
for part, closed in zip(pits.parts, pits.closed):
    ring = np.vstack([part, part[:1]]) if closed else part
    ax.plot(ring[:, 0], ring[:, 1], color=ACCENT, lw=1.6)
for part in lines.parts:
    ax.plot(part[:, 0], part[:, 1], color=HIGHLIGHT, lw=1.6, ls="--")
map_axes(ax, "Pit outline with its core, and section A-A'")
save(fig, "polylines")
```

</details>

```text
Polylines(1 features, 2 parts, crs: local grid)
  name: Utf8
Polylines(1 features, 1 parts, crs: local grid)
  name: Utf8
shape: (1, 3)
┌──────┬─────────────────────────────────┬──────────────┐
│ name ┆ geometry                        ┆ closed       │
│ ---  ┆ ---                             ┆ ---          │
│ str  ┆ list[list[struct[3]]]           ┆ list[bool]   │
╞══════╪═════════════════════════════════╪══════════════╡
│ pit  ┆ [[{60.0,80.0,0.0}, {60.0,240.0… ┆ [true, true] │
└──────┴─────────────────────────────────┴──────────────┘
same section: True
```

![polylines](polylines.png)

Rasters travel as GeoTIFF. `write_geotiff` writes each column of a 2D grid as a band, nulls as the `nodata` value
and the CRS in the GeoKeys, so GIS software opens it as a georeferenced raster; `read_geotiff` returns the same
grid. Rotated grids are supported, and a masked model is written with nodata in its absent cells.

<details><summary>Python</summary>

```python
cs.write_geotiff(folder / "grid.tif", grid)
raster = cs.read_geotiff(folder / "grid.tif")
print(raster)
print(f"{(folder / 'grid.tif').stat().st_size / 1e6:.2f} MB")
print(
    "same grid:", raster.origin == grid.origin, np.array_equal(raster["estimate"], estimate, equal_nan=True)
)
```

</details>

```text
BlockModel(regular, 78000 of 78000 cells, count [260, 300, 1], size [1.0, 1.0, 1.0], rotation [0.0, 0.0, 0.0])
  estimate: Float64
  variance: Float64
1.18 MB
same grid: True True
```

Full script: [`example_57.py`](example_57.py)
