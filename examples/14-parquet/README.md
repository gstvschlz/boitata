# 14. Storing containers in Parquet

`write_parquet` saves a `PointSet` or `BlockModel` with its geometry, layout and CRS in the file metadata;
`read_parquet` returns the same container. The file is ordinary Parquet, so polars, pandas or DuckDB read it as a
table.

<details><summary>Python</summary>

```python
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
in the file metadata. The estimator read back predicts exactly the same values.

<details><summary>Python</summary>

```python
kriging.to_parquet(folder / "kriging.parquet")
print(pl.read_parquet(folder / "kriging.parquet").head(3))
again = cs.OrdinaryKriging.from_parquet(folder / "kriging.parquet")
print("same estimates:", np.array_equal(again.predict(grid), estimate, equal_nan=True))
```

</details>

```text
shape: (3, 6)
┌──────┬──────┬─────┬───────┬──────┬────────────────┐
│ x    ┆ y    ┆ z   ┆ value ┆ hole ┆ error_variance │
│ ---  ┆ ---  ┆ --- ┆ ---   ┆ ---  ┆ ---            │
│ f64  ┆ f64  ┆ f64 ┆ f64   ┆ f64  ┆ f64            │
╞══════╪══════╪═════╪═══════╪══════╪════════════════╡
│ 11.0 ┆ 8.0  ┆ 0.0 ┆ 0.0   ┆ null ┆ 0.0            │
│ 8.0  ┆ 30.0 ┆ 0.0 ┆ 0.0   ┆ null ┆ 0.0            │
│ 9.0  ┆ 48.0 ┆ 0.0 ┆ 224.4 ┆ null ┆ 0.0            │
└──────┴──────┴─────┴───────┴──────┴────────────────┘
same estimates: True
```

Full script: [`example_14.py`](example_14.py)
