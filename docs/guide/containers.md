# Containers and Arrow

A container holds a table of columns plus the geometry that places each row in space. Every container stores its columns in the Arrow format, so Polars, pandas and PyArrow read them without a conversion step of your own.

## Six containers

<div class="bt-compare" markdown>

| Container | One row is | Geometry |
| --- | --- | --- |
| `Table` | A record | None |
| `PointSet` | A sample | x, y, z per row |
| `Drillholes` | A hole and its intervals | Collar, survey and interval tables |
| `BlockModel` | A block | Origin, block size, block counts, rotation |
| `Mesh` | A vertex or a triangle | Vertices and triangles |
| `Polylines` | A line or polygon feature | Parts made of vertices |

</div>

Every container answers the same calls: `len(data)`, `data["V"]` for one column as a NumPy array, `column_names`, `filter(mask)` for the rows where `mask` is true, `with_column` and `with_columns` to add columns, and `to_table()` for the whole table. The ones with geometry also give `coords` as an `(n, 3)` array, `x`, `y` and `z` for one axis, and `bounds` for the `(min, max)` corners. `PointSet`, `BlockModel`, `Polylines` and `Drillholes` convert with `to_polars()`, `to_pandas()` and `to_pyarrow()`.

A `Mesh` keeps one table for its vertices and one for its triangles (`with_vertex_column`, `with_face_column`); `mesh["V"]` looks in the vertex columns first and raises when a name is in both. `Drillholes` counts holes with `len`, but its `coords` and columns are the interval midpoints and interval columns, the same as `samples()`. A misspelled column name raises `MissingColumn` with the closest name it found.

## Columns and missing values

A column is numeric (stored as 64-bit floats), text or boolean. A missing value is an Arrow null, whatever the file used to mark it. Readers map the usual sentinels to null: `-99`, `-999`, `1e21`, text such as `NA` or `NULL`, and empty cells. Take a small CSV with one missing assay and one missing rock code:

```text
X,Y,Au,ROCK
0,0,1.2,oxide
10,0,-999,fresh
0,10,0.4,
```

```python
import boitata as bt
import numpy as np

table = bt.read_csv("samples.csv")
print(table["Au"])
print(table["ROCK"])
print(table.to_polars())
```

```text
[1.2 nan 0.4]
['oxide' 'fresh' None]
shape: (3, 4)
┌──────┬──────┬──────┬───────┐
│ X    ┆ Y    ┆ Au   ┆ ROCK  │
│ ---  ┆ ---  ┆ ---  ┆ ---   │
│ f64  ┆ f64  ┆ f64  ┆ str   │
╞══════╪══════╪══════╪═══════╡
│ 0.0  ┆ 0.0  ┆ 1.2  ┆ oxide │
│ 10.0 ┆ 0.0  ┆ null ┆ fresh │
│ 0.0  ┆ 10.0 ┆ 0.4  ┆ null  │
└──────┴──────┴──────┴───────┘
```

The `-999` became a null: `NaN` in the NumPy array, `null` in Polars. A missing text value comes out as `None`. A `NaN` you put into a container with `with_column` becomes a null too.

If a file marks missing values some other way, pass `nodata=`. It replaces the default list, so `nodata=[-1]` reads `-1` as null and keeps `-999` as a number.

## Units

A column can carry a unit. `read_csv` reads one from a header such as `Au [g/t]` or `Au (g/t)`, `units=` gives one to a column the file leaves bare, and `bt.set_units(columns={"Au": "g/t"})` declares them once for a project. `with_units` sets them by hand, and `convert_units` rescales a column within its kind. Grades, ratios, metal, ore mass and each currency are separate kinds, so converting `%` to `ratio%` or `kg metal` to `t` raises `InvalidInput`. Coordinates carry a `length_unit` too, and `to_length_unit` converts them.

A parameter takes a number in the unit of the data, or text such as `"150 ft"` or `"0.5 g/t"` converted to it: the `Search` radius, `Variogram` ranges, `experimental_variogram` lags, `composite` lengths, `runs` lengths, block sizes of `from_extents`, `swath` widths, declustering cells, cutoffs of `grade_tonnage`, `compare_models`, `runs`, `spatial_bootstrap`, indicator and multigaussian kriging and the simulations, `Capping` caps, the `IndicatorKriging` threshold and densities such as `"2.7 t/m3"`. The [units example](../examples/02-data-and-geometry/25-units/example_02_25.md) walks through them.

!!! pitfall "Pitfall"
    Statistics such as `describe` skip nulls, but estimators, transforms and variograms refuse them: `fit` raises `InvalidInput: values must be finite; drop missing values first`. Keep the rows that have a value before you fit:

    ```python
    assayed = table.filter(~np.isnan(table["Au"]))
    ```

    boitatá leaves the choice to you, because dropping rows without telling you would hide a data problem.

## Coordinates are 3D

A `PointSet` always stores three coordinates. Give it `(n, 2)` coordinates, or a table with no `Z` column, and it sets `z = 0`:

```python
samples = bt.PointSet.from_table(table)
print(samples.coords)
```

```text
[[ 0.  0.  0.]
 [10.  0.  0.]
 [ 0. 10.  0.]]
```

`from_table` looks for columns named `X`, `Y` and `Z`; pass `x=`, `y=` and `z=` when yours are named `EAST`, `NORTH` and `RL`. A 2D grid is a `BlockModel` with one block in z.

## Adding columns

Containers do not change in place. `with_column` returns a new container with the extra column, so assign the result:

```python
samples = samples.with_column("Au_log", np.log(samples["Au"]))
print(samples)
```

```text
PointSet(3 points, crs: none)
  Au: Float64
  ROCK: Utf8
  Au_log: Float64
```

`with_columns` adds several at once from a dict or a data frame.

## A column name stands for a column

An argument that takes one value per row also takes a column name, as long as the call can see the container. Functions that take the values first get the container through `data=`; estimators read it from the samples you fit them on:

```python
bt.describe("Au", data=samples)               # same as bt.describe(samples["Au"])
kriging.fit(samples, "Au", holes="HOLE_ID", domain_column="ZONE")  # values, holes, domains by name
kriging.predict(grid, domain_column="ZONE")                        # target domains by name
```

## To Polars, pandas and NumPy

Containers expose the Arrow C stream interface, so libraries that read Arrow take them directly. `to_table()` puts the coordinates in as `x`, `y` and `z` columns.

```python
import polars as pl

frame = pl.DataFrame(samples)                  # x, y, z, Au, ROCK, Au_log
frame = samples.to_polars()                    # the same
frame = samples.to_pandas()                    # a pandas DataFrame
values = samples["Au"]                         # one column as a NumPy array
```

Grouping, joining and pivoting belong to Polars or pandas; boitatá keeps to the spatial work.

## Block models

A block model is a grid of boxes. Four things fix every block: the `origin` (the outer corner of the first block), the block `size` along each axis, the `count` of blocks along each axis, and a `rotation`. A regular grid stores no coordinates; the block centers follow from those numbers.

<figure class="bt-figure">
--8<-- "svg/g-block-grid.svg"
<figcaption><b>Figure 1.</b> <code>BlockModel(origin=(0, 0, 0), size=(10, 10, 5), count=(4, 3, 1), rotation=(30, 0, 0))</code>. The grid turns 30° clockwise about its origin corner. Rows run along x first, then y, then z, so the row number of a block is also its position in the grid.</figcaption>
</figure>

```python
model = bt.BlockModel(origin=(0, 0, 0), size=(10, 10, 5), count=(4, 3, 1), rotation=(30, 0, 0))
print(model)
print(model.coords[:2].round(2))
```

```text
BlockModel(regular, 12 of 12 cells, count [4, 3, 1], size [10.0, 10.0, 5.0], rotation [30.0, 0.0, 0.0])
[[ 6.83  1.83  2.5 ]
 [15.49 -3.17  2.5 ]]
```

Block 0 has its center 5 m along x and 5 m along y from the origin, turned 30° clockwise: (6.83, 1.83). Block 1 is the next one along the rotated x axis.

The rotation is `(azimuth, dip, rake)` in degrees, the same triple the variogram and the search use. The grid also carries a CRS string, which Parquet files keep. Containers with different declared CRSs refuse to meet; an `EPSG:` code matches WKT whose top-level ID names it.

### Three layouts

A model does not have to store every block. The same class, columns and calls work in three layouts:

<div class="bt-compare" markdown>

| Layout | Stores | Use it for | Made with |
| --- | --- | --- | --- |
| Regular | Every block, no index | Estimation grids | `BlockModel(...)`, `from_extents` |
| Masked | A subset of blocks, with their sorted block index | Blocks inside a domain or above a cutoff | `model.filter(keep)` |
| Sub-blocked | Smaller blocks inside some parents, with parent index and extents | Thin or irregular solids | `model.subblock(...)`, `BlockModel.from_meshes` |

</div>

```python
rich = model.filter(model.x > 10)
print(rich)
print(rich.index)
```

```text
BlockModel(masked, 11 of 12 cells, count [4, 3, 1], size [10.0, 10.0, 5.0], rotation [30.0, 0.0, 0.0])
[ 1  2  3  4  5  6  7  8  9 10 11]
```

Block 0 has its center at x = 6.83, so the mask drops it and keeps the other 11; `index` lists the block numbers kept. You convert between layouts explicitly: `to_regular()` puts a masked model back on the full grid, and `regularize(target)` averages one model onto the blocks of another, such as sub-blocks onto their parents.

The examples go further with [block models](../examples/01-first-steps/02-block-models/example_01_02.md), [sub-blocks](../examples/02-data-and-geometry/06-sub-blocks/example_02_06.md), [models larger than memory](../examples/02-data-and-geometry/10-large-models/example_02_10.md) and [storing containers in Parquet](../examples/01-first-steps/04-parquet/example_01_04.md); [saving models](saving.md) and the [containers and I/O API](../api/containers/index.md) cover the rest.
