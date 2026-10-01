# Containers and Arrow

A container holds a table of columns plus the geometry that places each row in space. Every container stores its columns in the Arrow format, and polars, pandas and pyarrow read them without a conversion step of your own.

!!! learn "What you'll learn"
    - How a container stores columns and what a missing value looks like.
    - How to move data to polars, pandas and NumPy.
    - How a block model describes its grid with a handful of numbers.

    Prerequisites: [How Boitatá is organized](organization.md).

## Six containers

<div class="bt-compare" markdown>

| Container | One row is | Geometry |
| --- | --- | --- |
| `Table` | a record | none |
| `PointSet` | a sample | x, y, z per row |
| `Drillholes` | a hole and its intervals | collar, survey and interval tables |
| `BlockModel` | a block | origin, block size, block counts, rotation |
| `Mesh` | a vertex or a triangle | vertices and triangles |
| `Polylines` | a line or polygon feature | parts made of vertices |

</div>

`Table`, `PointSet`, `BlockModel` and `Polylines` answer the same few calls: `len(data)`, `data["V"]` for one column as a NumPy array, `to_table()` for the whole table, and `with_column` to add a column. A `Mesh` keeps one table for its vertices and one for its triangles (`with_vertex_column`, `with_face_column`). `Drillholes` hands its samples over as a `PointSet` through `samples()` or `composite()`.

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

The `-999` became a null: `NaN` in the NumPy array, `null` in polars. A missing text value comes out as `None`. A `NaN` you put into a container with `with_column` is stored as null too.

If a file marks missing values some other way, pass `nodata=`. It replaces the default list, so `nodata=[-1]` reads `-1` as null and keeps `-999` as a number.

!!! pitfall "Pitfall"
    Statistics such as `describe` skip nulls, but estimators, transforms and variograms refuse them: `fit` raises `InvalidInput: values must be finite; drop missing values first`. Keep the rows that have a value before you fit:

    ```python
    assayed = table.filter(~np.isnan(table["Au"]))
    ```

    Boitatá leaves the choice to you, because dropping rows without telling you would hide a data problem.

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

Any argument that takes one value per row also takes a column name, as long as the call can see the container. Functions that take the values first get the container through `data=`; estimators read it from the samples they are fitted on:

```python
bt.describe("Au", data=samples)               # same as bt.describe(samples["Au"])
kriging.fit(samples, "Au", holes="HOLE_ID", domain_column="ZONE")  # values, holes, domains by name
kriging.predict(grid, domain_column="ZONE")                        # target domains by name
```

## To polars, pandas and NumPy

Containers expose the Arrow C stream interface, so libraries that read Arrow take them directly. `to_table()` puts the coordinates in as `x`, `y` and `z` columns.

```python
import polars as pl

frame = pl.DataFrame(samples)                  # x, y, z, Au, ROCK, Au_log
frame = samples.to_table().to_polars()         # the same
frame = samples.to_table().to_pandas()         # a pandas DataFrame
values = samples["Au"]                         # one column as a NumPy array
```

Grouping, joining and pivoting belong to polars or pandas; Boitatá keeps to the spatial work.

## Block models

A block model is a grid of boxes. Four things fix every block: the `origin` (the outer corner of the first block), the block `size` along each axis, the `count` of blocks along each axis, and a `rotation`. Coordinates are never stored for a regular grid; the block centers follow from those numbers.

<figure class="bt-figure">
--8<-- "svg/g-block-grid.svg"
<figcaption><b>Figure 1.</b> <code>BlockModel(origin=(0, 0, 0), size=(10, 10, 5), count=(4, 3, 1), rotation=(30, 0, 0))</code>. The grid turns 30° clockwise about its origin corner. Rows run along x first, then y, then z, so the row number of a block is also its position in the grid.</figcaption>
</figure>

```python
model = bt.BlockModel(origin=(0, 0, 0), size=(10, 10, 5), count=(4, 3, 1), rotation=(30, 0, 0))
print(model)
print(model.centroids[:2].round(2))
```

```text
BlockModel(regular, 12 of 12 cells, count [4, 3, 1], size [10.0, 10.0, 5.0], rotation [30.0, 0.0, 0.0])
[[ 6.83  1.83  2.5 ]
 [15.49 -3.17  2.5 ]]
```

Block 0 has its center 5 m along x and 5 m along y from the origin, turned 30° clockwise: (6.83, 1.83). Block 1 is the next one along the rotated x axis.

The rotation is `(azimuth, dip, rake)` in degrees, the same triple the variogram and the search use. The grid also carries a CRS string, kept through Parquet files.

### Three layouts

A model does not have to store every block. The same class, columns and calls work in three layouts:

<div class="bt-compare" markdown>

| Layout | Stores | Use it for | Made with |
| --- | --- | --- | --- |
| regular | every block, no index | estimation grids | `BlockModel(...)`, `from_extents` |
| masked | a subset of blocks, with their sorted block index | blocks inside a domain or above a cutoff | `model.mask(keep)` |
| sub-blocked | smaller blocks inside some parents, with parent index and extents | thin or irregular solids | `model.subblock(...)`, `BlockModel.from_meshes` |

</div>

```python
rich = model.mask(model.centroids[:, 0] > 10)
print(rich)
print(rich.index)
```

```text
BlockModel(masked, 11 of 12 cells, count [4, 3, 1], size [10.0, 10.0, 5.0], rotation [30.0, 0.0, 0.0])
[ 1  2  3  4  5  6  7  8  9 10 11]
```

Block 0 has its center at x = 6.83, so the mask drops it and keeps the other 11; `index` lists the block numbers kept. Conversions are always explicit: `to_regular()` puts a masked model back on the full grid, and `regularize(target)` averages one model onto the blocks of another, such as sub-blocks onto their parents.

!!! seealso "See also"
    - [Block models](../examples/01-first-steps/02-block-models/example_01_02.md): building, masking and slicing grids.
    - [Sub-blocks](../examples/02-data-and-geometry/06-sub-blocks/example_02_06.md): splitting blocks at a solid's boundary.
    - [Models larger than memory](../examples/02-data-and-geometry/10-large-models/example_02_10.md): a block model processed chunk by chunk.
    - [Storing containers in Parquet](../examples/01-first-steps/04-parquet/example_01_04.md) and [Saving models](saving.md).
    - [Containers and I/O API](../api/containers/index.md).
