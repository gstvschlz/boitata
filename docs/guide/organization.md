# how boitatá is organized

boitatá has five kinds of object, and each kind takes the same calls wherever it appears. once you know which kind you hold, you can guess most of what it accepts. you need basic python and numpy, and no geostatistics.

## the map

<figure class="bt-figure">
--8<-- "svg/g-api-map.svg"
<figcaption><b>figure 1.</b> data enters through loaders, lives in containers, passes through functions, value objects and estimators, and leaves as plots and files. <code>with_column</code> puts results back onto a container. the rust core does the arithmetic for containers, functions and estimators.</figcaption>
</figure>

one short run touches every layer:

```python
import boitata as bt

samples = bt.datasets.walker_lake()                                  # 1. get data: a PointSet
experimental = bt.experimental_variogram(samples, "V", 10.0, 120.0)  # 3. a function
variogram = experimental.fit("spherical")                            # 3. a value object
search = bt.Search(radius=60, max_samples=24, min_samples=4)         # 3. a value object
grid = bt.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(52, 60))  # 2. a container
kriging = bt.OrdinaryKriging(variogram, search).fit(samples, "V")    # 3. an estimator
grid = grid.with_column("V", kriging.predict(grid))                  # results back on the container
fig, ax = bt.plot.histogram("V", data=grid)                          # 4. a plot
```

```text
PointSet(470 points, crs: none)
  ID: Float64
  V: Float64
  U: Float64
  T: Float64
BlockModel(regular, 3120 of 3120 cells, count [52, 60, 1], size [5.0, 5.0, 1.0], rotation [0.0, 0.0, 0.0])
  V: Float64
```

## five kinds of object

**containers hold data.** `Table`, `PointSet`, `Drillholes`, `BlockModel`, `Mesh` and `Polylines` store columns plus the geometry that places each row: coordinates for points, origin and block size for a grid, vertices and triangles for a mesh. [containers and arrow](containers.md) covers them.

**value objects describe a model.** a `Variogram`, a `Structure`, a `Search`, a `Categories` scheme or a `HighGrade` restriction holds parameters and no data. you build one, read its properties, pass it to an estimator and save it as JSON. a value object never changes after you build it.

**estimators and transforms learn from data.** `NormalScore`, `Capping`, `OrdinaryKriging`, `SGS` and their relatives share one lifecycle: build with options, `fit` to samples, then `transform` or `predict`. [fit, transform, predict](estimators.md) walks through it.

**functions compute a result in one call.** exploratory statistics (`describe`, `describe_by`, `swath`, `contact`, `grade_tonnage`) and experimental variograms (`experimental_variogram`, `variogram_map`) take data and return numbers, a `Table` or an `ExperimentalVariogram`. they fit and remember nothing.

**modules draw and fetch.** `bt.plot` draws on matplotlib and returns `fig, ax`; every plot accepts `ax=` to draw into your own axes. `bt.datasets` downloads the teaching datasets once and caches them.

<div class="bt-compare" markdown>

| kind | holds | you call | saves as |
| --- | --- | --- | --- |
| container | columns and geometry | `with_column`, `filter`, `mask`, `["V"]` | parquet |
| value object | parameters | properties, `to_json` | JSON |
| transform | what `fit` learned | `fit`, `transform`, `inverse_transform` | JSON |
| estimator | the fitted samples | `fit`, `predict`, `cross_validate` | parquet |
| function | nothing | one call | its result |

</div>

!!! key "key idea"
    ask what kind of object you hold. a container answers `with_column`, a value object answers `to_json`, an estimator answers `fit` and `predict`, and a function answers once and keeps nothing.

## guessing a call

the names follow a few rules, so you can often write a call you have not seen before.

- **full names.** `OrdinaryKriging`, `NormalScore`, `experimental_variogram`. only standard acronyms stay short: SGS, SIS, PCA, MAF, CRS.
- **the same argument names everywhere.** `coords` for locations, `values` for the variable, `targets` for where to estimate, `weights`, `categories`, `domains` or `domain_column`, `seed`, `search`, `variogram`. ranges come as `(major, semi-major, minor)`; a `rotation` is one `(azimuth, dip, rake)` triple in degrees, with azimuth clockwise from north and dip positive down.
- **a column name stands for a column.** where a function takes one value per row, you can pass the column name and the container: `kriging.fit(samples, "V")`, or `bt.describe("V", data=samples)` for calls that take the values first.
- **options are keyword-only.** you must name every argument that has a default, so `bt.Search(60, 24)` fails and `bt.Search(radius=60, max_samples=24)` works:

```python
bt.Search(60, 24)
```

```text
TypeError: Search.__new__() takes 1 positional arguments but 2 were given
```

- **methods return new objects.** `with_column`, `filter` and `mask` return a new container and leave the old one as it was, so assign the result: `grid = grid.with_column(...)`.
- **american spelling**, `z` up, right-handed axes, 3D internally: points given as `(n, 2)` get `z = 0`.

## errors

the errors boitatá raises itself derive from `bt.BoitataError` and from the builtin exception you would expect, so you can catch either. python's own checks, such as a missing keyword, raise a plain `TypeError`.

<div class="bt-compare" markdown>

| error | also a | raised when |
| --- | --- | --- |
| `InvalidInput` | `ValueError` | an argument is out of range, or a file holds another object |
| `MissingColumn` | `KeyError` | the container has no column of that name |
| `FileError` | `OSError` | a file cannot be read or written |

</div>

```python
try:
    samples["Au"]
except KeyError as error:
    print(type(error).__name__, error)
```

```text
MissingColumn no column "Au"; columns: ID, V, U, T
```

the message lists the columns the container has, so you can spot a typo.

!!! pitfall "pitfall"
    readers turn sentinels such as `-999` into nulls, and statistics such as `describe` skip them. estimators, transforms and variograms do not: a missing value there raises `InvalidInput: values must be finite; drop missing values first`. drop the rows with `filter` before `fit`; [containers and arrow](containers.md) shows how.

## rust underneath

the numerical work runs in rust, in double precision. kriging runs in parallel over targets and simulation over realizations. each realization draws from a seed derived from yours, so a run gives the same numbers with one thread or many. the python layer checks the arguments and hands the work down; you never call the rust layer yourself.

## API map

<div class="grid cards bt-api-map" markdown>

-   **[containers and I/O](../api/containers/index.md)**

    ---

    `Table`, `PointSet`, `BlockModel`, `Mesh`, `Polylines`, `read_csv`, `read_shapefile`, `read_geotiff`

-   **[drill holes and blocks](../api/drillholes/index.md)**

    ---

    `Drillholes`, `check_drillholes`, `merge_intervals`, `assign_domain`, `block_shell`

-   **[datasets and plots](../api/datasets/index.md)**

    ---

    `bt.datasets`, `bt.plot`, `bt.plot3d`

-   **[EDA and validation](../api/eda/index.md)**

    ---

    `describe`, `describe_by`, `swath`, `contact`, `grade_tonnage`, `Categories`

-   **[transforms](../api/transforms/index.md)**

    ---

    `NormalScore`, `Capping`, `HermiteAnamorphosis`, `PPMT`, `PCA`, `MAF`

-   **[variography](../api/variography/index.md)**

    ---

    `Variogram`, `Structure`, `experimental_variogram`, `variogram_map`

-   **[estimation](../api/estimation/index.md)**

    ---

    `Search`, `OrdinaryKriging`, `SimpleKriging`, `BlockKriging`, `IndicatorKriging`

-   **[simulation](../api/simulation/index.md)**

    ---

    `SGS`, `TurningBands`, `SIS`, `Plurigaussian`, `SNESIM`

-   **[modeling](../api/modeling/index.md)**

    ---

    `ImplicitModel`: surfaces and solids from contacts and orientations

</div>
