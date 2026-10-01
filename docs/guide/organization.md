# How Boitatá is organized

Boitatá has five kinds of object, and each kind takes the same calls wherever it appears. Once you know which kind you hold, you can guess most of what it accepts.

!!! learn "What you'll learn"
    - The five layers of the library, from loading data to the Rust core.
    - How containers, value objects, estimators and functions differ.
    - The naming and argument rules that let you guess a call.
    - Which errors to catch.

    Prerequisites: basic Python and NumPy. No geostatistics needed.

## The map

<figure class="bt-figure">
--8<-- "svg/g-api-map.svg"
<figcaption><b>Figure 1.</b> Data enters through loaders, lives in containers, is worked on by functions, value objects and estimators, and leaves as plots and files. Results go back onto a container with <code>with_column</code>. The Rust core does the arithmetic for containers, functions and estimators.</figcaption>
</figure>

One short run touches every layer:

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

## Five kinds of object

**Containers hold data.** `Table`, `PointSet`, `Drillholes`, `BlockModel`, `Mesh` and `Polylines` store columns plus the geometry that places each row: coordinates for points, origin and block size for a grid, vertices and triangles for a mesh. [Containers and Arrow](containers.md) covers them.

**Value objects describe a model.** A `Variogram`, a `Structure`, a `Search`, a `Categories` scheme or a `HighGrade` restriction holds parameters and no data. You build one, read its properties, pass it to an estimator and save it as JSON. A value object never changes after you build it.

**Estimators and transforms learn from data.** `NormalScore`, `Capping`, `OrdinaryKriging`, `SGS` and their relatives follow one lifecycle: build with options, `fit` to samples, then `transform` or `predict`. [fit, transform, predict](estimators.md) walks through it.

**Functions compute a result in one call.** Exploratory statistics (`describe`, `describe_by`, `swath`, `contact`, `grade_tonnage`) and experimental variograms (`experimental_variogram`, `variogram_map`) take data and return numbers, a `Table` or an `ExperimentalVariogram`. Nothing is fitted or remembered.

**Modules draw and fetch.** `bt.plot` draws on matplotlib and returns `fig, ax`; every plot accepts `ax=` to draw into your own axes. `bt.datasets` downloads the teaching datasets once and caches them.

<div class="bt-compare" markdown>

| Kind | Holds | You call | Saves as |
| --- | --- | --- | --- |
| Container | columns and geometry | `with_column`, `filter`, `mask`, `["V"]` | Parquet |
| Value object | parameters | properties, `to_json` | JSON |
| Transform | what `fit` learned | `fit`, `transform`, `inverse_transform` | JSON |
| Estimator | the fitted samples | `fit`, `predict`, `cross_validate` | Parquet |
| Function | nothing | one call | its result |

</div>

!!! key "Key idea"
    Ask what kind of object you hold. A container answers `with_column`, a value object answers `to_json`, an estimator answers `fit` and `predict`, and a function answers once and keeps nothing.

## Guessing a call

The names follow a few rules, so you can often write a call you have not seen before.

- **Full names.** `OrdinaryKriging`, `NormalScore`, `experimental_variogram`. Only standard acronyms stay short: SGS, SIS, PCA, MAF, CRS.
- **The same argument names everywhere.** `coords` for locations, `values` for the variable, `targets` for where to estimate, `weights`, `categories`, `domains` or `domain_column`, `seed`, `search`, `variogram`. Ranges come as `(major, semi-major, minor)`; a `rotation` is one `(azimuth, dip, rake)` triple in degrees, with azimuth clockwise from north and dip positive down.
- **A column name stands for a column.** Where a function takes one value per row, you can pass the column name and the container: `kriging.fit(samples, "V")`, or `bt.describe("V", data=samples)` for calls that take the values first.
- **Options are keyword-only.** Every argument with a default must be named, so `bt.Search(60, 24)` fails and `bt.Search(radius=60, max_samples=24)` works:

```python
bt.Search(60, 24)
```

```text
TypeError: Search.__new__() takes 1 positional arguments but 2 were given
```

- **Methods return new objects.** `with_column`, `filter` and `mask` return a new container and leave the old one as it was, so assign the result: `grid = grid.with_column(...)`.
- **American spelling**, `z` up, right-handed axes, 3D internally: points given as `(n, 2)` get `z = 0`.

## Errors

Errors Boitatá raises itself derive from `bt.BoitataError` and also from the builtin exception you would expect, so you can catch either. Python's own checks, such as a missing keyword, raise plain `TypeError`.

<div class="bt-compare" markdown>

| Error | Also a | Raised when |
| --- | --- | --- |
| `InvalidInput` | `ValueError` | an argument is out of range, or a file holds another object |
| `MissingColumn` | `KeyError` | a column name is not in the container |
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

The message lists the columns the container has, so you can spot a typo.

!!! pitfall "Pitfall"
    Readers turn sentinels such as `-999` into nulls, and statistics such as `describe` skip them. Estimators, transforms and variograms do not: a missing value there raises `InvalidInput: values must be finite; drop missing values first`. Drop the rows with `filter` before `fit`. [Containers and Arrow](containers.md) shows how.

## Rust underneath

The numerical work runs in Rust, in double precision. Kriging runs in parallel over targets and simulation over realizations, and each realization draws from a seed derived from yours, so a run gives the same numbers with one thread or many. The Python layer checks the arguments and hands the work down. You never call the Rust layer yourself.

## API map

<div class="grid cards bt-api-map" markdown>

-   **[Containers and I/O](../api/containers/index.md)**

    ---

    `Table`, `PointSet`, `BlockModel`, `Mesh`, `Polylines`, `read_csv`, `read_shapefile`, `read_geotiff`

-   **[Drill holes and blocks](../api/drillholes/index.md)**

    ---

    `Drillholes`, `check_drillholes`, `merge_intervals`, `assign_domain`, `block_shell`

-   **[Datasets and plots](../api/datasets/index.md)**

    ---

    `bt.datasets`, `bt.plot`, `bt.plot3d`

-   **[EDA and validation](../api/eda/index.md)**

    ---

    `describe`, `describe_by`, `swath`, `contact`, `grade_tonnage`, `Categories`

-   **[Transforms](../api/transforms/index.md)**

    ---

    `NormalScore`, `Capping`, `HermiteAnamorphosis`, `PPMT`, `PCA`, `MAF`

-   **[Variography](../api/variography/index.md)**

    ---

    `Variogram`, `Structure`, `experimental_variogram`, `variogram_map`

-   **[Estimation](../api/estimation/index.md)**

    ---

    `Search`, `OrdinaryKriging`, `SimpleKriging`, `BlockKriging`, `IndicatorKriging`

-   **[Simulation](../api/simulation/index.md)**

    ---

    `SGS`, `TurningBands`, `SIS`, `Plurigaussian`, `SNESIM`

-   **[Modeling](../api/modeling/index.md)**

    ---

    `ImplicitModel`: surfaces and solids from contacts and orientations

</div>

!!! seealso "See also"
    - [Containers and Arrow](containers.md): columns, nulls and block model geometry.
    - [fit, transform, predict](estimators.md): the estimator lifecycle.
    - [Saving models](saving.md): JSON and Parquet.
    - [Which page do I need?](finder.md): from a question to a page.
