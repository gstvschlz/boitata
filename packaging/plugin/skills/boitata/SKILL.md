---
name: boitata
description: How to write correct code with boitatá (`import boitata as bt`), the Rust-core Python geostatistics library - drill holes, compositing, block models, declustering, transforms, variograms, kriging, simulation, recoverable resources and model checks. Use whenever code imports `boitata`, or the user asks for geostatistics or resource-estimation work in Python and boitatá is installed or wanted.
---

# boitatá

boitatá is imported as `bt`. Install it with `pip install "boitata[all]"` (Python ≥ 3.11).

```python
import boitata as bt
```

## Look it up before you write it

Do not write a boitatá call from memory. The installed package carries its own reference, and it matches the
installed version exactly:

```sh
python -c "import boitata, pathlib; print(boitata.__version__, pathlib.Path(boitata.__file__).with_name('_boitata.pyi'))"
```

- Search that `_boitata.pyi` stub for the class or function to get its exact signature.
- `python -c "import boitata as bt; help(bt.OrdinaryKriging.fit)"` prints the NumPy-style docstring, with every
  argument, its unit and its default.
- `bt.__all__` lists every public name; `dir(bt.datasets)` and `dir(bt.plot)` list the datasets and the plots.
- The docs at https://gstvschlz.github.io/boitata/ hold a worked example for most tasks. The page
  https://gstvschlz.github.io/boitata/guide/finder/ maps a question to the example that answers it.

If a name is not in `bt.__all__`, it does not exist. Say so and offer the closest one that does; do not invent it.

## Kinds of object

Ask which kind you hold, and the calls follow.

| Kind | Examples | You call | Saves as |
| --- | --- | --- | --- |
| Container | `Table`, `PointSet`, `Drillholes`, `BlockModel`, `Mesh`, `Polylines` | `with_column`, `filter`, `["V"]`, `coords` | Parquet |
| Value object | `Variogram`, `Structure`, `Search`, `Categories`, `HighGrade` | properties, `to_json` | JSON |
| Transform | `NormalScore`, `Capping`, `PPMT`, `PCA`, `MAF` | `fit`, `transform`, `inverse_transform` | JSON |
| Estimator | `OrdinaryKriging`, `BlockKriging`, `InverseDistance` | `fit`, `predict`, `cross_validate` | Parquet |
| Simulator | `SGS`, `DSS`, `SIS`, `TurningBands` | `fit`, `simulate` | Parquet |
| Function | `describe`, `swath`, `experimental_variogram` | one call | its result |

One run through every kind:

```python
samples = bt.datasets.clustered_metal_survey()
experimental = bt.experimental_variogram(samples, "V", 10.0, 120.0)
variogram = experimental.fit("spherical")
search = bt.Search(radius=60, max_samples=24, min_samples=4)
grid = bt.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(52, 60))
kriging = bt.OrdinaryKriging(variogram, search).fit(samples, "V")
estimate, variance = kriging.predict(grid, return_variance=True)
grid = grid.with_column("V", estimate).with_column("V_variance", variance)
check = kriging.cross_validate()
fig, ax = bt.plot.histogram("V", data=grid)
```

## Rules that hold everywhere

- **Options go to the constructor, data to `fit`, targets to `predict`, `transform` or `simulate`.** `fit` returns
  the object, so calls chain.
- **Options are keyword-only.** `bt.Search(60, 24)` raises `TypeError`; write `bt.Search(radius=60, max_samples=24)`.
- **Containers never change in place.** `with_column`, `with_columns` and `filter` return a new container, so assign
  the result: `grid = grid.with_column(...)`.
- **A column name stands for a column** wherever the call can see the container: `kriging.fit(samples, "V")`,
  `bt.describe("V", data=samples)`.
- **Results are NumPy arrays in target order.** For a block model that is the row order, so the array goes straight
  into `with_column`. A target the search could not fill is `NaN`. `predict(..., diagnostics=True)` returns a
  `Table` with slope, efficiency and sample counts per target.
- **Geometry is 3D.** `(n, 2)` coordinates get `z = 0`; a 2D grid is a `BlockModel` with one block in z. Axes are
  right-handed with z up.
- **Ranges are `(major, semi-major, minor)`; a rotation is `(azimuth, dip, rake)` in degrees**, azimuth clockwise
  from north, dip positive down. The variogram, the search and the block model share that triple.
- **A `BlockModel` origin is the outer corner of the first block**, not its center. Rows run along x, then y, then z.
- **Names are spelled out**, in American spelling: `OrdinaryKriging`, `NormalScore`, `experimental_variogram`. Only
  SGS, DSS, SIS, PCA, MAF and CRS stay short.
- **Simulations take `seed`**, and the same seed gives the same realizations whatever the thread count.
- **These are not scikit-learn estimators.** They take coordinates plus one column of values, have no `get_params`
  and do not drop into a scikit-learn `Pipeline`.

## Missing values

Readers turn sentinels (`-99`, `-999`, `1e21`, `NA`, `NULL`, empty cells) into nulls, which show as `NaN` in NumPy.
Statistics such as `describe` skip them. Estimators, transforms and variograms refuse them with
`InvalidInput: values must be finite; drop missing values first`. Drop the rows yourself, and tell the user how
many went:

```python
assayed = samples.filter(~np.isnan(samples["Au"]))
```

## Errors

boitatá's own errors derive from `bt.BoitataError` and from a builtin: `InvalidInput` is a `ValueError`,
`MissingColumn` a `KeyError`, `FileError` an `OSError`. `MissingColumn` lists the columns the container has. Read
the message before changing the code; it names the argument at fault.

## Saving

Parameters go to JSON: `variogram.to_json()` and `bt.Variogram.from_json(text)`, the same for a fitted transform.
Columns go to Parquet: `bt.write_parquet(path, data)` and `bt.read_parquet(path)` for containers,
`obj.to_parquet(path)` and `Class.from_parquet(path)` for fitted estimators and simulation results. `write_csv`
drops block size, rotation, CRS and layout, so keep working files in Parquet.

## Data

`bt.datasets` generates every dataset from a seed, offline. Use one for a demonstration or a test in place of
inventing numbers. Grouping, joining and pivoting belong to Polars or pandas: `data.to_polars()`,
`data.to_pandas()`.

## Before you report a result

Run the code. Then check what the method promises: `cross_validate()` for a kriging plan, the histogram and
variogram of the realizations for a simulation, a swath or a global mean against the declustered samples for a
block model. Report the numbers you got, in the units of the data.
