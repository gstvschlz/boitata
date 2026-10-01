# saving models

boitatá saves in two open formats: JSON for objects that hold parameters, parquet for objects that hold columns. both read back to the same object, and a rerun from the saved files reproduces the numbers bit for bit.

## which object saves how

the kind of object sets the format. parameters fit in a short JSON string; columns go to parquet.

<div class="bt-compare" markdown>

| object | format | write | read |
| --- | --- | --- | --- |
| value objects: `Variogram`, `Structure`, `Coregionalization`, `Search`, `HighGrade`, `Categories` | JSON | `obj.to_json()` | `Variogram.from_json(text)` |
| fitted transforms: `NormalScore`, `Capping`, `PPMT`, `PCA`, `MAF`, … and `Declustering`, `Trend` | JSON | `obj.to_json()` | `NormalScore.from_json(text)` |
| containers: `PointSet`, `BlockModel`, `Polylines`, `Table` | parquet | `bt.write_parquet(path, data)` | `bt.read_parquet(path)` |
| fitted estimators and simulators: `OrdinaryKriging`, `SGS`, …, `ImplicitModel`, `LocalAnisotropy` | parquet | `obj.to_parquet(path)` | `OrdinaryKriging.from_parquet(path)` |
| results: `SimulationSummary`, `CategoricalIndicatorSummary`, … | parquet | `obj.to_parquet(path)` | `SimulationSummary.from_parquet(path)` |
| `Mesh` | OBJ, STL, DXF | `bt.write_mesh(path, mesh)` | `bt.read_mesh(path)` |

</div>

`Drillholes` has no file of its own: keep the collar, survey and interval tables you built it from, and save composites as a `PointSet`.

## JSON for parameters

`to_json` returns a string you can write wherever you like. the class's `from_json` builds the object back:

```python
from pathlib import Path

import boitata as bt
import numpy as np

variogram = bt.Variogram([("spherical", 0.9, 40.0)], nugget=0.1, rotation=(160, 0, 0), ratios=(0.35, 1.0))
Path("variogram.json").write_text(variogram.to_json())
print(Path("variogram.json").read_text())

same = bt.Variogram.from_json(Path("variogram.json").read_text())
print(same.to_json() == variogram.to_json())
```

```text
{"type":"Variogram","format":1,"nugget":0.1,"structures":[{"model":"Spherical","sill":0.9,"range":40.0}],"anisotropy":{"azimuth":160.0,"dip":0.0,"rake":0.0,"major":1.0,"semi":0.35,"minor":1.0}}
True
```

the file is plain text with a `type` and a `format` number. read it as the wrong class and `from_json` raises an error and builds nothing:

```python
bt.Search.from_json(variogram.to_json())
```

```text
InvalidInput: expected a Search, found "Variogram"
```

a fitted transform keeps what it learned, so a reloaded `NormalScore` gives the same scores as the original. these objects also pickle through the same JSON, so they pass between processes.

## parquet for columns

`write_parquet` saves a container's columns as an ordinary parquet file, with its geometry, layout and CRS as JSON in the file's metadata. `read_parquet` returns the same kind of container:

```python
grid = bt.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(52, 60), crs="EPSG:32611")
grid = grid.with_column("V", np.arange(len(grid), dtype=float))
bt.write_parquet("grid.parquet", grid)

back = bt.read_parquet("grid.parquet")
print(back)
print(back.crs, back.size, back.count)
```

```text
BlockModel(regular, 3120 of 3120 cells, count [52, 60, 1], size [5.0, 5.0, 1.0], rotation [0.0, 0.0, 0.0])
  V: Float64
EPSG:32611 [5.0, 5.0, 1.0] [52, 60, 1]
```

any parquet reader opens the file as a table. the geometry sits under the `boitata` metadata key:

```text
{"count":[52,60,1],"crs":"EPSG:32611","kind":"block_model","layout":"regular","origin":[2.5,2.5,0.0],"rotation":[0.0,0.0,0.0],"size":[5.0,5.0,1.0]}
```

a regular model stores no coordinates: the row number is the block number. a masked model adds its block index as a column, and a sub-blocked model adds the parent index and extents.

a fitted estimator saves the same way. its samples become columns; its variogram, search and options go into the metadata as JSON:

```python
samples = bt.datasets.walker_lake()
kriging = bt.OrdinaryKriging(variogram, bt.Search(radius=60)).fit(samples, "V")
kriging.to_parquet("kriging.parquet")
again = bt.OrdinaryKriging.from_parquet("kriging.parquet")
print(np.array_equal(again.predict(grid), kriging.predict(grid), equal_nan=True))
```

```text
True
```

!!! key "key idea"
    parameters go to JSON with `to_json`; anything with columns goes to parquet. both formats are open, so other tools read your files without boitatá.

!!! pitfall "pitfall"
    `write_csv` keeps the block centers and the columns and drops block size, rotation, CRS and layout. use CSV to hand data to someone else, and parquet to keep your own work.

the first-steps examples rerun [a saved five-piece workflow](../examples/01-first-steps/03-saving-and-loading/example_01_03.md) to the same estimate and [store masked models, fitted estimators, categorical results and polylines in parquet](../examples/01-first-steps/04-parquet/example_01_04.md). [models larger than memory](../examples/02-data-and-geometry/10-large-models/example_02_10.md) processes a parquet block model chunk by chunk, and [mesh files](../examples/02-data-and-geometry/07-mesh-files/example_02_07.md) covers OBJ, STL and DXF.
