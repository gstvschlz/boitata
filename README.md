# ceres

Geostatistics in Rust with a Python interface.

## About

ceres covers the geostatistical workflow: data I/O, declustering and transforms, variography, kriging and simulation.
The core is Rust, parallel and reproducible for any thread count; everything is available from Python, with data exchanged as NumPy arrays or Arrow tables (pyarrow, polars, pandas).

## Install

From source, with [mise](https://mise.jdx.dev):

```sh
mise run py:build
```

## Use

```python
import ceres as cs

samples = cs.PointSet.from_table(cs.read_csv("samples.csv"))
model = cs.Variogram([("spherical", 1.0, 50.0)], nugget=0.1)
ok = cs.OrdinaryKriging(model, cs.Search(radius=100)).fit(samples.coords, samples["grade"])
grid = cs.BlockModel(origin=(0, 0), size=(5, 5), count=(100, 100))
estimate = ok.predict(grid)
```

See [examples](examples/README.md).

## License

MIT
