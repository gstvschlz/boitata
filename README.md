<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gstvschlz/ceres/main/docs/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/gstvschlz/ceres/main/docs/assets/logo.svg" alt="ceres" width="120">
  </picture>
</p>

<h1 align="center">ceres</h1>

<p align="center">
  Geostatistics in Rust with a Python interface.<br>
  <a href="https://gstvschlz.github.io/ceres/">Documentation</a> ·
  <a href="https://gstvschlz.github.io/ceres/examples/">Examples</a> ·
  <a href="https://gstvschlz.github.io/ceres/api/containers/">API</a>
</p>

## About

ceres covers the geostatistical workflow: data I/O, declustering and transforms, variography, kriging and simulation.
The core is Rust, parallel and reproducible for any thread count; everything is available from Python, with data exchanged as NumPy arrays or Arrow tables (pyarrow, polars, pandas).

## Install

Not on PyPI yet. Install from git, which builds the Rust core (Rust ≥ 1.97 and a C compiler):

```sh
pip install "ceres[all] @ git+https://github.com/gstvschlz/ceres"
```

Extras: `plot` (matplotlib), `3d` (pyvista), `all` (also polars, pandas, pyarrow).
See the [install page](https://gstvschlz.github.io/ceres/install/) for uv, poetry, conda, pixi, wheels and offline installs.

## Use

```python
import ceres as cs

samples = cs.PointSet.from_table(cs.read_csv("samples.csv"))
model = cs.Variogram([("spherical", 1.0, 50.0)], nugget=0.1)
ok = cs.OrdinaryKriging(model, cs.Search(radius=100)).fit(samples.coords, samples["grade"])
grid = cs.BlockModel(origin=(0, 0), size=(5, 5), count=(100, 100))
estimate = ok.predict(grid)
```

See [examples](https://github.com/gstvschlz/ceres/blob/main/examples/README.md).

## Examples

- [First steps](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#first-steps): a quick tour, storing containers in Parquet
- [Data and geometry](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#data-and-geometry): drill holes, composites, block models, solids, meshes, GIS files
- [Exploratory analysis](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#exploratory-analysis): declustering, top cuts, contacts, swaths, data spacing
- [Transforms](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#transforms): normal scores, log-ratios, multivariate factors, imputation
- [Spatial continuity](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#spatial-continuity): experimental variograms, fitting, coregionalization
- [Kriging](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#kriging): point and block kriging, cokriging, indicators, search
- [Categories and domains](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#categories-and-domains): rock-type proportions, probabilities, SIS, plurigaussian
- [Stochastic simulation](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#stochastic-simulation): SGS, turning bands, cosimulation
- [Recoverable resources](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#recoverable-resources): tonnes and grade above cutoff on mining blocks
- [Checking models](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#checking-models): cross-validation, realization checks, classification
- [Geological modeling](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#geological-modeling): grade shells, contact surfaces, layered horizons
- [Case studies](https://github.com/gstvschlz/ceres/blob/main/examples/README.md#case-studies): four deposits, from raw tables to a result

## License

MIT
