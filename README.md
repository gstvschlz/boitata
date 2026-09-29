<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gstvschlz/ceres/main/docs/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/gstvschlz/ceres/main/docs/assets/logo.svg" alt="ceres" width="120">
  </picture>
</p>

<h1 align="center">ceres</h1>

<p align="center">
  My Python tooling for day-to-day geostatistics.<br>
  <a href="https://gstvschlz.github.io/ceres/">Documentation</a> ·
  <a href="https://gstvschlz.github.io/ceres/examples/01-first-steps/">Examples</a> ·
  <a href="https://gstvschlz.github.io/ceres/api/containers/">API</a>
</p>

ceres holds the tools I use for resource work: drill hole checks, statistics, transforms, variograms, kriging, simulation and model checks. It runs from Python; the numerical core is Rust.

## Install

```sh
pip install "ceresgeo[all]"
```

Published on PyPI as `ceresgeo`, imported as `ceres`. Wheels for Linux, macOS and Windows, Python ≥ 3.11.
Extras: `plot` (matplotlib), `3d` (pyvista), `all` (also polars, pandas, pyarrow).
See the [install page](https://gstvschlz.github.io/ceres/install/) for uv, poetry, conda, pixi and offline installs.

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
