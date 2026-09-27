---
hide: [navigation, toc]
---

<div class="ceres-hero" markdown>
<img class="logo-light" src="assets/logo.svg" alt="">
<img class="logo-dark" src="assets/logo-dark.svg" alt="">
<div markdown>
<h1>ceres</h1>

Geostatistics in Rust with a Python interface. Parallel, and the same result on any number of threads.

[Get started](#install){ .md-button .md-button--primary }
[Examples](examples/tutorials/index.md){ .md-button }
[API reference](api/containers.md){ .md-button }
</div>
</div>

<div class="grid cards" markdown>

-   :material-database-outline: **Data and I/O**

    ---

    Point sets, drillholes and block models with Arrow columns; CSV, GSLIB and Parquet; zero-copy to NumPy,
    polars and pandas.

-   :material-chart-bell-curve: **Transforms and EDA**

    ---

    Declustering, normal score and anamorphosis, log-ratios, and statistics for exploratory analysis.

-   :material-vector-curve: **Variography**

    ---

    Experimental variograms, model fitting and the linear model of coregionalization.

-   :material-cube-outline: **Estimation**

    ---

    Simple, ordinary and indicator kriging, cokriging, IDW and cross-validation over anisotropic searches.

-   :material-dice-multiple-outline: **Simulation**

    ---

    SGS, SIS and turning bands, with reproducible realizations from a single seed.

-   :material-layers-triple-outline: **Modeling and validation**

    ---

    Desurveying, compositing, domaining, implicit modeling and checks of estimates against the data.

</div>

## Install

ceres is not on PyPI yet. Install it from git (this builds the Rust core, so it needs Rust and a C compiler):

```bash
pip install "ceres[all] @ git+https://github.com/gstvschlz/ceres"
```

The [install page](install.md) covers uv, poetry, conda and pixi, wheels, extras and offline installs.

## A first estimate

```python
import ceres as cs

samples = cs.datasets.walker_lake()
xy, v = samples.coords, samples["V"]
variogram = cs.experimental_variogram(xy, v, lag=10, max_lag=120).fit("spherical")
grid = cs.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
kriging = cs.OrdinaryKriging(variogram, cs.Search(radius=80, max_samples=24)).fit(xy, v)
estimate, variance = kriging.predict(grid, return_variance=True)
scores = cs.NormalScore().fit_transform(v)
gaussian = cs.experimental_variogram(xy, scores, lag=10, max_lag=120).fit("spherical")
summary = cs.SGS(gaussian, cs.Search(radius=80)).fit(xy, v).simulate(grid, n=50, cutoffs=[500])
risk = summary.probability_above[0]
```

The [examples](examples/tutorials/index.md) work through every topic on open datasets; the API pages list every
class and function.
