# ceres

Geostatistics in Rust with a Python interface: data containers and I/O, exploratory analysis, transforms,
variography, estimation, simulation, validation and implicit modelling. Computations run in parallel and give the
same result on any number of threads.

```bash
pip install ceres            # numpy only
pip install "ceres[plot]"    # with matplotlib for ceres.plot
pip install "ceres[3d]"      # with pyvista for ceres.plot3d
```

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

The [examples](examples/index.md) work through every topic on open datasets; the API pages list every class and
function.
