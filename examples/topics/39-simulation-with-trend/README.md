# 39. Simulation with a trend

Far from the data, SGS draws from the global histogram, wherever it is. A trend known everywhere can steer it instead.

<details><summary>Python</summary>

```python
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import map_axes, save
from matplotlib.colors import PowerNorm

samples = cs.datasets.walker_lake()
xy, v = samples.coords, samples["V"]
weights = cs.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights
grid = cs.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
```

</details>

Here the trend is a moving-window average of V within 40 m, built with an existing estimator; any model or estimate
would do. `trend=` gives it at the data to `fit` and at the nodes to `simulate`, as an array or the name of a
BlockModel column.

<details><summary>Python</summary>

```python
window = cs.MovingAverage(cs.Search(radius=40, max_samples=200)).fit(xy, v)
trend = window.predict(xy)
trended = grid.with_column("trend", window.predict(grid))
```

</details>

The data are normal-scored within 8 equal-probability classes of the trend, the stepwise conditional transform of
topic 16 on (trend, V), which leaves scores independent of the trend. Those scores are simulated with their own
variogram, and every node is back-transformed with the histogram of its trend class. Plain SGS, for comparison,
uses the variogram of global normal scores; both are scaled to a unit sill.

<details><summary>Python</summary>

```python
pair = np.column_stack([trend, v])
scores = cs.StepwiseConditional(classes=8).fit(pair, weights=weights).transform(pair)[:, 1]
y = cs.NormalScore(tails=(0.0, v.max())).fit_transform(v, weights=weights)


def unit_sill(values):
    fitted = cs.experimental_variogram(xy, values, 10.0, 150.0).fit("spherical")
    (structure,) = fitted.structures
    return cs.Variogram(
        [("spherical", structure.sill / fitted.sill, structure.range)], nugget=fitted.nugget / fitted.sill
    )


score_variogram, plain_variogram = unit_sill(scores), unit_sill(y)
print(score_variogram, plain_variogram, sep="\n")

search = cs.Search(radius=100, max_samples=24)
with_trend = cs.SGS(score_variogram, search, classes=8).fit(xy, v, weights=weights, trend=trend)
by_trend = with_trend.simulate(trended, n=20, seed=5, keep=True, trend="trend").realizations
plain = cs.SGS(plain_variogram, search).fit(xy, v, weights=weights)
by_sgs = plain.simulate(grid, n=20, seed=5, keep=True).realizations
```

</details>

```text
Variogram(nugget=0.19038648061160174, structures=[Structure("spherical", sill=0.8096135193883982, range=22.62299198336531)], rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0))
Variogram(nugget=0.3405226282644493, structures=[Structure("spherical", sill=0.6594773717355508, range=51.76429044224697)], rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0))
```

<details><summary>Python</summary>

```python
node_trend = trended["trend"]
cov = np.cov(v, trend, aweights=weights)
print(f"correlation with the trend: data {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.2f}", end="")
for name, reals in (("SGS", by_sgs), ("SGS with trend", by_trend)):
    print(f", {name} {np.mean([np.corrcoef(r, node_trend)[0, 1] for r in reals]):.2f}", end="")
edges = np.quantile(node_trend, [0.25, 0.5, 0.75])
at_data, at_nodes = np.digitize(trend, edges), np.digitize(node_trend, edges)
print(f"\n{'mean V (ppm) by trend quartile':<32}{'data':>6}{'SGS':>6}{'SGS with trend':>16}")
for k in range(4):
    data_mean = np.average(v[at_data == k], weights=weights[at_data == k])
    plain_mean, trend_mean = (reals[:, at_nodes == k].mean() for reals in (by_sgs, by_trend))
    print(f"{f'quartile {k + 1}':<32}{data_mean:6.0f}{plain_mean:6.0f}{trend_mean:16.0f}")
```

</details>

```text
correlation with the trend: data 0.52, SGS 0.45, SGS with trend 0.50
mean V (ppm) by trend quartile    data   SGS  SGS with trend
quartile 1                         106   140             110
quartile 2                         267   259             260
quartile 3                         320   329             323
quartile 4                         450   452             454
```

<details><summary>Python</summary>

```python
shape = (60, 52)
extent = (0.5, 260.5, 0.5, 300.5)
norm = PowerNorm(0.5, vmin=0, vmax=1500)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), layout="constrained")
panels = [
    (node_trend, "Moving-window trend, 40 m"),
    (by_sgs[0], "SGS, realization 1"),
    (by_trend[0], "SGS with the trend, realization 1"),
]
for ax, (image, title) in zip(axes, panels):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, norm=norm)
    map_axes(ax, title)
fig.colorbar(im, ax=axes, shrink=0.8, label="V (ppm)")
save(fig, "trend")
```

</details>

![trend](trend.png)

Plain SGS already follows the trend where data are dense, but pulls the low-trend quarter up towards the global
mean; with the trend, each quarter keeps the declustered mean of its data, and the realizations correlate with the
trend about as much as the data do.

Full script: [`example_39.py`](example_39.py)
