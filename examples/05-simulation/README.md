# 5. Sequential Gaussian simulation

Kriging gives one smooth map. Simulation draws many maps that each honour the samples, the declustered histogram and
the variogram; together they measure uncertainty.

<details><summary>Python</summary>

```python
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GREY, HIGHLIGHT, INK, LIGHT, map_axes, save
from matplotlib.colors import PowerNorm

samples = cs.datasets.walker_lake()
truth = cs.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
azimuth = cs.Variogram.from_json((HERE.parent / "03-variography" / "model.json").read_text()).rotation[0]
```

</details>

Normal scores with declustering weights, tails bounded to 0 and the largest sample, and their variogram along
N170° and N260° scaled to a unit sill:

<details><summary>Python</summary>

```python
weights = cs.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights
y = cs.NormalScore(tails=(0.0, v.max())).fit_transform(v, weights=weights)

lag, max_lag = 10.0, 120.0
major = cs.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth).fit("spherical")
minor = cs.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth + 90).fit("spherical")
total = major.sill
a_major = major.structures[0].range
ratio = min(minor.structures[0].range / a_major, 1.0)
gaussian = cs.Variogram(
    [("spherical", major.structures[0].sill / total, a_major)],
    nugget=major.nugget / total,
    rotation=(azimuth, 0, 0),
    ratios=(ratio, 1.0),
)
print(gaussian)
```

</details>

```text
Variogram(nugget=0.32126839519606537, structures=[Structure("spherical", sill=0.6787316048039347, range=82.07160980744193)], rotation=(170.0, 0.0, 0.0), ratios=(0.4270832349370336, 1.0))
```

SGS normal-scores the data itself, simulates along a random path and back-transforms. 50 realizations, the same on
any number of threads. `simulate` returns a summary accumulated while it runs: the mean, variance and quantiles at
every node, the probability and mean above each cutoff, and each realization's global mean and share above the
cutoffs. Realizations are kept only when asked for, here to check them.

<details><summary>Python</summary>

```python
grid = cs.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
sgs = cs.SGS(gaussian, cs.Search(radius=100, max_samples=24)).fit(xy, v, weights=weights)
summary = sgs.simulate(grid, n=50, seed=42, cutoffs=[500.0], quantiles=[0.1, 0.9], realizations=True)
reals = summary.realizations
etype = summary.mean
p500 = summary.probability_above[0]

nodes = grid.centroids.astype(int)
true_at_nodes = truth[nodes[:, 1] - 1, nodes[:, 0] - 1]
means = summary.realization_mean
print(f"realization means {means.min():.0f}-{means.max():.0f}, true {true_at_nodes.mean():.0f}")
print(f"realization variance {reals.var(axis=1).mean():.0f}, true {true_at_nodes.var():.0f}")
low, high = np.quantile(summary.realization_above[0], [0.1, 0.9])
print(f"area above 500 ppm: P10 {low:.1%}, P90 {high:.1%}, true {np.mean(true_at_nodes > 500):.1%}")
```

</details>

```text
realization means 282-320, true 276
realization variance 72667, true 62312
area above 500 ppm: P10 21.8%, P90 25.1%, true 18.9%
```

Each realization looks like the truth; their mean is smooth like kriging and their spread measures uncertainty.

<details><summary>Python</summary>

```python
shape = (60, 52)
extent = (0.5, 260.5, 0.5, 300.5)
norm = PowerNorm(0.5, vmin=0, vmax=1500)
fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), layout="constrained")
for ax, image, title in (
    (axes[0, 0], true_at_nodes, "True V at grid nodes"),
    (axes[0, 1], reals[0], "Realization 1"),
    (axes[0, 2], reals[1], "Realization 2"),
    (axes[1, 0], etype, "Mean of 50 realizations"),
):
    im = ax.imshow(image.reshape(shape), origin="lower", extent=extent, norm=norm)
    map_axes(ax, title)
fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="V (ppm)")
fig.colorbar(im, ax=axes[1, 0], shrink=0.8, label="V (ppm)")
spread = axes[1, 1].imshow(summary.std.reshape(shape), origin="lower", extent=extent, cmap="cividis")
map_axes(axes[1, 1], "Spread across realizations")
fig.colorbar(spread, ax=axes[1, 1], shrink=0.8, label="standard deviation (ppm)")
prob = axes[1, 2].imshow(p500.reshape(shape), origin="lower", extent=extent, vmin=0, vmax=1)
map_axes(axes[1, 2], "Probability V > 500 ppm")
axes[1, 2].contour(
    (true_at_nodes > 500).reshape(shape).astype(float),
    levels=[0.5],
    origin="lower",
    extent=extent,
    colors=HIGHLIGHT,
    linewidths=0.8,
)
fig.colorbar(prob, ax=axes[1, 2], shrink=0.8, label="probability; true V > 500 outlined")
save(fig, "maps")
```

</details>

![maps](maps.png)

Each realization reproduces the declustered histogram and the model variogram:

<details><summary>Python</summary>

```python
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
grid_v = np.sort(true_at_nodes)
for r in reals:
    a.plot(np.sort(r), np.linspace(0, 1, r.size), color=LIGHT, lw=0.8)
a.plot(grid_v, np.linspace(0, 1, grid_v.size), color=INK, lw=1.4, label="truth")
order = np.argsort(v)
a.step(
    v[order],
    np.cumsum(weights[order]) / weights.sum(),
    color=ACCENT,
    lw=1.4,
    where="post",
    label="declustered samples",
)
a.plot([], [], color=LIGHT, label="50 realizations")
a.set(xlim=(0, 1600), xlabel="V (ppm)", ylabel="Cumulative probability", title="Histogram reproduction")
a.legend(loc="lower right")

xyz = grid.centroids
for r in reals[:20]:
    scores = cs.NormalScore().fit_transform(r)
    exp = cs.experimental_variogram(xyz, scores, lag, max_lag, azimuth=azimuth)
    b.plot(exp.lags, exp.gammas, color=LIGHT, lw=0.8)
data_exp = cs.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth)
b.plot(data_exp.lags, data_exp.gammas, "o", color=ACCENT, ms=4, label="normal scores of samples")
h = np.linspace(0, max_lag, 200)
b.plot(h, gaussian.gamma(h), color=HIGHLIGHT, lw=1.4, label="model")
b.plot([], [], color=LIGHT, label="20 realizations")
b.axhline(1.0, color=GREY, lw=0.8, ls="--")
b.set(xlim=(0, max_lag), ylim=(0, 1.4), xlabel="Lag distance (m)", ylabel="γ(h) of normal scores")
b.set_title(f"Variogram reproduction, N{azimuth:.0f}°")
b.legend(loc="lower right")
save(fig, "reproduction")
```

</details>

![reproduction](reproduction.png)

[Chapter 9](../09-change-of-support/README.md) averages realizations over mining blocks.

Full script: [`example_05.py`](example_05.py)
