# 18. Validation and classification

Block kriging of Walker Lake `V` on 10 × 10 m blocks, then the checks a resource estimate needs: global and local
bias against the declustered data, leave-one-out and k-fold cross-validation, kriging efficiency, slope of regression
and the neighbourhood diagnostics per block, a classification from them and the sample spacing, and a majority filter
that removes isolated blocks. The exhaustive grid confirms what the slope of regression predicts.

<details><summary>Python</summary>

```python
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GREY, LIGHT, map_axes, save
from matplotlib.colors import ListedColormap

samples = cs.datasets.walker_lake()
truth = cs.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
model = cs.Variogram.from_json((HERE.parent / "03-variography" / "model.json").read_text())
weights = cs.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights

blocks = cs.BlockModel(origin=(0.5, 0.5), size=(10, 10), count=(26, 30))
search = cs.Search(radius=80, max_samples=24, min_samples=4, rotation=(170, 0, 0), ratios=(0.5, 1.0))
kriging = cs.BlockKriging(model, search, size=(10, 10), discretization=(5, 5, 1)).fit(xy, v)
d = kriging.predict(blocks, diagnostics=True)
true_blocks = truth.reshape(30, 10, 26, 10).mean(axis=(1, 3)).ravel()

bias = cs.global_bias(d["value"], v, data_weights=weights)
print(
    f"blocks {bias['estimate_mean']:.0f} ppm, declustered data {bias['data_mean']:.0f} ppm ({bias['relative']:+.1%})"
)
print(f"true mean {true_blocks.mean():.0f} ppm")
```

</details>

```text
blocks 296 ppm, declustered data 293 ppm (+0.8%)
true mean 278 ppm
```

Local bias shows in swaths: mean grade in 20 m slices along easting and northing, for the blocks, the declustered
samples and the truth. Blocks track the truth slice by slice and are smoother than the samples, whose slice means
scatter where few samples fall.

<details><summary>Python</summary>

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
centroids = blocks.centroids
for ax, axis, name in ((axes[0], "x", "Easting (m)"), (axes[1], "y", "Northing (m)")):
    series = [
        cs.swath(xy, v, 20.0, axis=axis, weights=weights),
        cs.swath(centroids, d["value"], 20.0, axis=axis),
        cs.swath(centroids, true_blocks, 20.0, axis=axis),
    ]
    cs.plot.swath(series, labels=["declustered samples", "blocks", "truth"], ax=ax)
    for line, color in zip(ax.lines, (GREY, ACCENT, "black"), strict=True):
        line.set_color(color)
    ax.legend()
    ax.set(xlabel=name, ylabel="V (ppm)")
save(fig, "swaths")
```

</details>

![swaths](swaths.png)

Cross-validation re-estimates every sample from the others with the same variogram and search. Leave-one-out
removes one sample at a time; k-fold removes a whole fold, sample `i` going to fold `i % k`, so each estimate sees
data a fraction `1/k` sparser. Errors grow as folds get fewer: leave-one-out judges the model at the sample
spacing, which in the clustered areas is finer than most blocks see. A slope above 1 and a standardized squared
error below 1 hold at every k: high estimates slightly understate the samples, and the kriging variance is too
large.

<details><summary>Python</summary>

```python
point_kriging = cs.OrdinaryKriging(model, search).fit(xy, v)
for folds in (None, 10, 5, 2):
    cv = point_kriging.cross_validate(folds=folds)
    print(
        f"{'leave-one-out' if folds is None else f'{folds}-fold':>13}: RMSE {cv.rmse:5.1f} ppm, "
        f"correlation {cv.correlation:.2f}, slope {cv.slope:.2f}, "
        f"standardized squared error {cv.standardized_squared_error:.2f}"
    )
```

</details>

```text
leave-one-out: RMSE 189.1 ppm, correlation 0.78, slope 1.08, standardized squared error 0.65
      10-fold: RMSE 192.8 ppm, correlation 0.77, slope 1.07, standardized squared error 0.67
       5-fold: RMSE 196.9 ppm, correlation 0.76, slope 1.06, standardized squared error 0.69
       2-fold: RMSE 205.5 ppm, correlation 0.73, slope 1.06, standardized squared error 0.69
```

Kriging efficiency compares the block variance with the kriging variance: 1 for a perfectly known block, 0 or less
for one no better known than the global mean. The slope of regression of true on estimated grades is 1 when
estimates are conditionally unbiased; below 1, high estimates overstate and low ones understate the truth. Both
fall away from the samples:

<details><summary>Python</summary>

```python
shape, extent = (30, 26), (0.5, 260.5, 0.5, 300.5)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
for ax, key, title in (
    (axes[0], "efficiency", "Kriging efficiency"),
    (axes[1], "slope", "Slope of regression"),
):
    im = ax.imshow(d[key].reshape(shape), origin="lower", extent=extent, vmin=0, vmax=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color=GREY, linewidths=0)
    map_axes(ax, title)
fig.colorbar(im, ax=axes, shrink=0.8)
save(fig, "diagnostics")
```

</details>

![diagnostics](diagnostics.png)

The truth checks both. Globally, the regression of true on estimated block grades has a slope close to the mean
predicted slope, so the estimate is nearly conditionally unbiased. Blocks with efficiency of 0.7 or more are
clearly closer to the truth; below that efficiency separates them poorly, one reason classification also uses
sample spacing.

<details><summary>Python</summary>

```python
observed = np.polyfit(d["value"], true_blocks, 1)[0]
print(f"slope of regression: predicted mean {d['slope'].mean():.2f}, observed {observed:.2f}")
groups = np.digitize(d["efficiency"], [0.5, 0.7])
for g, label in enumerate(("efficiency < 0.5", "0.5 to 0.7", ">= 0.7")):
    k = groups == g
    error = np.sqrt(np.mean((d["value"][k] - true_blocks[k]) ** 2))
    corr = np.corrcoef(d["value"][k], true_blocks[k])[0, 1]
    print(f"{label:>16}: {k.sum():3d} blocks, RMSE {error:5.1f} ppm, correlation with truth {corr:.2f}")
```

</details>

```text
slope of regression: predicted mean 0.94, observed 1.03
efficiency < 0.5: 234 blocks, RMSE 110.2 ppm, correlation with truth 0.68
      0.5 to 0.7: 324 blocks, RMSE 112.2 ppm, correlation with truth 0.70
          >= 0.7: 222 blocks, RMSE  83.5 ppm, correlation with truth 0.93
```

The other diagnostics say why a block is weak. `mean_distance` is the mean distance to the samples used, and
`max_samples_reached` flags searches that stopped at 24 samples before the ellipse ran out: those blocks sit in
denser data and have the higher slope. `negative_weight_sum` adds the weights below zero, which samples screened
by closer ones receive. They are strongest on the sparse regular pattern between the clusters, and blocks with
strongly negative sums carry the largest errors, overstate the truth, and give the only negative grades. `lagrange`
(the Lagrange multiplier) and `n_holes` (distinct drill holes used; each Walker Lake sample is its own) complete
the set.

<details><summary>Python</summary>

```python
full = d["max_samples_reached"] == 1
for label, k in (("full search", full), ("ellipse ran out", ~full)):
    print(
        f"{label:>15}: {k.mean():4.0%} of blocks, mean distance {d['mean_distance'][k].mean():4.1f} m, "
        f"mean slope {d['slope'][k].mean():.2f}"
    )
groups = np.digitize(d["negative_weight_sum"], [-0.08, -0.04])
for g, label in enumerate(("below -0.08", "-0.08 to -0.04", "above -0.04")):
    k = groups == g
    error = np.sqrt(np.mean((d["value"][k] - true_blocks[k]) ** 2))
    print(
        f"negative weights {label:>14}: {k.sum():3d} blocks, RMSE {error:5.1f} ppm, "
        f"mean {d['value'][k].mean():3.0f} ppm against {true_blocks[k].mean():3.0f} true"
    )
negative = d["value"] < 0
print(
    f"{negative.sum()} negative estimates, negative weights {d['negative_weight_sum'][negative].max():.2f} or below"
)
```

</details>

```text
    full search:  89% of blocks, mean distance 28.2 m, mean slope 0.95
ellipse ran out:  11% of blocks, mean distance 40.5 m, mean slope 0.86
negative weights    below -0.08: 119 blocks, RMSE 135.9 ppm, mean 245 ppm against 202 true
negative weights -0.08 to -0.04: 358 blocks, RMSE 113.4 ppm, mean 247 ppm against 229 true
negative weights    above -0.04: 303 blocks, RMSE  85.2 ppm, mean 373 ppm against 366 true
6 negative estimates, negative weights -0.05 or below
```

<details><summary>Python</summary>

```python
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
for ax, key, title, cmap in (
    (axes[0], "mean_distance", "Mean distance to samples used (m)", "ceres"),
    (axes[1], "negative_weight_sum", "Sum of negative weights", plt.get_cmap("ceres").reversed()),
):
    im = ax.imshow(d[key].reshape(shape), origin="lower", extent=extent, cmap=cmap)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color=GREY, linewidths=0)
    map_axes(ax, title)
    fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, "neighbourhood")
```

</details>

![neighbourhood](neighbourhood.png)

Classification combines slope and efficiency with the distance to the nearest sample, from `neighborhood_stats`.
Rules apply in order and the first that holds wins; a 3 × 3 majority filter then absorbs isolated blocks into
their surroundings.

<details><summary>Python</summary>

```python
near = cs.neighborhood_stats(blocks, xy, v, k=8, radius=80)
criteria = {**d, "distance": near["nearest_dist"]}
rules = [
    ("measured", {"slope": (">=", 0.95), "efficiency": (">=", 0.7), "distance": ("<=", 7)}),
    ("indicated", {"slope": (">=", 0.9), "efficiency": (">=", 0.5)}),
]
classes = cs.classify(criteria, rules, default="inferred")
smoothed = cs.smooth_classes(blocks, classes, window=(3, 3, 1))
names = ["measured", "indicated", "inferred"]
for name in names:
    print(
        f"{name:>9}: {np.mean(classes == name):5.1%} of blocks, {np.mean(smoothed == name):5.1%} after smoothing"
    )
```

</details>

```text
 measured: 27.7% of blocks, 27.1% after smoothing
indicated: 39.5% of blocks, 42.1% after smoothing
 inferred: 32.8% of blocks, 30.9% after smoothing
```

<details><summary>Python</summary>

```python
colors = ListedColormap([ACCENT, "#9ebad6", LIGHT])
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
for ax, c, title in ((axes[0], classes, "Rules"), (axes[1], smoothed, "After a 3 × 3 majority filter")):
    code = np.select([c == n for n in names], range(3))
    ax.imshow(code.reshape(shape), origin="lower", extent=extent, cmap=colors, vmin=-0.5, vmax=2.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color=GREY, linewidths=0)
    map_axes(ax, title)
handles = [plt.Line2D([], [], marker="s", ls="", color=colors(i), label=n) for i, n in enumerate(names)]
fig.legend(handles=handles, loc="outside lower center", ncol=3)
save(fig, "classes")
```

</details>

![classes](classes.png)

Full script: [`example_18.py`](example_18.py)
