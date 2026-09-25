# 18. Validation and classification

Block kriging of Walker Lake `V` on 10 × 10 m blocks, then the checks a resource estimate needs: global bias
against the declustered data, kriging efficiency and slope of regression per block, a classification from them
and the sample spacing, and a majority filter that removes isolated blocks. The exhaustive grid confirms what the
slope of regression predicts.

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
blocks 295 ppm, declustered data 293 ppm (+0.8%)
true mean 278 ppm
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
efficiency < 0.5: 152 blocks, RMSE 109.4 ppm, correlation with truth 0.70
      0.5 to 0.7: 378 blocks, RMSE 116.1 ppm, correlation with truth 0.66
          >= 0.7: 250 blocks, RMSE  91.5 ppm, correlation with truth 0.92
```

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
 measured: 29.5% of blocks, 29.0% after smoothing
indicated: 46.8% of blocks, 49.6% after smoothing
 inferred: 23.7% of blocks, 21.4% after smoothing
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

Full script: [`example.py`](example.py)
