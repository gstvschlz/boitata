# 20. From drill holes to a classified model

One pass through a resource workflow on the massive sulphide (`MS`) lens of [chapter 19](../19-geological-model/README.md):
composites from the drill-hole CSVs, exploratory statistics, declustering, the normal-score variogram, ordinary
kriging with its diagnostics, a simulation summary for risk, classification, and the model saved to Parquet.

<details><summary>Python</summary>

```python
import tempfile

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GREY, HIGHLIGHT, LIGHT, save
from matplotlib.colors import ListedColormap
```

</details>

## Composites

Assays and logged lithology are merged, desurveyed and composited to 5 m within each lithology. The `MS` composites
inside the lens window are the data; `hole` names travel with them for the search and for classification.

<details><summary>Python</summary>

```python
tables = cs.datasets.drillhole_tables()
drillholes = cs.Drillholes(
    tables["collar"], tables["survey"], cs.merge_intervals(tables["assay"], tables["geology"])
)
composites = drillholes.composite(5.0, ["ZN"], domain="LITH")
xyz, zn = composites.coords, composites["ZN"]
lith = np.array(composites["LITH"], dtype=object)
lens = (xyz[:, 0] > 5300) & (xyz[:, 0] < 5500) & (xyz[:, 1] > 8100) & (xyz[:, 1] < 8400)
keep = lens & (lith == "MS") & ~np.isnan(zn)
xyz, zn = xyz[keep], zn[keep]
holes = np.array(composites["hole"], dtype=object)[keep]
print(f"{len(zn)} MS composites from {len(set(holes))} holes")
```

</details>

```text
432 MS composites from 147 holes
```

## Statistics, capping and declustering

Holes cluster where the lens is rich, so cell declustering weights the composites before any statistic. Capping
at the 99th percentile would remove 0.1 % of the metal, so grades are left uncapped.

<details><summary>Python</summary>

```python
weights = cs.cell_declustering(xyz, zn, sizes=np.arange(5, 80, 5)).weights
naive, declustered = cs.describe(zn), cs.describe(zn, weights)
print(f"mean {naive['mean']:.2f} % Zn, declustered {declustered['mean']:.2f} %, CV {declustered['cv']:.2f}")
caps = cs.capping(zn, weights)
for cap, fraction, removed in zip(caps["cap"], caps["fraction"], caps["metal_removed"], strict=True):
    print(f"cap {cap:5.1f} % Zn: {fraction:5.1%} of composites cut, {removed:5.1%} of the metal removed")

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
cs.plot.histogram(zn, weights, bins=np.arange(0, 44, 2), ax=a, color=LIGHT, edgecolor=GREY)
a.set(xlabel="Zn (%)", title="Declustered histogram")
cs.plot.probability(zn[zn > 0], weights[zn > 0], log=True, ax=b, color=ACCENT, ms=3)
b.set(xlabel="Zn (%)", title="Probability plot")
save(fig, "statistics")
```

</details>

```text
mean 9.04 % Zn, declustered 9.16 %, CV 0.94
cap  21.9 % Zn: 10.8% of composites cut,  6.1% of the metal removed
cap  24.7 % Zn:  5.0% of composites cut,  3.9% of the metal removed
cap  30.9 % Zn:  2.7% of composites cut,  1.7% of the metal removed
cap  38.4 % Zn:  1.8% of composites cut,  0.1% of the metal removed
cap  39.9 % Zn:  0.1% of composites cut,  0.0% of the metal removed
cap  41.0 % Zn:  0.1% of composites cut,  0.0% of the metal removed
```

![statistics](statistics.png)

## Variograms

Simulation needs the variogram of the normal scores, rescaled to a unit sill; kriging uses the variogram of the
grades. Both are fitted omnidirectionally here: the lens is too narrow across strike for reliable directional
pairs.

<details><summary>Python</summary>

```python
scores = cs.NormalScore().fit_transform(zn, weights=weights)
experimental = cs.experimental_variogram(xyz, scores, 10.0, 120.0)
fitted = experimental.fit("spherical")
sill = fitted.nugget + fitted.structures[0].sill
gaussian = cs.Variogram(
    [("spherical", fitted.structures[0].sill / sill, fitted.structures[0].range)], nugget=fitted.nugget / sill
)
grades = cs.experimental_variogram(xyz, zn, 10.0, 120.0).fit("spherical")
print(gaussian)

fig, ax = cs.plot.variogram(experimental, fitted, color=ACCENT)
ax.set(xlabel="Lag distance (m)", ylabel="γ(h) of normal scores", title="Normal-score variogram")
save(fig, "variogram")
```

</details>

```text
Variogram(nugget=0.2923448447944328, structures=[Structure("spherical", sill=0.7076551552055672, range=34.433139447946736)], rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0))
```

![variogram](variogram.png)

## Kriging and its checks

5 m blocks within 10 m of an `MS` composite stand in for the lens volume (chapter 19 models it properly). Ordinary
kriging takes at most four composites per hole, and returns kriging efficiency and slope of regression with each
estimate. The global mean is within 3 % of the declustered composites; by elevation the blocks follow the
composites, smoother as kriging should be, and damp the rich level at 860–880 m.

<details><summary>Python</summary>

```python
lo, hi = xyz.min(axis=0), xyz.max(axis=0)
origin = np.floor(lo / 5) * 5
count = tuple(int(c) for c in np.ceil((hi - origin) / 5))
blocks = cs.BlockModel(origin=tuple(origin), size=(5, 5, 5), count=count)
blocks = blocks.mask(cs.neighborhood_stats(blocks, xyz, zn, k=1, radius=50)["nearest_dist"] <= 10)

search = cs.Search(radius=60, max_samples=16, min_samples=4, max_per_hole=4)
kriged = cs.OrdinaryKriging(grades, search).fit(xyz, zn, holes=holes).predict(blocks, diagnostics=True)
bias = cs.global_bias(kriged["value"], zn, data_weights=weights)
print(
    f"{len(blocks)} blocks; mean {bias['estimate_mean']:.2f} % Zn against {bias['data_mean']:.2f} % declustered"
)

fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
cs.plot.swath(
    [
        cs.swath(xyz, zn, 20.0, axis="z", weights=weights),
        cs.swath(blocks.centroids, kriged["value"], 20.0, axis="z"),
    ],
    labels=["declustered composites", "blocks"],
    ax=ax,
)
ax.set(xlabel="Elevation (m)", ylabel="Zn (%)", title="Swath by elevation")
save(fig, "swath")
```

</details>

```text
5540 blocks; mean 8.87 % Zn against 9.16 % declustered
```

![swath](swath.png)

## Simulation for risk

Thirty sequential Gaussian simulations give, without storing them, the probability that each block exceeds
10 % Zn and the spread of the lens' share above that cutoff.

<details><summary>Python</summary>

```python
summary = cs.SGS(gaussian, cs.Search(radius=60, max_samples=16)).fit(xyz, zn, weights=weights, holes=holes)
summary = summary.simulate(blocks, n=30, seed=1, cutoffs=[10.0])
low, high = np.quantile(summary.realization_above[0], [0.1, 0.9])
print(
    f"share of blocks above 10 % Zn: P10 {low:.1%}, P90 {high:.1%}; kriged {np.mean(kriged['value'] > 10):.1%}"
)
```

</details>

```text
share of blocks above 10 % Zn: P10 32.6%, P90 36.7%; kriged 36.3%
```

## Classification

Measured blocks need a slope of regression of at least 0.8 and composites from three holes nearby; indicated,
a slope of 0.5. A 3 × 3 × 3 majority filter removes isolated blocks.

<details><summary>Python</summary>

```python
spacing = cs.neighborhood_stats(blocks, xyz, zn, k=8, radius=60, holes=holes)
rules = [
    ("measured", {"slope": (">=", 0.8), "holes": (">=", 3)}),
    ("indicated", {"slope": (">=", 0.5)}),
]
classes = cs.classify({**kriged, "holes": spacing["n_holes"]}, rules, default="inferred")
classes = cs.smooth_classes(blocks, classes, window=(3, 3, 3))
names = ["measured", "indicated", "inferred"]
for name in names:
    inside = classes == name
    print(f"{name:>9}: {inside.mean():5.1%} of blocks, mean {np.nanmean(kriged['value'][inside]):5.2f} % Zn")
```

</details>

```text
 measured: 46.5% of blocks, mean  9.45 % Zn
indicated: 53.2% of blocks, mean  8.34 % Zn
 inferred:  0.3% of blocks, mean 13.20 % Zn
```

On the east–west section through the middle of the lens:

<details><summary>Python</summary>

```python
blocks = (
    blocks.with_column("zn", kriged["value"])
    .with_column("p_above_10", summary.probability_above[0])
    .with_column("class", np.select([classes == n for n in names], range(3)).astype(float))
)
row = blocks.count[1] // 2
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
cs.plot.section(blocks, "zn", axis="y", index=row, ax=axes[0], vmin=0, vmax=25)
cs.plot.section(blocks, "p_above_10", axis="y", index=row, ax=axes[1], vmin=0, vmax=1)
cs.plot.section(
    blocks,
    "class",
    axis="y",
    index=row,
    ax=axes[2],
    colorbar=False,
    cmap=ListedColormap([ACCENT, "#9ebad6", LIGHT]),
    vmin=-0.5,
    vmax=2.5,
)
north = blocks.origin[1] + (row + 0.5) * 5
on_section = np.abs(xyz[:, 1] - north) < 5
for ax, title in zip(axes, ("Kriged Zn (%)", "P(Zn > 10 %)", "Class"), strict=True):
    ax.scatter(xyz[on_section, 0], xyz[on_section, 2], s=4, color=HIGHLIGHT, linewidths=0)
    ax.set_title(title)
    ax.set(xlabel="Easting (m)", ylabel="Elevation (m)")
handles = [
    plt.Line2D([], [], marker="s", ls="", color=c, label=n) for c, n in zip((ACCENT, "#9ebad6", LIGHT), names)
]
axes[2].legend(handles=handles, loc="lower right", fontsize=8)
save(fig, "section")
```

</details>

![section](section.png)

## Saving

The block model, with its geometry, mask and columns, goes to Parquet and back unchanged.

<details><summary>Python</summary>

```python
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "ms_lens.parquet"
    cs.write_parquet(path, blocks)
    stored = cs.read_parquet(path)
print(
    f"{len(stored)} blocks, columns {stored.attributes.column_names}, same Zn: {np.allclose(stored['zn'], blocks['zn'])}"
)
```

</details>

```text
5540 blocks, columns ['zn', 'p_above_10', 'class'], same Zn: True
```

Full script: [`example_20.py`](example_20.py)
