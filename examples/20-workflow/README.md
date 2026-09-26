# 20. From drill holes to a classified model

One pass through a resource workflow on the massive sulphide (`MS`) lens of [chapter 19](../19-geological-model/README.md):
composites from the drill-hole CSVs, the lens modelled from its contacts and kept within the drilling, exploratory
statistics, declustering, the normal-score variogram, ordinary kriging in search passes, block-support simulation
for risk, classification, and the model saved to Parquet.

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

Assays and logged lithology are merged and desurveyed by minimum curvature, with tangential desurvey kept for
comparison. Whole runs (`length=None`) show how thick the `MS` intercepts are. Overlapping assays are resolved first
by keeping the one that starts first, as in [chapter 6](../06-drillholes/README.md).

<details><summary>Python</summary>

```python
tables = cs.datasets.drillhole_tables()
assay = tables["assay"]
hole_id = np.array(assay["HOLEID"])
start, end = assay["FROM"], assay["TO"]
keep = np.ones(assay.num_rows, bool)
reach = {}
for i in np.lexsort((start, hole_id)):
    if start[i] < reach.get(hole_id[i], -np.inf):
        keep[i] = False
    else:
        reach[hole_id[i]] = end[i]
assay = cs.Table({c: np.asarray(assay[c])[keep] for c in assay.column_names})
intervals = cs.merge_intervals(assay, tables["geology"])
drillholes = cs.Drillholes(tables["collar"], tables["survey"], intervals, method="minimum_curvature")
tangential = cs.Drillholes(tables["collar"], tables["survey"], intervals, method="tangential")


def in_window(points):
    return (points[:, 0] > 5300) & (points[:, 0] < 5500) & (points[:, 1] > 8100) & (points[:, 1] < 8400)


runs = drillholes.composite(None, ["ZN"], domain="LITH")
run_length = runs["length"][in_window(runs.coords) & (np.array(runs["LITH"], dtype=object) == "MS")]
print(
    f"{len(run_length)} MS intercepts: median {np.median(run_length):.1f} m, "
    f"90 % shorter than {np.quantile(run_length, 0.9):.1f} m, longest {run_length.max():.1f} m"
)
```

</details>

```text
306 MS intercepts: median 2.2 m, 90 % shorter than 10.0 m, longest 35.3 m
```

Most intercepts are shorter than a 5 m composite, so each is one composite of its own length; in longer runs the
last piece, if under half a composite, joins the one before (`residual="merge"`) rather than standing alone.
Composites are taken within each lithology and sorted down each hole, in the lens window.

<details><summary>Python</summary>

```python
composites = drillholes.composite(5.0, ["ZN"], domain="LITH", residual="merge")
shift = np.linalg.norm(
    tangential.composite(5.0, ["ZN"], domain="LITH", residual="merge").coords - composites.coords, axis=1
)
xyz, zn = composites.coords, composites["ZN"]
lith = np.array(composites["LITH"], dtype=object)
hole = np.array(composites["hole"], dtype=object)
top, bottom, length = composites["from"], composites["to"], composites["length"]
order = np.lexsort((top, hole))
order = order[in_window(xyz[order])]
xyz, zn, lith, hole, top, bottom, length, shift = (
    a[order] for a in (xyz, zn, lith, hole, top, bottom, length, shift)
)
ms = lith == "MS"
print(
    f"{ms.sum()} MS composites, {length[ms].min():.1f}-{length[ms].max():.1f} m; tangential desurvey moves them "
    f"{np.median(shift[ms]):.2f} m (median), {shift[ms].max():.1f} m at most"
)
```

</details>

```text
392 MS composites, 0.1-7.0 m; tangential desurvey moves them 0.26 m (median), 6.8 m at most
```

## The lens and the drilled volume

As in chapter 19, the lens is a potential field pinned to 0 at every `MS` contact down a hole, +1 in `MS` and −1
around it. The field continues the lens along its plunge past the last hole, so blocks are kept only inside the
convex hull of the `MS` composites: no block is estimated from data entirely on one side of it.

<details><summary>Python</summary>

```python
change = (hole[1:] == hole[:-1]) & (ms[1:] != ms[:-1]) & np.isclose(bottom[:-1], top[1:])
contacts = drillholes.at(list(hole[:-1][change]), bottom[:-1][change])
coded = ms | np.r_[change, False] | np.r_[False, change] | (np.random.default_rng(1).random(len(ms)) < 0.1)
field = cs.Variogram([("spherical", 1.0, 150.0)], rotation=(16, 26, 90), ratios=(0.45, 0.22))
lens = cs.ImplicitModel("kriging", variogram=field, drift_degree=0).fit(
    xyz[coded], np.where(ms[coded], 1.0, -1.0), boundaries=contacts
)

keep = ms & ~np.isnan(zn)
xyz, zn, holes, length = xyz[keep], zn[keep], hole[keep], length[keep]
hull = cs.convex_hull(xyz)
lo, hi = hull.bounds
origin = np.floor(np.array(lo) / 5) * 5
count = tuple(int(c) for c in np.ceil((np.array(hi) - origin) / 5))
grid = cs.BlockModel(origin=tuple(origin), size=(5, 5, 5), count=count)
in_lens = lens.evaluate(grid.centroids) > 0
in_hull = hull.contains(grid.centroids)
blocks = grid.mask(in_lens & in_hull)
print(
    f"{len(contacts)} contacts; {in_lens.sum()} blocks in the lens, {len(blocks)} of them inside the hull "
    f"({len(blocks) * 125 / 1e3:.0f} thousand m3)"
)
```

</details>

```text
582 contacts; 2736 blocks in the lens, 2068 of them inside the hull (258 thousand m3)
```

## Statistics and declustering

Holes cluster where the lens is rich, so cell declustering weights the composites before any statistic. Capping
at 31 % Zn would cut 2.7 % of the composites and 1.7 % of the metal; instead they stay, restricted in the search
below.

<details><summary>Python</summary>

```python
weights = cs.cell_declustering(xyz, zn, sizes=np.arange(5, 80, 5)).weights
weights /= weights.mean()
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
mean 9.03 % Zn, declustered 9.23 %, CV 0.93
cap  21.9 % Zn: 10.6% of composites cut,  5.9% of the metal removed
cap  24.6 % Zn:  5.0% of composites cut,  3.9% of the metal removed
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
Variogram(nugget=0, structures=[Structure("spherical", sill=1, range=29.377377682207207)], rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0))
```

![variogram](variogram.png)

## Kriging in passes

The first pass wants eight composites within 30 m, about the variogram range, from at least two holes; blocks it
leaves go to a 60 m pass. In both, composites above 30 % Zn inform only blocks within 15 m, so a rich intercept
does not spread through the lens.

<details><summary>Python</summary>

```python
def passes(high_grade=None):
    return [
        cs.Search(radius=r, max_samples=16, min_samples=m, max_per_hole=4, high_grade=high_grade)
        for r, m in ((30, 8), (60, 4))
    ]


ok = cs.OrdinaryKriging(grades, passes(high_grade=(30.0, 15.0))).fit(xyz, zn, holes=holes)
kriged = ok.predict(blocks, diagnostics=True)
free = cs.OrdinaryKriging(grades, passes()).fit(xyz, zn, holes=holes).predict(blocks)
bias = cs.global_bias(kriged["value"], zn, data_weights=weights)
print(f"pass 1: {np.mean(kriged['pass'] == 1):.0%} of blocks, pass 2: {np.mean(kriged['pass'] == 2):.0%}")
print(
    f"mean {bias['estimate_mean']:.2f} % Zn against {bias['data_mean']:.2f} % declustered; "
    f"{np.mean(free):.2f} % without the high-grade restriction"
)
```

</details>

```text
pass 1: 80% of blocks, pass 2: 20%
mean 9.29 % Zn against 9.23 % declustered; 9.41 % without the high-grade restriction
```

## Simulation at block support

Thirty sequential Gaussian simulations on 2.5 m nodes, eight per block (`discretize(2)`), with the same high-grade
restriction. `blocks=` averages each realization over the nodes of every 5 m block before summarizing, so the
probability above 10 % Zn is that of the block grade, which is what a stope mines.

<details><summary>Python</summary>

```python
nodes = blocks.discretize(2)

search = cs.Search(radius=60, max_samples=16, max_per_hole=4, high_grade=(30.0, 15.0))
sgs = cs.SGS(gaussian, search).fit(xyz, zn, weights=weights, holes=holes)
summary = sgs.simulate(nodes, n=30, seed=1, cutoffs=[10.0], blocks=blocks)
low, high = np.quantile(summary.realization_above[0], [0.1, 0.9])
print(f"{len(nodes)} nodes in {len(blocks)} blocks")
print(
    f"blocks above 10 % Zn: P10 {low:.1%}, P90 {high:.1%} of the lens; kriged {np.mean(kriged['value'] > 10):.1%}"
)

fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
cs.plot.swath(
    [
        cs.swath(xyz, zn, 20.0, axis="z", weights=weights),
        cs.swath(blocks.centroids, kriged["value"], 20.0, axis="z"),
        cs.swath(blocks.centroids, summary.mean, 20.0, axis="z"),
    ],
    labels=["declustered composites", "kriged blocks", "mean of 30 simulations"],
    ax=ax,
)
ax.set(xlabel="Elevation (m)", ylabel="Zn (%)", title="Swath by elevation")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
save(fig, "swath")
```

</details>

```text
16544 nodes in 2068 blocks
blocks above 10 % Zn: P10 38.1%, P90 45.8% of the lens; kriged 38.2%
```

![swath](swath.png)

By elevation the kriged blocks and the mean of the simulations agree, and follow the composites more smoothly, as
they should, damping the rich level at 860–880 m.

## Classification

Measured blocks come from the first pass with a slope of regression of at least 0.8; indicated, from either pass
with a slope of 0.5. A 3 × 3 × 3 majority filter removes isolated blocks.

<details><summary>Python</summary>

```python
rules = [
    ("measured", {"pass": ("<=", 1), "slope": (">=", 0.8)}),
    ("indicated", {"slope": (">=", 0.5)}),
]
classes = cs.classify(kriged, rules, default="inferred")
classes = cs.smooth_classes(blocks, classes, window=(3, 3, 3))
names = ["measured", "indicated", "inferred"]
for name in names:
    inside = classes == name
    print(f"{name:>9}: {inside.mean():5.1%} of blocks, mean {np.nanmean(kriged['value'][inside]):5.2f} % Zn")
```

</details>

```text
 measured: 50.7% of blocks, mean  9.61 % Zn
indicated: 13.7% of blocks, mean  8.95 % Zn
 inferred: 35.6% of blocks, mean  8.97 % Zn
```

Drill spacing gives a second opinion that does not depend on the variogram fit. `hole_distance` takes, for each
block, the nearest composite of every hole, and averages the distances to the `n` nearest holes, so a hole with
many composites near a block still counts once. Measured blocks are on average within half the variogram range
(15 m) of three holes; indicated, within the range (30 m) of two; holes beyond the 60 m of the second pass do not
count.

<details><summary>Python</summary>

```python
spacing = cs.hole_distance(blocks, xyz, holes, n=[2, 3], search=cs.Search(radius=60))
rules = [
    ("measured", {"three": ("<=", 15)}),
    ("indicated", {"two": ("<=", 30)}),
]
by_distance = cs.classify({"two": spacing[2], "three": spacing[3]}, rules, default="inferred")
by_distance = cs.smooth_classes(blocks, by_distance, window=(3, 3, 3))
print(f"{'':>9}  " + "".join(f"{n:>10}" for n in names) + "   (rows: pass and slope, columns: distance)")
for name in names:
    print(f"{name:>9}: " + "".join(f"{np.mean((classes == name) & (by_distance == m)):10.1%}" for m in names))
print(f"same class for {np.mean(classes == by_distance):.0%} of blocks")
```

</details>

```text
             measured indicated  inferred   (rows: pass and slope, columns: distance)
 measured:      50.3%      0.4%      0.0%
indicated:       9.1%      4.5%      0.0%
 inferred:       1.8%     29.0%      4.8%
same class for 60% of blocks
```

Where the slope is high, spacing agrees: almost every block measured by slope is measured by distance too. Spacing
is the more generous of the two elsewhere, placing most blocks the slope leaves inferred within 30 m of two holes.
Most intercepts give a hole one composite in the lens, so a block near two holes may still rest on few data; the
slope sees that and the distance does not. Rules can combine both, as in chapter 18.

On the east–west section through the middle of the lens, the simulated block grade is drawn with
`plot.uncertain`: the mean of the realizations sets the colour and their standard deviation, over the spread of
all simulated blocks, fades it to white. Blocks along the holes keep their colour; the western ones, informed
only by composites off the section, fade almost to white, and are the inferred ones.

<details><summary>Python</summary>

```python
std_all = np.sqrt(summary.variance.mean() + summary.mean.var())
blocks = (
    blocks.with_column("zn", kriged["value"])
    .with_column("mean", summary.mean)
    .with_column("uncertainty", summary.std / std_all)
    .with_column("p_above_10", summary.probability_above[0])
    .with_column("class", np.select([classes == n for n in names], range(3)).astype(float))
)
rows = blocks.index // count[0] % count[1]
row = int(np.bincount(rows).argmax())
on_row = blocks.centroids[rows == row]
xlim, ylim = ((on_row[:, j].min() - 12.5, on_row[:, j].max() + 12.5) for j in (0, 2))
north = on_row[0, 1]
on_section = np.abs(xyz[:, 1] - north) < 5

fig = plt.figure(figsize=(11, 6.4), layout="constrained")
axes = fig.subplots(2, 4, height_ratios=[4, 1])
grade = plt.Normalize(0, 25)
cs.plot.section(blocks, "zn", axis="y", index=row, ax=axes[0, 0], colorbar=False, cmap="viridis", norm=grade)
cs.plot.uncertain(
    "mean",
    "uncertainty",
    block_model=blocks,
    axis="y",
    index=row,
    norm=grade,
    label="Zn (%)",
    legend_ax=axes[1, 1],
    ax=axes[0, 1],
)
cs.plot.section(blocks, "p_above_10", axis="y", index=row, ax=axes[0, 2], colorbar=False, vmin=0, vmax=1)
colors = ListedColormap([ACCENT, "#9ebad6", LIGHT])
cs.plot.section(
    blocks, "class", axis="y", index=row, ax=axes[0, 3], colorbar=False, cmap=colors, vmin=-0.5, vmax=2.5
)
titles = ("Kriged Zn", "Simulated block Zn", "P(block Zn > 10 %)", "Class")
for ax, title in zip(axes[0], titles, strict=True):
    ax.scatter(xyz[on_section, 0], xyz[on_section, 2], s=4, color=HIGHLIGHT, linewidths=0)
    ax.set(title=title, xlabel="Easting (m)", ylabel="Elevation (m)", xlim=xlim, ylim=ylim)
for ax, image, label in (
    (axes[1, 0], axes[0, 0].images[0], "Zn (%)"),
    (axes[1, 2], axes[0, 2].images[0], "probability"),
):
    ax.axis("off")
    fig.colorbar(image, cax=ax.inset_axes([0.1, 0.6, 0.8, 0.15]), orientation="horizontal", label=label)
axes[1, 3].axis("off")
handles = [plt.Line2D([], [], marker="s", ls="", color=colors(i), label=n) for i, n in enumerate(names)]
axes[1, 3].legend(handles=handles, loc="upper center", fontsize=8)
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
2068 blocks, columns ['zn', 'mean', 'uncertainty', 'p_above_10', 'class'], same Zn: True
```

Full script: [`example_20.py`](example_20.py)
