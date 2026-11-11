# From drill holes to a classified model

A resource estimate of three stacked sulfide lenses, end to end: drill-hole tables to composites inside each
lens wireframe, declustered statistics and a capping check, the Zn variogram, a sub-blocked model built from the
wireframes, Zn and density kriged in search passes, block-support simulation for risk, validation, classification,
and tonnes and metal per lens saved to Parquet.

<details><summary>Python</summary>

```python
import tempfile

import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, LIGHT, save
```

</details>

## Drill holes

Collars, surveys and assays become desurveyed drill holes; each assay sits at the midpoint of its interval.

<details><summary>Python</summary>

```python
data = bt.datasets.stacked_sulphide_lenses()
drillholes = bt.Drillholes.from_tables(data)
samples = drillholes.samples()
print(f"{len(drillholes)} holes, {len(samples)} assays")
```

</details>

```text
289 holes, 19247 assays
```

## Samples inside each lens

Each lens wireframe is a closed mesh, so it has an inside. Each assay takes the name of the lens around its
midpoint, and 2 m composites never cross from one lens to another or into the host rock.

<details><summary>Python</summary>

```python
lenses = {name: data[name] for name in ("lens_1", "lens_2", "lens_3")}
for name, mesh in lenses.items():
    print(f"{name}: closed {mesh.is_closed}, {mesh.volume / 1e6:.2f} Mm3")

lens = np.full(len(samples), "host", dtype=object)
for name, mesh in lenses.items():
    lens[mesh.contains(samples.coords)] = name
intervals = samples.with_columns({"LENS": list(lens)}).attributes
drillholes = bt.Drillholes(data["collars"], data["surveys"], intervals)
composites = drillholes.composite(2.0, ["ZN_PCT", "DENSITY"], domain="LENS", residual="merge")
composites = composites.filter(composites["LENS"] != "host")
names = composites["LENS"]
for name in lenses:
    inside = names == name
    measured = np.isfinite(composites["DENSITY"][inside]).sum()
    holes = len(set(composites["HOLE_ID"][inside]))
    print(f"{name}: {inside.sum()} composites from {holes} holes, {measured} with a density")
```

</details>

```text
lens_1: closed True, 2.88 Mm3
lens_2: closed True, 1.91 Mm3
lens_3: closed True, 1.67 Mm3
lens_1: 792 composites from 112 holes, 434 with a density
lens_2: 427 composites from 78 holes, 298 with a density
lens_3: 275 composites from 50 holes, 182 with a density
```

## Declustering and capping

Holes cluster where a lens is rich, so each lens is cell-declustered on its own. The capping table then shows how
much metal the top composites carry.

<details><summary>Python</summary>

```python
weights = np.zeros(len(composites))
for name in lenses:
    inside = names == name
    zn = composites["ZN_PCT"][inside]
    declustering = bt.cell_declustering(composites.coords[inside], zn, sizes=np.arange(10, 105, 5))
    weights[inside] = declustering.weights / declustering.weights.mean()
    print(
        f"{name}: mean {zn.mean():.2f} % Zn, declustered {declustering.mean:.2f} % ({declustering.cell_size:.0f} m cells)"
    )
composites = composites.with_column("weight", weights)

caps = bt.capping("ZN_PCT", weights="weight", data=composites)
for cap, fraction, removed in zip(caps["cap"], caps["fraction"], caps["metal_removed"], strict=True):
    print(f"cap {cap:5.1f} % Zn: {fraction:5.1%} of composites cut, {removed:5.1%} of the metal removed")

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
bt.plot.histogram(
    "ZN_PCT", weights="weight", data=composites, bins=np.arange(0, 38, 2), ax=a, color=LIGHT, edgecolor=GRAY
)
a.set(xlabel="Zn (%)", title="Declustered histogram, three lenses")
bt.plot.probability("ZN_PCT", weights="weight", data=composites, log=True, ax=b, color=ACCENT, ms=3)
b.set(xlabel="Zn (%)", title="Probability plot")
save(fig, "statistics")
```

</details>

```text
lens_1: mean 5.38 % Zn, declustered 5.18 % (25 m cells)
lens_2: mean 6.32 % Zn, declustered 5.84 % (25 m cells)
lens_3: mean 7.06 % Zn, declustered 6.87 % (10 m cells)
cap  11.3 % Zn: 10.0% of composites cut,  8.8% of the metal removed
cap  14.9 % Zn:  5.0% of composites cut,  4.4% of the metal removed
cap  18.7 % Zn:  2.5% of composites cut,  2.0% of the metal removed
cap  23.3 % Zn:  1.0% of composites cut,  0.8% of the metal removed
cap  27.3 % Zn:  0.5% of composites cut,  0.3% of the metal removed
cap  33.2 % Zn:  0.1% of composites cut,  0.0% of the metal removed
```

![statistics](statistics.png)

Declustering lowers every lens mean, most in lens 1. Capping the top 1 % of composites at 23.3 % Zn removes 0.8 %
of the metal, and the probability plot shows no break in the upper tail: leave the grades uncapped.

## Zn variogram of lens 1

Lens 1 has the most composites. A thin lens leaves few pairs across it, so the variogram is omnidirectional;
kriging uses the variogram of the grades, simulation that of the declustered normal scores, rescaled to a unit
sill. Both are fitted up to 60 m, just past the 45 m where γ first reaches its plateau: the longer lags hold three
to four times as many pairs, so with the fit weighted by pair count their scatter about the plateau would set the
model and stretch the range to the end of the lags.

<details><summary>Python</summary>

```python
one = composites.filter(names == "lens_1")
grades = bt.experimental_variogram(one, "ZN_PCT", 10.0, 150.0)
variogram = bt.experimental_variogram(one, "ZN_PCT", 10.0, 60.0).fit("spherical")
scores = bt.NormalScore().fit(one["ZN_PCT"], weights=one["weight"])
fitted = bt.experimental_variogram(one.coords, scores.transform(one["ZN_PCT"]), 10.0, 60.0).fit("spherical")
gaussian = fitted.standardized()
for label, model in (("Zn", variogram), ("normal scores", gaussian)):
    s = model.structures[0]
    print(f"{label}: nugget {model.nugget:.2f}, spherical sill {s.sill:.2f}, range {s.range:.0f} m")

fig, ax = bt.plot.variogram(grades, variogram=variogram, color=ACCENT)
ax.set(xlabel="Lag distance (m)", ylabel="γ(h), Zn (%²)", title="Zn variogram, lens 1")
save(fig, "variogram")
```

</details>

```text
Zn: nugget 6.86, spherical sill 8.25, range 56 m
normal scores: nugget 0.50, spherical sill 0.50, range 54 m
```

![variogram](variogram.png)

## A sub-blocked model

Parent blocks of 10 m are split into 5 m sub-blocks wherever a wireframe cuts them, so the model keeps each
lens's volume; blocks outside every lens are dropped.

<details><summary>Python</summary>

```python
low = np.min([m.bounds[0] for m in lenses.values()], axis=0)
high = np.max([m.bounds[1] for m in lenses.values()], axis=0)
origin = np.floor(low / 10) * 10
count = [int(c) for c in np.ceil((high - origin) / 10)]
blocks = bt.BlockModel.from_meshes(
    origin, (10, 10, 10), count, [(mesh, "inside", name) for name, mesh in lenses.items()], 2, column="LENS"
)
block_lens = blocks["LENS"]
for name, mesh in lenses.items():
    volume = blocks.volumes[block_lens == name].sum()
    print(
        f"{name}: {np.sum(block_lens == name)} blocks, {volume / 1e6:.3f} Mm3 against {mesh.volume / 1e6:.3f} Mm3"
    )
```

</details>

```text
lens_1: 7220 blocks, 2.880 Mm3 against 2.878 Mm3
lens_2: 6057 blocks, 1.910 Mm3 against 1.908 Mm3
lens_3: 5424 blocks, 1.664 Mm3 against 1.665 Mm3
```

## Kriging in passes

Each lens is kriged from its own composites (`domain_column`). The first pass wants six composites within the
variogram range, at most two per hole, so at least three holes; blocks it leaves go to a pass at twice the range,
and a last one reaches every block. Two per hole matters here: holes with long intercepts tend to be the richer
ones, and would otherwise fill the search with their own composites. Density, measured on fewer samples, is
kriged the same way with the shape of the Zn variogram: kriging weights do not depend on the sill.

<details><summary>Python</summary>

```python
reach = variogram.structures[0].range
passes = [
    bt.Search(reach, max_samples=12, min_samples=6, max_per_hole=2),
    bt.Search(2 * reach, max_samples=12, min_samples=4, max_per_hole=2),
    bt.Search(250, max_samples=12, max_per_hole=2),
]
zn = bt.OrdinaryKriging(variogram, passes).fit(composites, "ZN_PCT", holes="HOLE_ID", domain_column="LENS")
kriged = zn.predict(blocks, diagnostics=True, domain_column="LENS")
measured = composites.drop_null("DENSITY")
density = bt.OrdinaryKriging(variogram, passes).fit(
    measured, "DENSITY", holes="HOLE_ID", domain_column="LENS"
)
blocks = blocks.with_columns(
    {
        "zn": kriged["value"],
        "density": density.predict(blocks, domain_column="LENS"),
        "pass": kriged["pass"],
        "slope": kriged["slope"],
    }
)
for p in (1, 2, 3):
    print(f"pass {p}: {np.mean(kriged['pass'] == p):.0%} of blocks")
print(f"density {np.nanmin(blocks['density']):.2f}-{np.nanmax(blocks['density']):.2f} t/m3")
```

</details>

```text
pass 1: 78% of blocks
pass 2: 22% of blocks
pass 3: 0% of blocks
density 2.92-4.54 t/m3
```

## Simulation at block support

Kriging smooths; the risk in a stope comes from simulation. Thirty sequential Gaussian simulations of lens 1
run on 5 m nodes, eight to each 10 m parent block the lens touches, and `blocks=` averages each realization
over those blocks, so the probability above 5 % Zn is that of a 10 m block. The first two kriging passes serve as
the simulation's search.

<details><summary>Python</summary>

```python
grid = bt.BlockModel(origin, (10, 10, 10), count)
parents = grid.filter(np.isin(np.arange(len(grid)), blocks.index[block_lens == "lens_1"]))
nodes = parents.discretize(2)
sgs = bt.SGS(gaussian, passes[:2]).fit(one, "ZN_PCT", weights="weight", holes="HOLE_ID")
summary = sgs.simulate(nodes, n=30, seed=1, cutoffs=[5.0], blocks=parents)
low, high = np.quantile(summary.realization_above[:, 0], [0.1, 0.9])
print(f"{len(parents)} parent blocks: P10 {low:.0%}, P90 {high:.0%} of them above 5 % Zn")
sure = np.mean(summary.probability_above[:, 0] > 0.9)
print(f"mean {summary.mean.mean():.2f} % Zn; blocks above 5 % in more than 90 % of realizations: {sure:.0%}")
```

</details>

```text
4394 parent blocks: P10 36%, P90 46% of them above 5 % Zn
mean 4.96 % Zn; blocks above 5 % in more than 90 % of realizations: 1%
```

In eight realizations out of ten, between 36 and 46 % of the parent blocks exceed 5 % Zn, yet only 1 % of them do
so in more than 90 % of the realizations: at this drill spacing few single blocks are certain ore, although the
share of ore across the lens is well known.

## Validation

The kriged blocks, weighted by volume, should match the declustered composites of each lens, and a nearest-neighbor
model, the other unbiased reference, and follow them along strike and down the lens.

<details><summary>Python</summary>

```python
nearest = bt.NearestNeighbor(bt.Search(250, max_samples=1)).fit(composites, "ZN_PCT", domain_column="LENS")
blocks = blocks.with_column("nn", nearest.predict(blocks, domain_column="LENS"))
for name in lenses:
    inside = block_lens == name
    bias = bt.global_bias(
        blocks["zn"][inside],
        composites["ZN_PCT"][names == name],
        weights=blocks.volumes[inside],
        data_weights=weights[names == name],
    )
    nn = np.average(blocks["nn"][inside], weights=blocks.volumes[inside])
    print(
        f"{name}: kriged {bias['estimate_mean']:.2f} % Zn, declustered composites {bias['data_mean']:.2f} % "
        f"({bias['relative']:+.1%}), nearest neighbor {nn:.2f} % ({bias['estimate_mean'] / nn - 1:+.1%})"
    )

in_one = blocks.filter(block_lens == "lens_1")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
for ax, (axis, label) in zip(axes, (("y", "Northing (m)"), ("z", "Elevation (m)")), strict=True):
    bt.plot.swath(
        [
            bt.swath(one, "ZN_PCT", 40.0, axis=axis),
            bt.swath(in_one, "nn", 40.0, axis=axis, weights=in_one.volumes),
            bt.swath(in_one, "zn", 40.0, axis=axis, weights=in_one.volumes),
        ],
        labels=["Composites", "Nearest neighbor", "Kriged blocks"],
        ax=ax,
    )
    ax.set(xlabel=label, ylabel="Zn (%)")
    ax.get_legend().remove()
axes[0].set_title("Lens 1 swaths")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
save(fig, "swath")
```

</details>

```text
lens_1: kriged 4.80 % Zn, declustered composites 5.18 % (-7.2%), nearest neighbor 4.96 % (-3.2%)
lens_2: kriged 5.74 % Zn, declustered composites 5.84 % (-1.8%), nearest neighbor 5.67 % (+1.2%)
lens_3: kriged 6.49 % Zn, declustered composites 6.87 % (-5.6%), nearest neighbor 6.27 % (+3.5%)
```

![swath](swath.png)

In lens 1 the kriged mean lies 3 % below the nearest-neighbor model and 7 % below the declustered composites; in
lenses 2 and 3 it is 2 and 6 % below the declustered composites but 1 and 4 % above nearest neighbor. The two references disagree with each other as much as with the model: nearest neighbor spreads each
edge composite over the blocks around it, and cell declustering depends on the cell size it settles on. Along
northing and elevation, kriged blocks and nearest neighbor follow the same course, and the composites swing
around both, as raw data do.

## Classification

Measured blocks come from the first pass with a slope of regression of at least 0.6; indicated, from the first
two passes with a slope of 0.3; the rest is inferred. With a nugget of 45 % of the sill and a 56 m range, only
blocks close to several holes qualify as measured.

<details><summary>Python</summary>

```python
rules = [
    ("measured", {"pass": ("<=", 1), "slope": (">=", 0.6)}),
    ("indicated", {"pass": ("<=", 2), "slope": (">=", 0.3)}),
]
classes = bt.classify(kriged, rules, default="inferred")
categories = bt.Categories(["measured", "indicated", "inferred"], colors=[ACCENT, "#9ebad6", LIGHT])
blocks = blocks.with_column("class", categories.encode(classes))
for name in categories.names:
    print(f"{name:>9}: {np.mean(classes == name):.0%} of blocks")
```

</details>

```text
 measured: 36% of blocks
indicated: 42% of blocks
 inferred: 21% of blocks
```

Lens 1 is seen face on, on the plane through its middle, with its outline on that plane and its composites as
dots. The plane's pole is the least spread direction of the wireframe's vertices.

<details><summary>Python</summary>

```python
parents = parents.with_column("p_above_5", summary.probability_above[:, 0])
vertices = lenses["lens_1"].coords
pole = np.linalg.eigh(np.cov(vertices.T))[1][:, 0]
pole *= np.sign(pole[2])
strike = (np.degrees(np.arctan2(pole[0], pole[1])) - 90) % 360
dip = np.degrees(np.arccos(pole[2]))
print(f"lens 1 strikes {strike:03.0f}°, dips {dip:.0f}°")
plane = (tuple(vertices.mean(axis=0)), strike, dip)
fig, axes = plt.subplots(3, 1, figsize=(7, 10), layout="constrained", sharex=True)
bt.plot.section(blocks, "zn", plane=plane, ax=axes[0], colorbar=False, vmin=0, vmax=12)
bt.plot.section(parents, "p_above_5", plane=plane, ax=axes[1], colorbar=False, vmin=0, vmax=1)
bt.plot.section(blocks, "class", plane=plane, ax=axes[2], colorbar=False, scheme=categories)
for ax in axes[:2]:
    fig.colorbar(ax.images[0], cax=ax.inset_axes([0.7, 1.04, 0.28, 0.04]), orientation="horizontal")
bt.plot.category_legend(categories, axes[2], loc="lower right", bbox_to_anchor=(1, 1), ncol=3)
titles = ("Kriged Zn (%)", "P(10 m block Zn > 5 %)", "Class")
for ax, title in zip(axes, titles, strict=True):
    bt.plot.slab(one, plane=plane, thickness=60, meshes=[lenses["lens_1"]], s=3, color=HIGHLIGHT, ax=ax)
    ax.set(title=title, aspect="equal")
save(fig, "section")
```

</details>

```text
lens 1 strikes 024°, dips 56°
```

![section](section.png)

## Tonnes and metal

Tonnes are block volume × kriged density and metal is tonnes × Zn, per lens and class, in all of each lens and
above a 5 % Zn cutoff.

<details><summary>Python</summary>

```python
print(f"{'':20}{'kt':>8}{'Zn %':>7}{'kt Zn':>8}")
for groups in (block_lens, classes):
    table = bt.grade_tonnage(
        "zn", [0.0, 5.0], weights=blocks.volumes, density="density", categories=groups, data=blocks
    )
    for category, cutoff, tonnes, grade, metal in zip(
        *(table[c] for c in ("category", "cutoff", "tonnage", "mean_grade", "metal")), strict=True
    ):
        print(f"{category:10} ≥ {cutoff:1.0f} % Zn{tonnes / 1e3:8.0f}{grade:7.2f}{metal / 1e5:8.1f}")
```

</details>

```text
                          kt   Zn %   kt Zn
lens_1     ≥ 0 % Zn    9714   4.93   479.0
lens_1     ≥ 5 % Zn    3767   7.16   269.8
lens_2     ≥ 0 % Zn    6602   5.95   392.6
lens_2     ≥ 5 % Zn    3762   7.76   291.9
lens_3     ≥ 0 % Zn    5975   6.63   396.4
lens_3     ≥ 5 % Zn    4402   7.58   333.7
all        ≥ 0 % Zn   22291   5.69  1268.1
all        ≥ 5 % Zn   11931   7.50   895.4
indicated  ≥ 0 % Zn    9279   5.65   524.1
indicated  ≥ 5 % Zn    4880   7.48   364.8
inferred   ≥ 0 % Zn    4112   5.54   228.0
inferred   ≥ 5 % Zn    2256   7.05   159.0
measured   ≥ 0 % Zn    8899   5.80   515.9
measured   ≥ 5 % Zn    4796   7.75   371.5
all        ≥ 0 % Zn   22291   5.69  1268.1
all        ≥ 5 % Zn   11931   7.50   895.4
```

The three lenses hold 22.3 Mt at 5.69 % Zn, 1.27 Mt of zinc, of which 11.9 Mt at 7.50 % lie above 5 % Zn. Lens 1
carries 38 % of the metal. Measured blocks, 8.9 Mt at 5.80 % Zn, gather around the holes; inferred ones fill the
edges of the lenses and the gaps between holes.

The model, sub-blocks, lens names, grades, classes and all, goes to Parquet and back unchanged.

<details><summary>Python</summary>

```python
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "lenses.parquet"
    bt.write_parquet(path, blocks)
    stored = bt.read_parquet(path)
print(f"{len(stored)} blocks, columns {stored.column_names}")
same = np.array_equal(stored.extents, blocks.extents) and np.array_equal(stored["zn"], blocks["zn"])
print(f"same sub-blocks and Zn: {same}")
```

</details>

```text
18701 blocks, columns ['LENS', 'zn', 'density', 'pass', 'slope', 'nn', 'class']
same sub-blocks and Zn: True
```

Full script: [`example_12_02.py`](example_12_02.py)
