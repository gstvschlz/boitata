# 88. Stepped sections

`cs.plot.fence` slices a block model on several parallel planes stepped along a corridor, all sharing one color
scale: a fence of cross sections through the stacked sulphide lenses, replacing the hand-rolled `plt.subplots` loop
of topic 84's maps with a single call.

<details><summary>Python</summary>

```python
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import save
```

</details>

## A grade model across the lenses

Zinc kriged from the 5 m composites (as in topic 85) onto a block model spanning the deposit, cells of 20 m,
masked to the topography.

<details><summary>Python</summary>

```python
data = cs.datasets.stacked_sulphide_lenses()
intervals = cs.merge_intervals(data["assays"], data["lithology"])
drillholes = cs.Drillholes(data["collars"], data["surveys"], intervals)
composites = drillholes.composite(5.0, ["ZN_PCT"], categories=["LITH"])
known = composites.filter(~np.isnan(composites["ZN_PCT"]))

layers = (23.0, 55.0, 0.0)
variogram = cs.Variogram([("spherical", 0.8, 200.0)], nugget=0.2, rotation=layers, ratios=(0.8, 0.2))
search = cs.Search(300.0, max_samples=24, max_per_hole=6, rotation=layers, ratios=(1.0, 0.4))

center = np.array([12350.0, 29900.0, 0.0])
across = np.array([np.sin(np.radians(113.0)), np.cos(np.radians(113.0)), 0.0])
along = np.array([np.sin(np.radians(23.0)), np.cos(np.radians(23.0)), 0.0])
origin = center - 600.0 * across - 300.0 * along + (0.0, 0.0, -400.0)
model = cs.BlockModel(origin, (20.0, 20.0, 20.0), (60, 30, 40), rotation=(23.0, 0.0, 0.0))
topography = data["topography"]
rows = topography.row_at(model.centroids[:, :2])
model = model.mask((rows >= 0) & (model.centroids[:, 2] < topography["Z"][rows]))
kriged = cs.OrdinaryKriging(variogram, search).fit(known, "ZN_PCT", holes="HOLE_ID").predict(model)
model = model.mask(~np.isnan(kriged)).with_columns({"ZN_PCT": kriged[~np.isnan(kriged)]})
print(f"{len(model):,} blocks of 20 m")
```

</details>

```text
51,013 blocks of 20 m
```

## The old way: a manual loop

Topic 84 drew its two maps by hand: `plt.subplots(1, n, sharey=True)`, one `cs.plot.section(..., colorbar=False,
ax=ax)` per panel, `vmin`/`vmax` matched by hand across them, and one `fig.colorbar` at the end reading the last
panel's image. The same recipe, repeated for five sections stepped 125 m apart along strike:

<details><summary>Python</summary>

```python
positions = [center + t * along for t in np.linspace(-250.0, 250.0, 5)]
fig, axes = plt.subplots(1, len(positions), figsize=(13, 4), layout="constrained", sharey=True)
for ax, position in zip(axes, positions, strict=True):
    cs.plot.section(model, "ZN_PCT", plane=(position, 113.0, 90.0), vmin=0, vmax=8, colorbar=False, ax=ax)
for ax in axes[1:]:
    ax.set_ylabel("")
fig.colorbar(axes[-1].images[0], ax=axes, shrink=0.8, label="ZN_PCT")
save(fig, "manual")
```

</details>

![manual](manual.png)

## `cs.plot.fence`

The same five sections in one call: `azimuth` and `dip` shared by every panel, and `vmin`/`vmax` found across
all of them unless given, as `section` takes them.

<details><summary>Python</summary>

```python
fig, axes = cs.plot.fence(model, "ZN_PCT", positions, azimuth=113.0, dip=90.0)
save(fig, "fence")
```

</details>

![fence](fence.png)

Full script: [`example_88.py`](example_88.py)
