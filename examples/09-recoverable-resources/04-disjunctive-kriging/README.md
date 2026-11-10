# Disjunctive kriging

Disjunctive kriging estimates, at each node, any function of the grade from kriged Hermite factors of a Gaussian
anamorphosis; here, the probability that V exceeds a cutoff. The exhaustive metal survey grid shows where the grade
exceeds it.

<details><summary>Python</summary>

```python
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, INK, save

samples = bt.datasets.clustered_metal_survey()
truth = bt.datasets.clustered_metal_survey_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
weights = bt.cell_declustering(samples, "V", sizes=np.arange(2.5, 102.5, 2.5)).weights
```

</details>

A Hermite anamorphosis of the declustered samples, and the variogram of its Gaussian scores fitted along and across
N170°, the major axis found in [variogram fitting](../../05-spatial-continuity/02-variogram-fitting/README.md):

<details><summary>Python</summary>

```python
anam = bt.HermiteAnamorphosis(degree=40).fit(v, weights=weights)
y = anam.transform(v)
azimuths = (170, 260)
experimental = [bt.experimental_variogram(xy, y, 10, 120, azimuth=a) for a in azimuths]
gaussian = bt.Variogram.fit_directional(experimental, [(a, 0) for a in azimuths], rotation=[170, 0, 0])
print(gaussian)
```

</details>

```text
Variogram  sill 0.8982, 1 structure
  rotation  azimuth 170°, dip 0°, rake 0°
  ratios    semi/major 0.4627, minor/major 1
  structure    sill  major   semi  minor
  nugget     0.1314
  spherical  0.7668  103.1  47.68  103.1
```

`predict_tonnage` gives P(V > cutoff) on a 5 m grid. Binned against the truth, a calibrated estimate would sit on the
diagonal:

<details><summary>Python</summary>

```python
grid = bt.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
dk = bt.DisjunctiveKriging(anam, gaussian, bt.Search(radius=100, max_samples=24), order=20).fit(xy, v)
cutoff = 500.0
p = dk.predict_tonnage(grid, cutoff)
nodes = grid.coords.astype(int)
above = truth[nodes[:, 1] - 1, nodes[:, 0] - 1] > cutoff
edges = np.linspace(0, 1, 11)
bins = np.clip(np.digitize(p, edges) - 1, 0, 9)
counts = np.bincount(bins, minlength=10)
keep = counts > 20
predicted = np.array([p[bins == k].mean() if counts[k] else np.nan for k in range(10)])
observed = np.array([above[bins == k].mean() if counts[k] else np.nan for k in range(10)])
print(f"mean predicted P(V > {cutoff:.0f}) {p.mean():.3f}, true proportion {above.mean():.3f}")
for k in np.flatnonzero(keep):
    print(
        f"bin {edges[k]:.1f}–{edges[k + 1]:.1f}: {counts[k]} nodes, predicted {predicted[k]:.2f}, observed {observed[k]:.2f}"
    )

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
extent = (0.5, 260.5, 0.5, 300.5)
image = a.imshow(grid.grid(np.clip(p, 0, 1))[0], origin="lower", extent=extent, vmin=0, vmax=1)
a.contour(
    grid.grid(above.astype(float))[0],
    levels=[0.5],
    origin="lower",
    extent=extent,
    colors=HIGHLIGHT,
    linewidths=0.8,
)
a.set_aspect("equal")
a.set(title=f"P(V > {cutoff:.0f} ppm)", xlabel="Easting (m)", ylabel="Northing (m)")
fig.colorbar(image, ax=a, shrink=0.8, label=f"Probability; true V > {cutoff:.0f} outlined")
b.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
b.scatter(predicted[keep], observed[keep], s=np.sqrt(counts[keep]) * 4, color=ACCENT)
b.set(
    xlim=(0, 1),
    ylim=(0, 1),
    xlabel="Predicted probability",
    ylabel=f"Observed frequency at {p.size:,} nodes".replace(",", " "),
    title="Calibration",
)
b.set_aspect("equal")
b.text(0.03, 0.92, "Marker area ∝ nodes per bin", color=INK, fontsize=8)
save(fig, "disjunctive")
```

</details>

```text
mean predicted P(V > 500) 0.188, true proportion 0.187
bin 0.0–0.1: 1675 nodes, predicted 0.02, observed 0.01
bin 0.1–0.2: 426 nodes, predicted 0.15, observed 0.10
bin 0.2–0.3: 284 nodes, predicted 0.25, observed 0.24
bin 0.3–0.4: 186 nodes, predicted 0.35, observed 0.37
bin 0.4–0.5: 147 nodes, predicted 0.45, observed 0.44
bin 0.5–0.6: 106 nodes, predicted 0.54, observed 0.57
bin 0.6–0.7: 90 nodes, predicted 0.65, observed 0.74
bin 0.7–0.8: 79 nodes, predicted 0.75, observed 0.91
bin 0.8–0.9: 56 nodes, predicted 0.85, observed 0.95
bin 0.9–1.0: 71 nodes, predicted 0.98, observed 1.00
```

![disjunctive](disjunctive.png)

On average the estimate says 18.8% of the nodes exceed 500 ppm; 18.7% do. The error sits in the low bins, which hold
most nodes: where it predicts 0.1 to 0.4, you observe half to two thirds of that. From 0.5 to 0.9 the error reverses,
and more nodes exceed the cutoff than it predicts.

Full script: [`example_09_04.py`](example_09_04.py)
