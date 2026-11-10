# MIK localization

Multiple indicator kriging reaches the selective blocks inside a panel without a Gaussian model. Kriged at each panel
centroid, its conditional distribution describes point grades; an affine correction shrinks it to block support, and
ranked blocks share its bands. The exhaustive metal survey grid gives the true 10 m blocks.

<details><summary>Python</summary>

```python
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, INK, save

samples = bt.datasets.clustered_metal_survey()
truth = bt.datasets.clustered_metal_survey_exhaustive()["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
weights = bt.cell_declustering(samples, "V", sizes=np.arange(2.5, 102.5, 2.5)).weights
size = 10
true_smu = truth[:, :250].reshape(30, size, 25, size).mean(axis=(1, 3))
```

</details>

The grade variogram, fitted along and across N170°
([variogram fitting](../../05-spatial-continuity/02-variogram-fitting/README.md)), and one omnidirectional indicator
variogram at each decile:

<details><summary>Python</summary>

```python
azimuths = (170, 260)
experimental = [bt.experimental_variogram(xy, v, 10, 120, azimuth=a) for a in azimuths]
grade = bt.Variogram.fit_directional(
    experimental, [(a, 0) for a in azimuths], ["spherical", "spherical"], rotation=[170, 0, 0]
)
deciles = np.quantile(v, np.linspace(0.1, 0.9, 9))
indicators = [
    bt.experimental_variogram(xy, (v <= t).astype(float), 10, 120).fit("spherical") for t in deciles
]
print(grade)
```

</details>

```text
Variogram  sill 78396.3, 2 structures
  rotation  azimuth 170°, dip 0°, rake 0°
  ratios    semi/major 0.4316, minor/major 1
  structure     sill  major   semi  minor
  nugget     11558.1
  spherical   8962.1  10.62  4.585  10.62
  spherical  57876.1     99  42.73     99
```

The 50 × 50 m panels over the western 250 m hold 25 blocks of 10 m each, ranked by their ordinary block kriging.
`localize` kriges the indicators at each panel centroid and shrinks the distribution about its mean by the variance
factor f: the variance of 10 m blocks within a panel over that of points within it, computed here from the grade
variogram. The block ranked i receives the mean of the i-th of 25 equal-probability bands.

<details><summary>Python</summary>

```python
search = bt.Search(radius=100, max_samples=24, min_samples=4)
panels = bt.BlockModel(origin=(0.5, 0.5), size=(50, 50), count=(5, 6))
smus = panels.discretize(5)
kriged = bt.BlockKriging(grade, search, size=(size, size), discretization=(5, 5, 1)).fit(xy, v).predict(smus)
smus = smus.with_column("kriged", kriged)
mik = bt.MultipleIndicatorKriging(indicators, search, deciles, tails=(0.0, v.max()))
mik.fit(xy, v, weights=weights)
localized = mik.localize(smus, "kriged", panels, variance_factor=grade)["localized"]
for label, values in (("kriged", kriged), ("localized", localized)):
    r = bt.compare(values, true_smu.ravel())["correlation"]
    print(f"{label}: variance {values.var():.0f}, correlation with truth {r:.2f}")
print(f"true blocks: variance {true_smu.var():.0f}")
```

</details>

```text
kriged: variance 41398, correlation with truth 0.92
localized: variance 54909, correlation with truth 0.85
true blocks: variance 50084
```

Grade-tonnage of the three sets of blocks:

<details><summary>Python</summary>

```python
cutoffs = np.linspace(0, 1000, 41)


def empirical(values):
    tonnage = np.array([(values >= c).mean() for c in cutoffs])
    grade = np.array([values[values >= c].mean() if (values >= c).any() else np.nan for c in cutoffs])
    return tonnage, grade


true_curve, kriged_curve, mik_curve = empirical(true_smu.ravel()), empirical(kriged), empirical(localized)
for c in (300, 500, 800):
    k = np.searchsorted(cutoffs, c)
    print(
        f"above {c} ppm: localized {mik_curve[0][k]:.1%}, kriged {kriged_curve[0][k]:.1%}, true {true_curve[0][k]:.1%}"
    )

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained", width_ratios=(1, 1.3))
extent = (0.5, 250.5, 0.5, 300.5)
image = a.imshow(np.reshape(localized, (30, 25)), origin="lower", extent=extent, vmin=0, vmax=1000)
a.vlines(np.arange(50.5, 250, 50), 0.5, 300.5, color="white", lw=0.6, alpha=0.7)
a.hlines(np.arange(50.5, 300, 50), 0.5, 250.5, color="white", lw=0.6, alpha=0.7)
a.set_aspect("equal")
a.set(title="Localized indicator kriging", xlabel="Easting (m)", ylabel="Northing (m)")
fig.colorbar(image, ax=a, shrink=0.8, label="V (ppm); white lines bound the 50 m panels")
for (tonnage, _), color, style, label in (
    (true_curve, INK, {"lw": 3, "alpha": 0.35}, "True blocks"),
    (mik_curve, ACCENT, {"lw": 1.4, "ls": "--"}, "Localized indicator kriging"),
    (kriged_curve, GRAY, {"lw": 1.2, "ls": ":"}, "Kriged blocks"),
):
    b.plot(cutoffs, tonnage, color=color, label=label, **style)
b.set(
    title=f"Proportion of {size} × {size} m blocks above cutoff",
    xlabel="Cutoff V (ppm)",
    ylabel="Proportion of blocks",
)
b.legend(fontsize=8)
save(fig, "localized-mik")
```

</details>

```text
above 300 ppm: localized 47.5%, kriged 42.3%, true 38.9%
above 500 ppm: localized 19.9%, kriged 14.0%, true 16.9%
above 800 ppm: localized 4.4%, kriged 1.6%, true 2.7%
```

![localized-mik](localized-mik.png)

The localized blocks spread wider than the true ones (variance 54 909 against 50 084), while kriging smooths them to
41 398. Block by block they follow the truth a little less well than kriging (correlation 0.85 against 0.92). The
affine correction keeps the shape of each point distribution, so its long upper tail survives the shrinking: at 500
ppm the localized blocks put 19.9% above cutoff against a true 16.9%, overshooting by as much as kriging falls short.
[Uniform conditioning](../../09-recoverable-resources/02-uniform-conditioning/README.md) localizes the same panels
from a Gaussian model.

Full script: [`example_09_03.py`](example_09_03.py)
