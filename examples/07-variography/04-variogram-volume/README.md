# Variogram volume

A variogram map shows γ on one plane; in 3D the plane of best continuity is itself unknown. `variogram_volume` bins
every pair by its lag vector into a cube of cells, the 3D variogram map, and reads the principal axes of continuity from
a few hundred directions spread over the sphere. In each direction it finds the lag where γ reaches half the sill. Under
geometric anisotropy these lags trace an ellipsoid with the axes and ratios of the range ellipsoid, and its rotation and
ratios go straight into `Variogram`. The axes here come from Zn composites across three stacked sulfide lenses, compared
with the orientation of the lens solids.

<details><summary>Python</summary>

```python
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, save
```

</details>

## Composites and the lens orientation

Zn is high in the lenses and low in the rock around them, so the lenses are the continuous bodies. Composites inside
the lenses alone would say little about their orientation: a lens is a few meters thick, so every pair across it is a
downhole pair. The example keeps all 2 m composites and takes Zn in logarithms to tame its skew. The orientation of
each lens solid is the plane through its vertices, whose pole is the direction of least spread.

<details><summary>Python</summary>

```python
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes.from_tables(data)
composites = holes.composite(2.0, ["ZN_PCT"])
composites = composites.drop_null("ZN_PCT")
log_zn = np.log(composites["ZN_PCT"])
print(f"{len(composites)} composites")


def unit(azimuth, dip):
    az, dip = np.radians(azimuth), np.radians(dip)
    return np.array([np.cos(dip) * np.sin(az), np.cos(dip) * np.cos(az), -np.sin(dip)])


def plane(pole):
    """Dip and dip direction of the plane with this pole."""
    pole = pole if pole[2] > 0 else -pole
    return np.degrees(np.arccos(pole[2])), np.degrees(np.arctan2(pole[0], pole[1])) % 360


poles = []
for i in (1, 2, 3):
    vertices = data[f"lens_{i}"].coords
    pole = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)[2][2]
    poles.append(pole if pole[2] > 0 else -pole)
    print(f"lens {i}: dips {plane(pole)[0]:.0f} degrees towards {plane(pole)[1]:03.0f}")
lens_pole = np.mean(poles, axis=0)
lens_pole /= np.linalg.norm(lens_pole)
```

</details>

```text
16070 composites
lens 1: dips 56 degrees towards 114
lens 2: dips 54 degrees towards 112
lens 3: dips 55 degrees towards 112
```

## The volume and its axes

Lags of 15 m reach 225 m on each axis. Each direction sums the pairs within 15 degrees of it; a wider cone averages in
directions across the lenses and flattens the recovered dip.

<details><summary>Python</summary>

```python
volume = bt.variogram_volume(composites, log_zn, 15.0, 225.0, tolerance=15.0)
print("rotation (azimuth, dip, rake):", np.round(volume.rotation, 1))
print("ranges (major, semi, minor):  ", np.round(volume.ranges, 0))
print("ratios:                       ", np.round(volume.ratios, 2))
for name, (az, dip) in zip(("major", "semi-major", "minor"), volume.axes):
    print(f"{name:>10} axis: azimuth {az:05.1f}, dip {dip:4.1f}")

minor = unit(*volume.axes[2])
angle = np.degrees(np.arccos(abs(minor @ lens_pole)))
dip, direction = plane(minor)
lens_dip, lens_direction = plane(lens_pole)
print(f"plane of the major and semi-major axes: dips {dip:.0f} towards {direction:03.0f}")
print(f"mean lens plane:                        dips {lens_dip:.0f} towards {lens_direction:03.0f}")
print(f"minor axis to lens pole: {angle:.1f} degrees")
```

</details>

```text
rotation (azimuth, dip, rake): [184.7  23.3 129.5]
ranges (major, semi, minor):   [272. 238.  34.]
ratios:                        [0.87 0.13]
     major axis: azimuth 184.7, dip 23.3
semi-major axis: azimuth 069.1, dip 45.1
     minor axis: azimuth 292.8, dip 35.8
plane of the major and semi-major axes: dips 54 towards 113
mean lens plane:                        dips 55 towards 112
minor axis to lens pole: 0.9 degrees
```

The major axis plunges 23° toward 185°, close to strike; the semi-major axis runs obliquely down dip; the minor axis
lies within a degree of the pole of the lenses. The plane of best continuity is the lens plane. Within it the two
ranges differ by only 13 %, so the direction of the major axis in that plane is the least certain part of the fit.

## Slices through the principal planes

`bt.plot.variogram_volume` slices the cube through two principal axes and draws the range ellipse. In the plane of the
lenses γ climbs slowly. Across them it climbs fast, then drops where the lag reaches the next lens.

<details><summary>Python</summary>

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, layout="constrained")
vmax = np.nanpercentile(volume.gammas, 98)
for ax, name in zip(axes, ("major-semi", "major-minor", "semi-minor")):
    bt.plot.variogram_volume(volume, plane=name, ax=ax, vmin=0, vmax=vmax)
    ax.set_title((name.replace("-", " and ") + " axes").capitalize())
    ax.set_xlabel(ax.get_xlabel() + " (m)")
axes[0].set_ylabel(axes[0].get_ylabel() + " (m)")
fig.colorbar(axes[0].collections[0], ax=axes, label="γ, log Zn", shrink=0.85)
save(fig, "slices")
```

</details>

![slices](slices.png)

## From the volume to a model

With the rotation and ratios held at the volume's values, experimental variograms along the three axes leave
`Variogram.fit_directional` only the nugget, sill and range to find. Across the lenses γ reaches the sill near 25 m
and then falls back, as pairs 130 m apart land in the next lens: a hole effect the single structure ignores.

<details><summary>Python</summary>

```python
experimentals = [
    bt.experimental_variogram(composites, log_zn, 15.0, 225.0, azimuth=az, dip=dip, tolerance=15.0)
    for az, dip in volume.axes
]
model = bt.Variogram.fit_directional(
    experimentals, volume.axes, "spherical", rotation=volume.rotation, ratios=volume.ratios
)
structure = model.structures[0]
print(f"nugget {model.nugget:.2f}, sill {structure.sill:.2f}, major range {structure.range:.0f} m")

fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
for exp, direction, color, name in zip(
    experimentals, volume.axes, (ACCENT, GRAY, HIGHLIGHT), ("Major", "Semi-major", "Minor")
):
    bt.plot.variogram(exp, variogram=model, direction=direction, ax=ax, color=color, label=name)
ax.legend(loc="lower right")
ax.set(xlabel="Lag distance (m)", ylabel="γ(h), log Zn", title="Along the principal axes")
save(fig, "axes")
```

</details>

```text
nugget 0.36, sill 2.18, major range 204 m
```

![axes](axes.png)

Full script: [`example_07_04.py`](example_07_04.py)
