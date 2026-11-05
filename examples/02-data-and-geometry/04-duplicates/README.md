# Duplicates

Collars entered twice, holes collared next to each other and samples assayed twice put several records at one place,
and kriging cannot weight two samples at the same location. `duplicates` groups the samples closer than a tolerance
and merges each group. The raw tables of the stacked sulfide lenses carry such errors on purpose.

<details><summary>Python</summary>

```python
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, map_axes, save

raw = bt.datasets.stacked_sulphide_lenses(raw=True)
collars = raw["collars"]
names, counts = np.unique(collars["HOLE_ID"], return_counts=True)
print(f"{len(collars)} collars, {len(names)} hole ids, repeated: {', '.join(names[counts > 1])}")
```

</details>

```text
290 collars, 288 hole ids, repeated: DD0032, DD0224
```

## Collars

A repeated id can sit at two locations. `duplicates` on the collar coordinates groups the collars within 5 m of
one another, transitively, and gives each group's first collar and its spread.

<details><summary>Python</summary>

```python
xyz = np.column_stack([collars["X"], collars["Y"], collars["Z"]])
close, group_collars = bt.duplicates(xyz, tolerance=5.0)
for k, spread in zip(close["group"], close["spread"], strict=True):
    print(f"{' and '.join(collars['HOLE_ID'][group_collars == k])}: {spread:.1f} m apart")
for name in names[counts > 1]:
    first, second = xyz[collars["HOLE_ID"] == name]
    print(f"{name} twice: {np.linalg.norm(first - second):.0f} m apart")
```

</details>

```text
DD0060 and DD0121: 3.0 m apart
DD0065 and RC0003 and RC0036: 4.8 m apart
DD0069 and DD0070: 3.1 m apart
DD0224 and DD0224: 0.0 m apart
RC0001 and RC0038: 4.0 m apart
RC0004 and RC0045: 1.3 m apart
DD0032 twice: 225 m apart
DD0224 twice: 0 m apart
```

DD0224 is one row entered twice: drop one. DD0032 names two collars 225 m apart, an id clash that `check_drillholes`
flags ([checking drill holes](../../02-data-and-geometry/01-check-drillholes/README.md)). DD0065 and RC0003 are two holes collared 3.9 m apart at
different dips. Nearby holes of two drilling types are [paired data](../../03-exploratory-analysis/01-paired-data/README.md).

## Samples

The grades of the raw assays are text, with `NS`, `<0.01` and `-999` among the numbers. Here anything other than a
plain number reads as missing. The default rules of [checking drill holes](../../02-data-and-geometry/01-check-drillholes/README.md) fix the other planted errors, except
overlaps, which stay so that the samples entered twice survive.

<details><summary>Python</summary>

```python
def number(text):
    return np.array([float(v) if str(v).replace(".", "", 1).isdigit() else np.nan for v in text])


assays = raw["assays"]
assays = bt.Table({**{c: assays[c] for c in ("HOLE_ID", "FROM", "TO")}, "ZN_PCT": number(assays["ZN_PCT"])})
tables = {"collar": collars, "survey": raw["surveys"], "assays": assays}
flags, _, _ = bt.check_drillholes(collars, raw["surveys"], {"assays": assays}, max_depth="LENGTH")
fixed, _ = bt.fix_drillholes(flags, tables, overlaps="keep")
samples = bt.Drillholes(fixed["collar"], fixed["survey"], fixed["assays"]).samples()
print(f"{len(samples)} samples at their midpoints")
```

</details>

```text
19055 samples at their midpoints
```

With a tolerance of 0, only samples at the same location are grouped: three intervals of DD0216 were entered twice,
with grades within 10 % of each other.

<details><summary>Python</summary>

```python
report, group = bt.duplicates(samples, tolerance=0.0)
zn, hole = samples["ZN_PCT"], samples["HOLE_ID"]
depth = (samples["FROM"] + samples["TO"]) / 2
for k in report["group"]:
    rows = np.flatnonzero(group == k)
    values = " and ".join(f"{v:.3f}" for v in zn[rows])
    print(f"{hole[rows[0]]} at {depth[rows[0]]:.1f} m: Zn {values} %")

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", width_ratios=[1.3, 1])
a.scatter(xyz[:, 0], xyz[:, 1], s=5, color=GRAY, label="Collars")
a.scatter(close["x"], close["y"], s=90, facecolors="none", edgecolors=HIGHLIGHT, label="Collars within 5 m")
for k, x, y in zip(close["group"], close["x"], close["y"], strict=True):
    a.annotate(
        " / ".join(dict.fromkeys(collars["HOLE_ID"][group_collars == k])),
        (x, y),
        (8, -3),
        textcoords="offset points",
        fontsize=8,
    )
map_axes(a, "Collars")
a.legend(loc="upper left")
rows = np.flatnonzero((hole == "DD0216") & (depth > 710) & (depth < 745))
b.plot(depth[rows], zn[rows], "o", ms=3, color=ACCENT)
twice = rows[group[rows] >= 0]
b.plot(depth[twice], zn[twice], "o", ms=8, mfc="none", color=HIGHLIGHT, label="Entered twice")
b.set(title="DD0216, 710 to 745 m", xlabel="Depth (m)", ylabel="Zn (%)")
b.legend(loc="upper right")
save(fig, "duplicates")
```

</details>

```text
DD0216 at 725.5 m: Zn 0.031 and 0.032 %
DD0216 at 727.5 m: Zn 0.035 and 0.033 %
DD0216 at 729.5 m: Zn 0.050 and 0.050 %
```

![duplicates](duplicates.png)

A tolerance of a meter or two would also group neighbors along a hole, such as short intervals on both sides of a
contact. `merge="mean"` replaces each group by one sample at its first location with the mean grade, and the `n`
column counts the samples behind each row. `"first"` keeps the first entry instead, and `"max"` the largest value.

<details><summary>Python</summary>

```python
merged = bt.duplicates(samples, merge="mean")
twins = merged["n"] > 1
means = ", ".join(f"{v:.3f}" for v in merged["ZN_PCT"][twins])
print(f"{len(merged)} samples, {int(twins.sum())} of them merged pairs, Zn {means} %")
```

</details>

```text
19052 samples, 3 of them merged pairs, Zn 0.032, 0.034, 0.050 %
```

Full script: [`example_02_04.py`](example_02_04.py)
