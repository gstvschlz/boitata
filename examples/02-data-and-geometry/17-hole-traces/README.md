# Hole traces with deviation flags

`bt.plot.holes` generalizes the single hand-rolled trace of [checking drill holes](../../02-data-and-geometry/01-check-drillholes/README.md) (one hole, `ax.plot(path["x"], path["z"])`) to any
set of holes at once: grouped by hole, one line each, labeled, in plan or projected on a section, with extra points
marked on top.

<details><summary>Python</summary>

```python
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import save
```

</details>

## The flagged stations

`check_drillholes` ([checking drill holes](../../02-data-and-geometry/01-check-drillholes/README.md)) flags two survey stations of `DD0197`, where the azimuth flips and back. Their
positions come from `Drillholes.at`, at the flagged stations' measured depths; `bt.plot.holes` itself knows
nothing about deviation, only about points to mark.

<details><summary>Python</summary>

```python
data = bt.datasets.stacked_sulphide_lenses(raw=True)
collar, survey = data["collars"], data["surveys"]
flags, _, _ = bt.check_drillholes(
    collar, survey, {"assays": data["assays"], "lithology": data["lithology"]}, max_depth="LENGTH"
)
deviated = np.asarray(flags["survey"]["deviation"], dtype=bool)
flagged_holes = np.asarray(survey["HOLE_ID"], dtype=object)[deviated]
flagged_depths = np.asarray(survey["DEPTH"], dtype=float)[deviated]
print(f"{deviated.sum()} flagged stations, in {sorted(set(flagged_holes))}")

drillholes = bt.Drillholes(collar, survey)
marked = drillholes.at(list(flagged_holes), flagged_depths)
```

</details>

```text
2 flagged stations, in ['DD0197']
```

## Plan and section

The holes within 40 m of `DD0197`'s collar, in plan view and on a section across strike (113°, as in [contact surfaces](../../11-geological-modeling/02-contact-surfaces/README.md),
[domain cleanup](../../02-data-and-geometry/14-domain-cleanup/README.md) and [domain change tables](../../07-categories-and-domains/02-domain-change-tables/README.md)) through it: one call each, instead of one `ax.plot` per hole.

<details><summary>Python</summary>

```python
at_dd0197 = np.asarray(collar["HOLE_ID"]) == "DD0197"
x0, y0, z0 = (float(np.asarray(collar[c])[at_dd0197][0]) for c in ("X", "Y", "Z"))
near = np.hypot(np.asarray(collar["X"]) - x0, np.asarray(collar["Y"]) - y0) < 40.0
nearby = set(np.asarray(collar["HOLE_ID"])[near])

paths = drillholes.paths()
local = paths.filter(np.isin(np.asarray(paths["HOLE_ID"]), list(nearby)))

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
bt.plot.holes(local, mark=marked, ax=axes[0])
axes[0].set_title("Plan")
bt.plot.holes(local, plane=((x0, y0, z0), 113.0, 90.0), mark=marked, ax=axes[1])
axes[1].set_title("Section across strike")
save(fig, "traces")
```

</details>

![traces](traces.png)

Full script: [`example_02_17.py`](example_02_17.py)
