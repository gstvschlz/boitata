# 6. Drillholes

5 277 holes with collar, survey (dip positive down, azimuth clockwise from north), assay and geology tables.

<details><summary>Python</summary>

```python
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GREY, INK, LIGHT, save
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
```

</details>

The tables are checked before anything else. `check_drillholes` flags every record of the collar, survey and interval
tables (duplicate ids, missing or sentinel values, from >= to, gaps, overlaps, angles out of range, depths past the
hole length, holes missing from a table, abrupt survey deviation) and summarises each check. `fix_drillholes` then
resolves them with one named rule per check; the log says what each rule changed. Overlapping assays keep the one
that starts first. Gaps are unsampled core and stay.

<details><summary>Python</summary>

```python
tables = cs.datasets.drillhole_tables()
checked = {"assay": tables["assay"], "geology": tables["geology"]}
flags, summary = cs.check_drillholes(tables["collar"], tables["survey"], checked, max_depth="DEPTH")
fixed, log = cs.fix_drillholes(flags, tables, overlaps="keep_first", deviation="drop", past_depth="keep")

for table, check, rows, holes in zip(summary["table"], summary["check"], summary["rows"], summary["holes"]):
    if rows:
        print(f"{table:>8} {check:<11} {rows:6.0f} rows in {holes:4.0f} holes")
for table, check, action, rows in zip(log["table"], log["check"], log["action"], log["rows"]):
    if rows:
        print(f"{action:>10} {rows:3.0f} {table} rows flagged {check}")
```

</details>

```text
  collar no_survey        2 rows in    2 holes
  collar no_assay       266 rows in  266 holes
  collar no_geology    3415 rows in 3415 holes
  survey deviation       12 rows in    8 holes
   assay gap           4903 rows in  802 holes
   assay overlap         22 rows in    4 holes
      drop  12 survey rows flagged deviation
keep_first  22 assay rows flagged overlap
```

A survey station is flagged when its direction turns more than 20° from the station above, so one wrong station is
flagged together with the station below it, as the dip of DH1432 flipping sign at 116 m. Here every flagged station
is dropped; each is shown against the station above it.

<details><summary>Python</summary>

```python
survey = tables["survey"]
hole_id, depth, dip, azimuth = np.array(survey["HOLEID"]), survey["DEPTH"], survey["DIP"], survey["AZIMUTH"]
order = np.lexsort((depth, hole_id))
above = dict(zip(order[1:], order[:-1]))
for i in np.flatnonzero(flags["survey"]["deviation"]):
    j = above[i]
    print(
        f"{hole_id[i]} {depth[j]:5.1f} -> {depth[i]:5.1f} m: dip {dip[j]:5.1f} -> {dip[i]:5.1f},"
        f" azimuth {azimuth[j]:5.1f} -> {azimuth[i]:5.1f}"
    )
```

</details>

```text
DH0480  28.0 ->  50.0 m: dip  14.6 ->  14.9, azimuth 355.1 ->  19.8
DH0646   0.0 ->   6.0 m: dip  36.0 -> -35.5, azimuth 337.6 -> 345.1
DH0963   0.0 ->  19.8 m: dip -20.0 -> -15.0, azimuth 157.6 -> 299.2
DH1432 104.0 -> 116.0 m: dip  41.8 -> -38.9, azimuth 304.0 -> 303.6
DH1432 116.0 -> 128.0 m: dip -38.9 ->  41.8, azimuth 303.6 -> 303.5
DH2797  90.0 -> 130.0 m: dip  11.0 ->  10.0, azimuth 337.6 -> 300.2
DH3167  30.0 ->  56.0 m: dip  49.0 ->  18.7, azimuth 293.6 -> 289.6
DH3167  56.0 ->  60.0 m: dip  18.7 ->  48.0, azimuth 289.6 -> 293.6
DH3382   6.0 ->  28.0 m: dip  46.0 ->  -1.2, azimuth 157.1 -> 270.6
DH3382  28.0 ->  30.0 m: dip  -1.2 ->  46.0, azimuth 270.6 -> 157.6
DH3382  30.0 ->  58.0 m: dip  46.0 ->  -1.0, azimuth 157.6 -> 270.6
DH3807 437.0 -> 485.0 m: dip  48.5 ->  27.5, azimuth 282.6 -> 288.1
```

Assays and lithology come in separate interval tables. `merge_intervals` splits both at every boundary so each piece
carries its grades and its lithology.

<details><summary>Python</summary>

```python
tables = fixed
assay, geology = tables["assay"], tables["geology"]
GRADES = ["ZN", "PB", "CU", "AG", "AU"]
intervals = cs.merge_intervals(assay, geology)
print(f"{assay.num_rows} assays + {geology.num_rows} geology intervals -> {intervals.num_rows} merged")
```

</details>

```text
81634 assays + 64431 geology intervals -> 137014 merged
```

`Drillholes` desurveys each hole from its collar and survey. Minimum curvature bends along a circular arc between
stations; tangential holds each station's direction down to the next one; balanced tangential gives half of each
segment to each end. On curved holes the methods drift apart with depth.

<details><summary>Python</summary>

```python
methods = ["minimum_curvature", "tangential", "balanced_tangential"]
holes = {m: cs.Drillholes(tables["collar"], tables["survey"], intervals, method=m) for m in methods}
dh = holes["minimum_curvature"]
reference = dh.samples().coords
print(dh)
for m in methods[1:]:
    shift = np.linalg.norm(holes[m].samples().coords - reference, axis=1)
    print(f"{m:>19} vs minimum curvature: median {np.median(shift):.2f} m, max {shift.max():.1f} m")
```

</details>

```text
Drillholes(5277 holes, 137014 intervals)
         tangential vs minimum curvature: median 0.31 m, max 43.0 m
balanced_tangential vs minimum curvature: median 0.00 m, max 3.8 m
```

Compositing to 2 m by `LITH` cuts intervals at every 2 m mark and never averages across a contact. A grade is the
mean over the length that carries a value, so unsampled core is not read as zero; that length is returned per grade
as `<grade>_length`, next to `length`, which also counts unsampled ground. Composites without assays are dropped.

<details><summary>Python</summary>

```python
composites = dh.composite(2.0, GRADES, domain="LITH")
paths = dh.paths()
print(
    f"{len(composites)} composites; {np.mean(composites['ZN_length'] < composites['length'] - 1e-9):.1%} partly unsampled for Zn"
)
```

</details>

```text
62506 composites; 16.0% partly unsampled for Zn
```

Traces in plan and a 50 m thick section with the Zn composites:

<details><summary>Python</summary>

```python
hole = np.array(paths["hole"])
xyz = np.c_[paths["x"], paths["y"], paths["z"]]
breaks = np.flatnonzero(hole[1:] != hole[:-1]) + 1
traces = np.split(xyz, breaks)

y0 = np.median(composites.coords[:, 1])
half = 25.0
in_slab = [t for t in traces if np.any(np.abs(t[:, 1] - y0) < half)]
inside = []
for t in in_slab:
    inslab = np.abs(t[:, 1] - y0) < half
    for run in np.split(np.arange(len(t)), np.flatnonzero(np.diff(inslab.astype(int))) + 1):
        if inslab[run[0]] and len(run) > 1:
            inside.append(t[run][:, [0, 2]])
comps = composites.coords
zn = composites["ZN"]
near = (np.abs(comps[:, 1] - y0) < half) & ~np.isnan(zn) & (zn > 0)

fig, (a, b) = plt.subplots(
    1, 2, figsize=(12, 5.2), layout="constrained", gridspec_kw={"width_ratios": [1, 1.6]}
)
a.add_collection(LineCollection([t[:, :2] for t in traces], colors=LIGHT, linewidths=0.4))
a.add_collection(LineCollection([t[:, :2] for t in in_slab], colors=ACCENT, linewidths=0.6))
a.axhspan(y0 - half, y0 + half, color=ACCENT, alpha=0.08, lw=0)
a.autoscale()
a.set_aspect("equal")
a.set(title=f"Plan: {len(dh)} desurveyed holes", xlabel="Easting (m)", ylabel="Northing (m)")
b.add_collection(LineCollection(inside, colors=LIGHT, linewidths=0.8))
points = b.scatter(comps[near, 0], comps[near, 2], c=zn[near], s=5, norm=LogNorm(0.05, 30), cmap="cividis")
b.autoscale()
b.set_aspect("equal")
b.set(
    title=f"Section: northing {y0:.0f} ± {half:.0f} m, 2 m composites",
    xlabel="Easting (m)",
    ylabel="Elevation (m)",
)
fig.colorbar(points, ax=b, shrink=0.7, label="Zn (%)")
save(fig, "holes")
```

</details>

![holes](holes.png)

Compositing regularizes support: most assays are 1 m, some much longer.

<details><summary>Python</summary>

```python
raw_len = assay["TO"] - assay["FROM"]
comp_len = composites["to"] - composites["from"]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
bins = np.arange(0, 6.25, 0.25)
a.hist(np.clip(raw_len, 0, 6), bins, color=LIGHT, edgecolor=GREY, lw=0.5, label="assays")
a.hist(comp_len, bins, histtype="step", color=ACCENT, lw=1.6, label="composites")
a.set(title="Interval lengths", xlabel="Length (m)", ylabel="Count")
a.legend()
logbins = np.logspace(-2, 1.7, 40)
raw_zn = assay["ZN"]
a_zn = raw_zn[raw_zn > 0]
c_zn = zn[zn > 0]
b.hist(
    a_zn,
    logbins,
    weights=np.full(a_zn.size, 1 / a_zn.size),
    color=LIGHT,
    edgecolor=GREY,
    lw=0.5,
    label=f"assays, CV {a_zn.std() / a_zn.mean():.2f}",
)
b.hist(
    c_zn,
    logbins,
    weights=np.full(c_zn.size, 1 / c_zn.size),
    histtype="step",
    color=ACCENT,
    lw=1.6,
    label=f"composites, CV {c_zn.std() / c_zn.mean():.2f}",
)
b.set_xscale("log")
b.set(title="Compositing narrows the Zn distribution", xlabel="Zn (%)", ylabel="Proportion")
b.legend(loc="upper left")
b.tick_params(axis="y", colors=INK)
save(fig, "compositing")
```

</details>

![compositing](compositing.png)

Zn by lithology:

<details><summary>Python</summary>

```python
lith = np.array(composites.attributes["LITH"])
names, counts = np.unique(lith[lith != ""], return_counts=True)
top = np.isin(lith, names[np.argsort(counts)[::-1][:8]])
fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
cs.plot.boxplot(zn[top], lith[top], sort=True, log=True, ax=ax)
ax.set(title="Zn of 2 m composites by lithology (P10, P25, median, P75, P90 and mean)", ylabel="Zn (%)")
save(fig, "domains")
```

</details>

![domains](domains.png)

Other supports. `length=None` gives one composite per run of a lithology. `intervals=` composites to given
intervals instead, here 10 m benches: the depths where each desurveyed path crosses a bench elevation. `residual=`
decides the fate of a run's tail shorter than `min_fraction` of the length: kept, merged into the previous composite
or dropped. Without `domain`, composites cross contacts and `categories=` gives the lithology covering most of each.

<details><summary>Python</summary>

```python
BENCH = 10.0
depth, z = paths["depth"], paths["z"]
cuts = {hole[i]: [0.0, depth[i]] for i in np.r_[breaks - 1, len(hole) - 1]}
for i in np.flatnonzero(hole[1:] == hole[:-1]):
    lo, hi = sorted((z[i], z[i + 1]))
    for level in np.arange(np.ceil(lo / BENCH) * BENCH, hi, BENCH):
        cuts[hole[i]].append(depth[i] + (level - z[i]) / (z[i + 1] - z[i]) * (depth[i + 1] - depth[i]))
benches = {"HOLEID": [], "FROM": [], "TO": []}
for h, c in cuts.items():
    c = np.unique(c)
    benches["HOLEID"] += [h] * (len(c) - 1)
    benches["FROM"] += list(c[:-1])
    benches["TO"] += list(c[1:])

modes = {
    "2 m": composites,
    "runs": dh.composite(None, GRADES, domain="LITH"),
    "10 m benches": dh.composite(None, GRADES, domain="LITH", intervals=benches),
    "2 m, merge < 1 m": dh.composite(2.0, GRADES, domain="LITH", residual="merge"),
    "2 m, drop < 1 m": dh.composite(2.0, GRADES, domain="LITH", residual="drop"),
    "2 m, majority LITH": dh.composite(2.0, GRADES, categories=["LITH"]),
}
for name, c in modes.items():
    print(f"{name:>18}: {len(c):6} composites, median length {np.median(c['length']):.1f} m")
```

</details>

```text
               2 m:  62506 composites, median length 2.0 m
              runs:  15761 composites, median length 5.5 m
      10 m benches:  24439 composites, median length 5.1 m
  2 m, merge < 1 m:  60149 composites, median length 2.0 m
   2 m, drop < 1 m:  57780 composites, median length 2.0 m
2 m, majority LITH:  56842 composites, median length 2.0 m
```

Metal balance: Σ grade × `<grade>_length` over the composites reproduces Σ grade × interval length over the assays,
for every grade and every mode except `drop`, which leaves its short tails out. Weighting by `length` instead
counts unsampled ground at the composite grade and inflates metal.

<details><summary>Python</summary>

```python
def metal(grade, length):
    return np.nansum(grade * length)


assayed = {g: metal(assay[g], assay["TO"] - assay["FROM"]) for g in GRADES}
print(f"{'':>18}  {'Zn metal':>10}  {'error':>8}  {'by length':>9}")
print(f"{'assays':>18}  {assayed['ZN']:10.1f}")
for name, c in modes.items():
    zn_metal = metal(c["ZN"], c["ZN_length"])
    error, naive = zn_metal / assayed["ZN"] - 1, metal(c["ZN"], c["length"]) / assayed["ZN"] - 1
    print(f"{name:>18}  {zn_metal:10.1f}  {error:+8.1e}  {naive:+9.1%}")
    if name != "2 m, drop < 1 m":
        for g in GRADES:
            assert abs(metal(c[g], c[f"{g}_length"]) / assayed[g] - 1) < 1e-9, (name, g)
print("metal balanced to 1e-9 for", ", ".join(GRADES))
```

</details>

```text
                      Zn metal     error  by length
            assays    325074.7
               2 m    325074.7  +0.0e+00      +8.0%
              runs    325074.7  +0.0e+00     +77.2%
      10 m benches    325074.7  +0.0e+00     +40.6%
  2 m, merge < 1 m    325074.7  +0.0e+00      +8.2%
   2 m, drop < 1 m    316029.6  -2.8e-02      +5.2%
2 m, majority LITH    325074.7  +0.0e+00      +7.5%
metal balanced to 1e-9 for ZN, PB, CU, AG, AU
```

Full script: [`example_06.py`](example_06.py)
