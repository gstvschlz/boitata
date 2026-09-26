"""
# 6. Drillholes

5 277 holes with collar, survey (dip positive down, azimuth clockwise from north), assay and geology tables.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GREY, INK, LIGHT, save
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

# %% [markdown]
# Assays and lithology come in separate interval tables. Within a table intervals must not overlap, so the few
# overlapping assays are resolved first, here by keeping the one that starts first. `merge_intervals` then splits both
# tables at every boundary so each piece carries its grades and its lithology.

# %%
tables = cs.datasets.drillhole_tables()
assay, geology = tables["assay"], tables["geology"]
GRADES = ["ZN", "PB", "CU", "AG", "AU"]

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
intervals = cs.merge_intervals(assay, geology)
print(f"{(~keep).sum()} overlapping assays dropped")
print(f"{assay.num_rows} assays + {geology.num_rows} geology intervals -> {intervals.num_rows} merged")


# %% [markdown]
# `Drillholes` desurveys each hole from its collar and survey. Minimum curvature bends along a circular arc between
# stations; tangential holds each station's direction down to the next one; balanced tangential gives half of each
# segment to each end. On curved holes the methods drift apart with depth.

# %%
methods = ["minimum_curvature", "tangential", "balanced_tangential"]
holes = {m: cs.Drillholes(tables["collar"], tables["survey"], intervals, method=m) for m in methods}
dh = holes["minimum_curvature"]
reference = dh.samples().coords
print(dh)
for m in methods[1:]:
    shift = np.linalg.norm(holes[m].samples().coords - reference, axis=1)
    print(f"{m:>19} vs minimum curvature: median {np.median(shift):.2f} m, max {shift.max():.1f} m")


# %% [markdown]
# Compositing to 2 m by `LITH` cuts intervals at every 2 m mark and never averages across a contact. A grade is the
# mean over the length that carries a value, so unsampled core is not read as zero; that length is returned per grade
# as `<grade>_length`, next to `length`, which also counts unsampled ground. Composites without assays are dropped.

# %%
composites = dh.composite(2.0, GRADES, domain="LITH")
paths = dh.paths()
print(
    f"{len(composites)} composites; {np.mean(composites['ZN_length'] < composites['length'] - 1e-9):.1%} partly unsampled for Zn"
)


# %% [markdown]
# Traces in plan and a 50 m thick section with the Zn composites:

# %%
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
points = b.scatter(comps[near, 0], comps[near, 2], c=zn[near], s=5, norm=LogNorm(0.05, 30), cmap="ceres")
b.autoscale()
b.set_aspect("equal")
b.set(
    title=f"Section: northing {y0:.0f} ± {half:.0f} m, 2 m composites",
    xlabel="Easting (m)",
    ylabel="Elevation (m)",
)
fig.colorbar(points, ax=b, shrink=0.7, label="Zn (%)")
save(fig, "holes")


# %% [markdown]
# Compositing regularizes support: most assays are 1 m, some much longer.

# %%
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


# %% [markdown]
# Zn by lithology:

# %%
lith = np.array(composites.attributes["LITH"])
names, counts = np.unique(lith[lith != ""], return_counts=True)
top = np.isin(lith, names[np.argsort(counts)[::-1][:8]])
fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
cs.plot.boxplot(zn[top], lith[top], sort=True, log=True, ax=ax)
ax.set(title="Zn of 2 m composites by lithology (P10, P25, median, P75, P90 and mean)", ylabel="Zn (%)")
save(fig, "domains")


# %% [markdown]
# Other supports. `length=None` gives one composite per run of a lithology. `intervals=` composites to given
# intervals instead, here 10 m benches: the depths where each desurveyed path crosses a bench elevation. `residual=`
# decides the fate of a run's tail shorter than `min_fraction` of the length: kept, merged into the previous composite
# or dropped. Without `domain`, composites cross contacts and `categories=` gives the lithology covering most of each.

# %%
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


# %% [markdown]
# Metal balance: Σ grade × `<grade>_length` over the composites reproduces Σ grade × interval length over the assays,
# for every grade and every mode except `drop`, which leaves its short tails out. Weighting by `length` instead
# counts unsampled ground at the composite grade and inflates metal.


# %%
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
