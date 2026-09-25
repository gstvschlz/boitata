import sys
from pathlib import Path

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from common import ACCENT, GREY, INK, LIGHT, fetch, save

dh = cs.Drillholes(
    cs.read_csv(fetch("drillholes/collar.csv")),
    cs.read_csv(fetch("drillholes/survey.csv")),
    cs.read_csv(fetch("drillholes/assay.csv")),
)
samples = dh.samples()
composites = dh.composite(2.0, ["ZN", "PB", "CU", "AG", "AU"])
paths = dh.paths()
print(dh)
print(f"{len(samples)} samples -> {len(composites)} composites of 2 m")

hole = np.array(paths["hole"])
xyz = np.c_[paths["x"], paths["y"], paths["z"]]
breaks = np.flatnonzero(hole[1:] != hole[:-1]) + 1
traces = np.split(xyz, breaks)

y0 = np.median(composites.coords[:, 1])
half = 25.0
in_slab = [t for t in traces if np.any(np.abs(t[:, 1] - y0) < half)]
inside = []
for t in in_slab:
    keep = np.abs(t[:, 1] - y0) < half
    for run in np.split(np.arange(len(t)), np.flatnonzero(np.diff(keep.astype(int))) + 1):
        if keep[run[0]] and len(run) > 1:
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
save(fig, HERE, "holes")

raw_len = samples["TO"] - samples["FROM"]
comp_len = composites["to"] - composites["from"]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
bins = np.arange(0, 6.25, 0.25)
a.hist(np.clip(raw_len, 0, 6), bins, color=LIGHT, edgecolor=GREY, lw=0.5, label="samples")
a.hist(comp_len, bins, histtype="step", color=ACCENT, lw=1.6, label="composites")
a.set(title="Interval lengths", xlabel="Length (m)", ylabel="Count")
a.legend()
logbins = np.logspace(-2, 1.7, 40)
raw_zn = samples["ZN"]
a_zn = raw_zn[raw_zn > 0]
c_zn = zn[zn > 0]
b.hist(
    a_zn,
    logbins,
    weights=np.full(a_zn.size, 1 / a_zn.size),
    color=LIGHT,
    edgecolor=GREY,
    lw=0.5,
    label=f"samples, CV {a_zn.std() / a_zn.mean():.2f}",
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
save(fig, HERE, "compositing")
