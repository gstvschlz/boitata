"""
# 3. Compositing

The stacked sulphide lenses: 16 995 assays, mostly 1 m or 2 m, taken only in and around the mineralized zones, and 1726
lithology intervals. Compositing brings the assays to one support without averaging across a contact or reading
unsampled core as zero, and without creating or losing metal.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, LIGHT, save

# %% [markdown]
# Assays and lithology come in separate interval tables. `merge_intervals` splits both at every boundary so each piece
# carries its grades and its lithology; pieces outside the assayed zones have no grades.

# %%
data = cs.datasets.stacked_sulphide_lenses()
collar, survey, assay, lithology = data["collars"], data["surveys"], data["assays"], data["lithology"]
GRADES = ["ZN_PCT", "PB_PCT", "CU_PCT", "AG_GPT", "AU_GPT"]
intervals = cs.merge_intervals(assay, lithology)
dh = cs.Drillholes(collar, survey, intervals)
print(f"{assay.num_rows} assays + {lithology.num_rows} lithology intervals -> {intervals.num_rows} merged")
print(dh)

# %% [markdown]
# Compositing to 2 m by `LITH` cuts intervals at every 2 m mark and never averages across a contact. A grade is the
# mean over the length that carries a value, so unsampled core is not read as zero; that length is returned per grade
# as `<grade>_length`, next to `length`, which also counts unsampled ground. Composites without assays are dropped.
# Here the assayed zones start and end at lithology contacts, so inside a lithology Zn is sampled everywhere; Au is
# not assayed in the RC holes.

# %%
composites = dh.composite(2.0, GRADES, domain="LITH")
partial = composites["ZN_PCT_length"] < composites["length"] - 1e-9
no_au = np.isnan(composites["AU_GPT"])
print(
    f"{len(composites)} composites; {partial.sum()} partly unsampled for Zn; {no_au.sum()} without Au (RC holes)"
)

# %% [markdown]
# Compositing regularizes support: the assays are mostly 1 m or 2 m, the composites 2 m, with shorter tails where a run of
# one lithology ends.

# %%
raw_len = assay["TO"] - assay["FROM"]
fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
bins = np.arange(0, 3.0, 0.125)
ax.hist(raw_len, bins, color=LIGHT, edgecolor=GRAY, lw=0.5, label="assays")
ax.hist(composites["length"], bins, histtype="step", color=ACCENT, lw=1.6, label="composites")
ax.set(title="Interval lengths", xlabel="Length (m)", ylabel="Count")
ax.legend(loc="upper left")
save(fig, "compositing")

# %% [markdown]
# Other supports. `length=None` gives one composite per run of a lithology. `intervals=` composites to given
# intervals instead, here 10 m benches: the depths where each desurveyed path crosses a bench elevation. `residual=`
# decides the fate of a run's tail shorter than `min_fraction` of the length: kept, merged into the previous composite
# or dropped. Without `domain`, composites cross contacts and `categories=` gives the lithology covering most of each.

# %%
BENCH = 10.0
paths = dh.paths()
hole, depth, z = np.array(paths["HOLE_ID"]), paths["depth"], paths["z"]
cuts = {h: [0.0, depth[hole == h].max()] for h in np.unique(hole)}
for i in np.flatnonzero(hole[1:] == hole[:-1]):
    lo, hi = sorted((z[i], z[i + 1]))
    for level in np.arange(np.ceil(lo / BENCH) * BENCH, hi, BENCH):
        cuts[hole[i]].append(depth[i] + (level - z[i]) / (z[i + 1] - z[i]) * (depth[i + 1] - depth[i]))
benches = {"HOLE_ID": [], "FROM": [], "TO": []}
for h, c in cuts.items():
    c = np.unique(c)
    benches["HOLE_ID"] += [h] * (len(c) - 1)
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
# counts unsampled ground at the composite grade and inflates metal, here where majority-`LITH` composites cross a
# contact into unassayed rock.


# %%
def metal(grade, length):
    return np.nansum(grade * length)


assayed = {g: metal(assay[g], raw_len) for g in GRADES}
print(f"{'':>18}  {'Zn metal':>10}  {'error':>8}  {'by length':>9}")
print(f"{'assays':>18}  {assayed['ZN_PCT']:10.1f}")
for name, c in modes.items():
    zn_metal = metal(c["ZN_PCT"], c["ZN_PCT_length"])
    error, naive = zn_metal / assayed["ZN_PCT"] - 1, metal(c["ZN_PCT"], c["length"]) / assayed["ZN_PCT"] - 1
    print(f"{name:>18}  {zn_metal:10.1f}  {error:+8.1e}  {naive:+9.1%}")
    if name != "2 m, drop < 1 m":
        for g in GRADES:
            assert abs(metal(c[g], c[f"{g}_length"]) / assayed[g] - 1) < 1e-9, (name, g)
print("metal balanced to 1e-9 for", ", ".join(GRADES))
