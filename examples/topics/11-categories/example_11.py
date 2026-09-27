"""
# 11. Categories

A tailings storage facility drilled by 107 sonic holes, logged as cover (CAP), sand (SAND), slimes (SLIME) and the clay
of the original ground (CLAY). A `Categories` scheme fixes the order, names and colors of the codes once, so that
shares, plots and later models all agree on them.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import ceres as cs
import matplotlib.pyplot as plt
from common import save

data = cs.datasets.tailings_reprocessing()
logged = cs.Drillholes(data["collars"], data["surveys"], data["lithology"]).samples()
length = logged["TO"] - logged["FROM"]
print(
    f"{len(logged)} intervals, {length.sum():.0f} m logged, {length.min():.2f} to {length.max():.1f} m long"
)

# %% [markdown]
# `encode` turns the logged names into integer codes in the order of the scheme, NaN for a name it does not know, and
# `shares` gives the proportion of each code, by count or weighted. The intervals differ in length a hundredfold, so
# the shares by length are the ones that describe the facility.

# %%
lithology = cs.Categories(
    ["CAP", "SAND", "SLIME", "CLAY"], colors=["#b8b8b8", "#d9b56c", "#5b7a99", "#7a4f35"]
)
codes = lithology.encode(logged["LITH"])
by_count, by_length = lithology.shares(codes), lithology.shares(codes, weights=length)
for name, a, b in zip(lithology.names, by_count, by_length, strict=True):
    print(f"{name:<6}{a:6.1%} of intervals {b:6.1%} of length")

# %% [markdown]
# SLIME is logged as often as SAND but in thinner intervals, and the few CLAY intervals are long: by length, SLIME falls
# from 47.7 % to 38.6 % and CLAY rises from 2.9 % to 11.0 %. `plot.proportions` draws the weighted shares as bars,
# the unweighted ones as ticks; `plot.category_swath` stacks the shares per slice, as `swath` does for a grade.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained", width_ratios=[1, 1.4, 1.4])
cs.plot.proportions(codes, weights=length, scheme=lithology, ax=axes[0])
axes[0].set_title("Share of length")
xyz = logged.coords
cs.plot.category_swath(xyz, codes, 50.0, axis="y", weights=length, scheme=lithology, ax=axes[1])
axes[1].set(title="Along northing, 50 m slices", xlabel="Northing (m)")
axes[1].get_legend().remove()
cs.plot.category_swath(xyz, codes, 2.0, axis="z", weights=length, scheme=lithology, ax=axes[2])
axes[2].set(title="In elevation, 2 m slices", xlabel="Elevation (m)")
save(fig, "categories")

# %% [markdown]
# The facility sorts its tailings: sand dominates the north and slimes the south, where the fines settled, while
# clay sits at the base and cover on top. A reprocessing plan that treats sand and slimes apart would model the two as
# separate domains.
#
# A scheme can also be read off the data. `Categories.from_values` keeps the names above `min_share` of the weight,
# sorted, and lumps the rest into `other`, last: here CAP, under 5 % of the length.

# %%
found = cs.Categories.from_values(logged["LITH"], weights=length, min_share=0.05)
print(found.names)
