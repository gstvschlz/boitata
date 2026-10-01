"""
# along-hole transition matrix and MDS

`transition_matrix` counts which rock type follows which down a hole. for each composite it tallies the class of the
composites one lag deeper in the same hole, into a `from`/`to` table shaped like `domain_change`'s: rows the shallower
rock, columns the deeper one, so `bt.plot.domain_change` draws it unmodified. `bt.plot.transition_mds` turns the row
frequencies into a dissimilarity and lays the rocks out in 2D by classical MDS, so rocks that often touch sit close
together.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import boitata as bt
import numpy as np
from common import save

# %% [markdown]
# ## composites and their rock type
#
# 5 m composites of the stacked sulphide lenses, majority lithology only (as in [domain cleanup](../../02-data-and-geometry/14-domain-cleanup/README.md)); the scheme groups
# the massive, semi-massive and stringer sulphides into one class.

# %%
data = bt.datasets.stacked_sulphide_lenses()
intervals = bt.merge_intervals(data["assays"], data["lithology"])
drillholes = bt.Drillholes(data["collars"], data["surveys"], intervals)
composites = drillholes.composite(5.0, [], categories=["LITH"])
scheme = bt.Categories(
    ["OB", "HWS", "VCL", "SUL", "FWV"],
    mapping={"MS": "SUL", "SMS": "SUL", "STR": "SUL"},
    other="DYK",
    colors=["#d9d9d9", "#a9bfd3", "#8c8c8c", "#c05a28", "#1f4e79", "#e0c080"],
)
depth = (composites["from"] + composites["to"]) / 2
table = bt.transition_matrix(depth, composites["LITH"], composites["HOLE_ID"], lag=5.0, scheme=scheme)
print(f"{len(composites):,} composites, {int(np.sum(table['count'])):,} pairs 5 m apart")

# %% [markdown]
# ## the matrix, reused from `domain_change`
#
# `frequency` holds each rock's row as shares, so the matrix plot needs no `relative=`. the diagonal (staying in the
# same rock) is outlined and left blank.

# %%
fig, ax = bt.plot.domain_change(table, value="frequency", fmt="{:.0%}")
ax.set(xlabel="Rock 5 m deeper", ylabel="Rock at a composite")
save(fig, "matrix")

# %% [markdown]
# ## MDS: who sits next to whom
#
# classical MDS turns `1 - (freq + freq.T) / 2` into a 2D layout. the volcaniclastic sits next to the sulphides it
# overlies, overburden and hanging wall cluster on their own, and the footwall and dyke sit apart, since either
# rarely borders the others.

# %%
fig, ax = bt.plot.transition_mds(table)
save(fig, "mds")
