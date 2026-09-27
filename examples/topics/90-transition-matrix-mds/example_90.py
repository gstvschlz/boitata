"""
# 90. Along-hole transition matrix and MDS

Which rock type follows which, going down a hole? `transition_matrix` tallies, for every composite, the class of
composites one lag deeper in the same hole, into a `from`/`to` table shaped exactly like `domain_change`'s: rows
the shallower rock, columns the deeper one, so `cs.plot.domain_change` draws it unmodified. `cs.plot.transition_mds`
turns the row frequencies into a dissimilarity and lays the rocks out in 2D by classical MDS, so ones that most
often lie next to each other sit close together.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import ceres as cs
import numpy as np
from common import save

# %% [markdown]
# ## Composites and their rock type
#
# 5 m composites of the stacked sulphide lenses, majority lithology only (as topic 80's cleanup); the scheme groups
# the massive, semi-massive and stringer sulphides into one class.

# %%
data = cs.datasets.stacked_sulphide_lenses()
intervals = cs.merge_intervals(data["assays"], data["lithology"])
drillholes = cs.Drillholes(data["collars"], data["surveys"], intervals)
composites = drillholes.composite(5.0, [], categories=["LITH"])
scheme = cs.Categories(
    ["OB", "HWS", "VCL", "SUL", "FWV"],
    mapping={"MS": "SUL", "SMS": "SUL", "STR": "SUL"},
    other="DYK",
    colors=["#d9d9d9", "#a9bfd3", "#8c8c8c", "#c05a28", "#1f4e79", "#e0c080"],
)
depth = (composites["from"] + composites["to"]) / 2
table = cs.transition_matrix(depth, composites["LITH"], composites["HOLE_ID"], lag=5.0, scheme=scheme)
print(f"{len(composites):,} composites, {int(np.sum(table['count'])):,} pairs 5 m apart")

# %% [markdown]
# ## The matrix, reused from `domain_change`
#
# `frequency` is each rock's row already turned into shares, so the matrix plot needs no `relative=`: the diagonal,
# staying in the same rock, is outlined and left blank.

# %%
fig, ax = cs.plot.domain_change(table, value="frequency", fmt="{:.0%}")
ax.set(xlabel="Rock 5 m deeper", ylabel="Rock at a composite")
save(fig, "matrix")

# %% [markdown]
# ## MDS: who sits next to whom
#
# `1 - (freq + freq.T) / 2` turned into a 2D layout by classical MDS: the volcaniclastic sits right next to the
# sulphides it directly overlies, overburden and hanging wall cluster on their own, and the footwall and dyke sit
# apart, matching how rarely either borders the others directly.

# %%
fig, ax = cs.plot.transition_mds(table)
save(fig, "mds")
