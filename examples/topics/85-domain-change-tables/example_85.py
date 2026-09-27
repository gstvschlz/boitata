"""
# 85. Domain change tables

Two categorical models of the same blocks, an old and a new domain model, the most likely category and a
realization, a model before and after cleanup, differ block by block. `domain_change` cross-tabulates them: the
tonnage moving from each class of the first model to each class of the second, and with a grade, the metal and mean
grade of every cell. Rows sum to the first model's classes, columns to the second's, the diagonal is what stays, and
the metal adds up to the model's. `cs.plot.domain_change` draws the table as a matrix.
"""

# %% [hidden]
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))
warnings.filterwarnings("ignore", ".*locations hold several samples")

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import save

# %% [markdown]
# ## Two rock models
#
# The section of topic 80 across the stacked sulphide lenses: the most likely rock from categorical indicator kriging
# of the logged rock types on cells of 10 m, and the same model once units under 5000 m³ are given to the rock around
# them. Zinc and density are kriged on the same cells from the 5 m composites, ignoring the rock types.

# %%
data = cs.datasets.stacked_sulphide_lenses()
intervals = cs.merge_intervals(data["assays"], data["lithology"])
drillholes = cs.Drillholes(data["collars"], data["surveys"], intervals)
composites = drillholes.composite(5.0, ["ZN_PCT", "DENSITY"], categories=["LITH"])
scheme = cs.Categories(
    ["OB", "HWS", "VCL", "SUL", "FWV"],
    mapping={"MS": "SUL", "SMS": "SUL", "STR": "SUL"},
    other="DYK",
    colors=["#d9d9d9", "#a9bfd3", "#8c8c8c", "#c05a28", "#1f4e79", "#e0c080"],
)
weights = cs.cell_declustering(composites, scheme.encode(composites["LITH"]), cell_size=50.0).weights
layers = (23.0, 55.0, 0.0)
variograms = [
    cs.Variogram([("spherical", 0.9, 400.0)], nugget=0.1, rotation=(0.0, 0.0, 0.0), ratios=(1.0, 0.05)),
    cs.Variogram([("spherical", 0.9, 500.0)], nugget=0.1, rotation=layers, ratios=(0.8, 0.2)),
    cs.Variogram([("spherical", 0.85, 400.0)], nugget=0.15, rotation=layers, ratios=(0.8, 0.2)),
    cs.Variogram([("spherical", 0.8, 150.0)], nugget=0.2, rotation=layers, ratios=(0.8, 0.25)),
    cs.Variogram([("spherical", 0.9, 500.0)], nugget=0.1, rotation=layers, ratios=(0.8, 0.2)),
    cs.Variogram([("spherical", 0.7, 40.0)], nugget=0.3),
]
passes = [
    cs.Search(150.0, max_samples=24, max_per_hole=6, rotation=layers, ratios=(1.0, 0.4)),
    cs.Search(300.0, max_samples=24, rotation=layers, ratios=(1.0, 0.4)),
]
cik = cs.CategoricalIndicatorKriging(variograms, passes, scheme=scheme)
cik.fit(composites, "LITH", weights=weights, holes="HOLE_ID")

center = np.array([12350.0, 29900.0, 0.0])
across = np.array([np.sin(np.radians(113.0)), np.cos(np.radians(113.0)), 0.0])
along = np.array([np.sin(np.radians(23.0)), np.cos(np.radians(23.0)), 0.0])
origin = center - 600.0 * across - 5.0 * along + [0.0, 0.0, -400.0]
section = cs.BlockModel(origin, (10.0, 10.0, 10.0), (120, 1, 80), rotation=(23.0, 0.0, 0.0))
topography = data["topography"]
rows = topography.row_at(section.centroids[:, :2])
section = section.mask((rows >= 0) & (section.centroids[:, 2] < topography["Z"][rows]))
most_likely = cik.predict(section).most_likely
section = section.mask(~np.isnan(most_likely))
section = section.with_columns({"rock": most_likely[~np.isnan(most_likely)]})

grade = cs.Variogram([("spherical", 0.8, 200.0)], nugget=0.2, rotation=layers, ratios=(0.8, 0.2))
search = cs.Search(300.0, max_samples=24, max_per_hole=6, rotation=layers, ratios=(1.0, 0.4))
for column in ["ZN_PCT", "DENSITY"]:
    known = composites.filter(~np.isnan(composites[column]))
    kriged = cs.OrdinaryKriging(grade, search).fit(known, column, holes="HOLE_ID").predict(section)
    section = section.with_columns({column: kriged})
section = section.mask(~np.isnan(section["ZN_PCT"]) & ~np.isnan(section["DENSITY"]))
section = section.with_columns({"clean": cs.remove_small_units(section, "rock", min_volume=5000.0)})
print(f"{len(section)} cells; {np.sum(section['rock'] != section['clean'])} change rock")

# %% [markdown]
# ## The table
#
# With the block volumes of the model as weights and the kriged density, `tonnage` is in tonnes; with the zinc
# grades each cell also holds its `metal` (tonnes × %) and `mean_grade`. One row per pair of rocks, in the order of
# the scheme. The rows of a rock sum to its tonnage before the cleanup, its column to its tonnage after, and the
# metal to the metal of the model: the cleanup moves tonnes and metal between rocks, it creates or loses none.

# %%
table = cs.domain_change("rock", "clean", density="DENSITY", grades="ZN_PCT", scheme=scheme, data=section)
k = len(scheme)
tonnes = np.asarray(table["tonnage"]).reshape(k, k)
block = section.volumes * section["DENSITY"]
before = [block[section["rock"] == c].sum() for c in range(k)]
after = [block[section["clean"] == c].sum() for c in range(k)]
print(f"rows are the rocks before: {np.allclose(tonnes.sum(axis=1), before)}")
print(f"columns are the rocks after: {np.allclose(tonnes.sum(axis=0), after)}")
print(f"unchanged: {np.trace(tonnes) / tonnes.sum():.1%} of {tonnes.sum() / 1e6:.1f} Mt")
zinc = np.sum(block * section["ZN_PCT"]) / 100
print(f"zinc in the table {np.sum(table['metal']) / 100 / 1e3:.2f} kt, in the model {zinc / 1e3:.2f} kt")

start, end = np.asarray(table["from"]), np.asarray(table["to"])
moved = table.filter((start != end) & (np.asarray(table["tonnage"]) > 0))
for a, b, t, g in zip(moved["from"], moved["to"], moved["tonnage"], moved["mean_grade"]):
    print(f"{a:>4} to {b:4} {t / 1e3:5.1f} kt at {g:.2f} % Zn")

# %% [markdown]
# Under 1 % of the tonnage changes rock, but not at random grades: the volcanic cells taken into the sulphides run
# near 1.8 % Zn, while the sulphide specks given to the footwall hold 0.3 %. Drawn as matrices, each row as shares of
# the rock it starts from, the tonnage on the left and the zinc on the right; the diagonal is outlined and left blank,
# so the grays show only what moves.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
for ax, value, title in zip(axes, ["tonnage", "metal"], ["Tonnage", "Zinc"]):
    cs.plot.domain_change(table, value=value, relative=True, fmt="{:.1%}", ax=ax)
    ax.set(title=f"{title}, share of each rock before", xlabel="After cleanup", ylabel="Most likely rock")
axes[1].set_ylabel("")
save(fig, "change")
