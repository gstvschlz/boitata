"""
# units

a column can carry a unit, kept in its arrow metadata and through parquet. units combine lengths, masses, grades,
ratios, angles and currencies, so `t/m3`, `USD/m` and `(g/t)^2` are units too. grades are a kind of their own: a
grade never converts to a plain ratio, and ore mass times a grade is metal (`kg metal`), never ore.
"""

# %% [hidden]
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import save

assays = bt.datasets.stacked_sulphide_lenses()["assays"]
folder = Path(tempfile.mkdtemp())
bt.write_csv(
    folder / "assays.csv",
    bt.Table({"ZN (%)": assays["ZN_PCT"], "AG [g/t]": assays["AG_GPT"], "DENSITY": assays["DENSITY"]}),
    progress=False,
)

# %% [markdown]
# ## units from the file
#
# `read_csv` splits a header such as `ZN (%)` or `AG [g/t]` into the column name and its unit. a column whose header
# says nothing takes its unit from `units=`.

# %%
table = bt.read_csv(folder / "assays.csv", units={"DENSITY": "t/m3"}, progress=False)
print(table.column_names)
print(table.units)

# %% [markdown]
# a project that always names its columns the same way declares the units once with `set_units`; readers and
# constructors then give those columns their unit whenever the file leaves it out. `with bt.units(...)` does the same
# for one block.

# %%
with bt.units(columns={"DENSITY": "t/m3"}):
    print(bt.read_csv(folder / "assays.csv", progress=False).units)

# %% [markdown]
# ## converting
#
# `convert_units` rescales a column to another unit of the same kind. `oz/t` is troy ounces per short ton, as North
# American reports write it.

# %%
silver = table.convert_units("AG", to="oz/t")
print(f"{table['AG'][:3]} g/t is {silver['AG'][:3]} oz/t")
density = table.convert_units("DENSITY", to="kg/m3")["DENSITY"]
print(f"mean density {np.nanmean(table['DENSITY']):.2f} t/m3 = {np.nanmean(density):.0f} kg/m3")

# %% [markdown]
# a conversion across kinds raises `InvalidInput`: a zinc grade in percent is not a ratio in percent.

# %%
try:
    table.convert_units("ZN", to="ratio%")
except bt.InvalidInput as error:
    print(error)

# %% [markdown]
# plots label their axes with the unit of the column.

# %%
fig, ax = bt.plot.histogram("AG", data=silver)
ax.set_title("silver grades, converted to oz/t")
save(fig, "histogram")
plt.show()

# %% [markdown]
# ## units through a workflow
#
# a column read from a container is a `UnitArray`: a numpy array that remembers its unit. composites keep the unit of
# the assays, an estimator fitted on a column estimates in its unit, and the kriging variance comes out in its square.
# `with_column` stores the unit with the new column.

# %%
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes(
    data["collars"], data["surveys"], data["assays"].with_units({"ZN_PCT": "%"}), length_unit="m"
)
composites = holes.composite(2.0, ["ZN_PCT"])
grid = bt.BlockModel.from_extents(
    data["lens_1"], size=(10, 10, 5), buffer=20, rotation=(22.5, 0.0, 55.0), length_unit="m"
)
kriging = bt.OrdinaryKriging(bt.Variogram([("spherical", 1.0, 60.0)]), bt.Search(radius=60, max_samples=12))
estimate, variance = kriging.fit(composites, "ZN_PCT").predict(grid, return_variance=True, progress=False)
print(f"estimate in {estimate.unit}, variance in {variance.unit}")
grid = grid.with_column("ZN", estimate)
print(grid.units)

# %% [markdown]
# arithmetic drops the unit, since the result may be in another one. give it back with `unit=`.

# %%
grid = grid.with_column("ZN_PPM", estimate * 1e4, unit="ppm")
print(grid.units)

# %% [markdown]
# ## lengths
#
# containers carry the length unit of their coordinates, and so do a `Search` and a `Variogram`. an estimator fitted
# with parameters in feet on data in metres converts the coordinates, so the estimate is the one the same lengths in
# metres give. containers in different units refuse to meet; `to_length_unit` converts one.

# %%
feet = bt.OrdinaryKriging(
    bt.Variogram([("spherical", 1.0, 60.0 / 0.3048)], length_unit="ft"),
    bt.Search(radius=60.0 / 0.3048, max_samples=12, length_unit="ft"),
)
in_feet = feet.fit(composites, "ZN_PCT").predict(grid, progress=False)
print(f"largest difference from the estimate in metres: {np.nanmax(np.abs(in_feet - estimate)):.1e} %")
try:
    kriging.predict(grid.to_length_unit("ft"), progress=False)
except bt.InvalidInput as error:
    print(error)

# %% [markdown]
# ## tonnage and metal
#
# with the block size in metres and a density in `t/m3`, `grade_tonnage` knows that tonnage is a mass: it reads in
# t, kt or Mt and metal in the units of the grade's metal, whichever reads best. metal never converts to ore tonnes.

# %%
grid = grid.with_column("DENSITY", np.full(len(grid), 3.1), unit="t/m3")
curve = bt.grade_tonnage("ZN", [0.0, 2.0, 4.0], data=grid, density="DENSITY")
print(curve.units)
print(curve.to_polars())
