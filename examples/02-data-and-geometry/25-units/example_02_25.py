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
