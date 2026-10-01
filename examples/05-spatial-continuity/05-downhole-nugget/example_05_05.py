"""
# downhole nugget

along a hole, samples sit one sample length apart, far closer than any two holes, so the downhole variogram shows the
nugget best. the data: Ni of a nickel laterite, sampled at 1 m down 448 vertical holes, horizon by horizon.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, save

tables = bt.datasets.nickel_laterite_profile()
intervals = bt.merge_intervals(tables["assays"], tables["horizons"])
drillholes = bt.Drillholes(tables["collars"], tables["surveys"], intervals)
runs = drillholes.composite(None, ["NI_PCT"], domain="HORIZON")
horizon, thickness = np.array(runs["HORIZON"]), np.array(runs["length"])
for name in ["FERR", "LIM", "SAP", "BRK"]:
    print(f"{name:5} mean thickness {thickness[horizon == name].mean():5.1f} m")


# %% [markdown]
# a downhole variogram needs several lags inside one horizon, and only the limonite `LIM` and the saprolite `SAP` are
# thick enough. `holes=` keeps only pairs down the same hole; lag `k` gathers pairs about `k` composite lengths apart.
# `nugget()` extrapolates a line through the first three lags to zero. composites never cross a horizon, and each
# horizon gets its own variogram, standardized by its variance, at three composite lengths.

# %%
horizons, lengths = ["LIM", "SAP"], [1.0, 1.5, 2.0]
nuggets, downhole = {}, {}
for length in lengths:
    composites = drillholes.composite(length, ["NI_PCT"], domain="HORIZON", residual="merge")
    horizon, ni = np.array(composites["HORIZON"]), composites["NI_PCT"]
    for name in horizons:
        keep = (horizon == name) & ~np.isnan(ni)
        exp = bt.experimental_variogram(
            composites.coords[keep],
            ni[keep],
            length,
            6 * length,
            standardize=True,
            holes=np.array(composites["HOLE_ID"])[keep],
        )
        nuggets[name, length], downhole[name, length] = exp.nugget(), exp
print("nugget / variance")
print("      " + "".join(f"{h:>7}" for h in horizons))
for length in lengths:
    print(f"{length:3.1f} m " + "".join(f"{nuggets[h, length]:7.2f}" for h in horizons))


# %% [markdown]
# at 1 m the limonite carries a fifth of its Ni variance in the nugget, the saprolite about a seventh; 1.5 m composites
# give nearly the same. at 2 m the first three lags span 6 m, most of the limonite's 7.4 m. the line then reaches zero
# across the bend of the variogram, and the nugget comes out larger in both horizons. carry the nugget of the shortest
# composites into the fits between holes, whose closest spacing is 25 m.

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.4), layout="constrained")
for name, color in (("LIM", ACCENT), ("SAP", GRAY)):
    exp = downhole[name, 1.0]
    a.plot(exp.lags, exp.gammas, "o-", color=color, ms=3, label=name)
    a.plot([0, exp.lags[0]], [exp.nugget(), exp.gammas[0]], ":", color=color)
    a.plot(0, exp.nugget(), "s", color=HIGHLIGHT, ms=5, clip_on=False)
a.set(xlim=(0, 6), ylim=(0, None), xlabel="Downhole lag (m)", ylabel="standardized γ(h)")
a.set_title("Downhole Ni variograms, 1 m composites")
a.legend(loc="lower right")
for name, color in zip(horizons, (ACCENT, GRAY)):
    b.plot(lengths, [nuggets[name, length] for length in lengths], "o-", color=color, ms=4, label=name)
b.set(xticks=lengths, ylim=(0, None), xlabel="Composite length (m)", ylabel="nugget / variance")
b.set_title("Extrapolated nugget by horizon")
b.legend(loc="lower right", ncol=2)
save(fig, "downhole")
