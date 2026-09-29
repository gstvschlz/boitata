"""
# Soft boundaries

A laterite grades from limonite (`LIM`) down into saprolite (`SAP`), and nickel does not step cleanly at the logged
contact. Fitted with `domains=` or `domain_column=`, an estimator informs each target from samples of its own domain
only: a hard boundary. `Search(..., soft=...)` opens it: samples of another domain within the soft distance inform a
target too, both ways or, with a dict, one way only. Kriging and SGS take the same search.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[2]))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, INK, save

data = cs.datasets.nickel_laterite_profile()
intervals = cs.merge_intervals(data["assays"], data["horizons"])
samples = cs.Drillholes(data["collars"], data["surveys"], intervals).samples()
samples = samples.filter(np.isin(np.asarray(samples["HORIZON"]), ["LIM", "SAP"]))
horizon = np.asarray(samples["HORIZON"])
for name in ("LIM", "SAP"):
    print(
        f"{name}: {np.sum(horizon == name)} samples of 1 m, mean {samples['NI_PCT'][horizon == name].mean():.2f} % Ni"
    )


def profile(points, values):
    return cs.contact(
        points,
        values,
        domain_column="HORIZON",
        holes="HOLE_ID",
        inside="SAP",
        outside="LIM",
        max_distance=7.0,
        bin=1,
    )


# %% [markdown]
# Down each hole, the mean grade against the distance to the `LIM`/`SAP` contact:

# %%
observed = profile(samples, "NI_PCT")
print("distance:", observed["distance"])
print("mean Ni: ", observed["mean"].round(2))

fig, ax = cs.plot.contact(observed, labels=("SAP", "LIM"))
fig.set_size_inches(6, 3.4)
ax.set(
    xlabel="Distance to the LIM/SAP contact along the hole (m)",
    ylabel="Mean Ni (%)",
    title="Ni across the contact",
)
save(fig, "contact")

# %% [markdown]
# `LIM` is flat up to the contact. `SAP` loses grade steadily toward it, from 1.86 % Ni 6 to 7 m below to 1.32 % in the
# last meter, and the step at the contact itself is small: the boundary is gradational on the saprolite side.
#
# One normal-score variogram serves both horizons: each is scored on its own, then the scores are pooled. The nugget
# and vertical range come from pairs down the holes, the horizontal range from pairs across them. Kriging weights do
# not depend on the sill, so the same model serves kriging and SGS.

# %%
scores = np.empty(len(samples))
for name in ("LIM", "SAP"):
    scores[horizon == name] = cs.NormalScore().fit_transform(samples["NI_PCT"][horizon == name])
down = cs.experimental_variogram(samples, scores, 1, 12, holes="HOLE_ID").fit("spherical")
across = cs.experimental_variogram(samples, scores, 25, 250, azimuth=0, tolerance=22.5).fit(
    "spherical", nugget=down.nugget
)
model = across.with_anisotropy((0, 0, 0), (1.0, down.structures[0].range / across.structures[0].range))
print(model)

# %% [markdown]
# The holes on the 50 m mesh do the estimating; the 25 m infill holes are held out to check it. Three rules share one
# search, flattened ten to one, near the seven to one of the variogram. The soft distance is measured in that
# ellipsoid, so 50 reaches 50 m across and 5 m up or down, most of the transition. The one-way rule lets `SAP` draw on
# `LIM` but not the other way round.

# %%
collars = data["collars"]
offset = [np.abs((collars[axis] + 25) % 50 - 25) for axis in ("X", "Y")]
infill = np.asarray(collars["HOLE_ID"])[(offset[0] > 12) | (offset[1] > 12)]
held_out = np.isin(np.asarray(samples["HOLE_ID"]), infill)
mesh, check = samples.filter(~held_out), samples.filter(held_out)
print(f"{len(collars) - len(infill)} mesh holes, {len(infill)} infill holes held out")

rules = {"hard": None, "soft both ways": 50.0, "soft, SAP from LIM": {("SAP", "LIM"): 50.0}}
searches = {
    name: cs.Search(200, max_samples=24, min_samples=4, max_per_hole=6, ratios=(1.0, 0.1), soft=soft)
    for name, soft in rules.items()
}
truth = check["NI_PCT"]
check_horizon = np.asarray(check["HORIZON"])
kriged = {}
for name, search in searches.items():
    ok = cs.OrdinaryKriging(model, search).fit(mesh, "NI_PCT", holes="HOLE_ID", domain_column="HORIZON")
    result = ok.predict(check, diagnostics=True, domain_column="HORIZON")
    kriged[name] = result["value"]
    error = result["value"] - truth
    rmse = {h: np.sqrt(np.mean(error[check_horizon == h] ** 2)) for h in ("LIM", "SAP")}
    other = {h: np.mean(result["n_other_domain"][check_horizon == h] > 0) for h in ("LIM", "SAP")}
    print(
        f"{name:>18}: RMSE LIM {rmse['LIM']:.3f}, SAP {rmse['SAP']:.3f}; "
        f"other horizon used by {other['LIM']:.0%} of LIM, {other['SAP']:.0%} of SAP targets"
    )

# %% [markdown]
# Most targets lie far from the contact, so the errors hardly move. The mean estimate by distance to the contact
# shows where the rules differ:


# %%
def means(values):
    return profile(check, values)["mean"].round(2)


distance = profile(check, "NI_PCT")["distance"]
print(f"{'distance':>18}:", distance)
print(f"{'held out':>18}:", means("NI_PCT"))
for name, values in kriged.items():
    print(f"{name:>18}:", means(values))

# %% [markdown]
# SGS takes the same searches and the same domains. Each horizon keeps its own normal-score table; a `LIM` sample
# informing a `SAP` node enters by its grade, scored through the `SAP` table. The mean of 20 realizations at the
# held-out samples:

# %%
simulated = {}
for name, search in searches.items():
    sgs = cs.SGS(model, search).fit(mesh, "NI_PCT", holes="HOLE_ID", domain_column="HORIZON")
    simulated[name] = sgs.simulate(check, n=20, seed=7, domain_column="HORIZON").mean
    print(f"{name:>18}:", means(simulated[name]))

# %%
styles = {"hard": (GRAY, "-"), "soft both ways": (HIGHLIGHT, "--"), "soft, SAP from LIM": (ACCENT, "-")}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True, layout="constrained")
for ax, estimates, title in ((axes[0], kriged, "Ordinary kriging"), (axes[1], simulated, "SGS, mean of 20")):
    for i, side in enumerate((distance < 0, distance > 0)):
        ax.plot(
            distance[side],
            means("NI_PCT")[side],
            "o",
            color=INK,
            ms=4,
            label=None if i else "held-out samples",
        )
        for name, values in estimates.items():
            color, style = styles[name]
            ax.plot(distance[side], means(values)[side], ls=style, color=color, label=None if i else name)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set(title=title, xlabel="Distance to the contact (m), SAP < 0 < LIM")
axes[0].set_ylabel("Mean Ni (%)")
axes[1].legend(loc="upper right")
save(fig, "profiles")

# %% [markdown]
# In the last meter of `SAP` the hard boundary kriges 1.55 % Ni against 1.28 % held out: it sees only saprolite,
# richer deeper down. Letting `SAP` draw on `LIM` brings it to 1.34 % and follows the gradient. Opened both ways, the
# boundary also lifts the first meter of `LIM` from 1.03 % to 1.21 %, where the held-out samples stay at 1.06 %. SGS
# agrees: 1.60 % hard and 1.24 % one way in the last meter of `SAP`, and 1.25 % in the first meter of `LIM` when soft
# both ways. The contact profile says which way to open a boundary; here only the saprolite side is gradational.
