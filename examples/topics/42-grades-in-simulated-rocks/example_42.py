"""
# 42. Simulation methods

Sequential indicator simulation (SIS) and plurigaussian simulation (PGS) simulate categories; continuous variables
are in topics 36 to 39. Several correlated grades are simulated through independent factors in topic 43.
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
from common import save

# %% [markdown]
# ## Categories
#
# Jura's rock types are known everywhere on the prediction grid, so simulated categories can be compared with the
# real geology. SIS krigs, at each node, the probability of every rock type from indicator variograms; PGS truncates
# a Gaussian field at thresholds set by the proportions, which orders the types. A hierarchical `rule` truncates
# several fields in turn: here the first sets Quaternary cover apart from the Jurassic, and the second orders the
# Jurassic stages from Argovian to Portlandian, so the cover may touch every stage but each stage touches only the
# next. A `Categories` scheme names the rock types and gives each a color: `encode` turns labels into the codes 0
# to 4 that the simulations take, `shares` gives their proportions, and `plot.category_colors` and
# `plot.category_legend` draw the codes in the scheme's colors.

# %%
train = cs.datasets.jura()["prediction"]
jura_grid = cs.datasets.jura()["grid"]
names = ["Argovian", "Kimmeridgian", "Sequanian", "Portlandian", "Quaternary"]
rock_types = cs.Categories(names, colors=["#1f4e79", "#6f9fc9", "#c9d9ea", "#c05a28", "#8c8c8c"])
rock = rock_types.encode(train["Rock"]).astype(int)
true_rock = rock_types.encode(jura_grid["Rock"]).astype(int)
proportions = rock_types.shares(rock)

indicator_models = []
for k in range(5):
    indicator = (rock == k).astype(float)
    if indicator.sum() >= 5:
        fitted = cs.experimental_variogram(train.coords, indicator, 0.1, 1.5).fit("spherical")
    else:
        fitted = cs.Variogram([("spherical", indicator.var(), 0.5)])
    indicator_models.append(fitted)

sis = cs.SIS(indicator_models, cs.Search(radius=1.5, max_samples=16)).fit(train, rock)
sis_summary = sis.simulate(jura_grid, n=10, seed=3, realizations=True)
by_sis = sis_summary.realizations
latent = cs.Variogram([("spherical", 1.0, 0.8)])
pgs = cs.Plurigaussian(latent, proportions=proportions).fit(train, rock)
by_pgs = pgs.simulate(jura_grid, n=1, seed=3, realizations=True).realizations[0]
stages = (1, [names.index(n) for n in ("Argovian", "Sequanian", "Kimmeridgian", "Portlandian")])
rule = (0, [stages, names.index("Quaternary")])
hierarchy = cs.Plurigaussian([latent, latent], proportions=proportions, rule=rule).fit(train, rock)
by_rule = hierarchy.simulate(jura_grid, n=1, seed=3, realizations=True).realizations[0]

simulated = (("SIS", by_sis[0]), ("PGS ordered", by_pgs), ("PGS rule", by_rule))
print(f"{'':>13}" + "".join(f"{n[:5]:>8}" for n in names))
for label, cats in (("samples", rock), ("true grid", true_rock), *simulated):
    print(f"{label:>13}" + "".join(f"{s:8.2f}" for s in rock_types.shares(cats)))
for label, cats in simulated:
    print(f"{label}: {np.mean(cats == true_rock):.0%} of nodes match the true rock type")
matches = np.mean(sis_summary.most_likely == true_rock)
print(
    f"SIS most likely type over 10 realizations: {matches:.0%} match, mean entropy {sis_summary.entropy.mean():.2f}"
)

# %%
cmap, norm = cs.plot.category_colors(rock_types)
fig, axes = plt.subplots(2, 2, figsize=(8.5, 10), layout="constrained")
panels = [
    (true_rock, "True rock types"),
    (by_sis[0], "SIS realization"),
    (by_pgs, "PGS realization, ordered"),
    (by_rule, "PGS realization, hierarchical rule"),
]
for ax, (cats, title) in zip(axes.flat, panels):
    ax.scatter(*jura_grid.coords[:, :2].T, c=cats, cmap=cmap, norm=norm, s=7, marker="s", linewidths=0)
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="X (km)", ylabel="Y (km)")
cs.plot.category_legend(rock_types, fig, loc="outside lower center", ncol=5)
save(fig, "categories")

# %% [markdown]
# SIS gives each type its own indicator variogram and matches the true type at half the nodes. Ordered PGS reproduces
# the proportions closely, but its rule only allows contacts between neighbors in the order, so Portlandian appears
# as specks along every Sequanian-Quaternary contact. The hierarchical rule puts Portlandian next to Kimmeridgian and
# under the cover, as in the true map, and matches about as many nodes as SIS. None recovers Portlandian's 5 % of the
# area from 3 of 259 samples.

# %% [markdown]
# Proportions need not be global. Given local proportions at the samples at `fit` and at the nodes at `simulate`,
# the rule's thresholds follow them, so each rock type is likelier where its samples cluster. Here they are the rock
# types of the samples averaged with Gaussian weights of 300 m, shrunk towards the global proportions where samples
# are sparse.

# %%
onehot = np.eye(5)[rock]


def local_proportions(xy, bandwidth=0.3):
    d2 = ((xy[:, None, :2] - train.coords[None, :, :2]) ** 2).sum(-1)
    w = np.exp(-0.5 * d2 / bandwidth**2)
    return (w @ onehot + proportions) / (w.sum(axis=1, keepdims=True) + 1)


at_nodes = local_proportions(jura_grid.coords)
hierarchy.fit(train.coords, rock, proportions=local_proportions(train.coords))
by_local = hierarchy.simulate(jura_grid, n=1, seed=3, realizations=True, proportions=at_nodes).realizations[0]
print(f"PGS rule, local proportions: {np.mean(by_local == true_rock):.0%} of nodes match the true rock type")

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), layout="constrained")
im = axes[0].scatter(
    *jura_grid.coords[:, :2].T, c=at_nodes[:, names.index("Argovian")], s=7, marker="s", linewidths=0
)
fig.colorbar(im, ax=axes[0], shrink=0.8, orientation="horizontal", label="local Argovian proportion")
axes[1].scatter(*jura_grid.coords[:, :2].T, c=by_local, cmap=cmap, norm=norm, s=7, marker="s", linewidths=0)
for ax, title in zip(axes, ("Local proportion of Argovian", "PGS realization, local proportions")):
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="X (km)", ylabel="Y (km)")
cs.plot.category_legend(rock_types, fig, loc="outside lower center", ncol=5)
save(fig, "local-proportions")

# %% [markdown]
# With local proportions, Argovian keeps to the north-west and the south where its samples are, and the realization
# matches the true rock type at 56 % of the nodes, against 49 % with global proportions.
#
# The latent variograms so far were guesses. Each rock type's indicator variogram follows from the rule and the latent
# variograms, so `fit_variograms` rescales the latent ranges until those implied variograms match the experimental
# ones; Portlandian, with 3 samples, is left out.

# %%
from common import ACCENT, GRAY

experimental = [
    cs.experimental_variogram(train.coords, (rock == k).astype(float), 0.1, 1.5)
    if names[k] != "Portlandian"
    else None
    for k in range(5)
]
guessed = cs.Plurigaussian([latent, latent], proportions=proportions, rule=rule)
fitted = cs.Plurigaussian([latent, latent], proportions=proportions, rule=rule).fit_variograms(experimental)
cover, stage = (v.structures[0].range for v in fitted.variograms)
print(f"fitted latent ranges: {cover:.2f} km for the cover field, {stage:.2f} km for the stages field")
fitted.fit(train.coords, rock, proportions=local_proportions(train.coords))
by_fitted = fitted.simulate(jura_grid, n=1, seed=3, realizations=True, proportions=at_nodes).realizations[0]
print(
    f"PGS rule, local proportions, fitted: {np.mean(by_fitted == true_rock):.0%} of nodes match the true rock type"
)

# %%
h = np.linspace(0, 1.5, 61)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), layout="constrained", sharey=True)
for ax, k in zip(axes, [k for k in range(5) if experimental[k] is not None]):
    ax.plot(experimental[k].lags, experimental[k].gammas, "o", color=GRAY, ms=4, label="experimental")
    ax.plot(h, guessed.indicator_variograms(h)[k], "--", color=GRAY, label="guessed ranges")
    ax.plot(h, fitted.indicator_variograms(h)[k], color=ACCENT, label="fitted ranges")
    ax.set(title=names[k], xlabel="lag (km)")
axes[0].set_ylabel("indicator semivariance")
axes[0].legend(frameon=False, loc="lower right")
save(fig, "latent-variograms")

# %% [markdown]
# The fitted cover field is short and the stages field long, so the stages form broad bands that the cover patches
# over; with 0.8 km on both, the stages varied too fast. The fitted variograms follow the experimental points of
# every rock type and bring the match to 59 %.

# %% [markdown]
# ## Grades within simulated rock types
#
# Simulated rock types can host the grade simulation. Fitted with `domains`, SGS normal-scores Co within each rock
# type, Argovian holding about half the Co of the others. Given the `(n, targets)` array of SIS realizations as
# `domains`, realization k of Co is simulated within realization k of the rock types, so the grades carry the
# uncertainty of the contacts; one row of labels, here the true rock types, holds the domains fixed.

# %%
co = train["Co"]
co_scores = np.empty(len(co))
for k in range(5):
    co_scores[rock == k] = cs.NormalScore().fit_transform(co[rock == k])
co_variogram = cs.experimental_variogram(train.coords, co_scores, 0.1, 1.5).fit("spherical")
cobalt = cs.SGS(co_variogram, cs.Search(radius=1.5, max_samples=16)).fit(train, "Co", domains=rock)
within_true = cobalt.simulate(jura_grid, n=10, seed=3, realizations=True, domains=true_rock).realizations
within_sis = cobalt.simulate(jura_grid, n=10, seed=3, realizations=True, domains=by_sis).realizations

argovian = (by_sis == 0).mean(axis=0)
unsure = (argovian > 0) & (argovian < 1)
print(f"{'Co (ppm)':<42}{'true rock types':>16}{'SIS rock types':>16}")
rows = [
    ("mean", lambda r: r.mean()),
    ("mean on true Argovian", lambda r: r[:, true_rock == 0].mean()),
    ("std where SIS is unsure of Argovian", lambda r: r.std(axis=0)[unsure].mean()),
    ("std elsewhere", lambda r: r.std(axis=0)[~unsure].mean()),
]
for label, stat in rows:
    print(f"{label:<42}{stat(within_true):16.2f}{stat(within_sis):16.2f}")
print(f"SIS is unsure whether {unsure.mean():.0%} of the nodes are Argovian")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
panels = [
    (within_true[0], "Co within true rock types", "Co (ppm)", 18),
    (within_sis[0], "Co within SIS realization 1", "Co (ppm)", 18),
    (within_sis.std(axis=0), "Co across SIS realizations", "standard deviation (ppm)", 5),
]
for ax, (image, title, label, top) in zip(axes, panels):
    im = ax.scatter(*jura_grid.coords[:, :2].T, c=image, s=7, marker="s", linewidths=0, vmin=0, vmax=top)
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="X (km)", ylabel="Y (km)")
    fig.colorbar(im, ax=ax, shrink=0.8, orientation="horizontal", label=label)
save(fig, "grades-in-rock-types")

# %% [markdown]
# Within the true rock types Co drops sharply at every Argovian contact. Within SIS's rock types the contacts move
# from one realization to the next, so the lean Argovian Co spreads over its uncertain margin: where SIS is unsure
# of Argovian, the spread of Co across realizations grows by about 40 %.
