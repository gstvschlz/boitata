"""
# Sequential indicator simulation

Sequential indicator simulation (SIS) draws categories. It visits the nodes along a random path, krigs the indicator
of each category with its own variogram from the samples and the nodes already drawn, and draws a category from
those probabilities. Each realization is one possible rock-type map; over many realizations, the frequency of each
category at a node is its probability. [Categorical indicator kriging](../../09-estimation/15-categorical-indicator-kriging/README.md) estimates such probabilities without
simulating.
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
from common import ACCENT, GRAY, save

# %% [markdown]
# ## Rock types
#
# The topsoil survey's rock types are known at each node of the prediction grid, so you can compare simulated maps with the real
# geology. A `Categories` scheme names the five types and gives each a color: `encode` turns labels into the codes 0
# to 4 that SIS takes, and `shares` gives their proportions.

# %%
data = bt.datasets.topsoil_metals()
train, grid = data["prediction"], data["grid"]
names = ["Argovian", "Kimmeridgian", "Sequanian", "Portlandian", "Quaternary"]
rock_types = bt.Categories(names, colors=["#0072b2", "#6f9fc9", "#c9d9ea", "#d55e00", "#8c8c8c"])
rock = rock_types.encode(train["Rock"]).astype(int)
true_rock = rock_types.encode(grid["Rock"]).astype(int)
print(f"{len(rock)} samples, {len(true_rock)} grid nodes")
print(f"{'':<13}{'samples':>8}{'share':>7}{'true grid':>11}")
for k, name in enumerate(names):
    print(f"{name:<13}{np.sum(rock == k):8d}{np.mean(rock == k):7.2f}{np.mean(true_rock == k):11.2f}")

# %% [markdown]
# ## Indicator variograms
#
# Each rock type gets the variogram of its indicator, 1 inside the type and 0 outside. Portlandian has too few
# samples for an experimental variogram, so it gets a short spherical model with the indicator's variance as sill.

# %%
experimental, variograms = [], []
for k in range(5):
    indicator = (rock == k).astype(float)
    if indicator.sum() >= 5:
        experimental.append(bt.experimental_variogram(train.coords, indicator, 0.1, 1.5))
        variograms.append(experimental[-1].fit("spherical"))
    else:
        experimental.append(None)
        variograms.append(bt.Variogram([("spherical", indicator.var(), 0.5)]))
for name, v in zip(names, variograms):
    s = v.structures[0]
    print(f"{name:<13}nugget {v.nugget:.3f}, sill {s.sill:.3f}, range {s.range:.2f} km")

# %%
h = np.linspace(0, 1.5, 61)
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), layout="constrained", sharey=True)
for ax, name, e, v in zip(axes, names, experimental, variograms):
    if e is not None:
        ax.plot(e.lags, e.gammas, "o", color=GRAY, ms=4, label="Experimental")
    ax.plot(h, v.gamma(h), color=ACCENT, label="Model")
    ax.set(title=name, xlabel="Lag (km)")
axes[0].set_ylabel("Indicator semivariance")
axes[0].legend(loc="lower right")
save(fig, "variograms")

# %% [markdown]
# ## Realizations
#
# `simulate` returns a summary over `n` realizations: `probabilities` holds each type's frequency, one row per node,
# `most_likely` the most frequent type, and `entropy`, scaled to [0, 1], how evenly the realizations disagree.
# `keep=True` keeps the maps.

# %%
sis = bt.SIS(variograms, bt.Search(radius=1.5, max_samples=16)).fit(train, rock)
summary = sis.simulate(grid, n=20, seed=3, keep=True)
maps = summary.realizations
shares = np.array([rock_types.shares(m) for m in maps])
print(f"{'':<13}{'true grid':>10}{'mean':>7}{'min':>7}{'max':>7}")
for k, name in enumerate(names):
    print(
        f"{name:<13}{np.mean(true_rock == k):10.2f}{shares[:, k].mean():7.2f}"
        f"{shares[:, k].min():7.2f}{shares[:, k].max():7.2f}"
    )
match = np.mean(maps == true_rock, axis=1)
print(f"a realization matches the true rock type at {match.min():.0%} to {match.max():.0%} of the nodes")
print(
    f"the most likely type matches at {np.mean(summary.most_likely == true_rock):.0%}; "
    f"mean entropy {summary.entropy.mean():.2f}"
)

# %%
cmap, norm = bt.plot.category_colors(rock_types)
xy = grid.coords[:, :2].T
fig, axes = plt.subplots(2, 3, figsize=(12, 9.5), layout="constrained")
categorical = [(true_rock, "True rock types"), (maps[0], "Realization 1"), (maps[1], "Realization 2")]
for ax, (cats, title) in zip(axes[0], categorical):
    ax.scatter(*xy, c=cats, cmap=cmap, norm=norm, s=7, marker="s", linewidths=0)
    ax.set_title(title)
axes[1, 0].scatter(*xy, c=summary.most_likely, cmap=cmap, norm=norm, s=7, marker="s", linewidths=0)
axes[1, 0].set_title("Most likely over 20 realizations")
for ax, values, title, label, color in (
    (axes[1, 1], summary.probabilities[:, 0], "P(Argovian)", "Probability", "cividis"),
    (axes[1, 2], summary.entropy, "Entropy", "Entropy", "Greys"),
):
    im = ax.scatter(*xy, c=values, s=7, marker="s", linewidths=0, vmin=0, vmax=1, cmap=color)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, orientation="horizontal", label=label)
for ax in axes.flat:
    ax.set_aspect("equal")
    ax.set(xlabel="X (km)", ylabel="Y (km)")
bt.plot.category_legend(rock_types, fig, loc="outside lower center", ncol=5)
save(fig, "realizations")

# %% [markdown]
# Each realization matches the true rock type at 44 % to 48 % of the nodes, and two realizations differ wherever the
# samples leave room. On average the proportions stay within 0.02 of the true grid, except Portlandian: no realization
# recovers its 5 % of the area from 3 of 259 samples. The most likely type matches at 57 %, more than any
# realization, though it is a smooth estimate and no possible map looks like it. Entropy peaks at the contacts and
# far from the samples.

# %% [markdown]
# ## Locally varying anisotropy
#
# `anisotropy=` takes a [LocalAnisotropy](../../09-estimation/14-local-anisotropy/README.md) as `SGS` does: each node
# searches its own ellipse and every category's indicator variogram turns with it. Here the directions follow the
# Argovian probability map on a regular grid, smoothed over 0.5 km, with across-strike ranges 40 % of along-strike:

# %%
box = bt.BlockModel(origin=(0.3, 0.4), size=(0.1, 0.1), count=(47, 54))
guide = bt.OrdinaryKriging(variograms[0], bt.Search(radius=1.5, max_samples=16)).fit(
    train, (rock == 0).astype(float)
)
lva = bt.LocalAnisotropy.from_grid(
    box.with_column("p", guide.predict(box)), "p", window=3, ratios=(0.4, 1.0)
).smooth(0.5)
anisotropic = sis.simulate(grid, n=20, seed=3, anisotropy=lva)
print(
    f"most likely type matches at {np.mean(anisotropic.most_likely == true_rock):.0%}; "
    f"mean entropy {anisotropic.entropy.mean():.2f}"
)

# %% [markdown]
# ## A shared path on block models
#
# On a regular or masked BlockModel, `simulate` walks one multigrid path for every realization: each node's neighbors
# and kriging weights are found once for the whole batch, and each realization still draws its own categories. On
# 100 000 nodes this ran about six times faster than a new random path per realization. The realizations share the
# path's order, which could make them more alike; across five seeds the share of nodes where two realizations differ
# and the mean entropy stayed within the seed-to-seed spread of the random path. `path="random"` keeps a new random
# path per realization:

# %%
for path in ("shared", "random"):
    s = sis.simulate(box, n=20, seed=3, path=path)
    print(f"{path:<7} mean entropy {s.entropy.mean():.2f}, Portlandian {s.proportions[:, 3].mean():.3f}")
