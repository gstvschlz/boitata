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
# ## A shared path on block models
#
# On a regular or masked BlockModel, `simulate` walks one multigrid path for every realization: each node's neighbors
# and kriging weights are found once for the whole batch, and each realization still draws its own categories. On
# 100 000 nodes this ran about six times faster than a new random path per realization. The realizations share the
# path's order, which could make them more alike; across five seeds the share of nodes where two realizations differ
# and the mean entropy stayed within the seed-to-seed spread of the random path. `path="random"` keeps a new random
# path per realization:

# %%
# One layer centred at z = 0, the elevation of the 2D samples, so a block holding a sample sits on it.
box = bt.BlockModel(origin=(0.3, 0.4, -0.5), size=(0.1, 0.1, 1.0), count=(47, 54, 1))
for path in ("shared", "random"):
    s = sis.simulate(box, n=20, seed=3, path=path)
    print(f"{path:<7} mean entropy {s.entropy.mean():.2f}, Portlandian {s.proportions[:, 3].mean():.3f}")

# %% [markdown]
# ## Cleaning short-scale noise
#
# SIS realizations are noisier at short range than their indicator variograms say. `clean`, an odd window side in
# blocks, replaces each block by the most probable category in the window around it (maximum a-posteriori selection,
# Deutsch 1998): nearer blocks weigh more, a block whose centre is a sample keeps its type and weighs 5 times more,
# and each type is scaled by its sample proportion over its proportion in the realization. The same 20 realizations
# with and without a 5 by 5 window, counting the type changes between neighboring blocks along X:


# %%
def changes(realizations):
    r = np.asarray(realizations).reshape(len(realizations), 54, 47)
    return np.mean(r[:, :, 1:] != r[:, :, :-1])


raw = sis.simulate(box, n=20, seed=3, keep=True)
cleaned = sis.simulate(box, n=20, seed=3, keep=True, clean=5)
for name, s in (("raw", raw), ("clean=5", cleaned)):
    print(
        f"{name:<8} type changes {changes(s.realizations):.3f}, Portlandian {s.proportions[:, 3].mean():.3f}"
    )

# %%
main = int(np.argmax(np.bincount(rock)))
lags = np.arange(1, 11)


def semivariance(realizations, k):
    r = (np.asarray(realizations).reshape(len(realizations), 54, 47) == k).astype(float)
    return [0.5 * np.mean((r[:, :, h:] - r[:, :, :-h]) ** 2) for h in lags]


print(
    f"{names[main]} semivariance at 0.1 km: without cleaning {semivariance(raw.realizations, main)[0]:.3f}, "
    f"clean=5 {semivariance(cleaned.realizations, main)[0]:.3f}, model {variograms[main].gamma([0.1])[0]:.3f}"
)
print(f"{names[main]} nugget {variograms[main].nugget:.3f}")
extent = (0.3, 5.0, 0.4, 5.8)
fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), layout="constrained")
axes[0, 0].scatter(*xy, c=true_rock, cmap=cmap, norm=norm, s=7, marker="s", linewidths=0)
axes[0, 0].set_title("True rock types")
for row, (name, s) in enumerate((("Without cleaning", raw), ("clean=5", cleaned))):
    for col in (1, 2):
        ax = axes[row, col]
        r = np.asarray(s.realizations)[col - 1].reshape(54, 47)
        ax.imshow(r, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        ax.plot(*train.coords[:, :2].T, ".", color="black", ms=2)
        ax.set_title(f"{name}, realization {col}")
ax = axes[1, 0]
ax.plot(0.1 * lags, semivariance(raw.realizations, main), "o-", color=GRAY, label="Without cleaning")
ax.plot(0.1 * lags, semivariance(cleaned.realizations, main), "o-", color=ACCENT, label="clean=5")
ax.plot(0.1 * lags, variograms[main].gamma(0.1 * lags), "--", color="black", label="Model")
ax.set(title=f"{names[main]} indicator, along X", xlabel="Lag (km)", ylabel="Semivariance")
ax.legend(loc="lower right")
for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]):
    ax.set_aspect("equal")
    ax.set(xlabel="X (km)", ylabel="Y (km)")
bt.plot.category_legend(rock_types, fig, loc="outside lower center", ncol=5)
save(fig, "cleaning")

# %% [markdown]
# Cleaning cuts the type changes between neighbors from 0.41 to 0.16 and turns scattered blocks into patches. Whether
# that is right depends on the variograms. Kimmeridgian's has a nugget of 0.139 out of 0.224, so the model itself
# expects blocks 0.1 km apart to differ often: the realizations without cleaning sit near it (0.128 against 0.150),
# and cleaning drops the semivariance to 0.047, far more continuous than the model. Here the noise is in the data,
# and cleaning removes it from the realizations too. Clean when realizations are noisier than their variograms, as
# with models of small nugget, and leave it off when the nugget is real. Cleaning cannot recover Portlandian either:
# its target is the 1 % share of the samples, against 5 % of the true grid, so the proportion term has nothing to
# pull toward.
