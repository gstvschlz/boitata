"""
# high-grade restriction

a rich sample informs every node its search reaches, so an isolated high value spreads its grade far beyond the ground
it represents. `bt.HighGrade` restricts samples above a threshold beyond a distance: `mode="drop"` leaves them out,
`mode="clamp"` keeps them at the threshold, and ranges with a rotation give the restriction its own ellipse. each search
pass carries its own restriction. the example block-kriges walker lake `V` in 5 m blocks and checks it against the
exhaustive block averages.
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
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, save

samples = bt.datasets.walker_lake()
xy, v = samples.coords[:, :2], samples["V"]
exhaustive = bt.datasets.walker_lake_exhaustive()["V"].reshape(300, 260)
grid = bt.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
truth = exhaustive.reshape(60, 5, 52, 5).mean(axis=(1, 3)).ravel()

azimuths = np.arange(0, 180, 22.5)
directional = [bt.experimental_variogram(samples, "V", 10.0, 120.0, azimuth=a) for a in azimuths]
model = bt.Variogram.fit_directional(
    directional, [(a, 0) for a in azimuths], ["spherical", "spherical"], weighting="count/gamma"
)
kriging = bt.BlockKriging(model, bt.Search(radius=80), (5, 5)).fit(samples, "V")


def search(high_grade=None, *, radius=80, min_samples=4):
    ellipse = {"rotation": model.rotation, "ratios": (0.5, 1.0)}
    return bt.Search(
        radius=radius, max_samples=24, min_samples=min_samples, octant=True, high_grade=high_grade, **ellipse
    )


# %% [markdown]
# ## four restrictions
#
# the threshold is 600 ppm, reached by almost a third of the samples, and the restricted distance is 20 m. the tuple
# `(600, 20)` is shorthand for `bt.HighGrade(600, 20)`, which drops. clamp keeps the far rich samples at 600 ppm, so
# they still pull the node up, by less. the ellipse restricts to 40 m along the continuity and 10 m across it. the
# passes restrict to 15 m in a tight first pass and to 30 m in the wide second, compared with the same passes
# unrestricted.
#
# blocks 20 to 50 m from the nearest rich sample form its halo: beyond the restricted distance, within the search. there
# an unrestricted estimate spreads rich values, and there the restriction acts.

# %%
threshold = 600
rich = xy[v > threshold]
nearest = np.sqrt(((grid.centroids[:, None, :2] - rich[None]) ** 2).sum(-1)).min(axis=1)
halo = (nearest > 20) & (nearest < 50)

searches = {
    "none": search(),
    "drop": search((threshold, 20)),
    "clamp": search(bt.HighGrade(threshold, 20, mode="clamp")),
    "ellipse": search(bt.HighGrade(threshold, (40, 10, 10), rotation=model.rotation)),
    "passes, none": [search(radius=30, min_samples=8), search()],
    "passes": [search((threshold, 15), radius=30, min_samples=8), search((threshold, 30))],
}
cutoffs = np.arange(100, 801, 50)
estimates, metal = {}, {}
print(f"{np.mean(v > threshold):.0%} of the samples above {threshold} ppm, {halo.sum()} halo blocks")
print(f"{'search':>13}  halo mean error  block RMSE  metal above 300 ppm  cross-validation mean error")
for name, s in searches.items():
    estimator = kriging.with_search(s)
    estimate = estimates[name] = estimator.predict(grid)
    metal[name] = [estimate[estimate > c].sum() / truth[truth > c].sum() for c in cutoffs]
    print(
        f"{name:>13}  {np.mean(estimate[halo] - truth[halo]):+10.1f} ppm  {np.sqrt(np.mean((estimate - truth) ** 2)):7.1f}"
        f"  {metal[name][4]:14.1%}  {estimator.cross_validate().mean_error:+18.1f} ppm"
    )

# %% [markdown]
# unrestricted, the halo blocks are overestimated by 16 ppm on average. dropping the rich samples beyond 20 m takes a
# third of that off, and clamp lands between the two, as it must: a far rich sample enters at 600 ppm instead of at its
# grade or not at all. the ellipse, which lets rich samples reach twice as far along the continuity, does almost as well
# as the circle. the restricted passes correct as much as the single restricted search.
#
# ## metal above cutoff
#
# metal above a cutoff is the sum of the block grades above it, here relative to the exhaustive truth.

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(8.8, 3.6), layout="constrained")
styles = {"none": (GRAY, "-"), "drop": (ACCENT, "-"), "clamp": (HIGHLIGHT, "-"), "ellipse": (ACCENT, ":")}
for name, (color, ls) in styles.items():
    a.plot(cutoffs, metal[name], color=color, ls=ls, lw=1.5, label=name)
a.axhline(1, color=INK, lw=0.8)
a.set(xlabel="Cutoff (ppm)", ylabel="Estimated / true metal", title="Metal above cutoff")
a.legend(loc="lower left")

names = list(searches)
errors = [np.mean(estimates[n][halo] - truth[halo]) for n in names]
colors = [GRAY if n.endswith("none") else ACCENT for n in names]
b.barh(names, errors, color=colors)
for y, e in enumerate(errors):
    b.text(e + 0.3, y, f"{e:+.1f}", va="center", color=INK)
b.axvline(0, color=INK, lw=0.8)
b.invert_yaxis()
b.set(xlabel="Mean error (ppm)", title="Halo blocks, 20 to 50 m from a rich sample")
b.spines["left"].set_color(LIGHT)
save(fig, "modes")

# %% [markdown]
# at low cutoffs the unrestricted estimate puts a little too much metal, and dropping brings it closest to the truth.
# from 400 ppm up every estimate falls about a fifth short: that is the smoothing of kriging, which no restriction
# cures. above 650 ppm the restriction even adds metal. inside dense clusters of rich samples, ordinary kriging gives
# the screened far ones small negative weights ([search](../../06-kriging/06-search/README.md)). dropping a negative
# weight on a high value raises the estimate, and clamping, which swaps it for a negative weight on the lower threshold,
# raises it too.
#
# the cross-validation mean error follows little of this: the samples sit in the clusters, where the restriction acts
# through those negative weights, and not in the halos. judge a restriction against the truth, or a simulated one.
