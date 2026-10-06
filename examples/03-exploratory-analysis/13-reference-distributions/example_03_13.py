"""
# reference distributions

a normal-score transform built on the data alone knows nothing beyond the highest and lowest sample: with few or
clustered samples, the tails follow a straight line toward chosen bounds. a reference distribution fitted to the data
fills them smoothly. `KernelDensity` spreads a weighted gaussian kernel over each sample, bounded by reflection or in
log space. `GaussianMixture` fits a few gaussians by expectation-maximization, for one variable or several. pass
either one as `NormalScore(reference=...)` to replace the empirical CDF of the data.
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
from common import ACCENT, GRAY, HIGHLIGHT, LIGHT, save

data = bt.datasets.vein_gold_grade_control()
intervals = bt.merge_intervals(data["assays"], data["lithology"])
holes = bt.Drillholes(data["collars"], data["surveys"], intervals)
composites = holes.composite(1.0, ["AU_GPT"], domain="LITH", categories=["VEIN"])
v4 = composites.filter((composites["LITH"] == "QV") & (composites["VEIN"] == "V4")).drop_null("AU_GPT")
au = v4["AU_GPT"]
weights = bt.cell_declustering(v4, "AU_GPT", cell_size=20.0).weights
print(
    f"{len(au)} composites of 1 m in vein V4, declustered mean Au {np.average(au, weights=weights):.2f} g/t"
)
print(f"highest {np.sort(au)[-3:].round(1)} g/t")

# %% [markdown]
# ## a smooth tail for one variable
#
# vein V4 of [top cuts](../../03-exploratory-analysis/05-top-cuts/README.md) has about a hundred composites. the
# empirical table stops at the highest one, and above it scores map linearly in probability to the upper tail bound,
# here that same maximum. three kernel densities, all fitted with the declustering weights and kept above zero: one
# reflects its kernels at 0 g/t, and two put them on log Au. silverman's rule sets the width from the spread of the
# weighted data and its effective number of samples; the last density takes a narrower width of 0.3 log units.

# %%
references = {
    "reflected at 0": bt.KernelDensity(lower=0.0).fit(au, weights=weights),
    "log space": bt.KernelDensity(log=True).fit(au, weights=weights),
    "log, width 0.3": bt.KernelDensity(log=True, bandwidth=0.3).fit(au, weights=weights),
}
transforms = {"empirical": bt.NormalScore(tails=(0.0, au.max())).fit(au, weights=weights)}
for name, kde in references.items():
    transforms[name] = bt.NormalScore(reference=kde).fit(au)
    print(
        f"{name:15} bandwidth {kde.bandwidth_:.2f}" + (" g/t" if name == "reflected at 0" else " (log units)")
    )

# %% [markdown]
# back-transforming a large standard-normal sample reads the distribution of each reference: its mean, its upper
# quantiles, and how often it goes past the highest composite.

# %%
z = np.random.default_rng(0).standard_normal(200_000)
print(f"{'':15}{'mean':>7}{'P50':>7}{'P99':>8}{'P99.9':>8}{'> max (%)':>11}")
for name, t in transforms.items():
    x = t.inverse_transform(z)
    q = np.quantile(x, [0.5, 0.99, 0.999])
    print(f"{name:15}{x.mean():7.2f}{q[0]:7.2f}{q[1]:8.1f}{q[2]:8.1f}{100 * np.mean(x > au.max()):11.2f}")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
bins = np.logspace(np.log10(au.min()), np.log10(au.max()), 25)
axes[0].hist(au, bins=bins, weights=weights / weights.sum(), color=LIGHT, label="declustered data")
grid = np.logspace(-2.5, 3, 400)
width = np.log10(bins[1] / bins[0])
styles = {"empirical": (GRAY, "-"), "reflected at 0": (ACCENT, ":"), "log space": (ACCENT, "-")}
styles["log, width 0.3"] = (HIGHLIGHT, "-")
for name, kde in references.items():
    color, ls = styles[name]
    axes[0].plot(grid, kde.pdf(grid) * grid * np.log(10) * width, color=color, ls=ls, label=name)
axes[0].set(xscale="log", xlabel="Au (g/t)", ylabel="Proportion per bin", title="Densities")
axes[0].legend(loc="upper left")
scores = np.linspace(-4, 4, 400)
for name, t in transforms.items():
    color, ls = styles[name]
    axes[1].plot(scores, t.inverse_transform(scores), color=color, ls=ls, label=name)
axes[1].axhline(au.max(), color=GRAY, lw=0.6, ls=":")
axes[1].set(yscale="log", xlabel="Normal score", ylabel="Au (g/t)", title="Back-transform")
axes[1].legend()
save(fig, "tails")

# %% [markdown]
# the empirical table never goes past 215 g/t, and the reflected kernels pass it by only a few g/t (P99.9 220 g/t).
# sized on the body at 4.1 g/t wide, they leave each high composite a spike of its own. kernels on log Au widen with the
# grade and carry the tail well past the highest composite, with P99.9 at 594 g/t for the silverman width. a wide log
# kernel inflates the mean by about exp(h²/2), 1.25 for h = 0.67, which lifts 15.95 g/t to 20.0. the width of 0.3 keeps
# the mean within 5 % (16.7 g/t) and still sets P99.9 at 323 g/t. compare the mean of a reference with the declustered
# mean before you simulate with it.

# %% [markdown]
# ## a mixture for two mineral associations
#
# log chalcocite and log tennantite of porphyry 1 form the L-shaped cloud of
# [multivariate transforms](../../04-transforms/05-multivariate-transforms/README.md) and
# [multivariate simulation](../../08-stochastic-simulation/06-multivariate-simulation/README.md): samples rich in one
# mineral are poor in the other. one gaussian draws an ellipse over the empty corner, and a mixture of a few follows the
# arms. with `components` left out, the fit keeps the count from 1 to 6 with the lowest bayesian information criterion
# (BIC).

# %%
porphyry = bt.datasets.porphyry_geometallurgy(deposit=1)["synthetic_drillholes"]
pair = np.log(np.column_stack([porphyry["calcosina"], porphyry["tenantita"]]))
names = ["log chalcocite (%)", "log tennantite (%)"]
mixture = bt.GaussianMixture(seed=0).fit(pair)
print(f"{len(pair)} composites; BIC by number of components:")
print("  " + ", ".join(f"{k}: {b:.0f}" for k, b in mixture.bic_.items()))
print("proportions", mixture.proportions_.round(2))

# %%
reference = mixture.sample(len(pair), seed=1)
single = bt.GaussianMixture(components=1).fit(pair).sample(len(pair), seed=1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", sharex=True, sharey=True)
for ax, xy, title in zip(
    axes,
    [pair, single, reference],
    ["Data", "One Gaussian", f"Mixture of {len(mixture.proportions_)}"],
    strict=True,
):
    ax.scatter(xy[:, 0], xy[:, 1], s=2, color=ACCENT, alpha=0.3, linewidths=0)
    ax.set(xlabel=names[0], title=title, xlim=(-9.5, 1.5), ylim=(-9.5, 0.5))
t = np.linspace(0, 2 * np.pi, 100)
for mean, cov in zip(mixture.means_, mixture.covariances_, strict=True):
    ellipse = mean + 2 * np.column_stack([np.cos(t), np.sin(t)]) @ np.linalg.cholesky(cov).T
    axes[2].plot(ellipse[:, 0], ellipse[:, 1], color=HIGHLIGHT, lw=0.8)
axes[0].set_ylabel(names[1])
save(fig, "mixture")

# %% [markdown]
# with 6817 samples, BIC keeps falling up to six components, because the stripes at the detection limits reward more.
# narrow components follow the stripes along the arms of the L and wide ones the scattered samples. a sample of the
# mixture leaves the corner nearly empty, where the single gaussian fills it. that sample is a smooth reference for both
# variables at once, as large as you need.
#
# ## imputation with a mixture
#
# `GaussianImputer(components=...)` fits the mixture to the normal scores and draws a missing score from the component
# its row belongs to, given its present scores. hide tennantite in every other hole and impute it back:

# %%
hole_ids = porphyry["DHID"]
hidden = np.isin(hole_ids, np.unique(hole_ids)[::2])
holed = pair.copy()
holed[hidden, 1] = np.nan
q80 = np.quantile(pair, 0.8, axis=0)
imputed = {
    "one Gaussian": bt.GaussianImputer(seed=0).fit(holed).transform(holed),
    "mixture": bt.GaussianImputer(components=None, seed=0).fit(holed).transform(holed),
}
print(f"{hidden.sum()} tennantite values hidden; both above their P80:")
print(f"  truth {np.mean(np.all(pair[hidden] > q80, axis=1)):.3f}")
for name, filled in imputed.items():
    print(f"  {name} {np.mean(np.all(filled[hidden] > q80, axis=1)):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", sharex=True, sharey=True)
titles = ["Hidden values", "Imputed, one Gaussian", "Imputed, mixture"]
for ax, xy, title in zip(axes, [pair, *imputed.values()], titles, strict=True):
    ax.scatter(pair[~hidden, 0], pair[~hidden, 1], s=2, color=LIGHT, linewidths=0)
    ax.scatter(xy[hidden, 0], xy[hidden, 1], s=2, color=HIGHLIGHT, alpha=0.4, linewidths=0)
    ax.set(xlabel=names[0], title=title)
axes[0].set_ylabel(names[1])
save(fig, "imputed")

# %% [markdown]
# of the hidden samples, 3.5 % have both minerals above their P80. one gaussian imputes 5.6 %: it draws tennantite from
# the correlation of the scores alone, and that correlation is weak, so high chalcocite gets typical tennantite. the
# mixture imputes 3.6 %, because it first picks the arm of the L that the chalcocite of the sample points to. the
# one-gaussian imputer of [imputation](../../04-transforms/06-imputation/README.md) stays the default; `components=None`
# lets BIC decide.
