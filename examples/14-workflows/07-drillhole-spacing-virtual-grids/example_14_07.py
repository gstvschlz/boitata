"""
# Drill-hole spacing from virtual grids

A mine plans quarterly production from 40 × 40 m areas of Walker Lake and needs each quarter's grade within ±15 %
at 90 % confidence. Which drill spacing delivers that? Wilde (2010) answers it by drilling simulated deposits:
take realizations conditioned to today's data as the truth, sample them on regular grids, simulate again from those
samples alone, and measure the uncertainty of each production volume. You run his virtual grids at 5 to 20 m, read
the spacing each criterion needs, and check the result against a reference study of the same data.
"""

# %% [hidden]
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))
warnings.filterwarnings("ignore", "Samples sharing")

# %%
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, save

# %% [markdown]
# !!! learn "What you'll learn"
#     - How Wilde's virtual grids turn a simulation model into an uncertainty-versus-spacing curve.
#     - What the maximum expected error (MEE) of a production volume measures, and how to read the share of volumes
#       that meet ±15 % at 90 %.
#     - Why the required spacing depends on the volume: a quarter needs much denser drilling than a year.
#
#     Prerequisites: [simulation at block support](../../08-stochastic-simulation/03-simulation-at-block-support/README.md)
#     for `window=` and `SimulationSummary.relative_error`.
#
# ## The data
#
# The 470 samples of V (ppm), declustered with cell weights, give the normal-score variogram: a nugget of 0.37 and
# one spherical structure of range 103 m along azimuth 170°, a third of that across. Turning bands with 500 bands
# simulate on 2 m nodes. The production volumes are moving windows. For a quarter, `blocks=` averages the nodes to
# 8 m panels and `window=(40, 40)` gives each panel the mean of the 5 × 5 panels around it, a 40 × 40 m window;
# for a year, 16 m panels and `window=(80, 80)`. A window spans an odd number of panels, so the panel side is a
# fifth of the window. The study counts only windows that lie wholly inside the area.

# %%
samples = bt.datasets.walker_lake()
xy, v = samples.coords, samples["V"]
weights = bt.cell_declustering(samples, "V", sizes=np.arange(2.5, 102.5, 2.5)).weights
scores = bt.NormalScore().fit(v, weights=weights).transform(v)
azimuths = (170, 260)
directional = [bt.experimental_variogram(xy, scores, 10, 120, azimuth=a) for a in azimuths]
fitted = bt.Variogram.fit_directional(directional, [(a, 0) for a in azimuths], rotation=[170, 0, 0])
gaussian = bt.Variogram(
    [(s.model, s.sill / fitted.sill, s.range) for s in fitted.structures],
    nugget=fitted.nugget / fitted.sill,
    rotation=fitted.rotation,
    ratios=fitted.ratios,
)
print(gaussian)

nodes = bt.BlockModel(origin=(1, 1), size=(2, 2), count=(130, 150))
panels = {
    side: bt.BlockModel(origin=(1, 1), size=(side / 5,) * 2, count=(1280 // side, 1480 // side))
    for side in (40, 80)
}
tb = bt.TurningBands(gaussian, bands=500, search=bt.Search(radius=100, max_samples=24)).fit(
    xy, v, weights=weights
)


def inside(side):
    """Panels whose window of `side` meters lies inside the panels."""
    c = panels[side].centroids[:, :2] - 1
    extent = np.array([1280 // side, 1480 // side]) * side / 5
    return ((c >= side / 2) & (c <= extent - side / 2)).all(axis=1)


print(
    f"{len(nodes.centroids)} nodes, {inside(40).sum()} quarterly and {inside(80).sum()} yearly windows inside"
)

# %% [markdown]
# !!! step "Step 1: Pick the truths"
#     Each truth is a realization conditioned to the 470 samples, so it honors what is known today and varies
#     where nothing is known. Twenty candidates are simulated and averaged to the quarterly windows;
#     `select_realizations` keeps three that span them, the medoids of a k-medoids clustering of their window
#     grades. A truth index is the realization's number in `simulate(seed=101)`, which `spacing_study` simulates
#     again.

# %%
candidates = tb.simulate(nodes, n=20, seed=101, blocks=panels[40], window=(40, 40), keep=True)
truths = bt.select_realizations(candidates, 3, seed=0)
means = candidates.realizations[:, inside(40)].mean(axis=1)
print(
    f"candidate means {means.min():.0f} to {means.max():.0f} ppm; truths {truths.tolist()}: "
    + ", ".join(f"{means[k]:.0f}" for k in truths)
    + " ppm"
)

# %% [markdown]
# !!! step "Step 2: Drill each truth on virtual grids"
#     `spacing_study` lays vertical holes on a square grid over the nodes for each spacing, reads each truth at
#     the holes, refits a copy of the simulator on those samples alone and simulates 50 realizations of the
#     windows. The model is 2D, so `composite_length=np.inf` gives one sample per hole. Each table row is one
#     window of one truth at one spacing.

# %%
spacings = [5, 7.5, 10, 12.5, 15, 20]
study = {
    side: bt.spacing_study(
        tb,
        nodes,
        spacings=spacings,
        truths=truths,
        seed=101,
        n=50,
        blocks=panels[side],
        window=(side, side),
        composite_length=np.inf,
    )
    for side in (40, 80)
}
for side, table in study.items():
    table = table.filter(inside(side)[np.asarray(table["row"], dtype=int)])
    study[side] = table
    print(f"{side} m windows: {table.num_rows} rows, columns {', '.join(table.column_names)}")

# %% [markdown]
# ??? math "The math"
#     For a window with realizations \(z^{(1)}, \dots, z^{(n)}\), mean \(\bar z\) and 5th and 95th percentiles
#     \(q_{05}, q_{95}\), the maximum expected error (Koppe et al., 2017) is
#     \[ \mathrm{MEE} = \frac{q_{95} - q_{05}}{2\,\bar z}. \]
#     A window with MEE ≤ 0.15 has 90 % of its realizations within about ±15 % of the mean.
#
# !!! step "Step 3: MEE per spacing"
#     The box plot shows the spread of MEE over the quarterly windows of the three truths; the right panel counts
#     the windows that meet ±15 % at 90 % for both volumes.

# %%
spacing_of = {side: table["spacing"] for side, table in study.items()}
share = {
    side: [np.mean(t["mee"][spacing_of[side] == s] <= 0.15) for s in spacings] for side, t in study.items()
}
print("spacing  median MEE 40 m  share ≤ 15 %: 40 m   80 m")
for i, s in enumerate(spacings):
    median = np.median(study[40]["mee"][spacing_of[40] == s])
    print(f"{s:7g}  {median:13.1%}  {share[40][i]:17.1%}  {share[80][i]:5.1%}")

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", width_ratios=[1.2, 1])
boxes = a.boxplot(
    [100 * study[40]["mee"][spacing_of[40] == s] for s in spacings],
    positions=spacings,
    widths=1.4,
    whis=(10, 90),
    showfliers=False,
    patch_artist=True,
    medianprops={"color": INK, "lw": 1.5},
)
for box in boxes["boxes"]:
    box.set(facecolor=LIGHT, edgecolor=GRAY)
a.axhline(15, color=HIGHLIGHT, ls="--", lw=1.2)
a.set(
    xlabel="Virtual grid spacing (m)",
    ylabel="MEE of the window (%)",
    xticks=spacings,
    xticklabels=[f"{s:g}" for s in spacings],
    title="40 × 40 m windows: MEE (P10–P90)",
)
for side, color in ((40, ACCENT), (80, HIGHLIGHT)):
    b.plot(spacings, 100 * np.array(share[side]), "o-", color=color, label=f"{side} × {side} m")
b.axhline(90, color=GRAY, ls="--", lw=1)
b.set(
    xlabel="Virtual grid spacing (m)",
    ylabel="Windows with MEE ≤ 15 % (%)",
    ylim=(0, 102),
    title="Share meeting ±15 % at 90 %",
)
b.legend()
save(fig, "virtual_grids")

# %% [markdown]
# The median quarterly MEE grows from 10.2 % at 5 m to 32.1 % at 20 m. At 5 m, 89.1 % of the quarterly windows
# meet the criterion; at 7.5 m, 53.1 %; at 10 m, 22.4 %. A yearly window covers four times the area, and 94.2 %
# of them still meet it at 10 m.
#
# !!! step "Step 4: Required spacing"
#     `uncertainty_curve` bins the windows by spacing and gives the P50 and P90 of MEE in each bin;
#     `required_spacing` interpolates where a quantile crosses 15 %. The P90 crossing is the spacing at which 90 %
#     of the windows meet ±15 %; the P50 crossing, at which the median window does.

# %%
edges = [4, 6, 8.5, 11, 13.5, 17, 22]
curves = {side: bt.uncertainty_curve("spacing", "mee", bins=edges, data=t) for side, t in study.items()}
required = {}
for side, curve in curves.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        required[side] = {q: bt.required_spacing(curve, column=q) for q in ("P50", "P90")}
    print(
        f"{side} m windows: P90 MEE at 5 m {curve['P90'][0]:.1%}; required spacing for the median window "
        f"{required[side]['P50']:.1f} m, for 90 % of windows {required[side]['P90']:.1f} m"
    )

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
for ax, (side, curve) in zip(axes, curves.items(), strict=True):
    bt.plot.uncertainty_curve(curve, required=required[side]["P90"], ax=ax)
    ax.set(title=f"{side} × {side} m windows", xlabel="Virtual grid spacing (m)", ylabel="MEE")
save(fig, "required_spacing")

# %% [markdown]
# The yearly volume needs holes every 11.4 m for 90 % of its windows, and its median window meets ±15 % up to
# 14.8 m. The quarterly volume misses the 90 % mark on every grid tried: its P90 MEE is 15.3 % at 5 m, so
# `required_spacing` returns NaN. Its median window crosses 15 % at 7.7 m.
#
# !!! key "Key idea"
#     The required spacing belongs to a volume and a confidence, not to the deposit. Averaging over a larger
#     volume cancels more of the local error, so a year tolerates holes twice as far apart as a quarter.
#
# !!! check "Check before you move on"
#     First, `covered` says whether the truth of a window lies within its MEE of the simulated mean; a calibrated
#     model covers about 90 % of the windows. Second, a reference study of the same data (Wilde's method on 2 m
#     nodes, three truths, 50 realizations) found 83.6 % of the quarterly windows within ±15 % at 5 m and 24.2 % at
#     10 m. Its holes sat half a spacing in from the edge, between the nodes. `planned_drillholes` with that
#     `offset` builds the same grids, and `spacing_study(plans=)` drills them on ten truths.

# %%
for side, table in study.items():
    print(f"{side} m windows covered: {np.mean(table['covered']):.1%}")
plans = {f"{s:g}": bt.planned_drillholes(nodes, s, offset=(s / 2 - 2, s / 2 - 2)) for s in (5, 10)}
reference = bt.spacing_study(
    tb,
    nodes,
    plans=plans,
    truths=10,
    seed=101,
    n=50,
    blocks=panels[40],
    window=(40, 40),
    composite_length=np.inf,
)
reference = reference.filter(inside(40)[np.asarray(reference["row"], dtype=int)])
plan = np.asarray(reference["plan"])
for name in plans:
    meets = reference["mee"][plan == name] <= 0.15
    truth_of = reference["realization"][plan == name]
    per_truth = [np.mean(meets[truth_of == k]) for k in range(10)]
    print(
        f"holes between nodes, {name} m: {meets.mean():.1%} of quarterly windows; per truth "
        f"{min(per_truth):.1%} to {max(per_truth):.1%}, standard deviation {np.std(per_truth, ddof=1):.1%}"
    )

# %% [markdown]
# The MEE covers the truth in 86.8 % of the quarterly windows and 86.0 % of the yearly ones, a little short of
# 90 %. With the holes between nodes, 84.5 % of the quarterly windows meet ±15 % at 5 m and 21.8 % at 10 m,
# against the reference's 83.6 % and 24.2 %. The share changes from truth to truth: at 10 m one truth scores
# 13.3 %, another 26.0 %, a standard deviation of 3.7 points. The reference averaged three truths, so its share
# carries a standard error of about 2 points, and gaps of 0.9 and 2.4 points lie within Monte-Carlo noise.
# The grids of Step 2 put holes on node centres, and they score 89.1 % at 5 m:
# a node that holds a sample keeps its value in every realization, which narrows its window.
#
# ## The decision
#
# Drill on 11 m or closer for yearly planning: at 10 m, 94.2 % of the 80 × 80 m windows meet ±15 % at 90 %.
# A quarter needs holes closer than 5 m, which no resource drilling pays for; plan quarters from grade control.
# [The learning curve](../../14-workflows/08-drillhole-spacing-learning-curve/README.md) reads the same
# relation from one simulation of the current data.
