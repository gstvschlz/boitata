"""
# Where to drill ten infill holes

The quarterly windows of Walker Lake fall into classes by their maximum expected error (MEE): measured when a
40 × 40 m window meets ±15 % at 90 % confidence, indicated when only the 80 × 80 m window around it does. Ten
infill holes should move as many indicated windows as possible to measured. You state that goal as an objective
over candidate collars, let four searches and one of your own choose the holes, and drill the plans on simulated
truths against random plans of ten holes.
"""

# %% [hidden]
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))
warnings.filterwarnings("ignore", "Samples sharing")

# %%
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, map_axes, save
from matplotlib.colors import ListedColormap

# %% [markdown]
# !!! learn "What you'll learn"
#     - How `DrillholePlan` turns a drilling goal into an objective over candidate holes, here a custom one built
#       from a learning curve.
#     - How the built-in searches `Greedy`, `Swap`, `ModifiedRandomSearch` and `Annealing` compare, and how to
#       plug in a search of your own.
#     - How to check a plan by virtual drilling with `spacing_study(plans=...)` against random plans.
#
#     Prerequisites: [virtual grids](../../14-workflows/07-drillhole-spacing-virtual-grids/README.md) for the
#     model, the windows and the MEE; [the learning curve](../../14-workflows/08-drillhole-spacing-learning-curve/README.md).
#
# ## The data
#
# The model repeats the virtual-grid study: the normal-score variogram of the declustered V, turning bands with
# 500 bands on 2 m nodes, 8 m panels carrying the 40 × 40 m window and 16 m panels carrying the 80 × 80 m one.

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


# %% [markdown]
# !!! step "Step 1: Classify the windows today"
#     One simulation of 100 realizations gives the MEE of every window. An 8 m panel is ore when the mean grade
#     of its 40 × 40 m window reaches 300 ppm; an ore panel is measured when its quarterly MEE is at most 15 %,
#     and indicated when the yearly window of the 16 m panel holding it meets 15 % instead. The indicated panels
#     are the targets.

# %%
today = {
    side: tb.simulate(
        nodes,
        n=100,
        seed=11,
        blocks=panels[side],
        window=(side, side),
        quantiles=[0.05, 0.95],
        progress=False,
    )
    for side in (40, 80)
}
quarter = today[40].relative_error()
centroids = panels[40].centroids
column, row = ((centroids[:, :2] - 1) // 16).astype(int).T
has_year = row < 1480 // 80
year_index = np.where(has_year, row * (1280 // 80) + column, 0)
year = np.where(has_year & inside(80)[year_index], today[80].relative_error()[year_index], np.inf)
ore = inside(40) & (today[40].mean >= 300)
classes = np.where(~ore, 0, np.where(quarter <= 0.15, 3, np.where(year <= 0.15, 2, 1)))
target = np.flatnonzero(classes == 2)
print("panels: waste {}, inferred {}, indicated {}, measured {}".format(*np.bincount(classes, minlength=4)))

# %% [markdown]
# Of the 1,184 panels, 474 hold ore: 226 inferred, 101 indicated and 147 measured.
#
# !!! step "Step 2: An objective from the learning curve"
#     A new hole lowers the data spacing around it, and the learning curve says how much MEE that buys. The data
#     spacing here is `data_spacing` with a 30 m search: with samples in plan and `composite_length` set to
#     4/3 of the radius, \(\sqrt{V / (c\, n)}\) becomes \(\sqrt{\pi r^2 / n}\), the side of the square each of
#     the \(n\) samples within 30 m stands for. `uncertainty_curve` gives the median MEE of the ore panels per
#     spacing bin.

# %%
radius = 30.0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    spacing = bt.data_spacing(centroids, xy, bt.Search(radius=radius), composite_length=4 * radius / 3)
known = ore & np.isfinite(spacing)
curve = bt.uncertainty_curve(spacing[known], quarter[known], bins=np.arange(4, 40, 3))
for s, p50, n in zip(curve["spacing"], curve["P50"], curve["n"], strict=True):
    print(f"spacing {s:5.1f} m: median MEE {p50:.1%} over {n:.0f} panels")

# %% [markdown]
# ??? math "The math"
#     Panel \(b\) has MEE \(e_b\) and spacing \(s_b\) today. With new holes its spacing drops to \(s'_b\), and
#     the learning curve \(L\) rescales its MEE to \(e'_b = e_b\, L(s'_b) / L(s_b)\). The panel scores its
#     progress towards 15 %,
#     \[ g_b = \mathrm{clip}\!\left(\frac{e_b - e'_b}{e_b - 0.15},\ 0,\ 1\right), \]
#     1 once converted and 0.5 halfway, and the objective sums \(g_b\) over the targets.
#
#     `DrillholePlan` calls the objective with a Table of metrics: ``block`` indexes the targets and
#     ``data_spacing`` holds \(s'_b\). It returns one score per row.

# %%
spacing_bins, median_mee = curve["spacing"], curve["P50"]
mee_today, spacing_today = quarter[target], spacing[target]


def learned(s):
    """Median MEE at spacing `s`; no sample within 30 m reads as the widest bin."""
    return np.interp(np.where(np.isnan(s), np.inf, s), spacing_bins, median_mee)


def predicted(metrics):
    b = np.asarray(metrics["block"], dtype=int)
    return mee_today[b] * learned(metrics["data_spacing"]) / learned(spacing_today[b])


def progress(metrics):
    b = np.asarray(metrics["block"], dtype=int)
    return np.clip((mee_today[b] - predicted(metrics)) / (mee_today[b] - 0.15), 0, 1)


# %% [markdown]
# !!! step "Step 3: Candidates and the problem"
#     `planned_drillholes` lays one vertical candidate every 5 m over the area, 3,120 collars. `exclude=` drops
#     those within 3.5 m of a sample, with a small circle around each one. `DrillholePlan` takes the
#     candidates, the targets, the existing samples and the objective, and allows ten holes.

# %%
cells = bt.BlockModel(origin=(0, 0), size=(5, 5), count=(52, 60))
candidates = bt.planned_drillholes(cells, 5.0)
angle = np.linspace(0, 2 * np.pi, 16, endpoint=False)
circles = [p[:2] + 3.5 * np.column_stack([np.cos(angle), np.sin(angle)]) for p in xy]
plan = bt.DrillholePlan(
    candidates,
    bt.OrdinaryKriging(gaussian, bt.Search(radius=radius, max_samples=8)),
    centroids[target],
    data=xy,
    objective=progress,
    n_holes=10,
    composite_length=4 * radius / 3,
    exclude=circles,
)
print(f"{len(plan.holes)} candidates, {plan.feasible([]).sum()} outside the circles")

# %% [markdown]
# !!! step "Step 4: Run the searches"
#     `optimize` hands the problem to a search and ranks the holes it returns. The four built-ins:
#
#     - `Greedy` adds the hole of largest gain, ten times.
#     - `Swap` starts from the greedy plan and makes the best one-for-one exchange until none helps.
#     - `ModifiedRandomSearch` moves one hole per iteration, 4,000 times, and keeps a move that raises the
#       objective. It moves weak holes more often: hole \(i\) moves with probability proportional to
#       \(\max C + \min C - C_i\), \(C_i\) the objective lost without it. A move goes within 25 m, or anywhere
#       one time in five.
#     - `Annealing` makes the same moves and also keeps a move that loses \(d\) with probability
#       \(e^{-d/T}\), the temperature \(T\) falling from 0.3 to 0.01; it returns the best plan it visits.
#
#     A search of your own needs a `search(problem, n)` method and nothing else. `Ranking` takes the ten holes
#     that gain most on their own, the "top ten" a spreadsheet would give.


# %%
class Ranking:
    def search(self, problem, n):
        gains = problem.gains([])
        return [int(i) for i in np.argsort(-gains, kind="stable")[:n]]


searches = {
    "ranking": Ranking(),
    "greedy": bt.Greedy(),
    "swap": bt.Swap(),
    "random search": bt.ModifiedRandomSearch(),
    "annealing": bt.Annealing(),
}
tables, chosen = {}, {}
print("search          objective  predicted conversions  seconds")
for name, search in searches.items():
    start = time.perf_counter()
    tables[name] = plan.optimize(search=search)
    seconds = time.perf_counter() - start
    chosen[name] = [plan.holes.index(h) for h in tables[name]["HOLE_ID"]]
    converted = int((predicted(plan.metrics(chosen[name])) <= 0.15).sum())
    print(f"{name:14s}  {plan.score(chosen[name]):9.2f}  {converted:21d}  {seconds:7.2f}")

# %% [markdown]
# The greedy plan scores 40.85 and converts 17 panels by the learning curve; annealing reaches 41.05, so greedy
# gets 99.5 % of it. Swap and the random search stop at 40.96 from the greedy start. The ranking scores 22.11:
# its ten holes crowd one gap, and each one adds little once its neighbors are drilled.
#
# !!! step "Step 5: Seeds"
#     The random searches depend on their seed. Five seeds of each show how far the result moves.

# %%
for name, kind in (("random search", bt.ModifiedRandomSearch), ("annealing", bt.Annealing)):
    values = [plan.optimize(search=kind(seed=seed))["CUMULATIVE"][-1] for seed in range(5)]
    print(f"{name}: objective {min(values):.2f} to {max(values):.2f} over seeds 0 to 4")

# %% [markdown]
# Every seed of the random search ends at the swap plan, 40.96. Annealing ends between 40.97 and 41.07: worse
# moves let it leave that plan, and the best of five seeds gains 0.5 % over greedy.
#
# !!! step "Step 6: The plan and its drilling order"
#     Each table lists the holes in drilling order: the hole of largest gain first, then the largest gain given
#     the holes above it. ``GAIN`` is that marginal gain, ``CUMULATIVE`` the objective so far and
#     ``CONTRIBUTION`` the objective lost if the hole leaves the final plan. The order holds whatever search
#     chose the plan, so you can stop drilling early and keep the best prefix.

# %%
best = tables["annealing"]
print("order  hole    east  north   gain  cumulative  contribution")
for r in range(best.num_rows):
    print(
        f"{best['ORDER'][r]:5.0f}  {best['HOLE_ID'][r]}  {best['X'][r]:5.1f}  {best['Y'][r]:5.1f}"
        f"  {best['GAIN'][r]:5.2f}  {best['CUMULATIVE'][r]:10.2f}  {best['CONTRIBUTION'][r]:12.2f}"
    )

colors = ListedColormap([LIGHT, "#e3b5a0", "#f1d27a", "#9cc59a"])
image = np.full(len(centroids), np.nan)
image[ore] = classes[ore]
image[~ore] = 0
fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), layout="constrained")
for ax, name in zip(axes, ("ranking", "annealing"), strict=True):
    ax.imshow(
        image.reshape(37, 32), origin="lower", extent=(1, 257, 1, 297), cmap=colors, vmin=-0.5, vmax=3.5
    )
    ax.scatter(*xy[:, :2].T, s=2, color=INK, linewidths=0)
    holes = tables[name]
    ax.scatter(holes["X"], holes["Y"], s=60, color=HIGHLIGHT, edgecolors="white", linewidths=1, zorder=3)
    for x, y, k in zip(holes["X"], holes["Y"], holes["ORDER"], strict=True):
        ax.annotate(
            f"{k:.0f}", (x, y), (5, 4), textcoords="offset points", fontsize=8, color=INK, weight="bold"
        )
    map_axes(ax, f"{name}: objective {holes['CUMULATIVE'][-1]:.1f}")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors(k)) for k in range(4)]
fig.legend(handles, ["waste", "inferred", "indicated", "measured"], loc="outside lower center", ncol=4)
save(fig, "plans")

# %% [markdown]
# The annealing plan spreads its holes over five indicated areas, two or three per area. The first hole gains
# 7.76; the last, 2.51. Hole 2 gains 5.54 when it goes in second but loses 4.14 when it leaves the full plan:
# holes 1 and 6 lie within 30 m of it and share its panels. The ranking puts all ten holes into one gap in the south-west, 5 m apart.
#
# !!! step "Step 7: Drill the plans on simulated truths"
#     The objective is a forecast. `spacing_study(plans=...)` checks it: it draws ten truths conditioned to
#     today's data, reads each truth at a plan's holes, refits the simulator on the existing samples plus those
#     holes (`existing=True`), simulates 100 realizations and returns the quarterly MEE of every panel. A target
#     converts when its MEE drops to 15 % or less. Twenty random plans of ten feasible holes give the reference.
#     `optimize` with a search that returns a fixed list builds their tables.


# %%
def drillholes(table):
    """Drillholes from the collars, directions and lengths of an `optimize` table."""
    ids = np.asarray(table["HOLE_ID"], dtype=object)
    zeros = np.zeros(len(ids))
    return bt.Drillholes(
        {"HOLE_ID": ids, "X": table["X"], "Y": table["Y"], "Z": table["Z"]},
        {"HOLE_ID": ids, "DEPTH": zeros, "AZIMUTH": table["AZIMUTH"], "DIP": table["DIP"]},
        {"HOLE_ID": ids, "FROM": zeros, "TO": table["LENGTH"]},
    )


class Fixed:
    def __init__(self, holes):
        self.holes = holes

    def search(self, problem, n):
        return self.holes


rng = np.random.default_rng(99)
feasible = np.flatnonzero(plan.feasible([]))
plans = {name: drillholes(table) for name, table in tables.items()}
for k in range(20):
    holes = rng.choice(feasible, 10, replace=False).tolist()
    plans[f"random {k}"] = drillholes(plan.optimize(search=Fixed(holes)))
study = bt.spacing_study(
    tb,
    nodes,
    plans=plans,
    truths=10,
    seed=101,
    n=100,
    blocks=panels[40],
    window=(40, 40),
    existing=True,
    composite_length=np.inf,
    progress=False,
)
on_target = np.isin(np.asarray(study["row"], dtype=int), target)
names, truth_of = np.asarray(study["plan"]), study["realization"]
converted = {
    name: np.array(
        [np.sum(study["mee"][on_target & (names == name) & (truth_of == k)] <= 0.15) for k in range(10)]
    )
    for name in plans
}
random = np.array([converted[f"random {k}"].mean() for k in range(20)])
print(f"random plans: {np.median(random):.1f} conversions (median), {random.min():.1f} to {random.max():.1f}")
for name in tables:
    counts = converted[name]
    print(f"{name:14s} {counts.mean():5.1f} conversions, {counts.min()} to {counts.max()} over the truths")

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
ax.boxplot(
    [np.concatenate([converted[f"random {k}"] for k in range(20)])] + [converted[name] for name in tables],
    tick_labels=["20 random\nplans", *tables],
    widths=0.5,
    showfliers=False,
    patch_artist=True,
    boxprops={"facecolor": LIGHT, "edgecolor": GRAY},
    medianprops={"color": INK, "lw": 1.5},
)
for i, name in enumerate(tables, start=2):
    ax.scatter(np.full(10, i), converted[name], s=10, color=ACCENT, zorder=3)
ax.set(ylabel="Indicated panels converted", title="Virtual drilling over ten truths")
save(fig, "validation")

# %% [markdown]
# A random plan converts 8.1 of the 101 targets in the median, and the best of twenty 9.9. The four searches
# convert 10.1 to 11.4, all above the best random plan, and differ by less than the spread over truths: annealing
# converts 5 on one truth and 16 on another. Greedy, at 99.5 % of the annealing objective, converts 11.4 and
# annealing 10.8.
#
# The ranking converts 14.7, the most of all, although its objective is half the others'. Its ten holes sit 5 m
# apart, denser than any part of the deposit today. The learning curve stops at 8.8 m, the densest bin, and the
# objective reads any spacing below it as that bin's MEE of 12.5 %, so it pays nothing for the extra holes of a
# cluster. Virtual drilling pays for them: ten holes in one gap bring several 40 × 40 m windows close to the
# densest data there is.
#
# !!! key "Key idea"
#     The searches agree within half a percent on this objective, and virtual drilling cannot tell their plans
#     apart. The objective decides where the holes go. A learning curve only knows the spacings the deposit
#     already has, so check a plan by drilling simulated truths before you trust its forecast.
#
# !!! check "Check before you move on"
#     `Swap` stops at a local optimum: no single exchange improves it. On small problems it often matches the
#     best plan over every subset; the tests check it on 49 candidates and three holes. With a budget instead
#     of a hole count, give `DrillholePlan` a `budget` and `cost_per_meter`, and `Greedy` ranks holes by gain per
#     unit cost.
#
# ## The decision
#
# Take the greedy plan: it runs in a fraction of a second, reaches 99.5 % of the annealing objective and converts
# 11.4 targets on the truths, against 8.1 for a random plan. Before drilling, revise the objective: the cluster
# of the ranking converts more, so a learning curve that reaches below 8.8 m, from virtual grids as on
# [the virtual-grid page](../../14-workflows/07-drillhole-spacing-virtual-grids/README.md), would reward a cluster
# as virtual drilling does.
