"""
# From drill holes to a 3D model

Three dipping zinc lenses, 289 drill holes and one notebook. You load the holes, composite them inside each lens,
build a sub-blocked model and a sparse grid of mining blocks below the ground, estimate and simulate zinc, check both
against the data, and look at the result in 3D. Each step takes a cell or two and links to the chapter or example
that covers it in depth.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1] / "examples"))

import pyvista as pv

pv.OFF_SCREEN = True

# %%
import time

import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, map_axes, save

# %% [markdown]
# ## Drill holes and topography
#
# The stacked sulphide lenses ship with collar, survey and assay tables, a wireframe per lens, and nothing else.
# `Drillholes` desurveys the holes. The collars sample the ground, so `topography` triangulates them into a surface
# and flags any collar that sits off the surface through its neighbors
# ([topography](../../examples/02-data-and-geometry/19-topography/example_02_19.md)).

# %%
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes(data["collars"], data["surveys"], data["assays"])
collars = bt.PointSet.from_table(data["collars"], z="Z")
ground = bt.topography(collars, cell=20.0)
print(f"{len(holes)} holes, {len(data['assays']):,} assays, {ground.flagged.sum()} collars flagged")

fig, ax = plt.subplots(figsize=(6, 5.5), layout="constrained")
bt.plot.section(ground.grid, "z", axis="z", ax=ax, cmap="terrain", colorbar=False)
fig.colorbar(ax.collections[0], ax=ax, shrink=0.7, label="Elevation (m)")
ax.scatter(*collars.coords[:, :2].T, s=3, color=INK)
map_axes(ax, "Topography from the collars")
save(fig, "topography")

# %% [markdown]
# ## Composites inside each lens
#
# Each assay takes the name of the lens around its midpoint, or `host`. Compositing to 2 m gives every sample the
# same support, and `domain="LENS"` stops a composite at a lens contact, so no composite mixes ore with host rock
# ([compositing](../../examples/02-data-and-geometry/03-compositing/example_02_03.md)).

# %%
lenses = {name: data[name] for name in ("lens_1", "lens_2", "lens_3")}
samples = holes.samples()
lens = np.full(len(samples), "host", dtype=object)
for name, mesh in lenses.items():
    lens[mesh.contains(samples.coords)] = name
holes = bt.Drillholes(data["collars"], data["surveys"], samples.with_column("LENS", list(lens)).attributes)
composites = holes.composite(2.0, ["ZN_PCT", "DENSITY"], domain="LENS", residual="merge")
names = np.array(composites["LENS"], dtype=object)
ore = composites.filter(names != "host")
ore_names = names[names != "host"]
print(f"{len(composites):,} composites of 2 m, {len(ore):,} inside a lens")

# %% [markdown]
# ## A sub-blocked model
#
# The lenses dip 60° to the east-southeast, so the model's parent blocks of 10 m are rotated with them. `from_meshes`
# splits each parent that a wireframe cuts into 2.5 m sub-blocks and keeps only blocks inside a lens. The model's
# volume matches the wireframes'
# ([sub-blocks](../../examples/02-data-and-geometry/06-sub-blocks/example_02_06.md)).

# %%
rotation = (22.5, 0.0, 55.0)
frame = bt.BlockModel.from_extents(*lenses.values(), size=(10, 10, 10), buffer=10, rotation=rotation)
blocks = bt.BlockModel.from_meshes(
    frame.origin,
    (10, 10, 10),
    frame.count,
    [(mesh, "inside", name) for name, mesh in lenses.items()],
    4,
    rotation=rotation,
    column="LENS",
)
block_lens = np.array(blocks["LENS"], dtype=object)
for name, mesh in lenses.items():
    inside = block_lens == name
    print(
        f"{name}: {inside.sum():,} blocks, {blocks.volumes[inside].sum() / 1e6:.2f} Mm³ "
        f"for a {mesh.volume / 1e6:.2f} Mm³ wireframe"
    )

# %% [markdown]
# ## Exploratory analysis
#
# The holes cluster where the lenses are rich, so the plain mean overstates zinc. Cell declustering weights each
# lens's composites on their own ([declustering](../02-describing-data/learn_02.md)).

# %%
weights = np.zeros(len(ore))
for name in lenses:
    inside = ore_names == name
    declustering = bt.cell_declustering(
        ore.coords[inside], ore["ZN_PCT"][inside], sizes=np.arange(10, 105, 5)
    )
    weights[inside] = declustering.weights / declustering.weights.mean()
    print(
        f"{name}: {inside.sum()} composites, mean {ore['ZN_PCT'][inside].mean():.2f} % Zn, "
        f"declustered {declustering.mean:.2f} %"
    )
ore = ore.with_column("weight", weights)

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
bt.plot.boxplot("ZN_PCT", list(ore_names), weights="weight", data=ore, ax=a)
a.set(ylabel="Zn (%)", title="Declustered Zn by lens")
bt.plot.probability("ZN_PCT", weights="weight", data=ore, log=True, ax=b, color=ACCENT, ms=3)
b.set(xlabel="Zn (%)", title="Probability plot, three lenses")
save(fig, "statistics")

# %% [markdown]
# Declustering lowers each lens mean by 0.8 to 1.0 % Zn, to about 5.2 %. The three lenses share one distribution,
# and the probability plot shows no break in the upper tail that would call for capping.
#
# ## Variography
#
# Lens 1 holds half the composites, so its variogram stands for all three. Kriging uses the variogram of the
# grades; turning bands simulates normal scores and needs theirs, rescaled to a unit sill
# ([spatial continuity](../03-spatial-continuity/learn_03.md)).

# %%
one = ore.filter(ore_names == "lens_1")
grades = bt.experimental_variogram(one, "ZN_PCT", 10.0, 150.0)
variogram = grades.fit("spherical")

scores = np.zeros(len(ore))
for name in lenses:
    inside = ore_names == name
    scores[inside] = bt.NormalScore().fit_transform(ore["ZN_PCT"][inside], weights=weights[inside])
fitted = bt.experimental_variogram(one, scores[ore_names == "lens_1"], 10.0, 150.0).fit("spherical")
gaussian = bt.Variogram(
    [(s.model, s.sill / fitted.sill, s.range) for s in fitted.structures], nugget=fitted.nugget / fitted.sill
)
for label, model in (("Zn", variogram), ("normal scores", gaussian)):
    s = model.structures[0]
    print(f"{label}: nugget {model.nugget:.2f}, spherical sill {s.sill:.2f}, range {s.range:.0f} m")

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
bt.plot.variogram(grades, variogram=variogram, ax=a, color=ACCENT)
a.set(xlabel="Lag distance (m)", ylabel="γ(h), Zn (%²)", title="Zn, lens 1")
for name, color in zip(lenses, (ACCENT, GRAY, HIGHLIGHT), strict=True):
    inside = ore_names == name
    experimental = bt.experimental_variogram(ore.coords[inside], scores[inside], 10.0, 150.0)
    b.plot(experimental.lags, experimental.gammas, "o", color=color, ms=4, label=name)
h = np.linspace(0, 150, 200)
b.plot(h, gaussian.gamma(h), color=INK, lw=1.4, label="model")
b.legend(loc="lower right")
b.set(xlabel="Lag distance (m)", ylabel="γ(h), normal scores", title="Normal scores, three lenses")
save(fig, "variogram")

# %% [markdown]
# Half the variance is nugget, and zinc loses its correlation beyond about 40 m. The normal scores of lenses 2 and
# 3 scatter around the same model.
#
# ## Domaining
#
# Kriging each lens from its own composites assumes the grade drops sharply at the lens contact. `contact` checks
# that assumption: it averages zinc by distance to the contact, down every hole that crosses one
# ([contact analysis](../../examples/03-exploratory-analysis/07-contacts/example_03_07.md)).

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained", sharey=True)
for ax, name in zip(axes, lenses, strict=True):
    profile = bt.contact(
        composites,
        "ZN_PCT",
        domain_column="LENS",
        holes="HOLE_ID",
        inside=name,
        outside="host",
        max_distance=20.0,
        bin=2.0,
    )
    bt.plot.contact(profile, labels=("lens", "host"), ax=ax, color=ACCENT)
    ax.set(title=name, xlabel="Distance to contact (m)", ylabel="")
axes[0].set_ylabel("Zn (%)")
save(fig, "contacts")
host = composites["ZN_PCT"][names == "host"]
print(f"host rock: {np.nanmean(host):.2f} % Zn over {np.isfinite(host).sum():,} composites")

# %% [markdown]
# Zinc reaches 8 to 10 % inside each lens and drops toward the contact; across it, the host averages 0.07 %. The
# contacts are hard, so host composites must not inform the lenses, and each lens gets its own estimate.
#
# ## Kriging
#
# Ordinary kriging runs in three search passes. The first wants six composites within the variogram range, at most
# two per hole; blocks it leaves go to a pass at twice the range, and the last pass reaches every block.
# `domain_column="LENS"` keeps each lens to its own composites. Density, measured on fewer samples, is kriged the
# same way, so tonnes follow the rock ([kriging](../04-kriging/learn_04.md)).

# %%
reach = variogram.structures[0].range
passes = [
    bt.Search(reach, max_samples=12, min_samples=6, max_per_hole=2),
    bt.Search(2 * reach, max_samples=12, min_samples=4, max_per_hole=2),
    bt.Search(300, max_samples=12, max_per_hole=2),
]
zn = bt.OrdinaryKriging(variogram, passes).fit(ore, "ZN_PCT", holes="HOLE_ID", domain_column="LENS")
kriged = zn.predict(blocks, diagnostics=True, domain_column="LENS")
measured = ore.filter(np.isfinite(ore["DENSITY"]))
density = bt.OrdinaryKriging(variogram, passes).fit(
    measured, "DENSITY", holes="HOLE_ID", domain_column="LENS"
)
blocks = blocks.with_columns(
    {"zn": kriged["value"], "density": density.predict(blocks, domain_column="LENS"), "pass": kriged["pass"]}
)
for p in (1, 2, 3):
    print(f"pass {p}: {np.mean(kriged['pass'] == p):.0%} of blocks")

# %% [markdown]
# ## A sparse grid of mining blocks
#
# Kriged sub-blocks give the metal in place. A mine digs fixed blocks, the selective mining units (SMUs), here
# cubes of 5 m on a regular grid over the lenses. You keep the cells whose center lies below the collar topography
# and inside a lens. The masked model stores those rows alone
# ([block models from extents](../../examples/02-data-and-geometry/12-block-model-from-extents/example_02_12.md)).

# %%
grid = bt.BlockModel.from_extents(*lenses.values(), size=(5, 5, 5), buffer=5, snap=True)
centers = grid.centroids
below = centers[:, 2] < ground.grid["z"][ground.grid.row_at(centers[:, :2])]
smu_lens = np.full(len(grid), "", dtype=object)
for name, mesh in lenses.items():
    smu_lens[below & mesh.contains(centers)] = name
keep = smu_lens != ""
smus = grid.mask(keep).with_column("LENS", list(smu_lens[keep]))
print(f"{len(grid):,} cells in the grid, {below.sum():,} below ground, {len(smus):,} SMUs kept")
print(
    f"SMUs: {smus.volumes.sum() / 1e6:.2f} Mm³ against {sum(m.volume for m in lenses.values()) / 1e6:.2f} Mm³ of lens"
)

# %% [markdown]
# Selecting by block center trims 3.5 % of the lens volume: thin edges where no center falls inside are lost, a first
# taste of dilution and ore loss at this block size.
#
# ## Turning bands
#
# A kriged SMU is too smooth to say how much ore lies above a cutoff. Turning bands draws 50 realizations of zinc,
# each lens from its own data, on eight nodes per SMU; `blocks=` averages each realization over its SMU, and
# `grade_tonnage_cutoffs=` builds a grade-tonnage curve per realization while it streams
# ([turning bands](../../examples/08-stochastic-simulation/04-turning-bands/example_08_04.md),
# [simulation](../05-simulation/learn_05.md)).

# %%
smus = smus.with_column("density", density.predict(smus, domain_column="LENS"))
nodes = smus.discretize(2)
parent = np.asarray(nodes["block"], dtype=np.int64)
nodes = nodes.with_column("LENS", list(np.asarray(smus["LENS"], dtype=object)[parent]))
tb = bt.TurningBands(gaussian, search=passes[1]).fit(
    ore, "ZN_PCT", weights="weight", holes="HOLE_ID", domain_column="LENS"
)
cutoffs = list(np.arange(0.0, 12.5, 0.5))
start = time.perf_counter()
summary = tb.simulate(
    nodes,
    n=50,
    seed=7,
    blocks=smus,
    domain_column="LENS",
    cutoffs=[5.0],
    grade_tonnage_cutoffs=cutoffs,
    density=smus["density"],
    keep=[0],
)
print(f"{summary.n} realizations on {len(nodes):,} nodes in {time.perf_counter() - start:.0f} s")
smus = smus.with_columns(
    {
        "etype": summary.mean,
        "realization": summary.realizations[0],
        "p_above_5": summary.probability_above[:, 0],
    }
)

# %% [markdown]
# A dip section across the three lenses compares the kriged sub-blocks with one realization on the SMUs and the
# probability that an SMU exceeds 5 % Zn.

# %%
center = np.mean([mesh.vertices.mean(axis=0) for mesh in lenses.values()], axis=0)
plane = (tuple(center), 111.0, 90.0)
panels = [
    (blocks, "zn", "Kriged Zn (%), sub-blocks", {"vmin": 0, "vmax": 12}),
    (smus, "realization", "Zn (%), one realization, SMUs", {"vmin": 0, "vmax": 12}),
    (smus, "p_above_5", "P(SMU Zn > 5 %)", {"vmin": 0, "vmax": 1, "cmap": "magma"}),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained", sharey=True)
for ax, (model, column, title, style) in zip(axes, panels, strict=True):
    bt.plot.section(model, column, plane=plane, ax=ax, colorbar=False, **style)
    fig.colorbar(ax.images[0], ax=ax, orientation="horizontal", shrink=0.8)
    ax.set(title=title, aspect="equal", xlim=axes[0].get_xlim(), ylim=axes[0].get_ylim())
    ax.set(xlabel="Along section, N111° (m)", ylabel="")
axes[0].set_ylabel("Elevation (m)")
save(fig, "sections")

# %% [markdown]
# ## Validation
#
# An estimate should keep the declustered mean of its data. Nearest neighbor, which copies the closest composite
# into each block, is a second unbiased reference, and the E-type, the mean of the realizations, a third. A swath
# plot follows all four along strike ([checking a model](../06-checking-a-model/learn_06.md)).

# %%
nearest = bt.NearestNeighbor(bt.Search(300, max_samples=1)).fit(ore, "ZN_PCT", domain_column="LENS")
blocks = blocks.with_column("nn", nearest.predict(blocks, domain_column="LENS"))
smu_names = np.asarray(smus["LENS"], dtype=object)
for name in lenses:
    inside = block_lens == name
    bias = bt.global_bias(
        blocks["zn"][inside],
        ore["ZN_PCT"][ore_names == name],
        weights=blocks.volumes[inside],
        data_weights=weights[ore_names == name],
    )
    nn = np.average(blocks["nn"][inside], weights=blocks.volumes[inside])
    print(
        f"{name}: declustered {bias['data_mean']:.2f}, kriged {bias['estimate_mean']:.2f}, "
        f"nearest neighbor {nn:.2f}, E-type {smus['etype'][smu_names == name].mean():.2f} % Zn"
    )

fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
bt.plot.swath(
    [
        bt.swath(ore, "ZN_PCT", 50.0, azimuth=21.0, weights="weight"),
        bt.swath(blocks, "nn", 50.0, azimuth=21.0, weights=blocks.volumes),
        bt.swath(blocks, "zn", 50.0, azimuth=21.0, weights=blocks.volumes),
        bt.swath(smus, "etype", 50.0, azimuth=21.0),
    ],
    labels=["declustered composites", "nearest neighbor", "kriged", "turning bands E-type"],
    ax=ax,
)
for line, color in zip(ax.lines, (INK, GRAY, ACCENT, HIGHLIGHT), strict=True):
    line.set_color(color)
ax.legend(loc="lower center", ncol=2)
ax.set(xlabel="Distance along strike, N021° (m)", ylabel="Zn (%)", title="Swath along strike, three lenses")
save(fig, "swath")

# %% [markdown]
# Kriging and the E-type sit within 0.3 % Zn of the declustered composites in every lens. Nearest neighbor runs
# 0.4 to 0.7 % lower in lenses 2 and 3, the two with the fewest composites.
#
# Simulated values must also reproduce the histogram and the variogram of the data. `check_realizations` compares
# 20 more realizations, drawn at the SMU centers so that their support matches the composites', with the
# declustered data along strike and down dip
# ([realization checks](../../examples/10-checking-models/04-realization-checks/example_10_04.md)).

# %%
points = tb.simulate(smus, n=20, seed=8, domain_column="LENS", keep=True)
check = bt.check_realizations(
    smus,
    points,
    ore,
    "ZN_PCT",
    weights="weight",
    variogram=gaussian,
    lag=10.0,
    max_lag=120.0,
    directions=[(21.0, 0.0), (111.0, 60.0)],
)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
bt.plot.histogram_reproduction(check, ax=a)
a.set(xlabel="Zn (%)", title="Histogram, 20 realizations")
bt.plot.variogram_reproduction(check, ax=b)
b.set(xlabel="Lag distance (m)", ylabel="γ(h), normal scores", title="Variogram along strike and down dip")
save(fig, "reproduction")

# %% [markdown]
# ## Grade and tonnage
#
# Tonnes are volume times kriged density. Kriging predicts more tonnes at low cutoffs and fewer at high cutoffs than
# the SMUs can deliver, because its blocks vary less than real 5 m blocks. The simulated curve carries its own
# uncertainty, the band between the 10th and 90th percentile realizations
# ([recoverable resources](../../examples/09-recoverable-resources/index.md)).

# %%
kriged_gt = bt.grade_tonnage("zn", cutoffs, weights=blocks.volumes, density="density", data=blocks)
simulated_gt = summary.grade_tonnage()
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
bt.plot.grade_tonnage({"kriged sub-blocks": kriged_gt, "turning bands, 5 m SMUs": simulated_gt}, ax=ax)
ax.set(xlabel="Zn cutoff (%)", ylabel="Tonnes above cutoff")
save(fig, "grade_tonnage")

at = np.asarray(kriged_gt["cutoff"]) == 5.0
print(
    f"kriged, above 5 % Zn: {kriged_gt['tonnage'][at][0] / 1e6:.1f} Mt at {kriged_gt['mean_grade'][at][0]:.2f} % Zn"
)
at = np.asarray(simulated_gt["cutoff"]) == 5.0
for p, tonnes, grade in zip(
    *(np.asarray(simulated_gt[c])[at] for c in ("probability", "tonnage", "mean_grade")), strict=True
):
    print(f"simulated P{p * 100:.0f}, above 5 % Zn: {tonnes / 1e6:.1f} Mt at {grade:.2f} % Zn")

# %% [markdown]
# Above 5 % Zn, kriging reports 11.3 Mt at 6.94 %; the SMUs deliver 9.2 to 10.0 Mt at 7.7 to 7.9 %, less ore at a
# higher grade.
#
# ## In 3D
#
# `bt.plot3d` draws containers on pyvista: the collar topography, the hole traces, the lens wireframes, and the
# SMUs more likely than not above 5 % Zn. Drag to rotate; in Colab the view runs in the browser
# ([3D views](../../examples/02-data-and-geometry/11-3d-views/example_02_11.md)).

# %%
low = np.min([mesh.bounds[0] for mesh in lenses.values()], axis=0) - 40
high = np.max([mesh.bounds[1] for mesh in lenses.values()], axis=0) + 40
traces = bt.plot3d.to_pyvista(holes).clip_box([low[0], high[0], low[1], high[1], low[2], 450], invert=False)
scene = bt.plot3d.plot(ground.mesh, color=LIGHT, opacity=0.4)
scene.add(traces, color=GRAY, line_width=1, opacity=0.6)
for mesh in lenses.values():
    scene.add(mesh, color=LIGHT, opacity=0.2)
scene.add(
    smus.mask(smus["p_above_5"] > 0.5),
    "p_above_5",
    cmap="magma",
    clim=(0.5, 1.0),
    scalar_bar_args={"title": "P(SMU Zn > 5 %)"},
)
scene.plotter.view_vector((1.0, -0.7, 0.3))
scene.plotter.camera.zoom(1.5)
scene.show()

# %% [hidden]
image = scene.plotter.screenshot(return_img=True, window_size=(1400, 900))
fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
ax.imshow(image)
ax.set_axis_off()
save(fig, "scene")

# %% [markdown]
# ## Where next
#
# The [learn chapters](../index.md) explain each step from first principles, and the
# [case study](../../examples/12-case-studies/02-drillholes-to-classified-model/example_12_02.md) on the same lenses goes
# on to classify the resource.
