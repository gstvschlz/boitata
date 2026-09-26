"""
# 19. Geological modelling

[Chapter 15](../15-implicit/README.md) modelled a grade shell. A geological unit is modelled from where the drill
holes cross its contacts, and from structural readings where the rock is measured. Here the massive sulphide
(`MS`) of the drillhole dataset is modelled from its logged contacts with three engines, then a synthetic fold
shows what plane and lineation readings add and how the field's gradient returns the dip. Last, both become
sub-blocked domain models.
"""

# %% [hidden]
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore", ".*locations hold several samples")

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GREY, HIGHLIGHT, INK, LIGHT, save

# %% [markdown]
# Every change between `MS` and another rock down a hole is a contact; `Drillholes.at` places it in space from the
# desurveyed path. The field is pinned to 0 at the contacts, +1 at the middle of each `MS` interval and −1 outside,
# using the intervals next to a contact and a sparse subset of the others, which keeps the systems small.

# %%
tables = cs.datasets.drillhole_tables()
dh = cs.Drillholes(tables["collar"], tables["survey"], tables["geology"])
logged = dh.samples()
xyz = logged.coords
hole = np.array(logged["HOLEID"], dtype=object)
lith = np.array(logged["LITH"], dtype=object)
top, bottom = logged["FROM"], logged["TO"]
window = (xyz[:, 0] > 5300) & (xyz[:, 0] < 5500) & (xyz[:, 1] > 8100) & (xyz[:, 1] < 8400)
order = np.lexsort((top, hole))[window[np.lexsort((top, hole))]]
xyz, hole, lith, top, bottom = xyz[order], hole[order], lith[order], top[order], bottom[order]

ms = lith == "MS"
change = (hole[1:] == hole[:-1]) & (ms[1:] != ms[:-1]) & np.isclose(bottom[:-1], top[1:])
contacts = dh.at(list(hole[:-1][change]), bottom[:-1][change])
beside = np.r_[change, False] | np.r_[False, change]
sparse = np.random.default_rng(1).random(len(ms)) < 0.08
coded = ms | beside | sparse
points, code = xyz[coded], np.where(ms[coded], 1.0, -1.0)
print(
    f"{len(set(hole))} holes, {ms.sum()} MS intervals, {len(contacts)} contacts, {coded.sum()} coded points"
)

# %% [markdown]
# The covariance of the `MS` intervals shows a steep lens striking N16°E, plunging 26° north and thin east–west.
# The `kriging` engine is a potential field: dual kriging with a covariance, here of 150 m along the plunge, 68 m up
# the lens and 33 m across it (`rotation` azimuth, dip, rake and `ratios`). The RBF takes the same anisotropy. The
# Gaussian process learns its own ranges along the `rotation` axes. The share of the window each model calls `MS`
# tells a sound model from one that invents bodies:

# %%
rotation, ratios = (16, 26, 90), (0.45, 0.22)


def potential(covariance):
    variogram = cs.Variogram([(covariance, 1.0, 150.0)], rotation=rotation, ratios=ratios)
    return cs.ImplicitModel("kriging", variogram=variogram, drift_degree=0)


models = {
    "kriging, cubic": potential("cubic"),
    "kriging": potential("spherical"),
    "RBF": cs.ImplicitModel("rbf", drift_degree=0, rotation=rotation, ratios=ratios),
    "GP": cs.ImplicitModel("gp", drift_degree=0, rotation=rotation),
}
volume = np.random.default_rng(0).uniform([5300, 8100, 650], [5500, 8400, 950], (20_000, 3))
for name, model in models.items():
    model.fit(points, code, boundaries=contacts)
    right = np.mean(np.sign(model.evaluate(points)) == code)
    share = np.mean(model.evaluate(volume) > 0)
    print(f"{name:>14}: {right:6.1%} of coded points on their side, {share:5.1%} of the window is MS")
report = models["GP"].report
print(f"GP noise variance {report['noise_variance']:.2f}, signal variance {report['signal_variance']:.2f}")

# %% [markdown]
# Kriging and the RBF honour every contact and all but a few coded points, yet with a cubic covariance kriging
# calls about ten times more of the window `MS`: the smooth cubic overshoots between codes a metre apart and grows bodies
# away from the holes. The spherical covariance, rougher at the origin, does not, and agrees with the RBF. The GP
# explains most of the codes as noise, so its field is a smooth trend that misses more than a quarter of them; its
# standard deviation still shows where the model rests on data and where it does not. On an east–west section
# across the lens:

# %%
north = np.median(xyz[ms, 1])
plane = ((0, north, 0), 90, 90)
east, elevation = np.meshgrid(np.linspace(5300, 5500, 201), np.linspace(650, 950, 301))
section = np.c_[east.ravel(), np.full(east.size, north), elevation.ravel()]
extent = (5300, 5500, 650, 950)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True, layout="constrained")
for ax, name, covariance in zip(axes[:2], ("kriging, cubic", "kriging"), ("cubic", "spherical"), strict=True):
    field = models[name].evaluate(section).reshape(east.shape)
    ax.contourf(east, elevation, field, levels=[0, np.inf], colors=[LIGHT])
    ax.contour(east, elevation, field, levels=[0], colors=ACCENT, linewidths=1.2)
    ax.set_title(f"Kriging, {covariance} covariance, northing {north:.0f} m")
_, variance = models["GP"].evaluate(section, variance=True)
sd = axes[2].imshow(
    np.sqrt(variance).reshape(east.shape), origin="lower", extent=extent, cmap="cividis", aspect="auto"
)
axes[2].set_title("GP standard deviation")
fig.colorbar(sd, ax=axes[2], shrink=0.8)
for ax in axes:
    cs.plot.slab(
        xyz[~ms], plane=plane, thickness=20, s=3, color=GREY, linewidths=0, label="other rock", ax=ax
    )
    cs.plot.slab(xyz[ms], plane=plane, thickness=20, s=5, color=HIGHLIGHT, linewidths=0, label="MS", ax=ax)
    cs.plot.slab(
        contacts,
        plane=plane,
        thickness=20,
        s=12,
        marker="x",
        color=INK,
        linewidths=0.8,
        label="contact",
        ax=ax,
    )
    ax.set(xlim=extent[:2], ylim=extent[2:])
for ax in axes[1:]:
    ax.set_ylabel("")
axes[0].legend(loc="lower left", markerscale=2)
save(fig, "section")

# %% [markdown]
# ## Structural readings
#
# Drill holes give contacts at a few places; mapping and oriented core give the dip of the surface at many more.
# A synthetic fold, `z = 100 + 30 sin(2πx / 400)` with its axis north–south, is known exactly. Five holes pierce it,
# twelve outcrops spread along it give its dip and dip direction, and fold-axis lineations (plunge 0, trend 0) are measured at six
# other places. Planes and lineations need the triharmonic kernel; one point above the surface sets which side is
# positive. `isosurface` extracts each modelled surface as a mesh, and `Mesh.vertical_distance` gives how far
# above it 400 points on the true surface lie: the elevation error over the whole fold. The same call flags blocks
# above or below topography.

# %%
rng = np.random.default_rng(4)


def surface(x):
    return 100 + 30 * np.sin(2 * np.pi * x / 400)


def true_dip(xy):
    slope = 30 * 2 * np.pi / 400 * np.cos(2 * np.pi * xy[:, 0] / 400)
    return np.degrees(np.arctan(np.abs(slope))), np.where(slope < 0, 90.0, 270.0)


def on_surface(xy):
    return np.c_[xy, surface(xy[:, 0])]


picks = on_surface(np.array([[40.0, 150], [170, 60], [310, 90], [420, 250], [560, 180]]))
outcrops = np.c_[np.linspace(20, 580, 12) + rng.uniform(-20, 20, 12), rng.uniform(0, 300, 12)]
planes = np.c_[on_surface(outcrops), np.column_stack(true_dip(outcrops))]
axis_readings = rng.uniform([0, 0], [600, 300], (6, 2))
lineations = np.c_[on_surface(axis_readings), np.zeros((6, 2))]
above = [[300.0, 150, 200]]
fits = {
    "5 holes": {},
    "5 holes, 12 planes": {"planes": planes},
    "5 holes, 4 planes": {"planes": planes[:4]},
    "5 holes, 4 planes, 6 lineations": {"planes": planes[:4], "lineations": lineations},
}
x = np.linspace(0, 600, 241)
z = np.linspace(0, 200, 801)
X, Z = np.meshgrid(x, z)
probe = rng.uniform([0, 0], [600, 300], (400, 2))
volume = cs.BlockModel(origin=(-5, -5, 0), size=(5, 5, 5), count=(122, 62, 40))
folds, depths = {}, {}
for name, readings in fits.items():
    folds[name] = cs.ImplicitModel(kernel="triharmonic").fit(above, [1.0], boundaries=picks, **readings)
    field = folds[name].evaluate(np.c_[X.ravel(), np.full(X.size, 150.0), Z.ravel()]).reshape(X.shape)
    depths[name] = z[np.argmin(np.abs(field), axis=0)]
    _, gradient = folds[name].evaluate(on_surface(probe), gradient=True)
    dip = np.degrees(np.arccos(np.abs(gradient[:, 2]) / np.linalg.norm(gradient, axis=1)))
    error = np.abs(folds[name].isosurface(volume).vertical_distance(on_surface(probe)))
    print(
        f"{name:>31}: surface within {np.median(error):4.1f} m (median), {error.max():4.1f} m (max); "
        f"dip within {np.median(np.abs(dip - true_dip(probe)[0])):3.1f}°"
    )

# %% [markdown]
# Twelve planes bring the surface from 14 m to 2 m of the truth (median) and the dip from 7° to under 2°. With
# only four planes, the lineations add the direction of the fold axis and improve both.

# %%
fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
ax.plot(x, surface(x), color=INK, lw=2.2, label="true surface")
for (name, depth), color, style in zip(
    depths.items(), (GREY, ACCENT, HIGHLIGHT, HIGHLIGHT), ("--", "-", ":", "-"), strict=True
):
    ax.plot(x, depth, color=color, ls=style, lw=1.2, label=name)
ax.scatter(picks[:, 0], picks[:, 2], color=INK, zorder=3, s=18, label="hole pierce points (all northings)")
ax.set(xlabel="Easting (m)", ylabel="Elevation (m)", title="Fold at northing 150 m", xlim=(0, 600))
ax.legend(ncol=2, loc="lower left", fontsize=8)
save(fig, "fold")

# %% [markdown]
# The gradient of the field is normal to the surface, so `evaluate(..., gradient=True)` returns the modelled dip and
# dip direction anywhere, here against the truth at the 400 probe points of the twelve-plane model:

# %%
_, gradient = folds["5 holes, 12 planes"].evaluate(on_surface(probe), gradient=True)
dip = np.degrees(np.arccos(np.abs(gradient[:, 2]) / np.linalg.norm(gradient, axis=1)))
direction = np.degrees(np.arctan2(gradient[:, 0], gradient[:, 1])) % 360
truth_dip, truth_direction = true_dip(probe)
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
cs.plot.scatter(truth_dip, dip, line=False, ax=a, color=ACCENT)
a.set(xlabel="True dip (°)", ylabel="Modelled dip (°)", title="Dip from the gradient")
b.hist(
    np.abs((direction - truth_direction + 180) % 360 - 180),
    bins=np.arange(0, 32, 2),
    color=LIGHT,
    edgecolor=GREY,
)
b.set(xlabel="Dip direction error (°)", ylabel="Probe points", title="Dip direction")
save(fig, "dip")

# %% [markdown]
# ## A domain block model
#
# `BlockModel.from_meshes` turns meshes into a sub-blocked model: `(mesh, rule, label)` domains in priority order,
# each sub-cell labelled by the first that holds its centre, `"inside"` a solid or `"below"` or `"above"` a surface
# such as topography. On 10 m blocks with 1 m sub-cells in elevation, the volume below the true fold and below the
# twelve-plane model. A surface given as a grid of elevations, the usual form of topography, is a 2D `BlockModel`
# with an elevation column; `grid_surface` triangulates it through the block centres, leaving holes where the
# elevation is missing. The true fold is such a grid at 5 m:

# %%
topography = cs.BlockModel(origin=(-7.5, -7.5, 0), size=(5, 5, 1), count=(123, 63, 1))
topography = topography.with_column("z", surface(topography.centroids[:, 0]))
surfaces = {
    "true fold": cs.grid_surface(topography, "z"),
    "12-plane model": folds["5 holes, 12 planes"].isosurface(volume),
}
xs = np.linspace(-5, 605, 6101)
exact = 310 * np.trapezoid(surface(xs), xs)
for name, mesh in surfaces.items():
    fold = cs.BlockModel.from_meshes(
        (-5, -5, 0),
        (10, 10, 10),
        (61, 31, 20),
        [(mesh, "below", "footwall")],
        subgrid=(1, 1, 10),
        fill="hanging wall",
    )
    below = fold.volumes[np.array(fold["domain"]) == "footwall"].sum()
    print(
        f"{name:>15}: {len(fold)} sub-blocks, footwall {below / 1e6:.3f} Mm3 ({below / exact - 1:+.2%} of the truth)"
    )

# %% [markdown]
# Sub-blocks follow the true fold to within 0.1 %; the rest of the model's shortfall is the surface's own error.
# The `MS` lens of the spherical kriging model, closed at the edges of its window, is a solid. `regularize` averages
# the units to 20 m blocks, each taking the unit that fills most of it. A lens about 30 m thick fills few 20 m
# blocks by more than half, so the label keeps a fraction of its volume; an `MS` indicator column averages to a
# proportion per block instead and keeps all of it.

# %%
window = cs.BlockModel(origin=(5300, 8100, 650), size=(10, 10, 10), count=(20, 30, 30))
lens = models["kriging"].isosurface(window, closed=True)
units = window.subblock([(lens, "inside", "MS")], 4, fill="other")
is_ms = np.array(units["domain"]) == "MS"
units = units.with_column("ms", is_ms.astype(float))
coarse = units.regularize(cs.BlockModel(origin=(5300, 8100, 650), size=(20, 20, 20), count=(10, 15, 15)))
print(f"MS volume: mesh {lens.volume / 1e6:.3f} Mm3, sub-blocks {units.volumes[is_ms].sum() / 1e6:.3f} Mm3")
labelled = coarse.volumes[np.array(coarse["domain"]) == "MS"].sum()
proportion = (coarse.volumes * coarse["fraction"] * coarse["ms"]).sum()
print(f"20 m blocks: labelled MS {labelled / 1e6:.3f} Mm3, MS proportion {proportion / 1e6:.3f} Mm3")
