"""
# contact surfaces

you can model a geological unit from where the drill holes cross its contacts. here three engines model lens 1 of
the stacked sulphide lenses from its logged contacts, and the supplied `lens_1.stl` solid scores each model.
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

# %%
data = bt.datasets.stacked_sulphide_lenses()
lens = data["lens_1"]
print(f"lens 1: closed {lens.is_closed}, {lens.volume / 1e6:.2f} Mm3")

# %% [markdown]
# the logs say `MS` or `SMS` for all three lenses; the intervals of lens 1 are those whose middle lies in its solid.
# each change between lens 1 and other rock down a hole is a contact, and `Drillholes.at` places it in space from the
# desurveyed path. the codes pin the field to 0 at the contacts, +1 at the middle of each lens interval and −1
# outside, from the intervals next to a contact and a sparse subset of the others.

# %%
holes = bt.Drillholes(data["collars"], data["surveys"], data["lithology"])
logged = holes.samples()
xyz = logged.coords
lo, hi = np.array(lens.bounds[0]) - 30, np.array(lens.bounds[1]) + 30
order = np.lexsort((logged["FROM"], logged["HOLE_ID"]))
order = order[np.all((xyz[order] > lo) & (xyz[order] < hi), axis=1)]
xyz = xyz[order]
hole = np.array(logged["HOLE_ID"], dtype=object)[order]
lith = np.array(logged["LITH"], dtype=object)[order]
bottom = np.asarray(logged["TO"])[order]

unit = np.isin(lith, ["MS", "SMS"]) & lens.contains(xyz)
change = (hole[1:] == hole[:-1]) & (unit[1:] != unit[:-1])
contacts = holes.at(list(hole[:-1][change]), bottom[:-1][change])
beside = np.r_[change, False] | np.r_[False, change]
sparse = np.random.default_rng(1).random(len(unit)) < 0.15
coded = unit | beside | sparse
points, code = xyz[coded], np.where(unit[coded], 1.0, -1.0)
print(
    f"{len(set(hole))} holes, {unit.sum()} lens intervals, {len(contacts)} contacts, {coded.sum()} coded points"
)

# %% [markdown]
# the lens strikes N22.5°E and dips 60° to the east-southeast, so the anisotropy axes run along strike, down dip and
# across the lens (`rotation` azimuth 22.5, dip 0, rake 60). the `kriging` engine is a potential field: dual kriging
# with a covariance, here of 200 m along strike, 100 m down dip and 20 m across (`ratios` 0.5 and 0.1). the RBF takes
# the same anisotropy; the gaussian process (GP) learns its own ranges along the `rotation` axes. on 5 m cells, each
# model's lens meets the solid: the share of the solid it covers, and the share of its own volume outside the solid.

# %%
rotation, ratios = (22.5, 0, 60), (0.5, 0.1)


def potential(covariance):
    variogram = bt.Variogram([(covariance, 1.0, 200.0)], rotation=rotation, ratios=ratios)
    return bt.ImplicitModel("kriging", variogram=variogram, degree=0)


models = {
    "kriging, cubic": potential("cubic"),
    "kriging": potential("spherical"),
    "RBF": bt.ImplicitModel("rbf", degree=0, rotation=rotation, ratios=ratios),
    "GP": bt.ImplicitModel("gp", degree=0, rotation=rotation),
}
cells = bt.BlockModel(origin=lo, size=(5, 5, 5), count=np.ceil((hi - lo) / 5).astype(int))
solid = lens.contains(cells.centroids)
for name, model in models.items():
    model.fit(points, code, boundaries=contacts)
    right = np.mean(np.sign(model.predict(points)) == code)
    inside = model.predict(cells) > 0
    print(
        f"{name:>14}: {right:6.1%} of coded points on their side, {inside.sum() * 125 / 1e6:.2f} Mm3, "
        f"covers {(inside & solid).sum() / solid.sum():.0%} of the solid, {(inside & ~solid).sum() / inside.sum():.0%} outside it"
    )
print(f"GP ranges {[round(r) for r in models['GP'].report['lengthscales']]} m")

# %% [markdown]
# kriging with a spherical covariance and the RBF honor each contact and agree with the solid except for a thin rind.
# the cubic covariance, smooth at the origin, overshoots between codes a few meters apart and swells the lens. the GP
# explains a sixth of the codes as noise and stretches its ranges along strike and down dip far beyond the window,
# so its lens is a slab that leaves the solid at both ends. on a vertical section down the dip, with the trace of the
# solid in black:

# %%
center = lens.vertices.mean(axis=0)
plane = (center, 112.5, 90)
u = np.array([np.sin(np.radians(112.5)), np.cos(np.radians(112.5)), 0.0])
normal = np.cross(u, [0, 0, 1])
along, elevation = np.meshgrid(np.arange(-160, 160.1, 2.0) + center @ u, np.arange(lo[2], hi[2], 2.0))
section = (center @ normal) * normal + along.reshape(-1, 1) * u + elevation.reshape(-1, 1) * [0, 0, 1]

fig, axes = plt.subplots(1, 4, figsize=(11, 5.6), sharey=True, layout="constrained")
for ax, (name, model) in zip(axes, models.items(), strict=True):
    field = model.predict(section).reshape(along.shape)
    ax.contourf(along, elevation, field, levels=[0, np.inf], colors=[LIGHT])
    ax.contour(along, elevation, field, levels=[0], colors=ACCENT, linewidths=1.2)
    style = {"plane": plane, "thickness": 20, "ax": ax}
    bt.plot.slab(points[code < 0], s=4, color=GRAY, label="other rock", meshes=lens, **style)
    bt.plot.slab(points[code > 0], s=6, color=HIGHLIGHT, label="lens 1", **style)
    bt.plot.slab(contacts, s=14, marker="x", color=INK, linewidths=0.8, label="contact", **style)
    ax.set(title=name, xlabel="Toward 112.5° (m)", xlim=(along.min(), along.max()), ylim=(lo[2], hi[2]))
for ax in axes[1:]:
    ax.set_ylabel("")
axes[0].legend(loc="upper right", markerscale=2)
save(fig, "section")

# %% [markdown]
# `isosurface(cells, closed=True)` turns a field into a solid mesh, which `BlockModel.from_meshes` and `subblock`
# turn into a domain model (see [sub-blocks](../../02-data-and-geometry/06-sub-blocks/README.md)). [structural data](../../11-geological-modeling/03-structural-data/README.md) adds plane and lineation readings.
