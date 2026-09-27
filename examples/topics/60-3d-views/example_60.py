"""
# 60. 3D views

`cs.plot3d` turns drill holes, points, meshes and block models into pyvista datasets and draws them
(``pip install ceres[3d]``). Here three stacked sulphide lenses are shown with the holes that cut them, their zinc
composites, a sub-blocked model of the lenses and a slice through a model rotated with them. Each scene renders
off-screen to an image.
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
import pyvista as pv
from common import GRAY, LIGHT, save

pv.OFF_SCREEN = True
BAR = {"title": "Zn (%)", "vertical": True, "height": 0.5, "position_x": 0.85, "position_y": 0.25}
STYLE = {"cmap": "cividis", "clim": (0, 10), "scalar_bar_args": BAR}


def show(plotter, title, view=(0.8, -0.6, 0.6)):
    """Renders a pyvista scene into a matplotlib figure."""
    plotter.view_vector(view)
    image = plotter.screenshot(return_img=True, window_size=(1400, 900))
    plotter.close()
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title)
    return fig


# %% [markdown]
# A `Drillholes` becomes one polyline per hole through its desurveyed stations, a `PointSet` points, a `Mesh`
# triangles. The holes are clipped to the box around the lenses; the composites inside the lenses are colored by
# zinc, the others left gray.

# %%
data = cs.datasets.stacked_sulphide_lenses()
holes = cs.Drillholes(data["collars"], data["surveys"], data["assays"])
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
composites = holes.composite(2.0, ["ZN_PCT"])
composites = composites.filter(~np.isnan(composites["ZN_PCT"]))
ore = np.any([lens.contains(composites.coords) for lens in lenses], axis=0)
print(f"{len(holes.holes)} holes, {len(composites):,} composites of 2 m, {ore.sum():,} inside a lens")

lo = np.min([lens.bounds[0] for lens in lenses], axis=0) - 50
hi = np.max([lens.bounds[1] for lens in lenses], axis=0) + 50
box = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
traces = cs.plot3d.to_pyvista(holes).clip_box(box, invert=False)
near = np.all((composites.coords > lo) & (composites.coords < hi), axis=1)

plotter = pv.Plotter(window_size=(1400, 900))
cs.plot3d.plot(traces, plotter=plotter, color=GRAY, line_width=1, opacity=0.3)
cs.plot3d.plot(composites.filter(near & ~ore), plotter=plotter, color=LIGHT, point_size=2)
cs.plot3d.plot(composites.filter(ore), values="ZN_PCT", plotter=plotter, point_size=5, **STYLE)
for lens in lenses:
    cs.plot3d.plot(lens, plotter=plotter, color=LIGHT, opacity=0.25)
save(show(plotter, "Drill holes, Zn composites and the three lenses"), "holes")

# %% [markdown]
# `from_meshes` sub-blocks a grid rotated with the lenses (topic 65) against the solids, and inverse distance fills
# the sub-blocks with the zinc of the composites inside the lenses. A masked or sub-blocked model becomes one
# hexahedron per row, at its parent's rotation.

# %%
rotation = (22.5, 0.0, 55.0)
size = (20, 20, 10)
frame = cs.BlockModel.from_extents(*lenses, size=size, buffer=20, rotation=rotation)
blocks = cs.BlockModel.from_meshes(
    frame.origin,
    size,
    frame.count,
    [(lens, "inside", f"lens {i}") for i, lens in enumerate(lenses, 1)],
    subgrid=4,
    fill="host",
    rotation=rotation,
)
blocks = blocks.mask(np.asarray(blocks["domain"], dtype=object) != "host")
search = cs.Search(radius=100, min_samples=1, max_samples=12)
idw = cs.InverseDistance(search, power=2).fit(composites.coords[ore], composites["ZN_PCT"][ore])
blocks = blocks.with_column("zn", idw.predict(blocks))
solid = sum(lens.volume for lens in lenses)
print(f"{len(blocks):,} sub-blocks, {blocks.volumes.sum() / 1e6:.2f} Mm3 for {solid / 1e6:.2f} Mm3 of lens")

plotter = pv.Plotter(window_size=(1400, 900))
cs.plot3d.plot(blocks, values="zn", plotter=plotter, **STYLE)
cs.plot3d.plot(traces, plotter=plotter, color=GRAY, line_width=1, opacity=0.4)
save(show(plotter, "Sub-blocks of the lenses, colored by Zn"), "subblocks")

# %% [markdown]
# A regular model keeps its geometry implicit: `to_pyvista` returns an image grid oriented by the model's rotation.
# `slices` cuts it through its center along the world axes; any pyvista cut works too. Filled with the
# inverse-distance zinc of all composites, the rotated grid is cut here across strike, through its center: a dip
# section where the three lenses are the high-grade bands. Blocks with no composite within 100 m stay empty.

# %%
everywhere = cs.InverseDistance(search, power=2).fit(composites.coords, composites["ZN_PCT"])
rotated = frame.with_column("zn", everywhere.predict(frame))
estimated = np.isfinite(rotated["zn"]).mean()
print(f"rotated grid {frame.count}: {estimated:.0%} of {len(frame):,} blocks estimated")
grid = cs.plot3d.to_pyvista(rotated)
strike = np.radians(rotation[0])
section = grid.slice(normal=(np.sin(strike), np.cos(strike), 0), origin=grid.center)
plotter = pv.Plotter(window_size=(1400, 900))
cs.plot3d.plot(section, values="zn", plotter=plotter, nan_opacity=0, **STYLE)
for lens in lenses:
    cs.plot3d.plot(lens, plotter=plotter, color=LIGHT, opacity=0.2)
save(show(plotter, "Dip section through a grid rotated with the lenses", view=(0.5, 1, 0.15)), "section")
