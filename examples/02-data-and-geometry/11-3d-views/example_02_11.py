"""
# 3D views

`bt.plot3d` draws drill holes, points, meshes and block models in a 3D scene that runs in the browser
(``pip install boitata[3d]`` for the notebook widget). The scenes show three stacked sulfide lenses with the holes
that cut them, their zinc composites, a sub-blocked model of the lenses and a section through a model rotated with
them. Each scene renders to an image with `screenshot`; in the documentation a click on the image loads the live
scene.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import boitata as bt
import numpy as np
from common import GRAY, LIGHT, show

ZN = {"cmap": "cividis", "clim": (0, 10), "label": "Zn (%)"}

# %% [markdown]
# A `Drillholes` without intervals draws its desurveyed traces, a `PointSet` points, a `Mesh` triangles. Only the
# holes collared over the lenses are drawn. Zinc colors the composites inside the lenses; the others stay gray.

# %%
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes.from_tables(data)
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
composites = holes.composite(2.0, ["ZN_PCT"])
composites = composites.drop_null("ZN_PCT")
ore = np.any([lens.contains(composites.coords) for lens in lenses], axis=0)
print(f"{len(holes.holes)} holes, {len(composites):,} composites of 2 m, {ore.sum():,} inside a lens")

lo = np.min([lens.bounds[0] for lens in lenses], axis=0) - 50
hi = np.max([lens.bounds[1] for lens in lenses], axis=0) + 50
xy = np.c_[data["collars"]["X"], data["collars"]["Y"]]
over = data["collars"].filter(np.all((xy > lo[:2]) & (xy < hi[:2]), axis=1))
traces = bt.Drillholes(over, data["surveys"])
near = np.all((composites.coords > lo) & (composites.coords < hi), axis=1)

scene = bt.plot3d.Scene()
scene.add(traces, name="holes", color=GRAY, line_width=1, opacity=0.5)
scene.add(composites.filter(near & ~ore), name="host", color=LIGHT, point_size=2)
scene.add(composites.filter(ore), "ZN_PCT", name="ore", point_size=6, **ZN)
for i, lens in enumerate(lenses, 1):
    scene.add(lens, name=f"lens {i}", color=GRAY, opacity=0.15)
scene.view(azimuth=305, dip=30)
show(scene, "holes", "Drill holes, Zn composites and the three lenses")

# %% [markdown]
# `from_meshes` sub-blocks a grid rotated with the lenses ([block model from extents](../../02-data-and-geometry/12-block-model-from-extents/README.md)) against the solids, and inverse distance fills
# the sub-blocks with the zinc of the composites inside the lenses. Each row draws as a box along the model's
# rotated axes, sub-blocks at their own size.

# %%
rotation = (22.5, 0.0, 55.0)
size = (20, 20, 10)
frame = bt.BlockModel.from_extents(*lenses, size=size, buffer=20, rotation=rotation)
blocks = bt.BlockModel.from_meshes(
    frame.origin,
    size,
    frame.count,
    [(lens, "inside", f"lens {i}") for i, lens in enumerate(lenses, 1)],
    subgrid=4,
    fill="host",
    rotation=rotation,
)
blocks = blocks.filter(blocks["domain"] != "host")
search = bt.Search(radius=100, min_samples=1, max_samples=12)
idw = bt.InverseDistance(search, power=2).fit(composites.filter(ore), "ZN_PCT")
blocks = blocks.with_column("zn", idw.predict(blocks))
solid = sum(lens.volume for lens in lenses)
print(f"{len(blocks):,} sub-blocks, {blocks.volumes.sum() / 1e6:.2f} Mm3 for {solid / 1e6:.2f} Mm3 of lens")

scene = bt.plot3d.Scene()
scene.add(blocks, "zn", name="sub-blocks", **ZN)
scene.add(traces, name="holes", color=GRAY, line_width=1, opacity=0.4)
scene.view(azimuth=305, dip=30)
show(scene, "subblocks", "Sub-blocks of the lenses, colored by Zn")

# %% [markdown]
# A regular model keeps its geometry implicit; the viewer builds its boxes from the centers, sizes and rotation.
# Filled with the inverse-distance zinc of all composites, the rotated grid is cut here by a `section` across
# strike, through its center and one block wide: a dip section where the three lenses are the high-grade bands.
# Blocks with no composite within 100 m are null and never drawn. `contour_step` draws 1% Zn isolines on the cut
# face itself, over the block colors; they update live as the section is dragged.

# %%
everywhere = bt.InverseDistance(search, power=2).fit(composites, "ZN_PCT")
rotated = frame.with_column("zn", everywhere.predict(frame))
estimated = np.isfinite(rotated["zn"]).mean()
print(f"rotated grid {frame.count}: {estimated:.0%} of {len(frame):,} blocks estimated")
center = rotated.coords.mean(axis=0)
across = np.radians(rotation[0] + 90)
reach = 400 * np.array([np.sin(across), np.cos(across), 0])
scene = bt.plot3d.Scene()
scene.add(rotated, "zn", name="rotated grid", contour_step=1.0, **ZN)
for i, lens in enumerate(lenses, 1):
    scene.add(lens, name=f"lens {i}", color=LIGHT, opacity=0.2)
scene.section([center - reach, center + reach], width=20)
scene.view(azimuth=rotation[0] + 180, dip=10)
show(scene, "section", "Dip section through a grid rotated with the lenses")

# %% [markdown]
# Zinc runs over two orders of magnitude, which a linear scale crushes into its bottom colors. `log=True` spreads
# it by decades, the color bar ticked to match. The assays of the holes over the lenses draw as `"tubes"`, shaded
# near the camera and plain lines further off, each swollen with its zinc by `radius_by`; `eye_dome` darkens each
# pixel by how far it lies behind its neighbors on screen, so overlapping holes read in depth.

# %%
assays = bt.Drillholes(over, data["surveys"], data["assays"])
scene = bt.plot3d.Scene(eye_dome=True)
scene.add(
    assays,
    "ZN_PCT",
    name="assays",
    representation="tubes",
    radius=4,
    radius_by="ZN_PCT",
    log=True,
    cmap="cividis",
    label="Zn",
)
scene.view(azimuth=305, dip=30)
show(scene, "log", "Composites as tubes on a log Zn scale, with eye-dome lighting")

# %% [markdown]
# `bins` steps the colors at cutoffs instead, each class in the colormap's color at its middle, with a legend of
# classes: here the composites inside the lenses below 1, 3 and 6% Zn and above. A scene of points alone starts
# with eye-dome lighting on.

# %%
scene = bt.plot3d.Scene()
scene.add(
    composites.filter(ore), "ZN_PCT", name="ore", point_size=6, bins=[1, 3, 6], cmap="cividis", label="Zn"
)
scene.view(azimuth=305, dip=30)
show(scene, "bins", "Zn composites inside the lenses in four classes")
