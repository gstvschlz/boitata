"""
# 22. 3D views

`cs.plot3d` turns drill holes, points, meshes and block models into pyvista datasets and draws them
(``pip install ceres[3d]``). Here the high-grade zinc of the drill-hole dataset is wrapped in a convex hull,
sub-blocked against it and viewed with the holes around it. Each scene renders off-screen to an image.
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
import pyvista as pv
from common import GRAY, LIGHT, save

pv.OFF_SCREEN = True
BAR = {"title": "Zn (%)", "vertical": True, "height": 0.5, "position_x": 0.85, "position_y": 0.25}


def show(plotter, title):
    """Renders a pyvista scene into a matplotlib figure."""
    plotter.camera_position = "iso"
    image = plotter.screenshot(return_img=True, window_size=(1400, 900))
    plotter.close()
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title)
    return fig


# %% [markdown]
# The holes of one cluster, their 2 m zinc composites, and the convex hull of the composites above 5 % Zn:

# %%
dh = cs.datasets.drillholes()
composites = dh.composite(2.0, ["ZN"])
xyz, zn = composites.coords, composites["ZN"]
window = ~np.isnan(zn) & (xyz[:, 0] > 4550) & (xyz[:, 0] < 4950) & (xyz[:, 1] > 7400) & (xyz[:, 1] < 7700)
local = cs.PointSet(xyz[window], {"ZN": zn[window]})
hull = cs.convex_hull(xyz[window][zn[window] > 5])
lo, hi = hull.bounds
print(f"{len(local)} composites, hull {hull.volume:,.0f} m3")

traces = cs.plot3d.to_pyvista(dh).clip_box([4550, 4950, 7400, 7700, -1e4, 1e4], invert=False)
plotter = pv.Plotter(window_size=(1400, 900))
cs.plot3d.plot(traces, plotter=plotter, color=GRAY, line_width=1)
cs.plot3d.plot(
    local, scalars="ZN", plotter=plotter, cmap="cividis", clim=(0, 15), point_size=5, scalar_bar_args=BAR
)
cs.plot3d.plot(hull, plotter=plotter, color=LIGHT, opacity=0.35)
save(show(plotter, "Drill holes, Zn composites and the hull of Zn > 5 %"), "holes")

# %% [markdown]
# `from_meshes` sub-blocks a 10 m grid against the hull; `to_pyvista` draws each sub-block as a hexahedron at its
# parent's rotation. Inverse-distance Zn fills the sub-blocks:

# %%
size = 10.0
count = np.ceil((np.array(hi) - lo) / size).astype(int)
blocks = cs.BlockModel.from_meshes(lo, (size,) * 3, count, [(hull, "inside", "ore")], subgrid=4)
blocks = blocks.mask(np.array(blocks["domain"]) == "ore")
search = cs.Search(radius=100, min_samples=1, max_samples=12)
blocks = blocks.with_column(
    "zn", cs.InverseDistance(search, power=2).fit(local.coords, local["ZN"]).predict(blocks)
)
print(f"{len(blocks)} sub-blocks, {blocks.volumes.sum():,.0f} m3")

grid = cs.plot3d.to_pyvista(blocks)
STYLE = {"cmap": "cividis", "clim": (0, 15), "scalar_bar_args": BAR}
plotter = pv.Plotter(window_size=(1400, 900))
cs.plot3d.plot(grid.clip("y", origin=grid.center), scalars="zn", plotter=plotter, **STYLE)
cs.plot3d.plot(traces, plotter=plotter, color=GRAY, line_width=1)
save(show(plotter, "Sub-blocks inside the hull, cut at its center, colored by Zn"), "subblocks")

# %% [markdown]
# A regular model keeps its geometry implicit: `to_pyvista` returns an image grid oriented by the model's rotation.
# Rotation turns the grid about its origin, so the origin is placed for the grid to cover the hull: at 30° azimuth the
# grid's y axis points 30° east of north and its x axis 30° south of east. `slices` cuts it through its center along
# the world axes.

# %%
side = np.hypot(*(np.array(hi) - lo)[:2])
x_axis = np.array([np.cos(np.pi / 6), -np.sin(np.pi / 6), 0])
y_axis = np.array([np.sin(np.pi / 6), np.cos(np.pi / 6), 0])
center = (np.array(lo) + hi) / 2
origin = center - side / 2 * (x_axis + y_axis) - [0, 0, (hi[2] - lo[2]) / 2]
n = int(np.ceil(side / size))
rotated = cs.BlockModel(origin=origin, size=(size,) * 3, count=(n, n, count[2]), rotation=(30, 0, 0))
rotated = rotated.with_column(
    "zn", cs.InverseDistance(search, power=2).fit(local.coords, local["ZN"]).predict(rotated)
)
plotter = cs.plot3d.slices(rotated, scalars="zn", nan_opacity=0, **STYLE)
cs.plot3d.plot(hull, plotter=plotter, style="wireframe", color=GRAY, opacity=0.3)
save(show(plotter, "Orthogonal slices of a grid rotated 30° in azimuth"), "slices")
