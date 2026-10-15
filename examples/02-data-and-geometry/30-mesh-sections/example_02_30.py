"""
# Mesh sections

`Mesh.section` cuts a mesh with a plane and returns the cut as `Polylines`: closed loops round a solid, open lines
across an open surface. `offsets` moves the plane along its normal, so one call gives a set of parallel sections,
each line tagged with its `offset`. Here the three stacked sulfide lenses are cut into level plans and into
sections across strike.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import tempfile

import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import map_axes, save

# %% [markdown]
# Face columns carry over: each line takes the columns of the faces it crosses, here the lens name. A line through
# faces with different values splits into open lines where the value changes, so a closed loop stays closed only
# when its faces agree.

# %%
data = bt.datasets.stacked_sulphide_lenses()
lenses = {}
for i in (1, 2, 3):
    lens = data[f"lens_{i}"]
    lenses[f"lens {i}"] = lens.with_face_column("lens", [f"lens {i}"] * len(lens.triangles))
lo = min(lens.bounds[0][2] for lens in lenses.values())
hi = max(lens.bounds[1][2] for lens in lenses.values())
levels = np.arange(np.ceil(lo / 50) * 50, hi, 50)
plans = {name: lens.section((0, 0, 0), (0, 0, 1), offsets=levels) for name, lens in lenses.items()}
for name, lines in plans.items():
    print(f"{name}: {lines}")

# %% [markdown]
# ## Level plans
#
# Horizontal sections every 50 m, one outline per lens and level. A loop round a solid wound outward runs
# counter-clockwise seen from the side the normal points to, here from above; a loop round a cavity runs the other
# way.

# %%
fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
cmap = plt.get_cmap("cividis")
for name, lines in plans.items():
    for f in range(len(lines)):
        ring = lines.parts[f]
        color = cmap((lines["offset"][f] - levels[0]) / np.ptp(levels))
        ax.plot(*np.r_[ring, ring[:1]][:, :2].T, color=color, linewidth=1.2)
image = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(levels[0], levels[-1]))
fig.colorbar(image, ax=ax, shrink=0.8, label="Level (m)")
map_axes(ax, "Lens outlines every 50 m")
save(fig, "plans")

# %% [markdown]
# Parallel sections sum back to the volume (Cavalieri): `Polylines.area` gives each level's plan area, times the
# spacing. With levels every 5 m the sum lands within a fraction of a percent of the volume the mesh encloses.

# %%
for name, lens in lenses.items():
    (_, _, z0), (_, _, z1) = lens.bounds
    step = 5.0
    lines = lens.section((0, 0, z0), (0, 0, 1), offsets=np.arange(step / 2, z1 - z0, step))
    print(f"{name}: sections {lines.area().sum() * step:,.0f} m3, mesh {lens.volume:,.0f} m3")

# %% [markdown]
# The sections are ordinary `Polylines`, so they go to GIS files as they are, with their elevation in z and the
# `lens` and `offset` columns as properties ([GeoJSON](../../02-data-and-geometry/28-geojson/README.md)):

# %%
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "lens_1_levels.geojson"
    bt.write_geojson(path, plans["lens 1"])
    back = bt.read_geojson(path)
    print(f"{path.name}: {len(back)} outlines, columns {back.attributes.column_names}")

# %% [markdown]
# ## Sections across strike
#
# The lenses strike 23°, so a section across strike looks along 23°: its normal. Sections every 100 m along strike
# either side of the center of the deposit, drawn in their own frame, distance across strike against elevation.

# %%
strike = np.radians(23.0)
normal = np.array([np.sin(strike), np.cos(strike), 0.0])
across = np.array([np.cos(strike), -np.sin(strike), 0.0])
center = np.array([12350.0, 29900.0, 0.0])
offsets = [-200.0, -100.0, 0.0, 100.0, 200.0]
fig, axes = plt.subplots(1, len(offsets), figsize=(12, 4.2), layout="constrained", sharex=True, sharey=True)
colors = dict(zip(lenses, ["#1f4e79", "#c05a28", "#8c8c8c"], strict=True))
for name, lens in lenses.items():
    lines = lens.section(center, normal, offsets=offsets)
    for f in range(len(lines)):
        ax = axes[offsets.index(lines["offset"][f])]
        ring = lines.parts[f] - center
        u, z = ring @ across, ring[:, 2] + center[2]
        ax.fill(u, z, facecolor=colors[lines["lens"][f]], alpha=0.6, label=lines["lens"][f])
for ax, offset in zip(axes, offsets, strict=True):
    ax.set_title(f"{offset:+.0f} m" if offset else "Center")
    ax.set_aspect("equal")
    ax.set_xlabel("Across strike (m)")
axes[0].set_ylabel("Elevation (m)")
handles, labels = axes[2].get_legend_handles_labels()
unique = dict(zip(labels, handles, strict=True))
fig.legend(unique.values(), unique.keys(), loc="outside lower center", ncols=3)
save(fig, "sections")
