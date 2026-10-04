"""
# block model volumes

`style="volume"` renders a regular block model on the GPU: each block a uniform cube of its color, null blocks
transparent, the color map and range shared with every other layer of the variable. a model larger than GPU memory
draws a coarser copy while the camera moves and the full one once it stops. masked and sub-blocked models draw as
cells; masking a regular model at a cutoff shows the blocks above it.
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
import pyvista as pv
from common import save

pv.OFF_SCREEN = True
BAR = {"vertical": True, "height": 0.5, "position_x": 0.85, "position_y": 0.25}


def image(scene, title, view=(0.8, -0.6, 0.6)):
    """Renders a scene into a matplotlib figure."""
    scene.view_vector(view)
    pixels = scene.screenshot(return_img=True, window_size=(1400, 900))
    scene.close()
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    ax.imshow(pixels)
    ax.set_axis_off()
    ax.set_title(title)
    return fig


# %% [markdown]
# zinc composites of the stacked sulphide lenses inform a rotated grid by inverse distance within 60 m; blocks with no
# composite in reach stay null. as a volume, opacity follows the grade: `opacity="linear"` (the default) leaves low
# grades faint, so the high-grade cores show through.

# %%
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes(data["collars"], data["surveys"], data["assays"])
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
composites = holes.composite(2.0, ["ZN_PCT"])
grid = bt.BlockModel.from_extents(*lenses, size=(10, 10, 5), buffer=20, rotation=(22.5, 0.0, 55.0))
search = bt.Search(radius=60, min_samples=1, max_samples=12)
idw = bt.InverseDistance(search, power=2).fit(composites.coords, composites["ZN_PCT"])
grid = grid.with_column("ZN_PCT", idw.predict(grid))

scene = bt.plot3d.Scene(window_size=(1400, 900))
scene.add(grid, "ZN_PCT", style="volume", clim=(0, 10), scalar_bar_args={"title": "Zn (%)", **BAR})
print(f"{len(grid):,} blocks, {np.isnan(grid['ZN_PCT']).mean():.0%} null")
save(image(scene, "Zn grade as a volume, opacity rising with grade"), "volume")

# %% [markdown]
# the blocks above a 4 % cutoff, masked out of the grid, draw as cells with their edges.

# %%
rich = grid.mask(np.nan_to_num(grid["ZN_PCT"]) > 4)
scene = bt.plot3d.Scene(window_size=(1400, 900))
scene.add(rich, "ZN_PCT", clim=(0, 10), show_edges=True, scalar_bar_args={"title": "Zn (%)", **BAR})
print(f"{len(rich):,} of {len(grid):,} blocks above 4 % Zn")
save(image(scene, "Blocks above 4 % Zn"), "cutoff")
