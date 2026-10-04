# interactive sections

`Scene.section` cuts every layer of a scene with one plane: blocks and lens surfaces show their intersection,
drill holes within `width` of the plane are clipped to that slab and projected onto it. `section_widget` puts a
plane in the window that recuts the scene each time it moves, and `view_section` turns the camera to look straight
at the plane in parallel projection, as a true-scale section. here each view renders off-screen to an image.

<details><summary>Python</summary>

```python
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from common import INK, save

pv.OFF_SCREEN = True
BAR = {"vertical": True, "height": 0.5, "position_x": 0.85, "position_y": 0.25}


def image(scene, title):
    """Renders a scene into a matplotlib figure."""
    pixels = scene.plotter.screenshot(return_img=True, window_size=(1400, 900))
    fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
    ax.imshow(pixels)
    ax.set_axis_off()
    ax.set_title(title)
    return fig
```

</details>

zinc composites of the stacked sulphide lenses inform a grid rotated with them. the scene holds the grid, the drill
holes and the three lens surfaces; one section across the lenses, at azimuth 112.5 and dipping 80, cuts them all.
holes within 15 m of the plane draw on it, the rest drop out.

<details><summary>Python</summary>

```python
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes(data["collars"], data["surveys"], data["assays"])
lenses = [data[f"lens_{i}"] for i in (1, 2, 3)]
composites = holes.composite(2.0, ["ZN_PCT"])
grid = bt.BlockModel.from_extents(*lenses, size=(20, 20, 10), buffer=20, rotation=(22.5, 0.0, 55.0))
search = bt.Search(radius=60, min_samples=1, max_samples=12)
idw = bt.InverseDistance(search, power=2).fit(composites.coords, composites["ZN_PCT"])
grid = grid.with_column("ZN_PCT", idw.predict(grid))

scene = bt.plot3d.Scene(window_size=(1400, 900))
scene.add(grid, "ZN_PCT", clim=(0, 10), scalar_bar_args={"title": "Zn (%)", **BAR})
scene.add(holes, color=INK, radius=2)
for lens in lenses:
    scene.add(lens, color="white")
center = grid.centroids.mean(axis=0)
scene.section(center, azimuth=112.5, dip=80, width=30)
print(f"section through {np.round(center).tolist()}")
scene.view_vector((0.1, -0.6, 0.8))
save(image(scene, "One plane cuts the grid, the lenses and the holes near it"), "cut")
```

</details>

```text
section through [12231.0, 29911.0, 48.0]
```

![cut](cut.png)

`view_section` looks along the normal of the plane with strike to the right and up dip upward. `section_widget`
would let the plane be dragged and turned in a window; each move calls `section` with the new origin and angles.

<details><summary>Python</summary>

```python
scene.view_section()
save(image(scene, "The same section, seen normal to the plane"), "section")
scene.close()
```

</details>

![section](section.png)

Full script: [`example_02_24.py`](example_02_24.py)
