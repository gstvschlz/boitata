"""Shared figure style: greys, one accent, one highlight."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "out"

INK = "#222222"
GREY = "#8c8c8c"
LIGHT = "#d9d9d9"
ACCENT = "#1f4e79"
HIGHLIGHT = "#c05a28"
CMAP = LinearSegmentedColormap.from_list("ceres", ["#f7f7f7", "#9ebad6", ACCENT, "#0b1f33"])

mpl.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK,
        "axes.edgecolor": GREY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=[ACCENT, GREY, HIGHLIGHT]),
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "image.cmap": "ceres",
    }
)
mpl.colormaps.register(CMAP, force=True)


def table(path):
    """CSV with a header row as a dict of float arrays."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.atleast_1d(data[name]) for name in data.dtype.names}


def grid(path, value="V", shape=(300, 260)):
    """Exhaustive Walker Lake grid as a (ny, nx) array."""
    return table(path)[value].reshape(shape)


def map_axes(ax, title):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")


def save(fig, chapter, name):
    path = ROOT / chapter / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
