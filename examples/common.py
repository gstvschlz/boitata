"""Figure style shared by the examples."""

import inspect
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from ceres.datasets import fetch  # noqa: F401

ROOT = Path(__file__).resolve().parent

INK = "#222222"
GREY = "#8c8c8c"
LIGHT = "#d9d9d9"
ACCENT = "#1f4e79"
HIGHLIGHT = "#c05a28"

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
        "image.cmap": "cividis",
    }
)


def map_axes(ax, title):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")


SAVED: list[Path] = []


def save(fig, name: str):
    """Writes `name`.png next to the calling script; in the docs gallery the figure is left for its scraper."""
    if "mkdocs_gallery" in sys.modules:
        return
    caller = inspect.currentframe().f_back.f_globals["__file__"]
    path = Path(caller).parent / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    SAVED.append(path)
