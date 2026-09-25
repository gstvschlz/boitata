"""Dataset download and figure style shared by the examples."""

import inspect
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
DATASETS = "https://raw.githubusercontent.com/gstvschlz/datasets/560bd39c9ec79dcd6565dc763009aa83cac68fb8"

INK = "#222222"
GREY = "#8c8c8c"
LIGHT = "#d9d9d9"
ACCENT = "#1f4e79"
HIGHLIGHT = "#c05a28"
CMAP = LinearSegmentedColormap.from_list("ceres", ["#f7f7f7", "#9ebad6", ACCENT, "#0b1f33"])
mpl.colormaps.register(CMAP, force=True)

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


def fetch(name: str) -> Path:
    """Local copy of `name` from the datasets repository, downloaded once."""
    path = ROOT / "data" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{DATASETS}/{name}", path)
    return path


def map_axes(ax, title):
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")


SAVED: list[Path] = []


def save(fig, name: str):
    """Writes `name`.png next to the calling script."""
    caller = inspect.currentframe().f_back.f_globals["__file__"]
    path = Path(caller).parent / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    SAVED.append(path)
