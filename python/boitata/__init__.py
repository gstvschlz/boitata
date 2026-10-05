import copyreg as _copyreg

from boitata import _boitata, datasets, plot, plot3d  # noqa: F401
from boitata._boitata import *
from boitata._boitata import __version__  # noqa: F401
from boitata._units import UnitArray, units  # noqa: F401
from boitata.checks import *
from boitata.compositions import *
from boitata.errors import BoitataError, FileError, InvalidInput, MissingColumn  # noqa: F401
from boitata.estimation import *
from boitata.geopackage import *
from boitata.pipeline import *
from boitata.planning import *
from boitata.spacing import *
from boitata.spatial import *
from boitata.steps import *

__all__ = [
    name
    for name in dir()
    if not name.startswith("_")
    and name
    not in (
        "checks",
        "compositions",
        "errors",
        "estimation",
        "geopackage",
        "pipeline",
        "spacing",
        "steps",
        "datasets",
        "plot",
        "plot3d",
    )
]


def _from_json(cls, text):
    return cls.from_json(text)


def _from_state(cls, meta, columns):
    return cls._from_state(meta, columns)


def _reduce(obj):
    if hasattr(obj, "_state"):
        return _from_state, (type(obj), *obj._state())
    return _from_json, (type(obj), obj.to_json())


for _cls in vars(_boitata).values():
    if isinstance(_cls, type) and (hasattr(_cls, "from_json") or hasattr(_cls, "_from_state")):
        _copyreg.pickle(_cls, _reduce)
