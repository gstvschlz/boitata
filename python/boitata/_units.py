from contextlib import contextmanager

import numpy as np

from boitata import _boitata

__all__ = ["UnitArray", "units"]


class UnitArray(np.ndarray):
    """A numpy array with the unit of the column or result it holds.

    Indexing and slicing keep `unit`; arithmetic and numpy functions return a plain array, so a unit is never
    carried into a value it no longer describes. `with_column` stores the unit with the column.
    """

    def __new__(cls, values, unit):
        array = np.asarray(values).view(cls)
        array.unit = unit
        return array

    def __array_finalize__(self, obj):
        self.unit = getattr(obj, "unit", None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if out is not None:
            kwargs["out"] = tuple(_plain(o) for o in out)
        return getattr(ufunc, method)(*(_plain(x) for x in inputs), **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return func(*_plain(args), **_plain(kwargs))

    def __reduce__(self):
        return UnitArray, (np.asarray(self), self.unit)

    def __repr__(self):
        return f"{np.asarray(self)!r} [{self.unit}]"


def _plain(x):
    if isinstance(x, UnitArray):
        return x.view(np.ndarray)
    if isinstance(x, tuple | list):
        return type(x)(_plain(v) for v in x)
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return x


def unit_of(values, data=None):
    """The unit `values` carries, or the unit of the column of `data` it names."""
    if isinstance(values, str):
        return getattr(data, "units", {}).get(values) if data is not None else None
    unit = getattr(values, "unit", None)
    return unit if isinstance(unit, str) else None


def tag(values, unit):
    return values if unit is None else UnitArray(values, unit)


def squared(unit):
    return None if unit is None else f"({unit})^2"


@contextmanager
def units(*, columns=None):
    """Project units for a ``with`` block, restored on exit.

    Parameters
    ----------
    columns : dict of str to str or None, optional
        Column name to unit, as in `set_units`.

    Examples
    --------
    >>> import boitata as bt
    >>> with bt.units(columns={"au": "g/t"}):
    ...     bt.Table({"au": [1.0]}).units
    {'au': 'g/t'}
    """
    saved = _boitata._unit_defaults()
    try:
        _boitata.set_units(columns=columns)
        yield
    finally:
        _boitata._restore_unit_defaults(saved)
