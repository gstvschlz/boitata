from contextlib import contextmanager

from boitata import _boitata

__all__ = ["units"]


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
