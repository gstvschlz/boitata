"""Cleaning steps on the columns and rows of a container, usable alone or in a `Pipeline`."""

import json
import re

import numpy as np

import boitata
from boitata._columns import column, names
from boitata.errors import InvalidInput

__all__ = [
    "Clip",
    "DropDuplicates",
    "DropNull",
    "FillNull",
    "Log",
    "Log10",
    "NormalizeText",
    "RenameColumns",
    "Replace",
    "Scale",
    "ToNull",
    "ToNumber",
]

FORMAT = 1


class _Step:
    """`fit` learns nothing unless a step says otherwise; `transform` returns a new container."""

    def fit(self, data):
        return self

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def to_json(self) -> str:
        return json.dumps({"type": type(self).__name__, "format": FORMAT, **vars(self)})

    @classmethod
    def from_json(cls, text: str):
        meta = json.loads(text)
        if meta.pop("type", None) != cls.__name__:
            raise InvalidInput(f"expected a {cls.__name__} JSON")
        if meta.pop("format", 0) > FORMAT:
            raise InvalidInput(f"format is newer than this boitata reads (up to {FORMAT}); upgrade boitata")
        step = cls.__new__(cls)
        vars(step).update(meta)
        return step

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in vars(self).items())})"

    def _targets(self, data, numeric=None):
        """The step's columns, or every column (of the given kind when `numeric` is set)."""
        if self.columns is not None:
            return [self.columns] if isinstance(self.columns, str) else list(self.columns)
        return [c for c in names(data) if numeric is None or _is_text(column(data, c)) != numeric]

    def _map(self, data, f, numeric=None):
        return _set(data, {c: f(_values(data, c)) for c in self._targets(data, numeric)})


class RenameColumns(_Step):
    """Renames columns: `mapping` first, then `strip` and `case` on the names it leaves.

    Parameters
    ----------
    mapping : dict, optional
        Old name to new name.
    case : {"lower", "upper", "snake"}, optional
        ``snake`` lowercases and joins words with underscores: ``"AuPpm"`` and ``"Au ppm"`` become ``"au_ppm"``.
    strip : bool, default False
        Remove leading and trailing whitespace.
    """

    def __init__(self, *, mapping=None, case=None, strip=False):
        if case not in (None, "lower", "upper", "snake"):
            raise InvalidInput("case must be 'lower', 'upper' or 'snake'")
        self.mapping, self.case, self.strip, self.names_ = dict(mapping or {}), case, strip, None

    def _rename(self, name):
        if name in self.mapping:
            return self.mapping[name]
        name = name.strip() if self.strip else name
        if self.case == "snake":
            name = re.sub(r"[^0-9a-z]+", "_", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()).strip("_")
        return {"lower": str.lower, "upper": str.upper}.get(self.case, str)(name)

    def fit(self, data):
        _check(data, self.mapping)
        renamed = {c: self._rename(c) for c in names(data)}
        clash = sorted({n for n in renamed.values() if list(renamed.values()).count(n) > 1})
        if clash:
            raise InvalidInput(f"renaming gives repeated column names: {', '.join(clash)}")
        self.names_ = renamed
        return self

    def transform(self, data):
        return _rename(data, self._fitted())

    def inverse_transform(self, data):
        return _rename(data, {new: old for old, new in self._fitted().items()})

    def _fitted(self):
        if self.names_ is None:
            raise InvalidInput("RenameColumns is not fitted; call fit first")
        return self.names_


class Clip(_Step):
    """Limits numeric columns to ``[lower, upper]``; nulls stay null."""

    def __init__(self, *, columns=None, lower=None, upper=None):
        self.columns, self.lower, self.upper = columns, lower, upper

    def transform(self, data):
        return self._map(data, lambda v: np.clip(v, self.lower, self.upper), numeric=True)


class FillNull(_Step):
    """Replaces nulls in `columns` (default all) with `value`."""

    def __init__(self, value, *, columns=None):
        self.value, self.columns = value, columns

    def transform(self, data):
        return self._map(data, lambda v: np.where(_null(v), self.value, v))


class Replace(_Step):
    """Replaces values equal to a key of `mapping` with its value, in `columns` (default all)."""

    def __init__(self, mapping, *, columns=None):
        self.mapping, self.columns = dict(mapping), columns

    def to_json(self) -> str:
        if not all(isinstance(k, str) for k in self.mapping):
            raise InvalidInput("Replace saves to JSON only with text keys")
        return super().to_json()

    def transform(self, data):
        def replace(v):
            out = v.copy() if _is_text(v) else v.astype(object)
            for old, new in self.mapping.items():
                out[v == old] = new
            return out if _is_text(v) else _numeric(out)

        return self._map(data, replace)


class ToNull(_Step):
    """Sets values equal to any of `sentinels` (e.g. ``-99``, ``"N/A"``) to null, in `columns` (default all)."""

    def __init__(self, sentinels, *, columns=None):
        self.sentinels = list(sentinels) if isinstance(sentinels, (list, tuple)) else [sentinels]
        self.columns = columns

    def transform(self, data):
        def to_null(v):
            hit = np.isin(v, [s for s in self.sentinels if isinstance(s, str) == _is_text(v)])
            return np.where(hit, None, v) if _is_text(v) else np.where(hit, np.nan, v)

        return self._map(data, to_null)


class ToNumber(_Step):
    """Parses text `columns` as numbers; nulls stay null. Raises on text that is not a number, such as ``"<0.01"``:
    replace it first with `Replace` or `ToNull`."""

    def __init__(self, columns):
        self.columns = columns

    def transform(self, data):
        def parse(v):
            if not _is_text(v):
                return v
            bad = sorted({s for s in v if s is not None and not _parses(s)})
            if bad:
                raise InvalidInput(f"not numbers: {', '.join(map(repr, bad[:5]))}")
            return _numeric([None if s is None else float(s) for s in v])

        return self._map(data, parse)


class NormalizeText(_Step):
    """Strips and recases the values of text columns, so ``"Ox "`` and ``"OX"`` become one category.

    Parameters
    ----------
    columns : str or list of str, optional
        Default every text column.
    case : {"lower", "upper"}, optional
    strip : bool, default True
    """

    def __init__(self, *, columns=None, case=None, strip=True):
        if case not in (None, "lower", "upper"):
            raise InvalidInput("case must be 'lower' or 'upper'")
        self.columns, self.case, self.strip = columns, case, strip

    def transform(self, data):
        def normalize(s):
            s = s.strip() if self.strip else s
            return s.lower() if self.case == "lower" else s.upper() if self.case == "upper" else s

        return self._map(
            data, lambda v: np.array([None if s is None else normalize(s) for s in v], object), False
        )


class DropNull(_Step):
    """Drops rows with a null in any of `columns` (default all)."""

    def __init__(self, *, columns=None):
        self.columns = columns

    def transform(self, data):
        null = np.zeros(_rows(data), bool)
        for c in self._targets(data):
            null |= _null(_values(data, c))
        return _filter(data, ~null)


class DropDuplicates(_Step):
    """Keeps the first sample of each group of samples at most `tolerance` apart (see `duplicates`)."""

    def __init__(self, *, tolerance=0.0):
        self.tolerance = tolerance

    def transform(self, data):
        coords = data.coords if hasattr(data, "coords") else data
        _, group = boitata.duplicates(coords, tolerance=self.tolerance)
        first = np.unique(group, return_index=True)[1]
        keep = group == -1
        keep[first] = True
        return _filter(data, keep)


class Scale(_Step):
    """Multiplies numeric `columns` by `factor`, e.g. ``0.001`` for ppb to ppm."""

    def __init__(self, columns, factor):
        if not factor:
            raise InvalidInput("factor must be nonzero")
        self.columns, self.factor = columns, factor

    def transform(self, data):
        return self._map(data, lambda v: v * self.factor)

    def inverse_transform(self, data):
        return self._map(data, lambda v: v / self.factor)


class Log(_Step):
    """Natural logarithm of numeric `columns`; null where a value is not positive."""

    base = np.e

    def __init__(self, columns):
        self.columns = columns

    def transform(self, data):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._map(data, lambda v: np.where(v > 0, np.log(v) / np.log(self.base), np.nan))

    def inverse_transform(self, data):
        return self._map(data, lambda v: np.power(self.base, v))


class Log10(Log):
    """Base-10 logarithm of numeric `columns`; null where a value is not positive."""

    base = 10.0


def _is_text(values):
    return np.asarray(values).dtype == object


def _parses(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def _numeric(values):
    return np.array([np.nan if v is None else v for v in values], float)


def _null(values):
    values = np.asarray(values)
    return np.array([v is None for v in values]) if _is_text(values) else np.isnan(values.astype(float))


def _values(data, name):
    values = np.asarray(column(data, name))
    return values if _is_text(values) else values.astype(float)


def _rows(data):
    return (
        len(data)
        if hasattr(data, "__len__") and not isinstance(data, dict)
        else len(_values(data, names(data)[0]))
    )


def _check(data, columns):
    for c in columns:
        column(data, c)


def _set(data, columns):
    if hasattr(data, "with_columns"):
        return data.with_columns(columns)
    if hasattr(data, "with_column"):
        for name, values in columns.items():
            data = data.with_column(name, values)
        return data
    if isinstance(data, boitata.Table):
        return boitata.Table({**{c: data[c] for c in data.column_names}, **columns})
    return {**data, **columns}


def _rename(data, mapping):
    table = {mapping.get(c, c): column(data, c) for c in names(data)}
    if hasattr(data, "with_attributes"):
        return data.with_attributes(boitata.Table(table))
    return boitata.Table(table) if isinstance(data, boitata.Table) else table


def _filter(data, keep):
    if hasattr(data, "filter"):
        return data.filter(keep)
    if isinstance(data, dict):
        return {k: np.asarray(v)[keep] for k, v in data.items()}
    raise InvalidInput(f"cannot drop rows of a {type(data).__name__}")
