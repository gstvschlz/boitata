"""Log-ratio transforms of compositions, on arrays or on the part columns of a container."""

import numpy as np

import boitata
from boitata._columns import column, names
from boitata.errors import InvalidInput
from boitata.steps import _Step

__all__ = ["ALR", "CLR", "ILR"]


class _LogRatio(_Step):
    shrink = 1

    def __init__(self, *, parts=None, total=1.0):
        if parts is not None and len(parts) < 2:
            raise InvalidInput("a composition needs at least 2 parts")
        self.parts, self.total = None if parts is None else list(parts), total

    def fit(self, data):
        if self.parts is not None:
            for p in self.parts:
                column(data, p)
        return self

    def transform(self, data):
        """Coordinates of each composition; rows with a missing part give null coordinates.

        On a container, the coordinate columns replace the part columns.
        """
        if self.parts is None:
            x = np.asarray(data, dtype=float)
            return _rows(self._forward, x, x.shape[1] - self.shrink)
        x = np.column_stack([column(data, p) for p in self.parts]).astype(float)
        out = _rows(self._forward, x, len(self._names()))
        return _replace(data, self.parts, dict(zip(self._names(), out.T, strict=True)))

    def inverse_transform(self, data):
        """Compositions closed to `total`; on a container, the part columns replace the coordinate columns."""
        if self.parts is None:
            y = np.asarray(data, dtype=float)
            return self.total * _rows(self._back, y, y.shape[1] + self.shrink)
        y = np.column_stack([column(data, c) for c in self._names()]).astype(float)
        out = self.total * _rows(self._back, y, len(self.parts))
        return _replace(data, self._names(), dict(zip(self.parts, out.T, strict=True)))


class CLR(_LogRatio):
    """Centered log-ratio: ``ln(x_i / g(x))``, one coordinate per part, summing to 0.

    Parameters
    ----------
    parts : list of str, optional
        Part columns of a container; without them `transform` takes an ``(n, D)`` array.
    total : float, default 1.0
        Sum of the compositions `inverse_transform` returns, e.g. 100 for percentages.
    """

    shrink = 0

    def _names(self):
        return [f"clr_{p}" for p in self.parts]

    def _forward(self, x):
        return boitata.clr(x)

    def _back(self, y):
        return boitata.clr_inverse(y)


class ALR(_LogRatio):
    """Additive log-ratio ``ln(x_i / x_r)`` against the reference part ``r``: ``D - 1`` coordinates.

    Parameters
    ----------
    parts : list of str, optional
        Part columns of a container; without them `transform` takes an ``(n, D)`` array.
    reference : str or int, optional
        Reference part, by name or position; default the last.
    total : float, default 1.0
        Sum of the compositions `inverse_transform` returns.
    """

    def __init__(self, *, parts=None, reference=None, total=1.0):
        super().__init__(parts=parts, total=total)
        if isinstance(reference, str):
            if self.parts is None or reference not in self.parts:
                raise InvalidInput(f"reference {reference!r} is not one of the parts")
            reference = self.parts.index(reference)
        self.reference = reference

    def _names(self):
        r = len(self.parts) - 1 if self.reference is None else self.reference
        return [f"alr_{p}" for i, p in enumerate(self.parts) if i != r]

    def _forward(self, x):
        return boitata.alr(x, reference=self.reference)

    def _back(self, y):
        return boitata.alr_inverse(y, reference=self.reference)


class ILR(_LogRatio):
    """Isometric log-ratio: ``D - 1`` orthonormal balances, so distances are Aitchison distances.

    Parameters
    ----------
    parts : list of str, optional
        Part columns of a container; without them `transform` takes an ``(n, D)`` array.
    basis : array_like of int, optional
        Sequential binary partition, ``(D - 1, D)`` signs as in `partition_basis`; default the pivot balances,
        the first part against the second, the first two against the third, and so on.
    total : float, default 1.0
        Sum of the compositions `inverse_transform` returns.
    """

    def __init__(self, *, parts=None, basis=None, total=1.0):
        super().__init__(parts=parts, total=total)
        if basis is not None:
            boitata.partition_basis(basis)
            basis = np.asarray(basis, dtype=int).tolist()
        self.basis = basis

    def _names(self):
        return [f"ilr_{i + 1}" for i in range(len(self.parts) - 1)]

    def _forward(self, x):
        return boitata.ilr(x, basis=self._basis())

    def _back(self, y):
        return boitata.ilr_inverse(y, basis=self._basis())

    def _basis(self):
        return None if self.basis is None else boitata.partition_basis(self.basis)


def _rows(f, x, width):
    """`f` on the complete rows of `x`; NaN rows elsewhere."""
    out = np.full((len(x), width), np.nan)
    ok = np.isfinite(x).all(axis=1)
    if ok.any():
        out[ok] = f(x[ok])
    return out


def _replace(data, drop, add):
    """`data` with the columns `drop` replaced by `add`, at the place of the first dropped column."""
    table, added = {}, False
    for c in names(data):
        if c in drop:
            if not added:
                table.update(add)
                added = True
        elif c not in add:
            table[c] = column(data, c)
    if hasattr(data, "with_attributes"):
        return data.with_attributes(boitata.Table(table))
    return boitata.Table(table) if isinstance(data, boitata.Table) else table
