"""A chain of transforms over the columns of a container."""

import inspect
import json
from functools import partial

import boitata
from boitata._columns import column
from boitata.errors import InvalidInput, MissingColumn
from boitata.steps import _set

__all__ = ["Pipeline"]

FORMAT = 1


class Pipeline:
    """Transforms applied in turn to a container, each on the output of the one before.

    Parameters
    ----------
    steps : sequence of (str, transform) or (str, transform, str or sequence of str)
        Name, transform and, for transforms on arrays, their columns. One column name calls the transform on that
        column (``NormalScore``, ``Capping``, ``BoxCox``, ...) and a list on those columns together (``PPMT``);
        either way the output replaces the columns read. A step without columns (``RenameColumns``, ``ToNull``,
        ``DropNull``, ...) takes and returns the whole container. Names must be distinct.

    Examples
    --------
    >>> pipe = bt.Pipeline([
    ...     ("names", bt.RenameColumns(case="snake")),
    ...     ("nulls", bt.ToNull([-99, -999])),
    ...     ("cap", bt.Capping(cap=40.0), "au"),
    ...     ("ns", bt.NormalScore(), "au"),
    ... ])
    >>> scores = pipe.fit_transform(composites, weights="w")
    >>> grades = pipe.inverse_transform(simulated)
    """

    def __init__(self, steps):
        self.steps = [(s[0], s[1], s[2] if len(s) > 2 else None) for s in steps]
        if len({name for name, _, _ in self.steps}) < len(self.steps):
            raise InvalidInput("step names must be distinct")
        self.fitted_ = False

    @property
    def named_steps(self) -> dict:
        return {name: step for name, step, _ in self.steps}

    def fit(self, data, *, weights=None):
        """Fits each step, with its `fit_transform`, on the output of the steps before it.

        Parameters
        ----------
        data : PointSet, BlockModel, Polylines, Table or mapping
        weights : array_like or str, optional
            Declustering weights, or their column, passed to the steps on columns whose `fit` takes weights.
        """
        self.fit_transform(data, weights=weights)
        return self

    def fit_transform(self, data, *, weights=None):
        """`fit`, returning the transformed container."""
        for name, step, columns in self.steps:
            weighted = weights is not None and columns is not None and _takes(step.fit_transform, "weights")
            method = partial(step.fit_transform, weights=weights) if weighted else step.fit_transform
            data = _apply(name, method, data, columns)
        self.fitted_ = True
        return data

    def transform(self, data):
        """Applies the fitted steps in order.

        Raises
        ------
        MissingColumn
            If `data` lacks a column a step reads.
        """
        self._fitted()
        for name, step, columns in self.steps:
            data = _apply(name, step.transform, data, columns)
        return data

    def inverse_transform(self, data):
        """Undoes the invertible steps in reverse order; steps without an inverse, such as capping, are skipped."""
        self._fitted()
        for name, step, columns in reversed(self.steps):
            if hasattr(step, "inverse_transform"):
                data = _apply(name, step.inverse_transform, data, columns)
        return data

    def to_json(self) -> str:
        """JSON of the steps and, once fitted, their fitted state."""
        steps = [{"name": n, "columns": c, "step": json.loads(s.to_json())} for n, s, c in self.steps]
        return json.dumps({"type": "Pipeline", "format": FORMAT, "fitted": self.fitted_, "steps": steps})

    @staticmethod
    def from_json(text: str) -> "Pipeline":
        """Reads `to_json` output; raises InvalidInput on another class's JSON or a newer format."""
        meta = json.loads(text)
        if meta.get("type") != "Pipeline":
            raise InvalidInput(f"expected a Pipeline, found {meta.get('type')}")
        if meta.get("format", 0) > FORMAT:
            raise InvalidInput(
                f"format {meta['format']} is newer than this boitata reads (up to {FORMAT}); upgrade boitata"
            )
        steps = []
        for s in meta["steps"]:
            cls = getattr(boitata, s["step"]["type"])
            steps.append((s["name"], cls.from_json(json.dumps(s["step"])), s["columns"]))
        pipe = Pipeline(steps)
        pipe.fitted_ = meta["fitted"]
        return pipe

    def _fitted(self):
        if not self.fitted_:
            raise InvalidInput("Pipeline is not fitted; call fit first")

    def __repr__(self):
        return f"Pipeline({[(n, type(s).__name__, c) for n, s, c in self.steps]})"


def _takes(method, arg):
    try:
        return arg in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _apply(name, method, data, columns):
    if columns is None:
        return method(data)
    for c in [columns] if isinstance(columns, str) else columns:
        try:
            column(data, c)
        except MissingColumn as e:
            raise MissingColumn(f"step {name!r}: {e.args[0]}") from None
    if isinstance(columns, str):
        return _set(data, {columns: method(columns, data=data)})
    out = method(data, columns=list(columns))
    return _set(data, {c: out[:, j] for j, c in enumerate(columns)})
