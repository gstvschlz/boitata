"""A chain of transforms over the columns of a container."""

import inspect
import json
from functools import partial

import boitata
from boitata._columns import names
from boitata.errors import InvalidInput, MissingColumn

__all__ = ["Pipeline"]

FORMAT = 1


class Pipeline:
    """Transforms applied in turn to columns of a container, each replacing the columns it reads.

    Parameters
    ----------
    steps : sequence of (str, transform, str or sequence of str)
        Name, transform and columns of each step. One column name calls the transform on that column
        (``NormalScore``, ``Capping``, ``BoxCox``, ...); a list of names calls it on those columns together
        (``PPMT``). Names must be distinct.

    Examples
    --------
    >>> pipe = bt.Pipeline([("cap", bt.Capping(cap=40.0), "au"), ("ns", bt.NormalScore(), "au")])
    >>> scores = pipe.fit_transform(composites, weights="w")
    >>> grades = pipe.inverse_transform(simulated)
    """

    def __init__(self, steps):
        self.steps = [(name, step, columns) for name, step, columns in steps]
        if len({name for name, _, _ in self.steps}) < len(self.steps):
            raise InvalidInput("step names must be distinct")
        self.columns_ = None

    @property
    def named_steps(self) -> dict:
        return {name: step for name, step, _ in self.steps}

    def fit(self, data, *, weights=None):
        """Fits each step, with its `fit_transform`, on the output of the steps before it.

        Parameters
        ----------
        data : PointSet, BlockModel, Polylines or mapping
        weights : array_like or str, optional
            Declustering weights, or their column, passed to the steps whose `fit` takes weights.
        """
        self.fit_transform(data, weights=weights)
        return self

    def fit_transform(self, data, *, weights=None):
        """`fit`, returning the transformed container."""
        read = []
        for _, _, columns in self.steps:
            read += [c for c in _list(columns) if c not in read]
        _check(data, read)
        for _, step, columns in self.steps:
            extra = (
                {"weights": weights} if weights is not None and _takes(step.fit_transform, "weights") else {}
            )
            data = _apply(partial(step.fit_transform, **extra), data, columns)
        self.columns_ = read
        return data

    def transform(self, data):
        """Applies the fitted steps in order.

        Raises
        ------
        MissingColumn
            If `data` lacks a column the pipeline was fitted on.
        """
        _check(data, self._fitted())
        for _, step, columns in self.steps:
            data = _apply(step.transform, data, columns)
        return data

    def inverse_transform(self, data):
        """Undoes the invertible steps in reverse order; steps without an inverse, such as capping, are skipped."""
        _check(data, self._fitted())
        for _, step, columns in reversed(self.steps):
            if hasattr(step, "inverse_transform"):
                data = _apply(step.inverse_transform, data, columns)
        return data

    def to_json(self) -> str:
        """JSON of the steps and, once fitted, their fitted state."""
        steps = [{"name": n, "columns": c, "step": json.loads(s.to_json())} for n, s, c in self.steps]
        return json.dumps({"type": "Pipeline", "format": FORMAT, "columns": self.columns_, "steps": steps})

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
        pipe.columns_ = meta["columns"]
        return pipe

    def _fitted(self):
        if self.columns_ is None:
            raise InvalidInput("Pipeline is not fitted; call fit first")
        return self.columns_

    def __repr__(self):
        return f"Pipeline({[(n, type(s).__name__, c) for n, s, c in self.steps]})"


def _list(columns):
    return [columns] if isinstance(columns, str) else list(columns)


def _takes(method, arg):
    try:
        return arg in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _check(data, columns):
    have = names(data)
    missing = [c for c in columns if c not in have]
    if missing:
        raise MissingColumn(
            f"no column {', '.join(map(repr, missing))}; the pipeline was fitted on {', '.join(columns)}"
        )


def _apply(method, data, columns):
    if isinstance(columns, str):
        return _with(data, {columns: method(columns, data=data)})
    out = method(data, columns=list(columns))
    return _with(data, {c: out[:, j] for j, c in enumerate(columns)})


def _with(data, columns):
    if hasattr(data, "with_column"):
        for name, values in columns.items():
            data = data.with_column(name, values)
        return data
    if hasattr(data, "keys"):
        return {**data, **columns}
    raise InvalidInput(
        f"Pipeline needs a PointSet, BlockModel, Polylines or mapping, not {type(data).__name__}"
    )
