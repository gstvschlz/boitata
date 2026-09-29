"""Checks that simulated realizations reproduce the histogram, variogram and correlations of their data."""

from dataclasses import dataclass

import numpy as np

from ceres._ceres import (
    NormalScore,
    Table,
    _realization_variograms,
    correlation,
    describe,
    experimental_variogram,
)
from ceres._columns import column
from ceres.errors import InvalidInput

__all__ = ["RealizationCheck", "check_realizations"]


@dataclass(frozen=True)
class RealizationCheck:
    """What `check_realizations` measured, per variable ``v`` (category for categorical realizations),
    realization ``r`` and direction ``d``.

    Attributes
    ----------
    names : list of str
        Variable names, or category codes as text.
    statistics : Table
        One row per variable and realization, realization 0 being the declustered data: ``variable``,
        ``realization`` and the statistics of `describe`, or ``proportion`` for categories.
    probabilities : ndarray, shape (p,)
        Cumulative probabilities at which quantiles are taken.
    quantiles, score_quantiles : ndarray, shape (v, r, p), or None
        Quantiles of each realization, in data units and in the normal scores of the declustered data, whose
        target is ``normal_ppf(probabilities)``. None for categories.
    data_quantiles : ndarray, shape (v, p), or None
        Declustered quantiles of the data.
    proportions : ndarray, shape (r, v), or None
        Share of the nodes in each category, per realization; None for continuous variables.
    data_proportions : ndarray, shape (v,), or None
        Declustered proportions of the data.
    directions : list of (float, float), or None
        ``(azimuth, dip)`` of each variogram direction; None when omnidirectional.
    variograms : list, or None
        ``variograms[v][r][d]``, the ExperimentalVariogram of realization ``r``; None without `lag`.
    data_variograms : list, or None
        ``data_variograms[v][d]``, the same for the data.
    models : list of Variogram or None
        Model each variable was simulated with, when given.
    correlations : ndarray, shape (r, v, v), or None
        Correlation matrix of each realization, with several variables.
    data_correlation : ndarray, shape (v, v), or None
        Declustered correlation matrix of the data.
    """

    names: list
    statistics: Table
    probabilities: np.ndarray
    quantiles: np.ndarray | None
    score_quantiles: np.ndarray | None
    data_quantiles: np.ndarray | None
    proportions: np.ndarray | None
    data_proportions: np.ndarray | None
    directions: list | None
    variograms: list | None
    data_variograms: list | None
    models: list | None
    correlations: np.ndarray | None
    data_correlation: np.ndarray | None

    @property
    def categorical(self) -> bool:
        return self.proportions is not None


def _stack(realizations):
    if hasattr(realizations, "realizations") or np.ndim(realizations) == 2:
        realizations = [realizations]
    out = []
    for r in realizations:
        r = getattr(r, "realizations", r)
        if r is None:
            raise InvalidInput("the summary holds no realizations; simulate with keep=")
        r = np.asarray(r)
        if r.ndim != 2:
            raise InvalidInput("realizations must be (n, targets) arrays")
        out.append(r)
    return out


def _quantiles(values, weights, p):
    s = describe(values, weights=weights, quantiles=p)
    return np.array([v for k, v in s.items() if k.startswith("P")])


def check_realizations(
    model,
    realizations,
    data,
    values,
    *,
    weights=None,
    variogram=None,
    lag=None,
    max_lag=None,
    directions=None,
    tolerance=22.5,
    normal_scores=True,
) -> RealizationCheck:
    """How well realizations on `model` reproduce the histogram, variogram and correlations of their data.

    Each realization's quantiles are compared with the declustered quantiles of the data, in data units and in
    the data's normal scores; its experimental variogram along `directions` with the data's and with the model it
    was drawn from; with several variables, its correlation matrix with the data's. Categorical realizations
    compare category proportions and indicator variograms instead. Realization variograms on a regular or masked
    BlockModel pair cells by index shifts, in parallel over realizations, identically on any number of threads.

    Parameters
    ----------
    model : BlockModel
        The nodes the realizations were simulated at.
    realizations : SimulationSummary, CategoricalSummary, list of SimulationSummary or array_like
        Summaries simulated with ``keep=`` (a list from `MultivariateSimulation`), or ``(n, targets)``
        arrays, one per variable; integer arrays are categories ``0..k``.
    data : PointSet or array_like, shape (m, 2) or (m, 3)
        Conditioning data, or their coordinates.
    values : str, array_like or list of them
        Data values, one per variable, or the columns of `data` holding them; NaN is dropped.
    weights : str or array_like, optional
        Declustering weights of the data.
    variogram : Variogram or list of Variogram, optional
        Model of each variable (of each category's indicator for categories), in normal scores when
        `normal_scores`.
    lag, max_lag : float, optional
        Lag width and largest lag of the variograms; without them no variograms are computed.
    directions : list of (float, float), optional
        ``(azimuth, dip)`` of each direction. Default: along the first `variogram`'s azimuth and dip, across it
        horizontally and, on a 3D grid, vertically; omnidirectional without `variogram`.
    tolerance : float
        Direction half-angle in degrees.
    normal_scores : bool
        Variograms of continuous variables in the normal scores of the declustered data, the units of Gaussian
        simulation models; else in data units.

    Returns
    -------
    RealizationCheck
    """
    reals = _stack(realizations)
    single = isinstance(values, str) or (np.ndim(values[0]) == 0 and not isinstance(values[0], str))
    values = [values] if single else list(values)
    if len(values) != len(reals):
        raise InvalidInput(f"{len(reals)} sets of realizations need as many values; got {len(values)}")
    columns = [np.asarray(column(data, v), dtype=float) for v in values]
    n = len(columns[0])
    w = np.ones(n) if weights is None else np.asarray(column(data, weights, "weights"), dtype=float)
    if any(len(c) != n for c in columns) or len(w) != n:
        raise InvalidInput("values and weights need one entry per sample")
    names = [v if isinstance(v, str) else str(j) for j, v in enumerate(values)]
    coords = getattr(data, "coords", None)
    if coords is None and np.ndim(data) == 2:
        coords = np.asarray(data, dtype=float)
    categorical = all(np.issubdtype(r.dtype, np.integer) for r in reals)
    p = np.round(np.linspace(0.01, 0.99, 99), 2)

    quantiles = score_quantiles = data_quantiles = proportions = data_proportions = None
    correlations = data_correlation = None
    if categorical:
        if len(reals) != 1:
            raise InvalidInput("check one set of categorical realizations at a time")
        codes, ok = columns[0], np.isfinite(columns[0])
        k = int(max(reals[0].max(), codes[ok].max())) + 1
        proportions = np.array([np.bincount(r, minlength=k) / r.size for r in reals[0]])
        data_proportions = np.bincount(codes[ok].astype(int), weights=w[ok], minlength=k) / w[ok].sum()
        names = [str(c) for c in range(k)]
        rows = [
            {"variable": name, "realization": r, "proportion": share}
            for c, name in enumerate(names)
            for r, share in enumerate([data_proportions[c], *proportions[:, c]])
        ]
        fields = [((codes[ok] == c).astype(float), ok, (reals[0] == c).astype(float)) for c in range(k)]
    else:
        rows, fields = [], []
        quantiles, score_quantiles, data_quantiles = [], [], []
        for name, v, r in zip(names, columns, reals, strict=True):
            ok = np.isfinite(v)
            rows.append({"variable": name, "realization": 0} | describe(v[ok], weights=w[ok]))
            rows += [{"variable": name, "realization": i + 1} | describe(x) for i, x in enumerate(r)]
            data_quantiles.append(_quantiles(v[ok], w[ok], p))
            quantiles.append(np.quantile(r, p, axis=1).T)
            tails = (min(v[ok].min(), r.min()), max(v[ok].max(), r.max()))
            ns = NormalScore(tails=tails).fit(v[ok], weights=w[ok])
            scores = ns.transform(r.ravel()).reshape(r.shape)
            score_quantiles.append(np.quantile(scores, p, axis=1).T)
            fields.append(
                (ns.transform(v[ok]), ok, scores) if normal_scores else (v[ok], ok, r.astype(float))
            )
        quantiles, score_quantiles = np.array(quantiles), np.array(score_quantiles)
        data_quantiles = np.array(data_quantiles)
        if len(reals) > 1:
            both = np.all([np.isfinite(c) for c in columns], axis=0)
            data_correlation = correlation(np.column_stack(columns)[both], weights=w[both])
            correlations = np.array([np.corrcoef(np.stack(r)) for r in zip(*reals, strict=True)])

    models = None
    if variogram is not None:
        models = [variogram] * len(fields) if not isinstance(variogram, list | tuple) else list(variogram)
        if len(models) != len(fields):
            raise InvalidInput(f"variogram: expected {len(fields)} models, got {len(models)}")
    if directions is None and models:
        azimuth, dip, _ = models[0].rotation
        directions = [(azimuth, dip), ((azimuth + 90.0) % 360.0, 0.0)]
        if model.count[2] > 1:
            directions.append((0.0, 90.0))
    variograms = data_variograms = None
    if lag is not None or max_lag is not None:
        if lag is None or max_lag is None:
            raise InvalidInput("variograms need both lag and max_lag")
        if coords is None:
            raise InvalidInput("variograms need data coordinates: a PointSet or an array of them")
        coords = np.asarray(coords, dtype=float)
        options = {"directions": directions, "tolerance": tolerance}
        variograms = [_realization_variograms(model, r, lag, max_lag, **options) for _, _, r in fields]
        data_variograms = [
            [
                experimental_variogram(coords[ok], x, lag, max_lag, **_direction(d, tolerance))
                for d in directions or [None]
            ]
            for x, ok, _ in fields
        ]
    statistics = Table({key: [row[key] for row in rows] for key in rows[0]})
    return RealizationCheck(
        names=names,
        statistics=statistics,
        probabilities=p,
        quantiles=quantiles,
        score_quantiles=score_quantiles,
        data_quantiles=data_quantiles,
        proportions=proportions,
        data_proportions=data_proportions,
        directions=None if directions is None else [tuple(map(float, d)) for d in directions],
        variograms=variograms,
        data_variograms=data_variograms,
        models=models,
        correlations=correlations,
        data_correlation=data_correlation,
    )


def _direction(d, tolerance):
    return {} if d is None else {"azimuth": d[0], "dip": d[1], "tolerance": tolerance}
