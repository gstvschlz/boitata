"""Estimators: `fit` on samples, `predict` at targets (arrays, PointSet or BlockModel)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ceres._ceres import Search, Variogram, _Estimator

__all__ = [
    "BayesianKriging",
    "BlockKriging",
    "CrossValidation",
    "FactorialKriging",
    "IndicatorKriging",
    "InverseDistance",
    "LocalLeastSquares",
    "MovingAverage",
    "MovingMedian",
    "NearestNeighbor",
    "OrdinaryKriging",
    "SimpleKriging",
    "UniversalKriging",
    "classify",
    "global_bias",
]

Searches = Search | Sequence[Search]


@dataclass(frozen=True)
class CrossValidation:
    """Leave-one-out results; NaN where a sample had too few neighbours."""

    actual: np.ndarray
    estimate: np.ndarray
    variance: np.ndarray

    @property
    def error(self) -> np.ndarray:
        return self.estimate - self.actual

    @property
    def mean_error(self) -> float:
        return float(np.nanmean(self.error))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.nanmean(self.error**2)))

    @property
    def correlation(self) -> float:
        ok = ~np.isnan(self.estimate)
        return float(np.corrcoef(self.actual[ok], self.estimate[ok])[0, 1])

    @property
    def slope(self) -> float:
        """Slope of the regression of actual on estimated values; below 1 means conditional bias."""
        ok = ~np.isnan(self.estimate)
        return float(np.polyfit(self.estimate[ok], self.actual[ok], 1)[0])

    @property
    def standardized_squared_error(self) -> float:
        """Mean of error² / variance; near 1 when the kriging variance is calibrated."""
        return float(np.nanmean(self.error**2 / self.variance))


class _Base:
    def __init__(self, method: str, search: Searches, variogram: Variogram | None = None, **options):
        self._engine = _Estimator(method, search, variogram, **options)

    def fit(self, coords, values, holes=None):
        """Stores the samples; `holes` (ids or names) tags them by drill hole for `max_per_hole`.

        Samples sharing a location keep the first one, with a warning naming their holes.
        """
        self._engine.fit(coords, values, holes)
        return self

    def predict(self, targets, return_variance: bool = False, anisotropy=None, diagnostics: bool = False):
        """Estimates at targets; NaN where the search found too few samples.

        `anisotropy` (a LocalAnisotropy) orients each target's variogram and search. With
        `diagnostics`, returns a dict of ``value``, ``variance``, ``efficiency`` (kriging
        efficiency), ``slope`` (slope of regression), ``n_samples`` and ``pass`` (the search,
        from 1, that filled each target).
        """
        return self._engine.predict(targets, return_variance, anisotropy, diagnostics)

    def cross_validate(self) -> CrossValidation:
        """Re-estimates every sample with itself left out, through the same search passes."""
        estimate, variance = self._engine.cross_validate()
        return CrossValidation(self._engine.values, estimate, variance)


class OrdinaryKriging(_Base):
    """Kriging with an unknown, locally constant mean (weights sum to 1)."""

    def __init__(self, variogram: Variogram, search: Searches):
        super().__init__("ordinary", search, variogram)


class SimpleKriging(_Base):
    """Kriging with a known global `mean`."""

    def __init__(self, variogram: Variogram, search: Searches, mean: float = 0.0):
        super().__init__("simple", search, variogram, mean=mean)


class IndicatorKriging(_Base):
    """Ordinary kriging of the indicator `value <= threshold`; estimates are probabilities."""

    def __init__(self, variogram: Variogram, search: Searches, threshold: float):
        super().__init__("indicator", search, variogram, threshold=threshold)


class UniversalKriging(_Base):
    """Kriging with a polynomial drift of `degree` in the coordinates."""

    def __init__(self, variogram: Variogram, search: Searches, degree: int = 1):
        super().__init__("universal", search, variogram, degree=degree)


class FactorialKriging(_Base):
    """Estimates only the selected components: `structures` by index, plus the nugget if asked."""

    def __init__(self, variogram: Variogram, search: Searches, structures, nugget: bool = False):
        super().__init__("factorial", search, variogram, structures=list(structures), nugget=nugget)


class BlockKriging(_Base):
    """Ordinary kriging of block averages; targets are block centres of `size`."""

    def __init__(self, variogram: Variogram, search: Searches, size, discretization=(4, 4, 1)):
        super().__init__("block", search, variogram, size=list(size), discretization=tuple(discretization))


class BayesianKriging(_Base):
    """Kriging with a Gaussian prior on the drift coefficients."""

    def __init__(self, variogram: Variogram, search: Searches, prior_mean, prior_variance, degree: int = 0):
        super().__init__(
            "bayesian",
            search,
            variogram,
            degree=degree,
            prior_mean=list(prior_mean),
            prior_variance=list(prior_variance),
        )


class InverseDistance(_Base):
    """Inverse-distance weighting; a `variogram` supplies anisotropic distances."""

    def __init__(self, search: Searches, power: float = 2.0, variogram: Variogram | None = None):
        super().__init__("inverse_distance", search, variogram, power=power)


class NearestNeighbor(_Base):
    def __init__(self, search: Searches, variogram: Variogram | None = None):
        super().__init__("nearest", search, variogram)


class MovingAverage(_Base):
    def __init__(self, search: Searches, variogram: Variogram | None = None):
        super().__init__("moving_average", search, variogram)


class MovingMedian(_Base):
    def __init__(self, search: Searches, variogram: Variogram | None = None):
        super().__init__("moving_median", search, variogram)


class LocalLeastSquares(_Base):
    """Local polynomial of `degree` fitted to the neighbours."""

    def __init__(self, search: Searches, degree: int = 1, variogram: Variogram | None = None):
        super().__init__("local_least_squares", search, variogram, degree=degree)


def global_bias(estimate, data, weights=None, data_weights=None) -> dict[str, float]:
    """Means of an estimate and of the data it came from, weighted by e.g. block volumes and
    declustering weights, and the relative difference ``estimate / data - 1``.
    """
    estimate, data = np.asarray(estimate, dtype=float), np.asarray(data, dtype=float)
    ok, okd = np.isfinite(estimate), np.isfinite(data)
    m = np.average(estimate[ok], weights=None if weights is None else np.asarray(weights)[ok])
    d = np.average(data[okd], weights=None if data_weights is None else np.asarray(data_weights)[okd])
    return {"estimate_mean": float(m), "data_mean": float(d), "relative": float(m / d - 1)}


_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


def classify(criteria, rules, default="unclassified") -> np.ndarray:
    """Labels each block with the first rule whose conditions all hold.

    Parameters
    ----------
    criteria : dict of str to array_like
        Per-block metrics, e.g. ``slope``, ``efficiency`` from ``predict(..., diagnostics=True)``
        and ``nearest_dist``, ``n_holes`` from `neighborhood_stats`.
    rules : sequence of (str, dict)
        ``(label, {metric: (op, threshold)})`` in priority order; `op` is one of
        ``<``, ``<=``, ``>``, ``>=``. NaN never satisfies a condition.
    default : str
        Label where no rule holds.

    Examples
    --------
    >>> classify(d, [("measured", {"slope": (">=", 0.8), "n_holes": (">=", 3)}),
    ...              ("indicated", {"slope": (">=", 0.5)})], default="inferred")
    """
    n = len(next(iter(criteria.values())))
    out = np.full(n, default, dtype=object)
    free = np.ones(n, dtype=bool)
    for label, conditions in rules:
        hit = free.copy()
        for name, (op, threshold) in conditions.items():
            if op not in _OPS:
                raise ValueError(f"unknown operator {op!r}; use one of {', '.join(_OPS)}")
            hit &= _OPS[op](np.asarray(criteria[name], dtype=float), threshold)
        out[hit] = label
        free &= ~hit
    return out.astype(str)
