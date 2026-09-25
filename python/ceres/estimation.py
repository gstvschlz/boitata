"""Estimators: `fit` on samples, `predict` at targets (arrays, PointSet or BlockModel)."""

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
]


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
    def standardized_squared_error(self) -> float:
        """Mean of error² / variance; near 1 when the kriging variance is calibrated."""
        return float(np.nanmean(self.error**2 / self.variance))


class _Base:
    def __init__(self, method: str, search: Search, variogram: Variogram | None = None, **options):
        self._engine = _Estimator(method, search, variogram, **options)
        self._values = None

    def fit(self, coords, values, holes=None):
        """Stores the samples; `holes` tags them by drill hole for `max_per_hole`."""
        self._engine.fit(coords, values, holes)
        self._values = np.asarray(values, dtype=float)
        return self

    def predict(self, targets, return_variance: bool = False):
        """Estimates at targets; NaN where the search found too few samples."""
        return self._engine.predict(targets, return_variance)

    def cross_validate(self) -> CrossValidation:
        """Re-estimates every sample with itself left out."""
        estimate, variance = self._engine.cross_validate()
        return CrossValidation(self._values, estimate, variance)


class OrdinaryKriging(_Base):
    """Kriging with an unknown, locally constant mean (weights sum to 1)."""

    def __init__(self, variogram: Variogram, search: Search):
        super().__init__("ordinary", search, variogram)


class SimpleKriging(_Base):
    """Kriging with a known global `mean`."""

    def __init__(self, variogram: Variogram, search: Search, mean: float = 0.0):
        super().__init__("simple", search, variogram, mean=mean)


class IndicatorKriging(_Base):
    """Ordinary kriging of the indicator `value <= threshold`; estimates are probabilities."""

    def __init__(self, variogram: Variogram, search: Search, threshold: float):
        super().__init__("indicator", search, variogram, threshold=threshold)


class UniversalKriging(_Base):
    """Kriging with a polynomial drift of `degree` in the coordinates."""

    def __init__(self, variogram: Variogram, search: Search, degree: int = 1):
        super().__init__("universal", search, variogram, degree=degree)


class FactorialKriging(_Base):
    """Estimates only the selected components: `structures` by index, plus the nugget if asked."""

    def __init__(self, variogram: Variogram, search: Search, structures, nugget: bool = False):
        super().__init__("factorial", search, variogram, structures=list(structures), nugget=nugget)


class BlockKriging(_Base):
    """Ordinary kriging of block averages; targets are block centres of `size`."""

    def __init__(self, variogram: Variogram, search: Search, size, discretization=(4, 4, 1)):
        super().__init__("block", search, variogram, size=list(size), discretization=tuple(discretization))


class BayesianKriging(_Base):
    """Kriging with a Gaussian prior on the drift coefficients."""

    def __init__(self, variogram: Variogram, search: Search, prior_mean, prior_variance, degree: int = 0):
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

    def __init__(self, search: Search, power: float = 2.0, variogram: Variogram | None = None):
        super().__init__("inverse_distance", search, variogram, power=power)


class NearestNeighbor(_Base):
    def __init__(self, search: Search, variogram: Variogram | None = None):
        super().__init__("nearest", search, variogram)


class MovingAverage(_Base):
    def __init__(self, search: Search, variogram: Variogram | None = None):
        super().__init__("moving_average", search, variogram)


class MovingMedian(_Base):
    def __init__(self, search: Search, variogram: Variogram | None = None):
        super().__init__("moving_median", search, variogram)


class LocalLeastSquares(_Base):
    """Local polynomial of `degree` fitted to the neighbours."""

    def __init__(self, search: Search, degree: int = 1, variogram: Variogram | None = None):
        super().__init__("local_least_squares", search, variogram, degree=degree)
