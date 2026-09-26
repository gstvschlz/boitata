"""Estimators: `fit` on samples, `predict` at targets (arrays, PointSet or BlockModel)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ceres._ceres import Search, Table, Variogram, _Estimator

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
    "calibrate_search",
    "classify",
    "global_bias",
]

Searches = Search | Sequence[Search]


@dataclass(frozen=True)
class CrossValidation:
    """Cross-validation results; NaN where a sample had too few neighbours."""

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

    def fit(self, coords, values, holes=None, error_variance=None, domains=None):
        """Stores the samples. Samples sharing a location keep the first one, with a warning naming their holes.

        Parameters
        ----------
        coords : array_like, shape (n, 2) or (n, 3)
        values : array_like, shape (n,)
        holes : array_like, optional
            Drill-hole ids or names, for `max_per_hole` and the ``n_holes`` diagnostic.
        error_variance : array_like, optional
            Variance of each sample's measurement error, for data of different quality (kriging only).
            It is added to the sample's diagonal entry in the kriging system, so the estimate no longer
            honours a noisy value and leans towards its neighbours. Cokriging, disjunctive kriging and the
            simulators do not take it.
        domains : array_like, optional
            Domain label of each sample: strings, numbers or booleans. A target is then estimated from the
            samples of its own domain, plus those of other domains within `Search` ``soft``. Samples
            sharing a location are dropped within a domain only; where a search reaches several at one
            location, it uses the one of the target's domain, or else the first. `predict` then needs
            domains too.
        """
        self._engine.fit(coords, values, holes, error_variance, domains)
        return self

    def predict(
        self, targets, return_variance: bool = False, anisotropy=None, diagnostics: bool = False, domains=None
    ):
        """Estimates at targets; NaN where the search found too few samples.

        Parameters
        ----------
        targets : array_like, PointSet or BlockModel
        return_variance : bool
            Also return the kriging variance.
        anisotropy : LocalAnisotropy, optional
            Orients each target's variogram and search.
        diagnostics : bool
            Return a dict of arrays instead: ``value``, ``variance``, ``efficiency`` (kriging efficiency),
            ``slope`` (slope of regression), ``n_samples``, ``pass`` (the search, from 1, that filled each
            target), ``n_holes`` (distinct holes among the samples used; untagged samples count one each),
            ``mean_distance`` (to the samples used), ``negative_weight_sum``, ``lagrange`` (the Lagrange
            multiplier; 0 for simple kriging, NaN where undefined), ``max_samples_reached`` (1 where the
            search returned `max_samples`), ``n_other_domain`` (samples used from domains other than the
            target's), ``support_variance`` (the variance of the target's support, the sill for points and
            sill minus the mean variogram within the block for blocks, nugget excluded) and
            ``estimate_variance`` (the variance of the estimator; the further below ``support_variance``,
            the smoother the estimates). NaN where unestimated.
        domains : array_like or label, optional
            Domain label of each target, or one label for all of them; required when fitted with domains.
            Targets of a domain without samples stay NaN.
        """
        return self._engine.predict(targets, return_variance, anisotropy, diagnostics, domains)

    def cross_validate(self, folds: int | None = None) -> CrossValidation:
        """Re-estimates every sample from the others, through the same search passes.

        Parameters
        ----------
        folds : int, optional
            Leave-one-out when None; otherwise k-fold, each sample estimated without the samples of its
            fold. Samples fitted with `holes` keep their holes whole: the ``j``-th of the sorted hole ids
            goes to fold ``j % folds``. An untagged sample ``i`` goes to fold ``i % folds``, so without
            holes and with `folds` equal to the number of samples this is leave-one-out. Folds span all
            domains; each held-out sample is estimated in its own domain.
        """
        estimate, variance = self._engine.cross_validate(folds)
        return CrossValidation(self._engine.values, estimate, variance)

    def with_search(self, search: Searches):
        """A copy of the estimator, fitted samples included, with `search` in place of its searches.

        Parameters
        ----------
        search : Search or sequence of Search
            One search, or passes: targets one leaves unestimated go to the next.
        """
        estimator = type(self).__new__(type(self))
        estimator._engine = self._engine.with_search(search)
        return estimator

    def to_parquet(self, path) -> None:
        """Writes the samples as Parquet columns and the parameters as JSON in the file metadata.

        Parameters
        ----------
        path : str or PathLike
        """
        self._engine.to_parquet(path, type(self).__name__)

    @classmethod
    def from_parquet(cls, path):
        """Reads `to_parquet` output; predictions match the saved estimator's bit for bit.

        Parameters
        ----------
        path : str or PathLike

        Raises
        ------
        InvalidInput
            If the file holds another class or a newer format.
        """
        estimator = cls.__new__(cls)
        estimator._engine = _Estimator.from_parquet(path, cls.__name__)
        return estimator


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


def calibrate_search(
    estimator,
    searches,
    targets,
    folds=None,
    weights=None,
    cutoffs=None,
    anamorphosis=None,
    cross_validation: bool = True,
    domains=None,
) -> Table:
    """Scores candidate searches for a fitted estimator, one row per scenario, to compare side by side.

    Each scenario re-estimates `targets` through `estimator.with_search`, so method, variogram and samples stay
    fixed. Scenarios run one after the other, each in parallel over the targets. No scenario is picked: the
    scores trade smoothing against conditional bias, and the choice is the user's.

    Parameters
    ----------
    estimator : kriging estimator
        Fitted, e.g. `BlockKriging` for blocks.
    searches : sequence
        The scenarios: each a `Search`, or a sequence of them as passes.
    targets : array_like, PointSet or BlockModel
        Where to estimate, usually the blocks.
    folds : int, optional
        Cross-validation folds; leave-one-out when None.
    weights : array_like, optional
        Declustering weights of the fitted samples, for the cross-validation scores and the global bias.
    cutoffs : sequence of float, optional
        Cutoffs for tonnage and metal ratios against the discrete Gaussian block-support reference.
    anamorphosis : HermiteAnamorphosis, optional
        Fitted on the samples (declustered if they are clustered); required with `cutoffs`. Its change-of-support
        coefficient is solved so that the block variance relative to the point variance matches the variogram's,
        ``block_variance / sill``.
    cross_validation : bool
        Adds the cross-validation scores. It runs at point support with the same variogram and passes, so block
        kriging is cross-validated as ordinary kriging.
    domains : array_like or label, optional
        Domain label of each target, or one label for all of them, as in `predict`; required when the estimator
        was fitted with domains. Cross-validation and the global bias then cover the samples of those domains only.

    Returns
    -------
    Table
        ``scenario`` (index in `searches`); ``estimate_variance`` (variance of the estimates),
        ``block_variance`` (sill minus the mean variogram within a target, nugget excluded; the sill for point
        estimators), ``variance_ratio`` (their ratio), ``model_variance_ratio`` (mean Var(Z*) of the targets over
        the block variance, the ratio the model predicts); ``slope_mean``, ``slope_p10``, ``efficiency_mean``,
        ``efficiency_p10`` (slope of regression and kriging efficiency, mean and 10th percentile);
        ``negative_weight_sum`` (mean); ``estimated`` and ``first_pass`` (fractions of the targets estimated and
        filled by the first pass); ``cv_rmse``, ``cv_mean_error`` (estimate minus actual) and ``cv_slope`` (of
        actual on estimate), declustered with `weights`; ``global_bias`` (mean estimate over the declustered data
        mean, minus 1); with `cutoffs`, ``tonnage_ratio_<cutoff>`` and ``metal_ratio_<cutoff>`` (estimated over
        reference proportion above the cutoff, and metal above it).

    Raises
    ------
    ValueError
        If `cutoffs` come without an anamorphosis or a variogram, `weights` do not match the fitted samples, or a
        scenario estimates no target.
    """
    if isinstance(searches, Search) or not len(searches):
        raise ValueError("searches must be a non-empty sequence of scenarios, each a Search or passes")
    if cutoffs is not None and anamorphosis is None:
        raise ValueError("cutoffs need a fitted HermiteAnamorphosis")
    engine = estimator._engine
    values = engine.values
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != values.shape:
            raise ValueError(
                f"{len(weights)} weights for {len(values)} fitted samples (samples sharing a location keep one)"
            )
    keep = np.ones(len(values), dtype=bool)
    if domains is not None and engine._sample_domains is not None:
        scalar = isinstance(domains, str) or np.ndim(domains) == 0
        wanted = {domains} if scalar else set(np.asarray(domains).tolist())
        keep = np.array([label in wanted for label in engine._sample_domains], dtype=bool)
    kept_weights = None if weights is None else weights[keep]
    cutoffs = [] if cutoffs is None else [float(c) for c in cutoffs]
    columns: dict[str, list[float]] = {}
    for i, scenario in enumerate(searches):
        candidate = engine.with_search(scenario)
        d = candidate.predict(targets, diagnostics=True, domains=domains)
        with np.errstate(divide="ignore", invalid="ignore"):
            row = {"scenario": i, **_scores(d, values[keep], kept_weights)}
        if cutoffs:
            estimate = d["value"][np.isfinite(d["value"])]
            row |= _recoveries(estimate, row["block_variance"], engine.variogram, anamorphosis, cutoffs)
        if cross_validation:
            estimate, _ = candidate._point_support().cross_validate(folds)
            row |= _cross_validation(values[keep], estimate[keep], kept_weights)
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return Table({key: np.asarray(column, dtype=float) for key, column in columns.items()})


def _scores(d, values, weights) -> dict[str, float]:
    ok = np.isfinite(d["value"])
    if not ok.any():
        raise ValueError("a scenario estimated no target; widen its search")
    estimate = d["value"][ok]
    block, spread = np.mean(d["support_variance"][ok]), np.var(estimate)
    return {
        "estimate_variance": float(spread),
        "block_variance": float(block),
        "variance_ratio": float(spread / block),
        "model_variance_ratio": float(np.mean(d["estimate_variance"][ok]) / block),
        "slope_mean": float(np.mean(d["slope"][ok])),
        "slope_p10": float(np.percentile(d["slope"][ok], 10)),
        "efficiency_mean": float(np.mean(d["efficiency"][ok])),
        "efficiency_p10": float(np.percentile(d["efficiency"][ok], 10)),
        "negative_weight_sum": float(np.mean(d["negative_weight_sum"][ok])),
        "estimated": float(ok.mean()),
        "first_pass": float(np.mean(d["pass"] == 1)),
        "global_bias": global_bias(estimate, values, data_weights=weights)["relative"],
    }


def _recoveries(estimate, block_variance, variogram, anamorphosis, cutoffs) -> dict[str, float]:
    if variogram is None or not np.isfinite(block_variance):
        raise ValueError("cutoffs need a kriging estimator with a variogram")
    target = min(block_variance / variogram.sill, 1.0) * anamorphosis.variance_
    low, high = 0.0, 1.0
    for _ in range(60):
        r = (low + high) / 2
        low, high = (r, high) if anamorphosis.block(r).variance_ < target else (low, r)
    reference = anamorphosis.block(high).grade_tonnage(cutoffs)
    row = {}
    for c, tonnage, metal in zip(cutoffs, reference["tonnage"], reference["metal"], strict=True):
        above = estimate >= c
        row[f"tonnage_ratio_{c:g}"] = float(np.mean(above)) / tonnage
        row[f"metal_ratio_{c:g}"] = float(np.mean(np.where(above, estimate, 0.0))) / metal
    return row


def _cross_validation(actual, estimate, weights) -> dict[str, float]:
    ok = np.isfinite(estimate)
    if not ok.any():
        return dict.fromkeys(("cv_rmse", "cv_mean_error", "cv_slope"), np.nan)
    a, e = actual[ok], estimate[ok]
    w = None if weights is None else weights[ok]
    da, de = a - np.average(a, weights=w), e - np.average(e, weights=w)
    return {
        "cv_rmse": float(np.sqrt(np.average((e - a) ** 2, weights=w))),
        "cv_mean_error": float(np.average(e - a, weights=w)),
        "cv_slope": float(np.average(da * de, weights=w) / np.average(de**2, weights=w)),
    }


_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


def classify(criteria, rules, default="unclassified", domains=None) -> np.ndarray:
    """Labels each block with the first rule whose conditions all hold.

    Parameters
    ----------
    criteria : dict of str to array_like
        Per-block metrics, e.g. ``slope``, ``efficiency`` from ``predict(..., diagnostics=True)``,
        ``nearest_dist``, ``n_holes`` from `neighborhood_stats` and distances from `hole_distance`.
    rules : sequence of (str, dict), or dict of domain to such a sequence
        ``(label, {metric: (op, threshold)})`` in priority order; `op` is one of
        ``<``, ``<=``, ``>``, ``>=``. NaN never satisfies a condition. With `domains`, a dict
        gives each domain its own rules; the ``None`` key covers domains not listed, and blocks
        of other domains get `default`.
    default : str
        Label where no rule holds.
    domains : array_like, optional
        Domain of each block, the keys of `rules`.

    Examples
    --------
    >>> classify(d, [("measured", {"slope": (">=", 0.8), "n_holes": (">=", 3)}),
    ...              ("indicated", {"slope": (">=", 0.5)})], default="inferred")
    """
    n = len(next(iter(criteria.values())))
    if domains is None:
        if isinstance(rules, dict):
            raise ValueError("rules by domain need domains")
        groups = [(np.ones(n, dtype=bool), rules)]
    else:
        domains = np.asarray(domains)
        if len(domains) != n:
            raise ValueError("one domain per block")
        if not isinstance(rules, dict):
            rules = {None: rules}
        rest = np.ones(n, dtype=bool)
        groups = []
        for key, group in rules.items():
            if key is not None:
                inside = domains == key
                groups.append((inside, group))
                rest &= ~inside
        if None in rules:
            groups.append((rest, rules[None]))
    out = np.full(n, default, dtype=object)
    for free, group in groups:
        free = free.copy()
        for label, conditions in group:
            hit = free.copy()
            for name, (op, threshold) in conditions.items():
                if op not in _OPS:
                    raise ValueError(f"unknown operator {op!r}; use one of {', '.join(_OPS)}")
                hit &= _OPS[op](np.asarray(criteria[name], dtype=float), threshold)
            out[hit] = label
            free &= ~hit
    return out.astype(str)
