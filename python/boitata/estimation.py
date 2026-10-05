"""Estimators: `fit` on samples, `predict` at targets (arrays, PointSet or BlockModel)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from boitata._boitata import (
    BlockModel,
    Declustering,
    Drillholes,
    Mesh,
    Search,
    Table,
    Variogram,
    _DrillholePlan,
    _Estimator,
    assign_domain,
    point_in_polygon,
)
from boitata._columns import column
from boitata._units import squared, tag, unit_of
from boitata.errors import InvalidInput

__all__ = [
    "BayesianKriging",
    "BlockKriging",
    "CategoricalCrossValidation",
    "CrossValidation",
    "DrillholePlan",
    "ExternalDriftKriging",
    "FactorialKriging",
    "IndicatorCrossValidation",
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
    "weight_declustering",
]

Searches = Search | Sequence[Search]


def _labelled(table, units):
    """`table` with the `units` that are known; a unit not understood stays off."""
    known = {k: v for k, v in units.items() if v is not None and k in table.column_names}
    try:
        return table.with_units(known)
    except InvalidInput:
        return table


@dataclass(frozen=True)
class CrossValidation:
    """Cross-validation results; NaN where a sample had too few neighbors."""

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


@dataclass(frozen=True)
class IndicatorCrossValidation(CrossValidation):
    """Cross-validation of multiple indicator kriging: `estimate` is the E-type mean, `variance` the conditional
    variance, `cdf` the ``(samples, thresholds)`` corrected probabilities and `pit` ``F*(actual)``, the probability of
    the sample's own distribution not exceeding its value; NaN where a sample had too few neighbors."""

    thresholds: list[float]
    cdf: np.ndarray
    pit: np.ndarray

    @property
    def brier(self) -> np.ndarray:
        """Mean squared difference between each threshold's probability and indicator; 0 is perfect."""
        indicator = self.actual[:, None] <= np.asarray(self.thresholds)
        return np.nanmean((self.cdf - indicator) ** 2, axis=0)

    def accuracy(self, p):
        """Fraction of samples inside their symmetric `p`-probability interval; `p` or above when accurate.

        Parameters
        ----------
        p : float or array_like
            Probabilities in [0, 1].
        """
        pit = self.pit[~np.isnan(self.pit)]
        half = np.asarray(p, dtype=float)[..., None] / 2
        return np.mean(np.abs(pit - 0.5) <= half, axis=-1)

    @property
    def goodness(self) -> float:
        """Goodness statistic, 1 - ∫ (3a(p) - 2)(accuracy(p) - p) dp with a(p) = 1 where accurate; 1 is perfect,
        and an interval too narrow costs twice one too wide."""
        p = (np.arange(100) + 0.5) / 100
        excess = self.accuracy(p) - p
        return float(1 - np.mean(np.where(excess >= 0, 1, -2) * excess))


@dataclass(frozen=True)
class CategoricalCrossValidation:
    """Cross-validation of categorical indicator kriging: `actual` holds the category codes, `probabilities` the
    ``(samples, categories)`` corrected probabilities and `names` the categories; NaN where a sample had too few
    neighbors."""

    actual: np.ndarray
    probabilities: np.ndarray
    names: list[str]

    @property
    def most_likely(self) -> np.ndarray:
        """Code of the most probable category, ties to the lowest; NaN where unestimated."""
        ok = ~np.isnan(self.probabilities).any(axis=1)
        return np.where(ok, np.argmax(np.nan_to_num(self.probabilities, nan=-1.0), axis=1), np.nan)

    @property
    def brier(self) -> np.ndarray:
        """Mean squared difference between each category's probability and indicator; 0 is perfect."""
        indicator = self.actual[:, None] == np.arange(len(self.names))
        return np.nanmean((self.probabilities - indicator) ** 2, axis=0)


class _Base:
    # Unit of the estimates when it is not the unit of the values, as for probabilities.
    _estimate_unit = None

    def __init__(self, method: str, search: Searches, variogram: Variogram | None = None, **options):
        self._engine = _Estimator(method, search, variogram, **options)

    @property
    def unit(self) -> str | None:
        """Unit of the estimates: that of the fitted values, ``ratio`` for probabilities; None if unknown."""
        values = getattr(self, "_unit", None)
        return values if values is None or self._estimate_unit is None else self._estimate_unit

    def fit(self, coords, values, *, holes=None, error_variance=None, domains=None, domain_column=None):
        """Stores the samples. Samples sharing a location keep the first one, with a warning naming their holes.

        Parameters
        ----------
        coords : array_like, PointSet or BlockModel
            Sample locations, ``(n, 2)`` or ``(n, 3)``, or a container whose columns `values`, `holes`,
            `error_variance` and `domain_column` may name.
        values : array_like or str
            Sample values, or their column in `coords`; their unit, if any, becomes the unit of the estimates.
        holes : array_like or str, optional
            Drill-hole ids or names, for `max_per_hole` and the ``n_holes`` diagnostic.
        error_variance : array_like or str, optional
            Variance of each sample's measurement error, for data of different quality (kriging only).
            It is added to the sample's diagonal entry in the kriging system, so the estimate no longer
            honors a noisy value and leans towards its neighbors. Cokriging, disjunctive kriging and the
            simulators do not take it.
        domains : array_like, optional
            Domain label of each sample: strings, numbers or booleans. A target is then estimated from the
            samples of its own domain, plus those of other domains within `Search` ``soft``. Samples
            sharing a location are dropped within a domain only; where a search reaches several at one
            location, it uses the one of the target's domain, or else the first. `predict` then needs
            domains too.
        domain_column : str, optional
            The column of `coords` holding the domains; instead of `domains`.
        """
        self._unit = unit_of(values, coords)
        self._engine.fit(
            coords,
            values,
            holes=holes,
            error_variance=error_variance,
            domains=domains,
            domain_column=domain_column,
        )
        return self

    def predict(
        self,
        targets,
        *,
        return_variance: bool = False,
        anisotropy=None,
        diagnostics: bool = False,
        domains=None,
        domain_column=None,
        progress: bool = True,
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
            Return a Table instead: ``value``, ``variance``, ``efficiency`` (kriging efficiency),
            ``slope`` (slope of regression), ``n_samples``, ``pass`` (the search, from 1, that filled each
            target), ``n_holes`` (distinct holes among the samples used; untagged samples count one each),
            ``mean_distance`` (to the samples used), ``negative_weight_sum``, ``lagrange`` (the Lagrange
            multiplier; 0 for simple kriging, NaN where undefined), ``max_samples_reached`` (1 where the
            search returned `max_samples`), ``n_other_domain`` (samples used from domains other than the
            target's), ``support_variance`` (the variance of the target's support, the sill for points and
            sill minus the mean variogram within the block for blocks, nugget excluded),
            ``estimate_variance`` (the variance of the estimator; the further below ``support_variance``,
            the smoother the estimates) and ``target_met`` (1 where the samples used reach the search's
            `target_slope` or `target_efficiency`, 0 where even `max_samples` fall short, NaN without
            one). NaN where unestimated.
        domains : array_like or label, optional
            Domain label of each target, or one label for all of them; required when fitted with domains.
            Targets of a domain without samples stay NaN.
        domain_column : str, optional
            The column of `targets` holding their domains; instead of `domains`.
        progress : bool, default True
            Show a `tqdm` progress bar.

        Returns
        -------
        ndarray, tuple of ndarray or Table
            Estimates in `unit`, variances in its square.
        """
        out = self._engine.predict(
            targets,
            return_variance=return_variance,
            anisotropy=anisotropy,
            diagnostics=diagnostics,
            domains=domains,
            domain_column=domain_column,
            progress=progress,
        )
        unit = self.unit
        if diagnostics:
            return _labelled(out, {"value": unit, "variance": squared(unit)})
        if return_variance:
            return tag(out[0], unit), tag(out[1], squared(unit))
        return tag(out, unit)

    def cross_validate(self, *, folds: int | None = None) -> CrossValidation:
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
        estimate, variance = self._engine.cross_validate(folds=folds)
        unit = self.unit
        return CrossValidation(
            tag(self._engine.values, getattr(self, "_unit", None)),
            tag(estimate, unit),
            tag(variance, squared(unit)),
        )

    def with_search(self, search: Searches):
        """A copy of the estimator, fitted samples included, with `search` in place of its searches.

        Parameters
        ----------
        search : Search or sequence of Search
            One search, or passes: targets one leaves unestimated go to the next.
        """
        estimator = type(self).__new__(type(self))
        estimator._engine = self._engine.with_search(search)
        estimator._unit = getattr(self, "_unit", None)
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

    def __init__(self, variogram: Variogram, search: Searches, *, mean: float = 0.0):
        super().__init__("simple", search, variogram, mean=mean)


class IndicatorKriging(_Base):
    """Ordinary kriging of the indicator `value <= threshold`; estimates are probabilities."""

    _estimate_unit = "ratio"

    def __init__(self, variogram: Variogram, search: Searches, threshold: float):
        super().__init__("indicator", search, variogram, threshold=threshold)


class UniversalKriging(_Base):
    """Kriging with a polynomial drift of `degree` in the coordinates."""

    def __init__(self, variogram: Variogram, search: Searches, *, degree: int = 1):
        super().__init__("universal", search, variogram, degree=degree)


class ExternalDriftKriging(_Base):
    """Kriging with a polynomial drift of `degree` (0 by default) plus one column per external
    drift variable, evaluated at the samples with `fit` and at the targets with `predict`."""

    def __init__(
        self, variogram: Variogram, search: Searches, drift: str | Sequence[str], *, degree: int = 0
    ):
        names = [drift] if isinstance(drift, str) else list(drift)
        super().__init__("external_drift", search, variogram, degree=degree, drift=names)


class FactorialKriging(_Base):
    """Estimates only the selected components: `structures` by index, plus the nugget if asked."""

    def __init__(self, variogram: Variogram, search: Searches, structures, *, nugget: bool = False):
        super().__init__("factorial", search, variogram, structures=list(structures), nugget=nugget)


class BlockKriging(_Base):
    """Ordinary kriging of block averages; targets are block centers of `size`."""

    def __init__(self, variogram: Variogram, search: Searches, size, *, discretization=(4, 4, 1)):
        super().__init__("block", search, variogram, size=list(size), discretization=tuple(discretization))


class BayesianKriging(_Base):
    """Kriging with a Gaussian prior on the drift coefficients."""

    def __init__(
        self, variogram: Variogram, search: Searches, prior_mean, prior_variance, *, degree: int = 0
    ):
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

    def __init__(self, search: Searches, *, power: float = 2.0, variogram: Variogram | None = None):
        super().__init__("inverse_distance", search, variogram, power=power)


class NearestNeighbor(_Base):
    def __init__(self, search: Searches, *, variogram: Variogram | None = None):
        super().__init__("nearest", search, variogram)


class MovingAverage(_Base):
    def __init__(self, search: Searches, *, variogram: Variogram | None = None):
        super().__init__("moving_average", search, variogram)


class MovingMedian(_Base):
    def __init__(self, search: Searches, *, variogram: Variogram | None = None):
        super().__init__("moving_median", search, variogram)


class LocalLeastSquares(_Base):
    """Local polynomial of `degree` fitted to the neighbors."""

    def __init__(self, search: Searches, *, degree: int = 1, variogram: Variogram | None = None):
        super().__init__("local_least_squares", search, variogram, degree=degree)


def weight_declustering(coords, values, targets, *, estimator: _Base) -> Declustering:
    """Declustering weights from estimation weights: each sample's weight is the sum of the weights it receives
    when `estimator` estimates `targets`, scaled to sum to the number of samples, as in `cell_declustering`.

    A sample alone in a sparse area informs many targets and gets a large weight; samples in a cluster share
    the targets around them. Targets the search leaves unestimated add nothing, so they should cover the domain
    and no more. Kriging can give negative weights, so a screened sample's weight can be below 0.

    Parameters
    ----------
    coords : array_like, PointSet or BlockModel
        Sample locations, ``(n, 2)`` or ``(n, 3)``, or a container whose column `values` may name.
    values : array_like or str
    targets : array_like, PointSet or BlockModel
        A dense set of points, or blocks (their centroids), covering the domain.
    estimator : OrdinaryKriging, SimpleKriging, UniversalKriging, BlockKriging, IndicatorKriging,
        InverseDistance, NearestNeighbor or MovingAverage
        Its search passes and variogram, and its weights; it is not fitted or changed. `NearestNeighbor` gives
        polygon declustering at the resolution of the targets.

    Returns
    -------
    Declustering
        ``weights`` per sample, samples sharing a location sharing one weight, and the declustered ``mean``;
        ``cell_size`` is NaN.
    """
    return estimator._engine._declustering(coords, values, targets)


def global_bias(estimate, data, *, weights=None, data_weights=None) -> dict[str, float]:
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
    *,
    folds=None,
    weights=None,
    cutoffs=None,
    anamorphosis=None,
    cross_validation: bool = True,
    domains=None,
    domain_column=None,
    data=None,
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
    weights : array_like or str, optional
        Declustering weights of the fitted samples, for the cross-validation scores and the global bias, or the
        column of `data` holding them.
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
    domain_column : str, optional
        The column of `targets` holding their domains; instead of `domains`.
    data : PointSet or Table, optional
        The fitted samples, when `weights` is a column name.

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
    InvalidInput
        If `cutoffs` come without an anamorphosis or a variogram, `weights` do not match the fitted samples, or a
        scenario estimates no target.
    """
    if isinstance(searches, Search) or not len(searches):
        raise InvalidInput("searches must be a non-empty sequence of scenarios, each a Search or passes")
    if cutoffs is not None and anamorphosis is None:
        raise InvalidInput("cutoffs need a fitted HermiteAnamorphosis")
    if domain_column is not None:
        if domains is not None:
            raise InvalidInput("give one of domains or domain_column")
        domains = column(targets, domain_column, "domain_column")
    engine = estimator._engine
    values = engine.values
    if isinstance(weights, str) and data is None:
        raise InvalidInput(f'weights names column "{weights}"; pass the fitted samples as data=')
    if weights is not None:
        weights = np.asarray(column(data, weights, "weights"), dtype=float)
        if weights.shape != values.shape:
            raise InvalidInput(
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
            estimate, _ = candidate._point_support().cross_validate(folds=folds)
            row |= _cross_validation(values[keep], estimate[keep], kept_weights)
        for key, value in row.items():
            columns.setdefault(key, []).append(value)
    return Table({key: np.asarray(column, dtype=float) for key, column in columns.items()})


def _scores(d, values, weights) -> dict[str, float]:
    ok = np.isfinite(d["value"])
    if not ok.any():
        raise InvalidInput("a scenario estimated no target; widen its search")
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
        raise InvalidInput("cutoffs need a kriging estimator with a variogram")
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


def classify(criteria, rules, *, default="unclassified", domains=None, domain_column=None) -> np.ndarray:
    """Labels each block with the first rule whose conditions all hold.

    Parameters
    ----------
    criteria : Table or dict of str to array_like
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
    domain_column : str, optional
        The column of `criteria` holding the domains; instead of `domains`.

    Examples
    --------
    >>> classify(d, [("measured", {"slope": (">=", 0.8), "n_holes": (">=", 3)}),
    ...              ("indicated", {"slope": (">=", 0.5)})], default="inferred")
    """
    n = criteria.num_rows if hasattr(criteria, "num_rows") else len(next(iter(criteria.values())))
    if domain_column is not None:
        if domains is not None:
            raise InvalidInput("give one of domains or domain_column")
        domains = column(criteria, domain_column, "domain_column")
    if domains is None:
        if isinstance(rules, dict):
            raise InvalidInput("rules by domain need domains")
        groups = [(np.ones(n, dtype=bool), rules)]
    else:
        domains = np.asarray(domains)
        if len(domains) != n:
            raise InvalidInput("one domain per block")
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
                    raise InvalidInput(f"unknown operator {op!r}; use one of {', '.join(_OPS)}")
                hit &= _OPS[op](np.asarray(column(criteria, name, "criteria"), dtype=float), threshold)

            out[hit] = label
            free &= ~hit
    return out.astype(str)


class DrillholePlan:
    """Candidate drill holes scored by the kriging metrics they would bring to target blocks.

    The problem half of drillhole optimization: it scores sets of candidates, given by index into
    ``candidates.holes``, and leaves the choice of set to a search. Kriging variance, slope and efficiency
    depend on sample locations only, so only the locations of `data` matter. Each block is kriged as
    ``estimator.predict(targets, diagnostics=True)`` would after fitting `data` plus the composites of the
    selected holes, in candidate order. A candidate reaches the blocks within the search radius of its
    composites in any search pass; only those are re-kriged to evaluate it, and its cached gain is dropped only
    when a change of plan touches them. Results do not depend on the number of threads.

    With domains, each block is kriged from the samples of its domain, and of others within `Search` ``soft``,
    as `predict` does; blocks of a domain without samples stay unestimated until a hole brings it some. The
    composites of `data` keep their logged domains; a candidate's composites, which have no log, take the domain
    of the block they fall in, or of the nearest target outside the blocks or when the targets are points.

    Parameters
    ----------
    candidates : Drillholes
        Holes that could be drilled, e.g. from `planned_drillholes`; they need intervals to composite.
    estimator : OrdinaryKriging, SimpleKriging, IndicatorKriging or BlockKriging
        Its variogram, search passes and support; fitted or not. Its fitted samples are not used, and when
        fitted with domains `domains` is required. Searches with a high-grade restriction or a target slope or
        efficiency are refused.
    targets : BlockModel, PointSet or array_like
        Where to krige: block centroids or points.
    data : Drillholes, PointSet or array_like
        Existing samples. Drillholes are composited at `composite_length` and keep their holes for
        `max_per_hole`; points count as untagged samples. Values are unused.
    objective : {"classification", "variance"} or callable
        The score of a plan is the sum over blocks of a score g(metrics), minus that sum with `data` alone.
        ``"classification"`` scores a block by its progress towards the class above its current one under
        `rules`. Each condition of that class gives 1 when met, 0 when met today but no longer, and otherwise
        ``clip((m - m0) / (threshold - m0), 0, 1)`` with ``m0`` today's metric; with ``P`` their mean and
        ``P0`` today's, the block scores ``weight * (P - P0) / (1 - P0)``, which is ``weight`` exactly when it
        reaches the class; ``clip`` gives 0 where its ratio is undefined, as from an infinite ``m0``. Blocks
        already in the best class, or without rules, score 0. ``"variance"`` scores
        ``weight * (variance0 - variance) / support_variance``. Both built-ins read an unestimated block as
        variance equal to its support variance and 0 for slope, efficiency and counts, and a NaN
        ``data_spacing`` as infinite. A callable takes a Table as
        `metrics` returns it, for some of the blocks, and returns one finite score per row; it is called at
        most once per `score`, `gains` or `loss`, on the rows of every block that call needs.
    rules : sequence of (str, dict), or dict of domain to such a sequence, optional
        As in `classify`, best class first: ``(label, {metric: (op, threshold)})`` with metrics among
        ``variance``, ``slope``, ``efficiency``, ``n_samples``, ``n_holes`` and ``data_spacing``; required for
        ``"classification"``. With `domains`, a dict gives each domain its own rules; the ``None`` key covers
        domains not listed, and blocks of other domains score 0.
    weights : array_like or str, optional
        Weight of each block, >= 0, or the column of `targets` holding them; 1 by default. The built-in
        objectives skip blocks of weight 0.
    n_holes : int, optional
        Most holes in a plan.
    budget : float, optional
        Most total cost of a plan.
    cost_per_meter : float, optional
        Cost of a hole per meter of its length; without it a hole costs its length, so `budget` is in meters.
    min_spacing : float
        Least distance in plan between the collars of two holes of a plan.
    exclude : array_like, Polylines, Mesh or a sequence of them, optional
        Zones candidates may not enter: polygons (``(n, 2)`` or ``(n, 3)`` vertices, or Polylines) hold
        collars in plan, closed meshes hold collars and composites.
    composite_length : float, optional
        Length of the composites of the candidates, and of `data` when Drillholes, and ``c`` of
        ``data_spacing``; by default the median interval length of `data`, which then must be Drillholes.
    domains : array_like, label or tuple, optional
        Domain label of each target, or one label for all of them, as in `predict`; or
        ``(target_domains, data_domains)``, the second one label per composite of `data`. Without data
        domains, the composites of `data` are labeled as the candidates'.
    domain_column : str or tuple of str, optional
        The column of `targets` and `data` holding their domains, or one name for each,
        ``(target_column, data_column)``, as in `hole_distance`; instead of `domains`. A Drillholes `data`
        takes it from its intervals and composites without crossing it.

    Raises
    ------
    InvalidInput
        If an input is missing or malformed, or the estimator is not supported.
    """

    def __init__(
        self,
        candidates: Drillholes,
        estimator,
        targets,
        *,
        data,
        objective="classification",
        rules=None,
        weights=None,
        n_holes: int | None = None,
        budget: float | None = None,
        cost_per_meter: float | None = None,
        min_spacing: float = 0.0,
        exclude=None,
        composite_length: float | None = None,
        domains=None,
        domain_column: str | tuple[str, str] | None = None,
    ):
        if not isinstance(candidates, Drillholes):
            raise InvalidInput("candidates must be Drillholes")
        if composite_length is None:
            if not isinstance(data, Drillholes) or data.interval_columns is None:
                raise InvalidInput("composite_length is required unless data are Drillholes with intervals")
            _, start, end = data.interval_columns
            intervals = data.samples()
            composite_length = float(np.median(intervals[end] - intervals[start]))
        if domain_column is not None and domains is not None:
            raise InvalidInput("give one of domains or domain_column")
        target_column, data_column = (
            domain_column if isinstance(domain_column, tuple) else (domain_column,) * 2
        )
        logged = None
        if isinstance(domains, tuple):
            domains, logged = domains
        elif target_column is not None:
            domains = column(targets, target_column, "domain_column")
        data_holes = None
        if isinstance(data, Drillholes):
            composited = data.composite(composite_length, [], domain=data_column)
            hole = composited[data.interval_columns[0]].astype(str)
            data, data_holes = composited.coords, np.unique(hole, return_inverse=True)[1].tolist()
            logged = logged if data_column is None else composited[data_column]
        elif hasattr(data, "coords"):
            logged = logged if data_column is None else column(data, data_column, "domain_column")
            data = data.coords
        data = np.asarray(data, dtype=float).reshape(-1, np.shape(data)[-1] if np.size(data) else 3)
        if data.shape[1] == 2:
            data = np.column_stack([data, np.zeros(len(data))])
        self._holes = candidates.holes
        composites = candidates.composite(composite_length, [])
        index = {h: i for i, h in enumerate(self._holes)}
        owner = np.array([index[h] for h in composites[candidates.interval_columns[0]]], dtype=int)
        paths = candidates.paths()
        names = paths[paths.column_names[0]]
        self._collars = candidates.at(self._holes, np.zeros(len(self._holes)))
        self._length = np.array([paths["depth"][names == h].max() for h in self._holes])
        self._azimuth, self._dip = _directions(paths, names, self._holes)
        self._n_holes, self._budget = n_holes, budget
        self._cost = self._length * (1.0 if cost_per_meter is None else float(cost_per_meter))
        excluded = np.zeros(len(self._holes), dtype=bool)
        zones = [] if exclude is None else [exclude] if _is_zone(exclude) else list(exclude)
        for zone in zones:
            if isinstance(zone, Mesh):
                excluded |= zone.contains(self._collars)
                excluded[owner[zone.contains(composites.coords)]] = True
            else:
                excluded |= point_in_polygon(self._collars[:, :2], zone)
        if isinstance(weights, str):
            weights = column(targets, weights, "weights")
        labels, rule_of = None, None
        if domains is None and logged is not None:
            raise InvalidInput("domains of the data need domains of the targets")
        if domains is not None:
            centroids = targets.centroids if hasattr(targets, "centroids") else None
            points = centroids if centroids is not None else getattr(targets, "coords", targets)
            points = np.asarray(points, dtype=float)
            domains = np.asarray(domains, dtype=object)
            if domains.ndim == 0:
                domains = np.full(len(points), domains.item(), dtype=object)
            if logged is None:
                logged = _located(targets, points, domains, data)
            elif len(logged) != len(data):
                raise InvalidInput(f"{len(logged)} data domains for {len(data)} composites")
            located = _located(targets, points, domains, composites.coords)
            labels = np.concatenate([domains, np.asarray(logged, dtype=object), located]).tolist()
        if isinstance(rules, dict) and not callable(objective):
            if domains is None:
                raise InvalidInput("rules by domain need domains")
            keys = list(rules)
            rule_of = [None] * len(domains)
            for i, label in enumerate(domains.tolist()):
                k = keys.index(label) if label in keys else keys.index(None) if None in keys else None
                rule_of[i] = k
            rules = [_rules(rules[k]) for k in keys]
        elif rules is not None and not callable(objective):
            rules = [_rules(rules)]
        self._engine = _DrillholePlan(
            estimator._engine,
            targets,
            data,
            data_holes,
            composites.coords,
            owner.tolist(),
            self._collars,
            self._cost.tolist(),
            excluded.tolist(),
            objective,
            None if callable(objective) else rules,
            rule_of,
            weights,
            n_holes,
            budget,
            float(min_spacing),
            labels,
            float(composite_length),
        )

    @property
    def holes(self) -> list[str]:
        """Candidate names; plans index into them."""
        return list(self._holes)

    @property
    def collars(self) -> np.ndarray:
        """``(n, 3)`` collar of each candidate."""
        return self._collars.copy()

    @property
    def length(self) -> np.ndarray:
        """Length of each candidate in meters."""
        return self._length.copy()

    @property
    def cost(self) -> np.ndarray:
        """Cost of each candidate: its length times `cost_per_meter`, or its length."""
        return self._cost.copy()

    @property
    def n_holes(self) -> int | None:
        """Most holes in a plan, or None."""
        return self._n_holes

    @property
    def budget(self) -> float | None:
        """Most total cost of a plan, or None."""
        return self._budget

    def score(self, selected) -> float:
        """The objective with the holes `selected`; 0 for none.

        Parameters
        ----------
        selected : sequence of int
            Distinct candidate indices.
        """
        return self._engine.score(_indices(selected))

    def gains(self, selected) -> np.ndarray:
        """Gain in the objective from adding each candidate to `selected`.

        Parameters
        ----------
        selected : sequence of int

        Returns
        -------
        ndarray
            One gain per candidate, ``-inf`` where `feasible` is False, so ``argmax`` picks a feasible hole.
        """
        return self._engine.gains(_indices(selected))

    def loss(self, selected) -> np.ndarray:
        """Loss in the objective from removing each of `selected`, in its order.

        Parameters
        ----------
        selected : sequence of int
        """
        return self._engine.loss(_indices(selected))

    def feasible(self, selected) -> np.ndarray:
        """Whether each candidate could join `selected`: not selected, outside `exclude`, within `n_holes` and
        `budget`, and at least `min_spacing` from every selected collar.

        Parameters
        ----------
        selected : sequence of int
        """
        return self._engine.feasible(_indices(selected))

    def neighbors(self, i: int, radius: float) -> np.ndarray:
        """Candidates other than `i` whose collars lie within `radius` of its collar in plan, in index order.

        Parameters
        ----------
        i : int
        radius : float
        """
        return self._engine.neighbors(int(i), float(radius))

    def metrics(self, selected) -> Table:
        """Kriging metrics of every block with the holes `selected`.

        Parameters
        ----------
        selected : sequence of int

        Returns
        -------
        Table
            ``block`` (index into `targets`), ``variance``, ``slope``, ``efficiency``, ``n_samples`` and
            ``n_holes`` as ``predict(..., diagnostics=True)`` gives them (NaN where unestimated),
            ``data_spacing``, ``support_variance`` and ``weight``. ``data_spacing`` is
            ``data_spacing(targets, samples, search, composite_length=composite_length)`` with the first search
            pass of the estimator: ``sqrt(V / (c n))``, ``V`` the volume of that Search's ellipsoid (its radius
            and its own rotation and ratios, not the variogram's), ``n`` the samples of any domain inside it
            and ``c`` `composite_length`; NaN where it holds none.
        """
        return self._engine.metrics(_indices(selected))

    def optimize(self, n: int | None = None, *, search=None) -> Table:
        """Chooses the holes to drill with `search`, and ranks them.

        Parameters
        ----------
        n : int, optional
            Most holes to choose; by default `n_holes`, or no limit but the budget and the other constraints.
        search : DrillholeSearch, optional
            Any object with a ``search(problem, n)`` method returning candidate indices; ``Swap(start=Greedy())``
            by default.

        Returns
        -------
        Table
            One row per chosen hole, in drilling order: ``ORDER`` (from 1), ``HOLE_ID``, the collar ``X``, ``Y``,
            ``Z``, ``AZIMUTH`` and ``DIP`` at the collar, ``LENGTH``, ``COST``, ``GAIN`` (the gain in the objective
            from adding the hole after those above it), ``CUMULATIVE`` (the objective with the holes up to it)
            and ``CONTRIBUTION`` (the loss in the objective from removing the hole from the chosen plan). The
            order is greedy within the chosen plan: each row is the hole of largest gain among those left, so
            drilling stops early with the best prefix, whatever search chose the plan.

        Raises
        ------
        InvalidInput
            If `search` has no ``search`` method, or returns repeated or unknown indices, more than `n` holes,
            or a plan that breaks the constraints.

        Examples
        --------
        >>> table = plan.optimize(10, search=bt.Annealing(seed=1))
        """
        from boitata.planning import Greedy, Swap

        search = Swap(start=Greedy()) if search is None else search
        if not callable(getattr(search, "search", None)):
            raise InvalidInput("search must have a search(problem, n) method")
        if n is None:
            n = self._n_holes if self._n_holes is not None else len(self._holes)
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise InvalidInput("n must be an integer >= 0")
        chosen = _indices(search.search(self, int(n)))
        if len(set(chosen)) != len(chosen) or (chosen and max(chosen) >= len(self._holes)):
            raise InvalidInput("the search returned repeated or unknown candidate indices")
        if len(chosen) > n:
            raise InvalidInput(f"the search returned {len(chosen)} holes for n = {n}")
        order, gain, left = [], [], chosen
        while left:
            gains = self.gains(order)[left]
            k = int(np.argmax(gains))
            if not np.isfinite(gains[k]):
                raise InvalidInput("the search returned a plan that breaks the constraints")
            order.append(left[k])
            gain.append(gains[k])
            left = left[:k] + left[k + 1 :]
        rows = np.array(order, dtype=int)
        return Table(
            {
                "ORDER": np.arange(1.0, len(order) + 1),
                "HOLE_ID": np.array([self._holes[i] for i in order], dtype=object),
                "X": self._collars[rows, 0],
                "Y": self._collars[rows, 1],
                "Z": self._collars[rows, 2],
                "AZIMUTH": self._azimuth[rows],
                "DIP": self._dip[rows],
                "LENGTH": self._length[rows],
                "COST": self._cost[rows],
                "GAIN": np.array(gain, dtype=float),
                "CUMULATIVE": np.cumsum(gain, dtype=float),
                "CONTRIBUTION": np.asarray(self.loss(order), dtype=float) if order else np.zeros(0),
            }
        )


def _directions(paths, names, holes):
    """Azimuth and dip of each hole's first path segment, in degrees; dip positive down."""
    first = {}
    for row, name in enumerate(names.tolist()):
        first.setdefault(name, row)
    start = np.array([first[h] for h in holes])
    xyz = np.column_stack([paths[c] for c in ("x", "y", "z")])
    step = xyz[np.minimum(start + 1, len(xyz) - 1)] - xyz[start]
    length = np.linalg.norm(step, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dip = np.degrees(np.arcsin(-step[:, 2] / length))
    return np.degrees(np.arctan2(step[:, 0], step[:, 1])) % 360, dip


def _is_zone(obj) -> bool:
    return isinstance(obj, Mesh) or hasattr(obj, "parts") or np.ndim(obj) == 2


def _indices(selected) -> list[int]:
    indices = np.asarray(selected).ravel()
    if indices.size and (indices.dtype.kind not in "iu" or indices.min() < 0):
        raise InvalidInput("selected must be candidate indices >= 0")
    return indices.astype(int).tolist()


def _located(targets, points, domains, located):
    """Domain of the target block each of `located` falls in, or of the nearest target."""
    out = np.empty(len(located), dtype=object)
    rows = targets.row_at(located) if isinstance(targets, BlockModel) else np.full(len(located), -1)
    inside = rows >= 0
    out[inside] = domains[rows[inside]]
    if (~inside).any():
        codes, index = np.unique(domains.astype(str), return_inverse=True)
        first = {c: domains[np.flatnonzero(index == k)[0]] for k, c in enumerate(codes)}
        nearest, _ = assign_domain(located[~inside], coords=points, domains=index.astype(str))
        out[~inside] = [first[codes[int(k)]] for k in nearest]
    return out


def _rules(rules):
    return [[(name, op, float(t)) for name, (op, t) in conditions.items()] for _, conditions in rules]
