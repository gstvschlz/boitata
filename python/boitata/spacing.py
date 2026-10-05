"""Drill-hole spacing studies: virtual drilling of simulated truths."""

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from boitata import _boitata
from boitata.errors import InvalidInput

__all__ = ["spacing_study"]


def spacing_study(
    simulator: "_boitata.SGS | _boitata.DSS | _boitata.TurningBands",
    targets: "_boitata.BlockModel",
    *,
    spacings: "Sequence[float | tuple[float, float]] | None" = None,
    plans: "Mapping[str, _boitata.Drillholes] | None" = None,
    rotation: float = 0.0,
    azimuth: float = 0.0,
    dip: float = 90.0,
    topography: "_boitata.Mesh | None" = None,
    truths: "int | Sequence[int]" = 10,
    n: int = 50,
    seed: int = 0,
    blocks: "_boitata.BlockModel | None" = None,
    window: "Sequence[float] | None" = None,
    groups: "ArrayLike | str | None" = None,
    sampling_error: float = 0.0,
    existing: bool = False,
    composite_length: float | None = None,
    quantiles: tuple[float, float] = (0.05, 0.95),
    tolerances: Sequence[float] = (0.15,),
    progress: bool = True,
) -> "_boitata.Table":
    """Uncertainty of the summary rows after drilling each plan, on simulated truths (virtual drilling).

    Each truth is a realization of the fitted `simulator` at `targets`, so it is conditioned to the current
    data. For each plan and truth, the plan's composites read the truth at the target holding their centre,
    optionally with a sampling error ``z * (1 + Y * sampling_error)``, ``Y`` standard normal. A copy of
    `simulator` is refitted on those virtual samples, without declustering weights (and with the existing data
    too when `existing`), simulates `n` realizations at `targets`, and its summary is checked against the truth
    averaged to the same rows, as `SimulationSummary.validate` does.

    Parameters
    ----------
    simulator : SGS, DSS or TurningBands
        Fitted, without trend, domains or secondary; its variogram and search serve every refit.
    targets : BlockModel
        Nodes the truths and the realizations are simulated at; planned holes cover them and composites read
        the truth of the block holding their centre.
    spacings : sequence of float or (float, float), optional
        Collar spacings, each one value or ``(along, across)``; the holes come from `planned_drillholes` with
        `rotation`, `azimuth`, `dip` and `topography`. Give one of `spacings` or `plans`.
    plans : dict of str to Drillholes, optional
        Named drilling plans with intervals, such as candidate holes chosen by an optimizer.
    rotation, azimuth, dip : float
        Collar grid rotation and hole direction, in degrees, as in `planned_drillholes`; only with `spacings`.
    topography : Mesh, optional
        Ground the collars of `spacings` sit on.
    truths : int or sequence of int
        Number of truths, realizations ``0 .. truths - 1``, or their indices, such as from
        `select_realizations`. Index ``k`` is realization ``k`` of ``simulator.simulate(targets, seed=seed)``
        whatever ``n``, so indices picked from a summary simulated with `seed` name the same realizations.
    n : int
        Realizations per plan and truth.
    seed : int
        Seeds the truths; plan ``i`` on truth ``k`` simulates (and draws its sampling error) with seed
        ``s(s(seed, k), i + 1)``, where ``s(seed, k)`` is the seed of realization ``k`` of a run.
    blocks, window, groups : optional
        Summary rows, as in `SGS.simulate`; by default the targets.
    sampling_error : float
        Relative standard deviation of the sampling error, 0 for exact samples.
    existing : bool
        Refit on the existing data plus the virtual samples (infill); otherwise on the virtual samples alone
        (greenfield).
    composite_length : float, optional
        Composite length of the planned holes; by default the median distance between consecutive samples of
        a hole in the fitted data, which needs `holes` at fit. ``numpy.inf`` gives one composite per hole, as
        for a 2D model.
    quantiles : (float, float)
        Symmetric quantiles whose half-width over the mean is the MEE; (0.05, 0.95) gives 90 % confidence.
    tolerances : sequence of float
        Relative tolerances of the precision columns.
    progress : bool
        Show a `tqdm` bar over the (plan, truth) pairs.

    Returns
    -------
    Table
        One row per plan, truth and summary row: ``plan`` (the name, or the spacing as ``"10"`` or
        ``"10x20"``), ``spacing`` (``sqrt(along * across)``, the side of the square with the same area per hole;
        NaN for `plans`), ``realization`` (the truth's index), ``row`` (the row index, or the label of
        `groups`), ``truth``, ``mean``, ``mee`` (`SimulationSummary.relative_error`), ``cv``,
        ``precision_<r>`` per tolerance, ``error`` and ``covered`` (as in `SimulationSummary.validate`).

    Raises
    ------
    InvalidInput
        If neither or both of `spacings` and `plans` are given, the geometry arguments come with `plans`, the
        simulator is not fitted or carries a trend, domains or a secondary, `targets` is not a BlockModel,
        `quantiles` is not a symmetric pair, `composite_length` is missing without fitted holes, or a plan has
        no composite inside the targets.

    Notes
    -----
    The realizations of each simulation run in parallel; plans and truths run one after the other, so the
    result does not depend on the thread count.

    Examples
    --------
    >>> study = bt.spacing_study(sim, nodes, spacings=[10, 20], truths=3, window=(40, 40), progress=False)
    >>> curve = bt.uncertainty_curve("spacing", "mee", data=study)
    """
    if not isinstance(simulator, (_boitata.SGS, _boitata.DSS, _boitata.TurningBands)):
        raise InvalidInput("simulator must be a fitted SGS, DSS or TurningBands")
    if not isinstance(targets, _boitata.BlockModel):
        raise InvalidInput("targets must be a BlockModel")
    if (spacings is None) == (plans is None):
        raise InvalidInput("give one of spacings or plans")
    geometry = (rotation, azimuth, dip, topography) != (0.0, 0.0, 90.0, None)
    if plans is not None and geometry:
        raise InvalidInput("rotation, azimuth, dip and topography go with spacings")
    data = _fitted_data(simulator)
    lo, hi = sorted(float(q) for q in quantiles) if len(quantiles) == 2 else (math.nan, math.nan)
    if not (0 < lo < hi < 1 and abs(lo + hi - 1) < 1e-9):
        raise InvalidInput("quantiles must be a symmetric pair such as (0.05, 0.95)")
    if not (math.isfinite(sampling_error) and sampling_error >= 0):
        raise InvalidInput("sampling_error must be finite and non-negative")
    if n < 1:
        raise InvalidInput("n must be at least 1")
    indices = _truth_indices(truths)
    if composite_length is None:
        composite_length = _median_interval(data)
    elif not composite_length > 0:
        raise InvalidInput("composite_length must be positive")

    if spacings is not None:
        named = []
        for s in spacings:
            along, across = (s, s) if np.ndim(s) == 0 else s
            name = f"{along:g}" if along == across else f"{along:g}x{across:g}"
            holes = _boitata.planned_drillholes(
                targets, (along, across), rotation=rotation, azimuth=azimuth, dip=dip, topography=topography
            )
            named.append((name, math.sqrt(along * across), holes))
    else:
        named = [(str(name), math.nan, holes) for name, holes in plans.items()]
    samples = [
        (name, spacing, *_composites(name, holes, composite_length, targets))
        for name, spacing, holes in named
    ]

    truth = simulator.simulate(targets, n=max(indices) + 1, seed=seed, keep=indices, progress=False)
    at_nodes = _kept(truth, indices)
    rows = {"blocks": blocks, "window": window, "groups": groups}
    at_rows, labels = _boitata._summary_rows(targets, at_nodes, **rows)

    out = {}
    with tqdm(total=len(samples) * len(indices), disable=not progress) as bar:
        for i, (name, spacing, coords, ids, hit) in enumerate(samples):
            for j, k in enumerate(indices):
                pair = _boitata._realization_seed(_boitata._realization_seed(seed, k), i + 1)
                z = at_nodes[j][hit]
                if sampling_error > 0:
                    z = z * (1 + np.random.default_rng(pair).standard_normal(len(z)) * sampling_error)
                ok = np.isfinite(z)
                xyz, values, holes = coords[ok], z[ok], ids[ok]
                if existing:
                    xyz = np.vstack([data["coords"], xyz])
                    values = np.concatenate([data["values"], values])
                    holes = np.concatenate([data["holes"], holes])
                refit = copy.copy(simulator).fit(xyz, values, holes=holes)
                s = refit.simulate(
                    targets,
                    n=n,
                    seed=pair,
                    quantiles=[lo, hi],
                    tolerances=list(tolerances),
                    progress=False,
                    **rows,
                )
                check = s.validate(at_rows[j], confidence=hi - lo)
                m = len(s.mean)
                columns = {
                    "plan": [name] * m,
                    "spacing": np.full(m, spacing),
                    "realization": np.full(m, float(k)),
                    "row": labels if labels is not None else np.arange(m, dtype=float),
                    "truth": check["truth"],
                    "mean": check["mean"],
                    "mee": s.relative_error(confidence=hi - lo),
                    "cv": s.cv,
                    **{f"precision_{r:g}": s.precision[:, t] for t, r in enumerate(tolerances)},
                    "error": check["error"],
                    "covered": check["covered"],
                }
                for key, value in columns.items():
                    out.setdefault(key, []).append(np.asarray(value, dtype=object if key == "plan" else None))
                bar.update()
    return _boitata.Table({key: np.concatenate(parts) for key, parts in out.items()})


def _fitted_data(simulator):
    """Coordinates, values and hole names of the data `simulator` was fitted on."""
    _, columns = simulator._state()
    if columns is None:
        raise InvalidInput("simulator is not fitted; call fit first")
    columns = dict(columns)
    extra = sorted({"trend", "domain", "secondary"} & columns.keys())
    if extra:
        raise InvalidInput(f"spacing_study needs a simulator fitted without {', '.join(extra)}")
    coords = np.column_stack([np.asarray(columns[a], dtype=float) for a in "xyz"])
    codes = columns.get("hole", [None] * len(coords))
    known = all(c is not None for c in codes)
    holes = np.array(
        [f"existing {int(c)}" if known else f"existing sample {i}" for i, c in enumerate(codes)], dtype=object
    )
    return {
        "coords": coords,
        "values": np.asarray(columns["value"], dtype=float),
        "holes": holes,
        "codes": np.asarray(codes, dtype=float) if known else None,
    }


def _median_interval(data):
    codes = data["codes"]
    if codes is None:
        raise InvalidInput("give composite_length: the simulator was fitted without holes")
    same = codes[1:] == codes[:-1]
    steps = np.linalg.norm(np.diff(data["coords"], axis=0), axis=1)[same]
    if not len(steps):
        raise InvalidInput("give composite_length: no hole of the fitted data has two samples")
    return float(np.median(steps))


def _truth_indices(truths):
    if np.ndim(truths) == 0:
        if int(truths) != truths or truths < 1:
            raise InvalidInput("truths must be a positive count or realization indices")
        return list(range(int(truths)))
    indices = [int(k) for k in np.asarray(truths).ravel()]
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        raise InvalidInput("truths must be distinct non-negative realization indices")
    return indices


def _composites(name, holes, length, targets):
    """Centres of `holes`' composites inside `targets`, their hole names and the target holding each."""
    points = holes.composite(length, [])
    where = targets.row_at(points.coords)
    inside = where >= 0
    if not inside.any():
        raise InvalidInput(f"plan {name}: no composite lies inside the targets")
    ids = np.asarray(points[points.attributes.column_names[0]], dtype=object)
    return points.coords[inside], ids[inside], where[inside]


def _kept(summary, indices):
    """The realizations of `summary` in the order of `indices`."""
    order = {k: r for r, k in enumerate(summary.kept)}
    return summary.realizations[[order[k] for k in indices]]
