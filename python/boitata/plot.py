"""Plots on matplotlib (the ``plot`` extra).

Every function draws on `ax` when given, else on a new figure, and returns ``(fig, ax)``; `scatter_matrix` and
`variograms` take and return a grid of `axes` instead, and `category_colors` and `category_legend` return what they
make.
"""

import numpy as np

from boitata._boitata import closure as _closure
from boitata._boitata import clr as _clr
from boitata._boitata import correlation as _correlation
from boitata._boitata import describe, normal_cdf, normal_ppf
from boitata._boitata import swath as _swath
from boitata._columns import column as _column
from boitata._columns import stack as _stack
from boitata.errors import InvalidInput

__all__ = [
    "biplot",
    "boxplot",
    "category_colors",
    "category_legend",
    "category_swath",
    "cdf",
    "completeness",
    "conditional",
    "contact",
    "correlation",
    "correlation_reproduction",
    "cross_validation",
    "declustering",
    "directions",
    "domain_change",
    "fence",
    "grade_tonnage",
    "histogram",
    "histogram_reproduction",
    "holes",
    "paired_bias",
    "probability",
    "proportions",
    "qq",
    "scatter",
    "scatter_matrix",
    "section",
    "slab",
    "strip_log",
    "swath",
    "ternary",
    "transition_mds",
    "uncertain",
    "uncertainty_curve",
    "variogram",
    "variogram_reproduction",
    "variogram_volume",
    "variograms",
]


def _axes(ax):
    if ax is not None:
        return ax.figure, ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "boitata.plot needs matplotlib: pip install 'boitata[plot]' or conda install -c conda-forge matplotlib"
        ) from e
    return plt.subplots()


def _accent():
    import matplotlib as mpl

    return mpl.rcParams["axes.prop_cycle"].by_key()["color"][0]


def _finite(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(values)
    return values[ok], weights[ok] / weights[ok].sum()


def _weights(data, weights):
    if weights is None and hasattr(data, "volumes"):
        return data.volumes
    return _column(data, weights, "weights")


def _quantiles(values, weights, p):
    s = describe(values, weights=weights, quantiles=p)
    return np.array([v for k, v in s.items() if k.startswith("P")])


def _stats(ax, series, labels, corner):
    rows = [("n", "{:,}"), ("mean", "{:.3g}"), ("CV", "{:.2f}"), ("P10", "{:.3g}"), ("P50", "{:.3g}")]
    rows.append(("P90", "{:.3g}"))
    columns = []
    for values, weights in series:
        s = describe(values, weights=weights, quantiles=[0.1, 0.5, 0.9])
        numbers = [s["n"], s["mean"], s["cv"], s["P10"], s["P50"], s["P90"]]
        columns.append([f.format(x) for (_, f), x in zip(rows, numbers, strict=True)])
    table = [[name] + [c[i] for c in columns] for i, (name, _) in enumerate(rows)]
    if len(series) > 1:
        table.insert(0, [""] + [str(label or k) for k, label in enumerate(labels)])
    widths = [max(len(r[j]) for r in table) for j in range(len(table[0]))]
    text = "\n".join(
        " ".join(cell.ljust(w) if j == 0 else cell.rjust(w) for j, (cell, w) in enumerate(zip(r, widths)))
        for r in table
    )
    x, ha = (0.97, "right") if "right" in corner else (0.03, "left")
    y, va = (0.97, "top") if "upper" in corner else (0.03, "bottom")
    box = {"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2}
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, family="monospace", fontsize=7, bbox=box)


def histogram(values, *, weights=None, bins=40, log=False, stats=False, data=None, ax=None, **kwargs):
    """Histogram of relative frequencies, declustered by `weights`; `log` bins on a log axis.

    Parameters
    ----------
    values : str or array_like
        Values; NaN is ignored.
    weights : str or array_like, optional
        Declustering weights.
    bins : int or array_like
        Number of bins or their edges.
    log : bool
        Log-spaced bins over the positive values and a log x axis.
    stats : bool
        Write the count, weighted mean, CV and P10, P50, P90 in the upper right corner.
    data : PointSet, BlockModel, Table or mapping, optional
        Container whose columns `values` and `weights` may name; a block model's volumes are the default weights.
    **kwargs
        Passed to ``ax.hist``.
    """
    fig, ax = _axes(ax)
    _axis_names(ax, data, x=values)
    values, weights = _column(data, values), _weights(data, weights)
    if stats:
        _stats(ax, [(values, weights)], [None], "upper right")
    v, w = _finite(values, weights)
    if log:
        v, w = v[v > 0], w[v > 0]
        if np.isscalar(bins):
            bins = np.geomspace(v.min(), v.max(), bins + 1)
        ax.set_xscale("log")
    kwargs.setdefault("edgecolor", "white")
    kwargs.setdefault("linewidth", 0.5)
    ax.hist(v, bins, weights=w, **kwargs)
    ax.set_ylabel("Frequency")
    return fig, ax


def probability(values, *, weights=None, log=False, cap=None, fences=None, data=None, ax=None, **kwargs):
    """Cumulative probability on a normal scale: a Gaussian (or, with `log`, lognormal) distribution is a line.

    Parameters
    ----------
    values : str or array_like
        Values; NaN is ignored.
    weights : str or array_like, optional
        Declustering weights.
    log : bool
        Log x axis.
    cap : float, optional
        Top cut, drawn as a dashed vertical line.
    fences : float, optional
        Outlier fences ``P25 - fences × IQR`` and ``P75 + fences × IQR`` (1.5 is customary, 3 for far outliers),
        from weighted quartiles of the values, or of their logarithm with `log`; drawn as dotted vertical lines
        labeled with the count of values beyond each.
    data : PointSet, BlockModel, Table or mapping, optional
        As in `histogram`.
    **kwargs
        Passed to ``ax.plot``.
    """
    fig, ax = _axes(ax)
    v, w = _finite(_column(data, values), _weights(data, weights))
    order = np.argsort(v)
    v, w = v[order], w[order]
    p = np.cumsum(w) - w / 2
    kwargs.setdefault("marker", ".")
    kwargs.setdefault("linestyle", "none")
    ax.plot(v, normal_ppf(p), **kwargs)
    ticks = np.array([0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    ticks = ticks[(ticks >= p[0]) & (ticks <= p[-1])]
    ax.set_yticks(normal_ppf(ticks), [f"{100 * t:g}" for t in ticks])
    _axis_names(ax, data, x=values)
    ax.set_ylabel("Cumulative probability (%)")
    if cap is not None:
        ax.axvline(cap, color="0.5", lw=0.8, ls="--", label=f"cap {cap:.3g}")
    if fences is not None:
        keep = v > 0 if log else np.full(v.size, True)
        x = np.log10(v[keep]) if log else v
        q1, q3 = _quantiles(x, w[keep], [0.25, 0.75])
        spread = fences * (q3 - q1)
        for fence, n in ((q1 - spread, np.sum(x < q1 - spread)), (q3 + spread, np.sum(x > q3 + spread))):
            fence = 10.0**fence if log else fence
            if v[0] <= fence <= v[-1]:
                ax.axvline(fence, color="0.3", lw=0.8, ls=":", label=f"fence {fence:.3g}, {n} beyond")
    if log:
        ax.set_xscale("log")
    return fig, ax


def cdf(values, *, weights=None, labels=None, log=False, stats=False, data=None, ax=None, **kwargs):
    """Cumulative distribution of one or several series overlaid, e.g. domains, or clustered and declustered.

    Parameters
    ----------
    values : str, array_like or list of them
        One series, or several; NaN is ignored.
    weights : array_like or list, optional
        Declustering weights: an array for one series; for several, one array or None each.
    labels : list of str, optional
        Legend entries.
    log : bool
        Log x axis.
    stats : bool
        Write the count, weighted mean, CV and P10, P50, P90 of each series in the upper left corner, which a
        cumulative distribution leaves empty.
    data : PointSet, BlockModel, Table or mapping, optional
        Container whose columns `values` and `weights` may name; a block model's volumes are the default weights.
    **kwargs
        Passed to every ``ax.step``.
    """
    fig, ax = _axes(ax)
    if isinstance(values, str) or np.ndim(values[0]) == 0:
        _axis_names(ax, data, x=values)
        values, weights = [values], [_weights(data, weights)]
    values = [_column(data, v) for v in values]
    weights = [_weights(data, None)] * len(values) if weights is None else [_column(data, w) for w in weights]
    labels = labels or [None] * len(values)
    if stats:
        _stats(ax, list(zip(values, weights, strict=True)), labels, "upper left")
    for v, w, label in zip(values, weights, labels, strict=True):
        v, w = _finite(v, w)
        order = np.argsort(v)
        ax.step(v[order], np.cumsum(w[order]), where="post", label=label, **kwargs)
    if any(labels):
        ax.legend(loc="lower right" if stats else "best")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cumulative probability")
    if log:
        ax.set_xscale("log")
    return fig, ax


def qq(x, y, *, x_weights=None, y_weights=None, quantiles=None, log=False, data=None, ax=None, **kwargs):
    """Quantiles of `y` against the same quantiles of `x`, with the 1:1 line on which equal distributions lie.

    Parameters
    ----------
    x, y : str or array_like
        Two samples, e.g. one domain and another, or composites and blocks; NaN is ignored.
    x_weights, y_weights : str or array_like, optional
        Declustering weights of each.
    quantiles : array_like, optional
        Probabilities; default 0.01 to 0.99 in steps of 0.01.
    log : bool
        Log axes.
    data : PointSet, BlockModel, Table or mapping, optional
        As in `histogram`, for both samples.
    **kwargs
        Passed to ``ax.plot``.
    """
    fig, ax = _axes(ax)
    p = np.linspace(0.01, 0.99, 99) if quantiles is None else quantiles
    qx = _quantiles(_column(data, x, "x"), _weights(data, x_weights), p)
    qy = _quantiles(_column(data, y, "y"), _weights(data, y_weights), p)
    kwargs.setdefault("marker", ".")
    kwargs.setdefault("linestyle", "none")
    ax.plot(qx, qy, **kwargs)
    lo, hi = min(qx.min(), qy.min()), max(qx.max(), qy.max())
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--", label="1:1")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    return fig, ax


def boxplot(
    values, categories, *, weights=None, sort=False, log=False, scheme=None, data=None, ax=None, **kwargs
):
    """One box per category: P25 to P75, median, P10 to P90 whiskers and the mean, all weighted.

    Parameters
    ----------
    values : str or array_like
        Values; NaN is ignored.
    categories : array_like
        Category (e.g. domain) of each value, its code when `scheme` is given; each box is labeled with its count.
    weights : str or array_like, optional
        Declustering weights.
    sort : bool
        Order boxes by median, else by category.
    log : bool
        Log value axis.
    scheme : Categories, optional
        Names and colors of the categories.
    data : PointSet, BlockModel, Table or mapping, optional
        Container whose columns `values`, `categories` and `weights` may name; a block model's volumes are the
        default weights.
    **kwargs
        Passed to ``ax.bxp``.
    """
    import matplotlib as mpl

    fig, ax = _axes(ax)
    names, colors, index, keep = _classes(_column(data, categories, "categories"), scheme)
    values = np.asarray(_column(data, values), dtype=float)[keep]
    weights = _weights(data, weights)
    weights = None if weights is None else np.asarray(weights, dtype=float)[keep]
    stats, codes = [], []
    for c in np.unique(index[~np.isnan(values)]):
        s = describe(values[index == c], weights=None if weights is None else weights[index == c])
        p10, q1, med, q3, p90 = (s[k] for k in ("P10", "P25", "P50", "P75", "P90"))
        stats.append(
            {
                "label": f"{names[c]}\nn = {s['n']:,}",
                "whislo": p10,
                "q1": q1,
                "med": med,
                "q3": q3,
                "whishi": p90,
                "mean": s["mean"],
                "fliers": [],
            }
        )
        codes.append(c)
    if sort:
        order = np.argsort([s["med"] for s in stats], kind="stable")
        stats, codes = [stats[i] for i in order], [codes[i] for i in order]
    color = mpl.rcParams["axes.prop_cycle"].by_key()["color"][0]
    kwargs.setdefault("patch_artist", True)
    kwargs.setdefault("showmeans", True)
    kwargs.setdefault("boxprops", {"facecolor": mpl.colors.to_rgba(color, 0.25), "edgecolor": color})
    kwargs.setdefault("medianprops", {"color": color, "lw": 2})
    kwargs.setdefault("whiskerprops", {"color": color})
    kwargs.setdefault("capprops", {"color": color})
    kwargs.setdefault("meanprops", {"marker": "o", "ms": 4, "mfc": "white", "mec": color})
    boxes = ax.bxp(stats, **kwargs)["boxes"]
    if colors is not None:
        for box, c in zip(boxes, codes, strict=True):
            box.set_facecolor(mpl.colors.to_rgba(colors[c], 0.6))
    if log:
        ax.set_yscale("log")
    return fig, ax


def variogram(experimental, *, variogram=None, direction=None, ax=None, **kwargs):
    """Experimental variogram, points sized by pair count, with `variogram` and its sill.

    Parameters
    ----------
    experimental : ExperimentalVariogram
    variogram : Variogram, optional
    direction : tuple of float, optional
        ``(azimuth, dip)`` in degrees along which an anisotropic `variogram` is drawn;
        without it, lags are taken as distances in the model's isotropic space.
    **kwargs
        Passed to ``ax.scatter`` and, for its color, to the model line.
    """
    fig, ax = _axes(ax)
    keep = np.asarray(experimental.counts) > 0
    lags, gammas = experimental.lags[keep], experimental.gammas[keep]
    kwargs.setdefault("s", np.sqrt(experimental.counts[keep]) * 2)
    points = ax.scatter(lags, gammas, **kwargs)
    if variogram is not None:
        h = np.linspace(0, lags.max() * 1.05, 201)[1:]
        if direction is None:
            gamma = variogram.gamma(h)
        else:
            az, dip = np.radians(direction[0]), np.radians(direction[1])
            unit = np.array([np.cos(dip) * np.sin(az), np.cos(dip) * np.cos(az), -np.sin(dip)])
            gamma = variogram.gamma_between(np.zeros((h.size, 3)), h[:, None] * unit)
        ax.plot(h, gamma, color=points.get_facecolor()[0])
        ax.axhline(variogram.sill, color="0.5", lw=0.8, ls="--")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Lag distance")
    ax.set_ylabel("γ(h)")
    return fig, ax


def variograms(variograms, *, model=None, labels=None, axes=None, **kwargs):
    """Every direct and cross variogram of a `VariogramSet` as a matrix of panels, with `model` if given.

    Panel ``(i, j)`` holds the direct variogram of variable ``i`` on the diagonal and the cross variogram of
    ``i`` and ``j`` above it; the lower triangle is left empty. Points are sized by their share of the largest pair
    count, one color per direction.

    Parameters
    ----------
    variograms : VariogramSet
    model : Coregionalization, optional
        Drawn as ``C_ij(0) - C_ij(h)`` along each direction, or east when the set is omnidirectional.
    labels : list of str, optional
        Variable names; default the set's column names or indices.
    axes : array of Axes, optional
        ``(nvar, nvar)`` axes to draw on; default a new figure.
    **kwargs
        Passed to every ``ax.scatter``.

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes
        ``(nvar, nvar)``.
    """
    import matplotlib as mpl

    n = variograms.nvar
    if axes is None:
        fig, _ = _axes(None)
        fig.clear()
        fig.set_size_inches(2.3 * n + 0.5, 1.9 * n + 0.3)
        fig.set_layout_engine("constrained")
        axes = fig.subplots(n, n, squeeze=False)
    axes = np.asarray(axes)
    fig = axes.flat[0].figure
    labels = labels or [str(i) if name is None else name for i, name in enumerate(variograms.names)]
    directions = variograms.directions or [(90.0, 0.0)]
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j < i:
                ax.set_axis_off()
                continue
            entries = variograms[i, j]
            entries = entries if variograms.directions else [entries]
            for (azimuth, dip), e, color in zip(directions, entries, colors * len(entries), strict=False):
                keep = e.counts > 0
                size = 4 + 30 * np.sqrt(e.counts[keep] / e.counts.max())
                style = {"s": size, "color": color, "linewidths": 0} | kwargs
                label = f"{azimuth:g}°" if variograms.directions else None
                ax.scatter(e.lags[keep], e.gammas[keep], label=label, **style)
                if model is not None and keep.any():
                    h = np.linspace(0, e.lags[keep].max() * 1.05, 101)[1:]
                    a, d = np.radians(azimuth), np.radians(dip)
                    unit = np.array([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), -np.sin(d)])
                    origin = np.zeros((h.size, 3))
                    c0 = model.cross_covariance(i, j, origin, origin)
                    ax.plot(
                        h, c0 - model.cross_covariance(i, j, origin, h[:, None] * unit), color=color, lw=1
                    )
            ax.set_xlim(left=0)
            if i == j:
                ax.set_ylim(bottom=0)
                ax.set_title(labels[i])
                ax.set_xlabel("Lag distance")
                ax.set_ylabel("γ(h)")
            else:
                ax.axhline(0, color="0.5", lw=0.6)
                ax.set_title(f"{labels[i]} × {labels[j]}")
    if variograms.directions:
        axes[0, 0].legend(fontsize=7, title="azimuth", title_fontsize=7)
    return fig, axes


def _variable(check, variable):
    if isinstance(variable, str):
        if variable not in check.names:
            raise InvalidInput(f"no variable {variable!r}; checked: {', '.join(check.names)}")
        return check.names.index(variable)
    return variable


def _band(label, band, n):
    return (
        f"{n} {label}" if tuple(band) == (0.0, 1.0) else f"{n} {label}, P{100 * band[0]:g}–P{100 * band[1]:g}"
    )


def histogram_reproduction(check, *, variable=0, scores=False, band=(0.0, 1.0), ax=None, **kwargs):
    """Cumulative distributions of the realizations as a band, against the declustered data's; for categories,
    the spread of each category's proportion across realizations against the data's.

    Parameters
    ----------
    check : RealizationCheck
        Result of ``check_realizations``.
    variable : int or str
        Variable to draw.
    scores : bool
        In the normal scores of the data, against the standard normal distribution, instead of data units.
    band : tuple of float
        Quantiles across realizations bounding the band; default their full range.
    **kwargs
        Passed to ``ax.fill_betweenx`` (``ax.vlines`` for categories).
    """
    fig, ax = _axes(ax)
    color = _accent()
    if check.categorical:
        shares = check.proportions
        lo, hi = np.quantile(shares, band, axis=0)
        x = np.arange(len(check.names))
        kwargs = {"color": "0.75", "lw": 6} | kwargs
        ax.vlines(x, lo, hi, label=_band("realizations", band, len(shares)), **kwargs)
        ax.plot(x, np.median(shares, axis=0), "_", color="0.3", ms=12, label="median")
        ax.plot(x, check.data_proportions, "o", color=color, ms=5, label="declustered data")
        ax.set_xticks(x, check.names)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Category")
        ax.set_ylabel("Proportion")
        ax.legend()
        return fig, ax
    v = _variable(check, variable)
    p = check.probabilities
    q = check.score_quantiles[v] if scores else check.quantiles[v]
    target = normal_ppf(p) if scores else check.data_quantiles[v]
    lo, hi = np.quantile(q, band, axis=0)
    kwargs = {"color": "0.85", "lw": 0} | kwargs
    ax.fill_betweenx(p, lo, hi, label=_band("realizations", band, len(q)), **kwargs)
    ax.plot(np.median(q, axis=0), p, color="0.5", lw=0.8, label="median realization")
    ax.plot(target, p, color=color, lw=1.4, label="standard normal" if scores else "declustered data")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{check.names[v]}, normal score" if scores else check.names[v])
    ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right")
    return fig, ax


def variogram_reproduction(check, *, variable=0, band=(0.0, 1.0), ax=None, **kwargs):
    """Experimental variograms of the realizations as a band per direction, with the data's as points and the
    model as a line.

    Parameters
    ----------
    check : RealizationCheck
        Result of ``check_realizations`` with ``lag`` and ``max_lag``.
    variable : int or str
        Variable, or category for its indicator, to draw.
    band : tuple of float
        Quantiles across realizations bounding each band; default their full range.
    **kwargs
        Passed to every ``ax.fill_between``.
    """
    import matplotlib as mpl

    if check.variograms is None:
        raise InvalidInput("the check holds no variograms; give check_realizations lag and max_lag")
    fig, ax = _axes(ax)
    v = _variable(check, variable)
    reals, data = check.variograms[v], check.data_variograms[v]
    model = None if check.models is None else check.models[v]
    directions = check.directions or [None]
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    for d, (direction, color) in enumerate(zip(directions, colors * len(directions), strict=False)):
        lags = reals[0][d].lags
        gammas = np.array([r[d].gammas for r in reals])
        lo, hi = np.quantile(gammas, band, axis=0)
        name = "omnidirectional" if direction is None else f"azimuth {direction[0]:g}°, dip {direction[1]:g}°"
        ax.fill_between(lags, lo, hi, **({"color": color, "alpha": 0.25, "lw": 0, "label": name} | kwargs))
        keep = data[d].counts > 0
        ax.plot(data[d].lags[keep], data[d].gammas[keep], "o", color=color, ms=4)
        if model is not None:
            h = np.linspace(0, lags.max() * 1.05, 201)[1:]
            if direction is None:
                gamma = model.gamma(h)
            else:
                az, dip = np.radians(direction[0]), np.radians(direction[1])
                unit = np.array([np.cos(dip) * np.sin(az), np.cos(dip) * np.cos(az), -np.sin(dip)])
                gamma = model.gamma_between(np.zeros((h.size, 3)), h[:, None] * unit)
            ax.plot(h, gamma, color=color, lw=1.2)
    ax.plot([], [], "o", color="0.4", ms=4, label="data")
    if model is not None:
        ax.plot([], [], color="0.4", lw=1.2, label="model")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Lag distance")
    ax.set_ylabel("γ(h)")
    ax.legend(loc="lower right")
    return fig, ax


def correlation_reproduction(check, *, band=(0.0, 1.0), ax=None, **kwargs):
    """Correlation of each pair of variables across realizations as a range, against the declustered data's.

    Parameters
    ----------
    check : RealizationCheck
        Result of ``check_realizations`` with several variables.
    band : tuple of float
        Quantiles across realizations bounding each range; default their full range.
    **kwargs
        Passed to ``ax.vlines``.
    """
    if check.correlations is None:
        raise InvalidInput("the check holds no correlations; check several variables together")
    fig, ax = _axes(ax)
    i, j = np.triu_indices(len(check.names), 1)
    r = check.correlations[:, i, j]
    lo, hi = np.quantile(r, band, axis=0)
    x = np.arange(len(i))
    ax.vlines(
        x, lo, hi, **({"color": "0.75", "lw": 6, "label": _band("realizations", band, len(r))} | kwargs)
    )
    ax.plot(x, np.median(r, axis=0), "_", color="0.3", ms=12, label="median")
    ax.plot(x, check.data_correlation[i, j], "o", color=_accent(), ms=5, label="declustered data")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(x, [f"{check.names[a]} × {check.names[b]}" for a, b in zip(i, j, strict=True)])
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Correlation")
    ax.legend()
    return fig, ax


def _principal(rotation):
    """Rows: major, semi-major and minor unit vectors (East, North, Up) of ``(azimuth, dip, rake)``."""
    (sa, ca), (sd, cd), (sr, cr) = ((np.sin(t), np.cos(t)) for t in np.radians(rotation))
    r1 = np.array([[sa, ca, 0], [-ca, sa, 0], [0, 0, 1]])
    r2 = np.array([[cd, 0, -sd], [0, 1, 0], [sd, 0, cd]])
    r3 = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    return r3 @ r2 @ r1


def variogram_volume(volume, *, plane="major-semi", ellipse=True, ax=None, **kwargs):
    """Slice of a variogram volume through two of its principal axes, with the range ellipse.

    Parameters
    ----------
    volume : VariogramVolume
    plane : {"major-semi", "major-minor", "semi-minor"}
        Principal axes spanning the slice, drawn along x and y.
    ellipse : bool
        Draw the range ellipse and the two axes.
    **kwargs
        Passed to ``ax.pcolormesh`` (e.g. ``cmap``, ``vmax``).
    """
    names = ("major", "semi", "minor")
    pair = plane.split("-")
    if len(pair) != 2 or not set(pair) <= set(names) or pair[0] == pair[1]:
        raise InvalidInput(f"plane must be two of {names} joined by '-', got {plane!r}")
    i, j = (names.index(p) for p in pair)
    axes = _principal(volume.rotation)
    lags = np.asarray(volume.lags)
    gammas = np.asarray(volume.gammas)
    n, width = lags.size, lags[1] - lags[0]
    s, t = np.meshgrid(lags, lags)
    cell = np.rint((s[..., None] * axes[i] + t[..., None] * axes[j]) / width).astype(int) + n // 2
    inside = np.all((cell >= 0) & (cell < n), axis=-1)
    image = np.full(s.shape, np.nan)
    image[inside] = gammas[tuple(cell[inside].T)]
    fig, ax = _axes(ax)
    kwargs.setdefault("shading", "nearest")
    ax.pcolormesh(lags, lags, image, **kwargs)
    if ellipse:
        a, b = volume.ranges[i], volume.ranges[j]
        angle = np.linspace(0, 2 * np.pi, 181)
        ax.plot(a * np.cos(angle), b * np.sin(angle), color="white", lw=1.2)
        ax.plot([-a, a], [0, 0], color="white", lw=0.8)
        ax.plot([0, 0], [-b, b], color="white", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(lags[0], lags[-1])
    ax.set_ylim(lags[0], lags[-1])
    ax.set_xlabel(f"Lag along {pair[0]} axis")
    ax.set_ylabel(f"Lag along {pair[1]} axis")
    return fig, ax


def scatter(x, y, *, line=True, data=None, ax=None, **kwargs):
    """`y` against `x` with the 1:1 line and, if `line`, the least-squares regression of `y` on `x`.

    The regression slope of true on estimated values below 1 flags conditional bias. `x` and `y` may name columns
    of `data`.
    """
    fig, ax = _axes(ax)
    _axis_names(ax, data, x=x, y=y)
    x, y = np.asarray(_column(data, x, "x"), dtype=float), np.asarray(_column(data, y, "y"), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    kwargs.setdefault("s", 6)
    kwargs.setdefault("alpha", 0.5)
    ax.scatter(x, y, **kwargs)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--", label="1:1")
    if line:
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(
            [lo, hi],
            [intercept + slope * lo, intercept + slope * hi],
            color="0.2",
            lw=1.2,
            label=f"slope {slope:.2f}",
        )
    return fig, ax


def ternary(data, *, parts=None, labels=None, grid=True, ax=None, **kwargs):
    """Compositions of three parts as points in an equilateral triangle, each vertex a pure part.

    Parameters
    ----------
    data : array_like, mapping, Table, PointSet or BlockModel
        ``(n, 3)`` parts, or a container holding the `parts` columns. Each row is closed before plotting, so the
        parts may be any three of a larger composition (a subcomposition).
    parts : list of 3 str, optional
        Columns of `data`, in vertex order: bottom left, bottom right, top.
    labels : list of 3 str, optional
        Vertex labels; default the part names.
    grid : bool
        Draw lines at every 20 % of each part.
    **kwargs
        Passed to ``ax.scatter``.
    """
    fig, ax = _axes(ax)
    x, labels = _stack(data, labels, parts)
    if x.shape[1] != 3:
        raise InvalidInput(f"a ternary diagram needs 3 parts, got {x.shape[1]}")
    ok = np.isfinite(x).all(axis=1) & (x >= 0).all(axis=1) & (x.sum(axis=1) > 0)
    x = x[ok] / x[ok].sum(axis=1, keepdims=True)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    if grid:
        for t in (0.2, 0.4, 0.6, 0.8):
            for i in range(3):
                a, b = corners[(i + 1) % 3], corners[(i + 2) % 3]
                ends = t * corners[i] + (1 - t) * np.array([a, b])
                ax.plot(*ends.T, color="0.88", lw=0.6, zorder=0)
    ax.plot(*np.vstack([corners, corners[:1]]).T, color="0.3", lw=0.8)
    kwargs.setdefault("s", 6)
    kwargs.setdefault("color", _accent())
    ax.scatter(*(x @ corners).T, **kwargs)
    for corner, label, offset in zip(corners, labels, ((-6, -12), (6, -12), (0, 6)), strict=True):
        ha = {-6: "right", 6: "left", 0: "center"}[offset[0]]
        ax.annotate(label, corner, xytext=offset, textcoords="offset points", ha=ha)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def biplot(data, *, parts=None, labels=None, ax=None, **kwargs):
    """Covariance biplot of the clr coordinates: samples as points, parts as rays from the origin.

    Ray length is the standard deviation of a part's clr coordinate; the distance between two ray tips is the standard
    deviation of the log-ratio of the two parts, so tips close together mark parts in near-constant proportion. The
    axes are the first two principal components, labeled with the share of total variance they hold.

    Parameters
    ----------
    data : array_like, mapping, Table, PointSet or BlockModel
        ``(n, D)`` positive parts, or a container holding the `parts` columns; rows with a missing part are skipped.
    parts : list of str, optional
        Columns of `data`; default all of them.
    labels : list of str, optional
        Ray labels; default the part names.
    **kwargs
        Passed to ``ax.scatter``.
    """
    fig, ax = _axes(ax)
    x, labels = _stack(data, labels, parts)
    x = x[np.isfinite(x).all(axis=1)]
    z = _clr(_closure(x))
    z -= z.mean(axis=0)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    n = len(z)
    share = s**2 / (s**2).sum()
    kwargs.setdefault("s", 4)
    kwargs.setdefault("color", "0.7")
    ax.scatter(*(u[:, :2] * np.sqrt(n - 1)).T, **kwargs)
    rays = vt[:2].T * s[:2] / np.sqrt(n - 1)
    for (dx, dy), label in zip(rays, labels, strict=True):
        ax.annotate("", (dx, dy), (0, 0), arrowprops={"arrowstyle": "->", "color": _accent(), "lw": 1.2})
        ax.annotate(
            label,
            (dx, dy),
            xytext=(4 * np.sign(dx), 4 * np.sign(dy)),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            color=_accent(),
        )
    ax.update_datalim(1.15 * rays)
    ax.autoscale_view()
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({100 * share[0]:.0f} % of variance)")
    ax.set_ylabel(f"PC2 ({100 * share[1]:.0f} % of variance)")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def correlation(
    data, *, columns=None, labels=None, weights=None, method="pearson", colorbar=True, ax=None, **kwargs
):
    """Correlation, rank correlation or covariance matrix as a heatmap, each cell written out.

    Parameters
    ----------
    data : array_like, mapping, Table, PointSet or BlockModel
        ``(n, d)`` values, or named columns; each pair uses the rows where both are present, as ``boitata.correlation`` does.
    columns : list of str, optional
        Columns of `data` to draw, in order; default all of them.
    labels : list of str, optional
        Axis labels; default the column names or indices.
    weights : array_like or str, optional
        Declustering weights, or their column in `data`.
    method : {"pearson", "spearman", "covariance"}
        What to show; correlations span -1 to 1, covariances ± their largest magnitude.
    colorbar : bool
        Add a color bar.
    **kwargs
        Passed to ``ax.imshow`` (e.g. ``cmap``).
    """
    fig, ax = _axes(ax)
    weights = _column(data, weights, "weights")
    data, labels = _stack(data, labels, columns)
    r = _correlation(data, weights=weights, method=method)
    top = np.nanmax(np.abs(r)) if method == "covariance" else 1.0
    kwargs.setdefault("cmap", "RdBu_r")
    im = ax.imshow(r, vmin=-top, vmax=top, **kwargs)
    for (i, j), x in np.ndenumerate(r):
        text = f"{x:.2f}" if method != "covariance" else f"{x:.3g}"
        ax.text(
            j, i, text, ha="center", va="center", fontsize=8, color="white" if abs(x) > 0.6 * top else "0.1"
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    if colorbar:
        name = {"pearson": "Correlation", "spearman": "Rank correlation", "covariance": "Covariance"}[method]
        fig.colorbar(im, ax=ax, shrink=0.8, label=name)
    return fig, ax


def declustering(result, *, naive=None, ax=None, **kwargs):
    """Declustered mean against cell size, the chosen size marked, and the naive mean for reference.

    Parameters
    ----------
    result : Declustering
        Result of ``boitata.cell_declustering`` scanning cell sizes.
    naive : float, optional
        Mean of the values without weights, drawn as a dotted line.
    **kwargs
        Passed to ``ax.plot``.
    """
    fig, ax = _axes(ax)
    if not len(result.sizes):
        raise InvalidInput("result has no cell size scan: give cell_declustering sizes, not cell_size")
    line = ax.plot(result.sizes, result.means, **kwargs)[0]
    if naive is not None:
        ax.axhline(naive, color="0.5", ls=":", lw=1)
        ax.annotate(
            f"naive mean {naive:.3g}", (1, naive), xycoords=("axes fraction", "data"), ha="right", va="bottom"
        )
    ax.plot(result.cell_size, result.mean, "o", color=line.get_color())
    ax.annotate(
        f"{result.mean:.3g} at {result.cell_size:.3g}",
        (result.cell_size, result.mean),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
    )
    ax.set_xlabel("Cell size")
    ax.set_ylabel("Declustered mean")
    return fig, ax


def conditional(x, y, *, bins=10, weights=None, log=False, data=None, ax=None, **kwargs):
    """Scatter of `y` against `x` with the mean and P10 to P90 of `y` in bins of `x`, drawn at each bin's median.

    Parameters
    ----------
    x, y : str or array_like
        Pairs of values; pairs with NaN are ignored.
    bins : int or array_like
        Number of bins holding about as many pairs each, or bin edges of `x`.
    weights : str or array_like, optional
        Declustering weights.
    log : bool
        Log axes; non-positive values are then ignored.
    data : PointSet, BlockModel, Table or mapping, optional
        Container whose columns `x`, `y` and `weights` may name; a block model's volumes are the default weights.
    **kwargs
        Passed to ``ax.scatter``.
    """
    fig, ax = _axes(ax)
    _axis_names(ax, data, x=x, y=y)
    x, y = np.asarray(_column(data, x, "x"), dtype=float), np.asarray(_column(data, y, "y"), dtype=float)
    weights = _weights(data, weights)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & ((x > 0) & (y > 0) if log else True)
    x, y, w = x[ok], y[ok], w[ok]
    edges = (
        np.quantile(x, np.linspace(0, 1, bins + 1)) if np.ndim(bins) == 0 else np.asarray(bins, dtype=float)
    )
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    inside = (x >= edges[0]) & (x <= edges[-1])
    rows = []
    for b in range(len(edges) - 1):
        keep = inside & (k == b)
        if keep.any():
            s = describe(y[keep], weights=w[keep], quantiles=[0.1, 0.9])
            rows.append((_quantiles(x[keep], w[keep], [0.5])[0], s["mean"], s["P10"], s["P90"]))
    cx, mean, p10, p90 = np.array(rows).T
    kwargs.setdefault("s", 4)
    kwargs.setdefault("color", "0.75")
    kwargs.setdefault("linewidths", 0)
    ax.scatter(x, y, **kwargs)
    color = _accent()
    ax.fill_between(cx, p10, p90, color=color, alpha=0.2, lw=0, label="P10 to P90")
    ax.plot(cx, mean, marker="o", ms=4, color=color, label="mean")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    return fig, ax


def completeness(data, *, columns=None, ax=None, **kwargs):
    """Rows by number of variables present, the complete rows in the accent color, each bar labeled.

    Parameters
    ----------
    data : array_like, mapping, Table, PointSet or BlockModel
        ``(n, d)`` values, or named columns; NaN is missing.
    columns : list of str, optional
        Columns of `data` to draw, in order; default all of them.
    **kwargs
        Passed to ``ax.bar``.
    """
    fig, ax = _axes(ax)
    data, _ = _stack(data, columns=columns)
    d = data.shape[1]
    counts = np.bincount(np.isfinite(data).sum(axis=1), minlength=d + 1)
    kwargs.setdefault("color", ["0.8"] * d + [_accent()])
    bars = ax.bar(np.arange(d + 1), counts, **kwargs)
    ax.bar_label(bars, [f"{c:,}" for c in counts], fontsize=7)
    ax.set_xticks(np.arange(d + 1))
    ax.set_xlabel(f"Variables present, of {d}")
    ax.set_ylabel("Rows")
    return fig, ax


def scatter_matrix(data, *, columns=None, labels=None, weights=None, log=False, bins=30, axes=None, **kwargs):
    """Pairwise scatters of the columns of `data`, with their histograms on the diagonal.

    Each scatter is annotated with the Pearson (``r``) and rank correlation of its pair, weighted by `weights` and
    over the rows where both values are present, as ``boitata.correlation`` computes them.

    Parameters
    ----------
    data : array_like, mapping, Table, PointSet or BlockModel
        ``(n, d)`` values, or named columns; NaN is ignored pair by pair.
    columns : list of str, optional
        Columns of `data` to draw, in order; default all of them.
    labels : list of str, optional
        Axis labels; default the column names or indices.
    weights : array_like or str, optional
        Declustering weights, or their column in `data`, used by the histograms and the correlations.
    log : bool or sequence of bool
        Log axes and bins, for every column or per column; non-positive values are then ignored.
    bins : int
        Number of histogram bins.
    axes : array of Axes, optional
        ``(d, d)`` axes to draw on; default a new figure.
    **kwargs
        Passed to every ``ax.scatter``.

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes
        ``(d, d)``; ``axes[i, j]`` has column ``j`` across and column ``i`` up.
    """
    weights = _column(data, weights, "weights")
    data, labels = _stack(data, labels, columns)
    d = data.shape[1]
    log = np.broadcast_to(log, d)
    for j in np.flatnonzero(log):
        data[data[:, j] <= 0, j] = np.nan
    pearson = _correlation(data, weights=weights)
    rank = _correlation(data, weights=weights, method="spearman")
    if axes is None:
        fig, _ = _axes(None)
        fig.clear()
        fig.set_size_inches(1.7 * d + 0.5, 1.7 * d + 0.3)
        fig.set_layout_engine("constrained")
        axes = fig.subplots(d, d, squeeze=False)
    axes = np.asarray(axes)
    fig = axes.flat[0].figure
    limits = []
    for j in range(d):
        v = data[:, j][np.isfinite(data[:, j])]
        lo, hi = (np.log10(v.min()), np.log10(v.max())) if log[j] else (v.min(), v.max())
        pad = 0.04 * (hi - lo) or 0.5
        limits.append(10.0 ** np.array([lo - pad, hi + pad]) if log[j] else np.array([lo - pad, hi + pad]))
    kwargs.setdefault("s", 4)
    kwargs.setdefault("alpha", 0.4)
    kwargs.setdefault("linewidths", 0)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            ax.set_xscale("log" if log[j] else "linear")
            ax.set_yscale("log" if log[i] else "linear")
            if i == j:
                edges = np.geomspace(*limits[j], bins + 1) if log[j] else np.linspace(*limits[j], bins + 1)
                bars = ax.twinx()
                histogram(data[:, j], weights=weights, bins=edges, log=log[j], ax=bars)
                bars.set_ylim(bottom=0)
                bars.yaxis.set_visible(False)
                bars.spines[:].set_visible(False)
            else:
                ax.scatter(data[:, j], data[:, i], **kwargs)
                text = f"r {pearson[i, j]:.2f}\nrank {rank[i, j]:.2f}"
                box = {"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1}
                ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=7, bbox=box)
            ax.set_xlim(limits[j])
            ax.set_ylim(limits[i])
            ax.tick_params(labelbottom=i == d - 1, labelleft=j == 0)
        axes[-1, i].set_xlabel(labels[i])
        axes[i, 0].set_ylabel(labels[i])
    return fig, axes


def section(
    model,
    values,
    *,
    axis="z",
    index=None,
    plane=None,
    origin=None,
    azimuth=90.0,
    dip=90.0,
    holes=None,
    width=None,
    resolution=None,
    colorbar=True,
    scheme=None,
    ax=None,
    **kwargs,
):
    """Slice of a block model across `axis` at cell `index` (default: the middle), or on any plane, at true scale,
    with the drill holes near it projected on top.

    Parameters
    ----------
    model : BlockModel
        Missing blocks are left blank. Drawn as true block edges (exact for masked and sub-blocked layouts)
        whenever the cut is normal to one of the model's own local axes: always along `axis`, and for an
        explicit `plane` only when it matches the model's `rotation`. Otherwise resampled onto a raster.
    values : str or array_like
        Column name, or one value per block.
    axis : {"x", "y", "z"}
        Axis normal to the slice.
    index : int, optional
        Cell index along `axis`.
    plane : tuple, optional
        ``(center, azimuth, dip)`` as in `slab`; replaces `axis` and `index`. The model is sampled on a raster in
        section coordinates.
    origin : array_like, optional
        ``(x, y, z)`` point on the plane; with `azimuth` and `dip`, the same as ``plane=(origin, azimuth, dip)``.
    azimuth, dip : float
        Bearing of the section line and dip of the plane through `origin`, in degrees (90 and 90: a vertical
        east-west section).
    holes : Drillholes, optional
        Traces within `width` of the plane, clipped to that slab, projected and labeled at their shallowest point
        in it.
    width : float, optional
        Full width of the slab around the plane that keeps `holes`; default the largest block edge.
    resolution : float, optional
        Raster step on `plane`; default half the smallest block edge.
    colorbar : bool
        Add a color bar labeled with the column name, or a legend with `scheme`.
    scheme : Categories, optional
        `values` are codes of these categories, drawn in their colors.
    **kwargs
        Passed to ``ax.imshow`` (e.g. ``cmap``, ``norm``, ``vmin``).
    """
    fig, ax = _axes(ax)
    if origin is not None:
        plane = (origin, azimuth, dip)
    resolved = plane if plane is not None else _axis_plane(model, axis, index)
    k = _aligned_axis(model, resolved)
    if k is not None:
        mappable = _blocks(ax, model, values, resolved, k, scheme, kwargs)
    else:
        (image,), extent = _image(ax, model, [values], axis, index, plane, resolution)
        _scheme_colors(scheme, kwargs)
        mappable = ax.imshow(image, origin="lower", extent=extent, **kwargs)
    if holes is not None:
        _traces(ax, holes, resolved, max(model.size) if width is None else width)
    _key(fig, ax, mappable, _name(model, values), colorbar, scheme)
    return fig, ax


def _traces(ax, holes, plane, width):
    from matplotlib.collections import LineCollection

    center, u, v, n = _frame(plane)
    paths = holes.paths()
    ids = np.asarray(paths[paths.column_names[0]], dtype=object)
    xyz = np.column_stack([np.asarray(paths[c], dtype=float) for c in "xyz"])
    for name in dict.fromkeys(ids):
        segments = _clip([xyz[ids == name]], center, n, width / 2) @ np.c_[u, v]
        if len(segments):
            ax.add_collection(LineCollection(segments, colors="tab:red", linewidths=1.2, label=str(name)))
            ax.annotate(str(name), segments[0, 0], xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.autoscale_view()


def _block_axes(rotation):
    p = _principal(rotation)
    return np.array([-p[1], p[0], p[2]])


def _angles_for_normal(axis):
    dip = np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0)))
    spread = np.hypot(axis[0], axis[1])
    azimuth = 90.0 if spread < 1e-12 else np.degrees(np.arctan2(-axis[1], axis[0]))
    return azimuth, dip


def _axis_plane(model, axis, index):
    k = {"x": 0, "y": 1, "z": 2}[axis]
    size, count = np.asarray(model.size, dtype=float), np.asarray(model.count, dtype=float)
    index = count[k] // 2 if index is None else float(index)
    local = size * count / 2
    local[k] = (index + 0.5) * size[k]
    axes = _block_axes(model.rotation)
    center = np.asarray(model.origin) + local @ axes
    azimuth, dip = _angles_for_normal(-axes[k] if k == 1 else axes[k])
    return tuple(center), azimuth, dip


def _aligned_axis(model, plane):
    n = _frame(plane)[3]
    matches = np.flatnonzero(np.abs(_block_axes(model.rotation) @ n) > 1 - 1e-9)
    return int(matches[0]) if len(matches) else None


def _blocks(ax, model, values, plane, k, scheme, kwargs):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    center, u, v, _ = _frame(plane)
    corners = model.corners
    depth = (corners - center) @ _block_axes(model.rotation)[k]
    keep = (depth.min(axis=1) <= 0) & (depth.max(axis=1) >= 0)
    if not keep.any():
        raise InvalidInput("the plane misses the block model")
    others = [a for a in range(3) if a != k]
    face = [sum(bit << others[i] for i, bit in enumerate(bits)) for bits in ((0, 0), (0, 1), (1, 1), (1, 0))]
    values = np.asarray(_column(model, values), dtype=float)
    keep &= np.isfinite(values)
    faces = corners[keep][:, face, :] @ np.c_[u, v]
    values = values[keep]
    _scheme_colors(scheme, kwargs)
    kwargs.setdefault("linewidths", 0)
    vmin, vmax = kwargs.pop("vmin", None), kwargs.pop("vmax", None)
    patches = PatchCollection([Polygon(f) for f in faces], **kwargs)
    patches.set_array(values)
    patches.set_clim(vmin, vmax)
    ax.add_collection(patches)
    ax.autoscale()
    _label(ax, u, v)
    return patches


def fence(
    model,
    values,
    positions,
    *,
    azimuth,
    dip=90.0,
    resolution=None,
    colorbar=True,
    scheme=None,
    axes=None,
    **kwargs,
):
    """One vertical (or dipping) section per row of `positions`, on the same `azimuth` and `dip`, sharing one
    color scale: a fence of parallel sections stepped along a corridor.

    Parameters
    ----------
    model : BlockModel
        As in `section`.
    values : str or array_like
        Column name, or one value per block.
    positions : array_like
        ``(n, 3)`` plane centers, one per panel, as `section`'s `plane` center.
    azimuth, dip : float
        Orientation shared by every panel's plane, in degrees; `dip` as in `section`'s `plane` (90 vertical).
    resolution : float, optional
        Raster step on each plane; default half the smallest block edge.
    colorbar : bool
        Add one color bar for every panel, labeled with the column name, or a legend with `scheme`.
    scheme : Categories, optional
        `values` are codes of these categories, drawn in their colors.
    axes : array of Axes, optional
        One per position; default a new figure.
    **kwargs
        Passed to every ``ax.imshow`` (e.g. ``cmap``, ``norm``, ``vmin``, ``vmax``).

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    n = len(positions)
    if axes is None:
        fig, _ = _axes(None)
        fig.clear()
        fig.set_size_inches(2.4 * n + 0.6, 4.4)
        fig.set_layout_engine("constrained")
        axes = fig.subplots(1, n, sharey=True, squeeze=False)[0]
    axes = np.asarray(axes)
    fig = axes.flat[0].figure
    images, extents = [], []
    for ax, center in zip(axes, positions, strict=True):
        (image,), extent = _image(ax, model, [values], "z", None, (center, azimuth, dip), resolution)
        images.append(image)
        extents.append(extent)
    _scheme_colors(scheme, kwargs)
    if scheme is None:
        kwargs.setdefault("vmin", np.nanmin([np.nanmin(image) for image in images]))
        kwargs.setdefault("vmax", np.nanmax([np.nanmax(image) for image in images]))
    ims = [
        ax.imshow(image, origin="lower", extent=extent, **kwargs)
        for ax, image, extent in zip(axes, images, extents, strict=True)
    ]
    for ax in axes[1:]:
        ax.set_ylabel("")
    _key(fig, axes[-1], ims[-1], _name(model, values), colorbar, scheme)
    return fig, axes


def _scheme_colors(scheme, kwargs):
    if scheme is not None:
        cmap, norm = category_colors(scheme)
        kwargs.setdefault("cmap", cmap)
        kwargs.setdefault("norm", norm)


def _name(data, column):
    """`column` and its unit in `data`, as an axis label; the unit alone for an array that carries one."""
    if not isinstance(column, str):
        unit = getattr(column, "unit", None)
        return f"({unit})" if isinstance(unit, str) else ""
    unit = getattr(data, "units", {}).get(column)
    return f"{column} ({unit})" if unit else column


def _axis_names(ax, data, x=None, y=None):
    """Labels the axes with the columns `x` and `y` name, if they name one."""
    if _name(data, x):
        ax.set_xlabel(_name(data, x))
    if _name(data, y):
        ax.set_ylabel(_name(data, y))


def _key(fig, ax, mappable, label, colorbar, scheme):
    if not colorbar:
        return
    if scheme is None:
        fig.colorbar(mappable, ax=ax, shrink=0.8, label=label or None)
    else:
        category_legend(scheme, ax, loc="upper left", bbox_to_anchor=(1.01, 1))


def _image(ax, model, columns, axis, index, plane, resolution):
    if plane is None:
        return _slice(ax, model, columns, axis, index)
    center, u, v, n = _frame(plane)
    size = np.asarray(model.size, dtype=float)
    step = resolution or size.min() / 2
    reach = np.linalg.norm(size) / 2
    centroids = model.centroids
    near = centroids[np.abs((centroids - center) @ n) <= reach] @ np.c_[u, v]
    if not len(near):
        raise InvalidInput("the plane misses the block model")
    lo = near.min(0) - reach
    counts = np.ceil((near.max(0) + reach - lo) / step).astype(int)
    gu, gv = np.meshgrid(*(lo[a] + (np.arange(counts[a]) + 0.5) * step for a in (0, 1)))
    offset = center - (center @ u) * u - (center @ v) * v
    rows = model.row_at(offset + gu.reshape(-1, 1) * u + gv.reshape(-1, 1) * v).reshape(gu.shape)
    if not (rows >= 0).any():
        raise InvalidInput("the plane misses the block model")
    (j0, j1), (i0, i1) = (np.flatnonzero((rows >= 0).any(axis=a))[[0, -1]] for a in (0, 1))
    rows = rows[i0 : i1 + 1, j0 : j1 + 1]
    images = []
    for column in columns:
        values = np.asarray(_column(model, column), dtype=float)
        images.append(np.where(rows >= 0, values[rows], np.nan))
    _label(ax, u, v)
    (u0, v0), (u1, v1) = lo + step * np.array([j0, i0]), lo + step * np.array([j1 + 1, i1 + 1])
    return images, (u0, u1, v0, v1)


def _frame(plane):
    center, azimuth, dip = plane
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise InvalidInput("plane center must be (x, y, z)")
    az, dip = np.radians(azimuth), np.radians(dip)
    u = np.array([np.sin(az), np.cos(az), 0.0])
    v = np.cos(dip) * np.array([-np.cos(az), np.sin(az), 0.0]) + np.array([0.0, 0.0, np.sin(dip)])
    return center, u, v, np.cross(u, v)


def _label(ax, u, v):
    names = ("Easting (m)", "Northing (m)", "Elevation (m)")
    for axis, w, default in ((ax.xaxis, u, "Along strike (m)"), (ax.yaxis, v, "Up dip (m)")):
        aligned = np.isclose(w, 1.0, atol=1e-9)
        axis.set_label_text(names[aligned.argmax()] if aligned.any() else default)
    ax.set_aspect("equal")


def slab(
    coords,
    values=None,
    *,
    plane,
    thickness,
    meshes=None,
    lines=None,
    labels=None,
    colorbar=True,
    scheme=None,
    ax=None,
    **kwargs,
):
    """Points within a slab around a plane, in section coordinates, with mesh traces, clipped lines and labels.

    Section coordinates are world coordinates along the strike and up the dip of the plane, so a vertical
    east-west section reads easting and elevation, and a horizontal plane easting and northing.

    Parameters
    ----------
    coords : PointSet or array_like
        ``(n, 3)`` coordinates.
    values : str or array_like, optional
        Column of `coords`, or one value per point, coloring them; default one color.
    plane : tuple
        ``(center, azimuth, dip)``: a point on the plane, the bearing of the section line and the dip of the plane
        (90 is vertical, 0 a plan), in degrees.
    thickness : float
        Full width of the slab.
    meshes : Mesh or list of Mesh, optional
        Drawn as their intersection with the plane.
    lines : list of array_like, optional
        ``(m, 3)`` polylines, e.g. drill-hole traces, clipped to the slab.
    labels : sequence of str, optional
        One per point; each distinct label is written once, above its highest point in the slab.
    colorbar : bool
        With `values`, add a color bar labeled with the column name, or a legend with `scheme`.
    scheme : Categories, optional
        `values` are codes of these categories, drawn in their colors.
    **kwargs
        Passed to ``ax.scatter`` (e.g. ``s``, ``cmap``, ``norm``, ``color``).
    """
    from matplotlib.collections import LineCollection

    fig, ax = _axes(ax)
    center, u, v, n = _frame(plane)
    half = thickness / 2
    uv = np.c_[u, v]
    points = coords
    coords = np.asarray(getattr(points, "coords", points), dtype=float)
    near = np.abs((coords - center) @ n) <= half
    xy = coords[near] @ uv

    if meshes is not None:
        for mesh in [meshes] if hasattr(meshes, "triangles") else meshes:
            ax.add_collection(LineCollection(_trace(mesh, center, n) @ uv, colors="0.2", linewidths=1))
    if lines is not None:
        segments = _clip([np.asarray(line, dtype=float) for line in lines], center, n, half)
        ax.add_collection(LineCollection(segments @ uv, colors="0.75", linewidths=0.8))

    if values is not None:
        kwargs["c"] = np.asarray(_column(points, values), dtype=float)[near]
        _scheme_colors(scheme, kwargs)
    kwargs.setdefault("s", 6)
    drawn = ax.scatter(xy[:, 0], xy[:, 1], **kwargs)
    if values is not None:
        _key(fig, ax, drawn, _name(points, values), colorbar, scheme)

    if labels is not None:
        names = np.asarray(labels, dtype=object)[near]
        for name in dict.fromkeys(names):
            top = xy[names == name][:, 1].argmax()
            ax.annotate(
                name,
                xy[names == name][top],
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    ax.autoscale()
    _label(ax, u, v)
    return fig, ax


def _trace(mesh, center, normal):
    corners = mesh.vertices[mesh.triangles]
    d = (corners - center) @ normal
    a, b = [0, 1, 2], [1, 2, 0]
    crossing = (d[:, a] > 0) != (d[:, b] > 0)
    t = d[:, a] / np.where(crossing, d[:, a] - d[:, b], 1.0)
    cuts = corners[:, a] + t[..., None] * (corners[:, b] - corners[:, a])
    return cuts[crossing].reshape(-1, 2, 3)


def _clip(lines, center, normal, half):
    segments = np.concatenate(
        [np.empty((0, 2, 3))] + [np.stack([line[:-1], line[1:]], axis=1) for line in lines]
    )
    d = (segments - center) @ normal
    step = d[:, 1] - d[:, 0]
    flat = step == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ta, tb = (-half - d[:, 0]) / step, (half - d[:, 0]) / step
    start = np.where(flat, np.where(np.abs(d[:, 0]) <= half, 0.0, 1.0), np.maximum(0, np.minimum(ta, tb)))
    end = np.where(flat, 1.0, np.minimum(1, np.maximum(ta, tb)))
    keep = start < end
    s, t = segments[keep], np.c_[start, end][keep]
    return s[:, :1] + t[..., None] * (s[:, 1:] - s[:, :1])


def _slice(ax, model, columns, axis, index):
    names = []
    for i, column in enumerate(columns):
        if not isinstance(column, str):
            model = model.with_column(f"_{i}", column)
            column = f"_{i}"
        names.append(column)
    grid = model.to_regular()
    nx, ny, nz = grid.count
    (x0, y0, z0), (sx, sy, sz) = grid.origin, grid.size
    x, y, z = (x0, x0 + nx * sx), (y0, y0 + ny * sy), (z0, z0 + nz * sz)
    k = {"x": 2, "y": 1, "z": 0}[axis]
    index = (nz, ny, nx)[k] // 2 if index is None else index
    images = [np.take(np.asarray(grid[n], dtype=float).reshape(nz, ny, nx), index, axis=k) for n in names]
    extent, labels = {
        "z": ((*x, *y), ("X", "Y")),
        "y": ((*x, *z), ("X", "Z")),
        "x": ((*y, *z), ("Y", "Z")),
    }[axis]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal")
    return images, extent


def swath(swaths, *, labels=None, y="mean", ax=None, **kwargs):
    """Mean, tonnage or metal per slice of one or several `swath` results, with the first one's counts as light bars.

    Parameters
    ----------
    swaths : Table or list of Table
        Results of ``boitata.swath``, e.g. composites and blocks with the same width.
    labels : list of str, optional
        Legend entries.
    y : {"mean", "tonnage", "metal"}
        What to draw per slice.
    **kwargs
        Passed to every ``ax.plot``.
    """
    fig, ax = _axes(ax)
    swaths = [swaths] if hasattr(swaths, "column_names") else list(swaths)
    labels = labels or [None] * len(swaths)
    first = swaths[0]
    bars = ax.twinx()
    width = np.diff(first["center"]).min() if len(first["center"]) > 1 else 1.0
    bars.bar(first["center"], first["n"], width=width, color="0.9", zorder=0)
    bars.set_ylabel("Count", color="0.5")
    bars.tick_params(axis="y", colors="0.5")
    ax.set_zorder(bars.get_zorder() + 1)
    ax.patch.set_visible(False)
    kwargs.setdefault("marker", ".")
    for s, label in zip(swaths, labels, strict=True):
        ax.plot(s["center"], s[y], label=label, **kwargs)
    if any(labels):
        ax.legend()
    ax.set_xlabel("Distance along swath")
    ax.set_ylabel(y.capitalize())
    return fig, ax


def _palette(k):
    import matplotlib as mpl

    return mpl.colormaps[mpl.rcParams["image.cmap"]](np.linspace(0.05, 0.85, k))


def _classes(categories, scheme):
    """Names, colors (None without a scheme), per-sample index and mask of samples with a category."""
    if scheme is None:
        names, index = np.unique(np.asarray(categories), return_inverse=True)
        return [str(n) for n in names], None, index, np.ones(index.size, bool)
    codes = np.asarray(categories, dtype=float)
    keep = ~np.isnan(codes)
    index = codes[keep].astype(int)
    if np.any((index != codes[keep]) | (index < 0) | (index >= len(scheme))):
        raise InvalidInput(f"categories must be codes of the scheme, 0 to {len(scheme) - 1}, or NaN")
    return scheme.names, _colors(scheme), index, keep


def _colors(scheme):
    if scheme.colors is not None:
        return scheme.colors
    gray = ["0.6"] if scheme.other is not None else []
    return [*_palette(len(scheme) - len(gray)), *gray]


def category_colors(scheme):
    """Color map and norm drawing code i of `scheme` in its color i.

    Parameters
    ----------
    scheme : Categories
        Colors of the categories; when it has none, spread over matplotlib's ``image.cmap`` with ``other`` in
        gray.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
    norm : matplotlib.colors.BoundaryNorm
        Boundaries at every code ± 0.5; pass both as ``cmap=`` and ``norm=`` to ``imshow`` or ``scatter``.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    k = len(scheme)
    return ListedColormap(_colors(scheme)), BoundaryNorm(np.arange(k + 1) - 0.5, k)


def category_legend(scheme, ax, **kwargs):
    """Legend of `scheme`: one patch per category, in code order, in the colors of `category_colors`.

    Parameters
    ----------
    scheme : Categories
    ax : matplotlib.axes.Axes or matplotlib.figure.Figure
        Where the legend goes.
    **kwargs
        Passed to ``ax.legend`` (e.g. ``loc``, ``ncol``).

    Returns
    -------
    matplotlib.legend.Legend
    """
    from matplotlib.patches import Patch

    handles = [Patch(color=c, label=n) for n, c in zip(scheme.names, _colors(scheme), strict=True)]
    return ax.legend(handles=handles, **kwargs)


def category_swath(
    coords,
    categories,
    width,
    *,
    azimuth=None,
    axis=None,
    weights=None,
    colors=None,
    scheme=None,
    ax=None,
    **kwargs,
):
    """Proportion of each category per slice along a direction, as stacked bars.

    Parameters
    ----------
    coords : PointSet, BlockModel or array_like
        ``(n, 2)`` or ``(n, 3)`` coordinates; a container's columns may be named by `categories` and `weights`.
    categories : str or array_like
        Category (e.g. lithology) of each sample; its code, NaN for none, when `scheme` is given.
    width, azimuth, axis
        Slices as in ``boitata.swath``: `width` along `azimuth` (degrees from north) or `axis` ("x", "y", "z").
    weights : str or array_like, optional
        Declustering weights or lengths.
    colors : sequence, optional
        One color per category, sorted; default the scheme's, else spread over matplotlib's ``image.cmap``
        with ``other`` in gray.
    scheme : Categories, optional
        Order, names and colors of the categories.
    **kwargs
        Passed to every ``ax.bar``.
    """
    fig, ax = _axes(ax)
    names, default, index, keep = _classes(_column(coords, categories, "categories"), scheme)
    weights = _weights(coords, weights)
    weights = None if weights is None else np.asarray(weights, dtype=float)[keep]
    coords = np.asarray(getattr(coords, "coords", getattr(coords, "centroids", coords)), dtype=float)[keep]
    colors = colors if colors is not None else default if default is not None else _palette(len(names))
    kwargs.setdefault("edgecolor", "white")
    kwargs.setdefault("linewidth", 0.3)
    bottom = 0.0
    for code, (name, color) in enumerate(zip(names, colors, strict=True)):
        indicator = (index == code).astype(float)
        s = _swath(coords, indicator, width, azimuth=azimuth, axis=axis, weights=weights)
        ax.bar(s["center"], s["mean"], width=width, bottom=bottom, color=color, label=str(name), **kwargs)
        bottom = bottom + s["mean"]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Distance along swath")
    ax.set_ylabel("Proportion")
    return fig, ax


def proportions(categories, *, weights=None, scheme=None, data=None, ax=None, **kwargs):
    """Proportion of each category as horizontal bars, weighted, with the unweighted proportions as ticks.

    Parameters
    ----------
    categories : array_like
        Category of each sample; its code, NaN for none, when `scheme` is given.
    weights : str or array_like, optional
        Declustering weights or lengths; the bars are then weighted and a dark tick marks each unweighted share.
    scheme : Categories, optional
        Order, names and colors of the categories.
    data : PointSet, BlockModel, Table or mapping, optional
        Container whose columns `categories` and `weights` may name; a block model's volumes are the default
        weights.
    **kwargs
        Passed to ``ax.barh``.
    """
    fig, ax = _axes(ax)
    names, colors, index, keep = _classes(_column(data, categories, "categories"), scheme)
    weights = _weights(data, weights)
    w = np.ones(index.size) if weights is None else np.asarray(weights, dtype=float)[keep]
    share = np.bincount(index, w, len(names)) / w.sum()
    rows = np.arange(len(names))
    kwargs.setdefault("color", _accent() if colors is None else colors)
    kwargs.setdefault("alpha", 0.6)
    ax.barh(rows, share, **kwargs)
    right = share
    if weights is not None:
        naive = np.bincount(index, minlength=len(names)) / index.size
        ax.plot(naive, rows, "|", color="0.2", ms=11, mew=1.5, label="unweighted")
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1.02))
        right = np.maximum(share, naive)
    for row, x, p in zip(rows, right, share, strict=True):
        ax.annotate(
            f"{100 * p:.1f} %", (x, row), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7
        )
    ax.set_xlim(0, 1.25 * right.max())
    ax.set_yticks(rows, names)
    ax.invert_yaxis()
    ax.set_xlabel("Proportion")
    return fig, ax


def strip_log(drillholes, hole, *, columns=(), categories=(), scheme=None, runs=None, ax=None, **kwargs):
    """Holes as vertical logs side by side: category bars, grade profiles as steps and ore runs shaded.

    Parameters
    ----------
    drillholes : Drillholes
        With an interval table.
    hole : str or sequence of str
        Hole or holes to draw, left to right.
    columns : sequence of str
        Grade columns, one track each, drawn as steps from zero to the largest value over the holes drawn.
    categories : str or sequence of str
        Categorical columns (e.g. lithology), one bar each.
    scheme : Categories or dict, optional
        Names and colors of the categories, or a Categories per column name; values outside it are left
        blank. Without one, values are sorted and spread over matplotlib's ``image.cmap``.
    runs : Table, optional
        From ``Drillholes.runs``; ore runs are shaded across the grade tracks.
    **kwargs
        Passed to every grade ``ax.plot``.
    """
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    names = drillholes.interval_columns
    if names is None:
        raise InvalidInput("drillholes has no interval table")
    id_col, from_col, to_col = names
    holes = [hole] if isinstance(hole, str) else list(hole)
    unknown = sorted(set(holes) - set(drillholes.holes))
    if unknown:
        raise InvalidInput(f"unknown holes: {', '.join(unknown)}")
    categories = [categories] if isinstance(categories, str) else list(categories)
    columns = list(columns)
    samples = drillholes.samples()
    ids = np.asarray(samples[id_col], dtype=object)
    top, bottom = np.asarray(samples[from_col], dtype=float), np.asarray(samples[to_col], dtype=float)

    colors, found = {}, {}
    for c in categories:
        s = scheme.get(c) if isinstance(scheme, dict) else scheme
        if s is None:
            values = np.asarray(samples[c], dtype=object)[np.isin(ids, holes)]
            found[c] = sorted({str(v) for v in values if v is not None})
        else:
            colors[c] = dict(zip(s.names, _colors(s), strict=True))
    spread = iter(_palette(sum(map(len, found.values()))))
    for c, names_c in found.items():
        colors[c] = {n: next(spread) for n in names_c}
    handles = [Patch(color=p, label=f"{c} {n}") for c in categories for n, p in colors[c].items()]

    width = len(categories) + 2 * len(columns) + 0.6
    scale = {}
    for c in columns:
        v = np.asarray(samples[c], dtype=float)[np.isin(ids, holes)]
        scale[c] = np.nanmax(v) if np.any(v > 0) else 1.0
    fig, ax = _axes(ax)
    ticks, labels = [], []
    run_color = _accent()
    for k, h in enumerate(holes):
        x0 = k * width
        m = ids == h
        f, t = top[m], bottom[m]
        for j, c in enumerate(categories):
            values = np.asarray(samples[c], dtype=object)[m]
            color = [colors[c].get(str(v), "none") if v is not None else "none" for v in values]
            ax.bar(x0 + j + 0.5, t - f, width=0.9, bottom=f, color=color, align="center")
            ticks.append(x0 + j + 0.5)
            labels.append(c)
        g0 = x0 + len(categories) + 0.1
        if runs is not None and columns:
            r = np.asarray(runs[id_col], dtype=object) == h
            ore = r & np.asarray(runs["ore"], dtype=bool)
            ax.bar(
                g0 + len(columns),
                np.asarray(runs["to"])[ore] - np.asarray(runs["from"])[ore],
                width=2 * len(columns),
                bottom=np.asarray(runs["from"])[ore],
                color=run_color,
                alpha=0.15,
                linewidth=0,
            )
        order = np.argsort(f)
        for j, c in enumerate(columns):
            v = np.asarray(samples[c], dtype=float)[m][order]
            x = g0 + 2 * j + 1.8 * np.clip(v / scale[c], 0.0, 1.0)
            ys = np.column_stack([f[order], t[order]]).ravel()
            xs = np.repeat(x, 2)
            gap = np.flatnonzero(f[order][1:] > t[order][:-1] + 1e-9) + 1
            ys, xs = np.insert(ys, 2 * gap, np.nan), np.insert(xs, 2 * gap, np.nan)
            kw = {"color": "0.2", "linewidth": 0.8, **kwargs}
            ax.plot(xs, ys, **kw)
            ax.axvline(g0 + 2 * j, color="0.8", linewidth=0.5)
            ticks.append(g0 + 2 * j + 0.9)
            labels.append(f"{c}\n0-{scale[c]:.3g}")
        ax.text(
            x0 + (width - 0.6) / 2,
            1.01,
            h,
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="center",
            va="bottom",
        )
    if runs is not None and columns:
        handles.append(Patch(color=run_color, alpha=0.15, label="ore run"))
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_xlim(-0.3, len(holes) * width - 0.3)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_ylabel("Depth (m)")
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    return fig, ax


def holes(paths, *, plane=None, mark=None, hole="HOLE_ID", labels=True, ax=None, **kwargs):
    """Drill-hole traces, one line per hole, in plan or projected on a section.

    Parameters
    ----------
    paths : Table
        Result of ``Drillholes.paths``: a hole id column, ``depth``, ``x``, ``y`` and ``z``.
    plane : tuple, optional
        ``(center, azimuth, dip)`` as in `section`'s `plane`; projects traces into that section instead of
        drawing plan-view ``(x, y)``.
    mark : array_like, optional
        ``(m, 3)`` extra points scattered on top, e.g. flagged stations from ``Drillholes.at``.
    hole : str
        Column of `paths` naming each row's hole.
    labels : bool
        Label each hole at its shallowest row.
    **kwargs
        Passed to every ``ax.plot``.
    """
    fig, ax = _axes(ax)
    ids = np.asarray(paths[hole], dtype=object)
    xyz = np.column_stack([np.asarray(paths[c], dtype=float) for c in ("x", "y", "z")])
    depth = np.asarray(paths["depth"], dtype=float)
    if plane is not None:
        _, u, v, _ = _frame(plane)
        uv = np.c_[u, v]
        xy = xyz @ uv
    else:
        xy = xyz[:, :2]
    kwargs.setdefault("color", "0.3")
    kwargs.setdefault("lw", 1.0)
    for name in dict.fromkeys(ids):
        m = ids == name
        order = np.argsort(depth[m])
        ax.plot(xy[m][order, 0], xy[m][order, 1], **kwargs)
        if labels:
            ax.annotate(str(name), xy[m][order[0]], xytext=(3, 3), textcoords="offset points", fontsize=7)
    if mark is not None:
        mark = np.asarray(mark, dtype=float)
        points = mark @ uv if plane is not None else mark[:, :2]
        ax.scatter(points[:, 0], points[:, 1], color=_accent(), marker="x", zorder=5)
    if plane is not None:
        _label(ax, u, v)
    else:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
    return fig, ax


def directions(anisotropy, *, plane=None, thickness=None, ax=None, **kwargs):
    """Major axis of each local anisotropy as a line through its location, on a plan or a section.

    Each line is the major axis projected on the view, so an axis plunging out of it draws short.

    Parameters
    ----------
    anisotropy : LocalAnisotropy
        Locations and angles (azimuth, dip, rake).
    plane : tuple, optional
        ``(center, azimuth, dip)`` of a section as in `slab`; default a plan.
    thickness : float, optional
        With `plane`, full width of the slab of locations drawn; default all.
    **kwargs
        Passed to ``ax.quiver`` (e.g. ``scale``, ``width``, ``color``).
    """
    fig, ax = _axes(ax)
    coords = np.asarray(anisotropy.coords, dtype=float)
    az, dip = np.radians(np.asarray(anisotropy.angles, dtype=float)[:, :2].T)
    major = np.c_[np.sin(az) * np.cos(dip), np.cos(az) * np.cos(dip), -np.sin(dip)]
    if plane is None:
        center, u, v, n = np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]
    else:
        center, u, v, n = _frame(plane)
    keep = np.abs((coords - center) @ n) <= (np.inf if thickness is None else thickness / 2)
    uv = np.c_[u, v]
    xy, d = coords[keep] @ uv, major[keep] @ uv
    for k, x in {
        "pivot": "middle",
        "headwidth": 0,
        "headlength": 0,
        "headaxislength": 0,
        "angles": "xy",
    }.items():
        kwargs.setdefault(k, x)
    ax.quiver(xy[:, 0], xy[:, 1], d[:, 0], d[:, 1], **kwargs)
    _label(ax, u, v)
    return fig, ax


def paired_bias(bias, *, ax=None, **kwargs):
    """Relative bias of `b` over `a` per bin of pairing distance, with the pair counts as light bars.

    Parameters
    ----------
    bias : Table
        Result of ``boitata.paired_bias``.
    **kwargs
        Passed to ``ax.plot``.
    """
    fig, ax = _axes(ax)
    lo, hi = np.asarray(bias["from"]), np.asarray(bias["to"])
    bars = ax.twinx()
    bars.bar(lo, bias["n"], width=hi - lo, align="edge", color="0.9", edgecolor="white", zorder=0)
    bars.set_ylabel("Pairs", color="0.5")
    bars.tick_params(axis="y", colors="0.5")
    ax.set_zorder(bars.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    kwargs.setdefault("marker", "o")
    ax.plot((lo + hi) / 2, 100 * np.asarray(bias["bias"]), **kwargs)
    ax.set_xlim(lo[0], hi[-1])
    ax.set_xlabel("Pairing distance")
    ax.set_ylabel("Bias of b over a (%)")
    return fig, ax


def uncertainty_curve(curve, *, threshold=0.15, required=None, ax=None):
    """Uncertainty quantiles against data spacing, the share within `threshold` on the right axis.

    Parameters
    ----------
    curve : Table
        Result of ``boitata.uncertainty_curve``; its ``P...`` columns are drawn, the first solid, the rest dashed.
    threshold : float
        Acceptable uncertainty, drawn as a horizontal line.
    required : float, optional
        Required spacing, e.g. from ``boitata.required_spacing``, marked on the threshold line.
    """
    fig, ax = _axes(ax)
    spacing = np.asarray(curve["spacing"])
    share = ax.twinx()
    share.plot(spacing, curve["share"], color="0.6", lw=1)
    share.set_ylim(0, 1)
    share.set_ylabel(f"Share at or below {threshold:g}", color="0.5")
    share.tick_params(axis="y", colors="0.5")
    ax.set_zorder(share.get_zorder() + 1)
    ax.patch.set_visible(False)
    color = _accent()
    names = [n for n in curve.column_names if n.startswith("P")]
    for k, name in enumerate(names):
        ax.plot(
            spacing,
            curve[name],
            color=color,
            ls="-" if k == 0 else "--",
            marker="o",
            label=name,
        )
    ax.axhline(threshold, color="0.5", lw=0.8, ls="--")
    if required is not None and np.isfinite(required):
        ax.plot([required], [threshold], "v", color="0.2", label=f"required {required:.3g}")
    ax.set_xlabel("Data spacing")
    ax.set_ylabel("Uncertainty")
    ax.legend()
    return fig, ax


def _curves(table, name):
    """Rows of each curve in a grade-tonnage Table, split by its ``model`` and ``category`` columns."""
    keys = [np.asarray(table[k]).astype(str) for k in ("model", "category") if k in table.column_names]
    keys = [k for k in keys if len(np.unique(k)) > 1]
    labels = (
        np.array([" ".join(parts) for parts in zip(*keys, strict=True)]) if keys else np.full(len(table), "")
    )
    out = []
    for label in dict.fromkeys(labels):
        text = " ".join(filter(None, (name, label)))
        out.append((text or None, np.flatnonzero(labels == label)))
    return out


def grade_tonnage(table, *, relative=False, ax=None, **kwargs):
    """Tonnage (solid) and mean grade (dashed, right axis) above cutoff, for one curve or several in one color each.

    Parameters
    ----------
    table : Table or dict of str to Table
        Result of ``boitata.grade_tonnage``, ``HermiteAnamorphosis.grade_tonnage``,
        ``UniformConditioning.grade_tonnage``, ``boitata.compare_models`` or ``SimulationSummary.grade_tonnage``;
        rows are split into curves by their ``model`` and ``category`` columns. A dict names several such tables.
        With a ``probability`` column, each curve is drawn at the probability nearest 0.5, in a band between the
        lowest and highest.
    relative : bool
        Tonnage as a fraction of each curve's tonnage at its lowest cutoff, to compare samples with blocks.
    **kwargs
        Passed to every ``ax.plot``.
    """
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    fig, ax = _axes(ax)
    tables = table.items() if isinstance(table, dict) else [(None, table)]
    curves = [(label, t, rows) for name, t in tables for label, rows in _curves(t, name)]
    grade = ax.twinx()
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = [
        Line2D([], [], color="0.3", label="Tonnage"),
        Line2D([], [], color="0.3", ls="--", label="Mean grade"),
    ]
    band = None
    for i, (label, t, rows) in enumerate(curves):
        color = colors[i % len(colors)]
        probability = (
            np.asarray(t["probability"], dtype=float)[rows] if "probability" in t.column_names else None
        )
        levels = [None] if probability is None else np.unique(probability)
        middle = None if probability is None else levels[np.argmin(np.abs(levels - 0.5))]

        def curve(level, t=t, rows=rows, probability=probability):
            r = rows if level is None else rows[probability == level]
            cutoff = np.asarray(t["cutoff"], dtype=float)[r]
            order = np.argsort(cutoff, kind="stable")
            tonnage = np.asarray(t["tonnage"], dtype=float)[r][order]
            return cutoff[order], tonnage, np.asarray(t["mean_grade"], dtype=float)[r][order]

        cutoff, tonnage, mean_grade = curve(middle)
        scale = tonnage[0] if relative else 1.0
        ax.plot(cutoff, tonnage / scale, color=color, **kwargs)
        grade.plot(cutoff, mean_grade, color=color, ls="--", **kwargs)
        if probability is not None and len(levels) > 1:
            (_, t_lo, g_lo), (_, t_hi, g_hi) = curve(levels[0]), curve(levels[-1])
            ax.fill_between(cutoff, t_lo / scale, t_hi / scale, color=color, alpha=0.2, lw=0)
            grade.fill_between(cutoff, g_lo, g_hi, color=color, alpha=0.1, lw=0)
            band = (levels[0], levels[-1])
        if label is not None:
            handles.append(Line2D([], [], color=color, lw=6, label=label))
    if band is not None:
        handles.append(
            Line2D([], [], color="0.3", lw=6, alpha=0.25, label=f"P{100 * band[0]:g} to P{100 * band[1]:g}")
        )
    ax.set_xlabel("Cutoff")
    ax.set_ylabel("Tonnage fraction" if relative else "Tonnage")
    ax.set_ylim(bottom=0)
    grade.set_ylabel("Mean grade above cutoff")
    grade.legend(handles=handles, loc="upper center", frameon=False)
    return fig, ax


def _accuracy_curve(cv):
    from boitata.estimation import IndicatorCrossValidation

    if isinstance(cv, IndicatorCrossValidation):
        return cv
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (cv.actual - cv.estimate) / np.sqrt(cv.variance)
    pit = np.full(z.size, np.nan)
    ok = np.isfinite(z)
    pit[ok] = normal_cdf(z[ok])
    return IndicatorCrossValidation(cv.actual, cv.estimate, cv.variance, [], np.empty((z.size, 0)), pit)


def cross_validation(cv, *, kind="scatter", coords=None, ax=None, **kwargs):
    """Cross-validation as actual against estimate, as an accuracy plot, or as its errors.

    Parameters
    ----------
    cv : CrossValidation or IndicatorCrossValidation
        Result of ``cross_validate``.
    kind : {"scatter", "accuracy", "errors"}
        ``"scatter"``: actual against estimate with the 1:1 line, the regression of actual on estimate (a slope
        below 1 is conditional bias) and the statistics. ``"accuracy"``: fraction of samples inside their
        symmetric p-probability interval against p, with the goodness statistic; on or above the diagonal is
        accurate. Kriging takes the Gaussian interval of its estimate and variance, indicator kriging its
        corrected distribution. ``"errors"``: estimate minus actual mapped at `coords`, or without them a
        histogram.
    coords : PointSet or array_like, optional
        Sample locations for the error map, in the order given to ``fit``.
    **kwargs
        Passed to ``ax.scatter`` (scatter, error map), ``ax.plot`` (accuracy) or ``ax.hist`` (histogram).
    """
    if kind not in ("scatter", "accuracy", "errors"):
        raise InvalidInput(f"kind must be 'scatter', 'accuracy' or 'errors', not {kind!r}")
    fig, ax = _axes(ax)
    ok = np.isfinite(cv.estimate) & np.isfinite(cv.actual)
    if kind == "scatter":
        estimate, actual = cv.estimate[ok], cv.actual[ok]
        kwargs.setdefault("s", 6)
        kwargs.setdefault("alpha", 0.5)
        ax.scatter(estimate, actual, **kwargs)
        lo, hi = min(estimate.min(), actual.min()), max(estimate.max(), actual.max())
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--", label="1:1")
        slope, intercept = np.polyfit(estimate, actual, 1)
        ax.plot(
            [lo, hi],
            [intercept + slope * lo, intercept + slope * hi],
            color="0.2",
            lw=1.2,
            label="regression",
        )
        rows = [
            ("n", f"{ok.sum():,}"),
            ("mean error", f"{cv.mean_error:.3g}"),
            ("RMSE", f"{cv.rmse:.3g}"),
            ("correlation", f"{cv.correlation:.2f}"),
            ("slope", f"{slope:.2f}"),
        ]
        if np.isfinite(cv.variance[ok]).any():
            rows.append(("error²/variance", f"{cv.standardized_squared_error:.2f}"))
        width = max(len(k) for k, _ in rows)
        text = "\n".join(f"{k.ljust(width)} {v}" for k, v in rows)
        box = {"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2}
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", family="monospace", fontsize=7, bbox=box)
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlabel("Estimate")
        ax.set_ylabel("Actual")
        ax.set_aspect("equal", adjustable="datalim")
    elif kind == "accuracy":
        curve = _accuracy_curve(cv)
        p = np.linspace(0, 1, 21)
        ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls="--")
        kwargs.setdefault("marker", ".")
        ax.plot(p, curve.accuracy(p), **kwargs)
        ax.text(0.97, 0.03, f"goodness {curve.goodness:.2f}", transform=ax.transAxes, ha="right", va="bottom")
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.set_xlabel("Probability interval p")
        ax.set_ylabel("Fraction of samples inside")
    elif coords is None:
        error = cv.error[ok]
        kwargs.setdefault("bins", 40)
        kwargs.setdefault("edgecolor", "white")
        kwargs.setdefault("linewidth", 0.5)
        ax.hist(error, weights=np.full(error.size, 1 / error.size), **kwargs)
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        ax.axvline(error.mean(), color="0.2", lw=1.2, label=f"mean {error.mean():.3g}")
        ax.legend(frameon=False)
        ax.set_xlabel("Error (estimate − actual)")
        ax.set_ylabel("Frequency")
    else:
        xy = np.asarray(getattr(coords, "coords", coords), dtype=float)
        if xy.ndim != 2 or len(xy) != len(cv.actual):
            raise InvalidInput(f"coords must be one location per sample, {len(cv.actual)} rows")
        error = cv.error[ok]
        top = np.abs(error).max()
        kwargs.setdefault("cmap", "RdBu_r")
        kwargs.setdefault("s", 12)
        points = ax.scatter(xy[ok, 0], xy[ok, 1], c=error, vmin=-top, vmax=top, **kwargs)
        fig.colorbar(points, ax=ax, shrink=0.8, label="Error (estimate − actual)")
        ax.set(xlabel="X", ylabel="Y", aspect="equal")
    return fig, ax


def contact(table, *, labels=("inside", "outside"), ax=None, **kwargs):
    """Mean grade against signed distance to a contact, the sample counts as light bars; a jump at zero is a hard
    contact, a gradual change a soft one.

    Parameters
    ----------
    table : Table
        Result of ``boitata.contact``; negative distances are inside.
    labels : tuple of str
        Names of the inside and outside domains, written on either side.
    **kwargs
        Passed to ``ax.plot``.
    """
    fig, ax = _axes(ax)
    distance, mean = np.asarray(table["distance"], dtype=float), np.asarray(table["mean"], dtype=float)
    n = np.asarray(table["n"], dtype=float)
    width = np.diff(distance).min() if distance.size > 1 else 1.0
    bars = ax.twinx()
    bars.bar(distance, n, width=width, color="0.9", edgecolor="white", zorder=0)
    bars.set_ylabel("Samples", color="0.5")
    bars.tick_params(axis="y", colors="0.5")
    ax.set_zorder(bars.get_zorder() + 1)
    ax.patch.set_visible(False)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("color", _accent())
    for side in (distance < 0, distance > 0):
        ax.plot(distance[side], mean[side], **kwargs)
    ax.axvline(0, color="0.3", lw=1)
    for x, text in ((distance.min() / 2, labels[0]), (distance.max() / 2, labels[1])):
        ax.text(x, 0.97, text, transform=ax.get_xaxis_transform(), ha="center", va="top", color="0.3")
    ax.set_xlabel("Distance to contact")
    ax.set_ylabel("Mean grade")
    return fig, ax


def domain_change(table, *, value="tonnage", relative=False, fmt=None, ax=None, **kwargs):
    """Cross-table of two categorical models as a matrix, rows the first model's classes, columns the second's,
    each cell labeled with its value. The diagonal, what is unchanged, is outlined and left blank so the grays scale
    to what moves.

    Parameters
    ----------
    table : Table
        Result of ``boitata.domain_change``.
    value : str
        Column to show: ``"tonnage"``, ``"metal"`` or ``"mean_grade"``.
    relative : bool
        Each row as a fraction of its total: the share of each class of the first model going to each class of
        the second (not with ``"mean_grade"``).
    fmt : str, optional
        Format of the cell labels; default ``"{:.0%}"`` when relative, else ``"{:,.3g}"``.
    **kwargs
        Passed to ``ax.imshow``.
    """
    from matplotlib.patches import Rectangle

    fig, ax = _axes(ax)
    names = list(dict.fromkeys(np.asarray(table["from"]).astype(str)))
    k = len(names)
    cells = np.asarray(table[value], dtype=float).reshape(k, k)
    if relative:
        with np.errstate(invalid="ignore", divide="ignore"):
            cells = cells / np.nansum(cells, axis=1, keepdims=True)
    fmt = fmt or ("{:.0%}" if relative else "{:,.3g}")
    kwargs = {"cmap": "Greys", "vmin": 0.0} | kwargs
    moved = np.where(np.eye(k, dtype=bool), np.nan, cells)
    im = ax.imshow(np.ma.masked_invalid(moved), **kwargs)
    for i in range(k):
        for j in range(k):
            if np.isfinite(cells[i, j]) and cells[i, j] != 0:
                color = "white" if i != j and im.norm(cells[i, j]) > 0.5 else "0.2"
                ax.text(j, i, fmt.format(cells[i, j]), ha="center", va="center", color=color)
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=_accent(), lw=1.5))
    ax.set_xticks(range(k), names)
    ax.set_yticks(range(k), names)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    return fig, ax


def transition_mds(table, *, ax=None, **kwargs):
    """Classical (Torgerson) MDS of an along-hole transition table: categories that transition into each other most
    often sit close together in 2D.

    The ``frequency`` column, one class's row shares, is symmetrized into a dissimilarity, ``1 - (freq + freq.T)
    / 2``, then scaled from the top two eigenvectors of its double-centered squared distances.

    Parameters
    ----------
    table : Table
        Result of ``boitata.transition_matrix``.
    **kwargs
        Passed to ``ax.scatter``.
    """
    fig, ax = _axes(ax)
    names = list(dict.fromkeys(np.asarray(table["from"]).astype(str)))
    k = len(names)
    freq = np.nan_to_num(np.asarray(table["frequency"], dtype=float).reshape(k, k))
    d = 1.0 - (freq + freq.T) / 2.0
    np.fill_diagonal(d, 0.0)
    j = np.eye(k) - np.ones((k, k)) / k
    b = -0.5 * j @ (d**2) @ j
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1][:2]
    coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0.0, None))
    if coords.shape[1] < 2:
        coords = np.column_stack([coords, np.zeros(k)])
    kwargs.setdefault("color", _accent())
    ax.scatter(coords[:, 0], coords[:, 1], **kwargs)
    for (x, y), name in zip(coords, names, strict=True):
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    return fig, ax


def _fade(values, uncertainty, cmap, norm):
    rgba = cmap(norm(np.ma.masked_invalid(values)))
    certainty = 1 - np.clip(np.nan_to_num(uncertainty, nan=1.0), 0, 1)
    rgba[..., :3] = 1 - (1 - rgba[..., :3]) * certainty[..., None]
    rgba[..., 3] = np.isfinite(values)
    return rgba


def uncertain(
    values,
    uncertainty,
    *,
    model=None,
    axis="z",
    index=None,
    plane=None,
    resolution=None,
    extent=None,
    cmap=None,
    norm=None,
    scheme=None,
    label=None,
    legend_ax=None,
    ax=None,
):
    """Image whose color gives a value and whose fading towards white gives how uncertain it is.

    The legend is a fan: the value runs across its angle, certainty along its radius, from white at the center
    (uncertain) to the full color on the arc (certain).

    Parameters
    ----------
    values : str or array_like
        2D image, e.g. the mean of the realizations on a section; NaN is left blank. With `model`, a column
        name or one value per block.
    uncertainty : str or array_like
        Same shape as `values`, from 0 (certain) to 1 (no information), e.g. the realizations' standard deviation
        over the global one. Clipped to [0, 1].
    model : BlockModel, optional
        Slice it as `section` does; missing blocks are left blank.
    axis : {"x", "y", "z"}
        With `model`, axis normal to the slice.
    index : int, optional
        With `model`, cell index along `axis` (default: the middle).
    plane : tuple, optional
        With `model`, ``(center, azimuth, dip)`` as in `slab`, replacing `axis` and `index`.
    resolution : float, optional
        Raster step on `plane`; default half the smallest block edge.
    extent : tuple of float, optional
        Passed to ``ax.imshow``; set from `model` when given.
    cmap : str or Colormap, optional
        Default: matplotlib's ``image.cmap``.
    norm : Normalize, optional
        Maps values to [0, 1]; default spans their range.
    scheme : Categories, optional
        `values` are codes of these categories (e.g. the most likely one), drawn in their colors and keyed by a
        legend instead of the fan; replaces `cmap` and `norm`.
    label : str, optional
        Value name written under the legend, or its title with `scheme`.
    legend_ax : Axes, optional
        Where to draw the legend; default below `ax`.
    """
    import matplotlib as mpl

    fig, ax = _axes(ax)
    if model is not None:
        (values, uncertainty), extent = _image(
            ax, model, [values, uncertainty], axis, index, plane, resolution
        )
    values = np.asarray(values, dtype=float)
    cmap = mpl.colormaps[cmap or mpl.rcParams["image.cmap"]] if not callable(cmap) else cmap
    if scheme is not None:
        cmap, norm = category_colors(scheme)
    norm = norm or mpl.colors.Normalize(np.nanmin(values), np.nanmax(values))
    ax.imshow(_fade(values, np.asarray(uncertainty, dtype=float), cmap, norm), origin="lower", extent=extent)
    if scheme is not None:
        where = {"loc": "center"} if legend_ax else {"loc": "upper left", "bbox_to_anchor": (1.01, 1)}
        category_legend(scheme, legend_ax or ax, title=label, **where)
        if legend_ax:
            legend_ax.axis("off")
        return fig, ax

    fan = legend_ax or ax.inset_axes([0.2, -0.6, 0.6, 0.4])
    x, y = np.meshgrid(np.linspace(-1, 1, 201), np.linspace(0, 1, 101))
    radius, angle = np.hypot(x, y), np.degrees(np.arctan2(y, x))
    share = (135 - angle) / 90
    inside = (radius <= 1) & (share >= 0) & (share <= 1)
    fraction = np.where(inside, share, np.nan)
    fan.imshow(_fade(norm.inverse(fraction), 1 - radius, cmap, norm), origin="lower", extent=(-1, 1, 0, 1))
    for share, align in ((0, "right"), (0.5, "center"), (1, "left")):
        theta = np.radians(135 - 90 * share)
        value = float(norm.inverse(share))
        fan.text(
            1.05 * np.cos(theta), 1.05 * np.sin(theta), f"{value:.3g}", ha=align, va="bottom", fontsize=8
        )
    edge = np.array([[-0.1, -0.1], [-0.8, 0.6]])
    fan.plot(*edge.T, color="0.4", lw=0.8, marker="o", ms=2)
    for (x, y), text in zip(edge, ("uncertain", "certain"), strict=True):
        fan.text(x - 0.06, y, text, ha="right", va="center", fontsize=7, color="0.4")
    if label:
        fan.text(0, -0.2, label, ha="center", va="top", fontsize=9)
    fan.set(xlim=(-1.5, 1.5), ylim=(-0.4, 1.2), aspect="equal")
    fan.axis("off")
    return fig, ax
