"""Plots on matplotlib (``pip install ceres[plot]``).

Every function draws on `ax` when given, else on a new figure, and returns ``(fig, ax)``.
"""

import numpy as np

from ceres._ceres import describe, normal_ppf

__all__ = [
    "boxplot",
    "cdf",
    "histogram",
    "probability",
    "qq",
    "scatter",
    "section",
    "swath",
    "uncertain",
    "variogram",
]


def _axes(ax):
    if ax is not None:
        return ax.figure, ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("ceres.plot needs matplotlib: pip install ceres[plot]") from e
    return plt.subplots()


def _finite(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(values)
    return values[ok], weights[ok] / weights[ok].sum()


def histogram(values, weights=None, bins=40, log=False, ax=None, **kwargs):
    """Histogram of relative frequencies, declustered by `weights`; `log` bins on a log axis.

    Parameters
    ----------
    values : array_like
        Values; NaN is ignored.
    weights : array_like, optional
        Declustering weights.
    bins : int or array_like
        Number of bins or their edges.
    log : bool
        Log-spaced bins over the positive values and a log x axis.
    **kwargs
        Passed to ``ax.hist``.
    """
    fig, ax = _axes(ax)
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


def probability(values, weights=None, log=False, ax=None, **kwargs):
    """Cumulative probability on a normal scale: a Gaussian (or, with `log`, lognormal) distribution is a line.

    Parameters
    ----------
    values : array_like
        Values; NaN is ignored.
    weights : array_like, optional
        Declustering weights.
    log : bool
        Log x axis.
    **kwargs
        Passed to ``ax.plot``.
    """
    fig, ax = _axes(ax)
    v, w = _finite(values, weights)
    order = np.argsort(v)
    v, w = v[order], w[order]
    p = np.cumsum(w) - w / 2
    kwargs.setdefault("marker", ".")
    kwargs.setdefault("linestyle", "none")
    ax.plot(v, normal_ppf(p), **kwargs)
    ticks = np.array([0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    ticks = ticks[(ticks >= p[0]) & (ticks <= p[-1])]
    ax.set_yticks(normal_ppf(ticks), [f"{100 * t:g}" for t in ticks])
    ax.set_ylabel("Cumulative probability (%)")
    if log:
        ax.set_xscale("log")
    return fig, ax


def cdf(values, weights=None, labels=None, log=False, ax=None, **kwargs):
    """Cumulative distribution of one or several series overlaid, e.g. domains, or clustered and declustered.

    Parameters
    ----------
    values : array_like or list of array_like
        One series, or several; NaN is ignored.
    weights : array_like or list, optional
        Declustering weights: an array for one series; for several, one array or None each.
    labels : list of str, optional
        Legend entries.
    log : bool
        Log x axis.
    **kwargs
        Passed to every ``ax.step``.
    """
    fig, ax = _axes(ax)
    if np.ndim(values[0]) == 0:
        values, weights = [values], [weights]
    weights = [None] * len(values) if weights is None else weights
    labels = labels or [None] * len(values)
    for v, w, label in zip(values, weights, labels, strict=True):
        v, w = _finite(v, w)
        order = np.argsort(v)
        ax.step(v[order], np.cumsum(w[order]), where="post", label=label, **kwargs)
    if any(labels):
        ax.legend()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cumulative probability")
    if log:
        ax.set_xscale("log")
    return fig, ax


def qq(x, y, x_weights=None, y_weights=None, quantiles=None, log=False, ax=None, **kwargs):
    """Quantiles of `y` against the same quantiles of `x`, with the 1:1 line on which equal distributions lie.

    Parameters
    ----------
    x, y : array_like
        Two samples, e.g. one domain and another, or composites and blocks; NaN is ignored.
    x_weights, y_weights : array_like, optional
        Declustering weights of each.
    quantiles : array_like, optional
        Probabilities; default 0.01 to 0.99 in steps of 0.01.
    log : bool
        Log axes.
    **kwargs
        Passed to ``ax.plot``.
    """
    fig, ax = _axes(ax)
    p = np.linspace(0.01, 0.99, 99) if quantiles is None else quantiles
    qx = describe(x, x_weights, quantiles=p)["quantiles"]
    qy = describe(y, y_weights, quantiles=p)["quantiles"]
    kwargs.setdefault("marker", ".")
    kwargs.setdefault("linestyle", "none")
    ax.plot(qx, qy, **kwargs)
    lo, hi = min(qx.min(), qy.min()), max(qx.max(), qy.max())
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--", label="1:1")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    return fig, ax


def boxplot(values, categories, weights=None, sort=False, log=False, ax=None, **kwargs):
    """One box per category: P25 to P75, median, P10 to P90 whiskers and the mean, all weighted.

    Parameters
    ----------
    values : array_like
        Values; NaN is ignored.
    categories : array_like
        Category (e.g. domain) of each value; each box is labelled with its count.
    weights : array_like, optional
        Declustering weights.
    sort : bool
        Order boxes by median, else by category.
    log : bool
        Log value axis.
    **kwargs
        Passed to ``ax.bxp``.
    """
    import matplotlib as mpl

    fig, ax = _axes(ax)
    values, categories = np.asarray(values, dtype=float), np.asarray(categories)
    weights = None if weights is None else np.asarray(weights, dtype=float)
    stats = []
    for c in np.unique(categories[~np.isnan(values)]):
        keep = categories == c
        s = describe(values[keep], None if weights is None else weights[keep])
        p10, q1, med, q3, p90 = s["quantiles"]
        stats.append(
            {
                "label": f"{c}\nn = {s['n']:,}",
                "whislo": p10,
                "q1": q1,
                "med": med,
                "q3": q3,
                "whishi": p90,
                "mean": s["mean"],
                "fliers": [],
            }
        )
    if sort:
        stats.sort(key=lambda s: s["med"])
    color = mpl.rcParams["axes.prop_cycle"].by_key()["color"][0]
    kwargs.setdefault("patch_artist", True)
    kwargs.setdefault("showmeans", True)
    kwargs.setdefault("boxprops", {"facecolor": mpl.colors.to_rgba(color, 0.25), "edgecolor": color})
    kwargs.setdefault("medianprops", {"color": color, "lw": 2})
    kwargs.setdefault("whiskerprops", {"color": color})
    kwargs.setdefault("capprops", {"color": color})
    kwargs.setdefault("meanprops", {"marker": "o", "ms": 4, "mfc": "white", "mec": color})
    ax.bxp(stats, **kwargs)
    if log:
        ax.set_yscale("log")
    return fig, ax


def variogram(experimental, model=None, direction=None, ax=None, **kwargs):
    """Experimental variogram, points sized by pair count, with `model` and its sill.

    Parameters
    ----------
    experimental : ExperimentalVariogram
    model : Variogram, optional
    direction : tuple of float, optional
        ``(azimuth, dip)`` in degrees along which an anisotropic `model` is drawn;
        without it, lags are taken as distances in the model's isotropic space.
    **kwargs
        Passed to ``ax.scatter`` and, for its colour, to the model line.
    """
    fig, ax = _axes(ax)
    keep = np.asarray(experimental.counts) > 0
    lags, gammas = experimental.lags[keep], experimental.gammas[keep]
    kwargs.setdefault("s", np.sqrt(experimental.counts[keep]) * 2)
    points = ax.scatter(lags, gammas, **kwargs)
    if model is not None:
        h = np.linspace(0, lags.max() * 1.05, 201)[1:]
        if direction is None:
            gamma = model.gamma(h)
        else:
            az, dip = np.radians(direction[0]), np.radians(direction[1])
            unit = np.array([np.cos(dip) * np.sin(az), np.cos(dip) * np.cos(az), -np.sin(dip)])
            gamma = model.gamma_between(np.zeros((h.size, 3)), h[:, None] * unit)
        ax.plot(h, gamma, color=points.get_facecolor()[0])
        ax.axhline(model.sill, color="0.5", lw=0.8, ls="--")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Lag distance")
    ax.set_ylabel("γ(h)")
    return fig, ax


def scatter(x, y, line=True, ax=None, **kwargs):
    """`y` against `x` with the 1:1 line and, if `line`, the least-squares regression of `y` on `x`.

    The regression slope of true on estimated values below 1 flags conditional bias.
    """
    fig, ax = _axes(ax)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    kwargs.setdefault("s", 6)
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("linewidths", 0)
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


def section(block_model, values, axis="z", index=None, colorbar=True, ax=None, **kwargs):
    """Slice of a block model across `axis` at cell `index` (default: the middle), in model coordinates.

    Parameters
    ----------
    block_model : BlockModel
        Regular or masked; missing blocks are left blank. Rotation is ignored.
    values : str or array_like
        Column name, or one value per block.
    axis : {"x", "y", "z"}
        Axis normal to the slice.
    index : int, optional
        Cell index along `axis`.
    colorbar : bool
        Add a colour bar labelled with the column name.
    **kwargs
        Passed to ``ax.imshow`` (e.g. ``cmap``, ``norm``, ``vmin``).
    """
    fig, ax = _axes(ax)
    (image,), extent = _slice(ax, block_model, [values], axis, index)
    im = ax.imshow(image, origin="lower", extent=extent, **kwargs)
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8, label=values if isinstance(values, str) else None)
    return fig, ax


def _slice(ax, block_model, columns, axis, index):
    names = []
    for i, column in enumerate(columns):
        if not isinstance(column, str):
            block_model = block_model.with_column(f"_{i}", column)
            column = f"_{i}"
        names.append(column)
    grid = block_model.to_regular()
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


def swath(swaths, labels=None, ax=None, **kwargs):
    """Mean per slice of one or several `swath` results, with the first one's counts as light bars.

    Parameters
    ----------
    swaths : dict or list of dict
        Results of ``ceres.swath``, e.g. composites and blocks with the same width.
    labels : list of str, optional
        Legend entries.
    **kwargs
        Passed to every ``ax.plot``.
    """
    fig, ax = _axes(ax)
    swaths = [swaths] if isinstance(swaths, dict) else list(swaths)
    labels = labels or [None] * len(swaths)
    first = swaths[0]
    bars = ax.twinx()
    width = np.diff(first["centres"]).min() if len(first["centres"]) > 1 else 1.0
    bars.bar(first["centres"], first["count"], width=width, color="0.9", zorder=0)
    bars.set_ylabel("Count", color="0.5")
    bars.tick_params(axis="y", colors="0.5")
    ax.set_zorder(bars.get_zorder() + 1)
    ax.patch.set_visible(False)
    kwargs.setdefault("marker", ".")
    for s, label in zip(swaths, labels, strict=True):
        ax.plot(s["centres"], s["mean"], label=label, **kwargs)
    if any(labels):
        ax.legend()
    ax.set_xlabel("Distance along swath")
    ax.set_ylabel("Mean")
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
    block_model=None,
    axis="z",
    index=None,
    extent=None,
    cmap="viridis",
    norm=None,
    label=None,
    legend_ax=None,
    ax=None,
):
    """Image whose colour gives a value and whose fading towards white gives how uncertain it is.

    The legend is a fan: the value runs across its angle, certainty along its radius, from white at the centre
    (uncertain) to the full colour on the arc (certain).

    Parameters
    ----------
    values : str or array_like
        2D image, e.g. the mean of the realizations on a section; NaN is left blank. With `block_model`, a column
        name or one value per block.
    uncertainty : str or array_like
        Same shape as `values`, from 0 (certain) to 1 (no information), e.g. the realizations' standard deviation
        over the global one. Clipped to [0, 1].
    block_model : BlockModel, optional
        Slice it as `section` does; missing blocks are left blank.
    axis : {"x", "y", "z"}
        With `block_model`, axis normal to the slice.
    index : int, optional
        With `block_model`, cell index along `axis` (default: the middle).
    extent : tuple of float, optional
        Passed to ``ax.imshow``; set from `block_model` when given.
    cmap : str or Colormap
    norm : Normalize, optional
        Maps values to [0, 1]; default spans their range.
    label : str, optional
        Value name written under the legend.
    legend_ax : Axes, optional
        Where to draw the legend; default below `ax`.
    """
    import matplotlib as mpl

    fig, ax = _axes(ax)
    if block_model is not None:
        (values, uncertainty), extent = _slice(ax, block_model, [values, uncertainty], axis, index)
    values = np.asarray(values, dtype=float)
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    norm = norm or mpl.colors.Normalize(np.nanmin(values), np.nanmax(values))
    ax.imshow(_fade(values, np.asarray(uncertainty, dtype=float), cmap, norm), origin="lower", extent=extent)

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
