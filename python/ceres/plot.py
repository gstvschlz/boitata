"""Plots on matplotlib (``pip install ceres[plot]``).

Every function draws on `ax` when given, else on a new figure, and returns ``(fig, ax)``.
"""

import numpy as np

from ceres._ceres import normal_ppf

__all__ = ["histogram", "probability", "scatter", "section", "swath", "variogram"]


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
    name = values if isinstance(values, str) else None
    if name is None:
        block_model = block_model.with_column("_values", values)
        name = "_values"
    grid = block_model.to_regular()
    nx, ny, nz = grid.count
    cube = np.asarray(grid[name], dtype=float).reshape(nz, ny, nx)
    (x0, y0, z0), (sx, sy, sz) = grid.origin, grid.size
    x, y, z = (x0, x0 + nx * sx), (y0, y0 + ny * sy), (z0, z0 + nz * sz)
    k = {"x": 2, "y": 1, "z": 0}[axis]
    index = cube.shape[k] // 2 if index is None else index
    image = np.take(cube, index, axis=k)
    extent, labels = {
        "z": ((*x, *y), ("X", "Y")),
        "y": ((*x, *z), ("X", "Z")),
        "x": ((*y, *z), ("Y", "Z")),
    }[axis]
    im = ax.imshow(image, origin="lower", extent=extent, **kwargs)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal")
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8, label=None if name == "_values" else name)
    return fig, ax


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
