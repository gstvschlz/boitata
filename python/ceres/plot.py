"""Plots on matplotlib (``pip install ceres[plot]``).

Every function draws on `ax` when given, else on a new figure, and returns ``(fig, ax)``; `scatter_matrix` takes
and returns a grid of `axes` instead.
"""

import numpy as np

from ceres._ceres import correlation, describe, normal_ppf

__all__ = [
    "boxplot",
    "cdf",
    "histogram",
    "probability",
    "qq",
    "scatter",
    "scatter_matrix",
    "section",
    "slab",
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


def probability(values, weights=None, log=False, cap=None, ax=None, **kwargs):
    """Cumulative probability on a normal scale: a Gaussian (or, with `log`, lognormal) distribution is a line.

    Parameters
    ----------
    values : array_like
        Values; NaN is ignored.
    weights : array_like, optional
        Declustering weights.
    log : bool
        Log x axis.
    cap : float, optional
        Top cut, drawn as a dashed vertical line.
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
    if cap is not None:
        ax.axvline(cap, color="0.5", lw=0.8, ls="--", label=f"cap {cap:.3g}")
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


def scatter_matrix(data, labels=None, weights=None, log=False, bins=30, axes=None, **kwargs):
    """Pairwise scatters of the columns of `data`, with their histograms on the diagonal.

    Each scatter is annotated with the Pearson (``r``) and rank correlation of its pair, weighted by `weights` and
    over the rows where both values are present, as ``ceres.correlation`` computes them.

    Parameters
    ----------
    data : array_like, mapping or Table
        ``(n, d)`` values, or ``d`` named columns; NaN is ignored pair by pair.
    labels : list of str, optional
        Column names; default the mapping keys or the column indices.
    weights : array_like, optional
        Declustering weights, used by the histograms and the correlations.
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
    if hasattr(data, "column_names") or hasattr(data, "keys"):
        names = list(data.column_names) if hasattr(data, "column_names") else list(data.keys())
        labels = names if labels is None else labels
        data = np.column_stack([np.asarray(data[k], dtype=float) for k in names])
    data = np.array(data, dtype=float)
    d = data.shape[1]
    labels = [str(j) for j in range(d)] if labels is None else list(labels)
    log = np.broadcast_to(log, d)
    for j in np.flatnonzero(log):
        data[data[:, j] <= 0, j] = np.nan
    pearson = correlation(data, weights)
    rank = correlation(data, weights, method="spearman")
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
                histogram(data[:, j], weights, bins=edges, log=log[j], ax=bars)
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
    block_model, values, axis="z", index=None, plane=None, resolution=None, colorbar=True, ax=None, **kwargs
):
    """Slice of a block model across `axis` at cell `index` (default: the middle), or on any `plane`.

    Parameters
    ----------
    block_model : BlockModel
        Missing blocks are left blank. Sliced along `axis`: regular or masked, in model coordinates, rotation
        ignored. On a `plane`: any layout and rotation.
    values : str or array_like
        Column name, or one value per block.
    axis : {"x", "y", "z"}
        Axis normal to the slice.
    index : int, optional
        Cell index along `axis`.
    plane : tuple, optional
        ``(centre, azimuth, dip)`` as in `slab`; replaces `axis` and `index`. The model is sampled on a raster in
        section coordinates.
    resolution : float, optional
        Raster step on `plane`; default half the smallest block edge.
    colorbar : bool
        Add a colour bar labelled with the column name.
    **kwargs
        Passed to ``ax.imshow`` (e.g. ``cmap``, ``norm``, ``vmin``).
    """
    fig, ax = _axes(ax)
    (image,), extent = _image(ax, block_model, [values], axis, index, plane, resolution)
    im = ax.imshow(image, origin="lower", extent=extent, **kwargs)
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8, label=values if isinstance(values, str) else None)
    return fig, ax


def _image(ax, block_model, columns, axis, index, plane, resolution):
    if plane is None:
        return _slice(ax, block_model, columns, axis, index)
    centre, u, v, n = _frame(plane)
    size = np.asarray(block_model.size, dtype=float)
    step = resolution or size.min() / 2
    reach = np.linalg.norm(size) / 2
    centroids = block_model.centroids
    near = centroids[np.abs((centroids - centre) @ n) <= reach] @ np.c_[u, v]
    if not len(near):
        raise ValueError("the plane misses the block model")
    lo = near.min(0) - reach
    counts = np.ceil((near.max(0) + reach - lo) / step).astype(int)
    gu, gv = np.meshgrid(*(lo[a] + (np.arange(counts[a]) + 0.5) * step for a in (0, 1)))
    offset = centre - (centre @ u) * u - (centre @ v) * v
    rows = block_model.row_at(offset + gu.reshape(-1, 1) * u + gv.reshape(-1, 1) * v).reshape(gu.shape)
    if not (rows >= 0).any():
        raise ValueError("the plane misses the block model")
    (j0, j1), (i0, i1) = (np.flatnonzero((rows >= 0).any(axis=a))[[0, -1]] for a in (0, 1))
    rows = rows[i0 : i1 + 1, j0 : j1 + 1]
    images = []
    for column in columns:
        values = np.asarray(block_model[column] if isinstance(column, str) else column, dtype=float)
        images.append(np.where(rows >= 0, values[rows], np.nan))
    _label(ax, u, v)
    (u0, v0), (u1, v1) = lo + step * np.array([j0, i0]), lo + step * np.array([j1 + 1, i1 + 1])
    return images, (u0, u1, v0, v1)


def _frame(plane):
    centre, azimuth, dip = plane
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (3,):
        raise ValueError("plane centre must be (x, y, z)")
    az, dip = np.radians(azimuth), np.radians(dip)
    u = np.array([np.sin(az), np.cos(az), 0.0])
    v = np.cos(dip) * np.array([-np.cos(az), np.sin(az), 0.0]) + np.array([0.0, 0.0, np.sin(dip)])
    return centre, u, v, np.cross(u, v)


def _label(ax, u, v):
    names = ("Easting (m)", "Northing (m)", "Elevation (m)")
    for axis, w, default in ((ax.xaxis, u, "Along strike (m)"), (ax.yaxis, v, "Up dip (m)")):
        aligned = np.isclose(w, 1.0, atol=1e-9)
        axis.set_label_text(names[aligned.argmax()] if aligned.any() else default)
    ax.set_aspect("equal")


def slab(
    points,
    values=None,
    *,
    plane,
    thickness,
    meshes=None,
    lines=None,
    labels=None,
    colorbar=True,
    ax=None,
    **kwargs,
):
    """Points within a slab around a plane, in section coordinates, with mesh traces, clipped lines and labels.

    Section coordinates are world coordinates along the strike and up the dip of the plane, so a vertical
    east-west section reads easting and elevation, and a horizontal plane easting and northing.

    Parameters
    ----------
    points : PointSet or array_like
        ``(n, 3)`` coordinates.
    values : str or array_like, optional
        Column of `points`, or one value per point, colouring them; default one colour.
    plane : tuple
        ``(centre, azimuth, dip)``: a point on the plane, the bearing of the section line and the dip of the plane
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
        With `values`, add a colour bar labelled with the column name.
    **kwargs
        Passed to ``ax.scatter`` (e.g. ``s``, ``cmap``, ``norm``, ``color``).
    """
    from matplotlib.collections import LineCollection

    fig, ax = _axes(ax)
    centre, u, v, n = _frame(plane)
    half = thickness / 2
    uv = np.c_[u, v]
    coords = np.asarray(getattr(points, "coords", points), dtype=float)
    near = np.abs((coords - centre) @ n) <= half
    xy = coords[near] @ uv

    if meshes is not None:
        for mesh in [meshes] if hasattr(meshes, "triangles") else meshes:
            ax.add_collection(LineCollection(_trace(mesh, centre, n) @ uv, colors="0.2", linewidths=1))
    if lines is not None:
        segments = _clip([np.asarray(line, dtype=float) for line in lines], centre, n, half)
        ax.add_collection(LineCollection(segments @ uv, colors="0.75", linewidths=0.8))

    if values is not None:
        kwargs["c"] = np.asarray(points[values] if isinstance(values, str) else values, dtype=float)[near]
    kwargs.setdefault("s", 6)
    drawn = ax.scatter(xy[:, 0], xy[:, 1], **kwargs)
    if values is not None and colorbar:
        fig.colorbar(drawn, ax=ax, shrink=0.8, label=values if isinstance(values, str) else None)

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


def _trace(mesh, centre, normal):
    corners = mesh.vertices[mesh.triangles]
    d = (corners - centre) @ normal
    a, b = [0, 1, 2], [1, 2, 0]
    crossing = (d[:, a] > 0) != (d[:, b] > 0)
    t = d[:, a] / np.where(crossing, d[:, a] - d[:, b], 1.0)
    cuts = corners[:, a] + t[..., None] * (corners[:, b] - corners[:, a])
    return cuts[crossing].reshape(-1, 2, 3)


def _clip(lines, centre, normal, half):
    segments = np.concatenate(
        [np.empty((0, 2, 3))] + [np.stack([line[:-1], line[1:]], axis=1) for line in lines]
    )
    d = (segments - centre) @ normal
    step = d[:, 1] - d[:, 0]
    flat = step == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ta, tb = (-half - d[:, 0]) / step, (half - d[:, 0]) / step
    start = np.where(flat, np.where(np.abs(d[:, 0]) <= half, 0.0, 1.0), np.maximum(0, np.minimum(ta, tb)))
    end = np.where(flat, 1.0, np.minimum(1, np.maximum(ta, tb)))
    keep = start < end
    s, t = segments[keep], np.c_[start, end][keep]
    return s[:, :1] + t[..., None] * (s[:, 1:] - s[:, :1])


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
    plane=None,
    resolution=None,
    extent=None,
    cmap=None,
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
    plane : tuple, optional
        With `block_model`, ``(centre, azimuth, dip)`` as in `slab`, replacing `axis` and `index`.
    resolution : float, optional
        Raster step on `plane`; default half the smallest block edge.
    extent : tuple of float, optional
        Passed to ``ax.imshow``; set from `block_model` when given.
    cmap : str or Colormap, optional
        Default: matplotlib's ``image.cmap``.
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
        (values, uncertainty), extent = _image(
            ax, block_model, [values, uncertainty], axis, index, plane, resolution
        )
    values = np.asarray(values, dtype=float)
    cmap = mpl.colormaps[cmap or mpl.rcParams["image.cmap"]] if not callable(cmap) else cmap
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
