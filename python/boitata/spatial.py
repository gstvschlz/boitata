"""Outlines, surfaces and collars built from scattered points."""

from dataclasses import dataclass

import numpy as np

from boitata import _boitata
from boitata._columns import column
from boitata.errors import InvalidInput

__all__ = ["Topography", "outline", "topography"]


def outline(points, *, method="convex", max_edge=None, buffer=0.0, categories=None, plane="xy"):
    """Closed polygons around points, one feature per category.

    Parameters
    ----------
    points : PointSet, BlockModel or array_like
        Points, or their ``(n, 2)`` or ``(n, 3)`` coordinates; a block model's centroids.
    method : {"convex", "concave"}
        ``concave`` erodes the convex hull, longest boundary edge first, while an edge exceeds `max_edge` and the
        polygon stays simple, so every point stays inside or on it.
    max_edge : float, optional
        Longest boundary edge of a concave outline, in coordinate units, e.g. 1.5 to 3 times the drill spacing.
    buffer : float
        Distance to offset the outline outward with round corners; negative shrinks it. A shrink that splits the
        outline gives several rings in the feature, and one that erases it raises.
    categories : array_like or str, optional
        A label per point, or its column; one outline per label, in ascending order. Points without a label are
        left out.
    plane : "xy" or (float, float)
        ``"xy"`` outlines in plan; ``(azimuth, dip)`` on a plane of that strike bearing and dip, in degrees, as in
        `plot.slab`. Vertices lie on the plane through each group's centroid: in plan, at its mean elevation.

    Returns
    -------
    Polylines
        Closed, one feature per category, with the labels in the column `categories` names (``category`` for an
        array).
    """
    if method not in ("convex", "concave"):
        raise InvalidInput("method must be 'convex' or 'concave'")
    if (method == "concave") != (max_edge is not None):
        raise InvalidInput("a concave outline needs max_edge, and only it takes one")
    coords = getattr(points, "coords", points)
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise InvalidInput("points must be (n, 2) or (n, 3) coordinates or a container")
    plane = None if isinstance(plane, str) and plane == "xy" else tuple(plane)
    if categories is None:
        groups, name = [(None, np.ones(len(coords), bool))], None
    else:
        name = categories if isinstance(categories, str) else "category"
        labels = np.asarray(column(points, categories, "categories"), dtype=object)
        if len(labels) != len(coords):
            raise InvalidInput(f"categories: expected {len(coords)} values, got {len(labels)}")
        present = sorted(
            {
                label
                for label in labels
                if label is not None and not (isinstance(label, float) and np.isnan(label))
            }
        )
        groups = [(label, labels == label) for label in present]
    parts, features = [], []
    for i, (_, rows) in enumerate(groups):
        rings = _boitata._outline(coords[rows], max_edge=max_edge, buffer=buffer, plane=plane)
        parts += rings
        features += [i] * len(rings)
    attributes = None if name is None else {name: [label for label, _ in groups]}
    return _boitata.Polylines(
        parts, closed=True, features=features, attributes=attributes, crs=getattr(points, "crs", None)
    )


@dataclass(frozen=True)
class Topography:
    """Result of `topography`.

    Attributes
    ----------
    grid : BlockModel
        2D model of elevation, column ``z``; null outside `outline` and, for the Delaunay surface, outside the points'
        hull.
    mesh : Mesh
        Delaunay surface through the points, clipped to `outline`.
    outline : Polylines or None
        The clip.
    residuals : ndarray
        Each point's elevation minus the surface through the other points at its location.
    flagged : ndarray of bool
        Points whose absolute residual exceeds `max_residual` and those of their neighbors on the Delaunay
        surface: a spike pulls its neighbors' residuals up too, but only the spike is flagged.
    max_residual : float
    """

    grid: object
    mesh: object
    outline: object
    residuals: np.ndarray
    flagged: np.ndarray
    max_residual: float


def topography(points, *, cell, estimator=None, extent=None, clip=None, max_residual=None):
    """Elevation surface and grid through points such as collars or survey shots, with suspect points flagged.

    Parameters
    ----------
    points : PointSet or array_like
        Points, or ``(n, 3)`` coordinates; their z is the elevation.
    cell : float or (float, float)
        Grid cell size along x and y.
    estimator : estimator, optional
        Fitted on the points' elevation and predicted at the cell centers, e.g. ``InverseDistance`` or
        ``OrdinaryKriging``; residuals come from its `cross_validate`. Default the Delaunay surface, linear within
        each triangle and exact at the points.
    extent : (float, float, float, float), optional
        ``(xmin, xmax, ymin, ymax)`` of the grid; default the clip's bounds, or the points'.
    clip : Polylines or False, optional
        Cells and triangles outside it are dropped. Default a concave `outline` of the points with `max_edge` 3 times
        and `buffer` once their median spacing; False keeps everything.
    max_residual : float, optional
        Residual beyond which a point is flagged, if no neighbor on the Delaunay surface has a larger one; default 3
        robust standard deviations, 1.4826 times the median absolute deviation of the residuals, and at least a millionth
        of the elevation range, so round-off on a flat or planar surface flags nothing. Flagged points still
        shape the surface.

    Returns
    -------
    Topography
    """
    coords = np.asarray(getattr(points, "coords", points), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise InvalidInput("points must be a PointSet or (n, 3) coordinates")
    if clip is None:
        spacing = float(np.nanmedian(_boitata.data_spacing(coords, coords, None)))
        clip = outline(coords, method="concave", max_edge=3 * spacing, buffer=spacing)
    elif clip is False:
        clip = None
    dx, dy = (cell, cell) if np.ndim(cell) == 0 else cell
    if extent is None:
        corners = coords if clip is None else clip.coords
        extent = (corners[:, 0].min(), corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max())
    x0, x1, y0, y1 = extent
    count = (max(1, int(np.ceil((x1 - x0) / dx))), max(1, int(np.ceil((y1 - y0) / dy))))
    grid = _boitata.BlockModel((x0, y0), (dx, dy), count)
    plan = np.c_[grid.coords[:, :2], np.zeros(len(grid))]
    mesh = _boitata._tin(coords)
    triangles = np.asarray(mesh.triangles)
    if estimator is None:
        z = -mesh.vertical_distance(plan)
        residuals = _boitata._tin_residuals(coords)
    else:
        estimator.fit(coords[:, :2], coords[:, 2])
        z = np.asarray(estimator.predict(grid), dtype=float)
        cv = estimator.cross_validate()
        residuals = cv.actual - cv.estimate
    if clip is not None:
        z = np.where(clip.contains(plan), z, np.nan)
        centers = np.asarray(mesh.coords)[triangles].mean(axis=1)
        mesh = _boitata.Mesh(mesh.coords, triangles[clip.contains(centers)])
    if max_residual is None:
        deviation = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
        max_residual = float(max(3 * 1.4826 * deviation, 1e-6 * (np.ptp(coords[:, 2]) or 1.0)))
    size = np.nan_to_num(np.abs(residuals))
    largest = size.copy()
    for a, b in ((0, 1), (1, 2), (2, 0)):
        np.maximum.at(largest, triangles[:, a], size[triangles[:, b]])
        np.maximum.at(largest, triangles[:, b], size[triangles[:, a]])
    flagged = (size > max_residual) & (size >= largest)
    return Topography(grid.with_column("z", z), mesh, clip, residuals, flagged, max_residual)
