"""Outlines, surfaces and collars built from scattered points."""

import numpy as np

from boitata import _boitata
from boitata._columns import column
from boitata.errors import InvalidInput

__all__ = ["outline"]


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
    if hasattr(points, "coords"):
        coords = points.coords
    elif hasattr(points, "centroids"):
        coords = points.centroids
    else:
        coords = points
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
