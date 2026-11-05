"""Shared pieces of the dataset generators: seeded Gaussian fields, histograms, holes and closed solids."""

import math

import numpy as np

from boitata._boitata import Mesh, Variogram, _gaussian_field, _realization_seed


def stream(seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng(_realization_seed(seed, k))


def field(points, variogram: Variogram, box, seed: int, k: int) -> np.ndarray:
    """Standard-Gaussian field at `points`; `box`, `seed` and `k` fix it, so any points read one truth."""
    lo, hi = ([*c, *[0.0] * (3 - len(c))] for c in box)
    return _gaussian_field(points, variogram, lo, hi, seed=_realization_seed(seed, k))


def phi(z) -> np.ndarray:
    return 0.5 * (1 + np.frompyfunc(math.erf, 1, 1)(np.asarray(z) / math.sqrt(2)).astype(np.float64))


def histogram(z, probabilities, values) -> np.ndarray:
    """Standard-Gaussian `z` read through a quantile table, the anamorphosis to a target histogram."""
    return np.interp(phi(z), probabilities, values)


def classes(z, names, proportions) -> np.ndarray:
    """Ordered categories from thresholds on a standard-Gaussian `z`."""
    return np.asarray(names, dtype=object)[np.searchsorted(np.cumsum(proportions)[:-1], phi(z))]


def ring(center, radii, n, wobble, phase, lobes=(3, 5, 7)) -> np.ndarray:
    """Closed irregular outline of `n` vertices, smooth and fixed by `phase`."""
    a = np.arange(n) * 2 * np.pi / n
    bump = sum(np.sin(k * a + phase * (i + 1)) / (i + 1) for i, k in enumerate(lobes))
    r = 1 + wobble * bump
    return np.column_stack([center[0] + radii[0] * r * np.cos(a), center[1] + radii[1] * r * np.sin(a)])


def inside(x, y, polygon) -> np.ndarray:
    px, py = polygon[:, 0], polygon[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    x, y = np.asarray(x)[..., None], np.asarray(y)[..., None]
    crosses = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = px + (y - py) * (qx - px) / (qy - py)
    return (crosses & (x < at)).sum(-1) % 2 == 1


def holes_path(collar, stations, dips, azimuths) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-curvature path of one hole: `stations` depths and their XYZ. Dip is positive downward."""
    d, a = np.radians(dips), np.radians(azimuths)
    t = np.column_stack([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), -np.sin(d)])
    cos = np.clip((t[:-1] * t[1:]).sum(1), -1, 1)
    bend = np.arccos(cos)
    small = bend < 1e-9
    factor = np.where(small, 1.0, 2 / np.where(small, 1.0, bend) * np.tan(bend / 2))
    step = (np.diff(stations) / 2 * factor)[:, None] * (t[:-1] + t[1:])
    return stations, np.asarray(collar) + np.vstack([np.zeros(3), np.cumsum(step, 0)])


def along(path, depth) -> np.ndarray:
    stations, xyz = path
    return np.column_stack([np.interp(depth, stations, xyz[:, i]) for i in range(3)])


def runs(hole, depth_to, labels, step) -> list[tuple]:
    """Consecutive equal `labels` of cells ending at `depth_to` as (hole, from, to, label)."""
    edges = np.flatnonzero(labels[1:] != labels[:-1])
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, len(labels) - 1]
    return [(hole, depth_to[s] - step, depth_to[e], labels[s]) for s, e in zip(starts, ends)]


def plate(mid, half, u, v, world) -> Mesh:
    """Closed solid between mid(u, v) -+ half(u, v), on the grid `u` by `v` placed by `world(u, v, w)`.

    Two sheets and four walls sharing their edge vertices, so it is watertight by construction.
    """
    uu, vv = np.meshgrid(u, v, indexing="ij")
    w, t = mid(uu, vv), np.maximum(half(uu, vv), 1e-3)
    nu, nv = uu.shape
    top, bottom = world(uu, vv, w + t).reshape(-1, 3), world(uu, vv, w - t).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    a, b, c, d = (x.ravel() for x in (i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1))
    shift = nu * nv
    sheets = np.vstack(
        [
            np.c_[a, b, c],
            np.c_[a, c, d],
            np.c_[a + shift, c + shift, b + shift],
            np.c_[a + shift, d + shift, c + shift],
        ]
    )
    at = lambda i, j: i * nv + j
    rim = [at(i, 0) for i in range(nu - 1)] + [at(nu - 1, j) for j in range(nv - 1)]
    rim += [at(i, nv - 1) for i in range(nu - 1, 0, -1)] + [at(0, j) for j in range(nv - 1, 0, -1)]
    p, q = np.array(rim), np.roll(rim, -1)
    walls = np.vstack([np.c_[p, p + shift, q + shift], np.c_[p, q + shift, q]])
    mesh = Mesh(np.vstack([top, bottom]), np.vstack([sheets, walls]).astype(np.int64))
    return mesh if mesh.volume >= 0 else Mesh(mesh.coords, mesh.triangles[:, ::-1])


def framed(f, origin):
    return lambda u, v, w: np.stack([u, v, w], -1) @ f + np.asarray(origin)


def to_disc(p, q, a, b):
    """The square (p, q) in [-1, 1] on the ellipse of semi-axes `a` and `b`; its edge lands on the outline."""
    return a * p * np.sqrt(1 - q**2 / 2), b * q * np.sqrt(1 - p**2 / 2)


def gridded(origin, size, count, height):
    """Heights of a regular grid of cell centers and their bilinear reader; the reader honors the grid exactly."""
    nx, ny = count
    y, x = np.mgrid[0:ny, 0:nx]
    xy = np.column_stack([origin[0] + size * (x.ravel() + 0.5), origin[1] + size * (y.ravel() + 0.5)])
    values = np.round(height(xy), 1).reshape(ny, nx)

    def read(points):
        points = np.atleast_2d(points)
        fx = np.clip((points[:, 0] - origin[0]) / size - 0.5, 0.0, nx - 1.000001)
        fy = np.clip((points[:, 1] - origin[1]) / size - 0.5, 0.0, ny - 1.000001)
        i, j = fx.astype(int), fy.astype(int)
        tx, ty = fx - i, fy - j
        return (
            values[j, i] * (1 - tx) * (1 - ty)
            + values[j, i + 1] * tx * (1 - ty)
            + values[j + 1, i] * (1 - tx) * ty
            + values[j + 1, i + 1] * tx * ty
        )

    return values.ravel(), read


def frame(azimuth: float, dip: float) -> np.ndarray:
    """Rows: along strike, down dip and the upward normal of a plane dipping `dip` toward `azimuth`."""
    a, d = math.radians(azimuth), math.radians(dip)
    strike = np.array([-math.cos(a), math.sin(a), 0.0])
    down = np.array([math.sin(a) * math.cos(d), math.cos(a) * math.cos(d), -math.sin(d)])
    return np.vstack([strike, down, np.cross(down, strike)])
