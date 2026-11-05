"""Generators of the 2D datasets and the seismic cube."""

import numpy as np

from boitata._boitata import BlockModel, PointSet, Polylines, Table, Variogram, object_training_image
from boitata.datasets._fields import classes, field, histogram, inside, ring, stream

P = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
METAL_STREAM = 40377
BEND = (14.548, 84.312)
BODIES = (0.298, 37.17, 141.942, 0.35)
SCATTER = 0.24093
LOW_ZONE = (0.311, 0.655)
RICH = 432.47
METAL_SHARE = (*np.linspace(0.0, 0.95, 20), 0.98, 0.995, 1.0)
METAL_HISTOGRAM = (
    0.0,
    0.0,
    4.8,
    20.7,
    42.1,
    67.8,
    93.8,
    123.2,
    154.8,
    187.5,
    221.2,
    256.2,
    294.1,
    335.4,
    380.0,
    429.3,
    485.2,
    551.5,
    634.0,
    758.3,
    900.0,
    1080.0,
    1631.2,
)


def _metal_grid(seed):
    seed += METAL_STREAM
    box = ((-60.0, -40.0), (321.0, 341.0))
    y, x = np.mgrid[1.0:301.0, 1.0:261.0]
    grid = np.column_stack([x.ravel(), y.ravel()])
    bend = BEND[0] * np.sin(grid[:, 1] / BEND[1] + 0.7)
    bent = grid + np.column_stack([bend, np.zeros(len(grid))])
    grades = Variogram(
        [("spherical", BODIES[0], BODIES[1]), ("spherical", 1 - BODIES[0], BODIES[2])],
        rotation=(170.0, 0.0, 0.0),
        ratios=(BODIES[3], 1.0),
    )
    bodies = field(bent, grades, box, seed, 0)
    bodies = (bodies - bodies.mean()) / bodies.std()
    rng = stream(seed, 4)
    grades_z = bodies + SCATTER * rng.standard_normal(len(grid))
    rank = np.argsort(np.argsort(grades_z))
    share = (rank + 0.5) / len(grid)
    low = share < LOW_ZONE[0]
    inner = LOW_ZONE[1] * bodies[low] + rng.standard_normal(low.sum())
    share[low] = (np.argsort(np.argsort(inner)) + 0.5) / len(grid)
    v = np.interp(share, METAL_SHARE, METAL_HISTOGRAM)
    other = field(bent, grades, box, seed, 1)
    other = (other - other.mean()) / other.std()
    mixed = np.argsort(np.argsort(grades_z + 0.85 * other))
    u = np.interp((mixed + 0.5) / len(grid), P, (0.0, 0.0, 0.12, 57.0, 784.0, 2289.0, 9500.0))
    zt = field(grid, Variogram([("spherical", 1.0, 60.0)]), box, seed, 2)
    return grid, np.round(v, 1), np.round(u, 1), zt


def clustered_metal_survey(seed: int = 0) -> PointSet:
    """470 samples of a 2D metal field, 195 on a jittered 20 m grid and 275 drilled where V is high.

    The grid samples hold no U. The preferential infill biases the naive mean of V high; compare it with
    `clustered_metal_survey_exhaustive`, the truth these samples read.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    PointSet
        ID, V and U (ppm), and type T (1 or 2), on a 260 x 300 m area.
    """
    _, v, u, zt = _metal_grid(seed)
    rng = stream(seed + METAL_STREAM, 3)
    at = lambda xy: (xy[:, 1].astype(int) - 1) * 260 + xy[:, 0].astype(int) - 1
    gy, gx = np.mgrid[10:300:20, 10:260:20]
    regular = np.column_stack([gx.ravel(), gy.ravel()]) + rng.integers(-2, 3, (195, 2))
    taken = set(map(tuple, regular.tolist()))
    infill = []
    rich = regular[v[at(regular)] > RICH]
    while len(infill) < 275:
        p = np.clip(rich[rng.integers(len(rich))] + rng.integers(-9, 10, 2), (8, 8), (251, 291))
        if tuple(p.tolist()) not in taken:
            taken.add(tuple(p.tolist()))
            infill.append(p)
    xy = np.vstack([regular, infill])
    i = at(xy)
    us = u[i].copy()
    us[:195] = np.nan
    return PointSet(
        xy.astype(np.float64),
        {
            "ID": np.arange(1.0, 471.0),
            "V": v[i],
            "U": us,
            "T": np.where(zt[i] > 1.0, 1.0, 2.0),
        },
    )


def clustered_metal_survey_exhaustive(seed: int = 0) -> Table:
    """The exhaustive truth of `clustered_metal_survey`: 78 000 values at 1 m spacing, x fastest.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    Table
        X, Y, U and V.
    """
    grid, v, u, _ = _metal_grid(seed)
    return Table({"X": grid[:, 0], "Y": grid[:, 1], "U": u, "V": v})


METALS = ("Cd", "Co", "Cr", "Cu", "Ni", "Pb", "Zn")
METAL_QUANTILES = {
    "Cd": (0.135, 0.21, 0.39, 1.07, 2.29, 4.2, 5.13),
    "Co": (1.55, 2.26, 3.92, 9.76, 13.5, 17.1, 20.6),
    "Cr": (3.3, 12.9, 20.5, 34.8, 48.7, 61.7, 70.0),
    "Cu": (3.5, 4.42, 7.58, 17.6, 48.8, 109.0, 166.0),
    "Ni": (1.98, 4.74, 7.9, 20.6, 29.3, 37.4, 53.2),
    "Pb": (19.0, 21.5, 30.8, 46.4, 80.5, 182.0, 300.0),
    "Zn": (25.0, 28.2, 42.3, 73.6, 110.0, 150.0, 260.0),
}
METAL_LOADINGS = {
    "Cd": (0.78, 0.15, 0.1),
    "Co": (0.86, 0.0, -0.10),
    "Cr": (0.85, 0.10, 0.10),
    "Cu": (0.15, 0.85, 0.0),
    "Ni": (0.9, 0.10, 0.0),
    "Pb": (0.20, 0.85, 0.05),
    "Zn": (0.8, 0.5, 0.05),
}
STAGES = ("Argovian", "Sequanian", "Kimmeridgian", "Portlandian")
ROCK_SHIFT = {
    "Argovian": {"Co": -1.1, "Ni": -1.0, "Cr": -0.5, "Cd": -0.2, "Zn": -0.4, "Cu": -0.4, "Pb": -0.3},
    "Kimmeridgian": {"Co": 0.3, "Ni": 0.4, "Cd": 0.05, "Zn": 0.05},
    "Sequanian": {"Cr": 0.3, "Cd": 0.25, "Cu": 0.25, "Pb": 0.25, "Zn": 0.2},
    "Portlandian": {"Cd": 0.5, "Cr": 0.3, "Ni": 0.25},
    "Quaternary": {"Cu": 0.25, "Ni": -0.1},
}


def topsoil_metals(seed: int = 0) -> dict[str, PointSet]:
    """Seven topsoil metals with rock and land-use covariates, in km.

    Ni, Co, Cr, Cd and Zn share a regional factor, Cu and Pb a local one, and the rock shifts both.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``prediction`` (259) and ``validation`` (100) PointSets (Landuse, Rock, Cd, Co, Cr, Cu, Ni, Pb, Zn), and the
        ``grid`` PointSet of Landuse and Rock at 50 m nodes.
    """
    box = ((0.0, 0.0), (5.5, 6.2))
    outline = ring((2.75, 2.95), (2.15, 2.5), 40, 0.09, 1.3 + seed)
    gy, gx = np.mgrid[0.1:5.95:0.05, 0.3:5.15:0.05]
    keep = inside(gx.ravel(), gy.ravel(), outline)
    grid = np.column_stack([gx.ravel()[keep], gy.ravel()[keep]]).round(2)
    stages = Variogram([("spherical", 1.0, 2.6)], rotation=(132.0, 0.0, 0.0), ratios=(0.45, 1.0))
    cover = Variogram([("spherical", 1.0, 0.55)])

    edges = np.quantile(field(grid, stages, box, seed, 1), (0.23, 0.54, 0.94))
    roof = np.quantile(field(grid, cover, box, seed, 12), 0.87)

    def rock(p):
        out = np.asarray(STAGES, dtype=object)[np.searchsorted(edges, field(p, stages, box, seed, 1))]
        out[field(p, cover, box, seed, 12) > roof] = "Quaternary"
        return out

    land = lambda p: classes(
        field(p, Variogram([("spherical", 1.0, 1.0)]), box, seed, 2),
        ("Forest", "Meadow", "Pasture", "Tillage"),
        (0.14, 0.56, 0.27, 0.03),
    )
    rng = stream(seed, 0)
    pool = rng.uniform((0.5, 0.5), (4.95, 5.5), (12000, 2))
    pool = pool[inside(pool[:, 0], pool[:, 1], outline)]
    rocks = rock(pool)
    rich = field(pool, Variogram([("spherical", 1.0, 0.25)]), box, seed, 13) > 0.5
    quota = [
        {"Argovian": 53, "Kimmeridgian": 85, "Sequanian": 63, "Portlandian": 3, "Quaternary": 55},
        {"Argovian": 23, "Kimmeridgian": 39, "Sequanian": 26, "Portlandian": 3, "Quaternary": 9},
    ]
    picked = [[], []]
    taken = []
    for point, name, pod in zip(pool, rocks, rich):
        for part in (0, 1):
            if part == 1 and pod:
                continue
            if quota[part][name] > 0 and (not taken or np.hypot(*(np.array(taken) - point).T).min() > 0.05):
                quota[part][name] -= 1
                picked[part].append(point)
                taken.append(point)
                break
    n = len(picked[0])
    xy = np.round(np.vstack(picked), 3)
    regional = Variogram([("spherical", 1.0, 1.45)], rotation=(132.0, 0.0, 0.0), ratios=(0.33, 1.0))
    local = Variogram([("spherical", 1.0, 0.18)])
    factors = [field(xy, v, box, seed, k) for k, v in zip((3, 4, 5), (regional, local, local))]
    rocks = rock(xy)
    samples = {"Landuse": land(xy), "Rock": rocks}
    pods = np.maximum(field(xy, Variogram([("spherical", 1.0, 0.25)]), box, seed, 13), 0.0)
    for k, name in enumerate(METALS):
        load = METAL_LOADINGS[name]
        z = sum(a * f for a, f in zip(load, factors))
        z = z + np.sqrt(max(1 - sum(a * a for a in load), 0.03)) * field(xy, local, box, seed, 6 + k)
        z = z + np.array([ROCK_SHIFT[r].get(name, 0.0) for r in rocks])
        if name == "Cd":
            z = z + 0.45 * pods + 0.25 * factors[0] * (factors[0] > 0)
        samples[name] = np.round(histogram(z / 1.1, P, METAL_QUANTILES[name]), 3)
    split = lambda s: {k: np.asarray(v)[s] for k, v in samples.items()}
    return {
        "prediction": PointSet(xy[:n], split(slice(0, n))),
        "validation": PointSet(xy[n:], split(slice(n, None))),
        "grid": PointSet(grid, {"Landuse": land(grid), "Rock": rock(grid)}),
    }


def coal_seam_thickness(seed: int = 0) -> dict:
    """A coal seam in a lease, with infill drilled where the seam is thick.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``boreholes``: PointSet of 295 holes (THICKNESS_M, ASH_PCT, CV_MJKG, CATEGORY).
        ``boundary``: Polylines, the lease polygon (48 vertices).
        ``grid``: BlockModel of 120 x 90 cells of 100 m, INSIDE the lease or not.
    """
    box = ((19000.0, 49000.0), (33000.0, 60000.0))
    boundary = ring((26000.0, 54300.0), (5900.0, 3900.0), 48, 0.035, 0.7 + seed)
    rng = stream(seed, 0)
    anisotropic = Variogram(
        [("spherical", 0.42, 4500.0), ("spherical", 0.13, 1200.0)],
        nugget=0.45,
        rotation=(35.0, 0.0, 0.0),
        ratios=(0.45, 1.0),
    )
    thickness = lambda p: (
        1.72 - 0.11 * (p[:, 0] - 26000.0) / 1000 + 0.55 * field(p, anisotropic, box, seed, 1)
    )
    gy, gx = np.mgrid[50400.0:58400.0:700.0, 20000.0:32100.0:700.0]
    regional = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-12, 12, (gx.size, 2))
    regional = regional[inside(regional[:, 0], regional[:, 1], boundary)]
    regional = regional[
        np.sort(np.argsort(np.hypot(*((regional - (26000.0, 54300.0)) * (1.0, 1.5)).T))[:145])
    ]
    px, py = rng.uniform((20100, 50500), (31800, 58000), (20000, 2)).T
    pool = np.column_stack([px, py])[inside(px, py, boundary)]
    t_pool = thickness(pool)
    odds = np.where(t_pool > 1.9, t_pool**2, 1e-4)
    chosen = rng.choice(len(pool), 150, replace=False, p=odds / odds.sum())
    xy = np.round(np.vstack([regional, pool[chosen]]), 1)
    t = np.clip(thickness(xy), 0.45, 5.6)
    n = len(xy)
    z = rng.standard_normal(n)
    ash = np.clip(11.8 - 2.4 * (t - 2.23) + 4.0 * z, 4.0, 25.0)
    cv = np.clip(33.0 - 0.34 * ash + rng.normal(0, 0.22, n), 23.0, 31.0)
    missing = rng.random(n) < 0.16
    labels = np.where(t > 1.4, "MECHANIZED", np.where(t > 0.8, "SELECTIVE", "UNECONOMIC")).astype(object)
    gy, gx = np.mgrid[50050.0:59000.0:100.0, 20050.0:32000.0:100.0]
    return {
        "boreholes": PointSet(
            xy,
            {
                "ID": np.arange(1.0, n + 1),
                "THICKNESS_M": np.round(t, 2),
                "ASH_PCT": np.where(missing, np.nan, np.round(ash, 1)),
                "CV_MJKG": np.where(missing, np.nan, np.round(cv, 2)),
                "CATEGORY": labels,
            },
        ),
        "boundary": Polylines([np.round(boundary, 1)], closed=True),
        "grid": BlockModel(
            (20000.0, 50000.0),
            (100.0, 100.0),
            (120, 90),
            attributes={"INSIDE": inside(gx.ravel(), gy.ravel(), boundary).astype(np.float64)},
        ),
    }


def soil_geochemistry_survey(seed: int = 0) -> dict:
    """Soil samples along survey lines, with detection limits and covariates.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``samples``: PointSet of 1227 samples (CU_PPM, ZN_PPM, AU_PPB, AS_PPM; ``AU_BDL`` and ``AS_BDL`` flag values
        below detection, which hold the limit).
        ``boundary``: Polylines, the survey area (36 vertices).
        ``covariates``: BlockModel of 120 x 80 cells of 50 m (ELEVATION_M, MAG_NT, LITHOLOGY, INSIDE).
    """
    box = ((2800.0, 6900.0), (9200.0, 11100.0))
    boundary = ring((6000.0, 9000.0), (2900.0, 1950.0), 36, 0.03, 2.1 + seed)
    corridor_angle = np.radians(55.0)
    corridor = lambda p: np.exp(
        -(
            (
                ((p[:, 0] - 6000.0) * np.cos(corridor_angle) - (p[:, 1] - 9200.0) * np.sin(corridor_angle))
                / 700.0
            )
            ** 2
        )
    )
    trend = Variogram([("gaussian", 1.0, 1800.0)], rotation=(35.0, 0.0, 0.0), ratios=(0.4, 1.0))
    mag_z = lambda p: field(p, trend, box, seed, 1)
    mag = lambda p: 51940.0 + 165.0 * mag_z(p) + 210.0 * corridor(p)
    lithology = lambda p: classes(
        field(p, Variogram([("spherical", 1.0, 2200.0)]), box, seed, 2),
        ("METASEDIMENT", "MAFIC_VOLCANIC", "FELSIC_VOLCANIC", "GRANITE"),
        (0.2, 0.36, 0.22, 0.22),
    )
    rng = stream(seed, 0)
    lines = [y for y in np.arange(7700.0, 11000.0, 200.0) if not 8000.0 <= y < 8400.0]
    stations = [
        (x + rng.normal(0, 3.0), y + rng.normal(0, 4.0))
        for y in lines
        for x in np.arange(3000.0, 9000.0, 50.0)
    ]
    xy = np.asarray(stations)
    xy = xy[inside(xy[:, 0], xy[:, 1], boundary)]
    xy = xy[np.sort(rng.permutation(len(xy))[: min(len(xy), 1227)])]
    xy = np.round(xy, 1)
    n = len(xy)
    local = Variogram([("spherical", 0.7, 700.0)], nugget=0.3)
    zc = field(xy, local, box, seed, 3)
    litho = lithology(xy)
    shift = np.array(
        [
            {"METASEDIMENT": -0.35, "MAFIC_VOLCANIC": 0.05, "FELSIC_VOLCANIC": 0.1, "GRANITE": 0.1}[k]
            for k in litho
        ]
    )
    relief = Variogram([("gaussian", 1.0, 1500.0)])
    base = 0.6 * zc + 0.75 * corridor(xy) + 0.17 * mag_z(xy) + 0.3 * field(xy, relief, box, seed, 7) + shift
    cu = np.clip(np.exp(3.68 + 0.8 * base), 10.0, 300.0)
    zn = np.clip(np.exp(4.2 + 0.2 * base + 0.5 * field(xy, local, box, seed, 4)), 16.0, 360.0)
    au = np.exp(0.72 + 1.2 * base + 0.75 * field(xy, local, box, seed, 5))
    as_ = np.exp(1.95 + 0.75 * base + 0.6 * field(xy, local, box, seed, 6))
    gy, gx = np.mgrid[7025.0:11000.0:50.0, 3025.0:9000.0:50.0]
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    elevation = 400.0 + 66.0 * field(grid, Variogram([("gaussian", 1.0, 1500.0)]), box, seed, 7)
    return {
        "samples": PointSet(
            xy,
            {
                "ID": np.arange(1.0, n + 1),
                "CU_PPM": np.round(cu),
                "ZN_PPM": np.where(rng.random(n) < 0.29, np.nan, np.round(zn)),
                "AU_PPB": np.round(np.maximum(au, 2.0), 1),
                "AS_PPM": np.round(np.maximum(as_, 5.0)),
                "AU_BDL": (au < 2.0).astype(np.float64),
                "AS_BDL": (as_ < 5.0).astype(np.float64),
            },
        ),
        "boundary": Polylines([np.round(boundary, 1)], closed=True),
        "covariates": BlockModel(
            (3000.0, 7000.0),
            (50.0, 50.0),
            (120, 80),
            attributes={
                "ELEVATION_M": np.round(elevation, 1),
                "MAG_NT": np.round(mag(grid), 0),
                "LITHOLOGY": lithology(grid),
                "INSIDE": inside(grid[:, 0], grid[:, 1], boundary).astype(np.float64),
            },
        ),
    }


def channel_training_image(seed: int = 0) -> BlockModel:
    """A 250 x 250 channel training image: sinuous sand channels along Y in shale.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    BlockModel
        250 x 250 x 1 cells of 1 m from the origin (0, 0), with ``facies`` 0 (shale) or 1 (sand); sand is about 28 % of
        the cells. The spacing is nominal: the image has no physical scale.
    """
    grid = BlockModel((0.0, 0.0), (1.0, 1.0), (250, 250))
    channels = {
        "shape": "channel",
        "code": 1,
        "proportion": 0.28,
        "width": (7.0, 11.0),
        "azimuth": (-12.0, 12.0),
        "amplitude": (8.0, 16.0),
        "wavelength": (70.0, 110.0),
    }
    image = object_training_image(grid, [channels], seed=seed)
    return BlockModel(
        grid.origin,
        grid.size,
        grid.count,
        attributes={"facies": np.asarray(image["facies"], dtype=np.float64)},
    )


def seismic_cube(seed: int = 0) -> BlockModel:
    """A post-stack seismic cube: dipping reflectors cut by two normal faults, with a strong reflector near 1625 ms.

    45 x 45 traces of 51 samples in 25 m bins, at two-way time 1600-1800 ms and 4 ms, rotated to the inline direction.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    BlockModel
        45 x 45 x 51 cells with ``amplitude`` and ``crs`` set to EPSG:23031. Z is minus the two-way time.
    """
    origin, size, count = (612635.6144137512, 6080361.3326072795, -1802.0), (25.0, 25.0, 4.0), (45, 45, 51)
    nx, ny, nz = count
    rng = stream(seed, 0)
    k, j, i = np.mgrid[0:nz, 0:ny, 0:nx]
    x, y = (i + 0.5) * size[0], (j + 0.5) * size[1]
    time = 1800.0 - (k + 0.5) * size[2]
    faults = sum(
        throw * (x - x0 - slope * (time - 1700.0) > 0)
        for x0, slope, throw in ((340.0, 2.2, 14.0), (780.0, 2.6, -10.0))
    )
    relief = 1.8 * np.sin(2 * np.pi * y / 900.0 + 0.6)
    shifted = time - 0.01 * x - 0.004 * y + relief - faults
    horizons = np.sort(rng.uniform(1540.0, 1880.0, 26))
    strengths = rng.normal(-150.0, 3200.0, 26)
    horizons = np.append(horizons, 1625.0 - 0.01 * 560.0 - 0.004 * 560.0)
    strengths = np.append(strengths, -15000.0)
    amplitude = np.zeros_like(shifted)
    phases = rng.uniform(0.0, 2 * np.pi, (len(horizons), 2))
    for t0, a, (p, q) in zip(horizons, strengths, phases):
        s = (shifted - t0) / 10.0
        character = 1 + 0.8 * np.sin(y / 60.0 + p) + 0.3 * np.sin(x / 140.0 + q)
        amplitude += a * character * (1 - 2 * s**2) * np.exp(-(s**2))
    amplitude += rng.normal(0.0, 350.0, amplitude.shape)
    return BlockModel(
        origin,
        size,
        count,
        rotation=(88.39878818158118, 0.0, 0.0),
        attributes={"amplitude": np.round(amplitude.ravel(), 2)},
        crs="EPSG:23031",
    )
