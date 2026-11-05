"""Generators of the drilled 3D datasets. Each deposit is a geology function that logs, assays and solids all read."""

import csv
import itertools
import math
import tempfile
from pathlib import Path

import numpy as np

from boitata._boitata import BlockModel, PointSet, Table, Variogram, read_csv
from boitata.datasets._fields import (
    along,
    field,
    frame,
    framed,
    gridded,
    holes_path,
    plate,
    runs,
    stream,
    to_disc,
)
from boitata.errors import InvalidInput

HOLE_COLUMNS = ("HOLE_ID", "X", "Y", "Z", "LENGTH", "TYPE")


def _lognormal(median, sigma, z):
    return median * np.exp(sigma * z)


def _drill(plan, rng, spacing, dip_drift=0.0, azimuth_drift=0.0, flatten=0.0, swing=0.0):
    """Collars and surveys of `plan` rows (id, x, y, z, length, dip, azimuth, type), and each hole's path."""
    collars, surveys, paths = [], [], {}
    for hole, x, y, z, length, dip, azimuth, kind in plan:
        stations = np.unique(np.r_[np.arange(0.0, length, spacing), length])
        dips = np.clip(
            dip + np.cumsum(rng.normal(-flatten, dip_drift, len(stations))) * (dip_drift > 0), -90, 90
        )
        azimuths = azimuth + np.cumsum(rng.normal(swing, azimuth_drift, len(stations))) * (azimuth_drift > 0)
        paths[hole] = holes_path((x, y, z), stations, dips, azimuths)
        collars.append((hole, round(x, 1), round(y, 1), round(z, 1), round(length, 2), kind))
        surveys += [
            (hole, round(s, 2), round(d, 2), round(a, 2)) for s, d, a in zip(stations, dips, azimuths)
        ]
    return collars, surveys, paths


def _table(rows, names):
    columns = list(zip(*rows)) if rows else [()] * len(names)
    return Table(
        {
            n: np.asarray(c, dtype=np.float64)
            if not isinstance(c[0], (str, type(None)))
            else np.asarray(c, dtype=object)
            for n, c in zip(names, columns)
        }
    )


def _intervals(lithology, nominal):
    """Sample intervals (hole, from, to, lith) that break at every contact."""
    out = []
    for hole, a, b, label in lithology:
        length = b - a
        n = max(math.ceil(length / nominal - 1e-9), 1)
        edges = np.linspace(a, b, n + 1)
        out += [(hole, round(float(s), 2), round(float(e), 2), label) for s, e in itertools.pairwise(edges)]
    return out


def _midpoints(paths, intervals):
    xyz = np.empty((len(intervals), 3))
    by_hole = {}
    for i, row in enumerate(intervals):
        by_hole.setdefault(row[0], []).append(i)
    for hole, ids in by_hole.items():
        depth = np.array([(intervals[i][1] + intervals[i][2]) / 2 for i in ids])
        xyz[ids] = along(paths[hole], depth)
    return xyz


def _log(paths, classify, step):
    """Lithology runs of every hole from `classify(xyz) -> labels`, read every `step` meters in one batch."""
    depths = {}
    for hole, path in paths.items():
        length = path[0][-1]
        n = max(int(length / step), 1)
        depths[hole] = np.linspace(length / n, length, n)
    points = np.vstack([along(paths[h], t - (t[0] / 2)) for h, t in depths.items()])
    labels = classify(points)
    out, at = [], 0
    for hole, to in depths.items():
        part = labels[at : at + len(to)]
        at += len(to)
        out += [
            (h, round(float(a), 2), round(float(b), 2), str(k)) for h, a, b, k in runs(hole, to, part, to[0])
        ]
    return out


def _surface(origin, size, count, values, name="Z"):
    return BlockModel(origin, (size, size), count, attributes={name: values})


def _lens_geometry():
    f = frame(112.5, 55.0)
    lenses = (
        ((12080.0, 29843.0, 174.0), 318.0, 186.0, 15.5, 0.4),
        ((12312.0, 30067.0, 57.0), 270.0, 225.0, 10.0, 1.9),
        ((12372.0, 29852.0, -96.0), 360.0, 155.0, 9.5, 3.1),
    )
    return f, lenses


def _lens_mid(u, v, phase):
    return 5.0 * np.sin(u / 95.0 + phase) + 3.5 * np.cos(v / 70.0 + 2 * phase)


def _lens_half(u, v, a, b, peak, phase):
    r2 = (u / a) ** 2 + (v / b) ** 2
    return (
        peak * np.maximum(1 - r2, 0.0) * (1 + 0.22 * np.sin(u / 48.0 + 2 * phase) * np.cos(v / 37.0 - phase))
    )


def _disc_plate(mid, half, a, b, n, world):
    grid = np.linspace(-1.0, 1.0, n)
    return plate(
        lambda p, q: mid(*to_disc(p, q, a, b)),
        lambda p, q: np.maximum(half(*to_disc(p, q, a, b)), 0.05),
        grid,
        grid,
        lambda p, q, w: world(*to_disc(p, q, a, b), w),
    )


def _sulphide_tables(seed):
    box = ((10700.0, 28800.0, -420.0), (13700.0, 31600.0, 430.0))
    f, lenses = _lens_geometry()
    rng = stream(seed, 0)
    topo_v = Variogram([("gaussian", 1.0, 1500.0)])
    topography, ground = gridded(
        (10790.0, 28890.0),
        20.0,
        (141, 131),
        lambda xy: 379.0 + 9.5 * field(xy, topo_v, box, seed, 1) + 0.008 * (xy[:, 0] - 12200.0),
    )
    centers = np.array([c for c, *_ in lenses])
    top = np.max((centers - centers[0]) @ f[2]) + 30.0
    bottom = np.min((centers - centers[0]) @ f[2]) - 22.0
    dyke = frame(20.0, 85.0)
    ms = Variogram([("spherical", 1.0, 75.0)], rotation=(22.5, 55.0, 0.0), ratios=(0.8, 0.5))

    def geology(xyz):
        out = np.empty(len(xyz), dtype=object)
        base = (xyz - centers[0]) @ f[2]
        out[:] = "VCL"
        out[base > top] = "HWS"
        out[base < bottom] = "FWV"
        out[np.abs((xyz - np.array([12290.0, 29600.0, 200.0])) @ dyke[2]) < 2.2] = "DYK"
        for c, a, b, peak, phase in lenses:
            d = (xyz - np.asarray(c)) @ f.T
            u, v, w = d[:, 0], d[:, 1], d[:, 2] - _lens_mid(d[:, 0], d[:, 1], phase)
            h = _lens_half(u, v, a, b, peak, phase)
            r2 = (u / a) ** 2 + (v / b) ** 2
            stringer = (r2 < 1) & (w < -h) & (w > -h - 13.0 * np.sqrt(np.maximum(1 - r2, 0.0)))
            out[stringer] = "STR"
            within = (h > 0.0) & (np.abs(w) < h)
            out[within] = "SMS"
            out[within & (np.abs(w) < 0.58 * h)] = "MS"
        ob = xyz[:, 2] > ground(xyz[:, :2]) - (2.0 + 4.0 * (np.sin(xyz[:, 0] / 130.0) + 1))
        out[ob] = "OB"
        return out

    strike = f[0]
    plan = []
    number = 0
    aims = [
        (np.array(lenses[0][0]), s, v)
        for s in np.arange(-420.0, 421.0, 60.0)
        for v in np.arange(-210.0, 211.0, 70.0)
    ]
    aims += [
        (np.array(lenses[1][0]), s, v)
        for s in np.arange(-400.0, 401.0, 80.0)
        for v in np.arange(-240.0, 241.0, 80.0)
    ]
    aims += [
        (np.array(lenses[2][0]), s, v)
        for s in np.arange(-480.0, 481.0, 80.0)
        for v in np.arange(-120.0, 121.0, 60.0)
    ]
    for start, s, v in aims:
        if number < 244:
            number += 1
            dip = 61.0 + rng.normal(0, 6.0)
            azimuth = 294.0 + rng.normal(0, 3.5)
            target = start + s * strike + v * f[1]
            target += rng.normal(0, 10.0, 3)
            d = np.array(
                [
                    math.sin(math.radians(azimuth)) * math.cos(math.radians(dip)),
                    math.cos(math.radians(azimuth)) * math.cos(math.radians(dip)),
                    -math.sin(math.radians(dip)),
                ]
            )
            collar = target.copy()
            for _ in range(3):
                z = ground(collar[None, :2])[0]
                length = (z - target[2]) / -d[2]
                collar = target - d * length
            collar[2] = ground(collar[None, :2])[0]
            total = max((collar[2] - target[2]) / -d[2] + rng.uniform(70.0, 150.0), 90.0)
            plan.append((f"DD{number:04d}", collar[0], collar[1], collar[2], total, dip, azimuth, "DD"))
    for k in range(45):
        dip, azimuth = 60.0 + rng.normal(0, 3.0), 294.0 + rng.normal(0, 2.0)
        d = np.array(
            [
                math.sin(math.radians(azimuth)) * math.cos(math.radians(dip)),
                math.cos(math.radians(azimuth)) * math.cos(math.radians(dip)),
                -math.sin(math.radians(dip)),
            ]
        )
        target = (
            np.array(lenses[0][0]) + rng.uniform(-200.0, 200.0) * strike + rng.uniform(-170.0, -120.0) * f[1]
        )
        collar = target.copy()
        for _ in range(3):
            collar = target - d * (ground(collar[None, :2])[0] - target[2]) / -d[2]
        collar[2] = ground(collar[None, :2])[0]
        plan.append(
            (
                f"RC{k + 1:04d}",
                collar[0],
                collar[1],
                collar[2],
                (collar[2] - target[2]) / -d[2] + rng.uniform(15.0, 35.0),
                dip,
                azimuth,
                "RC",
            )
        )
    collars, surveys, paths = _drill(plan, rng, 30.0, 0.25, 0.2, flatten=0.42, swing=0.25)
    lithology = _log(paths, geology, 0.25)

    mineral = {"MS", "SMS", "STR"}
    assayed = []
    for hole in paths:
        rows = [r for r in lithology if r[0] == hole]
        zones = [(a, b) for _, a, b, k in rows if k in mineral]
        if hole.startswith("RC"):
            assayed += [r for r in rows if r[3] != "OB" and not (r[3] not in mineral and rng.random() < 0.15)]
            continue
        if not zones:
            continue
        w = (along(paths[hole], paths[hole][0]) - centers[0]) @ f[2]
        crossing = paths[hole][0][(w < top + 15.0) & (w > bottom - 60.0)]
        lo = min([min(a for a, _ in zones) - 15.0, *crossing[:1]])
        hi = max([max(b for _, b in zones) + 15.0, *crossing[-1:]])
        for h, a, b, k in rows:
            if k != "OB" and b > lo and a < hi:
                assayed.append((h, max(a, lo), min(b, hi), k))
    samples = []
    for row in assayed:
        samples += _intervals([row], 1.0 if row[0].startswith("RC") else 2.0)
    xyz = _midpoints(paths, samples)
    n = len(samples)
    litho = np.array([s[3] for s in samples])
    common = field(xyz, ms, box, seed, 2)
    noise = rng.standard_normal((6, n))
    z = lambda k: 0.93 * common + 0.37 * noise[k]
    table = {
        "VCL": (0.041, 0.016, 0.007, 0.8, 0.013),
        "HWS": (0.04, 0.015, 0.002, 0.7, 0.013),
        "FWV": (0.045, 0.018, 0.027, 0.9, 0.013),
        "DYK": (0.046, 0.018, 0.015, 0.8, 0.016),
        "SMS": (2.88, 1.0, 0.229, 21.7, 0.306),
        "MS": (7.6, 2.82, 0.322, 56.0, 0.697),
        "STR": (0.324, 0.086, 0.839, 6.5, 0.475),
    }
    med = np.array([table[k] for k in litho])
    footwall = np.zeros(n)
    for c, a, b, peak, phase in lenses:
        d = (xyz - np.asarray(c)) @ f.T
        h = _lens_half(d[:, 0], d[:, 1], a, b, peak, phase)
        inside = (np.abs(d[:, 2]) < 2 * h + 1) & np.isin(litho, ["MS", "SMS"])
        footwall[inside] = np.clip(-(d[:, 2] / np.maximum(h, 1.0))[inside], -1.0, 1.0)
    zn = _lognormal(med[:, 0], 0.63, z(0))
    pb = _lognormal(med[:, 1], 0.7, 0.9 * z(0) + 0.3 * z(1))
    cu = _lognormal(med[:, 2] * np.exp(0.9 * footwall), 0.88, z(2))
    ag = _lognormal(med[:, 3], 0.76, 0.8 * z(1) + 0.4 * z(0))
    au = _lognormal(med[:, 4], 1.0, z(3))
    rc = np.array([s[0].startswith("RC") for s in samples])
    measured = np.isin(litho, list(mineral)) & (rng.random(n) < 0.47)
    density = 2.82 + 0.077 * (zn + pb + cu) + rng.normal(0, 0.03, n)
    density = np.where(measured & ~rc, np.round(np.clip(density, 2.82, 4.8), 2), np.nan)
    rows = [
        (
            s[0],
            s[1],
            s[2],
            round(float(a), 3),
            round(float(b), 3),
            round(float(c), 3),
            round(float(d), 1),
            None if r else round(float(e), 3),
            g,
        )
        for s, a, b, c, d, e, g, r in zip(
            samples, zn, pb, np.clip(cu, 0.0, 16.5), ag, np.clip(au, 0.0, 22.0), density, rc
        )
    ]
    names = ("HOLE_ID", "FROM", "TO", "ZN_PCT", "PB_PCT", "CU_PCT", "AG_GPT", "AU_GPT", "DENSITY")
    return collars, surveys, rows, names, lithology, topography, lenses, f


def stacked_sulphide_lenses(raw: bool = False, seed: int = 0) -> dict:
    """Three dipping polymetallic lenses cut by 244 diamond and 45 RC holes.

    Parameters
    ----------
    raw : bool
        Return the drillhole tables as logged, with planted errors to check and fix (duplicate and missing holes,
        overlapping, inverted and past-depth intervals, survey flips, text in numeric columns, twinned samples), read
        as-is: -99 and -999 stay numbers.
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``collars``, ``surveys``, ``assays`` (ZN_PCT, PB_PCT, CU_PCT, AG_GPT, AU_GPT, DENSITY) and ``lithology`` (LITH)
        tables. Unless `raw`, also ``lens_1`` to ``lens_3`` Mesh solids and ``topography``, a BlockModel of 141 x 131
        cells of 20 m with Z.
    """
    collars, surveys, assays, names, lithology, topography, lenses, f = _sulphide_tables(seed)
    if raw:
        return _planted(collars, surveys, assays, names, lithology, seed)
    out = {
        "collars": _table(collars, HOLE_COLUMNS),
        "surveys": _table(surveys, ("HOLE_ID", "DEPTH", "DIP", "AZIMUTH")),
        "assays": _table(assays, names),
        "lithology": _table(lithology, ("HOLE_ID", "FROM", "TO", "LITH")),
    }
    for i, (c, a, b, peak, phase) in enumerate(lenses):
        out[f"lens_{i + 1}"] = _disc_plate(
            lambda u, v, phase=phase: _lens_mid(u, v, phase),
            lambda u, v, a=a, b=b, peak=peak, phase=phase: _lens_half(u, v, a, b, peak, phase),
            a,
            b,
            90,
            framed(f, c),
        )
    out["topography"] = _surface((10790.0, 28890.0), 20.0, (141, 131), topography)
    return out


def _planted(collars, surveys, assays, names, lithology, seed):
    """The clean tables with data-entry errors planted, read back as logged: text stays text, -99 stays a number."""
    rng = stream(seed, 90)
    text = lambda rows: [
        ["" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in r] for r in rows
    ]
    collars, surveys, assays, lith = text(collars), text(surveys), text(assays), text(lithology)
    col = {n: i for i, n in enumerate(names)}
    grades = ("ZN_PCT", "PB_PCT", "CU_PCT", "AG_GPT", "AU_GPT")
    of = lambda rows, hole: [r for r in rows if r[0] == hole]
    counts = {}
    for r in assays:
        counts[r[0]] = counts.get(r[0], 0) + 1
    candidates = sorted(h for h, c in counts.items() if h.startswith("DD") and c >= 20)
    picks = [str(p) for p in rng.choice(candidates, 17, replace=False)]
    dup_id, dup_collar, no_assays, no_collar, overlap, inverted, beyond, gaps, no_survey = picks[:9]
    azi_flip, dip_flip, sentinels, written, twins, case, space_lith, space_collar = picks[9:]

    c = of(collars, dup_id)[0]
    second = [
        dup_id,
        f"{float(c[1]) + rng.uniform(180, 260):.1f}",
        f"{float(c[2]) - rng.uniform(120, 200):.1f}",
        f"{float(c[3]) + rng.uniform(-3, 3):.1f}",
        f"{float(c[4]) * rng.uniform(0.6, 0.8):.2f}",
        c[5],
    ]
    collars.insert(collars.index(c) + 1, second)
    c = of(collars, dup_collar)[0]
    collars.insert(collars.index(c) + 1, list(c))
    assays[:] = [r for r in assays if r[0] != no_assays]
    collars.remove(of(collars, no_collar)[0])
    rows = of(assays, overlap)
    for r in (rows[len(rows) // 3], rows[2 * len(rows) // 3]):
        r[1] = f"{float(r[1]) - rng.uniform(0.4, 0.9):.2f}"
    r = of(assays, inverted)[len(of(assays, inverted)) // 2]
    r[1], r[2] = r[2], r[1]
    rows = of(assays, beyond)
    of(collars, beyond)[0][4] = f"{float(rows[-3][1]) - 0.5:.2f}"
    rows = of(assays, gaps)
    for r in (rows[len(rows) // 4], rows[3 * len(rows) // 4]):
        assays.remove(r)
    surveys[:] = [r for r in surveys if r[0] != no_survey]
    r = of(surveys, azi_flip)[len(of(surveys, azi_flip)) // 2]
    r[3] = f"{(float(r[3]) + 180) % 360:.2f}"
    for r in of(surveys, dip_flip):
        r[2] = f"{-float(r[2]):.2f}"
    ore = {h for h, _, _, k in lithology if k == "MS"}
    sentinels = next((h for h in candidates if h in ore and h not in picks), sentinels)
    massive = [(float(a), float(b)) for h, a, b, k in lith if h == sentinels and k == "MS"]
    rows = of(assays, sentinels)
    rich = [r for r in rows if any(a <= (float(r[1]) + float(r[2])) / 2 < b for a, b in massive)]
    others = [r for r in rows if r not in rich]
    chosen = [rich[len(rich) // 2], *(others[i] for i in rng.choice(len(others), 4, replace=False))]
    for r, name, value in zip(
        chosen, ("ZN_PCT", "CU_PCT", "AG_GPT", "PB_PCT", "AU_GPT"), ("-999", "-999", "-99", "-99", "-999")
    ):
        r[col[name]] = value
    rows = of(assays, written)
    r1, r2, r3 = (rows[i] for i in rng.choice(len(rows), 3, replace=False))
    r1[col["CU_PCT"]] = "<0.01"
    r2[col["AU_GPT"]] = "<0.01"
    for name in grades:
        r3[col[name]] = "NS"
    rows = of(assays, twins)
    start = int(rng.integers(len(rows) // 4, len(rows) - 3))
    at = assays.index(rows[start + 2]) + 1
    for r in rows[start : start + 3]:
        t = list(r)
        for name in grades:
            if t[col[name]]:
                decimals = len(t[col[name]].split(".")[1]) if "." in t[col[name]] else 0
                t[col[name]] = f"{float(t[col[name]]) * (1 + rng.uniform(-0.08, 0.08)):.{decimals}f}"
        assays.insert(at, t)
        at += 1
    rows = of(assays, case)
    for r in rows[len(rows) // 2 : len(rows) // 2 + 5]:
        r[0] = r[0].lower()
    for r in of(lith, space_lith)[1:3]:
        r[0] += " "
    of(collars, space_collar)[0][0] += " "

    tables = {
        "collars": (HOLE_COLUMNS, collars),
        "surveys": (("HOLE_ID", "DEPTH", "DIP", "AZIMUTH"), surveys),
        "assays": (names, assays),
        "lithology": (("HOLE_ID", "FROM", "TO", "LITH"), lith),
    }
    with tempfile.TemporaryDirectory() as folder:
        out = {}
        for name, (head, rows) in tables.items():
            path = Path(folder) / f"{name}.csv"
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(head)
                writer.writerows(rows)
            out[name] = read_csv(path, nodata=[])
    return out


def iron_formation_plateau(seed: int = 0) -> dict:
    """An iron formation with exploration holes, grade-control blastholes and a block model.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``collars``, ``surveys``, ``assays`` (FE_PCT, SIO2_PCT, AL2O3_PCT, P_PCT, MN_PCT, LOI_PCT, DENSITY) and
        ``lithology`` (LITH) tables.
        ``blastholes``: PointSet of holes at mid-bench, with Z_TOP, Z_BOTTOM, FE_PCT, SIO2_PCT.
        ``block_model``: masked BlockModel of cells of 25 x 25 x 12 m with LITH.
        ``high_grade``, ``iron_formation``: Mesh solids.
        ``topography``: BlockModel of 121 x 121 cells of 25 m with Z.
    """
    box = ((43400.0, 6400.0, 280.0), (46600.0, 9600.0, 740.0))
    rng = stream(seed, 0)
    center = np.array([45090.0, 8000.0])
    radii = np.array([920.0, 830.0])
    rel = lambda xy: ((xy - center) / radii) ** 2
    relief = Variogram([("gaussian", 1.0, 500.0)])

    def height(xy):
        r = np.sqrt(rel(xy).sum(1))
        plateau = 0.5 * (1 - np.tanh((r - 1.12) / 0.16))
        return (
            520.0
            + 185.0 * plateau
            + 3.0 * field(xy, relief, box, seed, 1) * (0.5 + plateau)
            + 5.0 * plateau * np.sin(xy[:, 0] / 210.0)
        )

    topography, ground = gridded((43487.5, 6487.5), 25.0, (121, 121), height)

    def bottom(xy):
        return (
            470.0
            + 45.0 * np.sin(xy[:, 0] / 320.0 + 0.8) * np.cos(xy[:, 1] / 390.0)
            - 0.04 * (xy[:, 0] - 45090.0)
        )

    def if_top(xy):
        return ground(xy) - 1.0

    def if_half(xy):
        r2 = np.minimum(rel(xy).sum(1), 1.0)
        return (if_top(xy) - bottom(xy)) / 2 * np.sqrt(1 - r2)

    def if_mid(xy):
        return if_top(xy) - if_half(xy)

    def hg_half(xy):
        r2 = np.minimum(rel(xy).sum(1) / 0.97, 1.0)
        return (16.0 + 14.0 * np.sin(xy[:, 0] / 170.0) ** 2) * np.sqrt(1 - r2)

    def hg_mid(xy):
        return if_top(xy) - 2.0 - hg_half(xy)

    hematite = Variogram([("spherical", 1.0, 95.0)], rotation=(130.0, 0.0, 0.0), ratios=(0.7, 0.2))
    grade = Variogram([("spherical", 0.8, 120.0)], nugget=0.2, rotation=(130.0, 0.0, 0.0), ratios=(0.7, 0.25))

    def geology(xyz):
        xy = xyz[:, :2]
        half = if_half(xy)
        mid = if_mid(xy)
        inside = (np.abs(xyz[:, 2] - mid) < half) & (rel(xy).sum(1) < 1)
        depth = if_top(xy) - xyz[:, 2]
        zh = field(xyz, hematite, box, seed, 2)
        hg = (np.abs(xyz[:, 2] - hg_mid(xy)) < hg_half(xy)) & inside
        out = np.full(len(xyz), "MAF", dtype=object)
        out[ground(xy) - xyz[:, 2] < 5.0 + 3.0 * np.sin(xy[:, 1] / 90.0)] = "LAT"
        friable = depth < 95.0 + 30.0 * zh
        out[inside] = "IC"
        out[inside & friable] = "IF"
        out[
            inside
            & (field(xyz, Variogram([("spherical", 1.0, 120.0)], ratios=(0.8, 0.3)), box, seed, 9) > 0.55)
        ] = "IC"
        out[inside & (zh > 0.95)] = "HC"
        out[hg] = "HF"
        out[hg & (zh < -0.55)] = "IF"
        out[hg & (zh < -1.25)] = "IC"
        out[hg & (zh > 1.7)] = "HC"
        out[inside & (depth < 2.0 + 5.0 * (np.sin(xy[:, 0] / 75.0) > 0.3)) & (xyz[:, 2] > 640)] = "CG"
        return out

    plan = []
    angle = math.radians(40.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    k = 0
    for a in np.arange(-1300.0, 1301.0, 100.0):
        for b in np.arange(-1300.0, 1301.0, 100.0):
            p = center + rotation @ np.array([a, b]) + rng.normal(0, 3.0, 2)
            if rel(p[None])[0].sum() < 0.98 and k < 187:
                k += 1
                vertical = rng.random() < 0.75
                z = ground(p[None])[0]
                length = np.clip((z - bottom(p[None])[0]) * rng.uniform(0.72, 1.06), 60.0, 294.0)
                plan.append(
                    (
                        f"FD{k:04d}",
                        p[0],
                        p[1],
                        z,
                        float(length),
                        90.0 if vertical else 60.0,
                        220.0 if rng.random() < 0.5 else 40.0,
                        "DD",
                    )
                )
    collars, surveys, paths = _drill(plan, rng, 50.0, 0.02, 0.15)
    lithology = _log(paths, geology, 0.25)

    def two_lengths(lithology):
        ore = {"HF", "HC", "CG", "IF"}
        rows = []
        for hole, a, b, k in lithology:
            rows += _intervals([(hole, a, b, k)], 2.0 if k in ore else 4.0)
        return rows

    samples = two_lengths(lithology)
    xyz = _midpoints(paths, samples)
    n = len(samples)
    litho = np.array([s[3] for s in samples])
    common = field(xyz, grade, box, seed, 3)
    table = {
        "HF": (66.6, 0.008, 1.42, 0.94, 0.045, 0.246, 1.445, 3.3),
        "HC": (66.3, 0.01, 1.35, 0.6, 0.03, 0.14, 0.8, 4.2),
        "IF": (49.9, 0.12, 25.2, 1.1, 0.034, 0.155, 1.44, 2.9),
        "IC": (38.8, 0.3, 43.0, 0.45, 0.025, 0.09, 0.5, 3.45),
        "CG": (59.0, 0.05, 2.9, 4.8, 0.133, 0.258, 5.78, 2.9),
        "LAT": (34.0, 0.3, 18.7, 16.3, 0.069, 0.129, 10.2, 2.08),
        "MAF": (11.3, 0.7, 45.1, 13.1, 0.039, 0.121, 2.77, 2.8),
    }

    def assay(litho, common):
        n = len(litho)
        med = np.array([table[k] for k in litho])
        noise = rng.standard_normal((7, n))
        al = _lognormal(med[:, 3], 0.46, noise[1])
        p = _lognormal(med[:, 4], 0.55, noise[2])
        mn = _lognormal(med[:, 5], 0.7, noise[3])
        loi = _lognormal(med[:, 6], 0.34, noise[4])
        fe = med[:, 0] * np.exp(med[:, 1] * (0.7 * common + 0.7 * noise[0]))
        hematite = med[:, 2] < 5.0
        fe = np.where(
            hematite, med[:, 0] + 0.08 * noise[0] - 1.05 * (al - med[:, 3]) - 0.95 * (loi - med[:, 6]), fe
        )
        fe = np.clip(fe, 2.1, 68.5)
        oxides = fe * 1.4297 + al + p * 2.2914 + mn * 1.2912 + loi
        room = np.maximum(99.0 - oxides, 0.3)
        silica = np.minimum(_lognormal(med[:, 2], 0.3, noise[5]), room)
        silica = np.where(
            (med[:, 2] > 20.0) & (med[:, 2] < 45.0), room * (0.985 + 0.01 * np.tanh(noise[5])), silica
        )
        close = np.minimum(99.5 / (oxides + silica), 1.0)
        return (
            fe * close,
            silica * close,
            al * close,
            p * close,
            mn * close,
            loi * close,
            med[:, 7] * (1 + 0.04 * noise[6]),
        )

    fe, si, al, p, mn, loi, density = assay(litho, common)
    density = np.where(rng.random(n) < 0.3, np.round(density, 2), np.nan)
    rows = [
        (
            s[0],
            s[1],
            s[2],
            round(float(a), 2),
            round(float(b), 2),
            round(float(c), 2),
            round(float(d), 3),
            round(float(e), 3),
            round(float(g), 2),
            h,
        )
        for s, a, b, c, d, e, g, h in zip(samples, fe, si, al, p, mn, loi, density)
    ]
    names = (
        "HOLE_ID",
        "FROM",
        "TO",
        "FE_PCT",
        "SIO2_PCT",
        "AL2O3_PCT",
        "P_PCT",
        "MN_PCT",
        "LOI_PCT",
        "DENSITY",
    )

    pit = np.array([44745.0, 8240.0])
    gy, gx = np.mgrid[-180.0:181.0:10.0, -180.0:181.0:10.0]
    pattern = np.column_stack([gx.ravel(), gy.ravel()])
    pattern = pattern[(pattern[:, 0] / 148.0) ** 2 + (pattern[:, 1] / 142.0) ** 2 < 1.0][:651] + pit
    xy = np.tile(pattern, (3, 1))
    tops = np.repeat([696.0, 684.0, 672.0], len(pattern))
    offsets = np.array([-4.8, -2.4, 0.0, 2.4, 4.8])
    stacked = np.vstack([np.column_stack([xy, tops - 6.0 + o]) for o in offsets])
    bench_lith = geology(stacked)
    bench_common = field(stacked, grade, box, seed, 3)
    bfe, bsi, *_ = assay(bench_lith, bench_common)
    m = len(xy)
    bfe = bfe.reshape(5, m).mean(0)
    bsi = bsi.reshape(5, m).mean(0)
    bfe = np.clip(bfe + rng.normal(0, 1.0, m), 14.0, 68.9)
    bsi = np.maximum(bsi + rng.normal(0, 1.5, m), 0.1)
    blastholes = PointSet(
        np.column_stack([np.round(xy, 1), tops - 6.0]),
        {
            "ID": np.arange(1.0, m + 1),
            "Z_TOP": tops,
            "Z_BOTTOM": tops - 12.0,
            "FE_PCT": np.round(bfe, 2),
            "SIO2_PCT": np.round(bsi, 2),
        },
    )

    nx, ny, nz = 89, 81, 33
    k_, j_, i_ = np.mgrid[0:nz, 0:ny, 0:nx]
    centers = np.column_stack(
        [
            43925.0 + 25.0 * (i_.ravel() + 0.5),
            7050.0 + 25.0 * (j_.ravel() + 0.5),
            300.0 + 12.0 * (k_.ravel() + 0.5),
        ]
    )
    below = (centers[:, 2] < ground(centers[:, :2])) & (rel(centers[:, :2]).sum(1) < 1.0)
    index = np.flatnonzero(below)
    blocks = BlockModel(
        (43925.0, 7050.0, 300.0),
        (25.0, 25.0, 12.0),
        (nx, ny, nz),
        attributes={"LITH": geology(centers[below])},
        index=index.astype(np.uint64),
    )
    plan_world = lambda u, v, w: np.stack([center[0] + u, center[1] + v, w], -1)
    column = lambda fn: (
        lambda u, v: fn(np.column_stack([center[0] + u.ravel(), center[1] + v.ravel()])).reshape(u.shape)
    )
    solid = lambda mid, half: _disc_plate(
        column(mid), column(half), radii[0] * 0.999, radii[1] * 0.999, 110, plan_world
    )
    return {
        "collars": _table(collars, HOLE_COLUMNS),
        "surveys": _table(surveys, ("HOLE_ID", "DEPTH", "DIP", "AZIMUTH")),
        "assays": _table(rows, names),
        "lithology": _table(lithology, ("HOLE_ID", "FROM", "TO", "LITH")),
        "blastholes": blastholes,
        "block_model": blocks,
        "high_grade": solid(hg_mid, hg_half),
        "iron_formation": solid(if_mid, if_half),
        "topography": _surface((43487.5, 6487.5), 25.0, (121, 121), topography),
    }


VEINS = {
    "V1": ((7983.0, 15012.0, 856.0), 415.0, 490.0, 1.32, 0.3, 85.0),
    "V2": ((7959.0, 14998.0, 845.0), 258.0, 350.0, 0.64, 1.2, 85.0),
    "V3": ((8040.0, 15027.0, 872.0), 294.0, 380.0, 0.65, 2.4, 85.0),
    "V4": ((8012.0, 14980.0, 861.0), 175.0, 262.0, 0.49, 3.5, 58.0),
}


def _vein_half(u, v, a, b, peak, phase):
    envelope = np.sqrt(np.maximum(1 - (u / a) ** 2, 0.0)) * np.sqrt(np.maximum(1 - (v / b) ** 2, 0.0))
    return peak * envelope * (1 + 0.45 * np.sin(u / 37.0 + phase) * np.cos(v / 29.0 + 2 * phase))


def _vein_mid(u, v, phase):
    return 2.6 * np.sin(u / 62.0 + phase) + 1.6 * np.cos(v / 45.0 - phase)


def vein_gold_grade_control(seed: int = 0) -> dict:
    """Four steep gold veins with 118 diamond holes and about 2500 channels.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``collars``, ``surveys``, ``assays`` (AU_GPT, AG_GPT, AS_PPM) and ``lithology`` (LITH, VEIN) tables; ``vein_V1``
        to ``vein_V4`` Mesh solids; ``topography``, a BlockModel of 71 x 71 cells of 20 m with Z.
    """
    box = ((7200.0, 14200.0, 340.0), (8800.0, 15800.0, 960.0))
    f = frame(85.0, 70.0)
    rng = stream(seed, 0)
    relief = Variogram([("gaussian", 1.0, 520.0)])
    topography, ground = gridded(
        (7290.0, 14290.0), 20.0, (71, 71), lambda xy: 900.0 + 15.5 * field(xy, relief, box, seed, 1)
    )
    cover = lambda xy: 866.0 + 5.0 * np.sin(xy[:, 0] / 190.0) * np.cos(xy[:, 1] / 240.0)
    shoots = Variogram(
        [("spherical", 0.8, 58.0)], nugget=0.2, rotation=(355.0, 0.0, 70.0), ratios=(0.55, 0.12)
    )

    def geology(xyz):
        lith = np.full(len(xyz), "AND", dtype=object)
        vein = np.empty(len(xyz), dtype=object)
        for name, (c, a, b, peak, phase, azimuth) in VEINS.items():
            d = (xyz - np.asarray(c)) @ frame(azimuth, 70.0).T
            u, v, w = d[:, 0], d[:, 1], d[:, 2] - _vein_mid(d[:, 0], d[:, 1], phase)
            ok = (np.abs(u) < a) & (v > -2.0) & (v < b)
            h = _vein_half(u, np.maximum(v, 0.0), a, b, peak, phase)
            halo = ok & (np.abs(w) < h + 0.5 + 0.4 * peak)
            lith[halo] = "BX"
            vein[halo] = name
            core = ok & (np.abs(w) < h)
            lith[core] = "QV"
            vein[core] = name
        above = xyz[:, 2] > cover(xyz[:, :2])
        lith[above] = "RHY"
        vein[above] = None
        return lith, vein

    plan = []
    for k, y in enumerate(np.linspace(14625.0, 15490.0, 118)):
        x = 8200.0 + rng.normal(0, 22.0) + 18.0 * ((k % 3) - 1)
        z = ground(np.array([[x, y]]))[0]
        plan.append(
            (
                f"UD{k + 1:04d}",
                x,
                y,
                z,
                rng.uniform(300.0, 680.0) if k % 2 else rng.uniform(220.0, 400.0),
                50.0 + 8.0 * ((k % 4) / 3),
                260.5 + rng.normal(0, 1.5),
                "DD",
            )
        )
    number = 0
    for name in ("V1", "V2"):
        c, a, b, peak, phase, _ = VEINS[name]
        for level in np.arange(700.0, 841.0, 20.0):
            for y in np.arange(c[1] - a * 0.72, c[1] + a * 0.72, 3.0):
                guess = np.array([c[0], y, level])
                for _ in range(3):
                    d = (guess - np.asarray(c)) @ f.T
                    guess[0] -= (d[2] - _vein_mid(d[0], d[1], phase)) / f[2][0]
                d = (guess - np.asarray(c)) @ f.T
                half = _vein_half(d[0], max(d[1], 0.0), a, b, peak, phase)
                if half > 0.3 and number < 2492:
                    number += 1
                    wall = rng.uniform(0.9, 1.7)
                    length = 2 * half / abs(f[2][0]) + 2 * wall
                    plan.append(
                        (
                            f"CH{number:05d}",
                            guess[0] + length / 2,
                            y,
                            level,
                            length,
                            0.0,
                            260.5 + rng.normal(0, 2.5),
                            "CH",
                        )
                    )
    collars, surveys, paths = _drill(plan, rng, 45.0, 0.5, 0.25)
    surveys = [(h, d, 0.0, a) if h.startswith("CH") else (h, d, dp, a) for h, d, dp, a in surveys]

    def code(xyz):
        lith, vein = geology(xyz)
        return np.array([f"{a}|{b or ''}" for a, b in zip(lith, vein)], dtype=object)

    coded = _log({h: p for h, p in paths.items() if h.startswith("CH")}, code, 0.05)
    coded += _log({h: p for h, p in paths.items() if not h.startswith("CH")}, code, 0.5)
    lithology = [(h, a, b, k.split("|")[0], k.split("|")[1] or None) for h, a, b, k in coded]

    samples = []
    for h, a, b, k in coded:
        lith = k.split("|")[0]
        channel = h.startswith("CH")
        samples += _intervals([(h, a, b, lith)], 0.5 if channel else 2.0)
    samples = [s for s in samples if s[3] != "RHY" or rng.random() < 0.05]
    xyz = _midpoints(paths, samples)
    n = len(samples)
    litho = np.array([s[3] for s in samples])
    z = field(xyz, shoots, box, seed, 2)
    channel = np.array([s[0].startswith("CH") for s in samples])
    noise = rng.standard_normal((3, n))
    table = {"QV": (2.56, 1.75), "BX": (0.327, 1.4), "AND": (0.012, 1.05), "RHY": (0.008, 0.83)}
    ag_table = {"QV": (23.3, 1.94), "BX": (3.35, 1.61), "AND": (0.58, 0.99), "RHY": (0.52, 0.82)}
    as_table = {"QV": (279.0, 0.79), "BX": (286.0, 0.82), "AND": (28.0, 0.84), "RHY": (28.0, 0.84)}
    pick = lambda t, c: np.array([t[k][c] for k in litho])
    zau = 0.72 * z + 0.69 * noise[0]
    au = np.clip(pick(table, 0) * np.exp(pick(table, 1) * zau), 0.005, 521.0)
    quartz = np.flatnonzero(litho == "QV")
    drilled = quartz[~channel[quartz]]
    cut = quartz[channel[quartz]]
    gap = np.min(
        np.linalg.norm(xyz[cut, None] - xyz[drilled[:: max(len(drilled) // 400, 1)]], axis=2), axis=1
    )
    hole = samples[cut[np.argmin(gap)]][0]
    au[[i for i in cut if samples[i][0] == hole]] = 1192.0
    ag = np.clip(
        pick(ag_table, 0) * np.exp(pick(ag_table, 1) * (0.75 * zau + 0.66 * noise[1])), 0.02, 164000.0
    )
    as_ = np.maximum(pick(as_table, 0) * np.exp(pick(as_table, 1) * noise[2]), 1.0)
    as_ = np.where(channel, np.nan, np.round(as_))
    rows = [
        (s[0], s[1], s[2], round(float(a), 3), round(float(b), 2), c)
        for s, a, b, c in zip(samples, au, ag, as_)
    ]
    out = {
        "collars": _table(collars, HOLE_COLUMNS),
        "surveys": _table(surveys, ("HOLE_ID", "DEPTH", "DIP", "AZIMUTH")),
        "assays": _table(rows, ("HOLE_ID", "FROM", "TO", "AU_GPT", "AG_GPT", "AS_PPM")),
        "lithology": _table(lithology, ("HOLE_ID", "FROM", "TO", "LITH", "VEIN")),
        "topography": _surface((7290.0, 14290.0), 20.0, (71, 71), topography),
    }
    for name, (c, a, b, peak, phase, azimuth) in VEINS.items():
        out[f"vein_{name}"] = plate(
            lambda u, v, phase=phase: _vein_mid(u, v, phase),
            lambda u, v, a=a, b=b, peak=peak, phase=phase: np.maximum(
                _vein_half(u, v, a, b, peak, phase), 0.03
            ),
            np.linspace(-a, a, 121),
            np.linspace(0.0, b, 81),
            framed(frame(azimuth, 70.0), c),
        )
    return out


def phosphate_weathering_profile(seed: int = 0) -> dict:
    """Phosphate in weathering horizons under rolling topography.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``collars``, ``surveys``, ``assays`` (P2O5_PCT, P2O5_AP_PCT, CAO_PCT, SIO2_PCT, FE2O3_PCT, AL2O3_PCT, MGO_PCT) and
        ``horizons`` (HORIZON) tables; ``topography``, a BlockModel of 241 x 191 cells of 10 m with Z.
    """
    box = ((59700.0, 21950.0), (62300.0, 24050.0))
    rng = stream(seed, 0)
    hills = Variogram([("gaussian", 1.0, 850.0)])
    swell = Variogram([("spherical", 1.0, 220.0)])
    relief = lambda xy: field(xy, hills, box, seed, 1)
    topography, ground = gridded((59795.0, 22045.0), 10.0, (241, 191), lambda xy: 1144.0 + 37.0 * relief(xy))

    def surfaces(xy):
        h = relief(xy)
        s = field(xy, swell, box, seed, 2)
        soil = 2.2 + 1.2 * np.abs(s)
        alu = soil + np.maximum(19.0 + 5.0 * h + 4.0 * s, 4.0)
        oxi = alu + np.maximum(25.0 + 6.0 * h + 3.0 * field(xy, swell, box, seed, 3), 6.0)
        sap = oxi + np.maximum(21.0 + 3.0 * h + 3.0 * field(xy, swell, box, seed, 4), 5.0)
        return soil, alu, oxi, sap

    gy, gx = np.mgrid[22250.0:23800.0:100.0, 60000.0:62050.0:100.0]
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    keep = ((nodes[:, 0] - 61000.0) / 1085.0) ** 2 + ((nodes[:, 1] - 23000.0) / 830.0) ** 2 < 1.0
    nodes = (nodes[keep][:251] + rng.normal(0, 4.0, (min(keep.sum(), 251), 2))).round(1)
    surf = surfaces(nodes)
    plan = []
    for k, (x, y) in enumerate(nodes):
        z = ground(np.array([[x, y]]))[0]
        inclined = False
        kind = "DD" if rng.random() < 0.135 else "RC"
        length = surf[3][k] + rng.uniform(17.0, 34.0)
        length = length / math.sin(math.radians(60.0)) if inclined else length
        plan.append((f"PF{k + 1:04d}", x, y, z, length, 60.0 if inclined else 90.0, 180.0, kind))
    collars, surveys, paths = _drill(plan, rng, 25.0, 0.02, 0.15)
    horizon_rows = []
    names = ("SOIL", "ALU", "OXI", "SAP", "ROCK")
    for k, (hole, *_rest, length, dip, _a, _t) in enumerate(plan):
        scale = 1 / math.sin(math.radians(dip))
        ends = [surf[i][k] * scale for i in range(4)] + [length]
        starts = [0.0, *ends[:-1]]
        horizon_rows += [(hole, round(a, 2), round(b, 2), n) for a, b, n in zip(starts, ends, names)]
    samples = _intervals(horizon_rows, 5.0)
    xyz = _midpoints(paths, samples)
    n = len(samples)
    hz = np.array([s[3] for s in samples])
    grade = Variogram([("spherical", 0.75, 280.0)], nugget=0.25, ratios=(1.0, 0.15))
    common = field(xyz, grade, ((59700.0, 21950.0, 900.0), (62300.0, 24050.0, 1300.0)), seed, 5)
    noise = rng.standard_normal((7, n))
    table = {
        "SOIL": (1.81, 0.19, 0.27, 28.3, 28.4, 19.1, 0.43),
        "ALU": (4.6, 0.33, 0.46, 14.5, 34.6, 19.2, 0.44),
        "OXI": (12.1, 7.8, 11.0, 11.7, 28.6, 4.39, 1.87),
        "SAP": (8.55, 7.77, 14.7, 17.4, 19.2, 2.73, 7.22),
        "ROCK": (4.7, 4.47, 18.5, 18.9, 11.8, 1.8, 10.7),
    }
    med = np.array([table[k] for k in hz])
    p = _lognormal(med[:, 0], 0.33, 0.8 * common + 0.6 * noise[0])
    cao = _lognormal(med[:, 2], 0.3, 0.6 * common + 0.8 * noise[1])
    ap = np.minimum(np.minimum(_lognormal(med[:, 1], 0.33, 0.8 * common + 0.6 * noise[0]), cao / 1.318), p)
    si = _lognormal(med[:, 3], 0.28, noise[2])
    fe = _lognormal(med[:, 4], 0.23, noise[3])
    al = _lognormal(med[:, 5], 0.46, noise[4])
    mg = _lognormal(med[:, 6], 0.47, noise[5])
    total = p + cao + si + fe + al + np.where(np.isfinite(mg), mg, 0.0)
    shrink = np.minimum(94.5 / total, 1.0)
    p, ap, cao, si, fe, al, mg = (v * shrink for v in (p, ap, cao, si, fe, al, mg))
    mg = np.where(rng.random(n) < 0.2, np.nan, np.round(mg, 2))
    rows = [
        (
            s[0],
            s[1],
            s[2],
            round(float(a), 2),
            round(float(b), 2),
            round(float(c), 2),
            round(float(d), 2),
            round(float(e), 2),
            round(float(g), 2),
            h,
        )
        for s, a, b, c, d, e, g, h in zip(samples, np.maximum(p, 0.5), ap, cao, si, fe, al, mg)
    ]
    return {
        "collars": _table(collars, HOLE_COLUMNS),
        "surveys": _table(surveys, ("HOLE_ID", "DEPTH", "DIP", "AZIMUTH")),
        "assays": _table(
            rows,
            (
                "HOLE_ID",
                "FROM",
                "TO",
                "P2O5_PCT",
                "P2O5_AP_PCT",
                "CAO_PCT",
                "SIO2_PCT",
                "FE2O3_PCT",
                "AL2O3_PCT",
                "MGO_PCT",
            ),
        ),
        "horizons": _table(horizon_rows, ("HOLE_ID", "FROM", "TO", "HORIZON")),
        "topography": _surface((59795.0, 22045.0), 10.0, (241, 191), topography),
    }


def nickel_laterite_profile(seed: int = 0) -> dict:
    """A nickel laterite drilled by 448 vertical holes, 50 m mesh with 25 m infill.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``collars``, ``surveys``, 1 m ``assays`` (NI_PCT, CO_PCT, FE_PCT, MGO_PCT, SIO2_PCT, AL2O3_PCT) and ``horizons``
        (HORIZON) tables.
    """
    box = ((29900.0, 59900.0), (31100.0, 60800.0))
    rng = stream(seed, 0)
    gy, gx = np.mgrid[60000.0:60701.0:50.0, 30000.0:31001.0:50.0]
    mesh = np.column_stack([gx.ravel(), gy.ravel()])
    iy, ix = np.mgrid[60200.0:60501.0:25.0, 30200.0:30801.0:25.0]
    infill = np.column_stack([ix.ravel(), iy.ravel()])
    infill = infill[~((infill[:, 0] % 50 == 0) & (infill[:, 1] % 50 == 0))]
    nodes = np.vstack([mesh, infill])[:448] + rng.normal(0, 1.5, (448, 2))
    nodes = nodes.round(1)
    relief = Variogram([("gaussian", 1.0, 400.0)])
    bedrock = Variogram([("spherical", 0.7, 45.0), ("spherical", 0.3, 180.0)])
    zc = 359.0 + 12.5 * field(nodes, relief, box, seed, 1)
    z_brk = field(nodes, bedrock, box, seed, 2)
    z_lim = field(nodes, Variogram([("spherical", 1.0, 150.0)]), box, seed, 3)
    ferr = np.maximum(2.4 + 0.9 * rng.standard_normal(448), 0.1)
    has_ferr = rng.random(448) < 0.955
    lim = np.maximum(7.2 + 2.2 * z_lim + 0.8 * rng.standard_normal(448), 1.0)
    pits = field(nodes, Variogram([("spherical", 1.0, 32.0)]), box, seed, 5)
    sap = np.maximum(10.6 + 2.8 * z_brk + 26.0 * np.maximum(pits - 1.6, 0.0), 2.6)
    tail = rng.uniform(2.0, 5.0, 448)
    plan, horizons = [], []
    for k, (x, y) in enumerate(nodes):
        hole = f"NL{k + 1:04d}"
        a = ferr[k] if has_ferr[k] else 0.0
        b = a + lim[k]
        c = b + sap[k]
        length = float(np.ceil(c + tail[k]))
        plan.append((hole, x, y, zc[k], length, 90.0, 0.0, "RC"))
        if has_ferr[k]:
            horizons.append((hole, 0.0, round(a, 2), "FERR"))
        horizons += [
            (hole, round(a, 2), round(b, 2), "LIM"),
            (hole, round(b, 2), round(c, 2), "SAP"),
            (hole, round(c, 2), length, "BRK"),
        ]
    collars, surveys, paths = _drill(plan, rng, 1e6)
    samples = _intervals(horizons, 1.0)
    xyz = _midpoints(paths, samples)
    n = len(samples)
    hz = np.array([s[3] for s in samples])
    limits = {r[0]: {} for r in horizons}
    for hole, a, b, name in horizons:
        limits[hole][name] = (a, b)
    relative = np.array(
        [
            (0.5 * (s[1] + s[2]) - limits[s[0]][s[3]][0])
            / max(limits[s[0]][s[3]][1] - limits[s[0]][s[3]][0], 0.1)
            for s in samples
        ]
    )
    grade = Variogram([("spherical", 0.8, 110.0)], nugget=0.2, ratios=(1.0, 0.07))
    common = field(xyz, grade, ((29900.0, 59900.0, 280.0), (31100.0, 60800.0, 400.0)), seed, 4)
    noise = rng.standard_normal((6, n))
    table = {
        "FERR": (0.327, 0.021, 44.2, 0.44, 4.71, 8.11),
        "LIM": (0.992, 0.09, 41.2, 1.77, 9.66, 5.08),
        "SAP": (1.5, 0.032, 14.5, 16.9, 36.1, 1.65),
        "BRK": (0.264, 0.012, 5.58, 33.2, 36.1, 0.79),
    }
    med = np.array([table[k] for k in hz])
    base = np.where(hz == "SAP", 0.5 * (relative - 0.5), 0.0)
    ni = np.clip(_lognormal(med[:, 0], 0.42, 0.92 * common + 0.39 * noise[0]) * np.exp(base), 0.04, 8.7)
    co = np.clip(_lognormal(med[:, 1], 0.72, 0.8 * common + 0.6 * noise[1]), 0.001, 1.45)
    fe = np.clip(_lognormal(med[:, 2], 0.2, noise[2]), 1.9, 58.0)
    mg = np.clip(_lognormal(med[:, 3], 0.42, noise[3]), 0.1, 66.0)
    si = np.clip(_lognormal(med[:, 4], 0.24, noise[4]), 1.9, 63.0)
    al = np.clip(_lognormal(med[:, 5], 0.54, noise[5]), 0.12, 41.0)
    rows = [
        (
            s[0],
            s[1],
            s[2],
            round(float(a), 3),
            round(float(b), 3),
            round(float(c), 2),
            round(float(d), 2),
            round(float(e), 2),
            round(float(g), 2),
        )
        for s, a, b, c, d, e, g in zip(samples, ni, co, fe, mg, si, al)
    ]
    return {
        "collars": _table(collars, HOLE_COLUMNS),
        "surveys": _table(surveys, ("HOLE_ID", "DEPTH", "DIP", "AZIMUTH")),
        "assays": _table(
            rows, ("HOLE_ID", "FROM", "TO", "NI_PCT", "CO_PCT", "FE_PCT", "MGO_PCT", "SIO2_PCT", "AL2O3_PCT")
        ),
        "horizons": _table(horizons, ("HOLE_ID", "FROM", "TO", "HORIZON")),
    }


def tailings_reprocessing(seed: int = 0) -> dict:
    """A tailings storage facility drilled by 107 sonic holes.

    Parameters
    ----------
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        ``collars``, ``surveys``, ``assays`` (AU_GPT, CU_PPM, AS_PPM, CN_WAD_PPM, MOISTURE_PCT; ``CN_WAD_BDL`` flags values
        below detection) and ``lithology`` (LITH) tables; ``surface`` and ``original_ground``, BlockModels of 81 x 56
        cells of 10 m with Z.
    """
    box = ((4900.0, 1900.0), (5800.0, 2550.0))
    rng = stream(seed, 0)
    center = np.array([5350.0, 2225.0])
    radii = np.array([372.0, 246.0])
    valley = Variogram([("gaussian", 1.0, 420.0)])
    natural = lambda xy: 503.0 + 5.5 * field(xy, valley, box, seed, 1) + 0.017 * (xy[:, 1] - 2225.0)
    radius = lambda xy: np.sqrt((((xy - center) / radii) ** 2).sum(1))

    def surface(xy):
        r = radius(xy)
        pond = 518.9 + 1.0 * np.tanh((xy[:, 1] - 2225.0) / 110.0)
        crest = 8.0 * np.exp(-(((r - 1.08) / 0.05) ** 2))
        outside = 0.5 * (1 + np.tanh((r - 1.12) / 0.015))
        return (1 - outside) * (pond + crest) + outside * (natural(xy) + 0.0)

    surface_values, _ = gridded((4945.0, 1945.0), 10.0, (81, 56), surface)
    original_values, _ = gridded((4945.0, 1945.0), 10.0, (81, 56), natural)
    points = []
    while len(points) < 107:
        p = rng.uniform((5010.0, 2010.0), (5690.0, 2440.0))
        near = not points or np.hypot(*(np.array(points) - p).T).min() > 32.0
        if radius(p[None])[0] < 0.97 and (p[1] > 2140.0 or rng.random() < 0.42) and near:
            points.append(p)
    xy = np.round(points, 1)
    tops = surface(xy)
    bases = natural(xy)
    plan, lithology = [], []
    for k, (x, y) in enumerate(xy):
        hole = f"TS{k + 1:03d}"
        tailings = max(tops[k] - bases[k], 3.5)
        natural_part = rng.uniform(1.0, 2.5)
        length = round(tailings + natural_part, 1)
        plan.append((hole, x, y, tops[k], length, 90.0, 0.0, "SONIC"))
        depth = 0.0
        if rng.random() < 0.2:
            depth = round(rng.uniform(0.4, 1.8), 2)
            lithology.append((hole, 0.0, depth, "CAP"))
        slime = np.clip(0.6 - (y - 2225.0) / 420.0, 0.1, 0.95)
        current = "SLIME" if rng.random() < slime else "SAND"
        while depth < tailings - 0.05:
            step = min(max(rng.exponential(0.27 if current == "SAND" else 0.15), 0.05), tailings - depth)
            lithology.append((hole, round(depth, 2), round(depth + step, 2), current))
            depth = round(depth + step, 2)
            current = "SLIME" if rng.random() < slime * 0.8 + 0.1 * (current == "SAND") else "SAND"
        lithology.append((hole, round(depth, 2), length, "CLAY"))
    lithology = _merge(lithology)
    collars, surveys, paths = _drill(plan, rng, 1e6)
    samples = _intervals(lithology, 1.5)
    xyz = _midpoints(paths, samples)
    n = len(samples)
    litho = np.array([s[3] for s in samples])
    depth = np.array([(s[1] + s[2]) / 2 for s in samples])
    common = field(xyz[:, :2], Variogram([("spherical", 0.7, 180.0)], nugget=0.3), box, seed, 2)
    noise = rng.standard_normal((5, n))
    aged = np.exp(0.035 * (depth - 8.8))
    tail = np.isin(litho, ["SAND", "SLIME"])
    au = np.where(
        tail, _lognormal(0.4, 0.5, 0.6 * common + 0.8 * noise[0]) * aged, _lognormal(0.01, 0.48, noise[0])
    )
    cu = np.where(tail, _lognormal(665.0, 0.38, noise[1]), _lognormal(36.0, 0.37, noise[1]))
    as_ = np.select(
        [litho == "SLIME", litho == "SAND", litho == "CAP"],
        [
            _lognormal(602.0, 0.55, noise[2]),
            _lognormal(273.0, 0.56, noise[2]),
            _lognormal(237.0, 0.49, noise[2]),
        ],
        _lognormal(14.0, 0.53, noise[2]),
    )
    cn_true = np.select(
        [litho == "SLIME", litho == "SAND", litho == "CAP"],
        [_lognormal(1.08, 0.95, noise[3]), _lognormal(0.46, 0.95, noise[3]), _lognormal(0.28, 0.6, noise[3])],
        _lognormal(0.15, 0.4, noise[3]),
    )
    moisture = np.select(
        [litho == "SLIME", litho == "SAND", litho == "CAP"],
        [_lognormal(32.0, 0.1, noise[4]), _lognormal(14.0, 0.22, noise[4]), _lognormal(13.8, 0.2, noise[4])],
        _lognormal(18.2, 0.19, noise[4]),
    )
    capped = litho == "CAP"
    rows = [
        (
            s[0],
            s[1],
            s[2],
            np.nan if c else round(float(a), 3),
            np.nan if c else round(float(b)),
            round(float(d)),
            round(float(max(e, 0.5)), 2),
            round(float(np.clip(f, 5.0, 43.0)), 1),
            float(e < 0.5),
        )
        for s, a, b, d, e, f, c in zip(samples, au, cu, as_, cn_true, moisture, capped)
    ]
    return {
        "collars": _table(collars, HOLE_COLUMNS),
        "surveys": _table(surveys, ("HOLE_ID", "DEPTH", "DIP", "AZIMUTH")),
        "assays": _table(
            rows,
            (
                "HOLE_ID",
                "FROM",
                "TO",
                "AU_GPT",
                "CU_PPM",
                "AS_PPM",
                "CN_WAD_PPM",
                "MOISTURE_PCT",
                "CN_WAD_BDL",
            ),
        ),
        "lithology": _table(lithology, ("HOLE_ID", "FROM", "TO", "LITH")),
        "surface": _surface((4945.0, 1945.0), 10.0, (81, 56), surface_values),
        "original_ground": _surface((4945.0, 1945.0), 10.0, (81, 56), original_values),
    }


def _merge(rows):
    out = []
    for hole, a, b, k in rows:
        if b <= a:
            continue
        if out and out[-1][0] == hole and out[-1][3] == k and out[-1][2] == a:
            out[-1] = (hole, out[-1][1], b, k)
        else:
            out.append((hole, a, b, k))
    return out


MINERALS = ("clays", "chalcocite", "bornite", "chalcopyrite", "tennantite", "molibdenite", "pyrite")
MINERAL_CORRELATION = np.array(
    [
        [1.0, -0.5, 0.1, 0.0, -0.1, 0.0, -0.3],
        [-0.5, 1.0, 0.1, -0.5, 0.1, -0.1, 0.2],
        [0.1, 0.1, 1.0, 0.2, 0.35, 0.65, -0.75],
        [0.0, -0.5, 0.2, 1.0, 0.25, 0.5, -0.25],
        [-0.1, 0.1, 0.35, 0.25, 1.0, 0.4, -0.15],
        [0.0, -0.1, 0.65, 0.5, 0.4, 1.0, -0.55],
        [-0.3, 0.2, -0.75, -0.25, -0.15, -0.55, 1.0],
    ]
)
MINERAL_SCALE = {
    "clays": (3.4, 0.75, 43.0),
    "chalcocite": (0.05, 1.6, 2.9),
    "bornite": (0.075, 1.2, 4.0),
    "chalcopyrite": (0.8, 0.72, 7.5),
    "tennantite": (0.0065, 1.5, 0.67),
    "molibdenite": (0.012, 0.85, 0.48),
    "pyrite": (1.1, 1.05, 24.0),
}
ZONE_SHIFT = {
    1: {"clays": 0.35, "pyrite": -1.1, "chalcopyrite": -0.9, "bornite": -0.9, "chalcocite": -0.4},
    2: {"chalcocite": 2.2, "clays": 0.2, "chalcopyrite": -0.2},
    3: {"chalcopyrite": 0.45, "bornite": 1.5, "molibdenite": 0.6, "pyrite": -0.6},
    4: {"chalcopyrite": 0.25, "pyrite": 0.5, "molibdenite": 0.15},
    5: {"chalcopyrite": -0.6, "bornite": -0.9, "pyrite": 0.15, "molibdenite": -0.3, "clays": -0.1},
}


def _stripes(values, shrink):
    """Assays below the detection limits of the laboratories read as the limits, stripes on a log plot."""
    if shrink < 1.0:
        return values
    return np.where(
        values < 0.0005, 0.0005, np.where(values < 0.0008, 0.0008, np.where(values < 0.0012, 0.0012, values))
    )


def _mineral_fields(xyz, box, seed, zones, shrink=1.0):
    cholesky = np.linalg.cholesky(MINERAL_CORRELATION + 1e-3 * np.eye(7))
    structure = Variogram(
        [("spherical", 0.9, 300.0)], nugget=0.1, rotation=(0.0, 90.0, 0.0), ratios=(0.45, 0.45)
    )
    z = np.column_stack([field(xyz, structure, box, seed, 10 + k) for k in range(7)]) @ cholesky.T
    out = {}
    for i, name in enumerate(MINERALS):
        median, sigma, cap = MINERAL_SCALE[name]
        shift = np.array([ZONE_SHIFT[int(c)].get(name, 0.0) for c in zones])
        out[name] = np.minimum(median * np.exp(shrink * sigma * z[:, i] + shift), cap)
    arm = field(xyz, Variogram([("spherical", 1.0, 200.0)]), box, seed, 25)
    barren = field(xyz, Variogram([("spherical", 1.0, 160.0)]), box, seed, 26) > -0.35
    out["chalcocite"] = _stripes(
        np.where((arm < -0.18) | barren, out["chalcocite"] * 0.03, out["chalcocite"] * 1.5), shrink
    )
    out["tennantite"] = _stripes(
        np.where((arm > 0.18) | barren, out["tennantite"] * 0.03, out["tennantite"] * 3.0), shrink
    )
    return out


def _porphyry_geology(seed, deposit):
    box = ((-620.0, -900.0, 1650.0), (560.0, 760.0, 2720.0))
    relief = Variogram([("gaussian", 1.0, 520.0)])
    top = 2598.0 if deposit == 1 else 2578.0
    _, ground = gridded(
        (-625.0, -905.0),
        10.0,
        (120, 170),
        lambda xy: top + 12.0 * field(xy, relief, box, seed + deposit, 1) + 0.008 * xy[:, 1],
    )
    shift = (10.0 * deposit, -30.0 * deposit)

    def zones(xyz):
        depth = ground(xyz[:, :2]) - xyz[:, 2]
        r = np.hypot((xyz[:, 0] - shift[0]) / 330.0, (xyz[:, 1] - shift[1] + 80.0) / 560.0)
        out = np.full(len(xyz), 5.0)
        rock = r + np.maximum(depth - 380.0, 0.0) / 620.0 < 1.05
        out[rock] = 4.0
        out[rock & (r < 0.5) & (depth < 360.0)] = 3.0
        out[depth < 95.0 + 25.0 * np.cos(xyz[:, 0] / 95.0)] = 1.0
        band = (depth >= 95.0 + 25.0 * np.cos(xyz[:, 0] / 95.0)) & (
            depth < 120.0 + 25.0 * np.cos(xyz[:, 0] / 95.0)
        )
        out[band & rock] = 2.0
        return out

    return box, ground, zones


def _bwi(zones, z):
    return np.clip(16.9 + 2.9 * z + 0.6 * (zones == 2) - 0.4 * (zones == 1), 9.0, 29.0)


def _porphyry_holes(seed, deposit, count, step, azimuth_range):
    box, ground, zones = _porphyry_geology(seed, deposit)
    rng = stream(seed + deposit, 0)
    rows = []
    for dh in range(1, count + 1):
        x, y = rng.uniform((-400.0, -600.0), (380.0, 560.0))
        z = ground(np.array([[x, y]]))[0]
        dip = -rng.uniform(62.0, 86.0)
        azimuth = rng.uniform(*azimuth_range)
        depth_total = rng.uniform(60.0, 400.0)
        cosd = math.cos(math.radians(dip))
        for a in np.arange(0.0, depth_total, step):
            mid = a + step / 2
            rows.append(
                (
                    float(dh),
                    x + mid * cosd * math.sin(math.radians(azimuth)),
                    y + mid * cosd * math.cos(math.radians(azimuth)),
                    z + mid * math.sin(math.radians(dip)),
                    a,
                    a + step,
                    azimuth,
                    dip,
                )
            )
    return np.array(rows), box, ground, zones


def porphyry_geometallurgy(deposit: int = 1, seed: int = 0) -> dict:
    """Mineralogy, grindability and recovery of three porphyry deposits.

    Parameters
    ----------
    deposit : {1, 2, 3}
    seed : int
        Fixes the data; the default gives the data the examples use.

    Returns
    -------
    dict
        Deposit 1: ``block_model``, a masked BlockModel of 49 x 71 x 44 cells of 20 m; ``correlations``,
        ``grindability_distribution`` (bwi) and ``mineralogical_distributions`` tables; ``pseudo_drillholes`` and
        ``synthetic_drillholes`` PointSets.
        Deposits 2 and 3: ``drillholes`` PointSet and ``topography``, a BlockModel of 97 x 142 cells of 10 m with midz.

    Raises
    ------
    InvalidInput
        If `deposit` is not 1, 2 or 3.
    """
    if deposit not in (1, 2, 3):
        raise InvalidInput(f"deposit must be 1, 2 or 3, got {deposit!r}")
    rng = stream(seed + deposit, 5)
    if deposit > 1:
        holes, box, ground, zones = _porphyry_holes(
            seed, deposit, 190 if deposit == 2 else 167, 10.0, (-95.0, 110.0)
        )
        xyz = holes[:, 1:4]
        minz = zones(xyz)
        mineral = _mineral_fields(xyz, box, seed + deposit, minz)
        zb = field(xyz, Variogram([("spherical", 0.8, 160.0)], nugget=0.2), box, seed + deposit, 30)
        names = {"tennantite": "tenantite", "molibdenite": "molybdenite"}
        attrs = {
            "DHID": holes[:, 0],
            "from": holes[:, 4],
            "to": holes[:, 5],
            "azimut": holes[:, 6],
            "dip": holes[:, 7],
            "anomaly": np.minimum(rng.geometric(0.55, len(holes)), 5).astype(np.float64),
            "minz": minz,
        }
        attrs |= {names.get(k, k): v for k, v in mineral.items()}
        attrs["bwi"] = _bwi(minz, zb)
        attrs["recovery"] = np.clip(
            90.0 - 0.45 * mineral["clays"] + 3.0 * zb + 2.5 * rng.standard_normal(len(holes)), 60.0, 94.2
        )
        y, x = np.mgrid[0:142, 0:97]
        origin = (-515.0, -765.0)
        grid = np.column_stack([origin[0] + 10.0 * (x.ravel() + 0.5), origin[1] + 10.0 * (y.ravel() + 0.5)])
        return {
            "drillholes": PointSet(xyz, attrs),
            "topography": BlockModel(
                origin, (10.0, 10.0), (97, 142), attributes={"midz": np.round(ground(grid), 1)}
            ),
        }

    box, ground, zones = _porphyry_geology(seed, 1)
    nx, ny, nz = 49, 71, 44
    k, j, i = np.mgrid[0:nz, 0:ny, 0:nx]
    centers = np.column_stack(
        [
            -520.0 + 20.0 * (i.ravel() + 0.5),
            -870.0 + 20.0 * (j.ravel() + 0.5),
            1730.0 + 20.0 * (k.ravel() + 0.5),
        ]
    )
    rock = centers[:, 2] < ground(centers[:, :2])
    index = np.flatnonzero(rock)
    xyz = centers[rock]
    minz = zones(xyz)
    mineral = _mineral_fields(xyz, box, seed + 1, minz, shrink=0.72)
    zb = field(xyz, Variogram([("spherical", 1.0, 200.0)]), box, seed + 1, 30)
    cu = (
        0.7986 * mineral["chalcocite"]
        + 0.6331 * mineral["bornite"]
        + 0.3463 * mineral["chalcopyrite"]
        + 0.5175 * mineral["tennantite"]
    )
    blocks = {
        "ton": np.full(len(xyz), 1000.0),
        **mineral,
        "cu": cu,
        "mo": 0.6 * mineral["molibdenite"],
        "as": 0.2 * mineral["tennantite"],
        "rec": np.clip(88.0 - 0.35 * mineral["clays"] + 3.4 * zb, 60.0, 94.9),
        "bwi": np.clip(13.7 + 2.6 * zb, 9.0, 28.6),
    }
    model = BlockModel(
        (-520.0, -870.0, 1730.0),
        (20.0, 20.0, 20.0),
        (nx, ny, nz),
        attributes=blocks,
        index=index.astype(np.uint64),
    )

    holes, _, _, _ = _porphyry_holes(seed, 1, 154, 5.0, (-95.0, 140.0))
    hxyz = holes[:, 1:4]
    hz = zones(hxyz)
    hmin = _mineral_fields(hxyz, box, seed + 1, hz)
    spanish = dict(
        zip(
            MINERALS, ("arcilla", "calcosina", "bornita", "calcopirita", "tenantita", "molibdenita", "pirita")
        )
    )
    attrs = {
        "DHID": holes[:, 0],
        "from": holes[:, 4],
        "to": holes[:, 5],
        "azimut": holes[:, 6],
        "dip": holes[:, 7],
        "minz": hz,
    }
    attrs |= {spanish[k]: v for k, v in hmin.items()}
    attrs["bwi"] = _bwi(
        hz, field(hxyz, Variogram([("spherical", 0.8, 160.0)], nugget=0.2), box, seed + 1, 31)
    )

    py, px, pz = np.meshgrid(
        np.arange(-750.0, 651.0, 100.0),
        np.arange(-500.0, 451.0, 50.0),
        np.arange(1745.0, 2601.0, 15.0),
        indexing="ij",
    )
    pseudo = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
    pseudo = pseudo[pseudo[:, 2] < ground(pseudo[:, :2])]
    pseudo = pseudo[np.sort(rng.permutation(len(pseudo))[:3101])]

    ref = rng.standard_normal((741, 7)) @ np.linalg.cholesky(MINERAL_CORRELATION + 1e-3 * np.eye(7)).T
    ref_zone = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], 741, p=[0.3, 0.1, 0.15, 0.2, 0.25])
    distributions = {}
    for c, name in enumerate(MINERALS):
        median, sigma, cap = MINERAL_SCALE[name]
        shift = np.array([ZONE_SHIFT[int(a)].get(name, 0.0) for a in ref_zone])
        distributions[name] = np.round(np.minimum(median * np.exp(sigma * ref[:, c] + shift), cap), 2)
    bwi_ref = np.round(np.clip(rng.normal(17.0, 3.0, 1366), 9.0, 28.0), 2)
    bwi_ref[rng.random(1366) < 0.03] = np.nan
    scores = np.column_stack([np.log(np.maximum(hmin[m], 1e-6)) for m in MINERALS])
    measured = np.round(np.corrcoef(scores.T), 3)
    correlations = {"Nscore:data": np.asarray(MINERALS, dtype=object)}
    correlations |= {m: measured[c] for c, m in enumerate(MINERALS)}
    return {
        "block_model": model,
        "correlations": Table(correlations),
        "grindability_distribution": Table({"bwi": bwi_ref}),
        "mineralogical_distributions": Table(distributions),
        "pseudo_drillholes": PointSet(pseudo, {"minz": zones(pseudo)}),
        "synthetic_drillholes": PointSet(hxyz, attrs),
    }
