"""Frame rate and GPU memory of `bt.plot3d.Scene` layers on synthetic block models and drill holes.

Each case runs in its own process: build the layer, render once, then orbit the camera off-screen at the
interactive update rate (so volumes draw their coarse copy, as while dragging). GPU memory is the dedicated
memory of the case's process on Windows (performance counters), elsewhere the rise in ``nvidia-smi`` device
memory over the case, "n/a" without either.

Usage: python ci/bench_viewer.py [--sizes 1e5,1e6,1e7] [--holes 1000,10000] [--frames 60] [--csv out.csv]
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

CASES = [
    ("regular", "surface"),
    ("regular", "volume"),
    ("masked", "surface"),
    ("subblocked", "surface"),
    ("points", None),
    ("holes", None),
]
COLUMNS = ["kind", "style", "size", "cells", "build_s", "first_frame_s", "fps", "gpu_mib"]


def gpu_mib():
    if sys.platform == "win32":
        counter = rf"\GPU Process Memory(pid_{os.getpid()}_*)\Dedicated Usage"
        query = f"((Get-Counter '{counter}').CounterSamples | Measure-Object CookedValue -Sum).Sum"
        cmd, scale = ["powershell", "-NoProfile", "-Command", query], 2**-20
    elif shutil.which("nvidia-smi"):
        cmd, scale = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 1
    else:
        return None
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    try:
        return float(out.stdout.split()[0]) * scale
    except (IndexError, ValueError):
        return None


def blocks(kind, size, rng):
    import boitata as bt

    n = max(round(size ** (1 / 3)), 2)
    grid = {"origin": (0.0, 0.0, 0.0), "size": (10.0, 10.0, 5.0), "count": (n, n, n)}
    if kind == "subblocked":
        parent = np.repeat(np.arange(0, n**3, 2, dtype=np.uint64), 2)
        half = np.array([[0, 0, 0, 0.5, 1, 1], [0.5, 0, 0, 1, 1, 1]])
        extents = np.tile(half, (len(parent) // 2, 1))
        values = {"v": rng.lognormal(size=len(parent))}
        return bt.BlockModel.subblocked(**grid, parent=parent, extents=extents, attributes=values)
    model = bt.BlockModel(**grid, attributes={"v": rng.lognormal(size=n**3)})
    return model.mask(rng.random(n**3) < 0.5) if kind == "masked" else model


def holes(count, rng, intervals=40, length=4.0):
    import boitata as bt

    side = int(np.ceil(np.sqrt(count)))
    names = [f"H{i:05d}" for i in range(count)]
    collar = {
        "HOLE_ID": names,
        "X": np.arange(count) % side * 25.0,
        "Y": np.arange(count) // side * 25.0,
        "Z": np.full(count, 500.0),
    }
    survey = {
        "HOLE_ID": names,
        "DEPTH": np.zeros(count),
        "AZIMUTH": rng.uniform(0, 360, count),
        "DIP": rng.uniform(50, 90, count),
    }
    start = np.tile(np.arange(intervals) * length, count)
    table = {
        "HOLE_ID": np.repeat(names, intervals).tolist(),
        "FROM": start,
        "TO": start + length,
        "v": rng.lognormal(size=count * intervals),
    }
    return bt.Drillholes(collar, survey, table)


def case(kind, style, size, frames, seed):
    import boitata as bt

    rng = np.random.default_rng(seed)
    if kind == "holes":
        data = holes(int(size), rng)
    elif kind == "points":
        data = bt.PointSet(rng.uniform(0, 1000, (int(size), 3)), {"v": rng.lognormal(size=int(size))})
    else:
        data = blocks(kind, size, rng)
    base = gpu_mib()
    scene = bt.plot3d.Scene(off_screen=True, window_size=(1280, 720))
    t = time.perf_counter()
    scene.add(data, "v", style=style)
    build = time.perf_counter() - t
    window = scene.plotter.ren_win
    t = time.perf_counter()
    window.Render()
    window.WaitForCompletion()
    first = time.perf_counter() - t
    window.SetDesiredUpdateRate(15.0)
    camera = scene.plotter.camera
    t = time.perf_counter()
    for _ in range(frames):
        camera.Azimuth(360 / frames)
        window.Render()
        window.WaitForCompletion()
    fps = frames / (time.perf_counter() - t)
    used = gpu_mib()
    cells = sum(mesh.n_cells for mesh, *_ in scene._layers)
    scene.close()
    return {
        "kind": kind,
        "style": style or "default",
        "size": int(size),
        "cells": cells,
        "build_s": round(build, 3),
        "first_frame_s": round(first, 3),
        "fps": round(fps, 1),
        "gpu_mib": "n/a" if used is None else round(used - (base or 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", default="1e5,1e6,1e7", help="block and point counts")
    parser.add_argument("--holes", default="1000,10000", help="hole counts, 40 intervals of 4 m each")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900, help="seconds per case")
    parser.add_argument("--csv", help="also write the results here")
    parser.add_argument("--case", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        kind, style, size = args.case
        style = None if style == "default" else style
        print(json.dumps(case(kind, style, float(size), args.frames, args.seed)))
        return
    sizes = [float(s) for s in args.sizes.split(",")]
    counts = [float(s) for s in args.holes.split(",")]
    rows = []
    print(" ".join(f"{c:>13}" for c in COLUMNS), flush=True)
    for kind, style in CASES:
        for size in counts if kind == "holes" else sizes:
            cmd = [sys.executable, __file__, "--case", kind, style or "default", str(size)]
            cmd += ["--frames", str(args.frames), "--seed", str(args.seed)]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=args.timeout)
                row = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else None
                error = out.stderr.strip().splitlines()[-1:] if row is None else None
            except subprocess.TimeoutExpired:
                row, error = None, ["timeout"]
            if row is None:
                row = dict.fromkeys(COLUMNS, "-") | {"kind": kind, "style": style, "size": int(size)}
                row["fps"] = (error or ["failed"])[0][:13]
            rows.append(row)
            print(" ".join(f"{row[c]!s:>13}" for c in COLUMNS), flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
