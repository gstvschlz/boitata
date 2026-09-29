"""Build ceres, install it into clean environments and run the Python tests there.

Usage: python ci/compat.py [all | dist | venv | uv | pixi | conda]
"""

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
COMPAT = ROOT / ".compat"
TESTS = ROOT / "python" / "tests"
WIN = sys.platform == "win32"
CHECK = (
    "import ceres, pathlib, sys; p = pathlib.Path(ceres.__file__).resolve(); "
    "print('ceres', ceres.__version__, 'from', p); "
    "sys.exit(p.is_relative_to(pathlib.Path(sys.argv[1]).resolve()))"
)


class Skip(Exception):
    pass


def run(*cmd, cwd=ROOT, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def which(*names):
    for name in names:
        if path := shutil.which(name):
            return path
    raise Skip(f"{' / '.join(names)} not installed")


def artifact(pattern):
    found = sorted(DIST.glob(pattern))
    if not found:
        raise SystemExit(f"no dist/{pattern}; run `python ci/compat.py dist` first")
    return found[-1]


def fresh(path):
    shutil.rmtree(path, ignore_errors=True)
    return path


def pytest(python):
    with tempfile.TemporaryDirectory() as tmp:
        run(*python, "-c", CHECK, ROOT / "python", cwd=tmp)
        run(*python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", ROOT, TESTS, cwd=tmp)


def venv_python(env):
    return env / ("Scripts/python.exe" if WIN else "bin/python")


def dist():
    fresh(DIST)
    toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    run(which("uv"), "build", "--out-dir", DIST, env={**os.environ, "RUSTUP_TOOLCHAIN": toolchain})


def venv():
    env = fresh(COMPAT / "venv")
    run(sys.executable, "-m", "venv", env)
    python = venv_python(env)
    run(python, "-m", "pip", "install", "-q", f"ceresgeo[all] @ {artifact('*.whl').as_uri()}", "pytest")
    pytest([python])


def uv():
    uv = which("uv")
    env = fresh(COMPAT / "uv")
    run(uv, "venv", "-q", "--python", "3.11", env)
    python = venv_python(env)
    run(
        uv, "pip", "install", "-q", "--python", python, f"ceresgeo[all] @ {artifact('*.whl').as_uri()}", "pytest"
    )
    pytest([python])


def pixi():
    pixi = which("pixi")
    for env, target, flags in [
        ("min", artifact("*.whl"), []),
        ("latest", artifact("*.whl"), []),
        ("sdist", artifact("*.tar.gz"), ["--no-build-isolation"]),
    ]:
        python = [pixi, "run", "--manifest-path", ROOT / "pixi.toml", "-e", env, "python"]
        run(*python, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *flags, target)
        pytest(python)


def conda():
    tool = which("micromamba", "mamba", "conda")
    env = fresh(COMPAT / "conda")
    create = ["create", "-y"] if Path(tool).stem == "micromamba" else ["env", "create"]
    run(tool, *create, "-f", ROOT / "environment.yml", "-p", env)
    python = [tool, "run", "-p", env, "python"]
    run(*python, "-m", "pip", "install", "-q", "--no-deps", artifact("*.whl"))
    pytest(python)


MODES = {"dist": dist, "venv": venv, "uv": uv, "pixi": pixi, "conda": conda}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode != "all" and mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; expected all, {', '.join(MODES)}")
    results = {}
    for name in MODES if mode == "all" else [mode]:
        print(f"== {name}", flush=True)
        try:
            MODES[name]()
            results[name] = "ok"
        except Skip as e:
            results[name] = f"skipped: {e}"
        except subprocess.CalledProcessError as e:
            results[name] = f"failed: exit {e.returncode}"
            if name == "dist":
                break
        print(f"== {name}: {results[name]}", flush=True)
    print("\n".join(f"{k:6} {v}" for k, v in results.items()))
    sys.exit(any(v.startswith("failed") for v in results.values()))


if __name__ == "__main__":
    main()
