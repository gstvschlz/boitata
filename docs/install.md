# Install

ceres is not on PyPI or conda-forge yet; there will be no release there until the API settles. Install it from
the git repository, or from a wheel attached to a [GitHub Release](https://github.com/gstvschlz/ceres/releases)
once version tags exist. Python ≥ 3.11 is required.

```sh
pip install "ceres[all] @ git+https://github.com/gstvschlz/ceres"   # today, builds from source
pip install "./ceres-0.1.0-cp311-abi3-<platform>.whl[all]"         # a wheel downloaded from a Release
pip install "ceres[all]"                                            # later, once on PyPI
```

A git install compiles the Rust core, so it needs a Rust toolchain and a C compiler (see
[Building from source](#building-from-source)). A wheel needs neither: one `abi3` wheel per platform serves every
Python from 3.11.

## Environment managers

To install from a downloaded wheel, replace `git+https://github.com/gstvschlz/ceres` by the wheel's path.

### venv and pip

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "ceres[all] @ git+https://github.com/gstvschlz/ceres"
```

### uv

```sh
uv add "ceres[all] @ git+https://github.com/gstvschlz/ceres"         # in a uv project
uv pip install "ceres[all] @ git+https://github.com/gstvschlz/ceres"  # in any environment
```

### poetry

```sh
poetry add git+https://github.com/gstvschlz/ceres.git -E all
```

### conda or mamba

Install the dependencies from conda-forge, then ceres with pip and `--no-deps`, so that pip leaves the conda
packages alone. `rust`, `c-compiler` and `maturin` are only needed to build from git.

```sh
conda create -n geo -c conda-forge python numpy matplotlib pyvista polars pandas pyarrow \
    rust c-compiler "maturin>=1.10,<2" pip
conda activate geo
pip install --no-deps --no-build-isolation git+https://github.com/gstvschlz/ceres
```

### pixi

```sh
pixi add python numpy matplotlib pyvista polars pandas pyarrow
pixi add --pypi "ceres @ git+https://github.com/gstvschlz/ceres"
```

The build needs Rust on `PATH` ([rustup](https://rustup.rs)). With a wheel instead:
`pixi run pip install --no-deps ./ceres-0.1.0-cp311-abi3-<platform>.whl`.

## Extras

The only required dependency is numpy. The others are imported on first use, and the error names the missing
package.

| pip extra | Adds | conda-forge packages |
|-----------|------|----------------------|
| (none) | `numpy` | `numpy` |
| `plot` | `ceres.plot` | `matplotlib` (or `matplotlib-base`) |
| `3d` | `ceres.plot3d` | `pyvista` |
| `all` | the above, and `to_polars`, `to_pandas`, `to_pyarrow` | `matplotlib pyvista polars pandas pyarrow` |

## Minimum versions

| Package | Minimum |
|---------|---------|
| Python | 3.11 |
| numpy | 1.26 (1.x and 2.x both work) |
| matplotlib | 3.8 |
| pyvista | 0.45 |
| polars | 1.4 |
| pandas | 2.2 |
| pyarrow | 16 |

## Building from source

Either install Rust ≥ 1.97 with [rustup](https://rustup.rs) (the repository pins its toolchain in
`rust-toolchain.toml`) and a C compiler, or take all of it from conda-forge:

```sh
git clone https://github.com/gstvschlz/ceres && cd ceres
conda env create -f environment.yml       # rust, c-compiler, maturin, test dependencies
conda activate ceres
pip install --no-deps --no-build-isolation .
```

With rustup, `pip install .` is enough; pip fetches maturin itself.

## Headless 3D rendering

`ceres.plot3d` renders with VTK, which needs OpenGL. On a Linux server or container without a display, either
install `mesalib` from conda-forge (or the system `libEGL`) and set `PYVISTA_OFF_SCREEN=true`, or run under a
virtual display:

```sh
xvfb-run -a python script.py              # apt install xvfb
```

## Datasets

`ceres.datasets` downloads files from raw.githubusercontent.com on first use, checks their SHA-256 and caches
them in a folder per datasets commit under `%LOCALAPPDATA%\ceres` on Windows, and `$XDG_CACHE_HOME/ceres` or
`~/.cache/ceres` elsewhere (macOS included). Set `CERES_DATA` to use a folder of your own instead; the files then
sit directly in it.

Downloads honour the standard `HTTPS_PROXY`, `HTTP_PROXY` and `NO_PROXY` variables (and the system proxy
settings on Windows and macOS). Behind a proxy that re-signs TLS, point `SSL_CERT_FILE` at its CA bundle.

## Air-gapped installs

From a wheel: on a connected machine with the same OS, architecture and Python, download the ceres wheel from a
Release and its dependencies, then install without an index:

```sh
pip download "./ceres-0.1.0-cp311-abi3-<platform>.whl[all]" -d wheels   # connected machine
pip install --no-index --find-links wheels "ceres[all]"                  # offline machine
```

From source: vendor the Rust crates on a connected machine and copy the repository across.

```sh
mkdir -p .cargo && cargo vendor >> .cargo/config.toml
```

The offline machine then builds with Rust, a C compiler and maturin already installed:
`pip install --no-deps --no-build-isolation .`. For datasets, copy the commit folder of a filled cache and point
`CERES_DATA` at it.

## Contributors

`mise run compat` builds an sdist and a wheel into `dist/`, installs them into fresh environments and runs the
Python tests in each. `mise run compat <mode>` runs one mode; all but `dist` reuse what is already in `dist/`.

| Mode | Environment |
|------|-------------|
| `dist` | builds the sdist and wheel only |
| `venv` | `python -m venv` and pip, all extras |
| `uv` | uv, Python 3.11, all extras |
| `pixi` | `pixi.toml` environments: `min` (lowest supported versions), `latest`, `sdist` (built with conda-forge Rust) |
| `conda` | `environment.yml` with micromamba, mamba or conda |

A mode whose tool is not installed is skipped. `mise run conda:recipe` builds the draft conda-forge recipe in
`packaging/conda-forge/`.
