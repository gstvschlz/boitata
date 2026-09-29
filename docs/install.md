# Install

ceres is published on PyPI as `ceresgeo` and imported as `ceres`. It ships as one `abi3` wheel per platform
(Linux x86_64 and aarch64, macOS Intel and Apple silicon, Windows x64) that serves every Python from 3.11, so no
compiler is needed.

```sh
pip install "ceresgeo[all]"
```

```python
import ceres as cs
```

## Environment managers

### venv and pip

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "ceresgeo[all]"
```

### uv

```sh
uv add "ceresgeo[all]"            # in a uv project
uv pip install "ceresgeo[all]"    # in any environment
```

### poetry

```sh
poetry add "ceresgeo[all]"
```

### conda or mamba

Install the dependencies from conda-forge, then ceres with pip and `--no-deps`, so that pip leaves the conda
packages alone.

```sh
conda create -n geo -c conda-forge python numpy tqdm matplotlib pyvista polars pandas pyarrow pip
conda activate geo
pip install --no-deps ceresgeo
```

### pixi

```sh
pixi add python numpy tqdm matplotlib pyvista polars pandas pyarrow
pixi add --pypi "ceresgeo[all]"
```

## Extras

The required dependencies are numpy and tqdm. The others are imported on first use, and the error names the
missing package.

| pip extra | Adds | conda-forge packages |
|-----------|------|----------------------|
| (none) | `numpy`, `tqdm` | `numpy tqdm` |
| `plot` | `ceres.plot` | `matplotlib` (or `matplotlib-base`) |
| `3d` | `ceres.plot3d` | `pyvista` |
| `all` | the above, and `to_polars`, `to_pandas`, `to_pyarrow` | `matplotlib pyvista polars pandas pyarrow` |

## Minimum versions

| Package | Minimum |
|---------|---------|
| Python | 3.11 |
| numpy | 1.26 (1.x and 2.x both work) |
| tqdm | 4.66 |
| matplotlib | 3.8 |
| pyvista | 0.45 |
| polars | 1.4 |
| pandas | 2.2 |
| pyarrow | 16 |

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

On a connected machine with the same OS, architecture and Python, download the wheels, then install without an
index:

```sh
pip download "ceresgeo[all]" -d wheels                                 # connected machine
pip install --no-index --find-links wheels "ceresgeo[all]"             # offline machine
```

For datasets, copy the commit folder of a filled cache and point `CERES_DATA` at it.

## Contributors

Building ceres needs Rust ≥ 1.97 (pinned in `rust-toolchain.toml`) and a C compiler; `mise run build` compiles it
into the development environment. `mise run compat` builds a wheel into `dist/`, installs it into fresh
environments and runs the Python tests in each. `mise run compat <mode>` runs one mode; all but `dist` reuse what
is already in `dist/`.

| Mode | Environment |
|------|-------------|
| `dist` | builds the sdist and wheel only |
| `venv` | `python -m venv` and pip, all extras |
| `uv` | uv, Python 3.11, all extras |
| `pixi` | `pixi.toml` environments: `min` (lowest supported versions), `latest`, `sdist` (built with conda-forge Rust) |
| `conda` | `environment.yml` with micromamba, mamba or conda |

A mode whose tool is not installed is skipped. `mise run conda:recipe` builds the draft conda-forge recipe in
`packaging/conda-forge/`.
