# install

boitatá lives on PyPI and imports as `boitata`. each platform gets one `abi3` wheel (linux x86_64 and aarch64,
macos intel and apple silicon, windows x64) that serves every python from 3.11, so you need no compiler.

```sh
pip install "boitata[all]"
```

```python
import boitata as bt
```

## environment managers

### venv and pip

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "boitata[all]"
```

### uv

```sh
uv add "boitata[all]"            # in a uv project
uv pip install "boitata[all]"    # in any environment
```

### poetry

```sh
poetry add "boitata[all]"
```

### conda or mamba

install the dependencies from conda-forge, then boitatá with pip and `--no-deps`, so pip leaves the conda packages
alone.

```sh
conda create -n geo -c conda-forge python numpy tqdm matplotlib pyvista polars pandas pyarrow pip
conda activate geo
pip install --no-deps boitata
```

### pixi

```sh
pixi add python numpy tqdm matplotlib pyvista polars pandas pyarrow
pixi add --pypi "boitata[all]"
```

## extras

numpy and tqdm are the only required dependencies. boitatá imports the others on first use, and the error names the
missing package.

| pip extra | adds | conda-forge packages |
|-----------|------|----------------------|
| (none) | `numpy`, `tqdm` | `numpy tqdm` |
| `plot` | `boitata.plot` | `matplotlib` (or `matplotlib-base`) |
| `3d` | `boitata.plot3d` | `pyvista` |
| `all` | the above, `to_polars`, `to_pandas`, `to_pyarrow`, and progress bars drawn as notebook widgets | `matplotlib pyvista polars pandas pyarrow ipywidgets` |

## minimum versions

| package | minimum |
|---------|---------|
| python | 3.11 |
| numpy | 1.26 (1.x and 2.x both work) |
| tqdm | 4.66 |
| matplotlib | 3.8 |
| pyvista | 0.45 |
| polars | 1.4 |
| pandas | 2.2 |
| pyarrow | 16 |
| ipywidgets | 8 |

## headless 3D rendering

`boitata.plot3d` renders with VTK, which needs OpenGL. on a linux server or container without a display, either
install `mesalib` from conda-forge (or the system `libEGL`) and set `PYVISTA_OFF_SCREEN=true`, or run under a
virtual display:

```sh
xvfb-run -a python script.py              # apt install xvfb
```

## datasets

`boitata.datasets` downloads files from raw.githubusercontent.com on first use, checks their SHA-256 and caches
them in one folder per datasets commit, under `%LOCALAPPDATA%\boitata` on windows and under
`$XDG_CACHE_HOME/boitata` or `~/.cache/boitata` elsewhere (macos included). set `BOITATA_DATA` to use your own
folder; the files then sit in it directly.

downloads honor the `HTTPS_PROXY`, `HTTP_PROXY` and `NO_PROXY` variables (and the system proxy settings on windows
and macos). behind a proxy that re-signs TLS, point `SSL_CERT_FILE` at its CA bundle.

## air-gapped installs

on a connected machine with the same OS, architecture and python, download the wheels, then install them
without an index:

```sh
pip download "boitata[all]" -d wheels                                 # connected machine
pip install --no-index --find-links wheels "boitata[all]"             # offline machine
```

for datasets, copy the commit folder of a filled cache and point `BOITATA_DATA` at it.

## contributors

to build boitatá you need rust ≥ 1.97 (pinned in `rust-toolchain.toml`) and a C compiler. `mise run build` compiles it
into the development environment. `mise run compat` builds a wheel into `dist/`, installs it into fresh
environments and runs the python tests in each. `mise run compat <mode>` runs one mode; every mode except `dist`
reuses what `dist/` already holds.

| mode | environment |
|------|-------------|
| `dist` | builds the sdist and wheel only |
| `venv` | `python -m venv` and pip, all extras |
| `uv` | uv, python 3.11, all extras |
| `pixi` | `pixi.toml` environments: `min` (lowest supported versions), `latest`, `sdist` (built with conda-forge rust) |
| `conda` | `environment.yml` with micromamba, mamba or conda |

`mise run compat` skips a mode whose tool is missing. `mise run conda:recipe` builds the draft conda-forge recipe in
`packaging/conda-forge/`.
