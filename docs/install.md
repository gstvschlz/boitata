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
conda create -n geo -c conda-forge python numpy matplotlib anywidget polars pandas pyarrow pip
conda activate geo
pip install --no-deps boitata
```

### pixi

```sh
pixi add python numpy matplotlib anywidget polars pandas pyarrow
pixi add --pypi "boitata[all]"
```

## extras

numpy is the only required dependency. boitatá imports the others on first use, and the error names the
missing package.

| pip extra | adds | conda-forge packages |
|-----------|------|----------------------|
| (none) | `numpy` | `numpy` |
| `plot` | `boitata.plot` | `matplotlib` (or `matplotlib-base`) |
| `3d` | the notebook widget of `boitata.plot3d`, synced both ways with python; without it a scene shows as a standalone page | `anywidget` |
| `export` | `Scene.screenshot`, rendered in headless chromium | `playwright` |
| `all` | `plot`, `3d`, and `to_polars`, `to_pandas`, `to_pyarrow` | `matplotlib anywidget polars pandas pyarrow` |

## minimum versions

| package | minimum |
|---------|---------|
| python | 3.11 |
| numpy | 1.26 (1.x and 2.x both work) |
| matplotlib | 3.8 |
| anywidget | 0.9 |
| playwright | 1.45 |
| polars | 1.4 |
| pandas | 2.2 |
| pyarrow | 16 |

## 3D scenes and screenshots

`boitata.plot3d` draws in the browser with WebGL, so it needs no OpenGL or display on the machine running python.
`Scene.screenshot` renders the scene in headless chromium through playwright (the `export` extra), which downloads
chromium on first use; it runs on servers and in containers without a display. on a bare linux image, install
chromium's system libraries once:

```sh
playwright install --with-deps chromium
```

if jupyter runs in another environment than the kernel, install `anywidget` there too, or the notebook shows each
scene as a standalone page without two-way sync.

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
