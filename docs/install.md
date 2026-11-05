# Install

boitatá lives on PyPI and imports as `boitata`. Each platform gets one `abi3` wheel (Linux x86_64 and aarch64,
macOS Intel and Apple silicon, Windows x64) that serves every Python from 3.11, so you need no compiler.

```sh
pip install "boitata[all]"
```

```python
import boitata as bt
```

## Environment managers

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

### Poetry

```sh
poetry add "boitata[all]"
```

### conda or mamba

Install the dependencies from conda-forge, then boitatá with pip and `--no-deps`, so pip leaves the conda packages
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

## Extras

NumPy is the only required dependency. boitatá imports the others on first use, and the error names the
missing package.

| pip extra | Adds | conda-forge packages |
|-----------|------|----------------------|
| (none) | `numpy` | `numpy` |
| `plot` | `boitata.plot` | `matplotlib` (or `matplotlib-base`) |
| `3d` | The notebook widget of `boitata.plot3d`, synced both ways with Python; without it a scene shows as a standalone page | `anywidget` |
| `export` | `Scene.screenshot`, rendered in headless Chromium | `playwright` |
| `all` | `plot`, `3d`, and `to_polars`, `to_pandas`, `to_pyarrow` | `matplotlib anywidget polars pandas pyarrow` |

## Minimum versions

| Package | Minimum |
|---------|---------|
| Python | 3.11 |
| NumPy | 1.26 (1.x and 2.x both work) |
| Matplotlib | 3.8 |
| anywidget | 0.9 |
| Playwright | 1.45 |
| Polars | 1.4 |
| pandas | 2.2 |
| PyArrow | 16 |

## 3D scenes and screenshots

`boitata.plot3d` draws in the browser with WebGL, so it needs no OpenGL or display on the machine running Python.
`Scene.screenshot` renders the scene in headless Chromium through Playwright (the `export` extra), which downloads
Chromium on first use; it runs on servers and in containers without a display. On a bare Linux image, install
Chromium's system libraries once:

```sh
playwright install --with-deps chromium
```

If Jupyter runs in another environment than the kernel, install `anywidget` there too, or the notebook shows each
scene as a standalone page without two-way sync.

## Datasets

`boitata.datasets` generates each dataset on the call, from a seed. Nothing is downloaded or cached, so the datasets
work offline and behind any proxy.

## Air-gapped installs

On a connected machine with the same OS, architecture and Python, download the wheels, then install them
without an index:

```sh
pip download "boitata[all]" -d wheels                                 # connected machine
pip install --no-index --find-links wheels "boitata[all]"             # offline machine
```

## Contributors

To build boitatá you need Rust ≥ 1.97 (pinned in `rust-toolchain.toml`) and a C compiler. `mise run build` compiles it
into the development environment. `mise run compat` builds a wheel into `dist/`, installs it into fresh
environments and runs the Python tests in each. `mise run compat <mode>` runs one mode; every mode except `dist`
reuses what `dist/` already holds.

| Mode | Environment |
|------|-------------|
| `dist` | Builds the sdist and wheel only |
| `venv` | `python -m venv` and pip, all extras |
| `uv` | uv, Python 3.11, all extras |
| `pixi` | `pixi.toml` environments: `min` (lowest supported versions), `latest`, `sdist` (built with conda-forge Rust) |
| `conda` | `environment.yml` with micromamba, mamba or conda |

`mise run compat` skips a mode whose tool is missing. `mise run conda:recipe` builds the draft conda-forge recipe in
`packaging/conda-forge/`.
