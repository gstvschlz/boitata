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

## Claude plugin

Without help, Claude guesses at boitatá's calls from whatever it remembers. The plugin holds one skill, a short
page Claude reads when your code imports `boitata` or you ask for geostatistics work. It gives the kinds of object
and the calls each one takes, the rules every call follows (keyword-only options, containers that never change in
place, the `(azimuth, dip, rake)` rotation), how missing values and errors behave, and how to save each object.
It also tells Claude to read each signature from the stub and the docstrings of the boitatá you installed, so the
code matches your version.

### Claude Code

Add this repository as a marketplace, then install the plugin from it:

```text
/plugin marketplace add https://gitlab.com/scholzegustavo/boitata.git
/plugin install boitata@boitata
```

From a shell, the same two steps are `claude plugin marketplace add` and `claude plugin install`. The plugin
installs for your user by default; add `--scope project` to the second command to record it in the project's
`.claude/settings.json` for everyone who works on it.

`/plugin marketplace update boitata` fetches the current skill after a new release.

### Claude apps

In Claude on the web or the desktop, open **Customize > Plugins**, choose **Add > Add marketplace** and paste
`https://gitlab.com/scholzegustavo/boitata.git`. Then install `boitata` from the list. Claude runs code in its own
sandbox there, so ask it to `pip install "boitata[all]"` first.

The plugin adds no tools and sends nothing anywhere: it is the text in
[`packaging/claude/skills/boitata/SKILL.md`](https://gitlab.com/scholzegustavo/boitata/-/blob/main/packaging/claude/skills/boitata/SKILL.md),
which you can read before you install it.

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
