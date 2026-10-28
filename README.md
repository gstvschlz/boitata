<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://gitlab.com/scholzegustavo/boitata/-/raw/main/docs/assets/logo-animated-dark.svg">
    <img src="https://gitlab.com/scholzegustavo/boitata/-/raw/main/docs/assets/logo-animated.svg" alt="boitatá" width="120">
  </picture>
</p>

<h1 align="center">boitatá</h1>

<p align="center">
  My Python tooling for day-to-day geostatistics.<br>
  <a href="https://scholzegustavo.gitlab.io/boitata/">Docs</a> ·
  <a href="https://scholzegustavo.gitlab.io/boitata/examples/01-first-steps/">Examples</a> ·
  <a href="https://scholzegustavo.gitlab.io/boitata/api/containers/">API</a>
</p>

<p align="center">
  <a href="https://gitlab.com/scholzegustavo/boitata/-/releases"><img src="https://gitlab.com/scholzegustavo/boitata/-/badges/release.svg" alt="Latest release"></a>
  <a href="https://colab.research.google.com/github/gstvschlz/boitata/blob/main/notebooks/learn_00.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

> boitatá is a guardian spirit of underground resources in Brazilian folklore.

`boitatá` holds the tools I use for resource work: drill hole checks, statistics, transforms, variograms, kriging, simulation and model checks.

## Install

```sh
pip install "boitata[all]"
```

Published on PyPI and imported as `boitata`. Wheels for Linux, macOS and Windows, Python ≥ 3.11.
Extras: `plot` (Matplotlib), `3d` (the notebook widget of `bt.plot3d`), `export` (headless screenshots), `all` (`plot`, `3d`, Polars, pandas, PyArrow).
See the [install page](https://scholzegustavo.gitlab.io/boitata/install/) for uv, Poetry, conda, pixi and offline installs.

## Examples

The [examples folder](https://gitlab.com/scholzegustavo/boitata/-/blob/main/examples) holds worked pages in 14 sections, each
alternating text, code and its results; the [docs render them with figures](https://scholzegustavo.gitlab.io/boitata/examples/01-first-steps/).
For the geostatistics itself, the [learn track](https://scholzegustavo.gitlab.io/boitata/learn/01-samples-and-support/learn_01/)
runs from what a sample stands for to checking a model.

## License

MIT
