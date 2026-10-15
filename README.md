<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gstvschlz/boitata/main/docs/assets/logo-animated-dark.svg">
    <img src="https://raw.githubusercontent.com/gstvschlz/boitata/main/docs/assets/logo-animated.svg" alt="boitatá" width="120">
  </picture>
</p>

<h1 align="center">boitatá</h1>

<p align="center">
  My Python tooling for day-to-day geostatistics.<br>
  <a href="https://gstvschlz.github.io/boitata/">Docs</a> ·
  <a href="https://gstvschlz.github.io/boitata/examples/01-first-steps/">Examples</a> ·
  <a href="https://gstvschlz.github.io/boitata/api/containers/">API</a>
</p>

<p align="center">
  <a href=""><img src="https://github.com/gstvschlz/boitata/actions/workflows/release.yml/badge.svg" alt="CI/CD status"></a>
  <a href=""><img src="https://github.com/gstvschlz/boitata/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
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
See the [install page](https://gstvschlz.github.io/boitata/install/) for uv, Poetry, conda, pixi and offline installs.

## Examples

The [examples folder](https://github.com/gstvschlz/boitata/blob/main/examples) holds worked pages in 14 sections, each
alternating text, code and its results; the [docs render them with figures](https://gstvschlz.github.io/boitata/examples/01-first-steps/).
For the geostatistics itself, the [learn track](https://gstvschlz.github.io/boitata/learn/01-samples-and-support/learn_01/)
runs from what a sample stands for to checking a model.

## License

MIT
