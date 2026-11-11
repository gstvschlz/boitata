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
  <a href="https://gstvschlz.github.io/boitata/examples/01-getting-started/">Examples</a> ·
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

## Agent plugin

Without help, a coding agent guesses at boitatá's calls. The plugin gives it the library's rules and tells it to
read each signature from the installed package, so the code it writes matches your version. One plugin serves
Claude, Codex, ChatGPT and Antigravity.

In [Claude Code](https://claude.com/claude-code):

```text
/plugin marketplace add gstvschlz/boitata
/plugin install boitata@boitata
```

In [Codex](https://developers.openai.com/codex), and from there in the ChatGPT desktop app:

```sh
codex plugin marketplace add gstvschlz/boitata
codex plugin add boitata@boitata
```

In [Antigravity](https://antigravity.google):

```sh
agy plugin install https://github.com/gstvschlz/boitata/tree/main/packaging/plugin
```

The [install page](https://gstvschlz.github.io/boitata/install/#agent-plugin) covers the Claude apps, ChatGPT
workspaces and other agents, and says what the plugin holds and how to update it.

## Examples

The [examples folder](https://github.com/gstvschlz/boitata/blob/main/examples) holds worked pages in 14 sections, each
alternating text, code and its results; the [docs render them with figures](https://gstvschlz.github.io/boitata/examples/01-getting-started/).
For the geostatistics itself, the [learn track](https://gstvschlz.github.io/boitata/learn/01-samples-and-support/learn_01/)
runs from what a sample stands for to checking a model.

## License

MIT
