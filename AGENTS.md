# Boitatá

## Purpose

Boitatá is a Rust-core, Python-exposed geostatistics library for day-to-day
mining/geology workflows: variograms, kriging, simulation, domaining, block
models. Primary user is the author, for personal geostatistics work; it is
also published as open source for others to use.

## Commands

Pinned via `mise.toml` (Python, uv, pixi, micromamba) and `rust-toolchain.toml`
(Rust).

```sh
mise install        # one-time setup
mise run build       # cargo build --workspace
mise run test        # rust tests, then builds the python module, then pytest
mise run lint        # rustfmt --check, clippy -D warnings, ruff check + format --check
mise run verify       # lint + test — must pass before a PR
mise run examples     # reruns examples/*, rewrites their figures
mise run docs         # mkdocs serve
mise run docs-build   # mkdocs build --strict
mise run compat       # install via pip, uv, pixi and conda
```

## Layout

- `crates/` — Rust workspace: `core`, `blocks`, `coda`, `drillholes`, `eda`,
  `estimation`, `modeling`, `simulation`, `transforms`, `variogram`, `io`,
  and `py` (the pyo3 binding crate).
- `python/boitata/` — Python package (thin wrappers, `_boitata.pyi` stub,
  `plot.py`, `plot3d.py`, `datasets.py`, `checks.py`, `errors.py`).
- `python/tests/` — pytest suite.
- `examples/01-...` through `13-...` — numbered, narrative example scripts
  rendered into docs (`mise run examples`).
- `docs/` — mkdocs site (`index.md`, `install.md`, hooks for example pages).

New Rust algorithms live under the matching `crates/` subdomain; the pyo3
binding goes in `crates/py`; the Python-facing wrapper/stub goes in
`python/boitata/`.

## Conventions

See `CONTRIBUTING.md` for the full list (Conventional Commits, one change per
PR, public/seeded datasets only, no proprietary vendor formats, results must
not depend on thread count). In addition:

- Every feature ships together: Rust implementation, Python binding, `.pyi`
  stub, NumPy-style docstring, and an example. A new algorithm also needs a
  rerunnable theory check (e.g. OK weights sum to 1, kriging exact at data,
  SGS reproduces histogram/variogram within tolerance).
- Before writing docs or examples, read existing ones nearby and match their
  tone and structure — don't introduce a new voice.
- No code comments except where genuinely non-obvious (a hidden constraint,
  a workaround, a subtle invariant). Don't comment what the code already
  says.

## Domain glossary

Geostatistics/mining terminology (lens, SMU, composite, domain, kriging
neighborhood, etc.) is defined in `docs/` and in the relevant `crates/*`
module docs — read there rather than assuming a definition.

## Never do

- Never add AI attribution/co-author trailers to commits or PR descriptions
  in this repo (no `Co-Authored-By: Claude...`, no "Generated with Claude
  Code" footers).
- Never commit proprietary/vendor data formats, client data, or files over
  1 MB — tests and examples use `bt.datasets` or seeded synthetic data only.
- Never introduce thread-count-dependent results.
- Never push or force-push without asking first.
- Never call a task done without running `mise run verify`.

## Definition of done

- `mise run verify` passes (lint + test).
- New/changed Rust functionality has a matching Python binding, `.pyi` stub
  update, NumPy-style docstring, and an example.
- `mise run docs-build` succeeds if docs/examples changed.
