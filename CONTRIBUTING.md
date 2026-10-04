# Contributing to Boitatá

Bug reports, fixes, examples and new features are welcome. Open an issue first for anything
bigger than a small fix, so we can agree on the design before you write the code.

## Set up

Boitatá uses [mise](https://mise.jdx.dev) to pin Python, uv, pixi and micromamba; Rust comes
from `rust-toolchain.toml`. From a clone:

```sh
mise install
mise run test
```

## Before you open a pull request

- `mise run verify` runs `lint` (rustfmt, clippy with warnings as errors, ruff) and `test`
  (the Rust tests, then builds the Python module and runs pytest). It must pass.
- `mise run examples` reruns the scripts in `examples/` and rewrites their figures.
  Run it when you change an example.
- `mise run compat` installs Boitatá with pip, uv, pixi and conda. Run it when you change
  packaging or dependencies.
- `mise run bench-viewer` measures frame rate and GPU memory of `bt.plot3d` on large block
  models and drill holes (`--sizes 1e5 --csv out.csv` to narrow it). Run it when you change
  the 3D viewer.

Every feature ships in Rust and Python together: the binding, the `.pyi` stub, a NumPy
docstring and an example. A new algorithm also needs a theory check we can rerun, for
example ordinary kriging weights summing to 1, kriging being exact at the data, or SGS
reproducing the histogram and variogram within a tolerance.

## Pull requests

- Link one issue per pull request, and keep each pull request to one change.
- Write commit messages as [Conventional Commits](https://www.conventionalcommits.org)
  (`feat:`, `fix:`, `docs:` ...). The changelog comes from them.
- Use public datasets in tests and examples (`bt.datasets`) or seeded synthetic data.
  Don't commit client data or files over 1 MB.
- Only open or documented file formats; no readers or writers for proprietary vendor formats.
- Results must not depend on the number of threads.
- Say which platforms you tested on (Windows, Linux, macOS).

By contributing, you agree to license your work under the [MIT License](LICENSE).
