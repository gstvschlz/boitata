# Examples

Worked examples with the Python API on open datasets from [gstvschlz/datasets](https://github.com/gstvschlz/datasets).
Walker Lake has an exhaustive grid, so every result is checked against the truth.

| # | Topic | Dataset | Covers |
|---|---|---|---|
| 1 | [Data and declustering](01-data/README.md) | Walker Lake | `read_csv`, `PointSet`, `cell_declustering` |
| 2 | [Normal-score transform](02-normal-score/README.md) | Walker Lake | `NormalScore`, `normal_cdf` |
| 3 | [Variography](03-variography/README.md) | Walker Lake | `variogram_map`, `experimental_variogram`, `Variogram` |
| 4 | [Ordinary kriging](04-kriging/README.md) | Walker Lake | `BlockModel`, `Search`, `OrdinaryKriging`, cross-validation |

Run all with `mise run examples`; data is downloaded to `examples/data` on first use.
