# Examples

Worked examples with the Python API on open datasets from [gstvschlz/datasets](https://github.com/gstvschlz/datasets).
Walker Lake has an exhaustive grid, so every result is checked against the truth.

| # | Topic | Dataset | Covers |
|---|---|---|---|
| 1 | [Data and declustering](01-data/README.md) | Walker Lake | `read_csv`, `PointSet`, `cell_declustering` |
| 2 | [Normal-score transform](02-normal-score/README.md) | Walker Lake | `NormalScore`, `normal_cdf` |
| 3 | [Variography](03-variography/README.md) | Walker Lake | `variogram_map`, `experimental_variogram`, `Variogram` |
| 4 | [Ordinary kriging](04-kriging/README.md) | Walker Lake | `BlockModel`, `Search`, `OrdinaryKriging`, cross-validation |
| 5 | [Sequential Gaussian simulation](05-simulation/README.md) | Walker Lake | `SGS`, `NormalScore` tails, `probability_above` |
| 6 | [Drillholes](06-drillholes/README.md) | Drillholes | `Drillholes`, `merge_intervals`, compositing by domain |
| 7 | [Solids and block models](07-solids/README.md) | Drillholes | `Mesh`, `proportion`, `mask`, `block_shell` |
| 8 | [Cokriging and indicator kriging](08-cokriging/README.md) | Jura | `Coregionalization`, `Cokriging` collocated, `IndicatorKriging` |
| 9 | [Change of support](09-change-of-support/README.md) | Walker Lake | `HermiteAnamorphosis`, `change_of_support`, `DisjunctiveKriging` |
| 10 | [Compositional data](10-compositional/README.md) | Geomet porphyry 1 | `closure`, `ilr`, `PPMT` |

Run all with `mise run examples`; data is downloaded to `examples/data` on first use.
