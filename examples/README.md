# Examples

Worked examples with the Python API on open datasets from [gstvschlz/datasets](https://github.com/gstvschlz/datasets).
Walker Lake has an exhaustive grid, so every result is checked against the truth.

| # | Topic | Dataset | Covers |
|---|---|---|---|
| 1 | [Data and declustering](01-data/README.md) | Walker Lake | `read_csv`, `PointSet`, `cell_declustering` |
| 2 | [Normal-score transform](02-normal-score/README.md) | Walker Lake | `NormalScore`, `normal_cdf` |
| 3 | [Variography](03-variography/README.md) | Walker Lake | `variogram_map`, `experimental_variogram`, `Variogram.fit_directional` |
| 4 | [Ordinary kriging](04-kriging/README.md) | Walker Lake | `BlockModel`, `Search`, `OrdinaryKriging`, cross-validation |
| 5 | [Sequential Gaussian simulation](05-simulation/README.md) | Walker Lake | `SGS`, `NormalScore` tails, `SimulationSummary` |
| 6 | [Drillholes](06-drillholes/README.md) | Drillholes | `Drillholes`, `merge_intervals`, compositing by domain |
| 7 | [Solids and block models](07-solids/README.md) | Drillholes | `Mesh`, `proportion`, `mask`, `block_shell` |
| 8 | [Cokriging and indicator kriging](08-cokriging/README.md) | Jura | `Coregionalization`, `Cokriging` collocated, `IndicatorKriging`, `MultipleIndicatorKriging` |
| 9 | [Change of support](09-change-of-support/README.md) | Walker Lake | `HermiteAnamorphosis`, `change_of_support`, `UniformConditioning`, `MultipleIndicatorKriging.localize`, `localize`, `DisjunctiveKriging` |
| 10 | [Compositional data](10-compositional/README.md) | Geomet porphyry 1 | `closure`, `ilr`, `PPMT` |
| 11 | [Locally varying anisotropy](11-local-anisotropy/README.md) | Walker Lake | `LocalAnisotropy.from_grid`, `smooth`, `predict(anisotropy=)`, `SGS` |
| 12 | [Estimation methods and search](12-estimation-methods/README.md) | Walker Lake | `NearestNeighbor`, `InverseDistance`, `UniversalKriging`, `BlockKriging`, `Search` ellipsoid |
| 13 | [Simulation methods](13-simulation-methods/README.md) | Walker Lake, Jura | `TurningBands`, `SIS`, `Plurigaussian` |
| 14 | [Storing containers in Parquet](14-parquet/README.md) | Walker Lake | `write_parquet`, `read_parquet`, polars |
| 15 | [Implicit modelling](15-implicit/README.md) | Drillholes | `ImplicitModel` RBF and GP, `isosurface(closed=True)` |
| 16 | [Exploratory data analysis](16-eda/README.md) | Drillholes | `duplicates`, `pairs`, `paired_bias`, `describe`, `plot.boxplot`, `plot.cdf`, `plot.qq`, `capping`, `contact`, `swath`, `h_scatter`, `correlation` |
| 17 | [Multivariate transforms](17-multivariate/README.md) | Geomet porphyry 1 | `PCA`, `MAF`, `StepwiseConditional`, `PPMT`, `MultivariateSimulation`, `GaussianImputer` |
| 18 | [Validation and classification](18-validation/README.md) | Walker Lake | `predict(diagnostics=True)`, `cross_validate(folds=)`, `global_bias`, `calibrate_search`, `with_search`, `classify`, `smooth_classes` |
| 19 | [Geological modelling](19-geological-model/README.md) | Drillholes, synthetic fold | `Drillholes.at` contacts, `ImplicitModel` kriging/RBF/GP, planes, lineations, gradients, GP variance |
| 20 | [From drill holes to a classified model](20-workflow/README.md) | Drillholes | the whole chain: composites, EDA, declustering, variograms, kriging checks, SGS risk, classification, Parquet |
| 21 | [Models larger than memory](21-large-models/README.md) | Drillholes | 20 M blocks: `BlockModelFile`, `map_blocks`, `TurningBands.simulate_to_parquet` |

Each page alternates text, collapsed Python and its results. `mise run examples` reruns every `example_NN.py` and rewrites the pages; `cs.datasets` downloads the data once and caches it.
