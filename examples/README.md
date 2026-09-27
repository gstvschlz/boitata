# Examples

Worked examples with the Python API on open datasets from [gstvschlz/datasets](https://github.com/gstvschlz/datasets).
Tutorials follow one deposit from data to a result; topics show one feature each.

## Tutorials

| # | Tutorial | Dataset | Steps |
|---|---|---|---|
| 1 | [A coal seam in a lease](tutorials/01-coal-seam-in-a-lease/README.md) | Coal seam thickness | The lease and its grid; Declustering; An anisotropic variogram; Ordinary or universal kriging; Volume and tonnes; How sure is the total?; Classification by data spacing |
| 2 | [From drill holes to a classified model](tutorials/02-drillholes-to-classified-model/README.md) | Stacked sulphide lenses | Drill holes; Samples inside each lens; Declustering and capping; Zn variogram of lens 1; A sub-blocked model; Kriging in passes; Simulation at block support; Validation; Classification; Tonnes and metal |
| 3 | [Iron ore: several grades, one closure, reconciled](tutorials/03-iron-ore-multivariate/README.md) | Iron formation plateau | Composites by lithology; One closure; Ore domains of the model; Log-ratios, PPMT and simulation; Do the blocks still close?; Reconciliation with the blastholes; A bench in plan |

## Topics

| # | Topic | Dataset | Covers |
|---|---|---|---|
| | **Data** | | |
| 1 | [Checking drill holes](topics/01-check-drillholes/README.md) | Stacked sulphide lenses | `check_drillholes`, `Table`, `fix_drillholes`, `Drillholes` |
| 2 | [Desurveying drill holes](topics/02-desurvey/README.md) | Stacked sulphide lenses | `Drillholes`, `Table` |
| 3 | [Compositing](topics/03-compositing/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes` |
| 4 | [Duplicates](topics/04-duplicates/README.md) | Stacked sulphide lenses | `duplicates`, `Table`, `check_drillholes`, `fix_drillholes`, `Drillholes` |
| 5 | [Paired data](topics/05-paired-data/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `pairs`, `paired_bias`, `plot.paired_bias`, `plot.qq`, `plot.scatter` |
| 6 | [Statistics by domain](topics/06-statistics-by-domain/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `describe_by`, `plot.boxplot`, `plot.cdf`, `plot.qq` |
| 7 | [Declustering](topics/07-declustering/README.md) | Coal seam thickness | `cell_declustering`, `plot.declustering`, `polygon_declustering` |
| 8 | [Top cuts](topics/08-top-cuts/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `cell_declustering`, `capping`, `describe_by`, `plot.probability`, `capping_report` |
| 13 | [Exploratory data analysis](topics/13-correlations/README.md) | Drillholes (legacy) | `check_drillholes`, `fix_drillholes`, `merge_intervals`, `Drillholes`, `duplicates`, `cell_declustering`, `grade_tonnage`, `contact`, `swath`, `plot.swath`, `Categories`, `plot.proportions`, `plot.category_swath`, `data_spacing`, `plot.histogram`, `h_scatter`, `plot.scatter_matrix`, `plot.completeness`, `plot.correlation`, `plot.conditional` |
| | **Transforms** | | |
| 14 | [Normal-score transform](topics/14-normal-score/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `normal_cdf`, `plot.probability` |
| 15 | [Compositional data](topics/15-compositional/README.md) | Porphyry geometallurgy | `closure`, `ilr`, `PPMT`, `ilr_inverse` |
| 16 | [Multivariate transforms](topics/16-multivariate-transforms/README.md) | Porphyry geometallurgy | `PCA`, `MAF`, `StepwiseConditional`, `PPMT` |
| 17 | [Imputation](topics/17-imputation/README.md) | Stacked sulphide lenses | `merge_intervals`, `GaussianImputer` |
| | **Variography** | | |
| 18 | [Experimental variograms](topics/18-experimental-variograms/README.md) | Walker Lake | `variogram_map`, `experimental_variogram`, `plot.variogram` |
| 19 | [Variogram fitting](topics/19-variogram-fitting/README.md) | Walker Lake | `variogram_map`, `experimental_variogram`, `plot.variogram`, `Variogram` |
| 20 | [Downhole nugget](topics/20-downhole-nugget/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `experimental_variogram` |
| 21 | [Coregionalization](topics/21-coregionalization/README.md) | Jura | `experimental_variogram`, `Coregionalization`, `plot.variogram` |
| | **Estimation** | | |
| 22 | [Ordinary kriging](topics/22-ordinary-kriging/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `OrdinaryKriging`, `plot.scatter` |
| 23 | [Simple estimators](topics/23-simple-estimators/README.md) | Walker Lake | `BlockModel`, `experimental_variogram`, `Variogram`, `Search`, `NearestNeighbor`, `InverseDistance`, `MovingAverage`, `OrdinaryKriging`, `compare_models` |
| 24 | [Universal kriging](topics/24-universal-kriging/README.md) | Coal seam thickness | `detrend`, `experimental_variogram`, `plot.variogram`, `Search`, `OrdinaryKriging`, `UniversalKriging` |
| 25 | [Block kriging](topics/25-block-kriging/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `Search`, `BlockModel`, `BlockKriging`, `OrdinaryKriging` |
| 26 | [Search](topics/26-search/README.md) | Walker Lake | `BlockModel`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `Search` |
| 27 | [Cokriging](topics/27-cokriging/README.md) | Jura | `experimental_variogram`, `Coregionalization`, `Search`, `OrdinaryKriging`, `Cokriging` |
| 28 | [Indicator kriging](topics/28-indicator-kriging/README.md) | Jura | `experimental_variogram`, `Search`, `IndicatorKriging` |
| 29 | [Multiple indicator kriging](topics/29-multiple-indicator-kriging/README.md) | Jura | `Search`, `cell_declustering`, `experimental_variogram`, `MultipleIndicatorKriging`, `OrdinaryKriging`, `BlockModel` |
| 30 | [Soft boundaries](topics/30-soft-boundaries/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `contact`, `plot.contact`, `NormalScore`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `SGS` |
| 31 | [Locally varying anisotropy](topics/31-local-anisotropy/README.md) | Walker Lake | `BlockModel`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `Search`, `LocalAnisotropy`, `plot.directions`, `cell_declustering`, `NormalScore`, `SGS` |
| | **Change of support** | | |
| 32 | [Discrete Gaussian model](topics/32-discrete-gaussian-model/README.md) | Walker Lake | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `change_of_support` |
| 33 | [Uniform conditioning](topics/33-uniform-conditioning/README.md) | Walker Lake | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `change_of_support`, `Search`, `BlockModel`, `BlockKriging`, `UniformConditioning` |
| 34 | [MIK localization](topics/34-mik-localization/README.md) | Walker Lake | `cell_declustering`, `experimental_variogram`, `Variogram`, `Search`, `BlockModel`, `BlockKriging`, `MultipleIndicatorKriging` |
| 35 | [Disjunctive kriging](topics/35-disjunctive-kriging/README.md) | Walker Lake | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `BlockModel`, `DisjunctiveKriging`, `Search` |
| | **Simulation** | | |
| 36 | [Sequential Gaussian simulation](topics/36-sgs/README.md) | Walker Lake | `Variogram`, `cell_declustering`, `NormalScore`, `experimental_variogram`, `BlockModel`, `SGS`, `Search` |
| 37 | [Simulation at block support](topics/37-simulation-at-block-support/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `BlockKriging`, `localize` |
| 42 | [Simulation methods](topics/42-grades-in-simulated-rocks/README.md) | Walker Lake, Jura | `cell_declustering`, `Variogram`, `BlockModel`, `SGS`, `Search`, `TurningBands`, `MovingAverage`, `StepwiseConditional`, `experimental_variogram`, `Categories`, `SIS`, `Plurigaussian`, `plot.category_colors`, `plot.category_legend`, `NormalScore` |
| 43 | [Multivariate simulation](topics/43-multivariate-simulation/README.md) | Porphyry geometallurgy | `PointSet`, `cell_declustering`, `BlockModel`, `Search`, `PPMT`, `PCA`, `experimental_variogram`, `MultivariateSimulation`, `TurningBands` |
| | **Validation** | | |
| 44 | [Model checks](topics/44-model-checks/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `global_bias`, `validate_model`, `plot.cdf`, `swath`, `plot.swath` |
| 45 | [Cross-validation](topics/45-cross-validation/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation` |
| 46 | [Kriging diagnostics](topics/46-kriging-diagnostics/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `BlockKriging` |
| 47 | [Search calibration](topics/47-search-calibration/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `HermiteAnamorphosis`, `calibrate_search` |
| 48 | [Classification](topics/48-classification/README.md) | Coal seam thickness | `data_spacing`, `classify`, `experimental_variogram`, `Variogram`, `Search`, `BlockKriging`, `smooth_classes`, `Categories`, `plot.category_colors`, `plot.category_legend` |
| | **Modeling** | | |
| 49 | [Solids and block models](topics/49-solids/README.md) | Drillholes (legacy) | `Mesh`, `BlockModel`, `plot.section`, `plot.slab`, `block_shell`, `Search`, `InverseDistance`, `write_mesh`, `read_mesh` |
| 52 | [Implicit modeling](topics/52-grade-shells/README.md) | Drillholes (legacy) | `ImplicitModel`, `BlockModel`, `plot.slab` |
| 53 | [Geological modeling](topics/53-contact-surfaces/README.md) | Drillholes (legacy) | `Drillholes`, `Variogram`, `ImplicitModel`, `plot.slab`, `BlockModel`, `plot.scatter`, `grid_surface` |
| | **I/O and scale** | | |
| 57 | [Storing containers in Parquet](topics/57-parquet/README.md) | Walker Lake | `PointSet`, `Variogram`, `BlockModel`, `OrdinaryKriging`, `Search`, `write_parquet`, `write_csv`, `read_parquet`, `write_shapefile`, `read_shapefile`, `Polylines`, `write_geotiff`, `read_geotiff` |
| 59 | [Models larger than memory](topics/59-large-models/README.md) | Drillholes (legacy) | `cell_declustering`, `convex_hull`, `BlockModel`, `write_parquet`, `BlockModelFile`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `NormalScore`, `Variogram`, `SimpleKriging`, `map_blocks`, `TurningBands`, `plot.uncertain` |
| 60 | [3D views](topics/60-3d-views/README.md) | Drillholes (legacy) | `PointSet`, `convex_hull`, `plot3d.to_pyvista`, `plot3d.plot`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.slices` |
| | **More methods** | | |
| 61 | [Despiking](topics/61-despiking/README.md) | Soil geochemistry survey | `despike`, `NormalScore` |
| 62 | [Weight declustering](topics/62-weight-declustering/README.md) | Coal seam thickness | `cell_declustering`, `experimental_variogram`, `Search`, `NearestNeighbor`, `InverseDistance`, `OrdinaryKriging`, `weight_declustering` |
| 63 | [Smooth trend](topics/63-smooth-trend/README.md) | Coal seam thickness | `cell_declustering`, `detrend` |
| 64 | [Categorical indicator kriging](topics/64-categorical-indicator-kriging/README.md) | Stacked sulphide lenses | `Drillholes`, `Categories`, `cell_declustering`, `Variogram`, `Search`, `CategoricalIndicatorKriging`, `BlockModel`, `plot.section`, `plot.category_legend` |
| 65 | [Block model from extents](topics/65-block-model-from-extents/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel` |
| 66 | [Result plots](topics/66-result-plots/README.md) | Walker Lake, Nickel laterite profile | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation`, `MultipleIndicatorKriging`, `BlockModel`, `BlockKriging`, `cell_declustering`, `grade_tonnage`, `compare_models`, `plot.grade_tonnage`, `merge_intervals`, `Drillholes`, `contact`, `plot.contact` |
| 68 | [Multigaussian kriging](topics/68-multigaussian-kriging/README.md) | Walker Lake | `Variogram`, `cell_declustering`, `NormalScore`, `despike`, `experimental_variogram`, `BlockModel`, `MultigaussianKriging`, `Search`, `plot.scatter` |

Each page alternates text, collapsed Python and its results. `mise run examples` reruns every script
and rewrites the pages; `cs.datasets` downloads the data once and caches it. `render.py --index` writes this file.
