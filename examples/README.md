# Examples

Worked examples with the Python API on open datasets from [gstvschlz/datasets](https://github.com/gstvschlz/datasets).
Tutorials follow one deposit from data to a result; topics show one feature each.

## Tutorials

| # | Tutorial | Dataset | Steps |
|---|---|---|---|
| 1 | [A coal seam in a lease](tutorials/01-coal-seam-in-a-lease/README.md) | Coal seam thickness | The lease and its grid; Declustering; An anisotropic variogram; Ordinary or universal kriging; Volume and tonnes; How sure is the total?; Classification by data spacing |
| 2 | [From drill holes to a classified model](tutorials/02-drillholes-to-classified-model/README.md) | Stacked sulphide lenses | Drill holes; Samples inside each lens; Declustering and capping; Zn variogram of lens 1; A sub-blocked model; Kriging in passes; Simulation at block support; Validation; Classification; Tonnes and metal |
| 3 | [Iron ore: several grades, one closure, reconciled](tutorials/03-iron-ore-multivariate/README.md) | Iron formation plateau | Composites by lithology; One closure; Ore domains of the model; Log-ratios, PPMT and simulation; Do the blocks still close?; Reconciliation with the blastholes; A bench in plan |
| 4 | [A laterite profile: horizons, then grades](tutorials/04-laterite-profile/README.md) | Nickel laterite profile | Horizon contacts; Is the 50 m mesh enough?; A layered block model; Ni and Co by horizon; Grades into the blocks; Tonnes and grade by horizon |

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
| 9 | [Contacts](topics/09-contacts/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `contact`, `plot.contact` |
| 10 | [Swaths](topics/10-swaths/README.md) | Phosphate weathering profile | `merge_intervals`, `Drillholes`, `describe_by`, `swath`, `plot.swath` |
| 11 | [Categories](topics/11-categories/README.md) | Tailings reprocessing | `Drillholes`, `Categories`, `plot.proportions`, `plot.category_swath` |
| 12 | [Data spacing](topics/12-data-spacing/README.md) | Coal seam thickness | `data_spacing`, `plot.histogram`, `hole_distance`, `plot.section` |
| 13 | [Correlations](topics/13-correlations/README.md) | Iron formation plateau | `merge_intervals`, `Drillholes`, `correlation`, `plot.correlation`, `plot.scatter_matrix`, `plot.completeness`, `plot.conditional`, `h_scatter` |
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
| 36 | [Sequential Gaussian simulation](topics/36-sgs/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search` |
| 37 | [Simulation at block support](topics/37-simulation-at-block-support/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `BlockKriging`, `localize` |
| 38 | [Turning bands](topics/38-turning-bands/README.md) | Walker Lake | `BlockModel`, `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `SGS`, `Search`, `TurningBands` |
| 39 | [Simulation with a trend](topics/39-simulation-with-trend/README.md) | Walker Lake | `cell_declustering`, `BlockModel`, `MovingAverage`, `Search`, `StepwiseConditional`, `NormalScore`, `experimental_variogram`, `Variogram`, `SGS` |
| 42 | [Simulation methods](topics/42-grades-in-simulated-rocks/README.md) | Jura | `Categories`, `experimental_variogram`, `Variogram`, `SIS`, `Search`, `Plurigaussian`, `plot.category_colors`, `plot.category_legend`, `NormalScore`, `SGS` |
| 43 | [Multivariate simulation](topics/43-multivariate-simulation/README.md) | Porphyry geometallurgy | `PointSet`, `cell_declustering`, `BlockModel`, `Search`, `PPMT`, `PCA`, `experimental_variogram`, `MultivariateSimulation`, `TurningBands` |
| | **Validation** | | |
| 44 | [Model checks](topics/44-model-checks/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `global_bias`, `validate_model`, `plot.cdf`, `swath`, `plot.swath` |
| 45 | [Cross-validation](topics/45-cross-validation/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation` |
| 46 | [Kriging diagnostics](topics/46-kriging-diagnostics/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `BlockKriging` |
| 47 | [Search calibration](topics/47-search-calibration/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `HermiteAnamorphosis`, `calibrate_search` |
| 48 | [Classification](topics/48-classification/README.md) | Coal seam thickness | `data_spacing`, `classify`, `experimental_variogram`, `Variogram`, `Search`, `BlockKriging`, `smooth_classes`, `Categories`, `plot.category_colors`, `plot.category_legend` |
| | **Modeling** | | |
| 49 | [Solids](topics/49-solids/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `plot.section`, `plot.slab`, `block_shell` |
| 50 | [Sub-blocks](topics/50-sub-blocks/README.md) | Stacked sulphide lenses | `BlockModel`, `plot.slab`, `grid_surface`, `Categories`, `plot.section`, `Drillholes`, `Search`, `InverseDistance` |
| 51 | [Mesh files](topics/51-mesh-files/README.md) | Vein gold grade control | `read_mesh`, `plot.slab`, `write_mesh`, `Mesh` |
| 52 | [Grade shells](topics/52-grade-shells/README.md) | Iron formation plateau | `Drillholes`, `ImplicitModel`, `plot.slab`, `BlockModel` |
| 53 | [Contact surfaces](topics/53-contact-surfaces/README.md) | Stacked sulphide lenses | `Drillholes`, `Variogram`, `ImplicitModel`, `BlockModel`, `plot.slab` |
| 54 | [Structural data](topics/54-structural-data/README.md) |  | `BlockModel`, `ImplicitModel`, `plot.scatter` |
| 55 | [Layered surfaces](topics/55-layered-surfaces/README.md) | Phosphate weathering profile | `grid_surface`, `plot.section`, `Drillholes`, `InverseDistance`, `Search`, `BlockModel`, `Categories`, `plot.slab`, `plot.category_legend` |
| 56 | [Polygons](topics/56-polygons/README.md) | Coal seam thickness | `point_in_polygon`, `polygon_distance`, `plot.section`, `assign_domain`, `Categories`, `plot.category_colors`, `plot.category_legend`, `PolygonSelector` |
| | **I/O and scale** | | |
| 57 | [Storing containers in Parquet](topics/57-parquet/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `BlockModel`, `OrdinaryKriging`, `Search`, `write_parquet`, `write_csv`, `read_parquet`, `Categories`, `CategoricalIndicatorKriging`, `CategoricalIndicatorSummary`, `Polylines` |
| 58 | [Shapefiles and GeoTIFF](topics/58-gis-formats/README.md) | Soil geochemistry survey | `write_shapefile`, `read_shapefile`, `Polylines`, `Categories`, `BlockModel`, `write_geotiff`, `read_geotiff` |
| 59 | [Models larger than memory](topics/59-large-models/README.md) | Drillholes (legacy) | `cell_declustering`, `convex_hull`, `BlockModel`, `write_parquet`, `BlockModelFile`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `NormalScore`, `Variogram`, `SimpleKriging`, `map_blocks`, `TurningBands`, `plot.uncertain` |
| 60 | [3D views](topics/60-3d-views/README.md) | Drillholes (legacy) | `PointSet`, `convex_hull`, `plot3d.to_pyvista`, `plot3d.plot`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.slices` |
| | **More methods** | | |
| 61 | [Despiking](topics/61-despiking/README.md) | Soil geochemistry survey | `despike`, `NormalScore` |
| 62 | [Weight declustering](topics/62-weight-declustering/README.md) | Coal seam thickness | `cell_declustering`, `experimental_variogram`, `Search`, `NearestNeighbor`, `InverseDistance`, `OrdinaryKriging`, `weight_declustering` |
| 63 | [Smooth trend](topics/63-smooth-trend/README.md) | Coal seam thickness | `cell_declustering`, `detrend` |
| 64 | [Categorical indicator kriging](topics/64-categorical-indicator-kriging/README.md) | Stacked sulphide lenses | `Drillholes`, `Categories`, `cell_declustering`, `Variogram`, `Search`, `CategoricalIndicatorKriging`, `BlockModel`, `plot.section`, `plot.category_legend` |
| 65 | [Block model from extents](topics/65-block-model-from-extents/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel` |
| 66 | [Result plots](topics/66-result-plots/README.md) | Walker Lake, Nickel laterite profile | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation`, `MultipleIndicatorKriging`, `BlockModel`, `BlockKriging`, `cell_declustering`, `grade_tonnage`, `compare_models`, `plot.grade_tonnage`, `merge_intervals`, `Drillholes`, `contact`, `plot.contact` |
| 67 | [Variogram sets](topics/67-variogram-sets/README.md) | Jura, Walker Lake | `experimental_variograms`, `Coregionalization`, `plot.variograms`, `BlockModel`, `experimental_variogram` |
| 68 | [Multigaussian kriging](topics/68-multigaussian-kriging/README.md) | Walker Lake | `Variogram`, `cell_declustering`, `NormalScore`, `despike`, `experimental_variogram`, `BlockModel`, `MultigaussianKriging`, `Search`, `plot.scatter` |
| 69 | [Capping transform](topics/69-capping-transform/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `cell_declustering`, `Capping`, `describe_by`, `capping_report`, `plot.probability`, `BlockModel`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `NormalScore` |

Each page alternates text, collapsed Python and its results. `mise run examples` reruns every script
and rewrites the pages; `cs.datasets` downloads the data once and caches it. `render.py --index` writes this file.
