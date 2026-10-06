# Examples

Worked examples with the Python API on open datasets from [gstvschlz/datasets](https://github.com/gstvschlz/datasets).

## first steps

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [quick tour](01-first-steps/01-quick-tour/README.md) | Walker Lake | `cell_declustering`, `describe`, `plot.histogram`, `Capping`, `experimental_variogram`, `Variogram`, `plot.variogram`, `BlockModel`, `Search`, `OrdinaryKriging`, `plot.cross_validation`, `swath`, `plot.swath` |
| 2 | [block models](01-first-steps/02-block-models/README.md) | Synthetic | `BlockModel`, `Polylines`, `write_csv`, `write_parquet`, `read_parquet` |
| 3 | [saving and loading](01-first-steps/03-saving-and-loading/README.md) | Walker Lake | `cell_declustering`, `Capping`, `NormalScore`, `Variogram`, `experimental_variogram`, `Search`, `BlockModel`, `SimpleKriging`, `Declustering` |
| 4 | [storing containers in parquet](01-first-steps/04-parquet/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `BlockModel`, `OrdinaryKriging`, `Search`, `write_parquet`, `write_csv`, `read_parquet`, `Categories`, `CategoricalIndicatorKriging`, `CategoricalIndicatorSummary`, `Polylines` |

## data and geometry

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [checking drill holes](02-data-and-geometry/01-check-drillholes/README.md) | Stacked sulphide lenses | `check_drillholes`, `Table`, `fix_drillholes`, `Drillholes` |
| 2 | [desurveying drill holes](02-data-and-geometry/02-desurvey/README.md) | Stacked sulphide lenses | `Drillholes`, `Table` |
| 3 | [compositing](02-data-and-geometry/03-compositing/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes` |
| 4 | [duplicates](02-data-and-geometry/04-duplicates/README.md) | Stacked sulphide lenses | `duplicates`, `Table`, `check_drillholes`, `fix_drillholes`, `Drillholes` |
| 5 | [solids](02-data-and-geometry/05-solids/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `plot.section`, `plot.slab`, `block_shell` |
| 6 | [sub-blocks](02-data-and-geometry/06-sub-blocks/README.md) | Stacked sulphide lenses | `BlockModel`, `plot.slab`, `grid_surface`, `Categories`, `plot.section`, `Drillholes`, `Search`, `InverseDistance` |
| 7 | [mesh files](02-data-and-geometry/07-mesh-files/README.md) | Vein gold grade control | `read_mesh`, `plot.slab`, `write_mesh`, `Mesh` |
| 8 | [polygons](02-data-and-geometry/08-polygons/README.md) | Coal seam thickness | `point_in_polygon`, `polygon_distance`, `plot.section`, `assign_domain`, `Categories`, `plot.category_colors`, `plot.category_legend`, `Polylines`, `PolygonSelector` |
| 9 | [shapefiles, geopackage and GeoTIFF](02-data-and-geometry/09-gis-formats/README.md) | Soil geochemistry survey | `write_shapefile`, `read_shapefile`, `Polylines`, `write_geopackage`, `read_geopackage`, `Categories`, `BlockModel`, `write_geotiff`, `read_geotiff` |
| 10 | [models larger than memory](02-data-and-geometry/10-large-models/README.md) | Iron formation plateau | `Drillholes`, `BlockModel`, `write_parquet`, `BlockModelFile`, `detrend`, `map_blocks`, `NormalScore`, `experimental_variogram`, `Variogram`, `TurningBands`, `Search`, `plot.section` |
| 11 | [3D views](02-data-and-geometry/11-3d-views/README.md) | Stacked sulphide lenses | `Drillholes`, `plot3d.Scene`, `BlockModel`, `Search`, `InverseDistance` |
| 12 | [block model from extents](02-data-and-geometry/12-block-model-from-extents/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel` |
| 13 | [runs and strip logs](02-data-and-geometry/13-runs-and-strip-logs/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `Categories`, `plot.strip_log` |
| 14 | [domain cleanup](02-data-and-geometry/14-domain-cleanup/README.md) | Stacked sulphide lenses | `Drillholes`, `Categories`, `cell_declustering`, `Variogram`, `Search`, `CategoricalIndicatorKriging`, `BlockModel`, `remove_small_units`, `plot.section`, `contact_distance`, `buffer_domains`, `plot.category_legend` |
| 15 | [mesh-crossing interval splits](02-data-and-geometry/15-mesh-interval-splits/README.md) | Stacked sulphide lenses | `Drillholes`, `merge_intervals`, `describe_by` |
| 16 | [stepped sections](02-data-and-geometry/16-stepped-sections/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Variogram`, `Search`, `BlockModel`, `OrdinaryKriging`, `plot.section`, `plot.fence` |
| 17 | [hole traces with deviation flags](02-data-and-geometry/17-hole-traces/README.md) | Stacked sulphide lenses | `check_drillholes`, `Drillholes`, `plot.holes` |
| 18 | [outlines](02-data-and-geometry/18-outlines/README.md) | Stacked sulphide lenses | `PointSet`, `data_spacing`, `outline` |
| 19 | [topography from points](02-data-and-geometry/19-topography/README.md) | Stacked sulphide lenses | `PointSet`, `topography`, `plot.section`, `Drillholes`, `snap_to_surface` |
| 20 | [sections with holes](02-data-and-geometry/20-sections-with-holes/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Variogram`, `Search`, `BlockModel`, `OrdinaryKriging`, `plot.section` |
| 21 | [live scene](02-data-and-geometry/21-live-scene/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.Scene` |
| 22 | [meshes and points in a scene](02-data-and-geometry/22-meshes-and-points/README.md) | Stacked sulphide lenses | `Drillholes`, `PointSet`, `topography`, `plot3d.Scene` |
| 23 | [filtering block models](02-data-and-geometry/23-filtering-block-models/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.Scene` |
| 24 | [interactive sections](02-data-and-geometry/24-interactive-sections/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.Scene` |
| 25 | [units](02-data-and-geometry/25-units/README.md) | Stacked sulphide lenses | `write_csv`, `Table`, `units`, `InvalidInput`, `plot.histogram`, `Drillholes`, `BlockModel`, `OrdinaryKriging`, `Variogram`, `Search`, `grade_tonnage`, `MultipleIndicatorKriging`, `Capping`, `validate_model` |

## exploratory analysis

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [paired data](03-exploratory-analysis/01-paired-data/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `pairs`, `paired_bias`, `plot.paired_bias`, `plot.qq`, `plot.scatter` |
| 2 | [statistics by domain](03-exploratory-analysis/02-statistics-by-domain/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `describe_by`, `plot.boxplot`, `plot.cdf`, `plot.qq` |
| 3 | [declustering](03-exploratory-analysis/03-declustering/README.md) | Coal seam thickness | `cell_declustering`, `plot.declustering`, `polygon_declustering` |
| 4 | [weight declustering](03-exploratory-analysis/04-weight-declustering/README.md) | Coal seam thickness | `cell_declustering`, `experimental_variogram`, `Search`, `NearestNeighbor`, `InverseDistance`, `OrdinaryKriging`, `weight_declustering` |
| 5 | [top cuts](03-exploratory-analysis/05-top-cuts/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `cell_declustering`, `capping`, `describe_by`, `plot.probability`, `capping_report` |
| 6 | [despiking](03-exploratory-analysis/06-despiking/README.md) | Soil geochemistry survey | `despike`, `NormalScore` |
| 7 | [contacts](03-exploratory-analysis/07-contacts/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `contact`, `plot.contact` |
| 8 | [soft-boundary statistics](03-exploratory-analysis/08-soft-boundary-statistics/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `soft_boundary` |
| 9 | [swaths](03-exploratory-analysis/09-swaths/README.md) | Phosphate weathering profile | `merge_intervals`, `Drillholes`, `describe_by`, `swath`, `plot.swath` |
| 10 | [data spacing](03-exploratory-analysis/10-data-spacing/README.md) | Coal seam thickness, Iron formation plateau | `data_spacing`, `plot.section`, `plot.histogram`, `Drillholes`, `Search` |
| 11 | [correlations](03-exploratory-analysis/11-correlations/README.md) | Iron formation plateau | `merge_intervals`, `Drillholes`, `correlation`, `plot.correlation`, `plot.scatter_matrix`, `plot.completeness`, `plot.conditional`, `h_scatter` |
| 12 | [spatial bootstrap](03-exploratory-analysis/12-spatial-bootstrap/README.md) | Coal seam thickness | `cell_declustering`, `despike`, `NormalScore`, `Variogram`, `experimental_variogram`, `spatial_bootstrap` |
| 13 | [reference distributions](03-exploratory-analysis/13-reference-distributions/README.md) | Vein gold grade control, Porphyry geometallurgy | `merge_intervals`, `Drillholes`, `cell_declustering`, `KernelDensity`, `NormalScore`, `GaussianMixture`, `GaussianImputer` |
| 14 | [cleaning steps](03-exploratory-analysis/14-cleaning-steps/README.md) | Stacked sulphide lenses | `Pipeline`, `RenameColumns`, `ToNull`, `Replace`, `ToNumber`, `DropNull`, `plot.histogram` |

## transforms

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [normal-score transform](04-transforms/01-normal-score/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `normal_cdf`, `plot.probability` |
| 2 | [censored normal-score transform](04-transforms/02-censored-normal-score/README.md) | Tailings reprocessing, Soil geochemistry survey | `NormalScore`, `experimental_variogram`, `Search`, `ExternalDriftKriging` |
| 3 | [capping transform](04-transforms/03-capping-transform/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `cell_declustering`, `Capping`, `describe_by`, `capping_report`, `plot.probability`, `BlockModel`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `NormalScore` |
| 4 | [compositional data](04-transforms/04-compositional/README.md) | Porphyry geometallurgy | `closure`, `composition_center`, `variation_matrix`, `total_variance`, `PointSet`, `Pipeline`, `ILR`, `PPMT`, `plot.ternary`, `plot.biplot` |
| 5 | [multivariate transforms](04-transforms/05-multivariate-transforms/README.md) | Porphyry geometallurgy | `PCA`, `MAF`, `StepwiseConditional`, `PPMT` |
| 6 | [imputation](04-transforms/06-imputation/README.md) | Stacked sulphide lenses | `merge_intervals`, `GaussianImputer` |
| 7 | [spatial imputation](04-transforms/07-spatial-imputation/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `NormalScore`, `experimental_variogram`, `plot.variogram`, `GaussianImputer` |
| 8 | [smooth trend](04-transforms/08-smooth-trend/README.md) | Coal seam thickness | `cell_declustering`, `detrend` |
| 9 | [unfolding](04-transforms/09-unfolding/README.md) | Nickel laterite profile | `Drillholes`, `BlockModel`, `InverseDistance`, `Search`, `grid_surface`, `Unfold`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `plot.section`, `plot.slab` |
| 10 | [target distribution correction](04-transforms/10-target-distribution-correction/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `check_realizations`, `correct_distribution`, `plot.histogram_reproduction`, `plot.section`, `KernelDensity` |
| 11 | [pipeline](04-transforms/11-pipeline/README.md) | Walker Lake | `cell_declustering`, `Pipeline`, `Capping`, `NormalScore`, `PPMT` |

## spatial continuity

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [experimental variograms](05-spatial-continuity/01-experimental-variograms/README.md) | Walker Lake | `variogram_map`, `experimental_variogram`, `plot.variogram` |
| 2 | [variogram fitting](05-spatial-continuity/02-variogram-fitting/README.md) | Walker Lake | `variogram_map`, `experimental_variogram`, `plot.variogram`, `Variogram` |
| 3 | [variogram sets](05-spatial-continuity/03-variogram-sets/README.md) | Jura, Walker Lake | `experimental_variograms`, `Coregionalization`, `plot.variograms`, `BlockModel`, `experimental_variogram` |
| 4 | [variogram volume](05-spatial-continuity/04-variogram-volume/README.md) | Stacked sulphide lenses | `Drillholes`, `variogram_volume`, `plot.variogram_volume`, `experimental_variogram`, `Variogram`, `plot.variogram` |
| 5 | [downhole nugget](05-spatial-continuity/05-downhole-nugget/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `experimental_variogram` |
| 6 | [madogram](05-spatial-continuity/06-madogram/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `experimental_variogram`, `dissemination` |
| 7 | [coregionalization](05-spatial-continuity/07-coregionalization/README.md) | Jura | `experimental_variogram`, `Coregionalization`, `plot.variogram` |
| 8 | [intrinsic coregionalization](05-spatial-continuity/08-intrinsic-coregionalization/README.md) | Jura | `experimental_variograms`, `Coregionalization`, `plot.variogram`, `Search`, `Cokriging` |
| 9 | [local variogram parameters](05-spatial-continuity/09-local-variogram-parameters/README.md) | Walker Lake | `BlockModel`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `Search`, `LocalAnisotropy`, `local_variogram_parameters` |
| 10 | [from variogram to search plan](05-spatial-continuity/10-search-plans/README.md) | Stacked sulphide lenses | `Drillholes`, `variogram_volume`, `experimental_variogram`, `Variogram`, `plot.variogram`, `plot.slab`, `BlockModel`, `OrdinaryKriging`, `Search`, `hole_distance`, `plot.section` |
| 11 | [nugget inference](05-spatial-continuity/11-nugget-inference/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `experimental_variogram`, `pairs` |

## kriging

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [ordinary kriging](06-kriging/01-ordinary-kriging/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `OrdinaryKriging`, `plot.scatter` |
| 2 | [simple estimators](06-kriging/02-simple-estimators/README.md) | Walker Lake | `BlockModel`, `experimental_variogram`, `Variogram`, `Search`, `NearestNeighbor`, `InverseDistance`, `MovingAverage`, `OrdinaryKriging`, `compare_models` |
| 3 | [universal kriging](06-kriging/03-universal-kriging/README.md) | Coal seam thickness | `detrend`, `experimental_variogram`, `plot.variogram`, `Search`, `OrdinaryKriging`, `UniversalKriging` |
| 4 | [external drift kriging](06-kriging/04-external-drift-kriging/README.md) | Soil geochemistry survey | `experimental_variogram`, `Search`, `OrdinaryKriging`, `ExternalDriftKriging` |
| 5 | [block kriging](06-kriging/05-block-kriging/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `Search`, `BlockModel`, `BlockKriging`, `OrdinaryKriging` |
| 6 | [search](06-kriging/06-search/README.md) | Walker Lake | `BlockModel`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `Search` |
| 7 | [search calibration](06-kriging/07-search-calibration/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `HermiteAnamorphosis`, `calibrate_search` |
| 8 | [high-grade restriction](06-kriging/08-high-grade-restriction/README.md) | Walker Lake | `BlockModel`, `experimental_variogram`, `Variogram`, `BlockKriging`, `Search`, `HighGrade` |
| 9 | [cokriging](06-kriging/09-cokriging/README.md) | Jura | `experimental_variogram`, `Coregionalization`, `Search`, `OrdinaryKriging`, `Cokriging`, `BlockModel`, `LocalAnisotropy` |
| 10 | [indicator kriging](06-kriging/10-indicator-kriging/README.md) | Jura | `experimental_variogram`, `Search`, `IndicatorKriging` |
| 11 | [multiple indicator kriging](06-kriging/11-multiple-indicator-kriging/README.md) | Jura | `Search`, `cell_declustering`, `experimental_variogram`, `MultipleIndicatorKriging`, `OrdinaryKriging`, `BlockModel` |
| 12 | [multigaussian kriging](06-kriging/12-multigaussian-kriging/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `despike`, `experimental_variogram`, `Variogram`, `BlockModel`, `MultigaussianKriging`, `Search`, `plot.scatter` |
| 13 | [soft boundaries](06-kriging/13-soft-boundaries/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `contact`, `plot.contact`, `NormalScore`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `SGS` |
| 14 | [locally varying anisotropy](06-kriging/14-local-anisotropy/README.md) | Walker Lake | `BlockModel`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `Search`, `LocalAnisotropy`, `plot.directions`, `cell_declustering`, `NormalScore`, `SGS` |

## categories and domains

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [categories](07-categories-and-domains/01-categories/README.md) | Tailings reprocessing | `Drillholes`, `Categories`, `plot.proportions`, `plot.category_swath` |
| 2 | [domain change tables](07-categories-and-domains/02-domain-change-tables/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Categories`, `cell_declustering`, `Variogram`, `Search`, `CategoricalIndicatorKriging`, `BlockModel`, `OrdinaryKriging`, `remove_small_units`, `domain_change`, `plot.domain_change` |
| 3 | [along-hole transition matrix and MDS](07-categories-and-domains/03-transition-matrix-mds/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Categories`, `transition_matrix`, `plot.domain_change`, `plot.transition_mds` |
| 4 | [local proportions](07-categories-and-domains/04-local-proportions/README.md) | Tailings reprocessing | `Drillholes`, `Categories`, `cell_declustering`, `vertical_proportions`, `plot.category_legend`, `detrend`, `BlockModel`, `combine_proportions`, `Variogram`, `Search`, `SIS`, `plot.category_colors`, `check_realizations`, `plot.histogram_reproduction` |
| 5 | [categorical indicator kriging](07-categories-and-domains/05-categorical-indicator-kriging/README.md) | Stacked sulphide lenses | `Drillholes`, `Categories`, `cell_declustering`, `Variogram`, `Search`, `CategoricalIndicatorKriging`, `BlockModel`, `plot.section`, `plot.uncertain` |
| 6 | [sequential indicator simulation](07-categories-and-domains/06-sis/README.md) | Jura | `Categories`, `experimental_variogram`, `Variogram`, `SIS`, `Search`, `plot.category_colors`, `plot.category_legend` |
| 7 | [plurigaussian simulation](07-categories-and-domains/07-plurigaussian/README.md) | Jura | `Categories`, `Variogram`, `Plurigaussian`, `plot.category_colors`, `plot.category_legend`, `experimental_variogram` |
| 8 | [grades in simulated rock types](07-categories-and-domains/08-grades-in-simulated-rocks/README.md) | Jura | `Categories`, `experimental_variogram`, `Variogram`, `SIS`, `Search`, `NormalScore`, `SGS` |

## stochastic simulation

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [sequential gaussian simulation](08-stochastic-simulation/01-sgs/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search` |
| 2 | [SGS along a shared path](08-stochastic-simulation/02-sgs-shared-path/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `check_realizations`, `plot.histogram_reproduction`, `plot.variogram_reproduction` |
| 3 | [simulation at block support](08-stochastic-simulation/03-simulation-at-block-support/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `plot.grade_tonnage`, `BlockKriging`, `localize` |
| 4 | [turning bands](08-stochastic-simulation/04-turning-bands/README.md) | Walker Lake | `BlockModel`, `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `SGS`, `Search`, `TurningBands` |
| 5 | [simulation with a trend](08-stochastic-simulation/05-simulation-with-trend/README.md) | Walker Lake | `cell_declustering`, `BlockModel`, `MovingAverage`, `Search`, `StepwiseConditional`, `NormalScore`, `experimental_variogram`, `Variogram`, `SGS` |
| 6 | [multivariate simulation](08-stochastic-simulation/06-multivariate-simulation/README.md) | Porphyry geometallurgy | `PointSet`, `cell_declustering`, `BlockModel`, `Search`, `PPMT`, `PCA`, `experimental_variogram`, `MultivariateSimulation`, `TurningBands` |
| 7 | [collocated cosimulation](08-stochastic-simulation/07-collocated-cosimulation/README.md) | Jura | `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `SGS`, `check_realizations`, `plot.histogram_reproduction`, `plot.variogram_reproduction`, `DSS` |
| 8 | [direct sequential simulation](08-stochastic-simulation/08-direct-sequential-simulation/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `DSS`, `SGS`, `correct_distribution` |

## recoverable resources

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [discrete gaussian model](09-recoverable-resources/01-discrete-gaussian-model/README.md) | Walker Lake | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `change_of_support` |
| 2 | [uniform conditioning](09-recoverable-resources/02-uniform-conditioning/README.md) | Walker Lake | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `change_of_support`, `Search`, `BlockModel`, `BlockKriging`, `UniformConditioning` |
| 3 | [MIK localization](09-recoverable-resources/03-mik-localization/README.md) | Walker Lake | `cell_declustering`, `experimental_variogram`, `Variogram`, `Search`, `BlockModel`, `BlockKriging`, `MultipleIndicatorKriging` |
| 4 | [disjunctive kriging](09-recoverable-resources/04-disjunctive-kriging/README.md) | Walker Lake | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `BlockModel`, `DisjunctiveKriging`, `Search` |

## checking models

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [model checks](10-checking-models/01-model-checks/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `global_bias`, `validate_model`, `plot.cdf`, `swath`, `plot.swath` |
| 2 | [cross-validation](10-checking-models/02-cross-validation/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation` |
| 3 | [kriging diagnostics](10-checking-models/03-kriging-diagnostics/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `BlockKriging` |
| 4 | [realization checks](10-checking-models/04-realization-checks/README.md) | Walker Lake, Porphyry geometallurgy | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `check_realizations`, `plot.histogram_reproduction`, `plot.variogram_reproduction`, `PointSet`, `PPMT`, `TurningBands`, `MultivariateSimulation`, `plot.correlation_reproduction` |
| 5 | [classification](10-checking-models/05-classification/README.md) | Coal seam thickness | `data_spacing`, `classify`, `experimental_variogram`, `Variogram`, `Search`, `BlockKriging`, `smooth_classes`, `Categories`, `plot.category_colors`, `plot.category_legend` |
| 6 | [result plots](10-checking-models/06-result-plots/README.md) | Walker Lake, Nickel laterite profile | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation`, `MultipleIndicatorKriging`, `BlockModel`, `BlockKriging`, `cell_declustering`, `grade_tonnage`, `compare_models`, `plot.grade_tonnage`, `merge_intervals`, `Drillholes`, `contact`, `plot.contact` |
| 7 | [section validation plates](10-checking-models/07-section-validation-plates/README.md) | Stacked sulphide lenses | `BlockModel`, `Drillholes`, `InverseDistance`, `Search`, `plot.section`, `plot.slab`, `plot.scatter` |

## geological modeling

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [grade shells](11-geological-modeling/01-grade-shells/README.md) | Iron formation plateau | `Drillholes`, `ImplicitModel`, `plot.slab` |
| 2 | [contact surfaces](11-geological-modeling/02-contact-surfaces/README.md) | Stacked sulphide lenses | `Drillholes`, `Variogram`, `ImplicitModel`, `BlockModel`, `plot.slab` |
| 3 | [structural data](11-geological-modeling/03-structural-data/README.md) | Synthetic | `BlockModel`, `ImplicitModel`, `plot.scatter` |
| 4 | [layered surfaces](11-geological-modeling/04-layered-surfaces/README.md) | Phosphate weathering profile | `grid_surface`, `plot.section`, `Drillholes`, `InverseDistance`, `Search`, `BlockModel`, `Categories`, `plot.slab`, `plot.category_legend` |

## Case studies

| # | Example | Dataset | Steps |
|---|---|---|---|
| 1 | [A coal seam in a lease](12-case-studies/01-coal-seam-in-a-lease/README.md) | Coal seam thickness | The lease and its grid; Declustering; An anisotropic variogram; Ordinary or universal kriging; Volume and tonnes; How sure is the total?; Classification by data spacing |
| 2 | [From drill holes to a classified model](12-case-studies/02-drillholes-to-classified-model/README.md) | Stacked sulphide lenses | Drill holes; Samples inside each lens; Declustering and capping; Zn variogram of lens 1; A sub-blocked model; Kriging in passes; Simulation at block support; Validation; Classification; Tonnes and metal |
| 3 | [Iron ore: several grades, one closure, reconciled](12-case-studies/03-iron-ore-multivariate/README.md) | Iron formation plateau | Composites by lithology; One closure; Ore domains of the model; Log-ratios, PPMT and simulation; Do the blocks still close?; Reconciliation with the blastholes; A bench in plan |
| 4 | [A laterite profile: horizons, then grades](12-case-studies/04-laterite-profile/README.md) | Nickel laterite profile | Horizon contacts; Is the 50 m mesh enough?; A layered block model; Ni and Co by horizon; Grades into the blocks; Tonnes and grade by horizon |

## multiple-point statistics

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [SNESIM and its multigrid](13-multiple-point-statistics/01-snesim-multigrid/README.md) | Strebelle | `BlockModel`, `SNESIM` |
| 2 | [conditioning SNESIM](13-multiple-point-statistics/02-snesim-conditioning/README.md) | Strebelle | `BlockModel`, `SNESIM` |
| 3 | [continuous SNESIM](13-multiple-point-statistics/03-snesim-continuous/README.md) | F3 seismic | `BlockModel`, `SNESIM` |
| 4 | [rotation and affinity](13-multiple-point-statistics/04-snesim-rotation-affinity/README.md) | Strebelle, F3 seismic | `BlockModel`, `LocalAnisotropy`, `SNESIM` |
| 5 | [several training images](13-multiple-point-statistics/05-snesim-training-images-by-zone/README.md) | Strebelle | `BlockModel`, `object_training_image`, `SNESIM` |
| 6 | [soft data in SNESIM](13-multiple-point-statistics/06-snesim-soft-data/README.md) | Strebelle | `BlockModel`, `SNESIM` |
| 7 | [training images and their consistency](13-multiple-point-statistics/07-training-images/README.md) | Strebelle | `BlockModel`, `object_training_image`, `training_image_consistency` |
| 8 | [image quilting](13-multiple-point-statistics/08-image-quilting/README.md) | Strebelle | `BlockModel`, `ImageQuilting` |
| 9 | [continuous image quilting](13-multiple-point-statistics/09-image-quilting-continuous/README.md) | F3 seismic | `BlockModel`, `ImageQuilting`, `SNESIM` |
| 10 | [seismic volumes](13-multiple-point-statistics/10-seismic-volumes/README.md) | F3 seismic | `read_segy`, `BlockModel`, `object_training_image`, `write_segy`, `ImageQuilting` |

## Workflows

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Flag solid proportions in a block model](14-workflows/01-solid-proportions/README.md) | Vein gold grade control | `BlockModel`, `plot.section`, `Categories`, `plot.slab` |
| 2 | [Sub-block a model from solids](14-workflows/02-sub-blocks-from-solids/README.md) | Vein gold grade control | `BlockModel`, `Categories`, `plot.section`, `plot.slab`, `plot.category_legend` |
| 3 | [Find and fix degenerate solids](14-workflows/03-degenerate-solids/README.md) | Stacked sulphide lenses | `Mesh` |
| 4 | [Audit drill holes before modeling](14-workflows/04-drillhole-audit/README.md) | Stacked sulphide lenses | `check_drillholes`, `Table`, `fix_drillholes` |
| 5 | [Compare two estimates of one deposit](14-workflows/05-compare-two-estimates/README.md) | Walker Lake | `cell_declustering`, `experimental_variogram`, `Variogram`, `Search`, `BlockKriging`, `InverseDistance`, `BlockModel`, `global_bias`, `swath`, `plot.swath`, `compare_models`, `OrdinaryKriging` |
| 6 | [Grade–tonnage curves and the change in contained metal](14-workflows/06-grade-tonnage-and-metal/README.md) | Walker Lake | `experimental_variogram`, `Variogram`, `Search`, `BlockKriging`, `InverseDistance`, `BlockModel`, `grade_tonnage`, `compare_models`, `plot.grade_tonnage` |
| 7 | [Drill-hole spacing from virtual grids](14-workflows/07-drillhole-spacing-virtual-grids/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `TurningBands`, `Search`, `select_realizations`, `spacing_study`, `uncertainty_curve`, `required_spacing`, `plot.uncertainty_curve`, `planned_drillholes` |
| 8 | [Drill-hole spacing from a learning curve](14-workflows/08-drillhole-spacing-learning-curve/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `TurningBands`, `Search`, `data_spacing`, `uncertainty_curve`, `required_spacing`, `spacing_study` |

Each page alternates text, collapsed Python and its results. `mise run examples` reruns every script
and rewrites the pages; `bt.datasets` downloads the data once and caches it. `render.py --index` writes this file.
