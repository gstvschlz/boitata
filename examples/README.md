# Examples

Worked examples with the Python API on the synthetic datasets that `bt.datasets` generates on the call.

## Getting started

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Quick tour](01-getting-started/01-quick-tour/README.md) | Clustered metal survey | `cell_declustering`, `describe`, `plot.histogram`, `Capping`, `experimental_variogram`, `Variogram`, `plot.variogram`, `BlockModel`, `Search`, `OrdinaryKriging`, `compare`, `plot.cross_validation`, `swath`, `plot.swath` |
| 2 | [Block models](01-getting-started/02-block-models/README.md) | Synthetic | `BlockModel`, `Polylines`, `write_csv`, `write_parquet`, `read_parquet` |
| 3 | [Saving and loading](01-getting-started/03-saving-and-loading/README.md) | Clustered metal survey | `cell_declustering`, `Capping`, `NormalScore`, `Variogram`, `experimental_variogram`, `Search`, `BlockModel`, `SimpleKriging`, `Declustering` |
| 4 | [Storing containers in parquet](01-getting-started/04-parquet/README.md) | Clustered metal survey | `experimental_variogram`, `Variogram`, `BlockModel`, `OrdinaryKriging`, `Search`, `write_parquet`, `write_csv`, `read_parquet`, `Categories`, `CategoricalIndicatorKriging`, `CategoricalIndicatorSummary`, `Polylines` |

## Drill holes and samples

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Checking drill holes](02-drillholes-and-samples/01-check-drillholes/README.md) | Stacked sulphide lenses | `check_drillholes`, `Table`, `fix_drillholes`, `Drillholes` |
| 2 | [Audit drill holes before modeling](02-drillholes-and-samples/02-drillhole-audit/README.md) | Stacked sulphide lenses | `check_drillholes`, `Table`, `fix_drillholes` |
| 3 | [Desurveying drill holes](02-drillholes-and-samples/03-desurvey/README.md) | Stacked sulphide lenses | `Drillholes`, `Table` |
| 4 | [Compositing](02-drillholes-and-samples/04-compositing/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes` |
| 5 | [Duplicates](02-drillholes-and-samples/05-duplicates/README.md) | Stacked sulphide lenses | `duplicates`, `Table`, `check_drillholes`, `fix_drillholes`, `Drillholes` |
| 6 | [Runs and strip logs](02-drillholes-and-samples/06-runs-and-strip-logs/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `Categories`, `plot.strip_log` |
| 7 | [Domain cleanup](02-drillholes-and-samples/07-domain-cleanup/README.md) | Stacked sulphide lenses | `Drillholes`, `Categories`, `cell_declustering`, `Variogram`, `Search`, `CategoricalIndicatorKriging`, `BlockModel`, `remove_small_units`, `plot.section`, `contact_distance`, `buffer_domains`, `plot.category_legend` |
| 8 | [Mesh-crossing interval splits](02-drillholes-and-samples/08-mesh-interval-splits/README.md) | Stacked sulphide lenses | `Drillholes`, `merge_intervals`, `describe_by` |
| 9 | [Units](02-drillholes-and-samples/09-units/README.md) | Stacked sulphide lenses | `write_csv`, `Table`, `units`, `InvalidInput`, `plot.histogram`, `Drillholes`, `BlockModel`, `OrdinaryKriging`, `Variogram`, `Search`, `PointSet`, `grade_tonnage`, `MultipleIndicatorKriging`, `Capping`, `validate_model` |
| 10 | [Local mine grid](02-drillholes-and-samples/10-local-grid/README.md) | Synthetic | `LocalGrid`, `BlockModel`, `Drillholes` |
| 11 | [Coordinate reference systems](02-drillholes-and-samples/11-coordinate-systems/README.md) | Synthetic | `Polylines`, `PointSet`, `BlockModel`, `InvalidInput` |
| 12 | [Data spacing](02-drillholes-and-samples/12-data-spacing/README.md) | Coal seam thickness, Iron formation plateau | `data_spacing`, `plot.section`, `plot.histogram`, `Drillholes`, `Search` |
| 13 | [Drill-hole spacing from virtual grids](02-drillholes-and-samples/13-drillhole-spacing-virtual-grids/README.md) | Clustered metal survey | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `TurningBands`, `Search`, `select_realizations`, `spacing_study`, `uncertainty_curve`, `required_spacing`, `plot.uncertainty_curve`, `planned_drillholes` |
| 14 | [Drill-hole spacing from a learning curve](02-drillholes-and-samples/14-drillhole-spacing-learning-curve/README.md) | Clustered metal survey | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `TurningBands`, `Search`, `data_spacing`, `uncertainty_curve`, `required_spacing`, `spacing_study` |

## Geometry and block models

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Block model from extents](03-geometry-and-block-models/01-block-model-from-extents/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel` |
| 2 | [Filtering block models](03-geometry-and-block-models/02-filtering-block-models/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.Scene` |
| 3 | [Models larger than memory](03-geometry-and-block-models/03-large-models/README.md) | Iron formation plateau | `Drillholes`, `BlockModel`, `write_parquet`, `BlockModelFile`, `detrend`, `map_blocks`, `NormalScore`, `experimental_variogram`, `TurningBands`, `Search`, `plot.section` |
| 4 | [Sub-blocks](03-geometry-and-block-models/04-sub-blocks/README.md) | Stacked sulphide lenses | `BlockModel`, `plot.slab`, `grid_surface`, `Categories`, `plot.section`, `Drillholes`, `Search`, `InverseDistance` |
| 5 | [Solids](03-geometry-and-block-models/05-solids/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `plot.section`, `plot.slab`, `block_shell` |
| 6 | [Flag solid proportions in a block model](03-geometry-and-block-models/06-solid-proportions/README.md) | Vein gold grade control | `BlockModel`, `plot.section`, `Categories`, `plot.slab` |
| 7 | [Sub-block a model from solids](03-geometry-and-block-models/07-sub-blocks-from-solids/README.md) | Vein gold grade control | `BlockModel`, `Categories`, `plot.section`, `plot.slab`, `plot.category_legend` |
| 8 | [Find and fix degenerate solids](03-geometry-and-block-models/08-degenerate-solids/README.md) | Stacked sulphide lenses | `Mesh` |
| 9 | [Mesh booleans and cuts](03-geometry-and-block-models/09-mesh-booleans/README.md) | Stacked sulphide lenses | `Mesh` |
| 10 | [Polygons](03-geometry-and-block-models/10-polygons/README.md) | Coal seam thickness | `point_in_polygon`, `polygon_distance`, `plot.section`, `assign_domain`, `Categories`, `plot.category_colors`, `plot.category_legend`, `Polylines`, `PolygonSelector` |
| 11 | [Outlines](03-geometry-and-block-models/11-outlines/README.md) | Stacked sulphide lenses | `PointSet`, `data_spacing`, `outline` |
| 12 | [Topography from points](03-geometry-and-block-models/12-topography/README.md) | Stacked sulphide lenses | `PointSet`, `topography`, `plot.section`, `Drillholes`, `snap_to_surface` |
| 13 | [Mesh files](03-geometry-and-block-models/13-mesh-files/README.md) | Vein gold grade control | `write_mesh`, `read_mesh`, `plot.slab`, `Mesh` |
| 14 | [Shapefiles, GeoPackage, GeoTIFF and JPEG](03-geometry-and-block-models/14-gis-formats/README.md) | Soil geochemistry survey | `write_shapefile`, `read_shapefile`, `Polylines`, `write_geopackage`, `read_geopackage`, `Categories`, `BlockModel`, `write_geotiff`, `read_geotiff`, `write_jpeg`, `read_jpeg` |
| 15 | [GeoJSON](03-geometry-and-block-models/15-geojson/README.md) | Synthetic | `PointSet`, `Polylines`, `write_geojson`, `read_geojson` |
| 16 | [VTK files](03-geometry-and-block-models/16-vtk-files/README.md) | Vein gold grade control | `PointSet`, `write_vtk`, `read_vtk`, `BlockModel` |

## Viewing

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [3D views](04-viewing/01-3d-views/README.md) | Stacked sulphide lenses | `Drillholes`, `plot3d.Scene`, `BlockModel`, `Search`, `InverseDistance` |
| 2 | [Live scene](04-viewing/02-live-scene/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.Scene` |
| 3 | [Meshes and points in a scene](04-viewing/03-meshes-and-points/README.md) | Stacked sulphide lenses | `Drillholes`, `PointSet`, `topography`, `plot3d.Scene` |
| 4 | [Hole traces with deviation flags](04-viewing/04-hole-traces/README.md) | Stacked sulphide lenses | `check_drillholes`, `Drillholes`, `plot.holes` |
| 5 | [Stepped sections](04-viewing/05-stepped-sections/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Variogram`, `Search`, `BlockModel`, `OrdinaryKriging`, `plot.section`, `plot.fence` |
| 6 | [Sections with holes](04-viewing/06-sections-with-holes/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Variogram`, `Search`, `BlockModel`, `OrdinaryKriging`, `plot.section` |
| 7 | [Interactive sections](04-viewing/07-interactive-sections/README.md) | Stacked sulphide lenses | `Drillholes`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.Scene` |
| 8 | [Mesh sections](04-viewing/08-mesh-sections/README.md) | Stacked sulphide lenses | `write_geojson`, `read_geojson` |

## Exploratory analysis

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Paired data](05-exploratory-analysis/01-paired-data/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `pairs`, `paired_bias`, `plot.paired_bias`, `plot.qq`, `plot.scatter` |
| 2 | [Statistics by domain](05-exploratory-analysis/02-statistics-by-domain/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `describe_by`, `plot.boxplot`, `plot.cdf`, `plot.qq` |
| 3 | [Cleaning steps](05-exploratory-analysis/03-cleaning-steps/README.md) | Stacked sulphide lenses | `Pipeline`, `RenameColumns`, `ToNull`, `Replace`, `ToNumber`, `DropNull`, `plot.histogram` |
| 4 | [Declustering](05-exploratory-analysis/04-declustering/README.md) | Coal seam thickness | `cell_declustering`, `plot.declustering`, `polygon_declustering` |
| 5 | [Weight declustering](05-exploratory-analysis/05-weight-declustering/README.md) | Coal seam thickness | `cell_declustering`, `experimental_variogram`, `Search`, `NearestNeighbor`, `InverseDistance`, `OrdinaryKriging`, `weight_declustering` |
| 6 | [Top cuts](05-exploratory-analysis/06-top-cuts/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `cell_declustering`, `capping`, `describe_by`, `plot.probability`, `capping_report` |
| 7 | [Despiking](05-exploratory-analysis/07-despiking/README.md) | Soil geochemistry survey | `despike`, `NormalScore` |
| 8 | [Swaths](05-exploratory-analysis/08-swaths/README.md) | Phosphate weathering profile | `merge_intervals`, `Drillholes`, `describe_by`, `swath`, `plot.swath` |
| 9 | [Correlations](05-exploratory-analysis/09-correlations/README.md) | Iron formation plateau | `merge_intervals`, `Drillholes`, `correlation`, `plot.correlation`, `plot.scatter_matrix`, `plot.completeness`, `plot.conditional`, `h_scatter` |
| 10 | [Spatial bootstrap](05-exploratory-analysis/10-spatial-bootstrap/README.md) | Coal seam thickness | `cell_declustering`, `describe`, `despike`, `NormalScore`, `Variogram`, `experimental_variogram`, `spatial_bootstrap` |
| 11 | [Reference distributions](05-exploratory-analysis/11-reference-distributions/README.md) | Vein gold grade control, Porphyry geometallurgy | `merge_intervals`, `Drillholes`, `cell_declustering`, `KernelDensity`, `NormalScore`, `GaussianMixture`, `GaussianImputer` |

## Transforms

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Normal-score transform](06-transforms/01-normal-score/README.md) | Clustered metal survey | `cell_declustering`, `NormalScore`, `describe`, `normal_cdf`, `plot.probability` |
| 2 | [Censored normal-score transform](06-transforms/02-censored-normal-score/README.md) | Tailings reprocessing, Soil geochemistry survey | `NormalScore`, `experimental_variogram`, `Search`, `ExternalDriftKriging` |
| 3 | [Capping transform](06-transforms/03-capping-transform/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `cell_declustering`, `Capping`, `describe_by`, `capping_report`, `plot.probability`, `BlockModel`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `NormalScore` |
| 4 | [Compositional data](06-transforms/04-compositional/README.md) | Porphyry geometallurgy | `closure`, `composition_center`, `variation_matrix`, `total_variance`, `PointSet`, `Pipeline`, `ILR`, `PPMT`, `correlation`, `plot.ternary`, `plot.biplot` |
| 5 | [Multivariate transforms](06-transforms/05-multivariate-transforms/README.md) | Porphyry geometallurgy | `PCA`, `MAF`, `StepwiseConditional`, `PPMT` |
| 6 | [Imputation](06-transforms/06-imputation/README.md) | Stacked sulphide lenses | `merge_intervals`, `GaussianImputer`, `compare` |
| 7 | [Spatial imputation](06-transforms/07-spatial-imputation/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `NormalScore`, `experimental_variogram`, `plot.variogram`, `GaussianImputer`, `compare` |
| 8 | [Smooth trend](06-transforms/08-smooth-trend/README.md) | Coal seam thickness | `cell_declustering`, `detrend` |
| 9 | [Unfolding](06-transforms/09-unfolding/README.md) | Nickel laterite profile | `Drillholes`, `BlockModel`, `InverseDistance`, `Search`, `grid_surface`, `Unfold`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `plot.section`, `plot.slab` |
| 10 | [Target distribution correction](06-transforms/10-target-distribution-correction/README.md) | Clustered metal survey | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `check_realizations`, `correct_distribution`, `plot.histogram_reproduction`, `plot.section`, `KernelDensity` |
| 11 | [Pipeline](06-transforms/11-pipeline/README.md) | Clustered metal survey | `cell_declustering`, `Pipeline`, `Capping`, `NormalScore`, `PPMT` |

## Variography

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Experimental variograms](07-variography/01-experimental-variograms/README.md) | Clustered metal survey | `variogram_map`, `experimental_variogram`, `plot.variogram` |
| 2 | [Variogram fitting](07-variography/02-variogram-fitting/README.md) | Clustered metal survey | `variogram_map`, `experimental_variogram`, `plot.variogram`, `Variogram` |
| 3 | [Variogram sets](07-variography/03-variogram-sets/README.md) | Topsoil metals, Clustered metal survey | `experimental_variograms`, `Coregionalization`, `correlation`, `plot.variograms`, `BlockModel`, `experimental_variogram` |
| 4 | [Variogram volume](07-variography/04-variogram-volume/README.md) | Stacked sulphide lenses | `Drillholes`, `variogram_volume`, `plot.variogram_volume`, `experimental_variogram`, `Variogram`, `plot.variogram` |
| 5 | [Downhole nugget](07-variography/05-downhole-nugget/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `experimental_variogram` |
| 6 | [Madogram](07-variography/06-madogram/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `experimental_variogram`, `dissemination` |
| 7 | [Coregionalization](07-variography/07-coregionalization/README.md) | Topsoil metals | `experimental_variogram`, `Coregionalization`, `plot.variogram` |
| 8 | [Intrinsic coregionalization](07-variography/08-intrinsic-coregionalization/README.md) | Topsoil metals | `experimental_variograms`, `Coregionalization`, `plot.variogram`, `Search`, `Cokriging`, `compare` |
| 9 | [Local variogram parameters](07-variography/09-local-variogram-parameters/README.md) | Clustered metal survey | `BlockModel`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `Search`, `LocalAnisotropy`, `local_variogram_parameters`, `compare` |
| 10 | [From variogram to search plan](07-variography/10-search-plans/README.md) | Stacked sulphide lenses | `Drillholes`, `variogram_volume`, `experimental_variogram`, `Variogram`, `plot.variogram`, `plot.slab`, `BlockModel`, `OrdinaryKriging`, `Search`, `hole_distance`, `plot.section` |
| 11 | [Nugget inference](07-variography/11-nugget-inference/README.md) | Vein gold grade control | `merge_intervals`, `Drillholes`, `experimental_variogram`, `pairs` |

## Domains and geology

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Categories](08-domains-and-geology/01-categories/README.md) | Tailings reprocessing | `Drillholes`, `Categories`, `plot.proportions`, `plot.category_swath` |
| 2 | [Domain change tables](08-domains-and-geology/02-domain-change-tables/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Categories`, `cell_declustering`, `Variogram`, `Search`, `CategoricalIndicatorKriging`, `BlockModel`, `OrdinaryKriging`, `remove_small_units`, `domain_change`, `plot.domain_change` |
| 3 | [Along-hole transition matrix and MDS](08-domains-and-geology/03-transition-matrix-mds/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Categories`, `transition_matrix`, `plot.domain_change`, `plot.transition_mds` |
| 4 | [Local proportions](08-domains-and-geology/04-local-proportions/README.md) | Tailings reprocessing | `Drillholes`, `Categories`, `cell_declustering`, `vertical_proportions`, `plot.category_legend`, `detrend`, `BlockModel`, `combine_proportions`, `Variogram`, `Search`, `SIS`, `plot.category_colors`, `check_realizations`, `plot.histogram_reproduction` |
| 5 | [Contacts](08-domains-and-geology/05-contacts/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `contact`, `plot.contact` |
| 6 | [Soft-boundary statistics](08-domains-and-geology/06-soft-boundary-statistics/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `soft_boundary` |
| 7 | [Ensemble domains](08-domains-and-geology/07-ensemble-domains/README.md) | Stacked sulphide lenses | `merge_intervals`, `Drillholes`, `Categories`, `domains`, `plot.slab` |
| 8 | [Grade shells](08-domains-and-geology/08-grade-shells/README.md) | Iron formation plateau | `Drillholes`, `ImplicitModel`, `plot.slab` |
| 9 | [Contact surfaces](08-domains-and-geology/09-contact-surfaces/README.md) | Stacked sulphide lenses | `Drillholes`, `Variogram`, `ImplicitModel`, `BlockModel`, `plot.slab` |
| 10 | [Structural data](08-domains-and-geology/10-structural-data/README.md) | Synthetic | `BlockModel`, `ImplicitModel`, `plot.scatter` |
| 11 | [Layered surfaces](08-domains-and-geology/11-layered-surfaces/README.md) | Phosphate weathering profile | `grid_surface`, `plot.section`, `Drillholes`, `InverseDistance`, `Search`, `BlockModel`, `Categories`, `plot.slab`, `plot.category_legend` |
| 12 | [Large implicit models](08-domains-and-geology/12-large-implicit-models/README.md) | Iron formation plateau | `Drillholes`, `ImplicitModel`, `plot.slab` |

## Estimation

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Ordinary kriging](09-estimation/01-ordinary-kriging/README.md) | Clustered metal survey | `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `OrdinaryKriging`, `plot.scatter` |
| 2 | [Simple estimators](09-estimation/02-simple-estimators/README.md) | Clustered metal survey | `BlockModel`, `experimental_variogram`, `Variogram`, `Search`, `NearestNeighbor`, `InverseDistance`, `MovingAverage`, `OrdinaryKriging`, `compare`, `compare_models` |
| 3 | [Universal kriging](09-estimation/03-universal-kriging/README.md) | Coal seam thickness | `detrend`, `experimental_variogram`, `plot.variogram`, `Search`, `OrdinaryKriging`, `UniversalKriging` |
| 4 | [External drift kriging](09-estimation/04-external-drift-kriging/README.md) | Soil geochemistry survey | `experimental_variogram`, `Search`, `OrdinaryKriging`, `ExternalDriftKriging` |
| 5 | [Block kriging](09-estimation/05-block-kriging/README.md) | Clustered metal survey | `experimental_variogram`, `Variogram`, `Search`, `BlockModel`, `BlockKriging`, `OrdinaryKriging`, `compare` |
| 6 | [Search](09-estimation/06-search/README.md) | Clustered metal survey | `BlockModel`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `Search`, `compare` |
| 7 | [Search calibration](09-estimation/07-search-calibration/README.md) | Clustered metal survey | `experimental_variogram`, `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `HermiteAnamorphosis`, `calibrate_search`, `compare` |
| 8 | [High-grade restriction](09-estimation/08-high-grade-restriction/README.md) | Clustered metal survey | `BlockModel`, `experimental_variogram`, `Variogram`, `BlockKriging`, `Search`, `HighGrade`, `compare` |
| 9 | [Cokriging](09-estimation/09-cokriging/README.md) | Topsoil metals | `experimental_variogram`, `Coregionalization`, `Search`, `OrdinaryKriging`, `Cokriging`, `compare`, `BlockModel`, `LocalAnisotropy` |
| 10 | [Indicator kriging](09-estimation/10-indicator-kriging/README.md) | Topsoil metals | `experimental_variogram`, `Search`, `IndicatorKriging` |
| 11 | [Multiple indicator kriging](09-estimation/11-multiple-indicator-kriging/README.md) | Topsoil metals | `Search`, `cell_declustering`, `experimental_variogram`, `MultipleIndicatorKriging`, `OrdinaryKriging`, `compare`, `BlockModel` |
| 12 | [Multigaussian kriging](09-estimation/12-multigaussian-kriging/README.md) | Clustered metal survey | `cell_declustering`, `NormalScore`, `despike`, `experimental_variogram`, `BlockModel`, `MultigaussianKriging`, `Search`, `compare`, `plot.scatter` |
| 13 | [Soft boundaries](09-estimation/13-soft-boundaries/README.md) | Nickel laterite profile | `merge_intervals`, `Drillholes`, `contact`, `plot.contact`, `NormalScore`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `SGS` |
| 14 | [Locally varying anisotropy](09-estimation/14-local-anisotropy/README.md) | Clustered metal survey | `BlockModel`, `experimental_variogram`, `Variogram`, `OrdinaryKriging`, `Search`, `LocalAnisotropy`, `plot.directions`, `compare`, `cell_declustering`, `NormalScore`, `SGS` |
| 15 | [Categorical indicator kriging](09-estimation/15-categorical-indicator-kriging/README.md) | Stacked sulphide lenses | `Drillholes`, `Categories`, `cell_declustering`, `Variogram`, `Search`, `CategoricalIndicatorKriging`, `BlockModel`, `plot.section`, `plot.uncertain` |
| 16 | [Discrete Gaussian model](09-estimation/16-discrete-gaussian-model/README.md) | Clustered metal survey | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `change_of_support` |
| 17 | [Uniform conditioning](09-estimation/17-uniform-conditioning/README.md) | Clustered metal survey | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `change_of_support`, `Search`, `BlockModel`, `BlockKriging`, `UniformConditioning`, `compare` |
| 18 | [MIK localization](09-estimation/18-mik-localization/README.md) | Clustered metal survey | `cell_declustering`, `experimental_variogram`, `Variogram`, `Search`, `BlockModel`, `BlockKriging`, `MultipleIndicatorKriging`, `compare` |
| 19 | [Disjunctive kriging](09-estimation/19-disjunctive-kriging/README.md) | Clustered metal survey | `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `Variogram`, `BlockModel`, `DisjunctiveKriging`, `Search` |

## Simulation

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Sequential Gaussian simulation](10-simulation/01-sgs/README.md) | Clustered metal survey | `cell_declustering`, `NormalScore`, `experimental_variogram`, `BlockModel`, `SGS`, `Search` |
| 2 | [SGS along a shared path](10-simulation/02-sgs-shared-path/README.md) | Clustered metal survey | `cell_declustering`, `NormalScore`, `experimental_variogram`, `BlockModel`, `SGS`, `Search`, `check_realizations`, `plot.histogram_reproduction`, `plot.variogram_reproduction` |
| 3 | [Simulation at block support](10-simulation/03-simulation-at-block-support/README.md) | Clustered metal survey | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `plot.grade_tonnage`, `BlockKriging`, `localize`, `compare` |
| 4 | [Turning bands](10-simulation/04-turning-bands/README.md) | Clustered metal survey | `BlockModel`, `cell_declustering`, `NormalScore`, `experimental_variogram`, `SGS`, `Search`, `TurningBands` |
| 5 | [Simulation with a trend](10-simulation/05-simulation-with-trend/README.md) | Clustered metal survey | `cell_declustering`, `BlockModel`, `MovingAverage`, `Search`, `StepwiseConditional`, `NormalScore`, `experimental_variogram`, `SGS` |
| 6 | [Multivariate simulation](10-simulation/06-multivariate-simulation/README.md) | Porphyry geometallurgy | `PointSet`, `cell_declustering`, `BlockModel`, `Search`, `PPMT`, `PCA`, `experimental_variogram`, `MultivariateSimulation`, `TurningBands` |
| 7 | [Collocated cosimulation](10-simulation/07-collocated-cosimulation/README.md) | Topsoil metals | `NormalScore`, `experimental_variogram`, `BlockModel`, `Search`, `SGS`, `check_realizations`, `plot.histogram_reproduction`, `plot.variogram_reproduction`, `Variogram`, `DSS` |
| 8 | [Direct sequential simulation](10-simulation/08-direct-sequential-simulation/README.md) | Clustered metal survey | `cell_declustering`, `describe`, `NormalScore`, `experimental_variogram`, `BlockModel`, `Search`, `DSS`, `SGS`, `correct_distribution` |
| 9 | [Sequential indicator simulation](10-simulation/09-sis/README.md) | Topsoil metals | `Categories`, `experimental_variogram`, `Variogram`, `SIS`, `Search`, `plot.category_colors`, `plot.category_legend` |
| 10 | [Plurigaussian simulation](10-simulation/10-plurigaussian/README.md) | Topsoil metals | `Categories`, `Variogram`, `Plurigaussian`, `plot.category_colors`, `plot.category_legend`, `experimental_variogram` |
| 11 | [Grades in simulated rock types](10-simulation/11-grades-in-simulated-rocks/README.md) | Topsoil metals | `Categories`, `experimental_variogram`, `Variogram`, `SIS`, `Search`, `NormalScore`, `SGS` |
| 12 | [SNESIM and its multigrid](10-simulation/12-snesim-multigrid/README.md) | Channel training image | `BlockModel`, `SNESIM` |
| 13 | [Conditioning SNESIM](10-simulation/13-snesim-conditioning/README.md) | Channel training image | `BlockModel`, `SNESIM` |
| 14 | [Continuous SNESIM](10-simulation/14-snesim-continuous/README.md) | Seismic cube | `BlockModel`, `SNESIM`, `compare` |
| 15 | [Rotation and affinity](10-simulation/15-snesim-rotation-affinity/README.md) | Channel training image, Seismic cube | `BlockModel`, `LocalAnisotropy`, `SNESIM` |
| 16 | [Several training images](10-simulation/16-snesim-training-images-by-zone/README.md) | Channel training image | `BlockModel`, `object_training_image`, `SNESIM` |
| 17 | [Soft data in SNESIM](10-simulation/17-snesim-soft-data/README.md) | Channel training image | `BlockModel`, `SNESIM` |
| 18 | [Training images and their consistency](10-simulation/18-training-images/README.md) | Channel training image | `BlockModel`, `object_training_image`, `training_image_consistency` |
| 19 | [Image quilting](10-simulation/19-image-quilting/README.md) | Channel training image | `BlockModel`, `ImageQuilting` |
| 20 | [Continuous image quilting](10-simulation/20-image-quilting-continuous/README.md) | Seismic cube | `BlockModel`, `compare`, `ImageQuilting`, `SNESIM` |
| 21 | [Seismic volumes](10-simulation/21-seismic-volumes/README.md) | Seismic cube | `write_segy`, `read_segy`, `BlockModel`, `object_training_image`, `ImageQuilting` |

## Validation and reporting

| # | Example | Dataset | Covers |
|---|---|---|---|
| 1 | [Model checks](11-validation-and-reporting/01-model-checks/README.md) | Clustered metal survey | `experimental_variogram`, `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `global_bias`, `validate_model`, `plot.cdf`, `swath`, `plot.swath` |
| 2 | [Cross-validation](11-validation-and-reporting/02-cross-validation/README.md) | Clustered metal survey | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation` |
| 3 | [Kriging diagnostics](11-validation-and-reporting/03-kriging-diagnostics/README.md) | Clustered metal survey | `experimental_variogram`, `Variogram`, `BlockModel`, `Search`, `BlockKriging`, `compare` |
| 4 | [Realization checks](11-validation-and-reporting/04-realization-checks/README.md) | Clustered metal survey, Porphyry geometallurgy | `cell_declustering`, `NormalScore`, `experimental_variogram`, `Variogram`, `BlockModel`, `SGS`, `Search`, `check_realizations`, `plot.histogram_reproduction`, `plot.variogram_reproduction`, `PointSet`, `PPMT`, `TurningBands`, `MultivariateSimulation`, `plot.correlation_reproduction` |
| 5 | [Classification](11-validation-and-reporting/05-classification/README.md) | Coal seam thickness | `data_spacing`, `classify`, `experimental_variogram`, `Variogram`, `Search`, `BlockKriging`, `smooth_classes`, `Categories`, `plot.category_colors`, `plot.category_legend` |
| 6 | [Result plots](11-validation-and-reporting/06-result-plots/README.md) | Clustered metal survey, Nickel laterite profile | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation`, `MultipleIndicatorKriging`, `BlockModel`, `BlockKriging`, `cell_declustering`, `grade_tonnage`, `compare_models`, `plot.grade_tonnage`, `merge_intervals`, `Drillholes`, `contact`, `plot.contact` |
| 7 | [Section validation plates](11-validation-and-reporting/07-section-validation-plates/README.md) | Stacked sulphide lenses | `BlockModel`, `Drillholes`, `InverseDistance`, `Search`, `plot.section`, `plot.slab`, `plot.scatter` |
| 8 | [Compare two estimates of one deposit](11-validation-and-reporting/08-compare-two-estimates/README.md) | Clustered metal survey | `cell_declustering`, `describe`, `experimental_variogram`, `Variogram`, `Search`, `BlockKriging`, `InverseDistance`, `BlockModel`, `global_bias`, `swath`, `plot.swath`, `compare_models`, `OrdinaryKriging`, `compare` |
| 9 | [Grade-tonnage curves and the change in contained metal](11-validation-and-reporting/09-grade-tonnage-and-metal/README.md) | Clustered metal survey | `experimental_variogram`, `Variogram`, `Search`, `BlockKriging`, `InverseDistance`, `BlockModel`, `grade_tonnage`, `compare_models`, `plot.grade_tonnage` |

## Case studies

| # | Example | Dataset | Steps |
|---|---|---|---|
| 1 | [A coal seam in a lease](12-case-studies/01-coal-seam-in-a-lease/README.md) | Coal seam thickness | The lease and its grid; Declustering; An anisotropic variogram; Ordinary or universal kriging; Volume and tonnes; How sure is the total?; Classification by data spacing |
| 2 | [From drill holes to a classified model](12-case-studies/02-drillholes-to-classified-model/README.md) | Stacked sulphide lenses | Drill holes; Samples inside each lens; Declustering and capping; Zn variogram of lens 1; A sub-blocked model; Kriging in passes; Simulation at block support; Validation; Classification; Tonnes and metal |
| 3 | [Iron ore: several grades, one closure, reconciled](12-case-studies/03-iron-ore-multivariate/README.md) | Iron formation plateau | Composites by lithology; One closure; Ore domains of the model; Log-ratios, PPMT and simulation; Do the blocks still close?; Reconciliation with the blastholes; A bench in plan |
| 4 | [A laterite profile: horizons, then grades](12-case-studies/04-laterite-profile/README.md) | Nickel laterite profile | Horizon contacts; Is the 50 m mesh enough?; A layered block model; Ni and Co by horizon; Grades into the blocks; Tonnes and grade by horizon |

Each page alternates text, collapsed Python and its results. `mise run examples` reruns every script
and rewrites the pages; `bt.datasets` downloads the data once and caches it. `render.py --index` writes this file.
