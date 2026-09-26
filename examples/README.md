# Examples

Worked examples with the Python API on open datasets from [gstvschlz/datasets](https://github.com/gstvschlz/datasets).
Tutorials follow one deposit from data to a result; topics show one feature each.

## Tutorials

| # | Tutorial | Dataset | Steps |
|---|---|---|---|
| 2 | [From drill holes to a classified model](tutorials/02-drillholes-to-classified-model/README.md) | Drillholes (legacy) | Composites; The lens and the drilled volume; Statistics and declustering; Variograms; Kriging in passes; A soft boundary with SM; Simulation at block support; Classification; Comparing models; Saving |

## Topics

| # | Topic | Dataset | Covers |
|---|---|---|---|
| | **Data** | | |
| 3 | [Drillholes](topics/03-compositing/README.md) | Drillholes (legacy) | `check_drillholes`, `fix_drillholes`, `merge_intervals`, `Drillholes`, `plot.slab`, `plot.boxplot` |
| 7 | [Data and declustering](topics/07-declustering/README.md) | Walker Lake | `cell_declustering`, `plot.declustering` |
| 13 | [Exploratory data analysis](topics/13-correlations/README.md) | Drillholes (legacy) | `check_drillholes`, `fix_drillholes`, `merge_intervals`, `Drillholes`, `duplicates`, `pairs`, `paired_bias`, `plot.paired_bias`, `plot.qq`, `plot.scatter`, `cell_declustering`, `describe_by`, `plot.boxplot`, `plot.cdf`, `capping`, `plot.probability`, `capping_report`, `grade_tonnage`, `contact`, `swath`, `plot.swath`, `Categories`, `plot.proportions`, `plot.category_swath`, `data_spacing`, `plot.histogram`, `h_scatter`, `plot.scatter_matrix`, `plot.completeness`, `plot.correlation`, `plot.conditional` |
| | **Transforms** | | |
| 14 | [Normal-score transform](topics/14-normal-score/README.md) | Walker Lake | `cell_declustering`, `NormalScore`, `normal_cdf`, `plot.probability` |
| 15 | [Compositional data](topics/15-compositional/README.md) | Porphyry geometallurgy | `closure`, `ilr`, `PPMT`, `ilr_inverse` |
| 16 | [Multivariate transforms](topics/16-multivariate-transforms/README.md) | Porphyry geometallurgy | `PCA`, `MAF`, `StepwiseConditional`, `PPMT`, `cell_declustering`, `BlockModel`, `Search`, `PointSet`, `experimental_variogram`, `MultivariateSimulation`, `TurningBands`, `GaussianImputer` |
| | **Variography** | | |
| 19 | [Variography](topics/19-variogram-fitting/README.md) | Walker Lake, Drillholes (legacy) | `variogram_map`, `experimental_variogram`, `plot.variogram`, `Variogram`, `check_drillholes`, `fix_drillholes`, `merge_intervals`, `Drillholes` |
| | **Estimation** | | |
| 22 | [Ordinary kriging](topics/22-ordinary-kriging/README.md) | Walker Lake | `Variogram`, `BlockModel`, `Search`, `OrdinaryKriging`, `plot.scatter` |
| 23 | [Estimation methods and search](topics/23-simple-estimators/README.md) | Walker Lake | `Variogram`, `BlockModel`, `Search`, `NearestNeighbor`, `InverseDistance`, `OrdinaryKriging`, `UniversalKriging`, `BlockKriging` |
| 27 | [Cokriging and indicator kriging](topics/27-cokriging/README.md) | Jura | `experimental_variogram`, `Coregionalization`, `Search`, `OrdinaryKriging`, `Cokriging`, `plot.variogram`, `IndicatorKriging`, `cell_declustering`, `MultipleIndicatorKriging`, `BlockModel` |
| 31 | [Locally varying anisotropy](topics/31-local-anisotropy/README.md) | Walker Lake | `Variogram`, `BlockModel`, `OrdinaryKriging`, `Search`, `LocalAnisotropy`, `plot.directions`, `cell_declustering`, `SGS` |
| | **Change of support** | | |
| 32 | [Change of support and disjunctive kriging](topics/32-discrete-gaussian-model/README.md) | Walker Lake | `Variogram`, `cell_declustering`, `HermiteAnamorphosis`, `experimental_variogram`, `change_of_support`, `BlockModel`, `SGS`, `Search`, `BlockKriging`, `UniformConditioning`, `MultipleIndicatorKriging`, `localize`, `DisjunctiveKriging` |
| | **Simulation** | | |
| 36 | [Sequential Gaussian simulation](topics/36-sgs/README.md) | Walker Lake | `Variogram`, `cell_declustering`, `NormalScore`, `experimental_variogram`, `BlockModel`, `SGS`, `Search` |
| 42 | [Simulation methods](topics/42-grades-in-simulated-rocks/README.md) | Walker Lake, Jura | `cell_declustering`, `Variogram`, `BlockModel`, `SGS`, `Search`, `TurningBands`, `MovingAverage`, `StepwiseConditional`, `experimental_variogram`, `Categories`, `SIS`, `Plurigaussian`, `plot.category_colors`, `plot.category_legend`, `NormalScore` |
| | **Validation** | | |
| 44 | [Validation and classification](topics/44-model-checks/README.md) | Walker Lake | `Variogram`, `cell_declustering`, `BlockModel`, `Search`, `BlockKriging`, `validate_model`, `plot.cdf`, `swath`, `plot.swath`, `OrdinaryKriging`, `HermiteAnamorphosis`, `calibrate_search`, `neighborhood_stats`, `classify`, `smooth_classes`, `Categories`, `plot.category_colors`, `plot.category_legend` |
| | **Modeling** | | |
| 49 | [Solids and block models](topics/49-solids/README.md) | Drillholes (legacy) | `Mesh`, `BlockModel`, `plot.section`, `plot.slab`, `block_shell`, `Search`, `InverseDistance`, `write_mesh`, `read_mesh` |
| 52 | [Implicit modeling](topics/52-grade-shells/README.md) | Drillholes (legacy) | `ImplicitModel`, `BlockModel`, `plot.slab` |
| 53 | [Geological modeling](topics/53-contact-surfaces/README.md) | Drillholes (legacy) | `Drillholes`, `Variogram`, `ImplicitModel`, `plot.slab`, `BlockModel`, `plot.scatter`, `grid_surface` |
| | **I/O and scale** | | |
| 57 | [Storing containers in Parquet](topics/57-parquet/README.md) | Walker Lake | `PointSet`, `Variogram`, `BlockModel`, `OrdinaryKriging`, `Search`, `write_parquet`, `write_csv`, `read_parquet`, `write_shapefile`, `read_shapefile`, `Polylines`, `write_geotiff`, `read_geotiff` |
| 59 | [Models larger than memory](topics/59-large-models/README.md) | Drillholes (legacy) | `cell_declustering`, `convex_hull`, `BlockModel`, `write_parquet`, `BlockModelFile`, `experimental_variogram`, `Search`, `OrdinaryKriging`, `NormalScore`, `Variogram`, `SimpleKriging`, `map_blocks`, `TurningBands`, `plot.uncertain` |
| 60 | [3D views](topics/60-3d-views/README.md) | Drillholes (legacy) | `PointSet`, `convex_hull`, `plot3d.to_pyvista`, `plot3d.plot`, `BlockModel`, `Search`, `InverseDistance`, `plot3d.slices` |
| | **More methods** | | |
| 63 | [Smooth trend](topics/63-smooth-trend/README.md) | Coal seam thickness | `cell_declustering`, `detrend` |
| 66 | [Result plots](topics/66-result-plots/README.md) | Walker Lake, Nickel laterite profile | `experimental_variogram`, `Variogram`, `Search`, `OrdinaryKriging`, `plot.cross_validation`, `MultipleIndicatorKriging`, `BlockModel`, `BlockKriging`, `cell_declustering`, `grade_tonnage`, `compare_models`, `plot.grade_tonnage`, `merge_intervals`, `Drillholes`, `contact`, `plot.contact` |

Each page alternates text, collapsed Python and its results. `mise run examples` reruns every `example_NN.py`
and rewrites the pages; `cs.datasets` downloads the data once and caches it. `render.py --index` writes this file.
