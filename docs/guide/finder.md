# Which page do I need?

Find your question below and follow the link. **Learn** pages explain an idea from scratch; **examples** show one task on real data; **workflows** follow a whole problem from question to decision; **API** pages list every argument.

## New to geostatistics

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| What does a sample represent, and why does its size matter? | Learn [1. Samples and support](../learn/01-samples-and-support/learn_01.md) |
| How do I summarize a variable when the drilling is clustered? | Learn [2. Describing data](../learn/02-describing-data/learn_02.md) |
| What is a variogram and how do I read one? | Learn [3. Spatial continuity](../learn/03-spatial-continuity/learn_03.md) |
| How does kriging choose its weights? | Learn [4. Kriging](../learn/04-kriging/learn_04.md) |
| Why simulate when I can krige? | Learn [5. Simulation](../learn/05-simulation/learn_05.md) |
| How do I know a model is any good? | Learn [6. Checking a model](../learn/06-checking-a-model/learn_06.md) |
| How is the library laid out? | Guide [How Boitatá is organized](organization.md) |
| I want one pass from samples to a checked estimate. | [Quick tour](../examples/01-first-steps/01-quick-tour/example_01_01.md) |

</div>

## Drill holes, solids and grids

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I have a new drill hole database. Is it clean? | Workflow [Audit drill holes before modeling](../examples/14-workflows/04-drillhole-audit/example_14_04.md), then [checking drill holes](../examples/02-data-and-geometry/01-check-drillholes/example_02_01.md) |
| My holes have overlapping or missing intervals. | [Checking drill holes](../examples/02-data-and-geometry/01-check-drillholes/example_02_01.md) (`check_drillholes`, `fix_drillholes`) |
| Where is each sample in 3D? | [Desurveying drill holes](../examples/02-data-and-geometry/02-desurvey/example_02_02.md) |
| My assays have different lengths. | [Compositing](../examples/02-data-and-geometry/03-compositing/example_02_03.md) |
| Some collars or samples appear twice. | [Duplicates](../examples/02-data-and-geometry/04-duplicates/example_02_04.md) |
| An interval straddles a geological contact. | [Mesh-crossing interval splits](../examples/02-data-and-geometry/15-mesh-interval-splits/example_02_15.md) |
| Where does each hole run above cutoff? | [Runs and strip logs](../examples/02-data-and-geometry/13-runs-and-strip-logs/example_02_13.md) |
| I need a block model that covers my data. | [Block models](../examples/01-first-steps/02-block-models/example_01_02.md), [block model from extents](../examples/02-data-and-geometry/12-block-model-from-extents/example_02_12.md) |
| Which blocks fall inside a solid, and by how much? | [Solids](../examples/02-data-and-geometry/05-solids/example_02_05.md), workflow [Flag solid proportions](../examples/14-workflows/01-solid-proportions/example_14_01.md) |
| A thin solid gets the wrong volume on whole blocks. | [Sub-blocks](../examples/02-data-and-geometry/06-sub-blocks/example_02_06.md), and the sub-blocking tutorial in [Workflows](../examples/14-workflows/index.md) |
| My solid has holes, flipped faces or zero volume. | Workflow [Find and fix degenerate solids](../examples/14-workflows/03-degenerate-solids/example_14_03.md) |
| My rock-type model is full of isolated specks. | [Domain cleanup](../examples/02-data-and-geometry/14-domain-cleanup/example_02_14.md) |
| I have a lease boundary or pit outline. | [Polygons](../examples/02-data-and-geometry/08-polygons/example_02_08.md) |
| I need to read or write OBJ, STL, DXF, shapefiles or GeoTIFF. | [Mesh files](../examples/02-data-and-geometry/07-mesh-files/example_02_07.md), [GIS formats](../examples/02-data-and-geometry/09-gis-formats/example_02_09.md) |
| My block model does not fit in memory. | [Models larger than memory](../examples/02-data-and-geometry/10-large-models/example_02_10.md) |
| I want to see holes and blocks in 3D or on sections. | [3D views](../examples/02-data-and-geometry/11-3d-views/example_02_11.md), [stepped sections](../examples/02-data-and-geometry/16-stepped-sections/example_02_16.md), [hole traces](../examples/02-data-and-geometry/17-hole-traces/example_02_17.md) |

</div>

## Looking at the data

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| The rich zones were drilled more densely. What is the true mean? | [Declustering](../examples/03-exploratory-analysis/03-declustering/example_03_03.md), [weight declustering](../examples/03-exploratory-analysis/04-weight-declustering/example_03_04.md) |
| How do the statistics differ between domains? | [Statistics by domain](../examples/03-exploratory-analysis/02-statistics-by-domain/example_03_02.md) |
| A few extreme assays dominate the mean. | [Top cuts](../examples/03-exploratory-analysis/05-top-cuts/example_03_05.md), [capping transform](../examples/04-transforms/03-capping-transform/example_04_03.md) |
| Many samples share one value, such as the detection limit. | [Despiking](../examples/03-exploratory-analysis/06-despiking/example_03_06.md), [censored normal score](../examples/04-transforms/02-censored-normal-score/example_04_02.md) |
| Is the boundary between two domains sharp or gradual? | [Contacts](../examples/03-exploratory-analysis/07-contacts/example_03_07.md), [soft-boundary statistics](../examples/03-exploratory-analysis/08-soft-boundary-statistics/example_03_08.md) |
| Do two sampling campaigns agree? | [Paired data](../examples/03-exploratory-analysis/01-paired-data/example_03_01.md) |
| Does the grade drift along strike or with depth? | [Swaths](../examples/03-exploratory-analysis/09-swaths/example_03_09.md) |
| How closely spaced is my drilling? | [Data spacing](../examples/03-exploratory-analysis/10-data-spacing/example_03_10.md) |
| How sure am I of the mean itself? | [Spatial bootstrap](../examples/03-exploratory-analysis/12-spatial-bootstrap/example_03_12.md) |
| How do my variables relate to each other? | [Correlations](../examples/03-exploratory-analysis/11-correlations/example_03_11.md) |

</div>

## Transforms

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I need Gaussian scores for simulation or multigaussian kriging. | [Normal-score transform](../examples/04-transforms/01-normal-score/example_04_01.md) |
| My grades are parts of a whole and sum to 100 %. | [Compositional data](../examples/04-transforms/04-compositional/example_04_04.md) |
| I have several correlated grades to decorrelate. | [Multivariate transforms](../examples/04-transforms/05-multivariate-transforms/example_04_05.md) |
| Some samples lack one of the assays. | [Imputation](../examples/04-transforms/06-imputation/example_04_06.md), [spatial imputation](../examples/04-transforms/07-spatial-imputation/example_04_07.md) |
| My layer undulates between two surfaces. | [Unfolding](../examples/04-transforms/09-unfolding/example_04_09.md) |

</div>

## Variograms and search

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| How do I compute a variogram along a direction? | [Experimental variograms](../examples/05-spatial-continuity/01-experimental-variograms/example_05_01.md) |
| How do I fit a model to it? | [Variogram fitting](../examples/05-spatial-continuity/02-variogram-fitting/example_05_02.md) |
| How large is the nugget effect? | [Downhole nugget](../examples/05-spatial-continuity/05-downhole-nugget/example_05_05.md), [nugget inference](../examples/05-spatial-continuity/11-nugget-inference/example_05_11.md) |
| I have two or more variables to model together. | [Coregionalization](../examples/05-spatial-continuity/07-coregionalization/example_05_07.md) |
| The direction of continuity changes across the deposit. | [Local variogram parameters](../examples/05-spatial-continuity/09-local-variogram-parameters/example_05_09.md), [locally varying anisotropy](../examples/06-kriging/14-local-anisotropy/example_06_14.md) |
| How big should my search be? | [From variogram to search plan](../examples/05-spatial-continuity/10-search-plans/example_05_10.md), [search](../examples/06-kriging/06-search/example_06_06.md), [search calibration](../examples/06-kriging/07-search-calibration/example_06_07.md) |

</div>

## Estimating

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I want a first estimate of a grade. | [Ordinary kriging](../examples/06-kriging/01-ordinary-kriging/example_06_01.md) |
| I want inverse distance or nearest neighbor to compare against. | [Simple estimators](../examples/06-kriging/02-simple-estimators/example_06_02.md) |
| I need block averages, not point values. | [Block kriging](../examples/06-kriging/05-block-kriging/example_06_05.md) |
| The grade has a trend. | [Universal kriging](../examples/06-kriging/03-universal-kriging/example_06_03.md), [external drift kriging](../examples/06-kriging/04-external-drift-kriging/example_06_04.md) |
| High grades spread too far. | [High-grade restriction](../examples/06-kriging/08-high-grade-restriction/example_06_08.md) |
| A secondary variable is sampled more densely. | [Cokriging](../examples/06-kriging/09-cokriging/example_06_09.md) |
| I want the probability of exceeding a cutoff. | [Indicator kriging](../examples/06-kriging/10-indicator-kriging/example_06_10.md), [multiple indicator kriging](../examples/06-kriging/11-multiple-indicator-kriging/example_06_11.md), [multigaussian kriging](../examples/06-kriging/12-multigaussian-kriging/example_06_12.md) |
| Samples across a domain boundary should count a little. | [Soft boundaries](../examples/06-kriging/13-soft-boundaries/example_06_13.md) |

</div>

## Categories and domains

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| How do I name and color rock codes? | [Categories](../examples/07-categories-and-domains/01-categories/example_07_01.md) |
| How much did my domains change between two models? | [Domain change tables](../examples/07-categories-and-domains/02-domain-change-tables/example_07_02.md) |
| What proportion of each rock type is in each block? | [Local proportions](../examples/07-categories-and-domains/04-local-proportions/example_07_04.md), [categorical indicator kriging](../examples/07-categories-and-domains/05-categorical-indicator-kriging/example_07_05.md) |
| I need possible layouts of the rock types. | [Sequential indicator simulation](../examples/07-categories-and-domains/06-sis/example_07_06.md), [plurigaussian simulation](../examples/07-categories-and-domains/07-plurigaussian/example_07_07.md), [grades in simulated rock types](../examples/07-categories-and-domains/08-grades-in-simulated-rocks/example_07_08.md) |

</div>

## Simulating and recoverable resources

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I want many equally likely grade maps. | [Sequential Gaussian simulation](../examples/08-stochastic-simulation/01-sgs/example_08_01.md), [turning bands](../examples/08-stochastic-simulation/04-turning-bands/example_08_04.md) |
| My grid is large and simulation is slow. | [SGS along a shared path](../examples/08-stochastic-simulation/02-sgs-shared-path/example_08_02.md) |
| I need simulated values at block size. | [Simulation at block support](../examples/08-stochastic-simulation/03-simulation-at-block-support/example_08_03.md) |
| Several grades must stay correlated. | [Multivariate simulation](../examples/08-stochastic-simulation/06-multivariate-simulation/example_08_06.md), [collocated cosimulation](../examples/08-stochastic-simulation/07-collocated-cosimulation/example_08_07.md) |
| Geology looks like channels or lenses that a variogram cannot draw. | [SNESIM and its multigrid](../examples/13-multiple-point-statistics/01-snesim-multigrid/example_13_01.md), [training images](../examples/13-multiple-point-statistics/07-training-images/example_13_07.md), [image quilting](../examples/13-multiple-point-statistics/08-image-quilting/example_13_08.md) |
| How much ore is above cutoff at the size of a mining block? | [Discrete Gaussian model](../examples/09-recoverable-resources/01-discrete-gaussian-model/example_09_01.md), [uniform conditioning](../examples/09-recoverable-resources/02-uniform-conditioning/example_09_02.md) |
| I need a grade–tonnage curve and the metal it implies. | Workflow [grade–tonnage and metal change](../examples/14-workflows/06-grade-tonnage-and-metal/example_14_06.md), [result plots](../examples/10-checking-models/06-result-plots/example_10_06.md) |

</div>

## Checking a model

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| Does my estimate honor the data? | [Model checks](../examples/10-checking-models/01-model-checks/example_10_01.md), [section validation plates](../examples/10-checking-models/07-section-validation-plates/example_10_07.md) |
| Are my variogram and search any good? | [Cross-validation](../examples/10-checking-models/02-cross-validation/example_10_02.md) |
| Which blocks are well informed? | [Kriging diagnostics](../examples/10-checking-models/03-kriging-diagnostics/example_10_03.md) |
| Do my realizations reproduce the histogram and variogram? | [Realization checks](../examples/10-checking-models/04-realization-checks/example_10_04.md) |
| How do I classify the resource? | [Classification](../examples/10-checking-models/05-classification/example_10_05.md) |
| I have two estimates of one deposit. Which differs where? | Workflow [Compare two estimates](../examples/14-workflows/05-compare-two-estimates/example_14_05.md) |

</div>

## Surfaces, whole studies and files

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I want a grade shell or a contact surface from drill holes. | [Grade shells](../examples/11-geological-modeling/01-grade-shells/example_11_01.md), [contact surfaces](../examples/11-geological-modeling/02-contact-surfaces/example_11_02.md), [layered surfaces](../examples/11-geological-modeling/04-layered-surfaces/example_11_04.md) |
| I have measured orientations of the contact. | [Structural data](../examples/11-geological-modeling/03-structural-data/example_11_03.md) |
| I want to see a full study from raw tables to a result. | [Case studies](../examples/12-case-studies/index.md), such as [from drill holes to a classified model](../examples/12-case-studies/02-drillholes-to-classified-model/example_12_02.md) |
| I want to save a variogram, a transform or a fitted estimator. | Guide [Saving models](saving.md), [saving and loading](../examples/01-first-steps/03-saving-and-loading/example_01_03.md), [Parquet](../examples/01-first-steps/04-parquet/example_01_04.md) |
| What does an argument do? | API: [containers](../api/containers/index.md), [drill holes](../api/drillholes/index.md), [EDA](../api/eda/index.md), [transforms](../api/transforms/index.md), [variography](../api/variography/index.md), [estimation](../api/estimation/index.md), [simulation](../api/simulation/index.md), [modeling](../api/modeling/index.md), [datasets and plots](../api/datasets/index.md) |

</div>

!!! seealso "See also"
    - [Workflows](../examples/14-workflows/index.md): real problems from question to decision.
    - [Glossary](glossary.md): the terms these pages use.
