# Which page do I need?

Find your question and follow the link. **Learn** pages explain an idea from scratch, **Examples** show one task on real data, **Case studies** follow a whole problem from question to decision, and **API** pages list every argument.

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
| How is the library laid out? | Guide [how boitatá is organized](organization.md) |
| I want one pass from samples to a checked estimate. | [Quick tour](../examples/01-getting-started/01-quick-tour/example_01_01.md) |

</div>

## Drill holes, solids and grids

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I have a new drill hole database. Is it clean? | [Audit drill holes before modeling](../examples/02-drillholes-and-samples/02-drillhole-audit/example_02_02.md), then [checking drill holes](../examples/02-drillholes-and-samples/01-check-drillholes/example_02_01.md) |
| My holes have overlapping or missing intervals. | [Checking drill holes](../examples/02-drillholes-and-samples/01-check-drillholes/example_02_01.md) (`check_drillholes`, `fix_drillholes`) |
| Where is each sample in 3D? | [Desurveying drill holes](../examples/02-drillholes-and-samples/03-desurvey/example_02_03.md) |
| My assays have different lengths. | [Compositing](../examples/02-drillholes-and-samples/04-compositing/example_02_04.md) |
| Some collars or samples appear twice. | [Duplicates](../examples/02-drillholes-and-samples/05-duplicates/example_02_05.md) |
| An interval straddles a geological contact. | [Mesh-crossing interval splits](../examples/02-drillholes-and-samples/08-mesh-interval-splits/example_02_08.md) |
| Where does each hole run above cutoff? | [Runs and strip logs](../examples/02-drillholes-and-samples/06-runs-and-strip-logs/example_02_06.md) |
| I need a block model that covers my data. | [Block models](../examples/01-getting-started/02-block-models/example_01_02.md), [block model from extents](../examples/03-geometry-and-block-models/01-block-model-from-extents/example_03_01.md) |
| Which blocks fall inside a solid, and by how much? | [Solids](../examples/03-geometry-and-block-models/05-solids/example_03_05.md), [flag solid proportions](../examples/03-geometry-and-block-models/06-solid-proportions/example_03_06.md) |
| A thin solid gets the wrong volume on whole blocks. | [Sub-blocks](../examples/03-geometry-and-block-models/04-sub-blocks/example_03_04.md), and [sub-blocks from solids](../examples/03-geometry-and-block-models/07-sub-blocks-from-solids/example_03_07.md) |
| My solid has holes, flipped faces or zero volume. | [Find and fix degenerate solids](../examples/03-geometry-and-block-models/08-degenerate-solids/example_03_08.md) |
| My rock-type model is full of isolated specks. | [Domain cleanup](../examples/02-drillholes-and-samples/07-domain-cleanup/example_02_07.md) |
| I have a lease boundary or pit outline. | [Polygons](../examples/03-geometry-and-block-models/10-polygons/example_03_10.md) |
| I need to read or write OBJ, STL, PLY, OFF, MSH, DXF, shapefiles or GeoTIFF. | [Mesh files](../examples/03-geometry-and-block-models/13-mesh-files/example_03_13.md), [GIS formats](../examples/03-geometry-and-block-models/14-gis-formats/example_03_14.md) |
| My block model does not fit in memory. | [Models larger than memory](../examples/03-geometry-and-block-models/03-large-models/example_03_03.md) |
| I want to see holes and blocks in 3D or on sections. | [3D views](../examples/04-viewing/01-3d-views/example_04_01.md), [stepped sections](../examples/04-viewing/05-stepped-sections/example_04_05.md), [hole traces](../examples/04-viewing/04-hole-traces/example_04_04.md), [sections with holes](../examples/04-viewing/06-sections-with-holes/example_04_06.md), [live scene](../examples/04-viewing/02-live-scene/example_04_02.md) |

</div>

## Looking at the data

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| The rich zones were drilled more densely. What is the true mean? | [Declustering](../examples/05-exploratory-analysis/04-declustering/example_05_04.md), [weight declustering](../examples/05-exploratory-analysis/05-weight-declustering/example_05_05.md) |
| How do the statistics differ between domains? | [Statistics by domain](../examples/05-exploratory-analysis/02-statistics-by-domain/example_05_02.md) |
| A few extreme assays dominate the mean. | [Top cuts](../examples/05-exploratory-analysis/06-top-cuts/example_05_06.md), [capping transform](../examples/06-transforms/03-capping-transform/example_06_03.md) |
| Many samples share one value, such as the detection limit. | [Despiking](../examples/05-exploratory-analysis/07-despiking/example_05_07.md), [censored normal score](../examples/06-transforms/02-censored-normal-score/example_06_02.md) |
| Is the boundary between two domains sharp or gradual? | [Contacts](../examples/08-domains-and-geology/05-contacts/example_08_05.md), [soft-boundary statistics](../examples/08-domains-and-geology/06-soft-boundary-statistics/example_08_06.md) |
| Do two sampling campaigns agree? | [Paired data](../examples/05-exploratory-analysis/01-paired-data/example_05_01.md) |
| Does the grade drift along strike or with depth? | [Swaths](../examples/05-exploratory-analysis/08-swaths/example_05_08.md) |
| How closely spaced is my drilling? | [Data spacing](../examples/02-drillholes-and-samples/12-data-spacing/example_02_12.md) |
| How sure am I of the mean itself? | [Spatial bootstrap](../examples/05-exploratory-analysis/10-spatial-bootstrap/example_05_10.md) |
| How do my variables relate to each other? | [Correlations](../examples/05-exploratory-analysis/09-correlations/example_05_09.md) |

</div>

## Transforms

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I need Gaussian scores for simulation or multigaussian kriging. | [Normal-score transform](../examples/06-transforms/01-normal-score/example_06_01.md) |
| My grades are parts of a whole and sum to 100 %. | [Compositional data](../examples/06-transforms/04-compositional/example_06_04.md) |
| I have several correlated grades to decorrelate. | [Multivariate transforms](../examples/06-transforms/05-multivariate-transforms/example_06_05.md) |
| Some samples lack one of the assays. | [Imputation](../examples/06-transforms/06-imputation/example_06_06.md), [spatial imputation](../examples/06-transforms/07-spatial-imputation/example_06_07.md) |
| My layer undulates between two surfaces. | [Unfolding](../examples/06-transforms/09-unfolding/example_06_09.md) |

</div>

## Variograms and search

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| How do I compute a variogram along a direction? | [Experimental variograms](../examples/07-variography/01-experimental-variograms/example_07_01.md) |
| How do I fit a model to it? | [Variogram fitting](../examples/07-variography/02-variogram-fitting/example_07_02.md) |
| How large is the nugget effect? | [Downhole nugget](../examples/07-variography/05-downhole-nugget/example_07_05.md), [nugget inference](../examples/07-variography/11-nugget-inference/example_07_11.md) |
| I have two or more variables to model together. | [Coregionalization](../examples/07-variography/07-coregionalization/example_07_07.md) |
| The direction of continuity changes across the deposit. | [Local variogram parameters](../examples/07-variography/09-local-variogram-parameters/example_07_09.md), [locally varying anisotropy](../examples/09-estimation/14-local-anisotropy/example_09_14.md) |
| How big should my search be? | [From variogram to search plan](../examples/07-variography/10-search-plans/example_07_10.md), [search](../examples/09-estimation/06-search/example_09_06.md), [search calibration](../examples/09-estimation/07-search-calibration/example_09_07.md) |

</div>

## Estimating

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I want a first estimate of a grade. | [Ordinary kriging](../examples/09-estimation/01-ordinary-kriging/example_09_01.md) |
| I want inverse distance or nearest neighbor to compare against. | [Simple estimators](../examples/09-estimation/02-simple-estimators/example_09_02.md) |
| I need block averages, not point values. | [Block kriging](../examples/09-estimation/05-block-kriging/example_09_05.md) |
| The grade has a trend. | [Universal kriging](../examples/09-estimation/03-universal-kriging/example_09_03.md), [external drift kriging](../examples/09-estimation/04-external-drift-kriging/example_09_04.md) |
| High grades spread too far. | [High-grade restriction](../examples/09-estimation/08-high-grade-restriction/example_09_08.md) |
| A secondary variable is sampled more densely. | [Cokriging](../examples/09-estimation/09-cokriging/example_09_09.md) |
| I want the probability of exceeding a cutoff. | [Indicator kriging](../examples/09-estimation/10-indicator-kriging/example_09_10.md), [multiple indicator kriging](../examples/09-estimation/11-multiple-indicator-kriging/example_09_11.md), [multigaussian kriging](../examples/09-estimation/12-multigaussian-kriging/example_09_12.md) |
| Samples across a domain boundary should count a little. | [Soft boundaries](../examples/09-estimation/13-soft-boundaries/example_09_13.md) |

</div>

## Categories and domains

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| How do I name and color rock codes? | [Categories](../examples/08-domains-and-geology/01-categories/example_08_01.md) |
| Can the grades suggest domains on their own? | [Ensemble domains](../examples/08-domains-and-geology/07-ensemble-domains/example_08_07.md) |
| How much did my domains change between two models? | [Domain change tables](../examples/08-domains-and-geology/02-domain-change-tables/example_08_02.md) |
| What proportion of each rock type is in each block? | [Local proportions](../examples/08-domains-and-geology/04-local-proportions/example_08_04.md), [categorical indicator kriging](../examples/09-estimation/15-categorical-indicator-kriging/example_09_15.md) |
| I need possible layouts of the rock types. | [Sequential indicator simulation](../examples/10-simulation/09-sis/example_10_09.md), [plurigaussian simulation](../examples/10-simulation/10-plurigaussian/example_10_10.md), [grades in simulated rock types](../examples/10-simulation/11-grades-in-simulated-rocks/example_10_11.md) |

</div>

## Simulating and recoverable resources

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I want many equally likely grade maps. | [Sequential Gaussian simulation](../examples/10-simulation/01-sgs/example_10_01.md), [turning bands](../examples/10-simulation/04-turning-bands/example_10_04.md), [direct sequential simulation](../examples/10-simulation/08-direct-sequential-simulation/example_10_08.md) |
| My grid is large and simulation is slow. | [SGS along a shared path](../examples/10-simulation/02-sgs-shared-path/example_10_02.md) |
| I need simulated values at block size. | [Simulation at block support](../examples/10-simulation/03-simulation-at-block-support/example_10_03.md) |
| Several grades must stay correlated. | [Multivariate simulation](../examples/10-simulation/06-multivariate-simulation/example_10_06.md), [collocated cosimulation](../examples/10-simulation/07-collocated-cosimulation/example_10_07.md) |
| Geology looks like channels or lenses that a variogram cannot draw. | [SNESIM and its multigrid](../examples/10-simulation/12-snesim-multigrid/example_10_12.md), [training images](../examples/10-simulation/18-training-images/example_10_18.md), [image quilting](../examples/10-simulation/19-image-quilting/example_10_19.md) |
| How much ore is above cutoff at the size of a mining block? | [Discrete Gaussian model](../examples/09-estimation/16-discrete-gaussian-model/example_09_16.md), [uniform conditioning](../examples/09-estimation/17-uniform-conditioning/example_09_17.md) |
| I need a grade-tonnage curve and the metal it implies. | [Grade-tonnage and metal change](../examples/11-validation-and-reporting/09-grade-tonnage-and-metal/example_11_09.md), [result plots](../examples/11-validation-and-reporting/06-result-plots/example_11_06.md) |

</div>

## Checking a model

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| Does my estimate honor the data? | [Model checks](../examples/11-validation-and-reporting/01-model-checks/example_11_01.md), [section validation plates](../examples/11-validation-and-reporting/07-section-validation-plates/example_11_07.md) |
| Are my variogram and search any good? | [Cross-validation](../examples/11-validation-and-reporting/02-cross-validation/example_11_02.md) |
| Which blocks are well informed? | [Kriging diagnostics](../examples/11-validation-and-reporting/03-kriging-diagnostics/example_11_03.md) |
| Do my realizations reproduce the histogram and variogram? | [Realization checks](../examples/11-validation-and-reporting/04-realization-checks/example_11_04.md) |
| How do I classify the resource? | [Classification](../examples/11-validation-and-reporting/05-classification/example_11_05.md) |
| I have two estimates of one deposit. Which differs where? | [Compare two estimates](../examples/11-validation-and-reporting/08-compare-two-estimates/example_11_08.md) |

</div>

## Surfaces, whole studies and files

<div class="bt-compare" markdown>

| Question | Read |
| --- | --- |
| I want a grade shell or a contact surface from drill holes. | [Grade shells](../examples/08-domains-and-geology/08-grade-shells/example_08_08.md), [contact surfaces](../examples/08-domains-and-geology/09-contact-surfaces/example_08_09.md), [layered surfaces](../examples/08-domains-and-geology/11-layered-surfaces/example_08_11.md) |
| I have measured orientations of the contact. | [Structural data](../examples/08-domains-and-geology/10-structural-data/example_08_10.md) |
| My implicit model has tens of thousands of samples. | [Large implicit models](../examples/08-domains-and-geology/12-large-implicit-models/example_08_12.md) |
| I want to see a full study from raw tables to a result. | [Case studies](../examples/12-case-studies/index.md), such as [from drill holes to a classified model](../examples/12-case-studies/02-drillholes-to-classified-model/example_12_02.md) |
| I want to save a variogram, a transform or a fitted estimator. | Guide [saving models](saving.md), [saving and loading](../examples/01-getting-started/03-saving-and-loading/example_01_03.md), [Parquet](../examples/01-getting-started/04-parquet/example_01_04.md) |
| What does an argument do? | API: [containers](../api/containers/index.md), [drill holes](../api/drillholes/index.md), [EDA](../api/eda/index.md), [transforms](../api/transforms/index.md), [variography](../api/variography/index.md), [estimation](../api/estimation/index.md), [simulation](../api/simulation/index.md), [modeling](../api/modeling/index.md), [datasets and plots](../api/datasets/index.md) |

</div>

!!! seealso "See also"
    - [Case studies](../examples/12-case-studies/index.md): real problems from question to decision.
    - [Glossary](glossary.md): the terms these pages use.
