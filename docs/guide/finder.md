# which page do i need?

find your question and follow the link. **learn** pages explain an idea from scratch, **examples** show one task on real data, **workflows** follow a whole problem from question to decision, and **API** pages list every argument.

## new to geostatistics

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| what does a sample represent, and why does its size matter? | learn [1. samples and support](../learn/01-samples-and-support/learn_01.md) |
| how do i summarize a variable when the drilling is clustered? | learn [2. describing data](../learn/02-describing-data/learn_02.md) |
| what is a variogram and how do i read one? | learn [3. spatial continuity](../learn/03-spatial-continuity/learn_03.md) |
| how does kriging choose its weights? | learn [4. kriging](../learn/04-kriging/learn_04.md) |
| why simulate when i can krige? | learn [5. simulation](../learn/05-simulation/learn_05.md) |
| how do i know a model is any good? | learn [6. checking a model](../learn/06-checking-a-model/learn_06.md) |
| how is the library laid out? | guide [how boitatá is organized](organization.md) |
| i want one pass from samples to a checked estimate. | [quick tour](../examples/01-first-steps/01-quick-tour/example_01_01.md) |

</div>

## drill holes, solids and grids

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| i have a new drill hole database. is it clean? | workflow [audit drill holes before modeling](../examples/14-workflows/04-drillhole-audit/example_14_04.md), then [checking drill holes](../examples/02-data-and-geometry/01-check-drillholes/example_02_01.md) |
| my holes have overlapping or missing intervals. | [checking drill holes](../examples/02-data-and-geometry/01-check-drillholes/example_02_01.md) (`check_drillholes`, `fix_drillholes`) |
| where is each sample in 3D? | [desurveying drill holes](../examples/02-data-and-geometry/02-desurvey/example_02_02.md) |
| my assays have different lengths. | [compositing](../examples/02-data-and-geometry/03-compositing/example_02_03.md) |
| some collars or samples appear twice. | [duplicates](../examples/02-data-and-geometry/04-duplicates/example_02_04.md) |
| an interval straddles a geological contact. | [mesh-crossing interval splits](../examples/02-data-and-geometry/15-mesh-interval-splits/example_02_15.md) |
| where does each hole run above cutoff? | [runs and strip logs](../examples/02-data-and-geometry/13-runs-and-strip-logs/example_02_13.md) |
| i need a block model that covers my data. | [block models](../examples/01-first-steps/02-block-models/example_01_02.md), [block model from extents](../examples/02-data-and-geometry/12-block-model-from-extents/example_02_12.md) |
| which blocks fall inside a solid, and by how much? | [solids](../examples/02-data-and-geometry/05-solids/example_02_05.md), workflow [flag solid proportions](../examples/14-workflows/01-solid-proportions/example_14_01.md) |
| a thin solid gets the wrong volume on whole blocks. | [sub-blocks](../examples/02-data-and-geometry/06-sub-blocks/example_02_06.md), and the sub-blocking tutorial in [workflows](../examples/14-workflows/index.md) |
| my solid has holes, flipped faces or zero volume. | workflow [find and fix degenerate solids](../examples/14-workflows/03-degenerate-solids/example_14_03.md) |
| my rock-type model is full of isolated specks. | [domain cleanup](../examples/02-data-and-geometry/14-domain-cleanup/example_02_14.md) |
| i have a lease boundary or pit outline. | [polygons](../examples/02-data-and-geometry/08-polygons/example_02_08.md) |
| i need to read or write OBJ, STL, DXF, shapefiles or GeoTIFF. | [mesh files](../examples/02-data-and-geometry/07-mesh-files/example_02_07.md), [GIS formats](../examples/02-data-and-geometry/09-gis-formats/example_02_09.md) |
| my block model does not fit in memory. | [models larger than memory](../examples/02-data-and-geometry/10-large-models/example_02_10.md) |
| i want to see holes and blocks in 3D or on sections. | [3D views](../examples/02-data-and-geometry/11-3d-views/example_02_11.md), [stepped sections](../examples/02-data-and-geometry/16-stepped-sections/example_02_16.md), [hole traces](../examples/02-data-and-geometry/17-hole-traces/example_02_17.md), [sections with holes](../examples/02-data-and-geometry/20-sections-with-holes/example_02_20.md) |

</div>

## looking at the data

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| the rich zones were drilled more densely. what is the true mean? | [declustering](../examples/03-exploratory-analysis/03-declustering/example_03_03.md), [weight declustering](../examples/03-exploratory-analysis/04-weight-declustering/example_03_04.md) |
| how do the statistics differ between domains? | [statistics by domain](../examples/03-exploratory-analysis/02-statistics-by-domain/example_03_02.md) |
| a few extreme assays dominate the mean. | [top cuts](../examples/03-exploratory-analysis/05-top-cuts/example_03_05.md), [capping transform](../examples/04-transforms/03-capping-transform/example_04_03.md) |
| many samples share one value, such as the detection limit. | [despiking](../examples/03-exploratory-analysis/06-despiking/example_03_06.md), [censored normal score](../examples/04-transforms/02-censored-normal-score/example_04_02.md) |
| is the boundary between two domains sharp or gradual? | [contacts](../examples/03-exploratory-analysis/07-contacts/example_03_07.md), [soft-boundary statistics](../examples/03-exploratory-analysis/08-soft-boundary-statistics/example_03_08.md) |
| do two sampling campaigns agree? | [paired data](../examples/03-exploratory-analysis/01-paired-data/example_03_01.md) |
| does the grade drift along strike or with depth? | [swaths](../examples/03-exploratory-analysis/09-swaths/example_03_09.md) |
| how closely spaced is my drilling? | [data spacing](../examples/03-exploratory-analysis/10-data-spacing/example_03_10.md) |
| how sure am i of the mean itself? | [spatial bootstrap](../examples/03-exploratory-analysis/12-spatial-bootstrap/example_03_12.md) |
| how do my variables relate to each other? | [correlations](../examples/03-exploratory-analysis/11-correlations/example_03_11.md) |

</div>

## transforms

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| i need gaussian scores for simulation or multigaussian kriging. | [normal-score transform](../examples/04-transforms/01-normal-score/example_04_01.md) |
| my grades are parts of a whole and sum to 100 %. | [compositional data](../examples/04-transforms/04-compositional/example_04_04.md) |
| i have several correlated grades to decorrelate. | [multivariate transforms](../examples/04-transforms/05-multivariate-transforms/example_04_05.md) |
| some samples lack one of the assays. | [imputation](../examples/04-transforms/06-imputation/example_04_06.md), [spatial imputation](../examples/04-transforms/07-spatial-imputation/example_04_07.md) |
| my layer undulates between two surfaces. | [unfolding](../examples/04-transforms/09-unfolding/example_04_09.md) |

</div>

## variograms and search

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| how do i compute a variogram along a direction? | [experimental variograms](../examples/05-spatial-continuity/01-experimental-variograms/example_05_01.md) |
| how do i fit a model to it? | [variogram fitting](../examples/05-spatial-continuity/02-variogram-fitting/example_05_02.md) |
| how large is the nugget effect? | [downhole nugget](../examples/05-spatial-continuity/05-downhole-nugget/example_05_05.md), [nugget inference](../examples/05-spatial-continuity/11-nugget-inference/example_05_11.md) |
| i have two or more variables to model together. | [coregionalization](../examples/05-spatial-continuity/07-coregionalization/example_05_07.md) |
| the direction of continuity changes across the deposit. | [local variogram parameters](../examples/05-spatial-continuity/09-local-variogram-parameters/example_05_09.md), [locally varying anisotropy](../examples/06-kriging/14-local-anisotropy/example_06_14.md) |
| how big should my search be? | [from variogram to search plan](../examples/05-spatial-continuity/10-search-plans/example_05_10.md), [search](../examples/06-kriging/06-search/example_06_06.md), [search calibration](../examples/06-kriging/07-search-calibration/example_06_07.md) |

</div>

## estimating

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| i want a first estimate of a grade. | [ordinary kriging](../examples/06-kriging/01-ordinary-kriging/example_06_01.md) |
| i want inverse distance or nearest neighbor to compare against. | [simple estimators](../examples/06-kriging/02-simple-estimators/example_06_02.md) |
| i need block averages, not point values. | [block kriging](../examples/06-kriging/05-block-kriging/example_06_05.md) |
| the grade has a trend. | [universal kriging](../examples/06-kriging/03-universal-kriging/example_06_03.md), [external drift kriging](../examples/06-kriging/04-external-drift-kriging/example_06_04.md) |
| high grades spread too far. | [high-grade restriction](../examples/06-kriging/08-high-grade-restriction/example_06_08.md) |
| a secondary variable is sampled more densely. | [cokriging](../examples/06-kriging/09-cokriging/example_06_09.md) |
| i want the probability of exceeding a cutoff. | [indicator kriging](../examples/06-kriging/10-indicator-kriging/example_06_10.md), [multiple indicator kriging](../examples/06-kriging/11-multiple-indicator-kriging/example_06_11.md), [multigaussian kriging](../examples/06-kriging/12-multigaussian-kriging/example_06_12.md) |
| samples across a domain boundary should count a little. | [soft boundaries](../examples/06-kriging/13-soft-boundaries/example_06_13.md) |

</div>

## categories and domains

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| how do i name and color rock codes? | [categories](../examples/07-categories-and-domains/01-categories/example_07_01.md) |
| how much did my domains change between two models? | [domain change tables](../examples/07-categories-and-domains/02-domain-change-tables/example_07_02.md) |
| what proportion of each rock type is in each block? | [local proportions](../examples/07-categories-and-domains/04-local-proportions/example_07_04.md), [categorical indicator kriging](../examples/07-categories-and-domains/05-categorical-indicator-kriging/example_07_05.md) |
| i need possible layouts of the rock types. | [sequential indicator simulation](../examples/07-categories-and-domains/06-sis/example_07_06.md), [plurigaussian simulation](../examples/07-categories-and-domains/07-plurigaussian/example_07_07.md), [grades in simulated rock types](../examples/07-categories-and-domains/08-grades-in-simulated-rocks/example_07_08.md) |

</div>

## simulating and recoverable resources

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| i want many equally likely grade maps. | [sequential gaussian simulation](../examples/08-stochastic-simulation/01-sgs/example_08_01.md), [turning bands](../examples/08-stochastic-simulation/04-turning-bands/example_08_04.md) |
| my grid is large and simulation is slow. | [SGS along a shared path](../examples/08-stochastic-simulation/02-sgs-shared-path/example_08_02.md) |
| i need simulated values at block size. | [simulation at block support](../examples/08-stochastic-simulation/03-simulation-at-block-support/example_08_03.md) |
| several grades must stay correlated. | [multivariate simulation](../examples/08-stochastic-simulation/06-multivariate-simulation/example_08_06.md), [collocated cosimulation](../examples/08-stochastic-simulation/07-collocated-cosimulation/example_08_07.md) |
| geology looks like channels or lenses that a variogram cannot draw. | [SNESIM and its multigrid](../examples/13-multiple-point-statistics/01-snesim-multigrid/example_13_01.md), [training images](../examples/13-multiple-point-statistics/07-training-images/example_13_07.md), [image quilting](../examples/13-multiple-point-statistics/08-image-quilting/example_13_08.md) |
| how much ore is above cutoff at the size of a mining block? | [discrete gaussian model](../examples/09-recoverable-resources/01-discrete-gaussian-model/example_09_01.md), [uniform conditioning](../examples/09-recoverable-resources/02-uniform-conditioning/example_09_02.md) |
| i need a grade–tonnage curve and the metal it implies. | workflow [grade–tonnage and metal change](../examples/14-workflows/06-grade-tonnage-and-metal/example_14_06.md), [result plots](../examples/10-checking-models/06-result-plots/example_10_06.md) |

</div>

## checking a model

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| does my estimate honor the data? | [model checks](../examples/10-checking-models/01-model-checks/example_10_01.md), [section validation plates](../examples/10-checking-models/07-section-validation-plates/example_10_07.md) |
| are my variogram and search any good? | [cross-validation](../examples/10-checking-models/02-cross-validation/example_10_02.md) |
| which blocks are well informed? | [kriging diagnostics](../examples/10-checking-models/03-kriging-diagnostics/example_10_03.md) |
| do my realizations reproduce the histogram and variogram? | [realization checks](../examples/10-checking-models/04-realization-checks/example_10_04.md) |
| how do i classify the resource? | [classification](../examples/10-checking-models/05-classification/example_10_05.md) |
| i have two estimates of one deposit. which differs where? | workflow [compare two estimates](../examples/14-workflows/05-compare-two-estimates/example_14_05.md) |

</div>

## surfaces, whole studies and files

<div class="bt-compare" markdown>

| question | read |
| --- | --- |
| i want a grade shell or a contact surface from drill holes. | [grade shells](../examples/11-geological-modeling/01-grade-shells/example_11_01.md), [contact surfaces](../examples/11-geological-modeling/02-contact-surfaces/example_11_02.md), [layered surfaces](../examples/11-geological-modeling/04-layered-surfaces/example_11_04.md) |
| i have measured orientations of the contact. | [structural data](../examples/11-geological-modeling/03-structural-data/example_11_03.md) |
| i want to see a full study from raw tables to a result. | [case studies](../examples/12-case-studies/index.md), such as [from drill holes to a classified model](../examples/12-case-studies/02-drillholes-to-classified-model/example_12_02.md) |
| i want to save a variogram, a transform or a fitted estimator. | guide [saving models](saving.md), [saving and loading](../examples/01-first-steps/03-saving-and-loading/example_01_03.md), [parquet](../examples/01-first-steps/04-parquet/example_01_04.md) |
| what does an argument do? | API: [containers](../api/containers/index.md), [drill holes](../api/drillholes/index.md), [EDA](../api/eda/index.md), [transforms](../api/transforms/index.md), [variography](../api/variography/index.md), [estimation](../api/estimation/index.md), [simulation](../api/simulation/index.md), [modeling](../api/modeling/index.md), [datasets and plots](../api/datasets/index.md) |

</div>

!!! seealso "See also"
    - [Workflows](../examples/14-workflows/index.md): real problems from question to decision.
    - [Glossary](glossary.md): the terms these pages use.
