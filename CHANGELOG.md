# Changelog

## 0.5.0 - 2026-10-07

### Features

- [**breaking**] one spelling across containers
- conveniences that remove repeated code
- HTML representation in notebooks

### Fixes

- **plot3d:** wait for the viewer, not the #scene div, before screenshots

### Documentation

- **examples:** use the new API
## 0.4.3 - 2026-10-06

### Features

- **plot3d:** three.js viewer core in a private module
- **plot3d:** representations and motion quality in the three.js viewer
- **plot3d:** per-layer attribute filter on the GPU in the three.js viewer
- **plot3d:** section maker with straight and polyline cuts in the three.js viewer
- **plot3d:** click to inspect, screenshots, fullscreen and shortcuts in the three.js viewer
- **estimation:** [**breaking**] remove DrillholePlan and the drill-hole search strategies
- **plot3d:** [**breaking**] three.js viewer replaces pyvista

### Documentation

- use the bt alias in simulation docstring examples
## 0.4.2 - 2026-10-06

### Features

- [**breaking**] remove progress bars
## 0.4.1 - 2026-10-06

### Features

- **plot3d:** polyline sections, section drawer, layer toggles and trame in colab

### Fixes

- **learn:** keep pyvista below 0.49 in colab, whose IPython lacks guarded_eval

### Documentation

- **examples:** render the polyline curtain in interactive sections
- **learn:** tour estimates every SMU a lens touches, diluted, with an interactive scene

### CI

- run the colab tour in colab's runtime image
## 0.4.0 - 2026-10-06

### Features

- **transforms:** column names and containers in NormalScore, HermiteAnamorphosis, BoxCox and PPMT
- **py:** Pipeline over containers
- **eda:** cleaning steps for containers and pipelines
- **core:** column units
- **coda:** log-ratio transformers and partition bases
- **coda:** detection-limit replacement, simplex operations and compositional statistics
- **plot:** ternary diagram and clr biplot
- **blocks:** outline polygons from points
- **blocks:** topography from points
- **drillholes:** snap collars to a surface
- **simulation:** derived summary statistics
- **simulation:** grade-tonnage uncertainty
- **plot3d:** live Scene
- **plot:** section through origin, azimuth and dip with projected holes
- **core:** self-intersection check and hole filling for meshes
- **plot3d:** drillholes as one tube mesh with collar labels
- **plot3d:** block model volumes
- **plot3d:** Scene.screenshot with scale and transparent background
- **plot3d:** interactive sections
- **plot3d:** motion quality while the camera moves
- **simulation:** direct sequential simulation in data units
- **python:** bt.DSS direct sequential simulation
- **simulation:** collocated co-DSS with a secondary variable
- **simulation:** select_realizations picks representative realizations by k-medoids
- **drillholes:** planned_drillholes collar grid over targets
- **simulation:** window= and groups= production volumes in simulate
- **python:** SimulationSummary.groups names the rows of groups=
- **eda:** uncertainty curve and required spacing
- **eda:** [**breaking**] data_spacing as Cabral Pinto's equivalent spacing
- **simulation:** precision tolerances and validate on SimulationSummary
- **simulation:** spacing_study drills simulated truths on virtual grids
- **estimation:** DrillholePlan scores candidate holes by the kriging metrics they bring
- **estimation:** DrillholePlan with domains, per-domain rules and data_spacing
- **estimation:** DrillholePlan keeps the logged domains of existing data
- **estimation:** drillhole searches and DrillholePlan.optimize
- **estimation:** angular sectors in the search ellipsoid's major plane
- **core:** [**breaking**] dimension algebra for units
- units flow through estimates, simulations and transforms
- length units on containers, searches and variograms
- tonnage and metal with units
- parameters take text with units
- units through the remaining estimators and transforms
- units in swath, validation and bootstrap tables, and GIS readers
- more parameters take text with units
- streamed turning-bands summaries keep their units

### Fixes

- **core:** from_extents adds no layer on rotated grids
- **py:** readable error for newer model formats
- **plot3d:** motion quality thins the outer faces of masked and sub-blocked models
- **simulation:** DSS handles tied data, unsimulated domains and non-finite data
- **python:** DSS clamp share over simulated nodes; example corrects the histogram
- **simulation:** select_realizations rejects distances that overflow
- **drillholes:** inclined planned collars slide up the hole onto topography
- **eda:** required spacing is the largest spacing meeting the threshold; P50/P90 columns
- **eda:** data_spacing docs and example prose; section NaN regression test
- **simulation:** precision NaN where the mean is 0 or NaN; validate docs
- **estimation:** plan gains match re-kriging the plan with the hole added
- **estimation:** DrillholePlan refuses negative or non-integer hole indices
- **estimation:** annealing refreshes contributions on every accepted move; removal weights stay >= 0

### Performance

- **plot3d:** viewer benchmark
- **plot3d:** draw only the outer faces of masked and sub-blocked models
- **simulation:** build the DSS lookups once for all realizations
- **simulation:** spacing_study averages node truths to rows instead of simulating them twice

### Refactor

- **simulation:** run SGS through a loop generic over its units

### Documentation

- **examples:** spacing examples quantify the truth-to-truth noise behind the reference gap
- update badges in README.md
- **examples:** units in more methods, and a fresh gallery render
- **learn:** colab tour from drill holes to a 3D model

### Tests

- **plot3d:** mesh and point layers in a scene, with gallery example
## 0.3.0 - 2026-10-01

### Features

- save Transiogram, PolygonSelector, Unfold as JSON and MultivariateSimulation as Parquet
- **estimation:** cokriging with locally varying anisotropy
- **io:** read and write GeoPackage point, line and polygon layers
- **io:** read and write DXF polyface meshes, read DXF MESH and OBJ groups
- **estimation:** calibrated local search by target slope or efficiency
- Mesh.validate reports duplicate, non-manifold, winding and orientation problems per face, edge and vertex
- **learn:** colab notebooks for the learn chapters

### Fixes

- **variogram:** ignore rounding-level gains in the anisotropy descent
- **estimation:** scan and grid search measure distances in one frame so ties agree
- **notebooks:** build the python extension before generating notebooks

### Documentation

- **examples:** soft data, training images, image quilting and seismic volumes pages
- README.md changes
- prototype scaffolding for Learn, Guide, workflows, themes and teaching components
- swiss theme, learn chapters 2-6, workflows and guide rewrite
- drop the prototype page and the try-it exercises
- tighter home, install, guide, glossary and nav
- **learn:** tighten chapter prose
- **examples:** tighten workflow and case-study prose
- **examples:** tighter prose in chapters 01-11 and 13
## 0.2.0 - 2026-09-30

### Features

- **io:** read SEG-Y cubes into block models
- **io:** write block models as SEG-Y
- **py:** bind read_segy and write_segy
- **simulation:** training images from block model columns
- **simulation:** object-based training images
- **py:** bind object_training_image
- **datasets:** Strebelle training image and an F3 seismic crop
- **simulation:** consistency of a training image with the hard data
- **py:** bind training_image_consistency
- **simulation:** SNESIM with search trees, multigrid and servosystem
- **py:** bind SNESIM
- **simulation:** seam cuts for image quilting
- **simulation:** image quilting
- **py:** bind ImageQuilting
- **simulation:** soft probabilities in SNESIM
- **py:** SNESIM.simulate takes soft probabilities
- **simulation:** FFT correlations for image-quilting patch costs
- **simulation:** FFT path, soft and secondary data in image quilting
- **py:** soft and secondary data in ImageQuilting
- **simulation:** local anisotropy and per-zone training images in SNESIM
- **py:** SNESIM takes anisotropy and a training image per domain
- **simulation:** continuous training images in SNESIM through value classes
- **py:** SNESIM simulates continuous training images
- **simulation:** continuous SNESIM takes values that match their neighbors, quartile classes by default
- progress bars know their total and render as widgets in notebooks and filling bars on the docs site

### Performance

- **simulation:** bench image-quilting costs by FFT and direct scan

### Refactor

- **simulation:** let closed name the probabilities it checks
- [**breaking**] rename ceres to boitata

### Documentation

- add Codecov badge to README
- **examples:** multiple-point statistics chapter with SNESIM multigrid, conditioning and continuous pages
- **examples:** SNESIM rotation, affinity and training images by zone; black and white figures
- **examples:** sub-gallery headers and chapter-style links for multiple-point statistics
- Boitatá identity with a b-shaped voxel serpent logo, animated README mark and fire palette
- kriging-node serpent as the logo, animated on the README and site hero
- name Boitatá in prose docstrings
- README logo from a relative path so it renders while the repo is private
- README.md changes
- README.md changes

### CI

- run pytest with coverage and upload to Codecov
- run pytest under a virtual display for VTK

### Tests

- xfail the coregionalization azimuth check on Linux
- **py:** ImageQuilting.fit returns self
- **simulation:** continuous SNESIM with a training image per zone and local anisotropy
- **docs:** import mkdocs only where the hook needs it, so the hook test runs without it
## 0.1.0 - 2026-09-29

### Features

- **variogram:** port variogram crate from ctrl
- **transforms:** port transforms crate from ctrl
- **drillholes:** port drillholes crate from ctrl
- **blocks:** port blocks crate from ctrl
- **coda:** port coda crate from ctrl
- **estimation:** port estimation crate from ctrl
- **simulation:** port simulation crate from ctrl
- **modeling:** port modeling crate from ctrl
- **core:** add rotation, PointSet and BlockModel containers
- **io:** add CSV and GSLIB readers and writers
- **py:** add Python package with containers and CSV/GSLIB I/O
- **transforms:** add NscoreTable::forward for new values
- **py:** bind transforms
- **py:** bind variography
- **core:** add with_column to PointSet and BlockModel
- **estimation:** estimate many targets and leave-one-out in parallel
- **py:** bind estimation
- **simulation:** declustering weights and parallel ensembles
- **py:** bind simulation
- **py:** bind cokriging and disjunctive kriging
- **py:** bind drillholes
- **py:** bind compositional data transforms
- **py:** bind solids, polygons, domaining and block shells
- **drillholes:** merge interval tables at shared boundaries
- **py:** merge_intervals and composites without grades dropped
- **estimation:** k-d tree search neighbourhood with its own ellipsoid
- locally varying anisotropy
- **py:** turning bands step defaults to the variogram
- **io:** Parquet storage for tables, point sets and block models
- **py:** write_parquet and read_parquet
- **core:** sub-blocked block models
- **io:** store sub-blocked models in Parquet
- **py:** sub-blocked BlockModel, extents and volumes
- **simulation:** [**breaking**] stream uncertainty summaries instead of returning realizations
- **py:** implicit modelling bindings
- **py:** ceres.datasets loader
- **py:** ceres.plot on matplotlib
- EDA toolkit
- **transforms:** PCA, MAF, stepwise conditional transform and PPMT pipeline
- kriging efficiency, slope of regression and classification helpers
- **py:** geological modelling example, RBF anisotropy, GP variance and Drillholes.at
- **core:** Mesh container
- **io:** OBJ, STL and DXF mesh readers and writers
- **py:** indicator cutoff for ImplicitModel
- **io:** stream block models from Parquet in chunks
- **simulation:** out-of-core turning-bands summaries
- **blocks:** convex hull; plot.uncertain fades values by their uncertainty
- **blocks:** signed vertical distance to an open surface
- **simulation:** summaries at block support
- **variogram:** [**breaking**] covariance, correlogram and pairwise-relative estimators
- **drillholes:** whole-run, interval, residual and categorical compositing; tangential desurvey
- **estimation:** multi-pass search and high-grade restriction
- **simulation:** high-grade threshold in data units for normal-score simulators
- **drillholes:** per-grade sampled length on composites for exact metal balance
- **plot:** plot.uncertain slices block models like plot.section; radial certainty axis on the fan
- **variogram:** experimental cross-variograms and cross-covariances
- **estimation:** measurement error, neighbourhood diagnostics, k-fold from Python
- **estimation:** k-fold keeps drill holes whole
- **variogram:** fit nested structures with fixed or bounded parameters
- **plot:** weighted Q-Q, cumulative distribution and box plots by domain
- **core:** discretize a block model into nodes
- **estimation:** classification by drill-hole distance
- **py:** save and load transforms and model objects as JSON
- **py:** save and load fitted estimators as Parquet
- **py:** save and load simulators and simulation summaries as Parquet
- **variogram:** fit one anisotropic model jointly to many directions
- **eda:** summary per category, grade-tonnage from data and capping report
- **estimation:** multiple indicator kriging with order-relation correction, tails and summaries
- **blocks:** sub-blocks from prioritised meshes and regularisation between grids
- **simulation:** multivariate simulation through independent factors
- **blocks:** surface from a grid and mesh repair
- **drillholes:** check and fix collar, survey and interval tables
- **variogram:** fit a linear model of coregionalization
- **plot:** sections on any plane, slab of points with mesh traces, lines and labels
- **variogram:** reject coregionalization matrices that are not symmetric PSD
- **py:** save and load implicit models and local anisotropy
- **eda:** group and merge spatial duplicates
- **plot:** scatter-plot matrix
- **estimation:** soft domain boundaries
- **estimation:** search calibration
- **estimation:** calibrate searches by domain
- **transforms:** localized uniform conditioning over a panel grid
- **estimation:** localize multiple indicator kriging over a panel grid
- **simulation:** localize block realizations within panels
- **simulation:** simulate with a trend through stepwise conditional scores
- **simulation:** multi-pass search for SGS
- **simulation:** domains and soft boundaries for SGS
- **variogram:** fit the LMC anisotropy with its sill matrices
- **estimation:** cross-validation and diagnostics for multiple indicator kriging
- **eda:** paired-data analysis
- **simulation:** soft neighbours as grades and trend within each SGS domain
- **transforms:** impute missing variables before multivariate simulation
- **variogram:** nugget from downhole pairs
- **eda:** model against data validation
- **estimation:** block multiple indicator kriging
- **simulation:** domains and soft boundaries for TurningBands
- **py:** 3D plotting with pyvista
- **eda:** compare_models and grade_tonnage by category
- **simulation:** block support and trend when streaming turning bands
- **simulation:** grades simulated inside simulated domains
- **eda:** correlation, declustering, conditional and completeness plots; stats box and outlier fences
- **io:** point shapefile reader and writer
- **simulation:** plurigaussian with any number of latent fields and hierarchical rules
- **eda:** data spacing; category swath, proportions and anisotropy direction plots
- **core:** Polylines container for lines and polygons
- **simulation:** plurigaussian with locally varying proportions
- **io:** GeoTIFF reader and writer for 2D block models
- **io:** line and polygon shapefiles as Polylines
- **simulation:** fit plurigaussian latent variograms to indicator variograms
- **core:** Categories value object for named category codes
- **io:** Polylines as nested Arrow and Parquet
- **py:** category colours, legends and scheme= on section, slab and boxplot
- **datasets:** [**breaking**] mining datasets and HOLE_ID default
- **plot:** grade-tonnage, cross-validation and contact plots
- **plot:** keep the grade-tonnage legend clear of the curves
- **plot:** result plots gallery page
- **transforms:** smooth kernel trend with automatic bandwidth
- **transforms:** spatial despiking of tied values
- **estimation:** categorical indicator kriging
- **estimation:** declustering weights from estimation weights
- **core:** block models sized from data extents
- **estimation:** multigaussian kriging
- **variogram:** variogram sets and grid index shifts
- **transforms:** capping transform with caps fitted per domain
- **drillholes:** checks for dip sign, ID mismatches and text values
- **blocks:** smooth classes on sub-blocked models by volume
- **py:** with_column accepts text arrays
- **variogram:** downhole variogram restricted to a direction
- **simulation:** realization checks against histogram, variogram and correlations
- **variogram:** variogram volume and principal axes of continuity
- **core:** Polylines containment, distance and area; selectors honor holes
- **transforms:** spatial bootstrap of global statistics
- **simulation:** collocated cosimulation with a secondary variable in SGS
- **transforms:** kernel density and Gaussian mixture references for normal scores
- **simulation:** local category proportions for SIS
- **transforms:** spatial imputation conditioned on nearby samples
- **blocks:** remove small units, contact distance and buffer domains
- **blocks:** unfold coordinates between two bounding surfaces
- **drillholes:** ore and waste runs with mining rules, strip logs
- **variogram:** local variogram parameters and variograms along local directions
- **variogram:** intrinsic coregionalization and madogram with dissemination
- **estimation:** high-grade clamp mode and restriction ellipse
- **simulation:** correct realizations to a target distribution
- **eda:** domain change tables
- **drillholes:** split intervals where they cross a mesh
- **estimation:** external drift kriging
- **plot:** stepped sections and hole traces
- **eda:** along-hole transition matrix and MDS plot
- **eda:** soft-boundary sample statistics
- **transforms:** censored normal-score transform
- **core:** add BlockModel::corners for true block-edge rendering
- **py:** expose BlockModel.corners
- **plot:** section() draws true block edges when axis-aligned
- **simulation:** keep selected realizations
- **py:** keep= selects realizations
- **estimation:** grid of cells visited nearest first
- **simulation:** stream kept realizations to parquet, encoding while the next chunk simulates
- **estimation:** group searches that select the same neighbors
- **simulation:** lattice, multigrid path and offset templates
- **simulation:** SGS along a shared multigrid path
- **simulation:** shared-path domains, trend, passes, cosimulation and batches
- **py:** SGS path and batch options
- **io:** tqdm progress bar for GSLIB read and write
- **modeling:** tqdm progress bar for ImplicitModel predict and isosurface
- **io:** tqdm progress bar for CSV and mesh read and write
- **estimation:** tqdm progress bar for kriging predict
- **io:** tqdm progress bar for Parquet read and write
- **estimation:** tqdm progress bar for indicator and categorical predict
- **estimation:** tqdm progress bar for cokriging and disjunctive predict
- **simulation:** tqdm progress bar on every simulator

### Fixes

- make azimuth clockwise from north in anisotropy and block rotation
- **transforms:** compute the normal CDF to double precision
- **variogram:** keep anisotropy rotation through serialization
- **estimation:** clip indicator kriging to [0, 1]
- **transforms:** bound normal-score tails in probability space
- **estimation:** constrain ordinary cokriging only on variables present
- **drillholes:** correct minimum-curvature desurvey
- **simulation:** fast, anisotropic, cap-safe turning bands
- keep the first of samples sharing a location and warn with their holes
- **transforms:** stepwise class keys cannot overflow
- **drillholes:** merge_intervals raises on overlapping intervals
- **search:** index added samples in immutable tree blocks
- **estimation:** leave the nugget out of block covariance averages
- **estimation:** one sample per location across domains
- **simulation:** honour max_per_hole in SGS, TurningBands and SIS
- **py:** install hints for pip and conda
- **docs:** tutorial scripts named tutorial_NN.py
- **plot:** uncertain with category schemes
- **drillholes:** [**breaking**] outputs keep the hole column name
- **estimation:** octants follow the search ellipse, quadrants in 2D
- **examples:** tutorial 04 reads the HOLE_ID column
- **core:** repair keeps cavities wound inward
- **core:** Categories encodes numeric labels consistently
- **modeling:** isosurface honors masked models
- **transforms:** average cell declustering weights over origin offsets
- **eda:** contact bins stop at max_distance
- **simulation:** turning bands honors the nugget
- **core:** Mesh repr works for open meshes
- **blocks:** deterministic majority domains and NaN-safe vertical distance
- **examples:** topic 72 text encoding
- **simulation:** [**breaking**] realization seeds from a mixer
- **plot:** keep the true horizontal sign on y sections (#361)

### Performance

- **transforms:** kd-tree pair search for MAF
- **simulation:** benchmark turning-bands phases on a million nodes
- **simulation:** parallel batches within one SGS realization
- **variogram:** grid sweep in fixed chunks
- **blocks:** fast mesh sub-blocking
- **simulation:** reuse turning-band factorizations across realizations
- **blocks:** fast mesh distance
- **estimation:** stack Cholesky kriging kernel
- **estimation:** krige on the stack kernel up to 64 samples
- **io:** byte-stream-split floats, no float dictionaries
- **io:** encode parquet row groups and columns in parallel
- **estimation:** search a grid instead of a k-d tree
- **py:** write_parquet releases the GIL and keeps float32
- **simulation:** condition a batch of turning-bands realizations with one search
- **simulation:** sweep bands over a tile, bench many realizations, exact f64 quantiles
- **simulation:** batch the turning-bands factors of a multivariate simulation

### Refactor

- drop BlockGrid, rename pitch to rake and Nscore to NormalScore
- **py:** [**breaking**] American spelling in the Python API and gallery
- **io:** [**breaking**] nodata everywhere
- **py:** [**breaking**] resolve column names against containers
- **py:** [**breaking**] simulator columns and domain_column
- **py:** [**breaking**] diagnostics as a Table and estimator columns
- **py:** [**breaking**] variography, blocks and modeling signatures
- **py:** [**breaking**] transforms, change of support and localize
- **eda:** [**breaking**] column names for tonnage and validation
- **py:** [**breaking**] calibrate_search takes weights by array or data=
- **eda:** [**breaking**] column names for pairs and statistics
- **py:** [**breaking**] keyword-only transform weights and NaN mean grade above an empty cutoff
- **plot:** [**breaking**] plot signatures
- **simulation:** [**breaking**] categorical probabilities as (targets, categories)
- **estimation:** [**breaking**] per-target rows in continuous summaries
- **estimation:** stream the neighbor selection
- rename realizations= to keep=

### Documentation

- **examples:** add data and declustering example
- **variogram:** drop library reference in weighting doc
- **examples:** add normal-score and variography examples
- **examples:** rewrite examples on the Python API
- **examples:** add sequential Gaussian simulation example
- **coda:** drop library reference
- **examples:** add drillholes and solids examples
- **examples:** add cokriging, change of support and compositional examples
- **examples:** render pages from cell-structured scripts
- **examples:** add estimation and simulation method comparisons
- **examples:** add Parquet storage example
- **examples:** sub-block a solid's boundary
- **examples:** local bias swaths in the validation example
- mkdocs site with API reference and example gallery
- **examples:** end-to-end workflow from drill holes to a classified model
- mesh I/O in the API reference and a test that every public name is documented
- **examples:** stream a 20 million block drill-hole model; search for turning bands
- **examples:** place new features in their chapters; use them end to end in chapter 20
- **examples:** re-render chapter 19 section after the desurvey rewrite
- **examples:** re-render validation chapter against the nested variogram model
- **examples:** per-domain tables, capping report and grade-tonnage in the EDA chapter
- **examples:** re-render chapter 08 with multiple indicator kriging
- **examples:** re-render chapters 07 and 19 with sub-blocks from meshes
- **examples:** re-render chapters 13 and 17 with multivariate simulation
- **examples:** re-render chapters 07 and 19 with grid surfaces and mesh repair
- **examples:** cividis for every continuous colormap
- **examples:** re-render chapters 06, 16 and 20 with drillhole checks
- **examples:** fit chapter 08's coregionalization automatically
- **examples:** draw sections with plot.slab and plane sections
- **examples:** re-render chapters 06, 07, 15, 19 and 20 with plane sections
- **examples:** readable legend on the chapter 7 bench
- **examples:** re-render chapter 14 with implicit model persistence
- **examples:** duplicates and scatter matrix in chapter 16
- **examples:** re-render chapter 16 with duplicates and scatter matrix
- **examples:** MS/SM contact in chapter 16, soft boundary in chapter 20
- **examples:** re-render chapters 16 and 20
- **examples:** search calibration in chapter 18 against the exhaustive truth
- **examples:** uniform conditioning and localisation in chapter 9
- **examples:** localized indicator kriging in chapter 9
- **examples:** localized simulation in chapter 9
- **examples:** simulate chapter 20 in the kriging passes
- **examples:** simulate chapter 20 across hard and soft MS/SM boundaries
- **examples:** impute missing tennantite in chapter 17
- **api:** list GaussianImputer
- **examples:** list the downhole nugget in chapter 3
- subblock logo, brand palette and landing page
- **examples:** [**breaking**] tutorials and topics galleries
- **examples:** smooth trend topic
- **examples:** despiking topic
- **examples:** categorical indicator kriging topic
- **examples:** weight declustering topic
- **examples:** split variography into topics 18-20
- **examples:** block model from extents topic
- **examples:** split change of support into topics 32-35 and 37
- **examples:** split Jura estimation into topics 21, 27, 28 and 29
- **examples:** standalone Walker estimation topics 22-26
- **examples:** tutorial 2 on stacked sulphide lenses
- **examples:** regenerate index
- **examples:** tutorial 1, a coal seam in a lease
- **examples:** iron ore multivariate tutorial
- **examples:** tutorial 03 script named tutorial_03.py
- **examples:** regenerate index
- **examples:** transforms topics 14-17 and multivariate simulation 43
- **examples:** check drill holes, desurvey and compositing topics
- **examples:** re-render data topics
- **examples:** drop gallery log filter
- **examples:** duplicates, paired data, statistics by domain, declustering and top cuts topics
- **examples:** re-render topics 5 and 6
- **examples:** soft boundaries topic on the nickel laterite; standalone local anisotropy
- install page for pip, uv, conda and pixi
- **examples:** split validation into topics 44-48
- **examples:** link multigaussian topic to topic 29
- **examples:** grade shells, contact surfaces and structural data topics 52-54
- **examples:** laterite profile tutorial
- **examples:** layered surfaces and polygons topics
- **examples:** split solids into topics 49 solids, 50 sub-blocks and 51 mesh files
- **examples:** keep open meshes out of gallery backreferences
- **examples:** lens meshes arrive closed
- **examples:** contacts, swaths, categories, data spacing and correlations topics
- **examples:** index topics 9 to 13
- **examples:** split simulation into topics 36, 38 and 39
- **examples:** point topic 42 to multivariate simulation in topic 43
- **examples:** split parquet topic into parquet and GIS formats pages
- **examples:** gallery header for GIS formats page
- **examples:** gallery header for capping topic
- **examples:** capping topic on the quartz veins of topic 8
- **examples:** explain why the high-grade restriction raises nodes in topic 26
- **examples:** large models on the iron formation, 3D views on the sulphide lenses
- **examples:** re-render topics 59 and 60
- **examples:** split categorical simulation into topics 40 sis, 41 plurigaussian and 42 grades in simulated rocks
- **examples:** topic 4 unpacks the check details
- **examples:** re-render pages using cell declustering
- **examples:** re-render topic 38 with the nugget reproduced
- **examples:** re-render realization checks after rebase
- **examples:** re-render mesh topics with the open-mesh repr
- **simulation:** a kernel trend streamed to simulate_to_parquet must be filled beyond its data
- **examples:** closed lens meshes, faded most-likely panel, default turning-band step, high-grade solid described
- **examples:** re-render tutorials 02 and 03
- **examples:** re-render topic 68 and the index
- **examples:** tidy the spatial bootstrap page
- **examples:** index topic 75
- **examples:** re-render the gallery, match prose to the printed numbers, section headers
- **examples:** re-render topic 17 figures
- **examples:** re-render spatial imputation after rebase
- **examples:** topic 79 runs and strip logs
- **examples:** re-render the gallery after the seed and summary changes
- **examples:** topics 81 intrinsic coregionalization and 82 madogram
- **examples:** topic 83 high-grade restriction
- add a contributing guide and issue forms
- **examples:** topic 93 section validation plates
- **examples:** stream a kept realization in topic 59
- **examples:** SGS along a shared path
- **examples:** re-render section galleries after true-edge sections (#363)
- **examples:** group examples into 12 categories
- **examples:** name scripts example_CC_NN.py so gallery file names stay unique
- personal README tone and one API page per object
- API summaries from the first docstring sentence
- point README examples link at first steps
- **examples:** add a quick tour and move the Parquet page to slot 4
- **examples:** saving and loading value objects
- **examples:** add a block models guide
- **examples:** nugget inference page
- **examples:** from variogram to search plan
- **examples:** fix search-plan page links for the docs site
- **examples:** fix parquet link on saving page

### Build

- **py:** package metadata for PyPI and conda-forge
- draft conda-forge recipe
- environment.yml, pixi.toml and compatibility script
- **datasets:** pin closed lens meshes

### CI

- build wheels and GitHub releases
- publish wheels to PyPI as ceresgeo

### Tests

- criterion benches for kriging and simulation
- **simulation:** deterministic guard for the planar-grid search case
- **py:** API convention checks
- **py:** pass with only numpy installed
- guard against double-encoded text
- **simulation:** categorical probabilities as (targets, categories)
- **plot:** check section colors on the block collection
- **estimation:** benchmark searches among drill holes
