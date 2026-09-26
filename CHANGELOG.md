# Changelog

## 0.1.0 - 2026-09-26

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

### Performance

- **transforms:** kd-tree pair search for MAF
- **simulation:** benchmark turning-bands phases on a million nodes
- **simulation:** parallel batches within one SGS realization

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

### Build

- **py:** package metadata for PyPI and conda-forge

### Tests

- criterion benches for kriging and simulation
- **simulation:** deterministic guard for the planar-grid search case
- **py:** API convention checks
- **py:** pass with only numpy installed
