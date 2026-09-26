use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use simulation::{
    BlockSupport, CategoricalSummary as CoreCategorical, ContinuousOptions, ContinuousSummary,
    GibbsParams, PgsParams, Region, SgsParams, SisParams, TruncationRule, TurningBandsParams,
};
use variogram::Variogram as CoreVariogram;

use crate::args::{
    self, Point, array1, distinct, finite, optional_finite, pick, points, rows, same_length,
};
use crate::containers::PyBlockModel;
use crate::estimation::{Search, targets};
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};
use crate::variogram::Variogram;

fn err(e: simulation::SimError) -> PyErr {
    match e {
        simulation::SimError::Io(ceres_io::Error::Io(e)) => crate::error("FileError", e),
        e => invalid(e),
    }
}

fn not_fitted() -> PyErr {
    invalid("simulator is not fitted; call fit first")
}

/// `rows` as a `(rows, cols)` array, also when there are no rows.
fn matrix<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    rows: &[Vec<T>],
    cols: usize,
) -> Bound<'py, PyArray2<T>> {
    Array2::from_shape_vec((rows.len(), cols), rows.concat())
        .expect("rectangular rows")
        .into_pyarray(py)
}

fn int_rows(rows: &[Vec<usize>]) -> Vec<Vec<i64>> {
    rows.iter()
        .map(|r| r.iter().map(|&c| c as i64).collect())
        .collect()
}

/// Uncertainty at every target from `n` realizations of a continuous
/// variable. Per-cutoff and per-quantile arrays have one row per cutoff or
/// quantile.
#[pyclass(module = "ceres", name = "SimulationSummary", frozen)]
pub struct SimulationSummary(ContinuousSummary);

#[pymethods]
impl SimulationSummary {
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        crate::persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        crate::persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<crate::persist::Columns>)> {
        crate::persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<crate::persist::Columns>) -> PyResult<Self> {
        crate::persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

    #[getter]
    fn n(&self) -> usize {
        self.0.n
    }

    /// Mean of the realizations (E-type estimate).
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.mean.clone())
    }

    /// Variance across realizations.
    #[getter]
    fn variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.variance.clone())
    }

    #[getter]
    fn std<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.variance.iter().map(|v| v.sqrt()).collect())
    }

    #[getter]
    fn cutoffs(&self) -> Vec<f64> {
        self.0.cutoffs.clone()
    }

    /// `(cutoffs, targets)` fraction of realizations above each cutoff.
    #[getter]
    fn probability_above<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, &self.0.probability_above, self.0.mean.len())
    }

    /// `(cutoffs, targets)` mean of the values above each cutoff; NaN where
    /// no realization is above it.
    #[getter]
    fn mean_above<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, &self.0.mean_above, self.0.mean.len())
    }

    #[getter]
    fn quantiles(&self) -> Vec<f64> {
        self.0.quantiles.clone()
    }

    /// `(quantiles, targets)` values at each requested quantile.
    #[getter]
    fn quantile_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, &self.0.quantile_values, self.0.mean.len())
    }

    /// Mean of each realization over all targets, `(n,)`.
    #[getter]
    fn realization_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.realization_mean.clone())
    }

    /// `(cutoffs, n)` fraction of targets above each cutoff in each realization.
    #[getter]
    fn realization_above<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, &self.0.realization_above, self.0.n)
    }

    /// `(n, targets)` realizations when simulated with `realizations=True`.
    #[getter]
    fn realizations<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.0
            .realizations
            .as_ref()
            .map(|r| matrix(py, r, self.0.mean.len()))
    }

    fn __repr__(&self) -> String {
        format!(
            "SimulationSummary(n={}, targets={}, cutoffs={:?}, quantiles={:?})",
            self.0.n,
            self.0.mean.len(),
            self.0.cutoffs,
            self.0.quantiles
        )
    }
}

/// Uncertainty at every target from `n` realizations of categories.
#[pyclass(module = "ceres", name = "CategoricalSummary", frozen)]
pub struct CategoricalSummary(CoreCategorical);

#[pymethods]
impl CategoricalSummary {
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        crate::persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        crate::persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<crate::persist::Columns>)> {
        crate::persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<crate::persist::Columns>) -> PyResult<Self> {
        crate::persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

    #[getter]
    fn n(&self) -> usize {
        self.0.n
    }

    /// `(categories, targets)` fraction of realizations in each category.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, &self.0.probabilities, self.0.most_likely.len())
    }

    /// Most probable category per target; ties go to the lowest.
    #[getter]
    fn most_likely<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.0.most_likely.iter().map(|&c| c as i64).collect())
    }

    /// Entropy of the probabilities scaled to [0, 1]: 0 where every
    /// realization agrees, 1 where all categories are equally likely.
    #[getter]
    fn entropy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.entropy.clone())
    }

    /// `(n, categories)` share of targets in each category per realization.
    #[getter]
    fn proportions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, &self.0.proportions, self.0.probabilities.len())
    }

    /// `(n, targets)` realizations when simulated with `realizations=True`.
    #[getter]
    fn realizations<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<i64>>> {
        self.0
            .realizations
            .as_ref()
            .map(|r| matrix(py, &int_rows(r), self.0.most_likely.len()))
    }

    fn __repr__(&self) -> String {
        format!(
            "CategoricalSummary(n={}, targets={}, categories={})",
            self.0.n,
            self.0.most_likely.len(),
            self.0.probabilities.len()
        )
    }
}

/// Averages realizations at `grid` to the rows of `blocks`, each node weighted
/// by its volume when `targets` is a block model.
fn support(
    targets: &Bound<PyAny>,
    grid: &[Point],
    blocks: Option<PyRef<PyBlockModel>>,
) -> PyResult<Option<BlockSupport>> {
    let Some(blocks) = blocks else {
        return Ok(None);
    };
    let volumes = targets
        .cast::<PyBlockModel>()
        .ok()
        .map(|m| m.get().0.volumes());
    BlockSupport::new(grid, volumes.as_deref(), &blocks.0)
        .map(Some)
        .map_err(err)
}

fn averaged(support: &Option<BlockSupport>, values: Vec<f64>) -> simulation::Result<Vec<f64>> {
    match support {
        Some(s) => s.mean(&values),
        None => Ok(values),
    }
}

fn majority(
    support: &Option<BlockSupport>,
    categories: Vec<usize>,
    k: usize,
) -> simulation::Result<Vec<usize>> {
    match support {
        Some(s) => s.majority(&categories, k),
        None => Ok(categories),
    }
}

struct Data {
    locs: Vec<Point>,
    values: Vec<f64>,
    weights: Option<Vec<f64>>,
}

/// Coordinates keeping the first sample of each shared location, and the
/// kept rows; the others are reported by `holes` in a warning.
fn located(
    coords: &Bound<PyAny>,
    n: usize,
    what: &str,
    holes: Option<&Bound<PyAny>>,
) -> PyResult<(Vec<Point>, Vec<usize>)> {
    let locs = points(coords)?;
    same_length(locs.len(), n, what)?;
    let holes = args::holes(holes, locs.len())?;
    let keep = distinct(coords.py(), &locs, holes.as_ref().map(|h| &h.0[..]))?;
    Ok((pick(&locs, &keep), keep))
}

fn data(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    holes: Option<&Bound<PyAny>>,
) -> PyResult<Data> {
    let values = finite(values, "values")?;
    let weights = optional_finite(weights, "weights")?;
    if let Some(w) = &weights {
        same_length(values.len(), w.len(), "weights")?;
    }
    let (locs, keep) = located(coords, values.len(), "values", holes)?;
    Ok(Data {
        locs,
        values: pick(&values, &keep),
        weights: weights.map(|w| pick(&w, &keep)),
    })
}

/// Sequential Gaussian simulation. `variogram` is the normal-score variogram
/// (unit sill); data are normal-scored internally, with optional declustering
/// weights, and realizations are back-transformed.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "SGS")]
pub struct Sgs {
    variogram: CoreVariogram,
    search: estimation::Search,
    #[serde(skip)]
    data: Option<Data>,
}

#[pymethods]
impl Sgs {
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        crate::persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        crate::persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<crate::persist::Columns>)> {
        crate::persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<crate::persist::Columns>) -> PyResult<Self> {
        crate::persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

    #[new]
    fn new(variogram: Variogram, search: Search) -> PyResult<Self> {
        Ok(Self {
            variogram: variogram.0,
            search: search.plain("SGS")?,
            data: None,
        })
    }

    /// Samples sharing a location keep the first, with a warning naming their
    /// `holes`.
    #[pyo3(signature = (coords, values, weights=None, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.data = Some(data(coords, values, weights, holes)?);
        Ok(slf)
    }

    /// Summary of `n` realizations at `targets`, seeds `seed, seed + 1, …`,
    /// with the probability and mean above each of `cutoffs` and the values at
    /// `quantiles`; the realizations themselves only when `realizations`.
    /// `anisotropy` (a LocalAnisotropy) orients each node's variogram and search.
    /// With `blocks` (a coarser BlockModel), each realization is averaged to
    /// its blocks, weighted by node volume, and summarized at block support;
    /// nodes outside every block are ignored and a block holding no node is
    /// an error.
    #[pyo3(signature = (targets, n=100, seed=0, cutoffs=vec![], quantiles=vec![], realizations=false, anisotropy=None, blocks=None))]
    #[allow(clippy::too_many_arguments)]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        realizations: bool,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        blocks: Option<PyRef<PyBlockModel>>,
    ) -> PyResult<SimulationSummary> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let support = support(targets, &grid, blocks)?;
        let local = anisotropy.map(|a| a.at_targets(&grid));
        let options = ContinuousOptions {
            cutoffs,
            quantiles,
            keep: realizations,
        };
        py.detach(|| {
            simulation::continuous(n, &options, |k| {
                let params = SgsParams {
                    search: self.search.clone(),
                    seed: seed.wrapping_add(k as u64),
                };
                simulation::sgs(
                    &d.locs,
                    &d.values,
                    d.weights.as_deref(),
                    &grid,
                    &self.variogram,
                    &params,
                    local.as_ref(),
                )
                .and_then(|r| averaged(&support, r.values))
            })
        })
        .map(SimulationSummary)
        .map_err(err)
    }
}

/// Turning-bands simulation conditioned by kriging; same conventions as SGS.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "TurningBands")]
pub struct TurningBands {
    variogram: CoreVariogram,
    bands: usize,
    step: Option<f64>,
    search: Option<estimation::Search>,
    #[serde(skip)]
    data: Option<Data>,
}

#[pymethods]
impl TurningBands {
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        crate::persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        crate::persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<crate::persist::Columns>)> {
        crate::persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<crate::persist::Columns>) -> PyResult<Self> {
        crate::persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

    /// `bands` lines, each discretized every `step` metres along the major axis
    /// (default: a fiftieth of the shortest range). `search` is the
    /// neighbourhood of the conditioning kriging (default: the 32 nearest
    /// data at any distance); a radius near the range skips nodes far from
    /// the data, where conditioning changes nothing.
    #[new]
    #[pyo3(signature = (variogram, bands=300, step=None, search=None))]
    fn new(
        variogram: Variogram,
        bands: usize,
        step: Option<f64>,
        search: Option<Search>,
    ) -> PyResult<Self> {
        Ok(Self {
            variogram: variogram.0,
            bands,
            step,
            search: search.map(|s| s.plain("TurningBands")).transpose()?,
            data: None,
        })
    }

    /// Samples sharing a location keep the first, with a warning naming their
    /// `holes`.
    #[pyo3(signature = (coords, values, weights=None, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.data = Some(data(coords, values, weights, holes)?);
        Ok(slf)
    }

    /// Summary of `n` realizations; same options as `SGS.simulate`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (targets, n=100, seed=0, cutoffs=vec![], quantiles=vec![], realizations=false, blocks=None))]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        realizations: bool,
        blocks: Option<PyRef<PyBlockModel>>,
    ) -> PyResult<SimulationSummary> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let support = support(targets, &grid, blocks)?;
        let options = ContinuousOptions {
            cutoffs,
            quantiles,
            keep: realizations,
        };
        let (lo, hi) = simulation::bounds(&grid);
        py.detach(|| {
            let ensemble = simulation::TurningBandsEnsemble::new(
                &d.locs,
                &d.values,
                d.weights.as_deref(),
                lo,
                hi,
                &self.variogram,
                &self.params(seed),
                n,
            )?;
            simulation::continuous(n, &options, |k| {
                averaged(&support, ensemble.realization(k, &grid)?)
            })
        })
        .map(SimulationSummary)
        .map_err(err)
    }

    /// Summary of `n` realizations over the block model file `path`, written
    /// to `out` chunk by chunk with the input columns: `mean`, `variance`,
    /// `p_above_<c>` and `mean_above_<c>` per cutoff, `q<p>` per quantile.
    /// The same values as `simulate` on the whole model, in memory bounded by
    /// `rows` blocks plus the bands. Returns each realization's global
    /// `realization_mean` and `realization_above` (one row per cutoff).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, out, n=100, seed=0, cutoffs=vec![], quantiles=vec![], rows=1_000_000))]
    fn simulate_to_parquet<'py>(
        &self,
        py: Python<'py>,
        path: std::path::PathBuf,
        out: std::path::PathBuf,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        rows: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let options = ContinuousOptions {
            cutoffs,
            quantiles,
            keep: false,
        };
        let global = py
            .detach(|| {
                simulation::turning_bands_to_parquet(
                    path,
                    out,
                    &d.locs,
                    &d.values,
                    d.weights.as_deref(),
                    &self.variogram,
                    &self.params(seed),
                    n,
                    &options,
                    rows,
                )
            })
            .map_err(err)?;
        let result = pyo3::types::PyDict::new(py);
        result.set_item("realization_mean", array1(py, global.realization_mean))?;
        result.set_item(
            "realization_above",
            matrix(py, &global.realization_above, n),
        )?;
        Ok(result)
    }
}

impl TurningBands {
    fn params(&self, seed: u64) -> TurningBandsParams {
        let defaults = TurningBandsParams::default();
        TurningBandsParams {
            n_bands: self.bands,
            step: self.step,
            seed,
            search: self.search.clone().unwrap_or(defaults.search),
        }
    }
}

fn categories(obj: &Bound<PyAny>) -> PyResult<Vec<usize>> {
    obj.py()
        .import("numpy")?
        .call_method1("asarray", (obj, "int64"))?
        .call_method0("tolist")?
        .extract()
        .map_err(|_| invalid("categories must be non-negative integers"))
}

/// Sequential indicator simulation of categories `0..k`, one indicator
/// variogram per category.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "SIS")]
pub struct Sis {
    variograms: Vec<CoreVariogram>,
    search: estimation::Search,
    #[serde(skip)]
    data: Option<(Vec<Point>, Vec<usize>)>,
}

#[pymethods]
impl Sis {
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        crate::persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        crate::persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<crate::persist::Columns>)> {
        crate::persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<crate::persist::Columns>) -> PyResult<Self> {
        crate::persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

    #[new]
    fn new(variograms: Vec<Variogram>, search: Search) -> PyResult<Self> {
        Ok(Self {
            variograms: variograms.into_iter().map(|v| v.0).collect(),
            search: search.plain("SIS")?,
            data: None,
        })
    }

    #[pyo3(signature = (coords, categories, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        categories: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let cats = self::categories(categories)?;
        let (locs, keep) = located(coords, cats.len(), "categories", holes)?;
        let cats = pick(&cats, &keep);
        if cats.iter().any(|&c| c >= slf.variograms.len()) {
            return Err(invalid("every category needs a variogram"));
        }
        slf.data = Some((locs, cats));
        Ok(slf)
    }

    /// Summary of `n` realizations, seeds `seed, seed + 1, …`; the
    /// realizations themselves only when `realizations`. With `blocks` (a
    /// coarser BlockModel), each block takes the category filling most of its
    /// node volume, ties to the smallest, as in `BlockModel.regularize`; blocks as in
    /// `SGS.simulate`.
    #[pyo3(signature = (targets, n=100, seed=0, realizations=false, blocks=None))]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        realizations: bool,
        blocks: Option<PyRef<PyBlockModel>>,
    ) -> PyResult<CategoricalSummary> {
        let (locs, cats) = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let support = support(targets, &grid, blocks)?;
        let k = self.variograms.len();
        py.detach(|| {
            simulation::categorical(n, k, realizations, |i| {
                let params = SisParams {
                    search: self.search.clone(),
                    seed: seed.wrapping_add(i as u64),
                };
                simulation::sis(locs, cats, &grid, k, &self.variograms, &params)
                    .and_then(|r| majority(&support, r.categories, k))
            })
        })
        .map(CategoricalSummary)
        .map_err(err)
    }
}

/// Plurigaussian simulation. Give `proportions` for an ordered rule on one
/// field, or `regions` as `(y1_low, y1_high, y2_low, y2_high, facies)`.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "Plurigaussian")]
pub struct Plurigaussian {
    variograms: (CoreVariogram, CoreVariogram),
    rule: TruncationRule,
    two_fields: bool,
    #[serde(skip)]
    data: Option<(Vec<Point>, Vec<usize>)>,
}

#[pymethods]
impl Plurigaussian {
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        crate::persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        crate::persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<crate::persist::Columns>)> {
        crate::persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<crate::persist::Columns>) -> PyResult<Self> {
        crate::persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

    #[new]
    #[pyo3(signature = (variogram, second_variogram=None, proportions=None, regions=None))]
    fn new(
        variogram: Variogram,
        second_variogram: Option<Variogram>,
        proportions: Option<Vec<f64>>,
        regions: Option<Vec<(f64, f64, f64, f64, usize)>>,
    ) -> PyResult<Self> {
        let rule = match (proportions, regions) {
            (Some(p), None) => TruncationRule::from_proportions(&p),
            (None, Some(r)) => TruncationRule {
                regions: r
                    .into_iter()
                    .map(|(a, b, c, d, facies)| Region {
                        y1: (a, b),
                        y2: (c, d),
                        facies,
                    })
                    .collect(),
            },
            _ => return Err(invalid("give exactly one of proportions or regions")),
        };
        let two_fields = second_variogram.is_some();
        let second = second_variogram.unwrap_or_else(|| variogram.clone());
        Ok(Self {
            variograms: (variogram.0, second.0),
            rule,
            two_fields,
            data: None,
        })
    }

    #[pyo3(signature = (coords, facies, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        facies: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let facies = categories(facies)?;
        let (locs, keep) = located(coords, facies.len(), "facies", holes)?;
        let facies = pick(&facies, &keep);
        slf.data = Some((locs, facies));
        Ok(slf)
    }

    /// Summary of `n` realizations; same options as `SIS.simulate`.
    #[pyo3(signature = (targets, n=100, seed=0, realizations=false, blocks=None))]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        realizations: bool,
        blocks: Option<PyRef<PyBlockModel>>,
    ) -> PyResult<CategoricalSummary> {
        let (locs, facies) = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let support = support(targets, &grid, blocks)?;
        let k = self
            .rule
            .regions
            .iter()
            .map(|r| r.facies + 1)
            .max()
            .unwrap_or(1);
        py.detach(|| {
            simulation::categorical(n, k, realizations, |i| {
                let params = PgsParams {
                    seed: seed.wrapping_add(i as u64),
                    two_fields: self.two_fields,
                    ..Default::default()
                };
                simulation::plurigaussian(
                    locs,
                    facies,
                    &grid,
                    &self.variograms.0,
                    &self.variograms.1,
                    &self.rule,
                    &params,
                )
                .and_then(|f| majority(&support, f, k))
            })
        })
        .map(CategoricalSummary)
        .map_err(err)
    }
}

/// One Gaussian draw at `coords` honouring `bounds` (`(n, 2)` lower/upper).
#[pyfunction]
#[pyo3(signature = (coords, bounds, variogram, iterations=200, burn_in=50, seed=1))]
fn gibbs<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    bounds: &Bound<PyAny>,
    variogram: Variogram,
    iterations: usize,
    burn_in: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let locs = points(coords)?;
    let bounds: Vec<(f64, f64)> = rows(bounds, "bounds")?
        .into_iter()
        .map(|r| match r[..] {
            [lo, hi] => Ok((lo, hi)),
            _ => Err(invalid("bounds must have shape (n, 2)")),
        })
        .collect::<PyResult<_>>()?;
    same_length(locs.len(), bounds.len(), "bounds")?;
    let params = GibbsParams {
        iterations,
        burn_in,
        seed,
    };
    let out = simulation::gibbs(&locs, &bounds, &variogram.0, &params).map_err(err)?;
    Ok(array1(py, out).into_any())
}

fn data_columns(d: &Data) -> Columns {
    let mut columns = persist::point_columns(d.locs.iter().copied());
    columns.push(persist::column("value", d.values.iter().copied()));
    let weights = (0..d.values.len())
        .map(|i| d.weights.as_ref().map(|w| w[i]))
        .collect();
    columns.push(("weight".into(), weights));
    columns
}

fn data_from(found: &Found) -> PyResult<Data> {
    let (locs, values, weights) = (
        found.points()?,
        found.values("value")?,
        found.optional("weight")?,
    );
    same_length(locs.len(), values.len(), "value")?;
    Ok(Data {
        locs,
        values,
        weights: weights.into_iter().collect(),
    })
}

fn category_columns((locs, categories): &(Vec<Point>, Vec<usize>)) -> Columns {
    let mut columns = persist::point_columns(locs.iter().copied());
    let categories = categories.iter().map(|&c| c as f64);
    columns.push(persist::column("category", categories));
    columns
}

fn categories_from(found: &Found, k: usize) -> PyResult<(Vec<Point>, Vec<usize>)> {
    let (locs, categories) = (found.points()?, found.indices("category")?);
    same_length(locs.len(), categories.len(), "category")?;
    if categories.iter().any(|&c| c >= k) {
        return Err(invalid("category out of range"));
    }
    Ok((locs, categories))
}

impl Tabular for Sgs {
    fn columns(&self) -> Option<Columns> {
        self.data.as_ref().map(data_columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.data = Some(data_from(&columns)?);
        Ok(())
    }
}

impl Tabular for TurningBands {
    fn columns(&self) -> Option<Columns> {
        self.data.as_ref().map(data_columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.data = Some(data_from(&columns)?);
        Ok(())
    }
}

impl Tabular for Sis {
    fn columns(&self) -> Option<Columns> {
        self.data.as_ref().map(category_columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.data = Some(categories_from(&columns, self.variograms.len())?);
        Ok(())
    }
}

impl Tabular for Plurigaussian {
    fn columns(&self) -> Option<Columns> {
        self.data.as_ref().map(category_columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.data = Some(categories_from(&columns, usize::MAX)?);
        Ok(())
    }
}

/// The per-realization part of a summary; per-target arrays are columns.
#[derive(Serialize, Deserialize)]
struct ContinuousMeta {
    n: usize,
    #[serde(with = "ceres_core::nonfinite")]
    cutoffs: Vec<f64>,
    #[serde(with = "ceres_core::nonfinite")]
    quantiles: Vec<f64>,
    #[serde(with = "ceres_core::nonfinite")]
    realization_mean: Vec<f64>,
    #[serde(with = "ceres_core::nonfinite")]
    realization_above: Vec<Vec<f64>>,
    realizations: bool,
}

impl Serialize for SimulationSummary {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let c = &self.0;
        ContinuousMeta {
            n: c.n,
            cutoffs: c.cutoffs.clone(),
            quantiles: c.quantiles.clone(),
            realization_mean: c.realization_mean.clone(),
            realization_above: c.realization_above.clone(),
            realizations: c.realizations.is_some(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for SimulationSummary {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let m = ContinuousMeta::deserialize(d)?;
        if m.realization_mean.len() != m.n
            || m.realization_above.len() != m.cutoffs.len()
            || m.realization_above.iter().any(|r| r.len() != m.n)
        {
            return Err(serde::de::Error::custom(
                "per-realization arrays need n values",
            ));
        }
        Ok(Self(ContinuousSummary {
            n: m.n,
            mean: vec![],
            variance: vec![],
            cutoffs: m.cutoffs,
            probability_above: vec![],
            mean_above: vec![],
            quantiles: m.quantiles,
            quantile_values: vec![],
            realization_mean: m.realization_mean,
            realization_above: m.realization_above,
            realizations: m.realizations.then(Vec::new),
        }))
    }
}

impl Tabular for SimulationSummary {
    fn columns(&self) -> Option<Columns> {
        let c = &self.0;
        let column = |name: String, v: &Vec<f64>| persist::column(&name, v.iter().copied());
        let mut out = vec![
            column("mean".into(), &c.mean),
            column("variance".into(), &c.variance),
        ];
        for (i, cut) in c.cutoffs.iter().enumerate() {
            out.push(column(format!("p_above_{cut}"), &c.probability_above[i]));
            out.push(column(format!("mean_above_{cut}"), &c.mean_above[i]));
        }
        for (q, values) in c.quantiles.iter().zip(&c.quantile_values) {
            out.push(column(format!("q{q}"), values));
        }
        for (i, r) in c.realizations.iter().flatten().enumerate() {
            out.push(column(format!("realization_{i}"), r));
        }
        Some(out)
    }

    fn restore(&mut self, found: Found) -> PyResult<()> {
        let c = &mut self.0;
        let each = |names: Vec<String>| -> PyResult<Vec<Vec<f64>>> {
            names.iter().map(|n| found.values(n)).collect()
        };
        c.mean = found.values("mean")?;
        c.variance = found.values("variance")?;
        c.probability_above = each(c.cutoffs.iter().map(|x| format!("p_above_{x}")).collect())?;
        c.mean_above = each(
            c.cutoffs
                .iter()
                .map(|x| format!("mean_above_{x}"))
                .collect(),
        )?;
        c.quantile_values = each(c.quantiles.iter().map(|q| format!("q{q}")).collect())?;
        if c.realizations.is_some() {
            c.realizations = Some(each(
                (0..c.n).map(|i| format!("realization_{i}")).collect(),
            )?);
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct CategoricalMeta {
    n: usize,
    categories: usize,
    #[serde(with = "ceres_core::nonfinite")]
    proportions: Vec<Vec<f64>>,
    realizations: bool,
}

impl Serialize for CategoricalSummary {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let c = &self.0;
        CategoricalMeta {
            n: c.n,
            categories: c.probabilities.len(),
            proportions: c.proportions.clone(),
            realizations: c.realizations.is_some(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for CategoricalSummary {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let m = CategoricalMeta::deserialize(d)?;
        if m.proportions.len() != m.n || m.proportions.iter().any(|p| p.len() != m.categories) {
            return Err(serde::de::Error::custom(
                "proportions need n rows of categories",
            ));
        }
        Ok(Self(CoreCategorical {
            n: m.n,
            probabilities: vec![vec![]; m.categories],
            most_likely: vec![],
            entropy: vec![],
            proportions: m.proportions,
            realizations: m.realizations.then(Vec::new),
        }))
    }
}

impl Tabular for CategoricalSummary {
    fn columns(&self) -> Option<Columns> {
        let c = &self.0;
        let mut out = vec![
            persist::column("most_likely", c.most_likely.iter().map(|&k| k as f64)),
            persist::column("entropy", c.entropy.iter().copied()),
        ];
        for (k, p) in c.probabilities.iter().enumerate() {
            out.push(persist::column(
                &format!("probability_{k}"),
                p.iter().copied(),
            ));
        }
        for (i, r) in c.realizations.iter().flatten().enumerate() {
            let r = r.iter().map(|&k| k as f64);
            out.push(persist::column(&format!("realization_{i}"), r));
        }
        Some(out)
    }

    fn restore(&mut self, found: Found) -> PyResult<()> {
        let c = &mut self.0;
        c.most_likely = found.indices("most_likely")?;
        c.entropy = found.values("entropy")?;
        c.probabilities = (0..c.probabilities.len())
            .map(|k| found.values(&format!("probability_{k}")))
            .collect::<PyResult<_>>()?;
        if c.realizations.is_some() {
            let realizations = (0..c.n)
                .map(|i| found.indices(&format!("realization_{i}")))
                .collect::<PyResult<_>>()?;
            c.realizations = Some(realizations);
        }
        Ok(())
    }
}

enum Factor {
    Sgs(CoreVariogram, estimation::Search),
    Bands(CoreVariogram, TurningBandsParams),
}

struct Factors {
    transform: simulation::Decorrelation,
    locs: Vec<Point>,
    columns: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
}

/// Several correlated variables simulated through independent factors.
///
/// Every realization simulates each factor with its own seed, back-transforms
/// the factors together at the nodes, and only then averages to blocks.
///
/// Parameters
/// ----------
/// transform : PCA, MAF, StepwiseConditional or PPMT
///     Template, fitted afresh by `fit`, that turns the variables into
///     factors. PPMT, which also makes them Gaussian and takes declustering
///     weights, is the usual choice.
/// simulators : sequence of SGS or TurningBands
///     One per factor, in factor order, each with the normal-score variogram
///     of its factor.
#[pyclass(module = "ceres", name = "MultivariateSimulation")]
pub struct MultivariateSimulation {
    transform: Py<PyAny>,
    factors: Vec<Factor>,
    fitted: Option<Factors>,
}

#[pymethods]
impl MultivariateSimulation {
    #[new]
    fn new(transform: &Bound<PyAny>, simulators: Vec<Bound<PyAny>>) -> PyResult<Self> {
        if !crate::transforms::is_decorrelation(transform) {
            return Err(invalid(crate::transforms::NOT_DECORRELATION));
        }
        if simulators.is_empty() {
            return Err(invalid("need one simulator per factor"));
        }
        let factors = simulators
            .iter()
            .map(|s| {
                if let Ok(s) = s.cast::<Sgs>() {
                    let s = s.borrow();
                    Ok(Factor::Sgs(s.variogram.clone(), s.search.clone()))
                } else if let Ok(s) = s.cast::<TurningBands>() {
                    let s = s.borrow();
                    Ok(Factor::Bands(s.variogram.clone(), s.params(0)))
                } else {
                    Err(invalid("simulators must be SGS or TurningBands"))
                }
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            transform: transform.clone().unbind(),
            factors,
            fitted: None,
        })
    }

    /// Fits the transform to the data, then each simulator to its factor.
    /// Samples missing a variable are dropped with a warning; of samples
    /// sharing a location, the first is kept, with a warning naming their
    /// holes.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (n, 2) or (n, 3)
    /// data : array_like, shape (n, variables)
    ///     One column per simulator; NaN marks a missing variable.
    /// weights : array_like, optional
    ///     Declustering weights, for the transform when it takes them (PCA,
    ///     PPMT) and for each factor's normal scores.
    /// holes : array_like, optional
    ///     Drill-hole ids or names, for `max_per_hole`.
    #[pyo3(signature = (coords, data, weights=None, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        data: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = coords.py();
        let data = rows(data, "data")?;
        let p = slf.factors.len();
        if data.iter().any(|r| r.len() != p) {
            return Err(invalid(format!(
                "data must have {p} columns, one per simulator"
            )));
        }
        let locs = points(coords)?;
        same_length(locs.len(), data.len(), "data")?;
        let weights = optional_finite(weights, "weights")?;
        if let Some(w) = &weights {
            same_length(data.len(), w.len(), "weights")?;
        }
        let holes = args::holes(holes, data.len())?.map(|h| h.0);
        let complete: Vec<usize> = (0..data.len())
            .filter(|&i| data[i].iter().all(|v| v.is_finite()))
            .collect();
        let missing = data.len() - complete.len();
        if missing > 0 {
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            let message = format!("{missing} samples miss a variable; dropped them");
            PyErr::warn(py, &category, &std::ffi::CString::new(message)?, 1)?;
        }
        let (data, locs) = (pick(&data, &complete), pick(&locs, &complete));
        let weights = weights.map(|w| pick(&w, &complete));
        let holes = holes.map(|h| pick(&h, &complete));
        let transform = crate::transforms::decorrelation(
            slf.transform.bind(py),
            &data,
            weights.as_deref(),
            &locs,
        )?;
        let factors = transform.forward(&data);
        let keep = distinct(py, &locs, holes.as_deref())?;
        let factors = pick(&factors, &keep);
        slf.fitted = Some(Factors {
            transform,
            locs: pick(&locs, &keep),
            columns: (0..p)
                .map(|j| factors.iter().map(|r| r[j]).collect())
                .collect(),
            weights: weights.map(|w| pick(&w, &keep)),
        });
        Ok(slf)
    }

    /// Summaries of `n` realizations of every variable.
    ///
    /// Parameters are those of `SGS.simulate`; `anisotropy` needs SGS for
    /// every factor. Factor `j` of realization `k` is simulated with a seed
    /// mixed from `seed + k` and `j`, so no two factors share random numbers.
    ///
    /// Returns
    /// -------
    /// list of SimulationSummary
    ///     One per variable, in column order.
    #[pyo3(signature = (targets, n=100, seed=0, cutoffs=vec![], quantiles=vec![], realizations=false, anisotropy=None, blocks=None))]
    #[allow(clippy::too_many_arguments)]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        realizations: bool,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        blocks: Option<PyRef<PyBlockModel>>,
    ) -> PyResult<Vec<SimulationSummary>> {
        let f = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let support = support(targets, &grid, blocks)?;
        if anisotropy.is_some() && self.factors.iter().any(|f| matches!(f, Factor::Bands(..))) {
            return Err(invalid("anisotropy needs SGS for every factor"));
        }
        let local = anisotropy.map(|a| a.at_targets(&grid));
        let options = ContinuousOptions {
            cutoffs,
            quantiles,
            keep: realizations,
        };
        py.detach(|| {
            simulation::multivariate(
                n,
                seed,
                self.factors.len(),
                &f.transform,
                &options,
                support.as_ref(),
                |j, seed| {
                    let (locs, values, weights) = (&f.locs, &f.columns[j], f.weights.as_deref());
                    Ok(match &self.factors[j] {
                        Factor::Sgs(variogram, search) => {
                            let params = SgsParams {
                                search: search.clone(),
                                seed,
                            };
                            simulation::sgs(
                                locs,
                                values,
                                weights,
                                &grid,
                                variogram,
                                &params,
                                local.as_ref(),
                            )?
                        }
                        Factor::Bands(variogram, params) => {
                            let params = TurningBandsParams {
                                seed,
                                ..params.clone()
                            };
                            simulation::turning_bands(
                                locs, values, weights, &grid, variogram, &params,
                            )?
                        }
                    }
                    .values)
                },
            )
        })
        .map(|s| s.into_iter().map(SimulationSummary).collect())
        .map_err(err)
    }

    fn __repr__(&self) -> String {
        format!(
            "MultivariateSimulation(factors={}, fitted={})",
            self.factors.len(),
            self.fitted.is_some()
        )
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<MultivariateSimulation>()?;
    m.add_class::<Sgs>()?;
    m.add_class::<TurningBands>()?;
    m.add_class::<Sis>()?;
    m.add_class::<Plurigaussian>()?;
    m.add_function(wrap_pyfunction!(gibbs, m)?)?;
    m.add_class::<SimulationSummary>()?;
    m.add_class::<CategoricalSummary>()?;
    Ok(())
}
