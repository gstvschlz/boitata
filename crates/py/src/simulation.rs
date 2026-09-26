use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use simulation::{
    BlockSupport, CategoricalSummary as CoreCategorical, ContinuousOptions, ContinuousSummary,
    GibbsParams, Hierarchy, PgsParams, Region, SgsParams, SisParams, TruncationRule,
    TurningBandsParams,
};
use variogram::Variogram as CoreVariogram;

use crate::args::{
    self, Point, array1, distinct, finite, optional_finite, pick, points, rows, same_length,
};
use crate::containers::PyBlockModel;
use crate::estimation::{Label, Search, codes, fit_codes, labels, searches, targets};
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
    holes: Option<Vec<u32>>,
    trend: Option<Vec<f64>>,
    domains: Option<Vec<u32>>,
}

impl Data {
    /// Fails where the data cannot be transformed within each domain and
    /// `classes` trend classes.
    fn check(&self, classes: usize) -> PyResult<()> {
        let trend = self.trend.as_deref().map(|t| (t, classes));
        simulation::Transforms::fit(
            &self.values,
            self.weights.as_deref(),
            self.domains.as_deref(),
            trend,
        )
        .map(|_| ())
        .map_err(err)
    }
}

fn classes() -> usize {
    10
}

/// The trend at the `nodes` targets, given or read from the `targets` column
/// it names, when fitted with a trend.
fn trend_at(
    d: &Data,
    targets: &Bound<PyAny>,
    nodes: usize,
    trend: Option<&Bound<PyAny>>,
) -> PyResult<Option<Vec<f64>>> {
    let trend = match (&d.trend, trend) {
        (None, None) => return Ok(None),
        (Some(_), Some(t)) => t,
        (Some(_), None) => return Err(invalid("fitted with a trend; give trend at the targets")),
        (None, Some(_)) => return Err(invalid("trend at the targets needs trend at fit")),
    };
    let at_nodes = if trend.is_instance_of::<pyo3::types::PyString>() {
        let model = targets
            .cast::<PyBlockModel>()
            .map_err(|_| invalid("a trend column name needs BlockModel targets"))?;
        finite(&model.get_item(trend)?, "trend")?
    } else {
        finite(trend, "trend")?
    };
    same_length(nodes, at_nodes.len(), "trend")?;
    Ok(Some(at_nodes))
}

/// Coordinates keeping the first sample of each location shared within a
/// domain of `domains`, the kept rows and their hole codes; the others are
/// reported by `holes` in a warning.
fn located(
    coords: &Bound<PyAny>,
    n: usize,
    what: &str,
    holes: Option<&Bound<PyAny>>,
    domains: Option<&[u32]>,
) -> PyResult<(Vec<Point>, Vec<usize>, Option<Vec<u32>>)> {
    let locs = points(coords)?;
    same_length(locs.len(), n, what)?;
    let holes = args::holes(holes, locs.len())?;
    let names = holes.as_ref().map(|h| &h.0[..]);
    let keep = args::distinct_in(coords.py(), &locs, names, domains)?;
    let codes = holes.map(|h| pick(&h.1, &keep));
    Ok((pick(&locs, &keep), keep, codes))
}

/// The data of `fit` and the labels of their `domains`, which every soft
/// boundary of `search` must name.
#[allow(clippy::too_many_arguments)]
fn data(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    holes: Option<&Bound<PyAny>>,
    trend: Option<&Bound<PyAny>>,
    domains: Option<&Bound<PyAny>>,
    classes: usize,
    search: &[Search],
) -> PyResult<(Data, Option<Vec<Label>>)> {
    let (fitted, codes) = match domains {
        None => (None, None),
        Some(obj) => {
            let (fitted, codes) = fit_codes(obj, values.len()?)?;
            (Some(fitted), Some(codes))
        }
    };
    for s in search {
        s.resolve(fitted.as_deref())?;
    }
    let values = finite(values, "values")?;
    let weights = optional_finite(weights, "weights")?;
    if let Some(w) = &weights {
        same_length(values.len(), w.len(), "weights")?;
    }
    let trend = optional_finite(trend, "trend")?;
    if let Some(t) = &trend {
        same_length(values.len(), t.len(), "trend")?;
    }
    let codes = codes.as_deref();
    let (locs, keep, holes) = located(coords, values.len(), "values", holes, codes)?;
    let data = Data {
        locs,
        values: pick(&values, &keep),
        weights: weights.map(|w| pick(&w, &keep)),
        holes,
        trend: trend.map(|t| pick(&t, &keep)),
        domains: codes.map(|d| pick(d, &keep)),
    };
    data.check(classes)?;
    Ok((data, fitted))
}

/// The searches with soft boundaries by the code of the `fitted` domains.
fn resolved(search: &[Search], fitted: Option<&[Label]>) -> PyResult<Vec<estimation::Search>> {
    search.iter().map(|s| s.resolve(fitted)).collect()
}

/// Domain codes of `n` nodes labelled by `obj` in the `fitted` domains, for
/// `method`.
fn node_domains(
    fitted: Option<&[Label]>,
    obj: Option<&Bound<PyAny>>,
    n: usize,
    method: &str,
) -> PyResult<Option<Vec<u32>>> {
    codes(fitted, obj, n, method)?
        .map(|codes| {
            codes
                .into_iter()
                .zip(labels(obj.expect("given with codes"), Some(n))?)
                .map(|(c, l)| c.ok_or_else(|| invalid(format!("domain {l} has no samples"))))
                .collect()
        })
        .transpose()
}

/// Domain codes of `nodes` nodes for each of `n` realizations, for `method`:
/// one row shared by all from labels as in [`node_domains`], or one row per
/// realization from an `(n, nodes)` array of simulated domains.
fn realization_domains(
    fitted: Option<&[Label]>,
    obj: Option<&Bound<PyAny>>,
    nodes: usize,
    n: usize,
    method: &str,
) -> PyResult<Option<Vec<Vec<u32>>>> {
    let np = obj.map(|o| o.py().import("numpy")).transpose()?;
    let simulated = match (obj, &np) {
        (Some(o), Some(np)) => np.call_method1("ndim", (o,))?.extract::<usize>()? == 2,
        _ => false,
    };
    if !simulated {
        return Ok(node_domains(fitted, obj, nodes, method)?.map(|c| vec![c]));
    }
    let array = np.expect("given").call_method1("asarray", (obj,))?;
    let rows = array.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if rows.len() != n {
        return Err(invalid(format!(
            "domains: expected {n} realizations of domains, got {}",
            rows.len()
        )));
    }
    rows.iter()
        .map(|row| node_domains(fitted, Some(row), nodes, method))
        .collect()
}

/// Domain codes of realization `k`'s nodes among [`realization_domains`].
fn of_realization(rows: &Option<Vec<Vec<u32>>>, k: usize) -> Option<&[u32]> {
    rows.as_ref().map(|r| &r[k % r.len()][..])
}

/// Sequential Gaussian simulation. `variogram` is the normal-score variogram
/// (unit sill); data are normal-scored internally, with optional declustering
/// weights, and realizations are back-transformed.
///
/// With a trend at `fit`, the data are normal-scored within `classes`
/// equal-probability classes of the trend (a stepwise conditional transform
/// of ``[trend, values]``); `variogram` is then the variogram of those
/// scores, and each node is back-transformed within the class of its trend.
///
/// `search` is a Search, or a sequence of them as passes: each node takes
/// the first that finds `min_samples` among the data, as kriging by passes
/// does, or the last when none does, and is simulated from that pass's
/// neighbours among the data and the nodes already simulated.
///
/// With `domains` at `fit`, each domain is transformed on its own, with the
/// declustering weights and within its own trend classes, and its nodes are
/// back-transformed through its transform; `variogram` is the one
/// normal-score variogram of every domain. Search.soft lets data and nodes
/// already simulated of another domain inform a node within the soft
/// distance, each as its grade (and trend) transformed through the node's
/// domain; boundaries are hard without it. `high_grade` compares grades, of
/// data and of simulated nodes, with its threshold. One random path visits
/// the nodes of every domain. Simulated domains at `simulate`, one
/// realization of the domains per realization of the grades, carry the
/// uncertainty of the domains into the grades.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "SGS")]
pub struct Sgs {
    variogram: CoreVariogram,
    #[serde(deserialize_with = "one_or_more")]
    search: Vec<Search>,
    #[serde(default = "classes")]
    classes: usize,
    /// Labels of the fitted domains, indexed by code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domains: Option<Vec<Label>>,
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
    #[pyo3(signature = (variogram, search, classes=10))]
    fn new(variogram: Variogram, search: &Bound<PyAny>, classes: usize) -> PyResult<Self> {
        Ok(Self {
            variogram: variogram.0,
            search: searches(search)?,
            classes,
            domains: None,
            data: None,
        })
    }

    /// The search pass of every target.
    ///
    /// A target takes the first search that finds `min_samples` among the
    /// data, whatever the realization; with the same searches, data, holes
    /// and variogram anisotropy this is the ``"pass"`` of kriging's
    /// ``predict(diagnostics=True)``.
    ///
    /// Parameters
    /// ----------
    /// targets : array_like or BlockModel
    /// anisotropy : LocalAnisotropy, optional
    ///     As in `simulate`.
    /// domains : array_like or label, optional
    ///     As in `simulate`.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     The pass, 1-based; NaN where no search finds enough data, and
    ///     `simulate` uses the last.
    #[pyo3(signature = (targets, anisotropy=None, domains=None))]
    fn passes<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        domains: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let nodes = node_domains(self.domains.as_deref(), domains, grid.len(), "passes")?;
        let search = resolved(&self.search, self.domains.as_deref())?;
        let local = anisotropy.map(|a| a.at_targets(&grid));
        let passes = py
            .detach(|| {
                simulation::sgs_passes(
                    &d.locs,
                    &d.values,
                    d.holes.as_deref(),
                    zoned(d, &nodes),
                    &grid,
                    &self.variogram,
                    &search,
                    local.as_ref(),
                )
            })
            .map_err(err)?;
        let passes = passes
            .into_iter()
            .map(|p| p.map_or(f64::NAN, |p| (p + 1) as f64));
        Ok(array1(py, passes.collect()))
    }

    /// Takes the conditioning data. Samples sharing a location keep the
    /// first, with a warning naming their `holes`.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (n, 2) or (n, 3)
    /// values : array_like, shape (n,)
    /// weights : array_like, optional
    ///     Declustering weights, for every normal score.
    /// holes : array_like, optional
    ///     Drill-hole ids or names, for `max_per_hole`.
    /// trend : array_like, optional
    ///     Trend at the data, from any model or estimator; `simulate` then
    ///     needs the trend at the targets.
    /// domains : array_like or label, optional
    ///     Domain label of each sample, or one label for all: strings,
    ///     numbers or booleans. Samples sharing a location in different
    ///     domains are all kept.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     If a Search.soft names a domain without samples, or has no
    ///     `domains` to work on.
    #[pyo3(signature = (coords, values, weights=None, holes=None, trend=None, domains=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
        trend: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (d, fitted) = data(
            coords,
            values,
            weights,
            holes,
            trend,
            domains,
            slf.classes,
            &slf.search,
        )?;
        slf.data = Some(d);
        slf.domains = fitted;
        Ok(slf)
    }

    /// Summary of `n` realizations at `targets`, seeds `seed, seed + 1, …`,
    /// with the probability and mean above each of `cutoffs` and the values at
    /// `quantiles`; the realizations themselves only when `realizations`.
    /// `anisotropy` (a LocalAnisotropy) orients each node's variogram and search.
    /// With `blocks` (a coarser BlockModel), each realization is averaged to
    /// its blocks, weighted by node volume, and summarized at block support;
    /// nodes outside every block are ignored and a block holding no node is
    /// an error. `trend`, needed when fitted with one, is the trend at the
    /// targets: an array, or the name of a column of BlockModel targets; each
    /// node is back-transformed within its trend class before any averaging.
    /// `domains`, needed when fitted with them, labels the targets, or is one
    /// label for all; a target in a domain without samples raises
    /// InvalidInput. Simulated domains, an ``(n, targets)`` array such as
    /// the `realizations` of SIS or Plurigaussian, give each realization its
    /// own: realization ``k`` of the grades is simulated within row ``k``.
    #[pyo3(signature = (targets, n=100, seed=0, cutoffs=vec![], quantiles=vec![], realizations=false, anisotropy=None, blocks=None, trend=None, domains=None))]
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
        trend: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
    ) -> PyResult<SimulationSummary> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let fitted = self.domains.as_deref();
        let nodes = realization_domains(fitted, domains, grid.len(), n, "simulate")?;
        let search = resolved(&self.search, fitted)?;
        let at_nodes = trend_at(d, targets, grid.len(), trend)?;
        let trend = d
            .trend
            .as_deref()
            .zip(at_nodes.as_deref())
            .map(|(data, nodes)| simulation::Trend {
                data,
                nodes,
                classes: self.classes,
            });
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
                    search: search.clone(),
                    seed: seed.wrapping_add(k as u64),
                };
                simulation::sgs_in(
                    &d.locs,
                    &d.values,
                    d.weights.as_deref(),
                    d.holes.as_deref(),
                    d.domains.as_deref().zip(of_realization(&nodes, k)),
                    trend,
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

/// The domains of the data and of `nodes`, when fitted with them.
fn zoned<'a>(d: &'a Data, nodes: &'a Option<Vec<u32>>) -> Option<simulation::Domains<'a>> {
    d.domains.as_deref().zip(nodes.as_deref())
}

/// A Search, or a list of them from files written before SGS took passes.
fn one_or_more<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Search>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Form {
        One(Box<Search>),
        More(Vec<Search>),
    }
    Ok(match Form::deserialize(d)? {
        Form::One(s) => vec![*s],
        Form::More(s) => s,
    })
}

/// Turning-bands simulation conditioned by kriging; same conventions as SGS,
/// trend and domains included.
///
/// With `domains` at `fit`, each domain is transformed on its own, as in
/// SGS, and every domain shares the bands of a realization. A node is
/// conditioned by kriging the residuals of the data of its domain, and of
/// other domains within Search.soft, each of those as its grade (and trend)
/// transformed through the node's domain; the node is back-transformed
/// through its domain's transform.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "TurningBands")]
pub struct TurningBands {
    variogram: CoreVariogram,
    bands: usize,
    step: Option<f64>,
    search: Option<Search>,
    #[serde(default = "classes")]
    classes: usize,
    /// Labels of the fitted domains, indexed by code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domains: Option<Vec<Label>>,
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
    /// the data, where conditioning changes nothing. `classes` are the trend
    /// classes, as in SGS.
    #[new]
    #[pyo3(signature = (variogram, bands=300, step=None, search=None, classes=10))]
    fn new(
        variogram: Variogram,
        bands: usize,
        step: Option<f64>,
        search: Option<Search>,
        classes: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            variogram: variogram.0,
            bands,
            step,
            search,
            classes,
            domains: None,
            data: None,
        })
    }

    /// Takes the conditioning data. Samples sharing a location keep the
    /// first, with a warning naming their `holes`.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (n, 2) or (n, 3)
    /// values : array_like, shape (n,)
    /// weights : array_like, optional
    ///     Declustering weights, for every normal score.
    /// holes : array_like, optional
    ///     Drill-hole ids or names, for `max_per_hole`.
    /// trend : array_like, optional
    ///     Trend at the data, from any model or estimator; `simulate` then
    ///     needs the trend at the targets.
    /// domains : array_like or label, optional
    ///     Domain label of each sample, or one label for all: strings,
    ///     numbers or booleans. Samples sharing a location in different
    ///     domains are all kept.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     If a Search.soft names a domain without samples, or has no
    ///     `domains` to work on.
    #[pyo3(signature = (coords, values, weights=None, holes=None, trend=None, domains=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
        trend: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (d, fitted) = data(
            coords,
            values,
            weights,
            holes,
            trend,
            domains,
            slf.classes,
            slf.search.as_slice(),
        )?;
        slf.data = Some(d);
        slf.domains = fitted;
        Ok(slf)
    }

    /// Summary of `n` realizations; same options as `SGS.simulate`.
    /// `domains`, needed when fitted with them, labels the targets, or is one
    /// label for all; a target in a domain without samples raises
    /// InvalidInput. Simulated domains, an ``(n, targets)`` array, give
    /// realization ``k`` of the grades the domains of row ``k``.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (targets, n=100, seed=0, cutoffs=vec![], quantiles=vec![], realizations=false, blocks=None, trend=None, domains=None))]
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
        trend: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
    ) -> PyResult<SimulationSummary> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let fitted = self.domains.as_deref();
        let nodes = realization_domains(fitted, domains, grid.len(), n, "simulate")?;
        let at_nodes = trend_at(d, targets, grid.len(), trend)?;
        let params = self.params(seed, self.resolved()?);
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
                d.holes.as_deref(),
                d.domains.as_deref(),
                d.trend.as_deref().map(|t| (t, self.classes)),
                lo,
                hi,
                &self.variogram,
                &params,
                n,
            )?;
            simulation::continuous(n, &options, |k| {
                let nodes = of_realization(&nodes, k);
                let r = ensemble.realization(k, &grid, nodes, at_nodes.as_deref());
                averaged(&support, r?)
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
    ///
    /// Parameters
    /// ----------
    /// domains : array_like or label, optional
    ///     Needed when fitted with them: labels of the blocks in file order,
    ///     or one label for all.
    /// trend : str, optional
    ///     Needed when fitted with a trend: the column of `path` holding it.
    /// discretization : tuple of int, optional
    ///     Nodes per axis simulated in each block and averaged by volume, as
    ///     ``simulate(model.discretize(discretization), blocks=model)``;
    ///     default the centroid. A node takes its block's domain and trend.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, out, n=100, seed=0, cutoffs=vec![], quantiles=vec![], rows=1_000_000, domains=None, trend=None, discretization=None))]
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
        domains: Option<&Bound<PyAny>>,
        trend: Option<String>,
        discretization: Option<(usize, usize, usize)>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let method = "simulate_to_parquet";
        match (&d.trend, &trend) {
            (Some(_), None) => return Err(invalid("fitted with a trend: give trend, its column")),
            (None, Some(_)) => return Err(invalid("trend needs trend at fit")),
            _ => {}
        }
        let discretization = discretization.map_or([1, 1, 1], |(x, y, z)| [x, y, z]);
        if discretization.contains(&0) {
            return Err(invalid("discretization must be positive"));
        }
        let blocks = match domains {
            Some(_) => ceres_io::BlockModelReader::open(&path)
                .map_err(|e| err(simulation::SimError::Io(e)))?
                .len(),
            None => 0,
        };
        let nodes = node_domains(self.domains.as_deref(), domains, blocks, method)?;
        let params = self.params(seed, self.resolved()?);
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
                    d.holes.as_deref(),
                    zoned(d, &nodes),
                    d.trend
                        .as_deref()
                        .zip(trend.as_deref())
                        .map(|(t, column)| (t, self.classes, column)),
                    &self.variogram,
                    &params,
                    n,
                    &options,
                    rows,
                    discretization,
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
    /// The parameters with `search`, or the default search.
    fn params(&self, seed: u64, search: Option<estimation::Search>) -> TurningBandsParams {
        TurningBandsParams {
            n_bands: self.bands,
            step: self.step,
            seed,
            search: search.unwrap_or(TurningBandsParams::default().search),
        }
    }

    /// The search with soft boundaries by domain code.
    fn resolved(&self) -> PyResult<Option<estimation::Search>> {
        let domains = self.domains.as_deref();
        self.search.as_ref().map(|s| s.resolve(domains)).transpose()
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
    #[serde(skip)]
    holes: Option<Vec<u32>>,
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
            holes: None,
        })
    }

    /// `holes` tag the samples for `max_per_hole`; samples sharing a location
    /// keep the first, with a warning naming their holes.
    #[pyo3(signature = (coords, categories, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        categories: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let cats = self::categories(categories)?;
        let (locs, keep, holes) = located(coords, cats.len(), "categories", holes, None)?;
        let cats = pick(&cats, &keep);
        if cats.iter().any(|&c| c >= slf.variograms.len()) {
            return Err(invalid("every category needs a variogram"));
        }
        slf.data = Some((locs, cats));
        slf.holes = holes;
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
                let holes = self.holes.as_deref();
                simulation::sis(locs, cats, holes, &grid, k, &self.variograms, &params)
                    .and_then(|r| majority(&support, r.categories, k))
            })
        })
        .map(CategoricalSummary)
        .map_err(err)
    }
}

/// Plurigaussian simulation: facies from thresholding independent latent
/// Gaussian fields through a truncation rule.
///
/// Parameters
/// ----------
/// variograms : Variogram or sequence of Variogram
///     Unit-sill variogram of each latent field, in field order.
/// proportions : sequence of float, optional
///     Facies proportions. Alone, facies ``0..k`` are ordered along the first
///     field, so each touches only its neighbours in that order.
/// rule : int or tuple, optional
///     Hierarchical rule, with `proportions`: a facies, or ``(field,
///     [child, ...])``, which cuts `field` into one slice per child, in
///     order, sized by the proportions of the facies below it. Facies under
///     different children touch only across that cut; for instance
///     ``(0, [4, (1, [0, 1, 2, 3])])`` sets facies 4 apart on the first field
///     and orders the other four on the second.
/// regions : sequence of (bounds, int), optional
///     Explicit rule instead of `proportions`: boxes ``([(low, high), ...],
///     facies)`` of the fields' Gaussian values, one ``(low, high]`` per
///     field, unbounded past the last.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "Plurigaussian")]
pub struct Plurigaussian {
    variograms: Vec<CoreVariogram>,
    rule: TruncationRule,
    hierarchy: Option<Hierarchy>,
    #[serde(skip)]
    data: Option<(Vec<Point>, Vec<usize>)>,
    /// Local proportions at the data.
    #[serde(skip)]
    local: Option<Vec<Vec<f64>>>,
}

impl Plurigaussian {
    fn facies(&self) -> usize {
        self.rule
            .regions
            .iter()
            .map(|r| r.facies + 1)
            .max()
            .unwrap_or(1)
    }

    /// Local proportions of `n` sites, one row each, for a rule built from
    /// proportions.
    fn local_rows(&self, obj: &Bound<PyAny>, n: usize) -> PyResult<Vec<Vec<f64>>> {
        if self.hierarchy.is_none() {
            return Err(invalid(
                "local proportions need a rule built from proportions",
            ));
        }
        let rows = rows(obj, "proportions")?;
        same_length(n, rows.len(), "proportions")?;
        let k = self.facies();
        for row in &rows {
            if row.len() != k {
                return Err(invalid(format!("proportions must have shape (n, {k})")));
            }
            proportion_row(row)?;
        }
        Ok(rows)
    }
}

fn proportion_row(p: &[f64]) -> PyResult<()> {
    if p.is_empty() || p.iter().any(|&q| !q.is_finite() || q < 0.0) {
        return Err(invalid("proportions must be finite and non-negative"));
    }
    if p.iter().sum::<f64>() <= 0.0 {
        return Err(invalid("proportions must not all be zero"));
    }
    Ok(())
}

fn hierarchy(obj: &Bound<PyAny>) -> PyResult<Hierarchy> {
    if let Ok(facies) = obj.extract::<usize>() {
        return Ok(Hierarchy::Facies(facies));
    }
    let (field, children): (usize, Vec<Bound<PyAny>>) = obj
        .extract()
        .map_err(|_| invalid("rule entries are facies or (field, [children])"))?;
    Ok(Hierarchy::Split {
        field,
        children: children.iter().map(hierarchy).collect::<PyResult<_>>()?,
    })
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
    #[pyo3(signature = (variograms, proportions=None, rule=None, regions=None))]
    fn new(
        variograms: &Bound<PyAny>,
        proportions: Option<Vec<f64>>,
        rule: Option<&Bound<PyAny>>,
        regions: Option<Vec<(Vec<(f64, f64)>, usize)>>,
    ) -> PyResult<Self> {
        let variograms: Vec<CoreVariogram> = match variograms.extract::<Variogram>() {
            Ok(v) => vec![v.0],
            Err(_) => variograms
                .extract::<Vec<Variogram>>()?
                .into_iter()
                .map(|v| v.0)
                .collect(),
        };
        if variograms.is_empty() {
            return Err(invalid("give at least one variogram"));
        }
        let fields = variograms.len();
        let mut tree = None;
        let rule = match (proportions, rule, regions) {
            (Some(p), rule, None) => {
                proportion_row(&p)?;
                let hierarchy = match rule {
                    Some(r) => hierarchy(r)?,
                    None => Hierarchy::Split {
                        field: 0,
                        children: (0..p.len()).map(Hierarchy::Facies).collect(),
                    },
                };
                hierarchy.validate(p.len(), fields).map_err(err)?;
                tree.insert(hierarchy).rule(&p)
            }
            (None, None, Some(r)) => TruncationRule {
                regions: r
                    .into_iter()
                    .map(|(bounds, facies)| Region { bounds, facies })
                    .collect(),
            },
            _ => {
                return Err(invalid(
                    "give proportions, with an optional rule, or regions",
                ));
            }
        };
        if rule.regions.is_empty() {
            return Err(invalid("the rule has no regions"));
        }
        if rule.fields() > fields {
            return Err(invalid(format!(
                "the rule thresholds {} fields but {fields} variograms were given",
                rule.fields()
            )));
        }
        Ok(Self {
            variograms,
            rule,
            hierarchy: tree,
            data: None,
            local: None,
        })
    }

    /// Takes the conditioning data.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (n, 2) or (n, 3)
    /// facies : array_like, shape (n,)
    ///     Facies ``0..k`` of each sample.
    /// holes : array_like, optional
    ///     Drill-hole ids or names; samples sharing a location keep the first.
    /// proportions : array_like, shape (n, k), optional
    ///     Local facies proportions at the samples, for a rule built from
    ///     proportions; `simulate` then needs them at the targets.
    #[pyo3(signature = (coords, facies, holes=None, proportions=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        facies: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        proportions: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let facies = categories(facies)?;
        let local = proportions
            .map(|p| slf.local_rows(p, facies.len()))
            .transpose()?;
        let (locs, keep, _) = located(coords, facies.len(), "facies", holes, None)?;
        slf.local = local.map(|l| pick(&l, &keep));
        slf.data = Some((locs, pick(&facies, &keep)));
        Ok(slf)
    }

    /// Summary of `n` realizations; same options as `SIS.simulate`, and
    /// `proportions` of shape ``(targets, k)``, the local facies proportions
    /// at the targets, when `fit` had them at the samples.
    #[pyo3(signature = (targets, n=100, seed=0, realizations=false, blocks=None, proportions=None))]
    #[allow(clippy::too_many_arguments)]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        realizations: bool,
        blocks: Option<PyRef<PyBlockModel>>,
        proportions: Option<&Bound<PyAny>>,
    ) -> PyResult<CategoricalSummary> {
        let (locs, facies) = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let local = match (&self.local, proportions) {
            (Some(at_data), Some(p)) => Some((at_data, self.local_rows(p, grid.len())?)),
            (None, None) => None,
            _ => {
                return Err(invalid(
                    "give local proportions at both fit and simulate, or at neither",
                ));
            }
        };
        let support = support(targets, &grid, blocks)?;
        let k = self.facies();
        py.detach(|| {
            simulation::categorical(n, k, realizations, |i| {
                let params = PgsParams {
                    seed: seed.wrapping_add(i as u64),
                    ..Default::default()
                };
                let (vgs, rule) = (&self.variograms, &self.rule);
                match (&local, &self.hierarchy) {
                    (Some((at_data, at_grid)), Some(tree)) => simulation::plurigaussian_local(
                        locs, facies, &grid, vgs, tree, at_data, at_grid, &params,
                    ),
                    _ => simulation::plurigaussian(locs, facies, &grid, vgs, rule, &params),
                }
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

/// Localised grades of selective blocks from their simulated realizations.
///
/// Each panel pools the ``n`` realizations of the blocks it holds, sorts the
/// pooled values, and gives its block ranked ``i`` the mean of the ``i``-th
/// chunk of ``n`` sorted values. The blocks average to the pooled mean and
/// reproduce the pooled grade-tonnage curve at tonnages ``k / blocks``,
/// without a change-of-support model. Partial panels localise over the blocks
/// present.
///
/// Parameters
/// ----------
/// smus : BlockModel
///     Selective blocks nesting in the panels: same rotation, sizes dividing
///     the panel sizes, grids aligned; not sub-blocked.
/// ranking : str
///     Column of `smus` ordering the blocks within a panel, such as a direct
///     kriging or the E-type mean of the realizations; ties follow row order.
/// realizations : array_like
///     ``(n, len(smus))`` realizations at selective-block support, as from
///     ``simulate(..., blocks=smus, realizations=True).realizations``.
/// panels : BlockModel
/// name : str, optional
///     Name of the new column; ``"localized"`` by default.
///
/// Returns
/// -------
/// BlockModel
///     `smus` with the localised grades; null outside every panel.
///
/// Raises
/// ------
/// InvalidInput
///     If the blocks do not nest, a block inside a panel has a null rank, or
///     the realizations are not finite.
#[pyfunction]
#[pyo3(signature = (smus, ranking, realizations, panels, name=None))]
fn localize(
    py: Python,
    smus: PyRef<PyBlockModel>,
    ranking: &str,
    realizations: &Bound<PyAny>,
    panels: PyRef<PyBlockModel>,
    name: Option<&str>,
) -> PyResult<PyBlockModel> {
    let rank = crate::transforms::nullable(&smus, ranking)?;
    let reals = rows(realizations, "realizations")?;
    let (panel_model, smu_model) = (&panels.0, &smus.0);
    let out = py
        .detach(|| simulation::localize(panel_model, smu_model, &rank, &reals))
        .map_err(err)?;
    let column: arrow_array::Float64Array = out.into_iter().collect();
    Ok(PyBlockModel(
        smus.0
            .with_column(name.unwrap_or("localized"), std::sync::Arc::new(column))
            .map_err(invalid)?,
    ))
}

fn data_columns(d: &Data) -> Columns {
    let mut columns = persist::point_columns(d.locs.iter().copied());
    columns.push(persist::column("value", d.values.iter().copied()));
    let weights = (0..d.values.len())
        .map(|i| d.weights.as_ref().map(|w| w[i]))
        .collect();
    columns.push(("weight".into(), weights));
    columns.push(hole_column(d.holes.as_deref(), d.values.len()));
    if let Some(trend) = &d.trend {
        columns.push(persist::column("trend", trend.iter().copied()));
    }
    if let Some(domains) = &d.domains {
        columns.push(persist::column(
            "domain",
            domains.iter().map(|&c| f64::from(c)),
        ));
    }
    columns
}

fn hole_column(holes: Option<&[u32]>, n: usize) -> (String, Vec<Option<f64>>) {
    let holes = (0..n).map(|i| holes.map(|h| f64::from(h[i]))).collect();
    ("hole".into(), holes)
}

/// Hole codes, or `None` when the column is missing (files written before
/// holes were kept) or null.
fn holes_from(found: &Found, n: usize) -> PyResult<Option<Vec<u32>>> {
    let Ok(holes) = found.optional("hole") else {
        return Ok(None);
    };
    same_length(n, holes.len(), "hole")?;
    let holes: Option<Vec<f64>> = holes.into_iter().collect();
    holes
        .map(|h| {
            h.into_iter()
                .map(|v| Ok(persist::index(v)? as u32))
                .collect()
        })
        .transpose()
}

/// The data of `found`, whose domain codes must index the `fitted` labels.
fn data_from(found: &Found, classes: usize, fitted: Option<&[Label]>) -> PyResult<Data> {
    let (locs, values, weights) = (
        found.points()?,
        found.values("value")?,
        found.optional("weight")?,
    );
    same_length(locs.len(), values.len(), "value")?;
    let weights: Option<Vec<f64>> = weights.into_iter().collect();
    let trend = found
        .optional("trend")
        .ok()
        .map(|_| found.values("trend"))
        .transpose()?;
    let domains = found
        .optional("domain")
        .ok()
        .map(|_| found.indices("domain"))
        .transpose()?;
    if let Some(d) = &domains {
        same_length(locs.len(), d.len(), "domain")?;
    }
    let known = fitted.map_or(0, <[Label]>::len);
    match &domains {
        Some(d) if d.iter().any(|&c| c >= known) => {
            return Err(invalid("domain codes need their labels"));
        }
        None if known > 0 => return Err(invalid("domain labels need a domain column")),
        _ => {}
    }
    let data = Data {
        holes: holes_from(found, locs.len())?,
        locs,
        values,
        weights,
        trend,
        domains: domains.map(|d| d.into_iter().map(|c| c as u32).collect()),
    };
    data.check(classes)?;
    Ok(data)
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
        self.data = Some(data_from(&columns, self.classes, self.domains.as_deref())?);
        Ok(())
    }
}

impl Tabular for TurningBands {
    fn columns(&self) -> Option<Columns> {
        self.data.as_ref().map(data_columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.data = Some(data_from(&columns, self.classes, self.domains.as_deref())?);
        Ok(())
    }
}

impl Tabular for Sis {
    fn columns(&self) -> Option<Columns> {
        self.data.as_ref().map(|d| {
            let mut columns = category_columns(d);
            columns.push(hole_column(self.holes.as_deref(), d.0.len()));
            columns
        })
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        let data = categories_from(&columns, self.variograms.len())?;
        self.holes = holes_from(&columns, data.0.len())?;
        self.data = Some(data);
        Ok(())
    }
}

impl Tabular for Plurigaussian {
    fn columns(&self) -> Option<Columns> {
        self.data.as_ref().map(|d| {
            let mut columns = category_columns(d);
            if let Some(local) = &self.local {
                for j in 0..self.facies() {
                    let name = format!("proportion_{j}");
                    columns.push(persist::column(&name, local.iter().map(|row| row[j])));
                }
            }
            columns
        })
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        let data = categories_from(&columns, self.facies())?;
        if columns.values("proportion_0").is_ok() {
            let k = self.facies();
            let cols = (0..k)
                .map(|j| columns.values(&format!("proportion_{j}")))
                .collect::<PyResult<Vec<_>>>()?;
            for c in &cols {
                same_length(data.0.len(), c.len(), "proportions")?;
            }
            self.local = Some(
                (0..data.0.len())
                    .map(|i| cols.iter().map(|c| c[i]).collect())
                    .collect(),
            );
        }
        self.data = Some(data);
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
    Sgs(CoreVariogram, Vec<estimation::Search>),
    Bands(CoreVariogram, Box<TurningBandsParams>),
}

struct Factors {
    transform: simulation::Decorrelation,
    locs: Vec<Point>,
    columns: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    holes: Option<Vec<u32>>,
    imputed: Option<(transforms::GaussianImputer, Vec<Vec<f64>>)>,
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
                    let search = s
                        .search
                        .iter()
                        .map(|x| x.clone().plain("MultivariateSimulation"));
                    Ok(Factor::Sgs(
                        s.variogram.clone(),
                        search.collect::<PyResult<_>>()?,
                    ))
                } else if let Ok(s) = s.cast::<TurningBands>() {
                    let s = s.borrow();
                    let search = s.search.clone().map(|x| x.plain("MultivariateSimulation"));
                    let params = s.params(0, search.transpose()?);
                    Ok(Factor::Bands(s.variogram.clone(), Box::new(params)))
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
    /// Samples missing a variable are dropped with a warning, unless
    /// `impute`; of samples sharing a location, the first is kept, with a
    /// warning naming their holes.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (n, 2) or (n, 3)
    /// data : array_like, shape (n, variables)
    ///     One column per simulator; NaN marks a missing variable.
    /// weights : array_like, optional
    ///     Declustering weights, for the transform when it takes them (PCA,
    ///     StepwiseConditional, PPMT) and for each factor's normal scores.
    /// holes : array_like, optional
    ///     Drill-hole ids or names, for `max_per_hole` and the warning on
    ///     samples sharing a location.
    /// impute : bool, default False
    ///     Keep samples missing some variables: a `GaussianImputer` fitted to
    ///     the data fills them afresh in every realization, so the imputation
    ///     uncertainty reaches the realizations. The transform is fitted to
    ///     the complete samples. Samples missing every variable are dropped.
    #[pyo3(signature = (coords, data, weights=None, holes=None, impute=false))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        data: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
        impute: bool,
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
        let holes = args::holes(holes, data.len())?;
        let complete = |data: &[Vec<f64>]| -> Vec<usize> {
            (0..data.len())
                .filter(|&i| data[i].iter().all(|v| v.is_finite()))
                .collect()
        };
        let usable: Vec<usize> = if impute {
            (0..data.len())
                .filter(|&i| data[i].iter().any(|v| v.is_finite()))
                .collect()
        } else {
            complete(&data)
        };
        let missing = data.len() - usable.len();
        if missing > 0 {
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            let what = if impute { "every" } else { "a" };
            let message = format!("{missing} samples miss {what} variable; dropped them");
            PyErr::warn(py, &category, &std::ffi::CString::new(message)?, 1)?;
        }
        let (data, locs) = (pick(&data, &usable), pick(&locs, &usable));
        let weights = weights.map(|w| pick(&w, &usable));
        let holes = holes.map(|(names, ids)| (pick(&names, &usable), pick(&ids, &usable)));
        let full = complete(&data);
        let transform = crate::transforms::decorrelation(
            slf.transform.bind(py),
            &pick(&data, &full),
            weights.as_ref().map(|w| pick(w, &full)).as_deref(),
            &pick(&locs, &full),
        )?;
        let keep = distinct(py, &locs, holes.as_ref().map(|h| &h.0[..]))?;
        let imputed = if impute {
            let imputer =
                transforms::GaussianImputer::fit(&data, weights.as_deref()).map_err(invalid)?;
            Some((imputer, pick(&data, &keep)))
        } else {
            None
        };
        let factors = match imputed {
            Some(_) => vec![],
            None => transform.forward(&pick(&data, &keep)),
        };
        slf.fitted = Some(Factors {
            transform,
            locs: pick(&locs, &keep),
            columns: (0..p)
                .map(|j| factors.iter().map(|r| r[j]).collect())
                .collect(),
            weights: weights.map(|w| pick(&w, &keep)),
            holes: holes.map(|h| pick(&h.1, &keep)),
            imputed,
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
                |k, j, factor| {
                    let drawn: Vec<f64>;
                    let values = match &f.imputed {
                        None => &f.columns[j],
                        Some((imputer, data)) => {
                            let p = self.factors.len();
                            let rows = imputer
                                .impute(data, simulation::factor_seed(seed, k, p))
                                .map_err(|e| simulation::SimError::Transform(e.to_string()))?;
                            drawn = f.transform.forward(&rows).iter().map(|r| r[j]).collect();
                            &drawn
                        }
                    };
                    let (seed, locs) = (factor, &f.locs);
                    let (weights, holes) = (f.weights.as_deref(), f.holes.as_deref());
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
                                holes,
                                &grid,
                                variogram,
                                &params,
                                local.as_ref(),
                            )?
                        }
                        Factor::Bands(variogram, params) => {
                            let params = TurningBandsParams {
                                seed,
                                ..params.as_ref().clone()
                            };
                            simulation::turning_bands(
                                locs, values, weights, holes, &grid, variogram, &params,
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
    m.add_function(wrap_pyfunction!(localize, m)?)?;
    m.add_class::<SimulationSummary>()?;
    m.add_class::<CategoricalSummary>()?;
    Ok(())
}
