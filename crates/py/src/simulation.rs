use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use simulation::{
    BlockSupport, CategoricalSummary as CoreCategorical, ContinuousOptions, ContinuousSummary,
    GibbsParams, Hierarchy, ObjectSet, Param, PgsParams, Region, SgsParams, Shape, SisParams,
    TruncationRule, TurningBandsParams,
};
use variogram::Variogram as CoreVariogram;

use crate::args::{
    self, Point, array1, distinct, finite, floats, optional_finite, pick, points, rows, same_length,
};
use crate::categorical::by_target;
use crate::containers::PyBlockModel;
use crate::estimation::{Label, Search, codes, fit_codes, labels, searches, targets};
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};
use crate::progress::with_progress;
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

/// `keep=`: False, True or 0-based realization indices.
fn keep_arg(keep: Option<&Bound<PyAny>>) -> PyResult<simulation::Keep> {
    let Some(keep) = keep else {
        return Ok(simulation::Keep::None);
    };
    if let Ok(flag) = keep.cast::<pyo3::types::PyBool>() {
        return Ok(match flag.is_true() {
            true => simulation::Keep::All,
            false => simulation::Keep::None,
        });
    }
    if keep.is_instance_of::<pyo3::types::PyString>() {
        return Err(invalid("keep must be a bool or realization indices"));
    }
    keep.extract::<Vec<usize>>()
        .map(simulation::Keep::Indices)
        .map_err(|_| invalid("keep must be a bool or non-negative realization indices"))
}

fn int_rows(rows: &[Vec<usize>]) -> Vec<Vec<i64>> {
    rows.iter()
        .map(|r| r.iter().map(|&c| c as i64).collect())
        .collect()
}

/// Uncertainty at every target from `n` realizations of a continuous
/// variable. Per-cutoff and per-quantile arrays have one row per target and
/// one column per cutoff or quantile.
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

    /// `(targets, cutoffs)` fraction of realizations above each cutoff.
    #[getter]
    fn probability_above<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.probability_above, self.0.mean.len())
    }

    /// `(targets, cutoffs)` mean of the values above each cutoff; NaN where
    /// no realization is above it.
    #[getter]
    fn mean_above<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.mean_above, self.0.mean.len())
    }

    #[getter]
    fn quantiles(&self) -> Vec<f64> {
        self.0.quantiles.clone()
    }

    /// `(targets, quantiles)` values at each requested quantile.
    #[getter]
    fn quantile_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.quantile_values, self.0.mean.len())
    }

    /// Mean of each realization over all targets, `(n,)`.
    #[getter]
    fn realization_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.realization_mean.clone())
    }

    /// `(n, cutoffs)` fraction of targets above each cutoff in each realization.
    #[getter]
    fn realization_above<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.realization_above, self.0.n)
    }

    /// Indices of the realizations kept with `keep=`.
    #[getter]
    fn kept(&self) -> Vec<usize> {
        self.0.kept.clone()
    }

    /// `(len(kept), targets)` kept realizations; None when none were kept.
    #[getter]
    fn realizations<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        (!self.0.kept.is_empty()).then(|| matrix(py, &self.0.realizations, self.0.mean.len()))
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

    /// `(targets, categories)` fraction of realizations in each category.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.probabilities, self.0.most_likely.len())
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

    /// Indices of the realizations kept with `keep=`.
    #[getter]
    fn kept(&self) -> Vec<usize> {
        self.0.kept.clone()
    }

    /// `(len(kept), targets)` kept realizations; None when none were kept.
    #[getter]
    fn realizations<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<i64>>> {
        (!self.0.kept.is_empty()).then(|| {
            matrix(
                py,
                &int_rows(&self.0.realizations),
                self.0.most_likely.len(),
            )
        })
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
    /// A secondary variable at the data, for collocated cosimulation.
    secondary: Option<Vec<f64>>,
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
    let at_nodes = finite(&args::column(Some(targets), trend, "trend")?, "trend")?;
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

/// `domains`, or the `domain_column` of `data`; not both.
fn domain_arg<'py>(
    data: &Bound<'py, PyAny>,
    domains: Option<&Bound<'py, PyAny>>,
    domain_column: Option<&str>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match (domains, domain_column) {
        (Some(_), Some(_)) => Err(invalid("give one of domains or domain_column")),
        (None, Some(c)) => args::named(Some(data), c, "domain_column").map(Some),
        (d, None) => Ok(d.cloned()),
    }
}

/// `arg`, or the column of `coords` it names.
fn resolve<'py>(
    coords: &Bound<'py, PyAny>,
    arg: Option<&Bound<'py, PyAny>>,
    what: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    arg.map(|a| args::column(Some(coords), a, what)).transpose()
}

/// The column `name` of the block model file `path`, read `rows` blocks at a
/// time.
fn file_column<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    name: &str,
    rows: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let io = |e| err(simulation::SimError::Io(e));
    let reader = ceres_io::BlockModelReader::open(path).map_err(io)?;
    if !reader.column_names().iter().any(|c| c == name) {
        return Err(crate::table::missing(name, reader.column_names().to_vec()));
    }
    let parts = reader
        .chunks(rows, Some(&[name]))
        .map_err(io)?
        .map(|chunk| Bound::new(py, PyBlockModel(chunk.map_err(io)?))?.get_item(name))
        .collect::<PyResult<Vec<_>>>()?;
    py.import("numpy")?.call_method1("concatenate", (parts,))
}

/// The data of `fit`, each argument an array or a column of `coords`, and the
/// labels of their domains, which every soft boundary of `search` must name.
#[allow(clippy::too_many_arguments)]
fn data(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    holes: Option<&Bound<PyAny>>,
    trend: Option<&Bound<PyAny>>,
    domains: Option<&Bound<PyAny>>,
    domain_column: Option<&str>,
    secondary: Option<&Bound<PyAny>>,
    classes: usize,
    search: &[Search],
) -> PyResult<(Data, Option<Vec<Label>>)> {
    let values = finite(&args::column(Some(coords), values, "values")?, "values")?;
    let (fitted, codes) = match domain_arg(coords, domains, domain_column)? {
        None => (None, None),
        Some(obj) => {
            let (fitted, codes) = fit_codes(&obj, values.len())?;
            (Some(fitted), Some(codes))
        }
    };
    for s in search {
        s.resolve(fitted.as_deref())?;
    }
    let weights = optional_finite(resolve(coords, weights, "weights")?.as_ref(), "weights")?;
    if let Some(w) = &weights {
        same_length(values.len(), w.len(), "weights")?;
    }
    let holes = resolve(coords, holes, "holes")?;
    let holes = holes.as_ref();
    let trend = optional_finite(resolve(coords, trend, "trend")?.as_ref(), "trend")?;
    if let Some(t) = &trend {
        same_length(values.len(), t.len(), "trend")?;
    }
    let secondary = resolve(coords, secondary, "secondary")?;
    let secondary = optional_finite(secondary.as_ref(), "secondary")?;
    if let Some(s) = &secondary {
        same_length(values.len(), s.len(), "secondary")?;
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
        secondary: secondary.map(|s| pick(&s, &keep)),
    };
    data.check(classes)?;
    Ok((data, fitted))
}

/// The searches with soft boundaries by the code of the `fitted` domains.
fn resolved(search: &[Search], fitted: Option<&[Label]>) -> PyResult<Vec<estimation::Search>> {
    search.iter().map(|s| s.resolve(fitted)).collect()
}

/// Domain codes of `n` nodes labeled by `obj` in the `fitted` domains, for
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

/// The lattice of BlockModel targets, regular or masked.
fn lattice_of(targets: &Bound<PyAny>) -> Option<simulation::Lattice> {
    let model = targets.cast::<PyBlockModel>().ok()?;
    simulation::Lattice::from_model(&model.get().0)
}

/// Bytes a batch of realizations may take: 70 % of the memory free now.
fn memory_budget(py: Python) -> PyResult<u64> {
    let free: u64 = py
        .import("ceres._memory")?
        .getattr("available")?
        .call0()?
        .extract()?;
    Ok(free / 10 * 7)
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
/// neighbors among the data and the nodes already simulated.
///
/// With `domains` at `fit`, each domain is transformed on its own, with the
/// declustering weights and within its own trend classes, and its nodes are
/// back-transformed through its transform; `variogram` is the one
/// normal-score variogram of every domain. Search.soft lets data and nodes
/// already simulated of another domain inform a node within the soft
/// distance, each as its grade (and trend) transformed through the node's
/// domain; boundaries are hard without it. `high_grade` compares grades, of
/// data and of simulated nodes, with its threshold; a clamped neighbor is
/// kriged as the score of the threshold. One random path visits
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
    /// Correlation of primary and secondary scores, when fitted with a
    /// secondary variable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    correlation: Option<f64>,
    #[serde(skip)]
    data: Option<Data>,
}

impl Sgs {
    /// The secondary variable at the targets, one row shared by every
    /// realization or one per realization, and its transform, when fitted
    /// with one.
    fn secondary_at(
        &self,
        d: &Data,
        targets: &Bound<PyAny>,
        nodes: usize,
        n: usize,
        secondary: Option<&Bound<PyAny>>,
    ) -> PyResult<Option<(simulation::Secondary, Vec<Vec<f64>>)>> {
        let (at_data, secondary) = match (&d.secondary, secondary) {
            (None, None) => return Ok(None),
            (Some(a), Some(s)) => (a, args::column(Some(targets), s, "secondary")?),
            (Some(_), None) => {
                return Err(invalid(
                    "fitted with a secondary; give secondary at the targets",
                ));
            }
            (None, Some(_)) => {
                return Err(invalid("secondary at the targets needs secondary at fit"));
            }
        };
        let array = secondary
            .py()
            .import("numpy")?
            .call_method1("asarray", (&secondary, "float64"))?;
        let rows = match array.getattr("ndim")?.extract::<usize>()? {
            2 => args::rows(&array, "secondary")?,
            _ => vec![args::floats(&array, "secondary")?],
        };
        if rows.len() != 1 && rows.len() != n {
            return Err(invalid(format!(
                "secondary: expected {n} realizations, got {}",
                rows.len()
            )));
        }
        for row in &rows {
            same_length(nodes, row.len(), "secondary")?;
            if row.iter().any(|v| !v.is_finite()) {
                return Err(invalid("secondary must be finite at every target"));
            }
        }
        let fitted =
            simulation::Secondary::fit(at_data, d.weights.as_deref(), &[], self.correlation)
                .map_err(err)?;
        Ok(Some((fitted, rows)))
    }
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
    #[pyo3(signature = (variogram, search, *, classes=10))]
    fn new(variogram: Variogram, search: &Bound<PyAny>, classes: usize) -> PyResult<Self> {
        Ok(Self {
            variogram: variogram.0,
            search: searches(search)?,
            classes,
            domains: None,
            correlation: None,
            data: None,
        })
    }

    /// Correlation of the primary and secondary normal scores used in
    /// collocated cosimulation; None when fitted without `secondary`.
    #[getter]
    fn correlation(&self) -> Option<f64> {
        self.correlation
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
    /// domain_column : str, optional
    ///     As in `simulate`.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     The pass, 1-based; NaN where no search finds enough data, and
    ///     `simulate` uses the last.
    #[pyo3(signature = (targets, *, anisotropy=None, domains=None, domain_column=None))]
    fn passes<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let domains = domain_arg(targets, domains, domain_column)?;
        let nodes = node_domains(
            self.domains.as_deref(),
            domains.as_ref(),
            grid.len(),
            "passes",
        )?;
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
    /// coords : array_like, shape (n, 2) or (n, 3), PointSet or BlockModel
    /// values : array_like, shape (n,), or str
    ///     Values, or the column of `coords` holding them; so for `weights`,
    ///     `holes` and `trend`.
    /// weights : array_like or str, optional
    ///     Declustering weights, for every normal score.
    /// holes : array_like or str, optional
    ///     Drill-hole ids or names, for `max_per_hole`.
    /// trend : array_like or str, optional
    ///     Trend at the data, from any model or estimator; `simulate` then
    ///     needs the trend at the targets.
    /// domains : array_like or label, optional
    ///     Domain label of each sample, or one label for all: strings,
    ///     numbers or booleans. Samples sharing a location in different
    ///     domains are all kept.
    /// domain_column : str, optional
    ///     The column of `coords` holding the domains, instead of `domains`.
    /// secondary : array_like or str, optional
    ///     A secondary variable at the data, for collocated cosimulation:
    ///     `simulate` then needs it at every target, and draws each node
    ///     from the collocated simple cokriging of its normal score from
    ///     its neighbors and the secondary score at the node, under the
    ///     Markov model (the cross-covariance is `correlation` times the
    ///     primary covariance). The secondary is normal-scored with the
    ///     declustering weights.
    /// correlation : float, optional
    ///     Correlation of the primary and secondary normal scores, in
    ///     [-1, 1]; by default the weighted correlation of their scores at
    ///     the data. 0 simulates as without `secondary`.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     If a Search.soft names a domain without samples, or has no
    ///     `domains` to work on, or both `domains` and `domain_column` are
    ///     given, or `correlation` comes without `secondary` or outside
    ///     [-1, 1].
    #[pyo3(signature = (coords, values, *, weights=None, holes=None, trend=None, domains=None, domain_column=None, secondary=None, correlation=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
        trend: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
        secondary: Option<&Bound<PyAny>>,
        correlation: Option<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if correlation.is_some() && secondary.is_none() {
            return Err(invalid("correlation needs secondary"));
        }
        let (d, fitted) = data(
            coords,
            values,
            weights,
            holes,
            trend,
            domains,
            domain_column,
            secondary,
            slf.classes,
            &slf.search,
        )?;
        slf.correlation = match &d.secondary {
            None => None,
            Some(s) => {
                let scores = simulation::Transforms::fit(
                    &d.values,
                    d.weights.as_deref(),
                    d.domains.as_deref(),
                    d.trend.as_deref().map(|t| (t, slf.classes)),
                )
                .map_err(err)?
                .scores;
                let w = d.weights.as_deref();
                let fitted = simulation::Secondary::fit(s, w, &scores, correlation).map_err(err)?;
                Some(fitted.correlation)
            }
        };
        slf.data = Some(d);
        slf.domains = fitted;
        Ok(slf)
    }

    /// Summary of `n` realizations at `targets`, each seeded from `seed` and
    /// its index,
    /// with the probability and mean above each of `cutoffs` and the values at
    /// `quantiles`; the ``(n, targets)`` realizations only when `keep`.
    /// `anisotropy` (a LocalAnisotropy) orients each node's variogram and search.
    /// With `blocks` (a coarser BlockModel), each realization is averaged to
    /// its blocks, weighted by node volume, and summarized at block support;
    /// nodes outside every block are ignored and a block holding no node is
    /// an error. `trend`, needed when fitted with one, is the trend at the
    /// targets: an array, or the name of a column of PointSet or BlockModel targets; each
    /// node is back-transformed within its trend class before any averaging.
    /// `domains`, needed when fitted with them, labels the targets, or is one
    /// label for all; a target in a domain without samples raises
    /// InvalidInput. Simulated domains, an ``(n, targets)`` array such as
    /// the `realizations` of SIS or Plurigaussian, give each realization its
    /// own: realization ``k`` of the grades is simulated within row ``k``.
    /// `domain_column`, instead of `domains`, names the column of PointSet or
    /// BlockModel targets holding them. `secondary`, needed when fitted with
    /// one, is the secondary variable at the targets: an array, the name of
    /// a column of PointSet or BlockModel targets, or an ``(n, targets)``
    /// array such as the `realizations` of a simulation of the secondary,
    /// realization ``k`` then cosimulated with row ``k``.
    /// `path` orders the nodes. On a regular or masked BlockModel every
    /// realization follows, by default, one multigrid path (coarse cells
    /// first), so each node's neighbors and kriging weights are found once
    /// for a batch of realizations; a node on a datum takes its grade.
    /// "random" draws a new random path per realization, which points,
    /// sub-blocked models, local anisotropy, simulated domains, octant
    /// searches, high-grade restrictions and soft boundaries need and the
    /// default then uses; "shared" raises InvalidInput where it cannot run.
    /// `batch`, on a shared path, is the number of realizations simulated
    /// together, by default as many as fit in 70 % of the free memory; a run
    /// whose quantiles and kept realizations alone exceed that memory
    /// raises InvalidInput. Realizations do not depend on `batch`. `progress`
    /// shows a `tqdm` bar over the realizations.
    #[pyo3(signature = (targets, *, n=100, seed=0, cutoffs=vec![], quantiles=vec![], keep=None, anisotropy=None, blocks=None, trend=None, domains=None, domain_column=None, secondary=None, path=None, batch=None, progress=true))]
    #[allow(clippy::too_many_arguments)]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        keep: Option<&Bound<PyAny>>,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        blocks: Option<PyRef<PyBlockModel>>,
        trend: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
        secondary: Option<&Bound<PyAny>>,
        path: Option<&str>,
        batch: Option<usize>,
        progress: bool,
    ) -> PyResult<SimulationSummary> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        if !matches!(path, None | Some("shared" | "random")) {
            return Err(invalid(format!(
                "path must be 'shared' or 'random', got {:?}",
                path.unwrap_or_default()
            )));
        }
        let lattice = lattice_of(targets);
        let mut grid = match &lattice {
            Some(_) => None,
            None => Some(self::targets(targets)?),
        };
        let count = lattice
            .as_ref()
            .map_or_else(|| grid.as_ref().map_or(0, Vec::len), |l| l.len());
        let fitted = self.domains.as_deref();
        let domains = domain_arg(targets, domains, domain_column)?;
        let nodes = realization_domains(fitted, domains.as_ref(), count, n, "simulate")?;
        let secondary = self.secondary_at(d, targets, count, n, secondary)?;
        let search = resolved(&self.search, fitted)?;
        let at_nodes = trend_at(d, targets, count, trend)?;
        let unsupported = match (&lattice, anisotropy.is_some()) {
            (None, _) => Some("targets that are not a regular or masked BlockModel"),
            (_, true) => Some("local anisotropy"),
            _ => simulation::shared_unsupported(
                &search,
                false,
                nodes.as_ref().is_some_and(|rows| rows.len() > 1),
            ),
        };
        let lattice = match (path, unsupported) {
            (Some("random"), _) | (None, Some(_)) => None,
            (_, None) => lattice,
            (_, Some(why)) => {
                return Err(invalid(format!(
                    "path='shared' does not support {why}; use path='random'"
                )));
            }
        };
        if grid.is_none() && (lattice.is_none() || blocks.is_some()) {
            grid = Some(self::targets(targets)?);
        }
        let grid = grid.unwrap_or_default();
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
            keep: keep_arg(keep)?,
        };
        if let Some(lattice) = lattice {
            let kept = options.keep.kept(n).len();
            let budget = memory_budget(py)?;
            let size = simulation::shared_batch(
                lattice.len(),
                n,
                options.cutoffs.len(),
                !options.quantiles.is_empty(),
                kept,
                budget,
            )
            .map_err(err)?;
            let batch = batch.map_or(size, |b| b.max(1));
            let rows: Option<Vec<Vec<f64>>> = secondary.as_ref().map(|(fitted, rows)| {
                rows.iter()
                    .map(|r| r.iter().map(|&v| fitted.forward(v)).collect())
                    .collect()
            });
            let correlation = secondary
                .as_ref()
                .map_or(0.0, |(fitted, _)| fitted.correlation);
            let shared = simulation::SharedSgs {
                lattice: &lattice,
                search: &search,
                levels: None,
                seed,
            };
            return with_progress(py, Some(n as u64), progress, |counter| {
                let collocated = rows.as_deref().map(|scores| simulation::Collocated {
                    scores,
                    correlation,
                });
                simulation::continuous_in_batches(
                    n,
                    &options,
                    batch,
                    |range| {
                        simulation::sgs_shared(
                            &d.locs,
                            &d.values,
                            d.weights.as_deref(),
                            d.holes.as_deref(),
                            d.domains.as_deref().zip(of_realization(&nodes, 0)),
                            trend,
                            &self.variogram,
                            &shared,
                            range,
                            collocated.as_ref(),
                        )
                    },
                    |b, i| averaged(&support, b.realization(i)),
                    counter,
                )
            })?
            .map(SimulationSummary)
            .map_err(err);
        }
        with_progress(py, Some(n as u64), progress, |counter| {
            simulation::continuous(
                n,
                &options,
                |k| {
                    let params = SgsParams {
                        search: search.clone(),
                        seed: ceres_core::rng::realization_seed(seed, k as u64),
                    };
                    let domains = d.domains.as_deref().zip(of_realization(&nodes, k));
                    match &secondary {
                        None => simulation::sgs_in(
                            &d.locs,
                            &d.values,
                            d.weights.as_deref(),
                            d.holes.as_deref(),
                            domains,
                            trend,
                            &grid,
                            &self.variogram,
                            &params,
                            local.as_ref(),
                        ),
                        Some((fitted, rows)) => simulation::cosgs(
                            &d.locs,
                            &d.values,
                            d.weights.as_deref(),
                            d.holes.as_deref(),
                            domains,
                            trend,
                            &grid,
                            &self.variogram,
                            &params,
                            local.as_ref(),
                            fitted,
                            &rows[k % rows.len()],
                        ),
                    }
                    .and_then(|r| averaged(&support, r.values))
                },
                counter,
            )
        })?
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

    /// `bands` lines, each discretized every `step` meters along the major axis
    /// (default: a fiftieth of the shortest range). `search` is the
    /// neighborhood of the conditioning kriging (default: the 32 nearest
    /// data at any distance); a radius near the range skips nodes far from
    /// the data, where conditioning changes nothing. `classes` are the trend
    /// classes, as in SGS.
    #[new]
    #[pyo3(signature = (variogram, *, bands=300, step=None, search=None, classes=10))]
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
    /// coords : array_like, shape (n, 2) or (n, 3), PointSet or BlockModel
    /// values : array_like, shape (n,), or str
    ///     Values, or the column of `coords` holding them; so for `weights`,
    ///     `holes` and `trend`.
    /// weights : array_like or str, optional
    ///     Declustering weights, for every normal score.
    /// holes : array_like or str, optional
    ///     Drill-hole ids or names, for `max_per_hole`.
    /// trend : array_like or str, optional
    ///     Trend at the data, from any model or estimator; `simulate` then
    ///     needs the trend at the targets.
    /// domains : array_like or label, optional
    ///     Domain label of each sample, or one label for all: strings,
    ///     numbers or booleans. Samples sharing a location in different
    ///     domains are all kept.
    /// domain_column : str, optional
    ///     The column of `coords` holding the domains, instead of `domains`.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     If a Search.soft names a domain without samples, or has no
    ///     `domains` to work on, or both `domains` and `domain_column` are
    ///     given.
    #[pyo3(signature = (coords, values, *, weights=None, holes=None, trend=None, domains=None, domain_column=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
        trend: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (d, fitted) = data(
            coords,
            values,
            weights,
            holes,
            trend,
            domains,
            domain_column,
            None,
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
    /// realization ``k`` of the grades the domains of row ``k``;
    /// `domain_column` and `progress` as in `SGS.simulate`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (targets, *, n=100, seed=0, cutoffs=vec![], quantiles=vec![], keep=None, blocks=None, trend=None, domains=None, domain_column=None, progress=true))]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        keep: Option<&Bound<PyAny>>,
        blocks: Option<PyRef<PyBlockModel>>,
        trend: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
        progress: bool,
    ) -> PyResult<SimulationSummary> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let fitted = self.domains.as_deref();
        let domains = domain_arg(targets, domains, domain_column)?;
        let nodes = realization_domains(fitted, domains.as_ref(), grid.len(), n, "simulate")?;
        let at_nodes = trend_at(d, targets, grid.len(), trend)?;
        let params = self.params(seed, self.resolved()?);
        let support = support(targets, &grid, blocks)?;
        let options = ContinuousOptions {
            cutoffs,
            quantiles,
            keep: keep_arg(keep)?,
        };
        let (lo, hi) = simulation::bounds(&grid);
        with_progress(py, Some(n as u64), progress, |counter| {
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
            simulation::continuous_batched(
                n,
                &options,
                ensemble.batch(grid.len()),
                |ks| {
                    ensemble
                        .realizations(
                            ks,
                            &grid,
                            |k| of_realization(&nodes, k),
                            at_nodes.as_deref(),
                        )?
                        .into_iter()
                        .map(|r| averaged(&support, r))
                        .collect()
                },
                counter,
            )
        })?
        .map(SimulationSummary)
        .map_err(err)
    }

    /// Summary of `n` realizations over the block model file `path`, written
    /// to `out` chunk by chunk with the input columns: `mean`, `variance`,
    /// `p_above_<c>` and `mean_above_<c>` per cutoff, `q<p>` per quantile, and
    /// `realization_<k>` for each realization kept with `keep`.
    /// The same values as `simulate` on the whole model, in memory bounded by
    /// `rows` blocks plus the bands. Returns each realization's global
    /// `realization_mean` and ``(n, cutoffs)`` `realization_above`.
    ///
    /// Parameters
    /// ----------
    /// keep : bool or sequence of int, default False
    ///     Realizations written beside the summary as ``realization_<k>``:
    ///     none, all, or these 0-based indices.
    /// domains : array_like or label, optional
    ///     Needed when fitted with them: labels of the blocks in file order,
    ///     or one label for all.
    /// domain_column : str, optional
    ///     Instead of `domains`, the column of `path` holding them.
    /// trend : str, optional
    ///     Needed when fitted with a trend: the column of `path` holding it,
    ///     with no nulls. A kernel trend's `predict` is NaN beyond four
    ///     bandwidths of its data; fill those blocks first, e.g. with the
    ///     mean of the data.
    /// discretization : tuple of int, optional
    ///     Nodes per axis simulated in each block and averaged by volume, as
    ///     ``simulate(model.discretize(discretization), blocks=model)``;
    ///     default the centroid. A node takes its block's domain and trend.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, out, *, n=100, seed=0, cutoffs=vec![], quantiles=vec![], keep=None, rows=1_000_000, domains=None, domain_column=None, trend=None, discretization=None, progress=true))]
    fn simulate_to_parquet<'py>(
        &self,
        py: Python<'py>,
        path: std::path::PathBuf,
        out: std::path::PathBuf,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        keep: Option<&Bound<PyAny>>,
        rows: usize,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
        trend: Option<String>,
        discretization: Option<(usize, usize, usize)>,
        progress: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let domains = match (domains, domain_column) {
            (Some(_), Some(_)) => return Err(invalid("give one of domains or domain_column")),
            (None, Some(c)) => Some(file_column(py, &path, c, rows)?),
            (d, None) => d.cloned(),
        };
        let domains = domains.as_ref();
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
        let total = ceres_io::BlockModelReader::open(&path)
            .map_err(|e| err(simulation::SimError::Io(e)))?
            .len();
        let blocks = if domains.is_some() { total } else { 0 };
        let nodes = node_domains(self.domains.as_deref(), domains, blocks, method)?;
        let params = self.params(seed, self.resolved()?);
        let options = ContinuousOptions {
            cutoffs,
            quantiles,
            keep: keep_arg(keep)?,
        };
        let global = with_progress(py, Some(total as u64), progress, |counter| {
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
                counter,
            )
        })?
        .map_err(err)?;
        let result = pyo3::types::PyDict::new(py);
        result.set_item("realization_mean", array1(py, global.realization_mean))?;
        result.set_item(
            "realization_above",
            by_target(py, &global.realization_above, n),
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

/// Categories `obj`, or the column of `coords` it names.
pub(crate) fn categories(coords: &Bound<PyAny>, obj: &Bound<PyAny>) -> PyResult<Vec<usize>> {
    let obj = args::column(Some(coords), obj, "categories")?;
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
    /// Local proportions at the data.
    #[serde(skip)]
    local: Option<Vec<Vec<f64>>>,
}

/// `n` rows of `k` local proportions.
fn proportion_rows(obj: &Bound<PyAny>, n: usize, k: usize) -> PyResult<Vec<Vec<f64>>> {
    let rows = rows(obj, "proportions")?;
    same_length(n, rows.len(), "proportions")?;
    for row in &rows {
        if row.len() != k {
            return Err(invalid(format!("proportions must have shape (n, {k})")));
        }
        proportion_row(row)?;
    }
    Ok(rows)
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
            local: None,
        })
    }

    /// Takes the categories ``0..k`` at `coords`. `holes` tag the samples for
    /// `max_per_hole`; samples sharing a location keep the first, with a
    /// warning naming their holes. Both may name columns of `coords`.
    /// `proportions`, shape ``(n, k)`` (an array or a Table such as
    /// `combine_proportions` returns), are local category proportions at the
    /// samples; `simulate` then needs them at the targets, and krigs each
    /// indicator minus its local proportion by simple kriging, so every node
    /// is drawn towards its own proportions.
    #[pyo3(signature = (coords, categories, *, holes=None, proportions=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        categories: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        proportions: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let cats = self::categories(coords, categories)?;
        let k = slf.variograms.len();
        let local = proportions
            .map(|p| proportion_rows(p, cats.len(), k))
            .transpose()?;
        let holes = resolve(coords, holes, "holes")?;
        let (locs, keep, holes) = located(coords, cats.len(), "categories", holes.as_ref(), None)?;
        let cats = pick(&cats, &keep);
        if cats.iter().any(|&c| c >= k) {
            return Err(invalid("every category needs a variogram"));
        }
        slf.data = Some((locs, cats));
        slf.holes = holes;
        slf.local = local.map(|l| pick(&l, &keep));
        Ok(slf)
    }

    /// Summary of `n` realizations, each seeded from `seed` and its index; the ``(n, targets)``
    /// realizations themselves only when `keep`. With `blocks` (a
    /// coarser BlockModel), each block takes the category filling most of its
    /// node volume, ties to the smallest, as in `BlockModel.regularize`; blocks as in
    /// `SGS.simulate`. `proportions`, shape ``(targets, k)``, are the local
    /// category proportions at the targets, when `fit` had them at the samples.
    #[pyo3(signature = (targets, *, n=100, seed=0, keep=None, blocks=None, proportions=None, progress=true))]
    #[allow(clippy::too_many_arguments)]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        keep: Option<&Bound<PyAny>>,
        blocks: Option<PyRef<PyBlockModel>>,
        proportions: Option<&Bound<PyAny>>,
        progress: bool,
    ) -> PyResult<CategoricalSummary> {
        let (locs, cats) = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let k = self.variograms.len();
        let at_grid = match (&self.local, proportions) {
            (Some(_), Some(p)) => Some(proportion_rows(p, grid.len(), k)?),
            (None, None) => None,
            _ => {
                return Err(invalid(
                    "give local proportions at both fit and simulate, or at neither",
                ));
            }
        };
        let local = self.local.as_deref().zip(at_grid.as_deref());
        let support = support(targets, &grid, blocks)?;
        let keep = keep_arg(keep)?;
        with_progress(py, Some(n as u64), progress, |counter| {
            simulation::categorical(
                n,
                k,
                &keep,
                |i| {
                    let params = SisParams {
                        search: self.search.clone(),
                        seed: ceres_core::rng::realization_seed(seed, i as u64),
                    };
                    let holes = self.holes.as_deref();
                    simulation::sis(
                        locs,
                        cats,
                        holes,
                        &grid,
                        k,
                        &self.variograms,
                        &params,
                        local,
                    )
                    .and_then(|r| majority(&support, r.categories, k))
                },
                counter,
            )
        })?
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
///     field, so each touches only its neighbors in that order.
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
        proportion_rows(obj, n, self.facies())
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
    #[pyo3(signature = (variograms, *, proportions=None, rule=None, regions=None))]
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

    /// The latent fields' variograms.
    #[getter]
    fn variograms(&self) -> Vec<Variogram> {
        self.variograms.iter().cloned().map(Variogram).collect()
    }

    /// Rescales the ranges of each latent variogram, keeping its structures
    /// and anisotropy, so the indicator semivariograms the rule implies match
    /// the experimental ones.
    ///
    /// Parameters
    /// ----------
    /// experimental : sequence of ExperimentalVariogram or None
    ///     Omnidirectional semivariogram of each facies' indicator, in facies
    ///     order, not standardized; None skips a facies. Lags are weighted by
    ///     their pair counts.
    ///
    /// Returns
    /// -------
    /// Plurigaussian
    ///     This simulator, with the fitted variograms.
    fn fit_variograms<'py>(
        mut slf: PyRefMut<'py, Self>,
        experimental: Vec<Option<PyRef<crate::variogram::ExperimentalVariogram>>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if experimental.len() != slf.facies() {
            return Err(invalid(format!(
                "give one experimental variogram per facies ({})",
                slf.facies()
            )));
        }
        let refs: Vec<Option<&variogram::Experimental>> = experimental
            .iter()
            .map(|e| e.as_ref().map(|e| &e.0))
            .collect();
        slf.variograms = simulation::fit_latent(&slf.rule, &slf.variograms, &refs).map_err(err)?;
        Ok(slf)
    }

    /// Indicator semivariogram of each facies the rule and the latent
    /// variograms imply, along the variograms' major axes.
    ///
    /// Parameters
    /// ----------
    /// lags : array_like, shape (m,)
    ///
    /// Returns
    /// -------
    /// ndarray, shape (k, m)
    fn indicator_variograms<'py>(
        &self,
        py: Python<'py>,
        lags: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let lags = finite(lags, "lags")?;
        let k = self.facies();
        let columns: Vec<Vec<f64>> = lags
            .iter()
            .map(|&h| {
                let rho: Vec<f64> = self
                    .variograms
                    .iter()
                    .map(|v| v.cov(h.abs()) / v.total_sill())
                    .collect();
                self.rule.indicator_gammas(&rho, k)
            })
            .collect();
        let rows: Vec<Vec<f64>> = (0..k)
            .map(|f| columns.iter().map(|c| c[f]).collect())
            .collect();
        Ok(matrix(py, &rows, lags.len()))
    }

    /// Takes the conditioning data.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, shape (n, 2) or (n, 3), PointSet or BlockModel
    /// categories : array_like, shape (n,), or str
    ///     Facies ``0..k`` of each sample, or the column of `coords` holding
    ///     them.
    /// holes : array_like or str, optional
    ///     Drill-hole ids or names; samples sharing a location keep the first.
    /// proportions : array_like, shape (n, k), optional
    ///     Local facies proportions at the samples, for a rule built from
    ///     proportions; `simulate` then needs them at the targets.
    #[pyo3(signature = (coords, categories, *, holes=None, proportions=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        categories: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        proportions: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let facies = self::categories(coords, categories)?;
        let local = proportions
            .map(|p| slf.local_rows(p, facies.len()))
            .transpose()?;
        let holes = resolve(coords, holes, "holes")?;
        let (locs, keep, _) = located(coords, facies.len(), "categories", holes.as_ref(), None)?;
        slf.local = local.map(|l| pick(&l, &keep));
        slf.data = Some((locs, pick(&facies, &keep)));
        Ok(slf)
    }

    /// Summary of `n` realizations; same options as `SIS.simulate`, and
    /// `proportions` of shape ``(targets, k)``, the local facies proportions
    /// at the targets, when `fit` had them at the samples.
    #[pyo3(signature = (targets, *, n=100, seed=0, keep=None, blocks=None, proportions=None, progress=true))]
    #[allow(clippy::too_many_arguments)]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        keep: Option<&Bound<PyAny>>,
        blocks: Option<PyRef<PyBlockModel>>,
        proportions: Option<&Bound<PyAny>>,
        progress: bool,
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
        let keep = keep_arg(keep)?;
        with_progress(py, Some(n as u64), progress, |counter| {
            simulation::categorical(
                n,
                k,
                &keep,
                |i| {
                    let params = PgsParams {
                        seed: ceres_core::rng::realization_seed(seed, i as u64),
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
                },
                counter,
            )
        })?
        .map(CategoricalSummary)
        .map_err(err)
    }
}

/// One Gaussian draw at `coords` honoring `bounds` (`(n, 2)` lower/upper).
#[pyfunction]
#[pyo3(signature = (coords, bounds, variogram, *, iterations=200, burn_in=50, seed=0))]
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

/// Localized grades of selective blocks from their simulated realizations.
///
/// Each panel pools the ``n`` realizations of the blocks it holds, sorts the
/// pooled values, and gives its block ranked ``i`` the mean of the ``i``-th
/// chunk of ``n`` sorted values. The blocks average to the pooled mean and
/// reproduce the pooled grade-tonnage curve at tonnages ``k / blocks``,
/// without a change-of-support model. Partial panels localize over the blocks
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
/// panels : BlockModel
/// realizations : array_like
///     ``(n, len(smus))`` realizations at selective-block support, as from
///     ``simulate(..., blocks=smus, keep=True).realizations``.
/// name : str, default "localized"
///     Name of the new column.
///
/// Returns
/// -------
/// BlockModel
///     `smus` with the localized grades; null outside every panel.
///
/// Raises
/// ------
/// InvalidInput
///     If the blocks do not nest, a block inside a panel has a null rank, or
///     the realizations are not finite.
#[pyfunction]
#[pyo3(signature = (smus, ranking, panels, realizations, *, name="localized"))]
fn localize(
    py: Python,
    smus: PyRef<PyBlockModel>,
    ranking: &str,
    panels: PyRef<PyBlockModel>,
    realizations: &Bound<PyAny>,
    name: &str,
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
            .with_column(name, std::sync::Arc::new(column))
            .map_err(invalid)?,
    ))
}

/// A size or angle of an object set: a number, or a ``(min, max)`` range.
fn object_param(
    set: &Bound<pyo3::types::PyDict>,
    key: &str,
    default: Option<f64>,
) -> PyResult<Param> {
    let Some(value) = set.get_item(key)? else {
        return default
            .map(Param::Fixed)
            .ok_or_else(|| invalid(format!("an object set needs {key:?}")));
    };
    if let Ok(v) = value.extract::<f64>() {
        return Ok(Param::Fixed(v));
    }
    value
        .extract::<(f64, f64)>()
        .map(|(lo, hi)| Param::Uniform(lo, hi))
        .map_err(|_| invalid(format!("{key} must be a number or a (min, max) pair")))
}

/// An object set from its dict; `flat` for a 2D grid, where thickness and
/// the minor radius are optional.
fn object_set(set: &Bound<pyo3::types::PyDict>, flat: bool) -> PyResult<ObjectSet> {
    let text = |key: &str| -> PyResult<String> {
        set.get_item(key)?
            .ok_or_else(|| invalid(format!("an object set needs {key:?}")))?
            .extract()
            .map_err(|_| invalid(format!("{key} must be a string")))
    };
    let shape = text("shape")?;
    let keys: &[&str] = match shape.as_str() {
        "channel" => &["width", "thickness", "azimuth", "wavelength", "amplitude"],
        "ellipsoid" => &["radii", "azimuth", "dip", "rake"],
        _ => {
            return Err(invalid(format!(
                "shape must be 'channel' or 'ellipsoid', got {shape:?}"
            )));
        }
    };
    for key in set.keys() {
        let key: String = key.extract()?;
        if !["shape", "code", "proportion"].contains(&key.as_str()) && !keys.contains(&key.as_str())
        {
            return Err(invalid(format!(
                "a {shape} set takes code, proportion, {}; not {key:?}",
                keys.join(", ")
            )));
        }
    }
    let code = set
        .get_item("code")?
        .ok_or_else(|| invalid("an object set needs \"code\""))?
        .extract::<u8>()
        .map_err(|_| invalid("code must be an integer 0 to 254"))?;
    let proportion = set
        .get_item("proportion")?
        .ok_or_else(|| invalid("an object set needs \"proportion\""))?
        .extract::<f64>()
        .map_err(|_| invalid("proportion must be a number"))?;
    let unused = flat.then_some(1.0);
    let shape = if shape == "channel" {
        let amplitude = object_param(set, "amplitude", Some(0.0))?;
        let straight = set.get_item("amplitude")?.is_none();
        Shape::Channel {
            width: object_param(set, "width", None)?,
            thickness: object_param(set, "thickness", unused)?,
            azimuth: object_param(set, "azimuth", Some(0.0))?,
            wavelength: object_param(set, "wavelength", straight.then_some(1.0))?,
            amplitude,
        }
    } else {
        let radii = set
            .get_item("radii")?
            .ok_or_else(|| invalid("an ellipsoid set needs \"radii\""))?;
        let radii: Vec<Bound<PyAny>> = radii
            .try_iter()
            .and_then(|r| r.collect())
            .map_err(|_| invalid("radii must be (major, semi-major, minor)"))?;
        let radius = |r: &Bound<PyAny>| -> PyResult<Param> {
            if let Ok(v) = r.extract::<f64>() {
                return Ok(Param::Fixed(v));
            }
            r.extract::<(f64, f64)>()
                .map(|(lo, hi)| Param::Uniform(lo, hi))
                .map_err(|_| invalid("each radius must be a number or a (min, max) pair"))
        };
        let radii = match (&radii[..], flat) {
            ([a, b, c], _) => [radius(a)?, radius(b)?, radius(c)?],
            ([a, b], true) => [radius(a)?, radius(b)?, Param::Fixed(1.0)],
            _ => {
                return Err(invalid(match flat {
                    true => "radii must be (major, semi-major) or (major, semi-major, minor)",
                    false => "radii must be (major, semi-major, minor) on a 3D grid",
                }));
            }
        };
        Shape::Ellipsoid {
            radii,
            azimuth: object_param(set, "azimuth", Some(0.0))?,
            dip: object_param(set, "dip", Some(0.0))?,
            rake: object_param(set, "rake", Some(0.0))?,
        }
    };
    Ok(ObjectSet {
        code,
        proportion,
        shape,
    })
}

/// Categorical training image drawn from channels and ellipsoids.
///
/// The image starts as `background`. Each object set, in order, adds
/// objects of its shape with its code at random places until the code covers
/// its ``proportion`` of the cells. A later set overwrites the cells of the
/// earlier ones, so their final shares can end below their proportions; the
/// last object of a set can overshoot. Objects cut by the edges of the grid
/// are as frequent as whole ones.
///
/// Every size and angle of a set is a number, or a ``(min, max)`` range drawn
/// uniformly for each object. Lengths are in the units of the cell sizes and
/// angles in degrees, in world coordinates: azimuth clockwise from north, dip
/// positive down.
///
/// - ``{"shape": "channel", ...}``: a channel across the whole grid, with a
///   flat top and a lens-shaped cross-section, ``thickness`` deep at its
///   centreline. Keys ``width``, ``thickness`` (optional on a 2D grid),
///   ``azimuth`` (default 0), ``amplitude`` (how far the centreline swings to
///   each side, default 0 for straight channels) and ``wavelength`` (distance
///   between two bends on the same side, needed with ``amplitude``). On a 2D
///   grid, a sinuous band.
/// - ``{"shape": "ellipsoid", ...}``: keys ``radii`` (major, semi-major,
///   minor; the minor radius is optional on a 2D grid), ``azimuth``, ``dip``
///   and ``rake`` of the major axis (default 0). On a 2D grid, an ellipse
///   along ``azimuth``.
///
/// Every set also has ``code``, an integer 0 to 254, and ``proportion``,
/// above 0 and below 1.
///
/// Parameters
/// ----------
/// grid : BlockModel
///     The grid whose geometry the image takes, 2D (one layer) or 3D,
///     optionally rotated.
/// objects : sequence of dict
///     The object sets, in the order they are laid down.
/// background : int, default 0
///     Code of the cells no object covers, 0 to 254.
/// column : str, default "facies"
///     Name of the code column.
/// seed : int, default 0
///     The same seed and objects give the same image on any number of
///     threads.
///
/// Returns
/// -------
/// BlockModel
///     A regular model on the geometry of `grid` with the codes in `column`.
///
/// Raises
/// ------
/// InvalidInput
///     If a set has an unknown shape or key, a size is not positive, a range
///     is reversed, a proportion is outside (0, 1), or a set has not reached
///     its proportion after 100 000 objects.
///
/// Examples
/// --------
/// >>> grid = cs.BlockModel((0, 0, 0), (1, 1, 1), (120, 120, 1))
/// >>> ti = cs.object_training_image(
/// ...     grid,
/// ...     [
/// ...         {"shape": "channel", "code": 1, "proportion": 0.25, "width": 8,
/// ...          "azimuth": (-20, 20), "amplitude": 8, "wavelength": 60},
/// ...         {"shape": "ellipsoid", "code": 2, "proportion": 0.05, "radii": (10, 6)},
/// ...     ],
/// ...     seed=0,
/// ... )
#[pyfunction]
#[pyo3(signature = (grid, objects, *, background=0, column="facies", seed=0))]
fn object_training_image(
    py: Python,
    grid: PyRef<PyBlockModel>,
    objects: Vec<Bound<pyo3::types::PyDict>>,
    background: u8,
    column: &str,
    seed: u64,
) -> PyResult<PyBlockModel> {
    let geometry = *grid.0.geometry();
    let sets = objects
        .iter()
        .map(|set| object_set(set, geometry.count[2] == 1))
        .collect::<PyResult<Vec<_>>>()?;
    let mut model = py
        .detach(|| simulation::object_training_image(geometry, &sets, background, seed, column))
        .map_err(err)?;
    model.crs = grid.0.crs.clone();
    Ok(PyBlockModel(model))
}

/// Realizations, or an estimate, corrected to a target distribution.
///
/// Each realization's values are ranked and the value ranked ``i`` of ``m``
/// is mapped to the reference quantile at ``(i + 0.5) / m``, then moved
/// `strength` of the way from its original value: at 1 every realization
/// takes the reference histogram exactly, at 0 nothing changes. Ranks are
/// kept, ties in the order of the targets, so the spatial pattern of each
/// realization stays; the correction is exact in distribution only, its
/// variogram moves with the values. Realizations are corrected in parallel,
/// identically on any number of threads.
///
/// Parameters
/// ----------
/// values : SimulationSummary or array_like
///     A summary simulated with ``keep=``, its ``(len(kept), targets)``
///     realizations, or one ``(targets,)`` estimate; NaN stays NaN.
/// reference : array_like, KernelDensity or GaussianMixture
///     Data values, whose weighted distribution is the target (quantiles
///     interpolate between the midpoints of the cumulative weights, NaN
///     dropped), or a fitted one-variable distribution.
/// weights : array_like, optional
///     Declustering weights of the reference data.
/// strength : float
///     Share of the correction applied, in [0, 1].
/// realizations : sequence of int, optional
///     The realizations to correct, e.g. those a `check_realizations` shows
///     outside a tolerance; the others are returned unchanged. Default all.
///
/// Returns
/// -------
/// ndarray
///     The corrected values, shaped as `values` (``(n, targets)`` for a
///     summary).
///
/// Raises
/// ------
/// InvalidInput
///     If a summary holds no realizations, `strength` is outside [0, 1], a
///     realization index is out of range, or the reference is empty.
#[pyfunction]
#[pyo3(signature = (values, reference, *, weights=None, strength=1.0, realizations=None))]
fn correct_distribution<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    reference: &Bound<'py, PyAny>,
    weights: Option<&Bound<'py, PyAny>>,
    strength: f64,
    realizations: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, single) = if let Ok(s) = values.cast::<SimulationSummary>() {
        let s = s.get();
        if s.0.kept.is_empty() {
            return Err(invalid(
                "the summary holds no realizations; simulate with keep=",
            ));
        }
        (s.0.realizations.clone(), false)
    } else if let Ok(v) = floats(values, "values") {
        (vec![v], true)
    } else {
        (rows(values, "values")?, false)
    };
    let fitted;
    let empirical;
    let target: &dyn transforms::Reference = match crate::transforms::Reference::extract(reference)
    {
        Ok(r) => {
            fitted = r;
            fitted.distribution()
        }
        Err(_) => {
            let data = floats(reference, "reference")?;
            let w = optional_finite(weights, "weights")?;
            if let Some(w) = &w {
                same_length(data.len(), w.len(), "weights")?;
            }
            let keep: Vec<usize> = (0..data.len()).filter(|&i| !data[i].is_nan()).collect();
            let w = w.map(|w| pick(&w, &keep));
            empirical =
                simulation::Empirical::new(&pick(&data, &keep), w.as_deref()).map_err(err)?;
            &empirical
        }
    };
    let out = py
        .detach(|| {
            simulation::correct_distribution(&rows, target, strength, realizations.as_deref())
        })
        .map_err(err)?;
    Ok(match single {
        true => array1(py, out.into_iter().next().expect("one row")).into_any(),
        false => matrix(py, &out, rows.first().map_or(0, Vec::len)).into_any(),
    })
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
    if let Some(secondary) = &d.secondary {
        columns.push(persist::column("secondary", secondary.iter().copied()));
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
    let secondary = found
        .optional("secondary")
        .ok()
        .map(|_| found.values("secondary"))
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
        secondary,
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
    kept: Vec<usize>,
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
            kept: c.kept.clone(),
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
            kept: m.kept,
            realizations: vec![],
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
        for (k, r) in c.kept.iter().zip(&c.realizations) {
            out.push(column(format!("realization_{k}"), r));
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
        c.realizations = each(c.kept.iter().map(|k| format!("realization_{k}")).collect())?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct CategoricalMeta {
    n: usize,
    categories: usize,
    #[serde(with = "ceres_core::nonfinite")]
    proportions: Vec<Vec<f64>>,
    kept: Vec<usize>,
}

impl Serialize for CategoricalSummary {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let c = &self.0;
        CategoricalMeta {
            n: c.n,
            categories: c.probabilities.len(),
            proportions: c.proportions.clone(),
            kept: c.kept.clone(),
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
            kept: m.kept,
            realizations: vec![],
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
        for (k, r) in c.kept.iter().zip(&c.realizations) {
            let r = r.iter().map(|&v| v as f64);
            out.push(persist::column(&format!("realization_{k}"), r));
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
        c.realizations = c
            .kept
            .iter()
            .map(|k| found.indices(&format!("realization_{k}")))
            .collect::<PyResult<_>>()?;
        Ok(())
    }
}
/// Realizations of a turning-bands factor built together: their bands take
/// about 10 MB each.
const MULTIVARIATE_BATCH: usize = 8;

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
    /// coords : array_like, shape (n, 2) or (n, 3), PointSet or BlockModel
    /// data : array_like, shape (n, variables), or list of str
    ///     One column per simulator, or the columns of `coords` holding them;
    ///     NaN or null marks a missing variable.
    /// weights : array_like or str, optional
    ///     Declustering weights, for the transform when it takes them (PCA,
    ///     StepwiseConditional, PPMT) and for each factor's normal scores.
    /// holes : array_like or str, optional
    ///     Drill-hole ids or names, for `max_per_hole` and the warning on
    ///     samples sharing a location.
    /// impute : bool or GaussianImputer, optional
    ///     Keep samples missing some variables: a `GaussianImputer` fitted to
    ///     the data fills them afresh in every realization, so the imputation
    ///     uncertainty reaches the realizations. Given an imputer, its
    ///     `components`, `spatial` and `neighbors` are used, so with `spatial` the gaps
    ///     also follow nearby samples. The transform is fitted to the complete samples.
    ///     Samples missing every variable are dropped.
    #[pyo3(signature = (coords, data, *, weights=None, holes=None, impute=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        data: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
        impute: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = coords.py();
        let template = match impute {
            None => None,
            Some(t) => match t.cast::<crate::transforms::GaussianImputer>() {
                Ok(t) => Some(Some(t.borrow())),
                Err(_) => t
                    .extract::<bool>()
                    .map_err(|_| invalid("impute must be a bool or a GaussianImputer"))?
                    .then_some(None),
            },
        };
        let impute = template.is_some();
        let data = match data.extract::<Vec<String>>() {
            Ok(names) => {
                let columns = names
                    .iter()
                    .map(|n| floats(&args::named(Some(coords), n, "data")?, "data"))
                    .collect::<PyResult<Vec<_>>>()?;
                let n = columns.first().map_or(0, Vec::len);
                (0..n)
                    .map(|i| columns.iter().map(|c| c[i]).collect())
                    .collect()
            }
            Err(_) => rows(data, "data")?,
        };
        let p = slf.factors.len();
        if data.iter().any(|r: &Vec<f64>| r.len() != p) {
            return Err(invalid(format!(
                "data must have {p} columns, one per simulator"
            )));
        }
        let locs = points(coords)?;
        same_length(locs.len(), data.len(), "data")?;
        let weights = optional_finite(resolve(coords, weights, "weights")?.as_ref(), "weights")?;
        if let Some(w) = &weights {
            same_length(data.len(), w.len(), "weights")?;
        }
        let holes = args::holes(resolve(coords, holes, "holes")?.as_ref(), data.len())?;
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
        let imputed = match template {
            None => None,
            Some(t) => {
                let imputer = match t {
                    Some(t) => t.fit_like(&data, weights.as_deref())?,
                    None => transforms::GaussianImputer::fit(&data, weights.as_deref())
                        .map_err(invalid)?,
                };
                Some((imputer, pick(&data, &keep)))
            }
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
    /// mixed from `seed`, `k` and `j`, so no two factors share random numbers.
    ///
    /// Returns
    /// -------
    /// list of SimulationSummary
    ///     One per variable, in column order.
    ///
    /// `progress` shows a `tqdm` bar over the realizations.
    #[pyo3(signature = (targets, *, n=100, seed=0, cutoffs=vec![], quantiles=vec![], keep=None, anisotropy=None, blocks=None, progress=true))]
    #[allow(clippy::too_many_arguments)]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        keep: Option<&Bound<PyAny>>,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        blocks: Option<PyRef<PyBlockModel>>,
        progress: bool,
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
            keep: keep_arg(keep)?,
        };
        let bands_only = !grid.is_empty()
            && self.factors.iter().all(|factor| {
                matches!(factor, Factor::Bands(_, params)
                    if f.imputed.is_none() || params.search.high_grade.is_none())
            });
        with_progress(py, Some(n as u64), progress, |counter| {
            if bands_only {
                let p = self.factors.len();
                let (lo, hi) = simulation::bounds(&grid);
                return simulation::multivariate_batched(
                    n,
                    p,
                    &f.transform,
                    &options,
                    support.as_ref(),
                    MULTIVARIATE_BATCH.min(n),
                    |j, ks| {
                        let Factor::Bands(variogram, params) = &self.factors[j] else {
                            unreachable!("checked")
                        };
                        let values: Vec<Vec<f64>> = match &f.imputed {
                            None => vec![f.columns[j].clone()],
                            Some((imputer, data)) => ks
                                .clone()
                                .map(|k| {
                                    let rows = imputer
                                        .impute(
                                            data,
                                            Some(&f.locs),
                                            simulation::factor_seed(seed, k, p),
                                        )
                                        .map_err(|e| {
                                            simulation::SimError::Transform(e.to_string())
                                        })?;
                                    Ok(f.transform.forward(&rows).iter().map(|r| r[j]).collect())
                                })
                                .collect::<simulation::Result<_>>()?,
                        };
                        let seeds: Vec<u64> = ks
                            .clone()
                            .map(|k| simulation::factor_seed(seed, k, j))
                            .collect();
                        simulation::TurningBandsEnsemble::with_realizations(
                            &f.locs,
                            &values,
                            f.weights.as_deref(),
                            f.holes.as_deref(),
                            lo,
                            hi,
                            variogram,
                            params,
                            &seeds,
                        )?
                        .realizations(0..ks.len(), &grid, |_| None, None)
                    },
                    counter,
                );
            }
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
                                .impute(data, Some(&f.locs), simulation::factor_seed(seed, k, p))
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
                counter,
            )
        })?
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
    m.add_function(wrap_pyfunction!(object_training_image, m)?)?;
    m.add_function(wrap_pyfunction!(correct_distribution, m)?)?;
    m.add_class::<SimulationSummary>()?;
    m.add_class::<CategoricalSummary>()?;
    Ok(())
}
