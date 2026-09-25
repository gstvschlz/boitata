use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use simulation::{
    CategoricalSummary as CoreCategorical, ContinuousOptions, ContinuousSummary, GibbsParams,
    PgsParams, Region, SgsParams, SisParams, TruncationRule, TurningBandsParams,
};
use variogram::Variogram as CoreVariogram;

use crate::args::{
    self, Point, array1, distinct, finite, optional_finite, pick, points, rows, same_length,
};
use crate::estimation::{Search, targets};
use crate::invalid;
use crate::variogram::Variogram;

fn err(e: simulation::SimError) -> PyErr {
    invalid(e)
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
#[pyclass(module = "ceres", name = "SGS")]
pub struct Sgs {
    variogram: CoreVariogram,
    search: estimation::Search,
    data: Option<Data>,
}

#[pymethods]
impl Sgs {
    #[new]
    fn new(variogram: Variogram, search: Search) -> Self {
        Self {
            variogram: variogram.0,
            search: search.0,
            data: None,
        }
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
    #[pyo3(signature = (targets, n=100, seed=0, cutoffs=vec![], quantiles=vec![], realizations=false, anisotropy=None))]
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
    ) -> PyResult<SimulationSummary> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
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
                .map(|r| r.values)
            })
        })
        .map(SimulationSummary)
        .map_err(err)
    }
}

/// Turning-bands simulation conditioned by kriging; same conventions as SGS.
#[pyclass(module = "ceres", name = "TurningBands")]
pub struct TurningBands {
    variogram: CoreVariogram,
    bands: usize,
    step: Option<f64>,
    data: Option<Data>,
}

#[pymethods]
impl TurningBands {
    /// `bands` lines, each discretized every `step` metres along the major axis
    /// (default: a fiftieth of the shortest range).
    #[new]
    #[pyo3(signature = (variogram, bands=300, step=None))]
    fn new(variogram: Variogram, bands: usize, step: Option<f64>) -> Self {
        Self {
            variogram: variogram.0,
            bands,
            step,
            data: None,
        }
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
    #[pyo3(signature = (targets, n=100, seed=0, cutoffs=vec![], quantiles=vec![], realizations=false))]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        realizations: bool,
    ) -> PyResult<SimulationSummary> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let options = ContinuousOptions {
            cutoffs,
            quantiles,
            keep: realizations,
        };
        py.detach(|| {
            simulation::continuous(n, &options, |k| {
                let params = TurningBandsParams {
                    n_bands: self.bands,
                    step: self.step,
                    seed: seed.wrapping_add(k as u64),
                    ..Default::default()
                };
                simulation::turning_bands(
                    &d.locs,
                    &d.values,
                    d.weights.as_deref(),
                    &grid,
                    &self.variogram,
                    &params,
                )
                .map(|r| r.values)
            })
        })
        .map(SimulationSummary)
        .map_err(err)
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
#[pyclass(module = "ceres", name = "SIS")]
pub struct Sis {
    variograms: Vec<CoreVariogram>,
    search: estimation::Search,
    data: Option<(Vec<Point>, Vec<usize>)>,
}

#[pymethods]
impl Sis {
    #[new]
    fn new(variograms: Vec<Variogram>, search: Search) -> Self {
        Self {
            variograms: variograms.into_iter().map(|v| v.0).collect(),
            search: search.0,
            data: None,
        }
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
    /// realizations themselves only when `realizations`.
    #[pyo3(signature = (targets, n=100, seed=0, realizations=false))]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        realizations: bool,
    ) -> PyResult<CategoricalSummary> {
        let (locs, cats) = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let k = self.variograms.len();
        py.detach(|| {
            simulation::categorical(n, k, realizations, |i| {
                let params = SisParams {
                    search: self.search.clone(),
                    seed: seed.wrapping_add(i as u64),
                };
                simulation::sis(locs, cats, &grid, k, &self.variograms, &params)
                    .map(|r| r.categories)
            })
        })
        .map(CategoricalSummary)
        .map_err(err)
    }
}

/// Plurigaussian simulation. Give `proportions` for an ordered rule on one
/// field, or `regions` as `(y1_low, y1_high, y2_low, y2_high, facies)`.
#[pyclass(module = "ceres", name = "Plurigaussian")]
pub struct Plurigaussian {
    variograms: (CoreVariogram, CoreVariogram),
    rule: TruncationRule,
    two_fields: bool,
    data: Option<(Vec<Point>, Vec<usize>)>,
}

#[pymethods]
impl Plurigaussian {
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
    #[pyo3(signature = (targets, n=100, seed=0, realizations=false))]
    fn simulate(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        realizations: bool,
    ) -> PyResult<CategoricalSummary> {
        let (locs, facies) = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
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

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Sgs>()?;
    m.add_class::<TurningBands>()?;
    m.add_class::<Sis>()?;
    m.add_class::<Plurigaussian>()?;
    m.add_function(wrap_pyfunction!(gibbs, m)?)?;
    m.add_class::<SimulationSummary>()?;
    m.add_class::<CategoricalSummary>()?;
    Ok(())
}
