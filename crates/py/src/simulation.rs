use pyo3::prelude::*;
use pyo3::types::PyDict;
use simulation::{
    GibbsParams, PgsParams, Realization, Region, SgsParams, SisParams, TruncationRule,
    TurningBandsParams,
};
use variogram::Variogram as CoreVariogram;

use crate::args::{
    self, Point, array1, array2, distinct, finite, optional_finite, pick, points, rows, same_length,
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

/// Realizations as an `(n, targets)` array.
fn stack<'py>(py: Python<'py>, reals: Vec<Realization>) -> Bound<'py, PyAny> {
    let rows: Vec<Vec<f64>> = reals.into_iter().map(|r| r.values).collect();
    array2(py, &rows).into_any()
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

    /// `n` realizations at `targets`, seeds `seed, seed + 1, …`; `(n, targets)`.
    /// `anisotropy` (a LocalAnisotropy) orients each node's variogram and search.
    #[pyo3(signature = (targets, n=1, seed=0, anisotropy=None))]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let local = anisotropy.map(|a| a.at_targets(&grid));
        let params = SgsParams {
            search: self.search.clone(),
            seed,
        };
        let reals = py
            .detach(|| {
                simulation::sgs_ensemble(
                    &d.locs,
                    &d.values,
                    d.weights.as_deref(),
                    &grid,
                    &self.variogram,
                    &params,
                    local.as_ref(),
                    n,
                )
            })
            .map_err(err)?;
        Ok(stack(py, reals))
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

    #[pyo3(signature = (targets, n=1, seed=0))]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let d = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let params = TurningBandsParams {
            n_bands: self.bands,
            step: self.step,
            seed,
            ..Default::default()
        };
        let reals = py
            .detach(|| {
                simulation::turning_bands_ensemble(
                    &d.locs,
                    &d.values,
                    d.weights.as_deref(),
                    &grid,
                    &self.variogram,
                    &params,
                    n,
                )
            })
            .map_err(err)?;
        Ok(stack(py, reals))
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

    /// `(n, targets)` integer array of categories.
    #[pyo3(signature = (targets, n=1, seed=0))]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        n: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (locs, cats) = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let k = self.variograms.len();
        let reals: Vec<Vec<usize>> = py
            .detach(|| {
                use rayon::prelude::*;
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let params = SisParams {
                            search: self.search.clone(),
                            seed: seed.wrapping_add(i as u64),
                        };
                        simulation::sis(locs, cats, &grid, k, &self.variograms, &params)
                            .map(|r| r.categories)
                    })
                    .collect::<Result<_, _>>()
            })
            .map_err(err)?;
        let np = py.import("numpy")?;
        np.call_method1("asarray", (reals, "int64"))
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

    #[pyo3(signature = (targets, seed=0))]
    fn simulate<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (locs, facies) = self.data.as_ref().ok_or_else(not_fitted)?;
        let grid = self::targets(targets)?;
        let params = PgsParams {
            seed,
            two_fields: self.two_fields,
            ..Default::default()
        };
        let out = py
            .detach(|| {
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
            .map_err(err)?;
        py.import("numpy")?.call_method1("asarray", (out, "int64"))
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

fn realizations(obj: &Bound<PyAny>) -> PyResult<Vec<Realization>> {
    Ok(rows(obj, "realizations")?
        .into_iter()
        .map(|values| Realization { values })
        .collect())
}

/// Per-target mean, standard deviation and quantiles of `(n, targets)` realizations.
#[pyfunction]
#[pyo3(signature = (realizations, quantiles=vec![0.1, 0.5, 0.9]))]
fn summarize_realizations<'py>(
    py: Python<'py>,
    realizations: &Bound<PyAny>,
    quantiles: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let stats = simulation::node_stats(&self::realizations(realizations)?, &quantiles);
    let d = PyDict::new(py);
    d.set_item("mean", array1(py, stats.iter().map(|s| s.mean).collect()))?;
    d.set_item("std", array1(py, stats.iter().map(|s| s.std_dev).collect()))?;
    for (i, q) in quantiles.iter().enumerate() {
        d.set_item(
            format!("q{q}"),
            array1(py, stats.iter().map(|s| s.quantiles[i]).collect()),
        )?;
    }
    Ok(d)
}

/// Per-target fraction of realizations above `cutoff`.
#[pyfunction]
fn probability_above<'py>(
    py: Python<'py>,
    realizations: &Bound<PyAny>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let p = simulation::probability_above(&self::realizations(realizations)?, cutoff);
    Ok(array1(py, p).into_any())
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Sgs>()?;
    m.add_class::<TurningBands>()?;
    m.add_class::<Sis>()?;
    m.add_class::<Plurigaussian>()?;
    m.add_function(wrap_pyfunction!(gibbs, m)?)?;
    m.add_function(wrap_pyfunction!(summarize_realizations, m)?)?;
    m.add_function(wrap_pyfunction!(probability_above, m)?)?;
    Ok(())
}
