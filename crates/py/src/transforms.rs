use pyo3::prelude::*;
use pyo3::types::PyDict;
use transforms::{
    HermiteAnamorphosis, Nscore, Ppmt as CorePpmt, PpmtParams, Recovery, Trend as CoreTrend,
    UniformConditioning as CoreUc, Weights,
};

use crate::args::{
    array1, array2, finite, optional_finite, points, points_array, rows, same_length, triple,
};
use crate::invalid;

fn err(e: transforms::TransformError) -> PyErr {
    invalid(e)
}

fn not_fitted(name: &str) -> PyErr {
    invalid(format!("{name} is not fitted; call fit first"))
}

fn map<'py>(py: Python<'py>, values: Vec<f64>, f: impl Fn(f64) -> f64) -> Bound<'py, PyAny> {
    array1(py, values.into_iter().map(f).collect()).into_any()
}

fn recoveries<'py>(py: Python<'py>, r: &[Recovery]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let col = |f: fn(&Recovery) -> f64| array1(py, r.iter().map(f).collect());
    d.set_item("cutoff", col(|r| r.cutoff))?;
    d.set_item("tonnage", col(|r| r.tonnage))?;
    d.set_item("metal", col(|r| r.metal))?;
    d.set_item("mean_grade", col(|r| r.mean_grade))?;
    d.set_item("benefit", col(|r| r.benefit))?;
    Ok(d)
}

/// Normal-score transform through the (weighted) empirical CDF. Beyond the
/// data, values interpolate in probability toward `tails` (lower, upper),
/// which default to the data range.
#[pyclass(module = "ceres", name = "NormalScore")]
pub struct NormalScore {
    tails: Option<(f64, f64)>,
    fitted: Option<Nscore>,
}

impl NormalScore {
    fn fitted(&self) -> PyResult<&Nscore> {
        self.fitted
            .as_ref()
            .ok_or_else(|| not_fitted("NormalScore"))
    }
}

#[pymethods]
impl NormalScore {
    #[new]
    #[pyo3(signature = (tails=None))]
    fn new(tails: Option<(f64, f64)>) -> Self {
        Self {
            tails,
            fitted: None,
        }
    }

    #[pyo3(signature = (values, weights=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let values = finite(values, "values")?;
        let weights = optional_finite(weights, "weights")?;
        if let Some(w) = &weights {
            same_length(values.len(), w.len(), "weights")?;
        }
        let mut ns = transforms::nscore_transform(&values, weights.as_deref()).map_err(err)?;
        if let Some((lower, upper)) = slf.tails {
            ns.table = ns.table.with_tails(lower, upper);
        }
        slf.fitted = Some(ns);
        Ok(slf)
    }

    /// Scores of the fitted values, exact per rank.
    #[pyo3(signature = (values, weights=None))]
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let slf = Self::fit(slf, values, weights)?;
        Ok(array1(py, slf.fitted()?.scores.clone()).into_any())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let table = &self.fitted()?.table;
        Ok(map(py, finite(values, "values")?, |v| table.forward(v)))
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        scores: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let table = &self.fitted()?.table;
        Ok(map(py, finite(scores, "scores")?, |y| table.back(y)))
    }

    /// Transformation table: sorted values and their scores.
    #[getter]
    fn table_<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let t = &self.fitted()?.table;
        Ok((
            array1(py, t.values.clone()).into_any(),
            array1(py, t.scores.clone()).into_any(),
        ))
    }
}

/// Gaussian anamorphosis expanded in Hermite polynomials.
#[pyclass(module = "ceres", name = "HermiteAnamorphosis")]
pub struct Anamorphosis {
    degree: usize,
    fitted: Option<HermiteAnamorphosis>,
}

impl Anamorphosis {
    fn fitted(&self) -> PyResult<&HermiteAnamorphosis> {
        self.fitted
            .as_ref()
            .ok_or_else(|| not_fitted("HermiteAnamorphosis"))
    }

    pub fn inner(&self) -> PyResult<HermiteAnamorphosis> {
        self.fitted().cloned()
    }

    pub fn from_inner(a: HermiteAnamorphosis) -> Self {
        Self {
            degree: a.degree(),
            fitted: Some(a),
        }
    }
}

#[pymethods]
impl Anamorphosis {
    #[new]
    #[pyo3(signature = (degree=30))]
    fn new(degree: usize) -> Self {
        Self {
            degree,
            fitted: None,
        }
    }

    #[pyo3(signature = (values, weights=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let values = finite(values, "values")?;
        let weights = optional_finite(weights, "weights")?;
        let a = HermiteAnamorphosis::fit(&values, weights.as_deref(), slf.degree).map_err(err)?;
        slf.fitted = Some(a);
        Ok(slf)
    }

    /// Raw values to Gaussian scores.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = self.fitted()?;
        Ok(map(py, finite(values, "values")?, |z| a.forward(z)))
    }

    /// Gaussian scores to raw values.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        scores: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = self.fitted()?;
        Ok(map(py, finite(scores, "scores")?, |y| a.back(y)))
    }

    #[getter]
    fn coefficients_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(array1(py, self.fitted()?.coefficients.clone()).into_any())
    }

    #[getter]
    fn mean_(&self) -> PyResult<f64> {
        Ok(self.fitted()?.mean())
    }

    #[getter]
    fn variance_(&self) -> PyResult<f64> {
        Ok(self.fitted()?.variance())
    }

    /// Block anamorphosis for change-of-support coefficient `r` in (0, 1].
    fn block(&self, r: f64) -> PyResult<Self> {
        if !(r > 0.0 && r <= 1.0) {
            return Err(invalid("r must be in (0, 1]"));
        }
        Ok(Self::from_inner(self.fitted()?.block(r)))
    }

    /// Tonnage, metal, mean grade and benefit above each cutoff.
    fn grade_tonnage<'py>(
        &self,
        py: Python<'py>,
        cutoffs: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let cutoffs = finite(cutoffs, "cutoffs")?;
        recoveries(py, &transforms::grade_tonnage(self.fitted()?, &cutoffs))
    }
}

/// Box-Cox power transform; `lambda_=None` picks the least-skewed lambda.
#[pyclass(module = "ceres", name = "BoxCox")]
pub struct BoxCox {
    requested: Option<f64>,
    lambda: Option<f64>,
}

#[pymethods]
impl BoxCox {
    #[new]
    #[pyo3(signature = (lambda_=None))]
    fn new(lambda_: Option<f64>) -> Self {
        Self {
            requested: lambda_,
            lambda: lambda_,
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        values: &Bound<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let values = finite(values, "values")?;
        let lambda = match slf.requested {
            Some(l) => l,
            None => {
                let candidates: Vec<f64> = (-20..=20).map(|i| i as f64 / 10.0).collect();
                transforms::optimal_lambda(&values, &candidates).map_err(err)?
            }
        };
        slf.lambda = Some(lambda);
        Ok(slf)
    }

    #[getter]
    fn lambda_(&self) -> PyResult<f64> {
        self.lambda.ok_or_else(|| not_fitted("BoxCox"))
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let out = transforms::box_cox(&finite(values, "values")?, self.lambda_()?).map_err(err)?;
        Ok(array1(py, out).into_any())
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let lambda = self.lambda_()?;
        Ok(map(py, finite(values, "values")?, |y| {
            transforms::box_cox_inverse(y, lambda)
        }))
    }
}

/// Projection-pursuit multivariate transform to independent Gaussians.
#[pyclass(module = "ceres", name = "PPMT")]
pub struct Ppmt {
    params: PpmtParams,
    fitted: Option<CorePpmt>,
}

impl Ppmt {
    fn fitted(&self) -> PyResult<&CorePpmt> {
        self.fitted.as_ref().ok_or_else(|| not_fitted("PPMT"))
    }
}

#[pymethods]
impl Ppmt {
    #[new]
    #[pyo3(signature = (iterations=30, candidates=60, seed=1))]
    fn new(iterations: usize, candidates: usize, seed: u64) -> Self {
        Self {
            params: PpmtParams {
                iterations,
                candidates,
                seed,
            },
            fitted: None,
        }
    }

    /// `data` is `(n, d)`.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = rows(data, "data")?;
        slf.fitted = Some(CorePpmt::fit(&data, &slf.params).map_err(err)?);
        Ok(slf)
    }

    fn transform<'py>(&self, py: Python<'py>, data: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        Ok(array2(py, &self.fitted()?.forward(&rows(data, "data")?)).into_any())
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(array2(py, &self.fitted()?.back(&rows(data, "data")?)).into_any())
    }
}

/// Uniform conditioning of panel estimates to SMU recoveries.
#[pyclass(module = "ceres", name = "UniformConditioning", frozen)]
pub struct UniformConditioning(CoreUc);

#[pymethods]
impl UniformConditioning {
    /// `anamorphosis` is the fitted point anamorphosis; `r_smu`, `r_panel` the
    /// change-of-support coefficients of the SMU and the panel.
    #[new]
    fn new(anamorphosis: PyRef<Anamorphosis>, r_smu: f64, r_panel: f64) -> PyResult<Self> {
        Ok(Self(
            CoreUc::new(anamorphosis.inner()?, r_smu, r_panel).map_err(err)?,
        ))
    }

    fn panel_recovery<'py>(
        &self,
        py: Python<'py>,
        panel_grade: f64,
        cutoffs: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let cutoffs = finite(cutoffs, "cutoffs")?;
        recoveries(py, &self.0.panel_recovery(panel_grade, &cutoffs))
    }

    fn localized_grades<'py>(
        &self,
        py: Python<'py>,
        panel_grade: f64,
        n_smu: usize,
    ) -> Bound<'py, PyAny> {
        array1(py, self.0.localized_grades(panel_grade, n_smu)).into_any()
    }
}

/// Polynomial trend in the coordinates.
#[pyclass(module = "ceres", name = "Trend", frozen)]
pub struct Trend(CoreTrend);

#[pymethods]
impl Trend {
    #[getter]
    fn degree(&self) -> usize {
        self.0.degree
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array1(py, self.0.coeffs.clone()).into_any()
    }

    fn predict<'py>(&self, py: Python<'py>, coords: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let values = points(coords)?.iter().map(|p| self.0.eval(p)).collect();
        Ok(array1(py, values).into_any())
    }
}

/// Fits a polynomial trend of `degree` (0-2); returns the trend and residuals.
#[pyfunction]
#[pyo3(signature = (coords, values, degree=1))]
fn detrend<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    degree: usize,
) -> PyResult<(Trend, Bound<'py, PyAny>)> {
    let (locs, values) = (points(coords)?, finite(values, "values")?);
    same_length(locs.len(), values.len(), "values")?;
    let (trend, residuals) = transforms::detrend(&locs, &values, degree).map_err(err)?;
    Ok((Trend(trend), array1(py, residuals).into_any()))
}

/// Declustering weights (normalised to sum to n) and the declustered mean.
#[pyclass(module = "ceres", name = "Declustering", frozen)]
pub struct Declustering {
    #[pyo3(get)]
    mean: f64,
    #[pyo3(get)]
    cell_size: f64,
    weights: Vec<f64>,
    sizes: Vec<f64>,
    means: Vec<f64>,
}

#[pymethods]
impl Declustering {
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array1(py, self.weights.clone()).into_any()
    }

    /// Cell sizes scanned and the declustered mean at each; empty when fixed.
    #[getter]
    fn sizes<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array1(py, self.sizes.clone()).into_any()
    }

    #[getter]
    fn means<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array1(py, self.means.clone()).into_any()
    }

    fn __repr__(&self) -> String {
        format!(
            "Declustering(mean={:.6}, cell_size={}, n={})",
            self.mean,
            self.cell_size,
            self.weights.len()
        )
    }
}

fn declustering(w: Weights, sizes: Vec<f64>, means: Vec<f64>) -> Declustering {
    Declustering {
        mean: w.declustered_mean,
        cell_size: w.cell_size,
        weights: w.weights,
        sizes,
        means,
    }
}

/// Cell declustering. Without `cell_size`, scans `sizes` (default: 30 sizes up
/// to half the largest extent), each averaged over `offsets` grid origins, and
/// keeps the size with the lowest mean (`minimize=False`: highest).
#[pyfunction]
#[pyo3(signature = (coords, values, cell_size=None, sizes=None, offsets=25, minimize=true))]
fn cell_declustering(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    cell_size: Option<f64>,
    sizes: Option<&Bound<PyAny>>,
    offsets: usize,
    minimize: bool,
) -> PyResult<Declustering> {
    let (locs, values) = (points(coords)?, finite(values, "values")?);
    same_length(locs.len(), values.len(), "values")?;
    let origin = locs.iter().fold((f64::MAX, f64::MAX, f64::MAX), |m, p| {
        (m.0.min(p.0), m.1.min(p.1), m.2.min(p.2))
    });
    if let Some(size) = cell_size {
        let w = transforms::cell_weights(&locs, &values, size, origin).map_err(err)?;
        return Ok(declustering(w, vec![], vec![]));
    }
    let sizes = match sizes {
        Some(s) => finite(s, "sizes")?,
        None => {
            let extent = locs.iter().fold(0.0_f64, |e, p| {
                e.max(p.0 - origin.0)
                    .max(p.1 - origin.1)
                    .max(p.2 - origin.2)
            });
            (1..=30).map(|i| extent / 2.0 * i as f64 / 30.0).collect()
        }
    };
    let means = sizes
        .iter()
        .map(|&s| transforms::decluster_mean_over_offsets(&locs, &values, s, offsets))
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let (best, _) =
        transforms::optimal_cell_size(&locs, &values, &sizes, offsets, !minimize).map_err(err)?;
    let w = transforms::cell_weights(&locs, &values, best, origin).map_err(err)?;
    Ok(declustering(w, sizes, means))
}

/// Polygonal (nearest-neighbour area) declustering on a `nodes`-cell grid.
#[pyfunction]
#[pyo3(signature = (coords, values, nodes=10_000))]
fn polygon_declustering(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    nodes: usize,
) -> PyResult<Declustering> {
    let (locs, values) = (points(coords)?, finite(values, "values")?);
    same_length(locs.len(), values.len(), "values")?;
    let w = transforms::polygon_weights(&locs, &values, nodes).map_err(err)?;
    Ok(declustering(w, vec![], vec![]))
}

/// Point-to-block support correction keeping the mean: `f` is the variance
/// reduction factor `Var(Z_v) / Var(Z)`.
#[pyfunction]
#[pyo3(signature = (values, f, weights=None))]
fn affine_correction<'py>(
    py: Python<'py>,
    values: &Bound<PyAny>,
    f: f64,
    weights: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let w = optional_finite(weights, "weights")?;
    let out =
        transforms::affine_correction(&finite(values, "values")?, w.as_deref(), f).map_err(err)?;
    Ok(array1(py, out).into_any())
}

#[pyfunction]
#[pyo3(signature = (values, f, weights=None))]
fn indirect_lognormal_correction<'py>(
    py: Python<'py>,
    values: &Bound<PyAny>,
    f: f64,
    weights: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let w = optional_finite(weights, "weights")?;
    let out =
        transforms::indirect_lognormal_correction(&finite(values, "values")?, w.as_deref(), f)
            .map_err(err)?;
    Ok(array1(py, out).into_any())
}

/// Averages samples into blocks; returns centres, means and counts.
#[pyfunction]
#[pyo3(signature = (coords, values, block_size, origin=vec![0.0, 0.0, 0.0]))]
fn upscale<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    block_size: Vec<f64>,
    origin: Vec<f64>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (locs, values) = (points(coords)?, finite(values, "values")?);
    same_length(locs.len(), values.len(), "values")?;
    let blocks = transforms::upscale(
        &locs,
        &values,
        triple(block_size, 1.0, "block_size")?,
        triple(origin, 0.0, "origin")?,
    )
    .map_err(err)?;
    let centers: Vec<_> = blocks.iter().map(|b| b.center).collect();
    Ok((
        points_array(py, &centers).into_any(),
        array1(py, blocks.iter().map(|b| b.value).collect()).into_any(),
        array1(py, blocks.iter().map(|b| b.count as f64).collect()).into_any(),
    ))
}

/// Splits one block into `refine` sub-cells carrying its value.
#[pyfunction]
fn downscale<'py>(
    py: Python<'py>,
    center: Vec<f64>,
    block_size: Vec<f64>,
    value: f64,
    refine: (usize, usize, usize),
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let cells = transforms::downscale(
        triple(center, 0.0, "center")?,
        triple(block_size, 1.0, "block_size")?,
        value,
        refine,
    )
    .map_err(err)?;
    let centers: Vec<_> = cells.iter().map(|c| c.0).collect();
    Ok((
        points_array(py, &centers).into_any(),
        array1(py, cells.iter().map(|c| c.1).collect()).into_any(),
    ))
}

/// Standard normal CDF.
#[pyfunction]
fn normal_cdf<'py>(py: Python<'py>, x: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(map(py, crate::args::floats(x, "x")?, transforms::phi))
}

/// Standard normal quantile function.
#[pyfunction]
fn normal_ppf<'py>(py: Python<'py>, p: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(map(py, crate::args::floats(p, "p")?, transforms::probit))
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<NormalScore>()?;
    m.add_class::<Anamorphosis>()?;
    m.add_class::<BoxCox>()?;
    m.add_class::<Ppmt>()?;
    m.add_class::<UniformConditioning>()?;
    m.add_class::<Trend>()?;
    m.add_class::<Declustering>()?;
    m.add_function(wrap_pyfunction!(detrend, m)?)?;
    m.add_function(wrap_pyfunction!(cell_declustering, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_declustering, m)?)?;
    m.add_function(wrap_pyfunction!(affine_correction, m)?)?;
    m.add_function(wrap_pyfunction!(indirect_lognormal_correction, m)?)?;
    m.add_function(wrap_pyfunction!(upscale, m)?)?;
    m.add_function(wrap_pyfunction!(downscale, m)?)?;
    m.add_function(wrap_pyfunction!(normal_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(normal_ppf, m)?)?;
    Ok(())
}
