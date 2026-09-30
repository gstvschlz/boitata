use std::sync::Arc;

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use transforms::{
    HermiteAnamorphosis, KernelTrend, Maf as CoreMaf, NormalScore as CoreNormalScore,
    Pca as CorePca, Ppmt as CorePpmt, PpmtParams, Recovery, StepwiseConditional as CoreSct,
    Trend as CoreTrend, UniformConditioning as CoreUc, Weights,
};

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch};

use crate::args::{
    Point, array1, array2, bools, column, finite, floats, optional_finite, per_row, points,
    points_array, rows, same_length, triple,
};
use crate::containers::PyBlockModel;
use crate::invalid;
use crate::table::Table;

fn err(e: transforms::TransformError) -> PyErr {
    invalid(e)
}

fn not_fitted(name: &str) -> PyErr {
    invalid(format!("{name} is not fitted; call fit first"))
}

fn map<'py>(py: Python<'py>, values: Vec<f64>, f: impl Fn(f64) -> f64) -> Bound<'py, PyAny> {
    array1(py, values.into_iter().map(f).collect()).into_any()
}

fn recoveries(r: &[Recovery]) -> PyResult<Table> {
    let col = |f: fn(&Recovery) -> f64| -> ArrayRef {
        Arc::new(
            r.iter()
                .map(|r| Some(f(r)).filter(|v| v.is_finite()))
                .collect::<Float64Array>(),
        )
    };
    let columns = [
        ("cutoff", col(|r| r.cutoff)),
        ("tonnage", col(|r| r.tonnage)),
        ("mean_grade", col(|r| r.mean_grade)),
        ("metal", col(|r| r.metal)),
        ("benefit", col(|r| r.benefit)),
    ];
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// Normal-score transform through the (weighted) empirical CDF, or through a
/// smooth reference distribution.
///
/// Parameters
/// ----------
/// tails : (float, float), optional
///     Values at cumulative probability 0 and 1. Beyond the data, values
///     interpolate in probability toward them; they default to the data
///     range, or to the bounds of the reference.
/// reference : KernelDensity or GaussianMixture, optional
///     A fitted one-variable distribution to take the scores against, in
///     place of the data: its quantiles at scores -5 to 5 form the table, so
///     few or clustered data still get smooth tails. Fit the reference with
///     the declustering weights; `fit` then takes no weights.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "NormalScore")]
pub struct NormalScore {
    #[serde(with = "boitata_core::nonfinite")]
    tails: Option<(f64, f64)>,
    reference: Option<Reference>,
    fitted: Option<CoreNormalScore>,
}

/// A fitted reference distribution of `NormalScore`.
#[derive(Serialize, Deserialize)]
pub(crate) enum Reference {
    KernelDensity(transforms::KernelDensity),
    GaussianMixture(transforms::GaussianMixture),
}

impl Reference {
    pub(crate) fn extract(obj: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(kde) = obj.cast::<KernelDensity>() {
            return Ok(Self::KernelDensity(kde.borrow().fitted()?.clone()));
        }
        if let Ok(gm) = obj.cast::<GaussianMixture>() {
            let gm = gm.borrow().fitted()?.clone();
            if gm.dim() != 1 {
                return Err(invalid("reference needs a one-variable GaussianMixture"));
            }
            return Ok(Self::GaussianMixture(gm));
        }
        Err(invalid(
            "reference must be a KernelDensity or a GaussianMixture",
        ))
    }

    pub(crate) fn distribution(&self) -> &dyn transforms::Reference {
        match self {
            Self::KernelDensity(r) => r,
            Self::GaussianMixture(r) => r,
        }
    }

    fn transform(&self, values: &[f64]) -> CoreNormalScore {
        match self {
            Self::KernelDensity(r) => transforms::from_reference(values, r),
            Self::GaussianMixture(r) => transforms::from_reference(values, r),
        }
    }
}

impl NormalScore {
    fn fitted(&self) -> PyResult<&CoreNormalScore> {
        self.fitted
            .as_ref()
            .ok_or_else(|| not_fitted("NormalScore"))
    }
}

#[pymethods]
impl NormalScore {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, tails=None, reference=None))]
    fn new(tails: Option<(f64, f64)>, reference: Option<&Bound<PyAny>>) -> PyResult<Self> {
        Ok(Self {
            tails,
            reference: reference.map(Reference::extract).transpose()?,
            fitted: None,
        })
    }

    /// Fits the empirical (or reference) distribution.
    ///
    /// Parameters
    /// ----------
    /// values : array_like
    ///     Sample values, in the units to transform.
    /// weights : array_like, optional
    ///     Declustering weights; refused together with `reference`, whose own
    ///     `fit` takes them instead.
    /// censored : array_like of bool, optional
    ///     Marks values reported at their detection limit rather than
    ///     measured exactly (those cells still hold the limit as their
    ///     numeric value). Values keep their exact rank against every
    ///     uncensored value and against censored values at another limit;
    ///     censored values tied at the same limit have no true order between
    ///     them, so each such tie is shuffled with `seed` before scoring
    ///     instead of averaged into one repeated score. Refused together with
    ///     `reference`.
    /// seed : int, default 0
    ///     Seed for breaking ties among `censored` values at the same limit.
    #[pyo3(signature = (values, *, weights=None, censored=None, seed=0))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        censored: Option<&Bound<PyAny>>,
        seed: u64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let values = finite(values, "values")?;
        let weights = optional_finite(weights, "weights")?;
        if let Some(w) = &weights {
            same_length(values.len(), w.len(), "weights")?;
        }
        let censored = censored.map(|c| bools(c, "censored")).transpose()?;
        if let Some(c) = &censored {
            same_length(values.len(), c.len(), "censored")?;
        }
        if censored.is_some() && slf.reference.is_some() {
            return Err(invalid("reference does not take censored values"));
        }
        let mut ns = match (&slf.reference, &weights) {
            (Some(_), Some(_)) => {
                return Err(invalid(
                    "with a reference, give the weights to the reference's fit",
                ));
            }
            (Some(r), None) => r.transform(&values),
            (None, _) => match &censored {
                Some(c) => transforms::transform_censored(&values, c, weights.as_deref(), seed)
                    .map_err(err)?,
                None => transforms::normal_score(&values, weights.as_deref()).map_err(err)?,
            },
        };
        if let Some((lower, upper)) = slf.tails {
            ns.table = ns.table.with_tails(lower, upper);
        }
        slf.fitted = Some(ns);
        Ok(slf)
    }

    /// Scores of the fitted values, exact per rank.
    #[pyo3(signature = (values, *, weights=None, censored=None, seed=0))]
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        censored: Option<&Bound<PyAny>>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let slf = Self::fit(slf, values, weights, censored, seed)?;
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
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "HermiteAnamorphosis")]
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
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, degree=30))]
    fn new(degree: usize) -> Self {
        Self {
            degree,
            fitted: None,
        }
    }

    #[pyo3(signature = (values, *, weights=None))]
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

    #[pyo3(signature = (values, *, weights=None))]
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        values: &Bound<'py, PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        Self::fit(slf, values, weights)?.transform(py, values)
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

    /// Recoveries above each cutoff, as proportions of the whole.
    ///
    /// Returns
    /// -------
    /// Table
    ///     ``cutoff``; ``tonnage``, the proportion above cutoff; ``mean_grade``
    ///     above cutoff, null when nothing is above; ``metal``, ``tonnage × mean_grade``; and ``benefit``,
    ///     ``metal - cutoff × tonnage``.
    fn grade_tonnage(&self, cutoffs: &Bound<PyAny>) -> PyResult<Table> {
        let cutoffs = finite(cutoffs, "cutoffs")?;
        recoveries(&transforms::grade_tonnage(self.fitted()?, &cutoffs))
    }
}

/// Box-Cox power transform; `lambda_=None` picks the least-skewed lambda.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "BoxCox")]
pub struct BoxCox {
    requested: Option<f64>,
    lambda: Option<f64>,
}

#[pymethods]
impl BoxCox {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, lambda_=None))]
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

    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        Self::fit(slf, values)?.transform(py, values)
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
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "PPMT")]
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
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    /// `marginal` normal-scores each variable before the projections.
    #[new]
    #[pyo3(signature = (*, iterations=30, candidates=60, seed=0, marginal=true))]
    fn new(iterations: usize, candidates: usize, seed: u64, marginal: bool) -> Self {
        Self {
            params: PpmtParams {
                iterations,
                candidates,
                seed,
                marginal,
            },
            fitted: None,
        }
    }

    /// `data` is `(n, d)`; `weights` (e.g. declustering) shape the marginal scores.
    #[pyo3(signature = (data, *, weights=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = rows(data, "data")?;
        let weights = optional_finite(weights, "weights")?;
        slf.fitted = Some(CorePpmt::fit(&data, weights.as_deref(), &slf.params).map_err(err)?);
        Ok(slf)
    }

    #[pyo3(signature = (data, *, weights=None))]
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        data: &Bound<'py, PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        Self::fit(slf, data, weights)?.transform(py, data)
    }

    fn transform<'py>(&self, py: Python<'py>, data: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let fitted = self.fitted()?;
        Ok(array2(py, &fitted.forward(&table(data, fitted.dim())?)).into_any())
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fitted = self.fitted()?;
        Ok(array2(py, &fitted.back(&table(data, fitted.dim())?)).into_any())
    }
}

/// Imputes missing variables from the variables present in the same sample
/// and, with `spatial`, from nearby samples.
///
/// Each variable is normal-scored on its observed values; the mean and
/// covariance of the scores are fitted by expectation-maximization, missing
/// entries included. Each missing score is drawn from its Gaussian
/// distribution given the scores present in its row and back-transformed, so
/// the imputed values keep the histograms and correlations of the data.
///
/// With `spatial`, the scores follow an intrinsic model: every variable and
/// pair of variables shares the correlogram of `spatial`, scaled by the score
/// covariance (of the row's component, with a mixture). Rows are visited
/// along a random path and their missing scores drawn by simple cokriging
/// from the scores in the row and in the nearest samples, imputed ones
/// included.
///
/// Parameters
/// ----------
/// components : int or None, default 1
///     Gaussians mixed in the scores, e.g. 2 for two mineral associations
///     that one Gaussian cannot represent; None picks 1 to 6 by BIC. A
///     missing score is then drawn from a component picked by its
///     probability given the scores present in the row.
/// seed : int, default 0
///     Seed of the mixture and of the draws; the same seed imputes the same
///     values.
/// spatial : Variogram, optional
///     Normal-score variogram shared by all variables; its sill is rescaled
///     to one. A pure nugget imputes as without `spatial`.
/// neighbors : int, default 16
///     Nearest samples, in the variogram's anisotropic distance, that each
///     draw conditions on; those beyond the range are left out.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "GaussianImputer")]
pub struct GaussianImputer {
    components: Option<usize>,
    seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spatial: Option<variogram::Variogram>,
    #[serde(default = "default_neighbors")]
    neighbors: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    coords: Option<Vec<Point>>,
    fitted: Option<transforms::GaussianImputer>,
}

fn default_neighbors() -> usize {
    16
}

impl GaussianImputer {
    fn fitted(&self) -> PyResult<&transforms::GaussianImputer> {
        self.fitted
            .as_ref()
            .ok_or_else(|| not_fitted("GaussianImputer"))
    }

    /// An imputer fitted to `data`, with the settings of this one.
    pub(crate) fn fit_like(
        &self,
        data: &[Vec<f64>],
        weights: Option<&[f64]>,
    ) -> PyResult<transforms::GaussianImputer> {
        let imputer =
            transforms::GaussianImputer::fit_mixture(data, weights, self.components, self.seed)
                .map_err(err)?;
        match &self.spatial {
            None => Ok(imputer),
            Some(v) => imputer.with_spatial(v.clone(), self.neighbors).map_err(err),
        }
    }
}

#[pymethods]
impl GaussianImputer {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, components=Some(1), seed=0, spatial=None, neighbors=16))]
    fn new(
        components: Option<usize>,
        seed: u64,
        spatial: Option<PyRef<crate::variogram::Variogram>>,
        neighbors: usize,
    ) -> PyResult<Self> {
        if neighbors == 0 {
            return Err(invalid("neighbors must be at least 1"));
        }
        let spatial = spatial.map(|v| v.0.clone());
        if let Some(v) = &spatial
            && (!v.is_stationary() || !(v.total_sill() > 0.0 && v.total_sill().is_finite()))
        {
            return Err(invalid(
                "spatial needs a variogram with a finite, positive sill",
            ));
        }
        Ok(Self {
            components,
            seed,
            spatial,
            neighbors,
            coords: None,
            fitted: None,
        })
    }

    /// Fits the normal scores and their covariance.
    ///
    /// Parameters
    /// ----------
    /// data : array_like, shape (n, variables)
    ///     NaN marks a missing variable; each variable needs two values.
    /// coords : array_like, shape (n, 2) or (n, 3), or PointSet, optional
    ///     Sample locations, needed with `spatial`; `transform` uses them when
    ///     given none.
    /// weights : array_like, optional
    ///     Declustering weights, for the scores and the covariance.
    #[pyo3(signature = (data, *, coords=None, weights=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<PyAny>,
        coords: Option<&Bound<PyAny>>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = rows(data, "data")?;
        let coords = coords.map(points).transpose()?;
        match &coords {
            Some(c) => same_length(data.len(), c.len(), "coords")?,
            None if slf.spatial.is_some() => {
                return Err(invalid("a spatial GaussianImputer needs coords"));
            }
            None => {}
        }
        let weights = optional_finite(weights, "weights")?;
        slf.fitted = Some(slf.fit_like(&data, weights.as_deref())?);
        slf.coords = coords.filter(|_| slf.spatial.is_some());
        Ok(slf)
    }

    #[pyo3(signature = (data, *, coords=None, weights=None))]
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        data: &Bound<'py, PyAny>,
        coords: Option<&Bound<PyAny>>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        Self::fit(slf, data, coords, weights)?.transform(py, data, None)
    }

    /// `data` with every NaN replaced by an imputed value; the other values
    /// are returned unchanged.
    ///
    /// Parameters
    /// ----------
    /// data : array_like, shape (n, variables)
    /// coords : array_like, shape (n, 2) or (n, 3), or PointSet, optional
    ///     Locations of `data`, used with `spatial`; defaults to those given
    ///     to `fit`.
    #[pyo3(signature = (data, *, coords=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<PyAny>,
        coords: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let data = rows(data, "data")?;
        let given = coords.map(points).transpose()?;
        let coords = given
            .as_deref()
            .or(self.coords.as_deref())
            .filter(|_| self.spatial.is_some());
        if let Some(c) = coords {
            same_length(data.len(), c.len(), "coords")?;
        }
        let out = self
            .fitted()?
            .impute(&data, coords, self.seed)
            .map_err(err)?;
        Ok(array2(py, &out).into_any())
    }

    /// Correlation matrix of the normal scores.
    #[getter]
    fn correlation_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.fitted()?.correlation();
        let rows: Vec<Vec<f64>> = c.row_iter().map(|r| r.iter().copied().collect()).collect();
        Ok(array2(py, &rows).into_any())
    }
}

/// Rows of a 1-D (one variable) or 2-D array-like.
fn variables(data: &Bound<PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let array = data
        .py()
        .import("numpy")?
        .call_method1("asarray", (data, "float64"))?;
    match array.getattr("ndim")?.extract::<usize>()? {
        1 => Ok(floats(&array, "data")?
            .into_iter()
            .map(|v| vec![v])
            .collect()),
        _ => rows(&array, "data"),
    }
}

/// Weighted Gaussian kernel density of one variable: a smooth reference
/// distribution for `NormalScore` when the data are too few or too clustered
/// to define the tails.
///
/// Parameters
/// ----------
/// bandwidth : {"silverman", "scott"} or float, optional
///     Kernel width: Silverman's rule ``0.9 min(sd, IQR / 1.34) n^-1/5`` (default),
///     robust to skewed and bimodal data, Scott's rule ``1.06 sd n^-1/5``, or
///     a width, in log units when `log`. `n` is the effective number of
///     samples of the weights.
/// lower, upper : float, optional
///     Bounds of the values, kept by reflecting the kernels at them, e.g.
///     ``lower=0`` for grades.
/// log : bool, default False
///     Kernels on ``ln x``: positive values with a lognormal-like upper tail.
///
/// Attributes
/// ----------
/// bandwidth_ : float
///     The fitted kernel width.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "KernelDensity")]
pub struct KernelDensity {
    bandwidth: transforms::Bandwidth,
    lower: Option<f64>,
    upper: Option<f64>,
    log: bool,
    fitted: Option<transforms::KernelDensity>,
}

impl KernelDensity {
    fn fitted(&self) -> PyResult<&transforms::KernelDensity> {
        self.fitted
            .as_ref()
            .ok_or_else(|| not_fitted("KernelDensity"))
    }
}

#[pymethods]
impl KernelDensity {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, bandwidth=None, lower=None, upper=None, log=false))]
    fn new(
        bandwidth: Option<&Bound<PyAny>>,
        lower: Option<f64>,
        upper: Option<f64>,
        log: bool,
    ) -> PyResult<Self> {
        use transforms::Bandwidth;
        let bandwidth = match bandwidth {
            None => Bandwidth::Silverman,
            Some(b) => match (b.extract::<f64>(), b.extract::<String>().as_deref()) {
                (Ok(h), _) if h > 0.0 && h.is_finite() => Bandwidth::Given(h),
                (_, Ok("silverman")) => Bandwidth::Silverman,
                (_, Ok("scott")) => Bandwidth::Scott,
                _ => {
                    return Err(invalid(
                        "bandwidth must be 'silverman', 'scott' or a positive width",
                    ));
                }
            },
        };
        Ok(Self {
            bandwidth,
            lower,
            upper,
            log,
            fitted: None,
        })
    }

    /// Fits the density to `values`.
    ///
    /// Parameters
    /// ----------
    /// values : array_like or str
    ///     Finite values, or the column of `data` holding them.
    /// weights : array_like or str, optional
    ///     Declustering weights, or their column.
    /// data : PointSet, Table or mapping, optional
    ///     Holds the columns named by the other arguments.
    #[pyo3(signature = (values, *, weights=None, data=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        values: &Bound<'py, PyAny>,
        weights: Option<&Bound<'py, PyAny>>,
        data: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let values = finite(&column(data, values, "values")?, "values")?;
        let weights = match weights {
            Some(w) => Some(per_row(data, w, values.len(), "weights")?),
            None => None,
        };
        slf.fitted = Some(
            transforms::KernelDensity::fit(
                &values,
                weights.as_deref(),
                slf.bandwidth,
                slf.lower,
                slf.upper,
                slf.log,
            )
            .map_err(err)?,
        );
        Ok(slf)
    }

    /// Density at each of `x`, zero outside the bounds.
    fn pdf<'py>(&self, py: Python<'py>, x: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let kde = self.fitted()?;
        Ok(map(py, finite(x, "x")?, |v| kde.pdf(v)))
    }

    /// Probability of a value at most each of `x`.
    fn cdf<'py>(&self, py: Python<'py>, x: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let kde = self.fitted()?;
        Ok(map(py, finite(x, "x")?, |v| kde.cdf(v)))
    }

    /// Value at each cumulative probability of `p`, in (0, 1).
    fn quantile<'py>(&self, py: Python<'py>, p: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        use transforms::Reference;
        let kde = self.fitted()?;
        let p = finite(p, "p")?;
        if p.iter().any(|&q| q <= 0.0 || q >= 1.0) {
            return Err(invalid("p must be in (0, 1)"));
        }
        Ok(map(py, p, |q| kde.quantile(q)))
    }

    /// `n` values drawn from the density; the same `seed` gives the same
    /// values.
    #[pyo3(signature = (n, *, seed=0))]
    fn sample<'py>(&self, py: Python<'py>, n: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
        Ok(array1(py, self.fitted()?.sample(n, seed)).into_any())
    }

    #[getter]
    fn bandwidth_(&self) -> PyResult<f64> {
        Ok(self.fitted()?.bandwidth())
    }
}

/// Mixture of Gaussians fitted by expectation-maximization.
///
/// A smooth reference for one variable (`NormalScore(reference=...)`) or for
/// several, e.g. two mineral associations that one Gaussian cannot represent.
/// Rows may miss variables (NaN): each weighs the components by its present
/// entries. The components start from a k-means++ clustering drawn from
/// `seed`, so a seed gives one fit.
///
/// Parameters
/// ----------
/// components : int, optional
///     Number of Gaussians; by default the count from 1 to `max_components`
///     with the lowest BIC.
/// max_components : int, default 6
/// seed : int, default 0
///
/// Attributes
/// ----------
/// proportions_ : ndarray, shape (k,)
/// means_ : ndarray, shape (k, d)
/// covariances_ : ndarray, shape (k, d, d)
/// bic_ : dict
///     Bayesian information criterion by number of components fitted.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "GaussianMixture")]
pub struct GaussianMixture {
    components: Option<usize>,
    max_components: usize,
    seed: u64,
    fitted: Option<(transforms::GaussianMixture, Vec<(usize, f64)>)>,
}

impl GaussianMixture {
    fn fitted(&self) -> PyResult<&transforms::GaussianMixture> {
        self.fitted
            .as_ref()
            .map(|f| &f.0)
            .ok_or_else(|| not_fitted("GaussianMixture"))
    }
}

#[pymethods]
impl GaussianMixture {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, components=None, max_components=6, seed=0))]
    fn new(components: Option<usize>, max_components: usize, seed: u64) -> Self {
        Self {
            components,
            max_components,
            seed,
            fitted: None,
        }
    }

    /// Fits the mixture.
    ///
    /// Parameters
    /// ----------
    /// data : array_like, shape (n,) or (n, d)
    ///     One variable or `d`; NaN marks a missing entry.
    /// weights : array_like, optional
    ///     Declustering weights of the rows.
    #[pyo3(signature = (data, *, weights=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = variables(data)?;
        let weights = optional_finite(weights, "weights")?;
        let w = weights.as_deref();
        slf.fitted = Some(match slf.components {
            Some(k) => {
                let gm = transforms::GaussianMixture::fit(&data, w, k, slf.seed).map_err(err)?;
                let bic = vec![(k, gm.bic)];
                (gm, bic)
            }
            None => {
                let (gm, bics) =
                    transforms::GaussianMixture::select(&data, w, slf.max_components, slf.seed)
                        .map_err(err)?;
                (
                    gm,
                    bics.into_iter()
                        .enumerate()
                        .map(|(i, b)| (i + 1, b))
                        .collect(),
                )
            }
        });
        Ok(slf)
    }

    /// Mixture density at each complete row of `data`.
    fn pdf<'py>(&self, py: Python<'py>, data: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let gm = self.fitted()?;
        let pdf = variables(data)?
            .iter()
            .map(|r| gm.pdf(r))
            .collect::<Result<_, _>>()
            .map_err(err)?;
        Ok(array1(py, pdf).into_any())
    }

    /// Most probable component of each row given its present entries.
    fn predict<'py>(&self, py: Python<'py>, data: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let gm = self.fitted()?;
        let labels = variables(data)?
            .iter()
            .map(|r| {
                let p = gm.posterior(r)?;
                Ok((0..p.len())
                    .max_by(|&a, &b| p[a].total_cmp(&p[b]))
                    .unwrap_or(0) as i64)
            })
            .collect::<Result<Vec<i64>, transforms::TransformError>>()
            .map_err(err)?;
        Ok(numpy::PyArray1::from_vec(py, labels).into_any())
    }

    /// `n` rows drawn from the mixture, shape (n, d); the same `seed` gives
    /// the same rows.
    #[pyo3(signature = (n, *, seed=0))]
    fn sample<'py>(&self, py: Python<'py>, n: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
        Ok(array2(py, &self.fitted()?.sample(n, seed)).into_any())
    }

    #[getter]
    fn proportions_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(array1(py, self.fitted()?.proportions.clone()).into_any())
    }

    #[getter]
    fn means_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rows: Vec<Vec<f64>> = self
            .fitted()?
            .means
            .iter()
            .map(|m| m.iter().copied().collect())
            .collect();
        Ok(array2(py, &rows).into_any())
    }

    #[getter]
    fn covariances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let gm = self.fitted()?;
        let (k, d) = (gm.covariances.len(), gm.dim());
        let flat = gm
            .covariances
            .iter()
            .flat_map(|c| c.transpose().iter().copied().collect::<Vec<_>>())
            .collect();
        array1(py, flat).call_method1("reshape", ((k, d, d),))
    }

    #[getter]
    fn bic_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (_, bics) = self
            .fitted
            .as_ref()
            .ok_or_else(|| not_fitted("GaussianMixture"))?;
        let d = pyo3::types::PyDict::new(py);
        for (k, b) in bics {
            d.set_item(k, b)?;
        }
        Ok(d.into_any())
    }
}

/// Finite `(n, dim)` rows.
fn table(data: &Bound<PyAny>, dim: usize) -> PyResult<Vec<Vec<f64>>> {
    let data = rows(data, "data")?;
    if data.iter().any(|r| r.len() != dim) {
        return Err(invalid(format!("data must have {dim} columns")));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("data must be finite"));
    }
    Ok(data)
}

/// Principal components of the covariance (`standardize=True`: correlation)
/// matrix, by decreasing variance.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "PCA")]
pub struct Pca {
    standardize: bool,
    fitted: Option<(CorePca, usize)>,
}

impl Pca {
    fn fitted(&self) -> PyResult<&(CorePca, usize)> {
        self.fitted.as_ref().ok_or_else(|| not_fitted("PCA"))
    }
}

#[pymethods]
impl Pca {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, standardize=false))]
    fn new(standardize: bool) -> Self {
        Self {
            standardize,
            fitted: None,
        }
    }

    /// `data` is `(n, d)`; optional `weights`, e.g. declustering.
    #[pyo3(signature = (data, *, weights=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = rows(data, "data")?;
        let weights = optional_finite(weights, "weights")?;
        let pca = CorePca::fit(&data, weights.as_deref(), slf.standardize).map_err(err)?;
        slf.fitted = Some((pca, data[0].len()));
        Ok(slf)
    }

    #[pyo3(signature = (data, *, weights=None))]
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        data: &Bound<'py, PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        Self::fit(slf, data, weights)?.transform(py, data)
    }

    fn transform<'py>(&self, py: Python<'py>, data: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let (pca, dim) = self.fitted()?;
        Ok(array2(py, &pca.forward(&table(data, *dim)?)).into_any())
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (pca, dim) = self.fitted()?;
        Ok(array2(py, &pca.back(&table(data, *dim)?)).into_any())
    }

    /// Unit eigenvectors as rows.
    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (pca, _) = self.fitted()?;
        Ok(array2(py, &pca.components()).into_any())
    }

    /// Variance of each component (eigenvalues).
    #[getter]
    fn explained_variance_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (pca, _) = self.fitted()?;
        Ok(array1(py, pca.eigenvalues().to_vec()).into_any())
    }

    #[getter]
    fn explained_variance_ratio_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (pca, _) = self.fitted()?;
        let total: f64 = pca.eigenvalues().iter().sum();
        Ok(map(py, pca.eigenvalues().to_vec(), |l| l / total))
    }
}

/// Min/max autocorrelation factors: uncorrelated at lag 0 and at `lag`, from
/// most to least continuous. Pairs within `lag ± tolerance` (default `lag / 2`).
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "MAF")]
pub struct Maf {
    lag: f64,
    tolerance: Option<f64>,
    fitted: Option<(CoreMaf, usize)>,
}

impl Maf {
    fn fitted(&self) -> PyResult<&(CoreMaf, usize)> {
        self.fitted.as_ref().ok_or_else(|| not_fitted("MAF"))
    }
}

#[pymethods]
impl Maf {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (lag, *, tolerance=None))]
    fn new(lag: f64, tolerance: Option<f64>) -> Self {
        Self {
            lag,
            tolerance,
            fitted: None,
        }
    }

    /// `data` is `(n, d)` at `coords` `(n, 2 | 3)`.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<PyAny>,
        coords: &Bound<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (data, locs) = (rows(data, "data")?, points(coords)?);
        same_length(data.len(), locs.len(), "coords")?;
        let tolerance = slf.tolerance.unwrap_or(slf.lag / 2.0);
        let maf = CoreMaf::fit(&data, &locs, slf.lag, tolerance).map_err(err)?;
        slf.fitted = Some((maf, data[0].len()));
        Ok(slf)
    }

    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        data: &Bound<'py, PyAny>,
        coords: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        Self::fit(slf, data, coords)?.transform(py, data)
    }

    fn transform<'py>(&self, py: Python<'py>, data: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let (maf, dim) = self.fitted()?;
        Ok(array2(py, &maf.forward(&table(data, *dim)?)).into_any())
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (maf, dim) = self.fitted()?;
        Ok(array2(py, &maf.back(&table(data, *dim)?)).into_any())
    }

    /// Semivariogram of each factor at `lag`, increasing.
    #[getter]
    fn gammas_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (maf, _) = self.fitted()?;
        Ok(array1(py, maf.gammas().to_vec()).into_any())
    }
}

/// Stepwise conditional transform: each variable normal-scored within the
/// `classes` equal-probability classes of the variables before it.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "StepwiseConditional")]
pub struct StepwiseConditional {
    classes: usize,
    fitted: Option<CoreSct>,
}

impl StepwiseConditional {
    fn fitted(&self) -> PyResult<&CoreSct> {
        self.fitted
            .as_ref()
            .ok_or_else(|| not_fitted("StepwiseConditional"))
    }
}

#[pymethods]
impl StepwiseConditional {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, classes=10))]
    fn new(classes: usize) -> Self {
        Self {
            classes,
            fitted: None,
        }
    }

    /// Fits the class tables; tied values share a class.
    ///
    /// Parameters
    /// ----------
    /// data : array_like, shape (n, d)
    ///     Column order sets the conditioning order.
    /// weights : array_like, optional
    ///     Declustering weights, for every normal score.
    #[pyo3(signature = (data, *, weights=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = rows(data, "data")?;
        let weights = optional_finite(weights, "weights")?;
        slf.fitted = Some(CoreSct::fit(&data, weights.as_deref(), slf.classes).map_err(err)?);
        Ok(slf)
    }

    #[pyo3(signature = (data, *, weights=None))]
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        data: &Bound<'py, PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        Self::fit(slf, data, weights)?.transform(py, data)
    }

    fn transform<'py>(&self, py: Python<'py>, data: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let fitted = self.fitted()?;
        Ok(array2(py, &fitted.forward(&table(data, fitted.dim())?)).into_any())
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fitted = self.fitted()?;
        Ok(array2(py, &fitted.back(&table(data, fitted.dim())?)).into_any())
    }
}

/// Uniform conditioning of panel estimates to recoveries of the selective
/// blocks (SMUs) inside them.
///
/// Parameters
/// ----------
/// anamorphosis : HermiteAnamorphosis
///     Fitted point anamorphosis.
/// r_smu : float
///     Change-of-support coefficient of the selective blocks, in (0, 1].
/// r_panel : float, optional
///     Change-of-support coefficient of the panel estimates, in (0, `r_smu`].
///     When omitted, each panel's comes from its own kriging: the panel
///     methods then take every panel's estimate variance
///     ``C(V, V) - variance - 2 * lagrange`` (``estimate_variance`` in the
///     `predict` diagnostics), which carries the smoothing of each estimate.
///     The kriging variogram's sill should then be the anamorphosis variance.
///     A panel estimate more variable than the selective blocks takes
///     ``r_panel = r_smu``.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "UniformConditioning", frozen)]
pub struct UniformConditioning(CoreUc);

#[pymethods]
impl UniformConditioning {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (anamorphosis, r_smu, *, r_panel=None))]
    fn new(anamorphosis: PyRef<Anamorphosis>, r_smu: f64, r_panel: Option<f64>) -> PyResult<Self> {
        let point = anamorphosis.inner()?;
        Ok(Self(
            match r_panel {
                Some(r_panel) => CoreUc::new(point, r_smu, r_panel),
                None => CoreUc::per_panel(point, r_smu),
            }
            .map_err(err)?,
        ))
    }

    /// Recoveries of the selective blocks within one panel.
    ///
    /// Parameters
    /// ----------
    /// panel_grade : float
    ///     Estimated panel grade, clamped to the anamorphosis range.
    /// cutoffs : array_like
    /// estimate_variance : float, optional
    ///     The panel's estimate variance; required without `r_panel`, refused
    ///     with it.
    ///
    /// Returns
    /// -------
    /// Table
    ///     As `HermiteAnamorphosis.grade_tonnage`, in proportions of the panel.
    #[pyo3(signature = (panel_grade, cutoffs, *, estimate_variance=None))]
    fn panel_recovery(
        &self,
        panel_grade: f64,
        cutoffs: &Bound<PyAny>,
        estimate_variance: Option<f64>,
    ) -> PyResult<Table> {
        let cutoffs = finite(cutoffs, "cutoffs")?;
        let r = self
            .0
            .panel_recovery(panel_grade, estimate_variance, &cutoffs)
            .map_err(err)?;
        recoveries(&r)
    }

    /// Grades of the selective blocks of one panel, ascending.
    ///
    /// Parameters
    /// ----------
    /// panel_grade : float
    /// n_smu : int
    ///     Number of selective blocks in the panel.
    /// estimate_variance : float, optional
    ///     As in `panel_recovery`.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     The means of `n_smu` equal-probability bands of the panel's
    ///     selective-block distribution. They average to the panel grade, and
    ///     the top ``k`` recover the panel's metal at tonnage ``k / n_smu``.
    #[pyo3(signature = (panel_grade, n_smu, *, estimate_variance=None))]
    fn localized_grades<'py>(
        &self,
        py: Python<'py>,
        panel_grade: f64,
        n_smu: usize,
        estimate_variance: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let g = self
            .0
            .band_means(panel_grade, estimate_variance, n_smu)
            .map_err(err)?;
        Ok(array1(py, g).into_any())
    }

    /// Global grade-tonnage curve of the panels.
    ///
    /// Parameters
    /// ----------
    /// panels : BlockModel
    /// grade : str
    ///     Column of estimated panel grades; null panels are left out.
    /// cutoffs : array_like
    /// estimate_variance : str, optional
    ///     Column of the panels' estimate variances; required without
    ///     `r_panel`, refused with it.
    /// density : float, array_like or str, default 1.0
    ///     Density of each panel, or the column of `panels` holding it.
    ///
    /// Returns
    /// -------
    /// Table
    ///     ``cutoff``, ``tonnage`` (volume × density above cutoff, summed over
    ///     the panels), ``mean_grade`` (null when nothing is above),
    ///     ``metal`` (``tonnage × mean_grade``)
    ///     and ``benefit``.
    #[pyo3(signature = (panels, grade, cutoffs, *, estimate_variance=None, density=None))]
    fn grade_tonnage(
        &self,
        py: Python,
        panels: &Bound<PyBlockModel>,
        grade: &str,
        cutoffs: &Bound<PyAny>,
        estimate_variance: Option<&str>,
        density: Option<&Bound<PyAny>>,
    ) -> PyResult<Table> {
        let cutoffs = finite(cutoffs, "cutoffs")?;
        let model = panels.borrow();
        let (grade, variance) = panel_columns(&model, grade, estimate_variance)?;
        let density = match density {
            Some(d) => per_row(Some(panels.as_any()), d, grade.len(), "density")?,
            None => vec![1.0; grade.len()],
        };
        let curves = py
            .detach(|| self.0.grade_tonnage(&grade, variance.as_deref(), &cutoffs))
            .map_err(err)?;
        let mut sums = vec![(0.0, 0.0, 0.0); cutoffs.len()];
        for ((curve, volume), density) in curves.iter().zip(model.0.volumes()).zip(density) {
            for (s, r) in sums.iter_mut().zip(curve.iter().flatten()) {
                let w = volume * density;
                *s = (s.0 + w * r.tonnage, s.1 + w * r.metal, s.2 + w * r.benefit);
            }
        }
        let global: Vec<Recovery> = cutoffs
            .iter()
            .zip(sums)
            .map(|(&cutoff, (tonnage, metal, benefit))| Recovery {
                cutoff,
                tonnage,
                metal,
                mean_grade: metal / tonnage,
                benefit,
            })
            .collect();
        recoveries(&global)
    }

    /// Localized grades of the selective blocks nested in the panels.
    ///
    /// A panel holding ``n`` selective blocks splits its selective-block
    /// distribution into ``n`` equal-probability bands, and its block ranked
    /// ``i`` gets the mean of band ``i``: the blocks average to the panel
    /// grade and reproduce the panel's grade-tonnage curve at tonnages
    /// ``k / n``. Partial panels localize over the blocks present.
    ///
    /// Parameters
    /// ----------
    /// smus : BlockModel
    ///     Selective blocks nesting in the panels: same rotation, sizes
    ///     dividing the panel sizes, grids aligned; not sub-blocked.
    /// ranking : str
    ///     Column of `smus` ordering the blocks within a panel, such as a
    ///     direct kriging of the blocks; ties follow row order.
    /// panels : BlockModel
    /// grade : str
    ///     Column of estimated panel grades.
    /// estimate_variance : str, optional
    ///     Column of the panels' estimate variances; required without
    ///     `r_panel`, refused with it.
    /// name : str, default "localized"
    ///     Name of the new column.
    ///
    /// Returns
    /// -------
    /// BlockModel
    ///     `smus` with the localized grades; null in null panels and outside
    ///     every panel.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     If the blocks do not nest, or a block of an estimated panel has a
    ///     null rank.
    #[pyo3(signature = (smus, ranking, panels, grade, *, estimate_variance=None, name="localized"))]
    #[allow(clippy::too_many_arguments)]
    fn localize(
        &self,
        py: Python,
        smus: PyRef<PyBlockModel>,
        ranking: &str,
        panels: PyRef<PyBlockModel>,
        grade: &str,
        estimate_variance: Option<&str>,
        name: &str,
    ) -> PyResult<PyBlockModel> {
        let (values, variance) = panel_columns(&panels, grade, estimate_variance)?;
        let rank = nullable(&smus, ranking)?;
        let (panel_model, smu_model) = (&panels.0, &smus.0);
        let out = py
            .detach(|| {
                self.0
                    .localize(panel_model, &values, variance.as_deref(), smu_model, &rank)
            })
            .map_err(err)?;
        let column: Float64Array = out.into_iter().collect();
        Ok(PyBlockModel(
            smus.0
                .with_column(name, Arc::new(column))
                .map_err(invalid)?,
        ))
    }
}

/// A float column of `model`, null as None.
pub(crate) fn nullable(model: &PyBlockModel, name: &str) -> PyResult<Vec<Option<f64>>> {
    let column =
        model.0.attributes().column_by_name(name).ok_or_else(|| {
            crate::table::missing(name, crate::table::names(model.0.attributes()))
        })?;
    let values = arrow_cast::cast(column, &arrow_schema::DataType::Float64).map_err(invalid)?;
    Ok(values.as_primitive::<Float64Type>().iter().collect())
}

/// Panel grades and, if named, estimate variances.
type PanelColumns = (Vec<Option<f64>>, Option<Vec<Option<f64>>>);

fn panel_columns(
    panels: &PyBlockModel,
    grade: &str,
    estimate_variance: Option<&str>,
) -> PyResult<PanelColumns> {
    Ok((
        nullable(panels, grade)?,
        estimate_variance.map(|v| nullable(panels, v)).transpose()?,
    ))
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum TrendModel {
    Polynomial(CoreTrend),
    Kernel {
        kernel: KernelTrend,
        categories: Option<Vec<String>>,
    },
}

/// A trend in the coordinates, from `detrend`: a polynomial, or a smooth
/// kernel average of the samples.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "Trend", frozen)]
pub struct Trend(TrendModel);

impl Trend {
    fn kernel(&self) -> Option<&KernelTrend> {
        match &self.0 {
            TrendModel::Kernel { kernel, .. } => Some(kernel),
            TrendModel::Polynomial(_) => None,
        }
    }
}

/// One column per category, or an array, of per-row values; null where None.
fn trend_output<'py>(
    py: Python<'py>,
    rows: Vec<Option<Vec<f64>>>,
    categories: Option<&[String]>,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(names) = categories else {
        let values = rows
            .into_iter()
            .map(|r| r.map_or(f64::NAN, |r| r[0]))
            .collect();
        return Ok(array1(py, values).into_any());
    };
    let columns = names.iter().enumerate().map(|(c, name)| {
        let column: Float64Array = rows.iter().map(|r| r.as_ref().map(|r| r[c])).collect();
        (name.clone(), Arc::new(column) as ArrayRef)
    });
    let batch = RecordBatch::try_from_iter(columns).map_err(invalid)?;
    Table(batch).into_bound_py_any(py)
}

#[pymethods]
impl Trend {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    /// Polynomial degree; None for a kernel trend.
    #[getter]
    fn degree(&self) -> Option<usize> {
        match &self.0 {
            TrendModel::Polynomial(t) => Some(t.degree),
            TrendModel::Kernel { .. } => None,
        }
    }

    /// Polynomial coefficients (1, x, y, z, x², y², z², xy, xz, yz up to the
    /// degree); None for a kernel trend.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        match &self.0 {
            TrendModel::Polynomial(t) => Some(array1(py, t.coeffs.clone()).into_any()),
            TrendModel::Kernel { .. } => None,
        }
    }

    /// Kernel bandwidth in use; None for a polynomial.
    #[getter]
    fn bandwidth(&self) -> Option<f64> {
        self.kernel().map(|k| k.bandwidth)
    }

    /// Candidate bandwidths; None for a polynomial.
    #[getter]
    fn bandwidths<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.kernel()
            .map(|k| array1(py, k.bandwidths.clone()).into_any())
    }

    /// Leave-one-out error of each candidate bandwidth; None for a polynomial.
    #[getter]
    fn scores<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.kernel()
            .map(|k| array1(py, k.scores.clone()).into_any())
    }

    /// Category names of a categorical trend, else None.
    #[getter]
    fn categories(&self) -> Option<Vec<String>> {
        match &self.0 {
            TrendModel::Kernel { categories, .. } => categories.clone(),
            TrendModel::Polynomial(_) => None,
        }
    }

    /// The trend at `coords` (array, PointSet or BlockModel cells): an array,
    /// or for categories a Table of proportions, one column each. A kernel
    /// trend is null (NaN) beyond four bandwidths of every sample; fill it
    /// (e.g. with the data mean) where a simulation needs a trend everywhere.
    fn predict<'py>(&self, py: Python<'py>, coords: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let targets = points(coords)?;
        match &self.0 {
            TrendModel::Polynomial(t) => {
                Ok(array1(py, targets.iter().map(|p| t.eval(p)).collect()).into_any())
            }
            TrendModel::Kernel { kernel, categories } => {
                let rows = py.detach(|| kernel.eval(&targets)).map_err(err)?;
                trend_output(py, rows, categories.as_deref())
            }
        }
    }
}

/// Locations of `coords` and finite `values`, or the column of `coords` they name.
fn samples(coords: &Bound<PyAny>, values: &Bound<PyAny>) -> PyResult<(Vec<Point>, Vec<f64>)> {
    let locs = points(coords)?;
    let values = finite(&column(Some(coords), values, "values")?, "values")?;
    same_length(locs.len(), values.len(), "values")?;
    Ok((locs, values))
}

/// Fits a trend and returns it with the residuals, data minus trend.
///
/// Without `bandwidth`, the trend is a polynomial of `degree` in the
/// coordinates, fitted by least squares. With it, the trend is smooth: at each
/// location, the weighted average of the samples under an anisotropic Gaussian
/// kernel whose standard deviation is `bandwidth` along the major axis and
/// `ratios` times it along the semi-major and minor axes. Given several
/// bandwidths, the one with the least leave-one-out error is kept: the
/// weighted mean squared difference between each sample and the trend of the
/// others, a sample with no other within four bandwidths counting against the
/// global mean. Noisier data thus get a smoother trend.
///
/// Parameters
/// ----------
/// coords : array_like, PointSet or BlockModel
///     Sample locations, shape (n, 2) or (n, 3), or a container.
/// values : array_like or str
///     Sample values, or the column of `coords` holding them. With
///     `categorical`, category labels.
/// degree : int, default 1
///     Polynomial degree, 0 to 2, without `bandwidth`.
/// bandwidth : float or sequence of float, optional
///     Kernel standard deviation along the major axis, or candidates to
///     choose from.
/// rotation : tuple of float, default (0, 0, 0)
///     Azimuth, dip and rake of the kernel's major axis, in degrees.
/// ratios : tuple of float, default (1, 1)
///     Semi-major and minor over major bandwidth.
/// weights : array_like or str, optional
///     Declustering weights, or the column of `coords` holding them.
/// categorical : bool, default False
///     Smooth the indicator of each category instead: the trend is the local
///     proportions, which lie in [0, 1] and sum to 1, one per category in the
///     order of `Categories.from_values`.
/// scheme : Categories, optional
///     With `categorical`, the categories' order and names, which encode the
///     labels.
///
/// Returns
/// -------
/// Trend
///     The fitted trend; `predict` evaluates it anywhere, e.g. at the cells of
///     a BlockModel, as `trend=` of `SGS` or `TurningBands`.
/// numpy.ndarray or Table
///     Residuals at the samples; per category, indicator minus proportion.
#[pyfunction]
#[pyo3(signature = (coords, values, *, degree=1, bandwidth=None, rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0), weights=None, categorical=false, scheme=None))]
#[allow(clippy::too_many_arguments)]
fn detrend<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    degree: usize,
    bandwidth: Option<&Bound<PyAny>>,
    rotation: (f64, f64, f64),
    ratios: (f64, f64),
    weights: Option<&Bound<PyAny>>,
    categorical: bool,
    scheme: Option<PyRef<crate::categories::Categories>>,
) -> PyResult<(Trend, Bound<'py, PyAny>)> {
    if scheme.is_some() && !categorical {
        return Err(invalid("scheme needs categorical=True"));
    }
    let Some(bandwidth) = bandwidth else {
        if weights.is_some() || categorical {
            return Err(invalid("weights and categorical need a bandwidth"));
        }
        let (locs, values) = samples(coords, values)?;
        let (trend, residuals) = transforms::detrend(&locs, &values, degree).map_err(err)?;
        return Ok((
            Trend(TrendModel::Polynomial(trend)),
            array1(py, residuals).into_any(),
        ));
    };
    let bandwidths: Vec<f64> = match bandwidth.extract::<f64>() {
        Ok(b) => vec![b],
        Err(_) => bandwidth
            .extract()
            .map_err(|_| invalid("bandwidth must be a number or a sequence of numbers"))?,
    };
    let locs = points(coords)?;
    let (rows, categories) = if categorical {
        let labels = crate::args::texts(&column(Some(coords), values, "values")?, "values")?;
        let (names, codes) = crate::categories::coded(&labels, scheme.as_deref())?;
        let codes: Vec<u32> = codes
            .into_iter()
            .collect::<Option<_>>()
            .ok_or_else(|| invalid("values must not be null; drop missing values first"))?;
        let rows: Vec<Vec<f64>> = codes
            .iter()
            .map(|&c| {
                (0..names.len())
                    .map(|n| f64::from(u8::from(n == c as usize)))
                    .collect()
            })
            .collect();
        (rows, Some(names))
    } else {
        let values = finite(&column(Some(coords), values, "values")?, "values")?;
        (values.into_iter().map(|v| vec![v]).collect(), None)
    };
    same_length(locs.len(), rows.len(), "values")?;
    let weights = weights
        .map(|w| per_row(Some(coords), w, locs.len(), "weights"))
        .transpose()?;
    let rotation = [rotation.0, rotation.1, rotation.2];
    let ratios = [ratios.0, ratios.1];
    let (kernel, fitted) = py
        .detach(|| {
            let k = KernelTrend::fit(
                &locs,
                &rows,
                weights.as_deref(),
                &bandwidths,
                rotation,
                ratios,
            )?;
            let fitted = k.eval(&locs)?;
            Ok((k, fitted))
        })
        .map_err(err)?;
    let residuals = rows
        .iter()
        .zip(fitted)
        .map(|(v, t)| t.map(|t| v.iter().zip(t).map(|(v, t)| v - t).collect()))
        .collect();
    let residuals = trend_output(py, residuals, categories.as_deref())?;
    Ok((Trend(TrendModel::Kernel { kernel, categories }), residuals))
}

/// Declustering weights (normalized to sum to n) and the declustered mean.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "Declustering", frozen)]
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
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

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

pub fn declustering(w: Weights, sizes: Vec<f64>, means: Vec<f64>) -> Declustering {
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
/// keeps the size with the lowest mean (`minimize=False`: highest). The weights
/// are averaged over the same `offsets` origins, so they sum to the number of
/// samples and `mean` equals `means` at the kept size. `values` may name a
/// column of `coords`.
#[pyfunction]
#[pyo3(signature = (coords, values, *, cell_size=None, sizes=None, offsets=25, minimize=true))]
fn cell_declustering(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    cell_size: Option<f64>,
    sizes: Option<&Bound<PyAny>>,
    offsets: usize,
    minimize: bool,
) -> PyResult<Declustering> {
    let (locs, values) = samples(coords, values)?;
    if let Some(size) = cell_size {
        let w =
            transforms::cell_weights_over_offsets(&locs, &values, size, offsets).map_err(err)?;
        return Ok(declustering(w, vec![], vec![]));
    }
    let origin = locs.iter().fold((f64::MAX, f64::MAX, f64::MAX), |m, p| {
        (m.0.min(p.0), m.1.min(p.1), m.2.min(p.2))
    });
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
    let w = transforms::cell_weights_over_offsets(&locs, &values, best, offsets).map_err(err)?;
    Ok(declustering(w, sizes, means))
}

/// Breaks ties in values before a normal-score transform.
///
/// Tied samples, such as those at a detection limit, are ranked on the average
/// over their neighborhood of each sample's rank, within each radius in turn,
/// then on a seeded random draw. Several variables share one ordering: the
/// averaged rank is the mean over the variables.
///
/// Parameters
/// ----------
/// coords : array_like, PointSet or BlockModel
///     ``(n, 2)`` or ``(n, 3)`` sample coordinates, or a container.
/// values : array_like, str or list of str
///     ``(n,)`` values, ``(n, k)`` values of k variables, or column names of
///     ``coords``.
/// radii : sequence of float, optional
///     Neighborhood radii, used smallest first. Defaults to 1, 2, 4 and 8
///     times the median distance to the nearest sample.
/// seed : int, default 0
///     Seed of the last-resort random tie-break.
///
/// Returns
/// -------
/// numpy.ndarray or Table
///     The values with each tie spread over tiny increasing offsets (at most
///     1e-4 of the gap to the next value) in rank order, so no ties remain and
///     untied values keep their order. A Table when ``values`` are names.
#[pyfunction]
#[pyo3(signature = (coords, values, *, radii=None, seed=0))]
fn despike<'py>(
    py: Python<'py>,
    coords: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    radii: Option<Vec<f64>>,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let locs = points(coords)?;
    let names = match values.is_instance_of::<pyo3::types::PyString>() {
        true => None,
        false => values.extract::<Vec<String>>().ok(),
    };
    let mut matrix = false;
    let columns = match &names {
        Some(names) => names
            .iter()
            .map(|n| finite(&crate::args::named(Some(coords), n, "values")?, "values"))
            .collect::<PyResult<Vec<_>>>()?,
        None => {
            let values = column(Some(coords), values, "values")?;
            let ndim: usize = py
                .import("numpy")?
                .call_method1("ndim", (&values,))?
                .extract()?;
            if ndim == 2 {
                matrix = true;
                let rows = rows(&values, "values")?;
                let k = rows.first().map_or(0, Vec::len);
                (0..k)
                    .map(|j| rows.iter().map(|r| r[j]).collect())
                    .collect()
            } else {
                vec![finite(&values, "values")?]
            }
        }
    };
    for c in &columns {
        same_length(locs.len(), c.len(), "values")?;
    }
    let radii = radii.unwrap_or_else(|| transforms::default_radii(&locs));
    let out = transforms::despike(&locs, &columns, &radii, seed).map_err(err)?;
    if let Some(names) = names {
        let columns = names
            .iter()
            .zip(out)
            .map(|(n, c)| (n.as_str(), Arc::new(Float64Array::from(c)) as ArrayRef));
        let table = Table(RecordBatch::try_from_iter(columns).map_err(invalid)?);
        return Ok(Bound::new(py, table)?.into_any());
    }
    if matrix {
        let rows: Vec<Vec<f64>> = (0..locs.len())
            .map(|i| out.iter().map(|c| c[i]).collect())
            .collect();
        return Ok(array2(py, &rows).into_any());
    }
    Ok(array1(py, out.into_iter().next().unwrap_or_default()).into_any())
}

/// Uncertainty in global statistics of spatially correlated data.
///
/// Each realization draws unconditional Gaussian values at the sample
/// locations with the correlation of `variogram`, turns them into uniform
/// ranks and reads them through the declustered distribution of `values`. The
/// statistics of the resampled values then spread as far as the spatial
/// correlation allows: with a pure nugget they match the classical bootstrap,
/// with ranges much longer than the data extent the samples act as one.
///
/// Parameters
/// ----------
/// coords : array_like, PointSet or BlockModel
///     ``(n, 2)`` or ``(n, 3)`` sample coordinates, or a container.
/// values : array_like or str
///     ``(n,)`` values, or their column in `coords`.
/// variogram : Variogram
///     Normal-score variogram; only its correlation (the variogram over its
///     total sill) is used.
/// weights : array_like or str, optional
///     Declustering weights, or their column; default equal.
/// n : int, default 100
///     Number of realizations.
/// seed : int, default 0
///     Seed; each realization draws from a seed mixed from it.
/// quantiles : sequence of float, optional
///     Probabilities of quantile columns ``P10``, ``P50``, ...
/// cutoffs : sequence of float, optional
///     Cutoffs of proportion columns ``above 1.5``, ...: the fraction of
///     values above each cutoff.
///
/// Returns
/// -------
/// Table
///     One row per realization: ``mean``, then the quantile and proportion
///     columns.
#[pyfunction]
#[pyo3(signature = (coords, values, variogram, *, weights=None, n=100, seed=0, quantiles=vec![], cutoffs=vec![]))]
#[allow(clippy::too_many_arguments)]
fn spatial_bootstrap(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    variogram: PyRef<crate::variogram::Variogram>,
    weights: Option<&Bound<PyAny>>,
    n: usize,
    seed: u64,
    quantiles: Vec<f64>,
    cutoffs: Vec<f64>,
) -> PyResult<Table> {
    let locs = points(coords)?;
    let values = finite(&column(Some(coords), values, "values")?, "values")?;
    same_length(locs.len(), values.len(), "values")?;
    let w = weights
        .map(|w| per_row(Some(coords), w, values.len(), "weights"))
        .transpose()?;
    let b = transforms::spatial_bootstrap(
        &locs,
        &values,
        w.as_deref(),
        &variogram.0,
        n,
        seed,
        &quantiles,
        &cutoffs,
    )
    .map_err(err)?;
    let f = |v: Vec<f64>| Arc::new(Float64Array::from(v)) as ArrayRef;
    let mut columns = vec![("mean".to_string(), f(b.mean))];
    for (j, &p) in quantiles.iter().enumerate() {
        let q = b.quantiles.iter().map(|q| q[j]).collect();
        columns.push((crate::eda::quantile_name(p), f(q)));
    }
    for (j, &c) in cutoffs.iter().enumerate() {
        let a = b.above.iter().map(|a| a[j]).collect();
        columns.push((format!("above {c}"), f(a)));
    }
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// Polygonal (nearest-neighbor area) declustering on a `nodes`-cell grid.
#[pyfunction]
#[pyo3(signature = (coords, values, *, nodes=10_000))]
fn polygon_declustering(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    nodes: usize,
) -> PyResult<Declustering> {
    let (locs, values) = samples(coords, values)?;
    let w = transforms::polygon_weights(&locs, &values, nodes).map_err(err)?;
    Ok(declustering(w, vec![], vec![]))
}

/// Point-to-block support correction keeping the mean: `f` is the variance
/// reduction factor `Var(Z_v) / Var(Z)`.
#[pyfunction]
#[pyo3(signature = (values, f, *, weights=None))]
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
#[pyo3(signature = (values, f, *, weights=None))]
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

/// Averages samples into blocks of `size`; returns centers, means and counts.
#[pyfunction]
#[pyo3(signature = (coords, values, size, *, origin=vec![0.0, 0.0, 0.0]))]
fn upscale<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    size: Vec<f64>,
    origin: Vec<f64>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (locs, values) = samples(coords, values)?;
    let blocks = transforms::upscale(
        &locs,
        &values,
        triple(size, 1.0, "size")?,
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
    size: Vec<f64>,
    value: f64,
    refine: (usize, usize, usize),
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let cells = transforms::downscale(
        triple(center, 0.0, "center")?,
        triple(size, 1.0, "size")?,
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
    m.add_class::<GaussianImputer>()?;
    m.add_class::<KernelDensity>()?;
    m.add_class::<GaussianMixture>()?;
    m.add_class::<Pca>()?;
    m.add_class::<Maf>()?;
    m.add_class::<StepwiseConditional>()?;
    m.add_class::<UniformConditioning>()?;
    m.add_class::<Trend>()?;
    m.add_class::<Declustering>()?;
    m.add_function(wrap_pyfunction!(detrend, m)?)?;
    m.add_function(wrap_pyfunction!(cell_declustering, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_declustering, m)?)?;
    m.add_function(wrap_pyfunction!(despike, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(affine_correction, m)?)?;
    m.add_function(wrap_pyfunction!(indirect_lognormal_correction, m)?)?;
    m.add_function(wrap_pyfunction!(upscale, m)?)?;
    m.add_function(wrap_pyfunction!(downscale, m)?)?;
    m.add_function(wrap_pyfunction!(normal_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(normal_ppf, m)?)?;
    Ok(())
}

/// A fresh fit of `transform` (a PCA, MAF, StepwiseConditional or PPMT) to
/// `data` at `locs`; `weights` reach the transforms that take them.
pub(crate) fn decorrelation(
    transform: &Bound<PyAny>,
    data: &[Vec<f64>],
    weights: Option<&[f64]>,
    locs: &[crate::args::Point],
) -> PyResult<simulation::Decorrelation> {
    use simulation::Decorrelation as D;
    Ok(if let Ok(t) = transform.cast::<Pca>() {
        D::Pca(CorePca::fit(data, weights, t.borrow().standardize).map_err(err)?)
    } else if let Ok(t) = transform.cast::<Maf>() {
        let t = t.borrow();
        let tolerance = t.tolerance.unwrap_or(t.lag / 2.0);
        D::Maf(CoreMaf::fit(data, locs, t.lag, tolerance).map_err(err)?)
    } else if let Ok(t) = transform.cast::<StepwiseConditional>() {
        D::Stepwise(CoreSct::fit(data, weights, t.borrow().classes).map_err(err)?)
    } else if let Ok(t) = transform.cast::<Ppmt>() {
        D::Ppmt(CorePpmt::fit(data, weights, &t.borrow().params).map_err(err)?)
    } else {
        return Err(invalid(NOT_DECORRELATION));
    })
}

pub(crate) const NOT_DECORRELATION: &str =
    "transform must be a PCA, MAF, StepwiseConditional or PPMT";

pub(crate) fn is_decorrelation(obj: &Bound<PyAny>) -> bool {
    obj.cast::<Pca>().is_ok()
        || obj.cast::<Maf>().is_ok()
        || obj.cast::<StepwiseConditional>().is_ok()
        || obj.cast::<Ppmt>().is_ok()
}
