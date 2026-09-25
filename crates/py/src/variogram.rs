use pyo3::prelude::*;
use pyo3::types::PyTuple;
use transforms::dgm::{BlockDiscretization, block_average_correlation};
use variogram::surface::{PlaneMapParams, SurfaceParams, plane_map, variogram_surface};
use variogram::{
    Angles, Anisotropy, CoregStructure, Coregionalization as CoreCoreg, Direction, Estimator,
    Experimental, LagBins, Model, Structure as CoreStructure, Transiogram as CoreTransiogram,
    Variogram as CoreVariogram, Weighting, empirical_transiogram, experimental,
};

use crate::args::{Point, array1, array2, finite, floats, points, same_length, triple};
use crate::invalid;
use crate::transforms::Anamorphosis;

fn err(e: variogram::VarioError) -> PyErr {
    invalid(e)
}

pub fn model(name: &str, order: Option<f64>, exponent: Option<f64>) -> PyResult<Model> {
    Ok(
        match name
            .to_ascii_lowercase()
            .replace(['-', '_', ' '], "")
            .as_str()
        {
            "spherical" => Model::Spherical,
            "exponential" => Model::Exponential,
            "gaussian" => Model::Gaussian,
            "cubic" => Model::Cubic,
            "pentaspherical" => Model::PentaSpherical,
            "circular" => Model::Circular,
            "sinehole" | "hole" => Model::SineHole,
            "matern" => Model::Matern {
                order: order.unwrap_or(1.5),
            },
            "power" => Model::Power {
                exponent: exponent.unwrap_or(1.0),
            },
            other => return Err(invalid(format!("unknown model {other:?}"))),
        },
    )
}

fn model_name(m: Model) -> &'static str {
    match m {
        Model::Spherical => "spherical",
        Model::Exponential => "exponential",
        Model::Gaussian => "gaussian",
        Model::Cubic => "cubic",
        Model::PentaSpherical => "pentaspherical",
        Model::Circular => "circular",
        Model::SineHole => "sinehole",
        Model::Matern { .. } => "matern",
        Model::Power { .. } => "power",
    }
}

fn estimator(name: &str) -> PyResult<Estimator> {
    match name {
        "matheron" => Ok(Estimator::Matheron),
        "cressie-hawkins" | "cressie_hawkins" => Ok(Estimator::CressieHawkins),
        _ => Err(invalid(format!("unknown estimator {name:?}"))),
    }
}

fn weighting(name: &str) -> PyResult<Weighting> {
    match name {
        "uniform" => Ok(Weighting::Uniform),
        "count" => Ok(Weighting::ByCount),
        "count/gamma" => Ok(Weighting::ByCountOverGamma),
        _ => Err(invalid(format!(
            "unknown weighting {name:?}; use uniform, count or count/gamma"
        ))),
    }
}

fn bins(lag: f64, max_lag: f64) -> PyResult<LagBins> {
    if !(lag > 0.0 && max_lag > lag) {
        return Err(invalid("need 0 < lag < max_lag"));
    }
    Ok(LagBins {
        max_lag,
        lag_width: lag,
    })
}

pub fn anisotropy(rotation: (f64, f64, f64), ratios: (f64, f64)) -> PyResult<Option<Anisotropy>> {
    if rotation == (0.0, 0.0, 0.0) && ratios == (1.0, 1.0) {
        return Ok(None);
    }
    let angles = Angles {
        azimuth: rotation.0,
        dip: rotation.1,
        rake: rotation.2,
        major: 1.0,
        semi: ratios.0,
        minor: ratios.1,
    };
    Ok(Some(Anisotropy::new(angles).map_err(err)?))
}

/// One nested structure: `sill` is the partial sill, `range` the range along
/// the major axis.
#[pyclass(module = "ceres", name = "Structure", frozen, from_py_object)]
#[derive(Clone)]
pub struct Structure(pub CoreStructure);

#[pymethods]
impl Structure {
    #[new]
    #[pyo3(signature = (model, sill, range, order=None, exponent=None))]
    fn new(
        model: &str,
        sill: f64,
        range: f64,
        order: Option<f64>,
        exponent: Option<f64>,
    ) -> PyResult<Self> {
        if !(sill >= 0.0 && range > 0.0) {
            return Err(invalid("need sill >= 0 and range > 0"));
        }
        Ok(Self(CoreStructure::new(
            self::model(model, order, exponent)?,
            sill,
            range,
        )))
    }

    #[getter]
    fn model(&self) -> &'static str {
        model_name(self.0.model)
    }

    #[getter]
    fn sill(&self) -> f64 {
        self.0.sill
    }

    #[getter]
    fn range(&self) -> f64 {
        self.0.range
    }

    fn __repr__(&self) -> String {
        format!(
            "Structure({:?}, sill={}, range={})",
            self.model(),
            self.0.sill,
            self.0.range
        )
    }
}

fn structure(obj: &Bound<PyAny>) -> PyResult<CoreStructure> {
    if let Ok(s) = obj.extract::<Structure>() {
        return Ok(s.0);
    }
    let (name, sill, range): (String, f64, f64) = obj
        .extract()
        .map_err(|_| invalid("structures are Structure or (model, sill, range) tuples"))?;
    Ok(Structure::new(&name, sill, range, None, None)?.0)
}

/// Nugget plus nested structures sharing one anisotropy. `rotation` is
/// azimuth, dip, rake in degrees; `ratios` are semi-major/major and
/// minor/major range ratios.
#[pyclass(module = "ceres", name = "Variogram", frozen, from_py_object)]
#[derive(Clone)]
pub struct Variogram(pub CoreVariogram);

#[pymethods]
impl Variogram {
    #[new]
    #[pyo3(signature = (structures, nugget=0.0, rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0)))]
    fn new(
        structures: Vec<Bound<PyAny>>,
        nugget: f64,
        rotation: (f64, f64, f64),
        ratios: (f64, f64),
    ) -> PyResult<Self> {
        if nugget < 0.0 {
            return Err(invalid("nugget must be >= 0"));
        }
        Ok(Self(CoreVariogram {
            nugget,
            structures: structures.iter().map(structure).collect::<PyResult<_>>()?,
            anisotropy: anisotropy(rotation, ratios)?,
        }))
    }

    /// Weighted least-squares fit of one structure plus nugget.
    #[staticmethod]
    #[pyo3(signature = (experimental, model="spherical", weighting="count"))]
    fn fit(experimental: &ExperimentalVariogram, model: &str, weighting: &str) -> PyResult<Self> {
        let fitted = variogram::fit(
            &experimental.0,
            self::model(model, None, None)?,
            self::weighting(weighting)?,
        )
        .map_err(err)?;
        Ok(Self(fitted.variogram))
    }

    /// Same structures with a new anisotropy.
    #[pyo3(signature = (rotation, ratios))]
    fn with_anisotropy(&self, rotation: (f64, f64, f64), ratios: (f64, f64)) -> PyResult<Self> {
        let mut v = self.0.clone();
        v.anisotropy = anisotropy(rotation, ratios)?;
        Ok(Self(v))
    }

    #[getter]
    fn nugget(&self) -> f64 {
        self.0.nugget
    }

    #[getter]
    fn structures(&self) -> Vec<Structure> {
        self.0.structures.iter().copied().map(Structure).collect()
    }

    #[getter]
    fn rotation(&self) -> (f64, f64, f64) {
        self.0.anisotropy.as_ref().map_or((0.0, 0.0, 0.0), |a| {
            (a.angles.azimuth, a.angles.dip, a.angles.rake)
        })
    }

    #[getter]
    fn ratios(&self) -> (f64, f64) {
        self.0
            .anisotropy
            .as_ref()
            .map_or((1.0, 1.0), |a| (a.angles.semi, a.angles.minor))
    }

    #[getter]
    fn sill(&self) -> f64 {
        self.0.total_sill()
    }

    /// γ at scalar lags (along the major axis when anisotropic).
    fn gamma<'py>(&self, py: Python<'py>, h: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let g = floats(h, "h")?
            .into_iter()
            .map(|h| self.0.gamma(h))
            .collect();
        Ok(array1(py, g).into_any())
    }

    /// Covariance `C(h) = sill - γ(h)` at scalar lags.
    fn covariance<'py>(&self, py: Python<'py>, h: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let c = floats(h, "h")?.into_iter().map(|h| self.0.cov(h)).collect();
        Ok(array1(py, c).into_any())
    }

    /// γ between paired rows of `a` and `b`, honouring anisotropy.
    fn gamma_between<'py>(
        &self,
        py: Python<'py>,
        a: &Bound<PyAny>,
        b: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (a, b) = (points(a)?, points(b)?);
        same_length(a.len(), b.len(), "b")?;
        let g = a
            .iter()
            .zip(&b)
            .map(|(p, q)| self.0.gamma_points(p, q))
            .collect();
        Ok(array1(py, g).into_any())
    }

    fn to_json(&self) -> String {
        serde_json::to_string(&self.0).expect("variogram serializes")
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self(serde_json::from_str(text).map_err(invalid)?))
    }

    fn __repr__(&self) -> String {
        let parts: Vec<String> = self.structures().iter().map(|s| s.__repr__()).collect();
        format!(
            "Variogram(nugget={}, structures=[{}], rotation={:?}, ratios={:?})",
            self.0.nugget,
            parts.join(", "),
            self.rotation(),
            self.ratios()
        )
    }
}

/// Lag centres, semivariances and pair counts.
#[pyclass(module = "ceres", name = "ExperimentalVariogram", frozen)]
pub struct ExperimentalVariogram(pub Experimental);

#[pymethods]
impl ExperimentalVariogram {
    #[getter]
    fn lags<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array1(py, self.0.lags.clone()).into_any()
    }

    #[getter]
    fn gammas<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array1(py, self.0.gammas.clone()).into_any()
    }

    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array1(py, self.0.counts.iter().map(|&c| c as f64).collect()).into_any()
    }

    #[pyo3(signature = (model="spherical", weighting="count"))]
    fn fit(&self, model: &str, weighting: &str) -> PyResult<Variogram> {
        Variogram::fit(self, model, weighting)
    }

    fn __repr__(&self) -> String {
        format!("ExperimentalVariogram({} lags)", self.0.lags.len())
    }
}

fn samples(coords: &Bound<PyAny>, values: &Bound<PyAny>) -> PyResult<(Vec<Point>, Vec<f64>)> {
    let (locs, values) = (points(coords)?, finite(values, "values")?);
    same_length(locs.len(), values.len(), "values")?;
    Ok((locs, values))
}

/// Experimental variogram, omnidirectional unless `azimuth` is given.
#[pyfunction]
#[pyo3(signature = (coords, values, lag, max_lag, azimuth=None, dip=0.0, tolerance=22.5, bandwidth=None, estimator="matheron"))]
#[allow(clippy::too_many_arguments)]
fn experimental_variogram(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    lag: f64,
    max_lag: f64,
    azimuth: Option<f64>,
    dip: f64,
    tolerance: f64,
    bandwidth: Option<f64>,
    estimator: &str,
) -> PyResult<ExperimentalVariogram> {
    let (locs, values) = samples(coords, values)?;
    let direction = azimuth.map(|azimuth| Direction {
        azimuth,
        dip,
        tolerance,
        bandwidth,
    });
    let exp = experimental(
        &locs,
        &values,
        &bins(lag, max_lag)?,
        self::estimator(estimator)?,
        direction.as_ref(),
    )
    .map_err(err)?;
    Ok(ExperimentalVariogram(exp))
}

/// γ on a plane as an angle × lag grid, with the fitted range per angle.
#[pyclass(module = "ceres", name = "VariogramMap", frozen)]
pub struct VariogramMap {
    #[pyo3(get)]
    lags: Py<PyAny>,
    #[pyo3(get)]
    angles: Py<PyAny>,
    #[pyo3(get)]
    gammas: Py<PyAny>,
    #[pyo3(get)]
    counts: Py<PyAny>,
    #[pyo3(get)]
    ranges: Py<PyAny>,
}

fn grid(values: Vec<f64>, rows: usize) -> Vec<Vec<f64>> {
    let cols = values.len() / rows.max(1);
    values.chunks(cols.max(1)).map(<[f64]>::to_vec).collect()
}

/// Variogram map on the plane spanned by `u` and `v` (default: horizontal,
/// angles counter-clockwise from east).
#[pyfunction]
#[pyo3(signature = (coords, values, lag, max_lag, u=vec![1.0, 0.0, 0.0], v=vec![0.0, 1.0, 0.0], tolerance=22.5, steps=36, model="spherical"))]
#[allow(clippy::too_many_arguments)]
fn variogram_map(
    py: Python,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    lag: f64,
    max_lag: f64,
    u: Vec<f64>,
    v: Vec<f64>,
    tolerance: f64,
    steps: usize,
    model: &str,
) -> PyResult<VariogramMap> {
    let (locs, values) = samples(coords, values)?;
    let params = PlaneMapParams {
        bins: bins(lag, max_lag)?,
        tolerance,
        angle_steps: steps,
        model: self::model(model, None, None)?,
        ..Default::default()
    };
    let m = plane_map(
        &locs,
        &values,
        triple(u, 0.0, "u")?,
        triple(v, 0.0, "v")?,
        &params,
    )
    .map_err(err)?;
    let rows = m.angles.len();
    Ok(VariogramMap {
        lags: array1(py, m.lags).into_any().unbind(),
        angles: array1(py, m.angles).into_any().unbind(),
        gammas: array2(py, &grid(m.gammas, rows)).into_any().unbind(),
        counts: array2(
            py,
            &grid(m.counts.iter().map(|&c| c as f64).collect(), rows),
        )
        .into_any()
        .unbind(),
        ranges: array1(py, m.ranges.iter().map(|r| r.unwrap_or(f64::NAN)).collect())
            .into_any()
            .unbind(),
    })
}

/// Fitted range per direction over a hemisphere: returns `(azimuths, dips,
/// ranges)` arrays, NaN where too few pairs.
#[pyfunction]
#[pyo3(signature = (coords, values, lag, max_lag, azimuth_steps=24, dip_steps=6, tolerance=22.5, model="spherical"))]
#[allow(clippy::too_many_arguments)]
fn variogram_surface_ranges<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    lag: f64,
    max_lag: f64,
    azimuth_steps: usize,
    dip_steps: usize,
    tolerance: f64,
    model: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    let (locs, values) = samples(coords, values)?;
    let params = SurfaceParams {
        bins: bins(lag, max_lag)?,
        tolerance,
        azimuth_steps,
        dip_steps,
        model: self::model(model, None, None)?,
        ..Default::default()
    };
    let s = variogram_surface(&locs, &values, &params).map_err(err)?;
    let col = |f: fn(&variogram::surface::SurfaceDirection) -> f64| {
        array1(py, s.directions.iter().map(f).collect())
    };
    PyTuple::new(
        py,
        [
            col(|d| d.azimuth),
            col(|d| d.dip),
            col(|d| d.range.unwrap_or(f64::NAN)),
        ],
    )
}

/// Linear model of coregionalization: `nugget` and each structure's `sills`
/// are symmetric `nvar × nvar` matrices.
#[pyclass(module = "ceres", name = "Coregionalization", frozen)]
pub struct Coregionalization(pub CoreCoreg);

#[pymethods]
impl Coregionalization {
    #[new]
    #[pyo3(signature = (nugget, structures, rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0)))]
    fn new(
        nugget: Vec<Vec<f64>>,
        structures: Vec<(String, f64, Vec<Vec<f64>>)>,
        rotation: (f64, f64, f64),
        ratios: (f64, f64),
    ) -> PyResult<Self> {
        let n = nugget.len();
        let square = |m: &Vec<Vec<f64>>| m.len() == n && m.iter().all(|r| r.len() == n);
        if !square(&nugget) || structures.iter().any(|s| !square(&s.2)) {
            return Err(invalid("nugget and sills must all be nvar x nvar"));
        }
        let structures = structures
            .into_iter()
            .map(|(name, range, sills)| {
                Ok(CoregStructure {
                    model: model(&name, None, None)?,
                    range,
                    sills,
                })
            })
            .collect::<PyResult<_>>()?;
        let mut c = CoreCoreg::new(nugget, structures);
        c.anisotropy = anisotropy(rotation, ratios)?;
        Ok(Self(c))
    }

    #[getter]
    fn nvar(&self) -> usize {
        self.0.nvar
    }

    /// Cross-covariance `C_ij` between paired rows of `a` and `b`.
    fn cross_covariance<'py>(
        &self,
        py: Python<'py>,
        i: usize,
        j: usize,
        a: &Bound<PyAny>,
        b: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if i >= self.0.nvar || j >= self.0.nvar {
            return Err(invalid("variable index out of range"));
        }
        let (a, b) = (points(a)?, points(b)?);
        same_length(a.len(), b.len(), "b")?;
        let c = a
            .iter()
            .zip(&b)
            .map(|(p, q)| self.0.cross_cov(i, j, p, q))
            .collect();
        Ok(array1(py, c).into_any())
    }
}

/// Markov transiogram for categories with the given proportions.
#[pyclass(module = "ceres", name = "Transiogram", frozen)]
pub struct Transiogram(CoreTransiogram);

#[pymethods]
impl Transiogram {
    #[new]
    fn new(proportions: Vec<f64>, range: f64) -> PyResult<Self> {
        Ok(Self(CoreTransiogram::new(proportions, range).map_err(err)?))
    }

    /// `P(category j at x + h | category i at x)` as a matrix.
    fn matrix<'py>(&self, py: Python<'py>, h: f64) -> Bound<'py, PyAny> {
        array2(py, &self.0.matrix(h)).into_any()
    }
}

/// Empirical transition probabilities: `(lags, probs[lag, i, j], counts)`.
#[pyfunction]
fn experimental_transiogram<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    categories: Vec<usize>,
    lag: f64,
    max_lag: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    let locs = points(coords)?;
    same_length(locs.len(), categories.len(), "categories")?;
    let n = categories.iter().max().map_or(0, |m| m + 1);
    let t = empirical_transiogram(&locs, &categories, n, max_lag, lag).map_err(err)?;
    let np = py.import("numpy")?;
    let probs = np.call_method1("asarray", (t.probs,))?;
    PyTuple::new(
        py,
        [
            array1(py, t.lags).into_any(),
            probs,
            array1(py, t.counts.iter().map(|&c| c as f64).collect()).into_any(),
        ],
    )
}

/// Change-of-support coefficient `r` for blocks of `size` under the discrete
/// Gaussian model, and the matching block anamorphosis. `variogram` is the
/// variogram of the Gaussian scores (unit sill).
#[pyfunction]
#[pyo3(signature = (anamorphosis, variogram, size, discretization=(4, 4, 1)))]
fn change_of_support(
    anamorphosis: PyRef<Anamorphosis>,
    variogram: &Variogram,
    size: Vec<f64>,
    discretization: (usize, usize, usize),
) -> PyResult<(f64, Anamorphosis)> {
    let disc = BlockDiscretization {
        nx: discretization.0,
        ny: discretization.1,
        nz: discretization.2,
    };
    let (r, block) = transforms::change_of_support(
        &anamorphosis.inner()?,
        &variogram.0,
        (0.0, 0.0, 0.0),
        triple(size, 1.0, "size")?,
        &disc,
    )
    .map_err(invalid)?;
    Ok((r, Anamorphosis::from_inner(block)))
}

/// Mean Gaussian correlation between points of one block (`γ̄`-based).
#[pyfunction]
#[pyo3(signature = (variogram, size, discretization=(4, 4, 1)))]
fn block_correlation(
    variogram: &Variogram,
    size: Vec<f64>,
    discretization: (usize, usize, usize),
) -> PyResult<f64> {
    let disc = BlockDiscretization {
        nx: discretization.0,
        ny: discretization.1,
        nz: discretization.2,
    };
    Ok(block_average_correlation(
        &variogram.0,
        (0.0, 0.0, 0.0),
        triple(size, 1.0, "size")?,
        &disc,
    ))
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Structure>()?;
    m.add_class::<Variogram>()?;
    m.add_class::<ExperimentalVariogram>()?;
    m.add_class::<VariogramMap>()?;
    m.add_class::<Coregionalization>()?;
    m.add_class::<Transiogram>()?;
    m.add_function(wrap_pyfunction!(experimental_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(variogram_map, m)?)?;
    m.add_function(wrap_pyfunction!(variogram_surface_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(experimental_transiogram, m)?)?;
    m.add_function(wrap_pyfunction!(change_of_support, m)?)?;
    m.add_function(wrap_pyfunction!(block_correlation, m)?)?;
    Ok(())
}
