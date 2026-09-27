use ceres_core::{Geometry, Layout};
use numpy::IntoPyArray;
use numpy::ndarray::Array3;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use serde::{Deserialize, Serialize};
use transforms::dgm::{BlockDiscretization, block_average_correlation};
use variogram::surface::{PlaneMapParams, plane_map};
use variogram::volume::VolumeParams;
use variogram::{
    Angles, Anisotropy, AnisotropySpec, Bounds, CoregStructure, Coregionalization as CoreCoreg,
    Direction, Estimator, Experimental, LagBins, Model, NestedSpec, Structure as CoreStructure,
    StructureSpec, Support, Transiogram as CoreTransiogram, Variogram as CoreVariogram, Weighting,
    cross_experimental, downhole, empirical_transiogram, experimental, experimental_realizations,
    experimental_set, extrapolated_nugget, fit_coregionalization, fit_directional, fit_nested,
};

use crate::args::{Point, array1, array2, column, finite, floats, points, same_length, triple};
use crate::containers::PyBlockModel;
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
        "covariance" => Ok(Estimator::Covariance),
        "correlogram" => Ok(Estimator::Correlogram),
        "pairwise-relative" | "pairwise_relative" => Ok(Estimator::PairwiseRelative),
        _ => Err(invalid(format!("unknown estimator {name:?}"))),
    }
}

fn weighting(name: &str) -> PyResult<Weighting> {
    match name {
        "uniform" => Ok(Weighting::Uniform),
        "count" => Ok(Weighting::ByCount),
        "count/gamma" => Ok(Weighting::ByCountOverGamma),
        "count/distance" => Ok(Weighting::ByCountOverDistance),
        _ => Err(invalid(format!(
            "unknown weighting {name:?}; use uniform, count, count/gamma or count/distance"
        ))),
    }
}

#[derive(FromPyObject)]
enum Models {
    One(String),
    Many(Vec<String>),
}

#[derive(FromPyObject, Clone, Copy)]
enum Limit {
    Fixed(f64),
    Between(f64, f64),
}

impl Limit {
    fn bounds(self) -> Bounds {
        match self {
            Limit::Fixed(x) => (x, x),
            Limit::Between(lo, hi) => (lo, hi),
        }
    }
}

type Limits = Option<Vec<Option<Limit>>>;

fn per_structure(given: Limits, n: usize, what: &str) -> PyResult<Vec<Option<Bounds>>> {
    let Some(given) = given else {
        return Ok(vec![None; n]);
    };
    same_length(n, given.len(), what)?;
    Ok(given.into_iter().map(|l| l.map(Limit::bounds)).collect())
}

fn nested(
    model: Models,
    nugget: Option<Limit>,
    sills: Limits,
    ranges: Limits,
) -> PyResult<NestedSpec> {
    let names = match model {
        Models::One(name) => vec![name],
        Models::Many(names) => names,
    };
    let n = names.len();
    let (sills, ranges) = (
        per_structure(sills, n, "sills")?,
        per_structure(ranges, n, "ranges")?,
    );
    let structures = names
        .iter()
        .zip(sills.into_iter().zip(ranges))
        .map(|(name, (sill, range))| {
            Ok(StructureSpec {
                model: self::model(name, None, None)?,
                sill,
                range,
            })
        })
        .collect::<PyResult<_>>()?;
    Ok(NestedSpec {
        nugget: nugget.map(Limit::bounds),
        structures,
    })
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
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "Structure", frozen, from_py_object)]
#[derive(Clone)]
pub struct Structure(pub CoreStructure);

#[pymethods]
impl Structure {
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
    #[pyo3(signature = (model, sill, range, *, order=None, exponent=None))]
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

/// Nugget plus nested structures sharing one anisotropy. `rotation` is one
/// azimuth, dip, rake triple in degrees for the whole model, where `angles`
/// (as in `LocalAnisotropy`) holds one per location; `ratios` are
/// semi-major/major and minor/major range ratios.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "Variogram", frozen, from_py_object)]
#[derive(Clone)]
pub struct Variogram(pub CoreVariogram);

#[pymethods]
impl Variogram {
    #[new]
    #[pyo3(signature = (structures, *, nugget=0.0,rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0)))]
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

    /// Weighted least-squares fit of a nugget plus one to three nested structures.
    ///
    /// Every parameter is free (None), fixed (a float) or bounded (a
    /// ``(low, high)`` pair). For given ranges the nugget and sills are solved
    /// exactly; the ranges are searched on a grid and refined, deterministically.
    ///
    /// Parameters
    /// ----------
    /// experimental : ExperimentalVariogram
    /// model : str or sequence of str
    ///     One shape per structure, shortest range first: fitted ranges
    ///     increase in this order.
    /// weighting : {"count", "uniform", "count/gamma", "count/distance"}
    ///     Least-squares weight per lag: N(h), 1, N(h)/γ(h)² or N(h)/h².
    /// nugget : float or (float, float), optional
    /// sills, ranges : sequence of (None, float or (float, float)), optional
    ///     Partial sill and range of each structure, one entry per model.
    ///
    /// Returns
    /// -------
    /// Variogram
    ///     Isotropic; one free structure with a free nugget reproduces the
    ///     single-structure fit.
    #[staticmethod]
    #[pyo3(signature = (experimental, model=Models::One("spherical".into()), *, weighting="count", nugget=None, sills=None, ranges=None))]
    fn fit(
        experimental: &ExperimentalVariogram,
        model: Models,
        weighting: &str,
        nugget: Option<Limit>,
        sills: Limits,
        ranges: Limits,
    ) -> PyResult<Self> {
        let spec = nested(model, nugget, sills, ranges)?;
        let fitted =
            fit_nested(&experimental.0, &spec, self::weighting(weighting)?).map_err(err)?;
        Ok(Self(fitted.variogram))
    }

    /// One anisotropic model fitted jointly to experimental variograms in
    /// several directions.
    ///
    /// The structures share the anisotropy; their ranges are major-axis
    /// ranges. Nugget, sills and ranges are free, fixed or bounded as in
    /// `Variogram.fit`, and so is each angle and range ratio. Free ranges stay
    /// within the largest lag; bound them, e.g. ``ranges=[None, (50, 300)]``,
    /// to go further. The search is deterministic.
    ///
    /// Parameters
    /// ----------
    /// experimentals : sequence of ExperimentalVariogram
    /// directions : sequence of (float, float)
    ///     Azimuth and dip in degrees of each experimental variogram.
    /// model, weighting, nugget, sills, ranges
    ///     As in `Variogram.fit`.
    /// rotation : sequence of (None, float or (float, float)), optional
    ///     Azimuth, dip and rake in degrees; free angles are returned with
    ///     azimuth and rake in [0, 180).
    /// ratios : sequence of (None, float or (float, float)), optional
    ///     Semi-major/major and minor/major range ratios, in (0, 1] when free.
    ///     When every direction is horizontal, dip and rake default to 0 and
    ///     the minor ratio to 1: the fit is two-dimensional.
    ///
    /// Returns
    /// -------
    /// Variogram
    #[staticmethod]
    #[pyo3(signature = (experimentals, directions, model=Models::One("spherical".into()), *, weighting="count", nugget=None, sills=None, ranges=None, rotation=None, ratios=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit_directional(
        experimentals: Vec<PyRef<ExperimentalVariogram>>,
        directions: Vec<(f64, f64)>,
        model: Models,
        weighting: &str,
        nugget: Option<Limit>,
        sills: Limits,
        ranges: Limits,
        rotation: Limits,
        ratios: Limits,
    ) -> PyResult<Self> {
        same_length(experimentals.len(), directions.len(), "directions")?;
        let spec = nested(model, nugget, sills, ranges)?;
        let (rotation, ratios) = (
            per_structure(rotation, 3, "rotation")?,
            per_structure(ratios, 2, "ratios")?,
        );
        let aniso = AnisotropySpec {
            azimuth: rotation[0],
            dip: rotation[1],
            rake: rotation[2],
            semi: ratios[0],
            minor: ratios[1],
        };
        let exps: Vec<Experimental> = experimentals.iter().map(|e| e.0.clone()).collect();
        let fitted = fit_directional(
            &exps,
            &directions,
            &spec,
            &aniso,
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

    /// γ between paired rows of `a` and `b`, honoring anisotropy.
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

    /// JSON of the model.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
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

/// Lag centers, semivariances and pair counts; ``covariances`` holds C(h)
/// for the covariance estimator, ρ(h) for the correlogram, and is None
/// otherwise.
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

    #[getter]
    fn covariances<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.0.covariances.clone().map(|c| array1(py, c).into_any())
    }

    /// Weighted least-squares fit; see `Variogram.fit`.
    #[pyo3(signature = (model=Models::One("spherical".into()), *, weighting="count", nugget=None, sills=None, ranges=None))]
    fn fit(
        &self,
        model: Models,
        weighting: &str,
        nugget: Option<Limit>,
        sills: Limits,
        ranges: Limits,
    ) -> PyResult<Variogram> {
        Variogram::fit(self, model, weighting, nugget, sills, ranges)
    }

    /// Nugget extrapolated to zero lag from the first `lags` lags.
    ///
    /// Intercept of a pair-count-weighted straight line through them, with
    /// intercept and slope kept non-negative. Meant for a downhole variogram
    /// (``holes=`` in `experimental_variogram`), whose shortest lags are the
    /// closest pairs available.
    ///
    /// Parameters
    /// ----------
    /// lags : int
    ///     Number of shortest lags, at least 2.
    ///
    /// Returns
    /// -------
    /// float
    #[pyo3(signature = (*, lags=3))]
    fn nugget(&self, lags: usize) -> PyResult<f64> {
        extrapolated_nugget(&self.0, lags).map_err(err)
    }

    fn __repr__(&self) -> String {
        format!("ExperimentalVariogram({} lags)", self.0.lags.len())
    }
}

fn samples(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    what: &str,
) -> PyResult<(Vec<Point>, Vec<f64>)> {
    let values = finite(&column(Some(coords), values, what)?, what)?;
    let locs = points(coords)?;
    same_length(locs.len(), values.len(), what)?;
    Ok((locs, values))
}

enum Locations {
    Points(Vec<Point>),
    Grid(Geometry, Vec<u64>),
}

impl Locations {
    fn support(&self) -> Support<'_> {
        match self {
            Locations::Points(p) => Support::Points(p),
            Locations::Grid(g, cells) => Support::Grid(g, cells),
        }
    }

    fn len(&self) -> usize {
        match self {
            Locations::Points(p) => p.len(),
            Locations::Grid(_, cells) => cells.len(),
        }
    }
}

/// The cells of a regular or masked BlockModel with `method` None or
/// "grid", else the points of `coords`.
fn locations(coords: &Bound<PyAny>, method: Option<&str>) -> PyResult<Locations> {
    let grid = coords.cast::<PyBlockModel>().ok().and_then(|m| {
        let m = &m.get().0;
        match m.layout() {
            Layout::Regular => Some((*m.geometry(), (0..m.geometry().cells()).collect())),
            Layout::Masked(index) => Some((*m.geometry(), index.clone())),
            Layout::SubBlocked { .. } => None,
        }
    });
    match (method, grid) {
        (None | Some("grid"), Some((g, cells))) => Ok(Locations::Grid(g, cells)),
        (Some("grid"), None) => Err(invalid(
            "method=\"grid\" needs a regular or masked BlockModel",
        )),
        (None | Some("pairs"), _) => Ok(Locations::Points(points(coords)?)),
        (Some(m), _) => Err(invalid(format!(
            "unknown method {m:?}; use \"pairs\" or \"grid\""
        ))),
    }
}

/// Experimental variogram, omnidirectional unless `azimuth` is given, or the
/// cross-variogram of `values` and `other`.
///
/// Every estimator is returned in variogram form, so any of them can be fitted.
///
/// Parameters
/// ----------
/// coords : array_like, shape (n, 2) or (n, 3), PointSet or BlockModel
/// values : array_like, shape (n,), or str
///     Values, or the column of ``coords`` holding them; so are ``other``
///     (of ``other_coords`` when given) and ``holes``.
/// lag, max_lag : float
///     Lag-bin width and largest pair distance.
/// azimuth, dip, tolerance : float
///     Direction and cone half-angle in degrees; omnidirectional when
///     ``azimuth`` is None.
/// bandwidth : float, optional
///     Largest offset of a pair from the direction line.
/// estimator : {"matheron", "cressie-hawkins", "covariance", "correlogram", "pairwise-relative"}
///     Classical, robust, σ² − C(h), 1 − ρ(h), or the pairwise-relative
///     variogram (non-negative values only).
/// standardize : bool
///     Divide by the sample variance so the sill is 1. The correlogram and
///     pairwise-relative estimates are dimensionless and left unchanged; a
///     cross-variogram is divided by σ₁σ₂.
/// other : array_like, shape (m,), optional
///     Second variable, for its cross-variogram with ``values``: "matheron"
///     gives γ₁₂(h) = Σ Δz₁·Δz₂ / 2N, "covariance" gives C₁₂(0) − C₁₂(h) with
///     C₁₂(h) in ``covariances``. C₁₂(h) pairs ``values`` at the tail with
///     ``other`` at the head, so the directional cone is one-sided: reversing
///     ``azimuth`` estimates C₂₁(h); omnidirectional averages both.
/// other_coords : array_like, shape (m, 2) or (m, 3), optional
///     Where ``other`` is sampled when not at ``coords``. Only the
///     cross-covariance is defined then, with C₁₂(0) estimated from the pairs
///     closer than half a lag.
/// holes : array_like, shape (n,), optional
///     Hole id of each sample, for a downhole variogram: only pairs in the
///     same hole count, lag ``k`` gathers the pairs within ``lag / 2`` of
///     ``k * lag`` and reports their mean distance. With ``azimuth``, only
///     the pairs inside the direction cone count, e.g. the flat stretches of
///     bent holes. Not with ``other``.
/// method : {None, "pairs", "grid"}
///     How pairs are found. "pairs" compares every two samples, O(n²).
///     "grid" needs a regular or masked BlockModel: every pair of cells one
///     index offset apart has the same separation, so each offset within
///     ``max_lag`` and the direction cone is binned once and its pairs are
///     gathered by shifting cell indices, O(n x offsets), with the same pairs
///     and estimates as "pairs" up to round-off; offsets exactly on a lag or
///     cone boundary are decided once, where round-off can split their pairs
///     in the search. None takes "grid" for such a BlockModel without
///     ``holes`` or ``other_coords``, else "pairs".
///
/// Returns
/// -------
/// ExperimentalVariogram
#[pyfunction]
#[pyo3(signature = (coords, values, lag, max_lag, *, azimuth=None, dip=0.0, tolerance=22.5, bandwidth=None, estimator="matheron", standardize=false, other=None, other_coords=None, holes=None, method=None))]
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
    standardize: bool,
    other: Option<&Bound<PyAny>>,
    other_coords: Option<&Bound<PyAny>>,
    holes: Option<&Bound<PyAny>>,
    method: Option<&str>,
) -> PyResult<ExperimentalVariogram> {
    let (bins, estimator) = (bins(lag, max_lag)?, self::estimator(estimator)?);
    if holes.is_some() || other_coords.is_some() {
        if method == Some("grid") {
            return Err(invalid(
                "method=\"grid\" takes neither holes nor other_coords",
            ));
        }
        let (locs, values) = samples(coords, values, "values")?;
        let holes = holes
            .map(|h| column(Some(coords), h, "holes"))
            .transpose()?;
        let direction = azimuth.map(|azimuth| Direction {
            azimuth,
            dip,
            tolerance,
            bandwidth,
        });
        if let Some((_, holes)) = crate::args::holes(holes.as_ref(), locs.len())? {
            if other.is_some() {
                return Err(invalid("holes does not take other"));
            }
            let dir = direction.as_ref();
            let exp = downhole(&locs, &values, &holes, &bins, estimator, dir, standardize)
                .map_err(err)?;
            return Ok(ExperimentalVariogram(exp));
        }
        let (Some(at), Some(other)) = (other_coords, other) else {
            return Err(invalid("other_coords needs other"));
        };
        let (at, other) = samples(at, other, "other")?;
        let exp = cross_experimental(
            &locs,
            &values,
            Some(&at),
            &other,
            &bins,
            estimator,
            direction.as_ref(),
            standardize,
        )
        .map_err(err)?;
        return Ok(ExperimentalVariogram(exp));
    }
    let values = finite(&column(Some(coords), values, "values")?, "values")?;
    let locs = locations(coords, method)?;
    same_length(locs.len(), values.len(), "values")?;
    let direction = azimuth.map(|azimuth| Direction {
        azimuth,
        dip,
        tolerance,
        bandwidth,
    });
    let other = other
        .map(|o| finite(&column(Some(coords), o, "other")?, "other"))
        .transpose()?;
    let (at, dir) = (locs.support(), direction.as_ref());
    let exp = match other {
        None => experimental(at, &values, &bins, estimator, dir, standardize),
        Some(other) => {
            same_length(values.len(), other.len(), "other")?;
            cross_experimental(
                at,
                &values,
                None,
                &other,
                &bins,
                estimator,
                dir,
                standardize,
            )
        }
    }
    .map_err(err)?;
    Ok(ExperimentalVariogram(exp))
}

/// Direct and cross experimental variograms of several variables, indexed
/// by variable pair.
///
/// ``set[i, j]`` is the ExperimentalVariogram of variables ``i`` and ``j``
/// (their positions in ``values``, or their column names): the direct
/// variogram for ``i == j``, else the cross-variogram, the same for
/// ``(j, i)``. With ``directions`` each entry is a list, one per direction.
/// `Coregionalization.fit` takes the set as it is, directions included.
#[pyclass(module = "ceres", name = "VariogramSet", frozen)]
pub struct VariogramSet {
    entries: Vec<Vec<Option<Vec<Experimental>>>>,
    names: Vec<Option<String>>,
    directions: Option<Vec<(f64, f64)>>,
}

impl VariogramSet {
    fn index(&self, key: &Bound<PyAny>) -> PyResult<usize> {
        let found = match key.extract::<String>() {
            Ok(name) => self.names.iter().position(|n| n.as_deref() == Some(&name)),
            Err(_) => key
                .extract::<usize>()
                .ok()
                .filter(|&i| i < self.names.len()),
        };
        found.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("no variable {key} in the set"))
        })
    }
}

#[pymethods]
impl VariogramSet {
    /// Number of variables.
    #[getter]
    fn nvar(&self) -> usize {
        self.names.len()
    }

    /// Column name of each variable, None where given as an array.
    #[getter]
    fn names(&self) -> Vec<Option<String>> {
        self.names.clone()
    }

    /// ``(azimuth, dip)`` of each direction, None when omnidirectional.
    #[getter]
    fn directions(&self) -> Option<Vec<(f64, f64)>> {
        self.directions.clone()
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: (Bound<'py, PyAny>, Bound<'py, PyAny>),
    ) -> PyResult<Bound<'py, PyAny>> {
        let (i, j) = (self.index(&key.0)?, self.index(&key.1)?);
        let entry = self.entries[i.min(j)][i.max(j)]
            .as_ref()
            .expect("upper triangle");
        let wrap = |e: &Experimental| Bound::new(py, ExperimentalVariogram(e.clone()));
        if self.directions.is_none() {
            return Ok(wrap(&entry[0])?.into_any());
        }
        let list = entry.iter().map(wrap).collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, list)?.into_any())
    }

    fn __repr__(&self) -> String {
        let dirs = match &self.directions {
            None => "omnidirectional".into(),
            Some(d) => format!("{} directions", d.len()),
        };
        format!("VariogramSet({} variables, {dirs})", self.nvar())
    }
}

/// Every direct and cross experimental variogram of several variables at
/// the same samples.
///
/// Parameters
/// ----------
/// coords : array_like, shape (n, 2) or (n, 3), PointSet or BlockModel
/// values : sequence of array_like or str
///     One column per variable, or the column of ``coords`` holding it.
/// lag, max_lag : float
///     Lag-bin width and largest pair distance.
/// directions : sequence of (float, float), optional
///     Azimuth and dip in degrees of each direction; omnidirectional when
///     None.
/// tolerance, bandwidth, estimator, standardize, method
///     As in `experimental_variogram`, for every direction. Cross-variograms
///     take the "matheron" or "covariance" estimator.
///
/// Returns
/// -------
/// VariogramSet
///     ``set[i, j]`` as `experimental_variogram` gives it for ``values[i]``
///     with ``other=values[j]``.
#[pyfunction]
#[pyo3(signature = (coords, values, lag, max_lag, *, directions=None, tolerance=22.5, bandwidth=None, estimator="matheron", standardize=false, method=None))]
#[allow(clippy::too_many_arguments)]
fn experimental_variograms(
    coords: &Bound<PyAny>,
    values: Vec<Bound<PyAny>>,
    lag: f64,
    max_lag: f64,
    directions: Option<Vec<(f64, f64)>>,
    tolerance: f64,
    bandwidth: Option<f64>,
    estimator: &str,
    standardize: bool,
    method: Option<&str>,
) -> PyResult<VariogramSet> {
    let (bins, estimator) = (bins(lag, max_lag)?, self::estimator(estimator)?);
    let locs = locations(coords, method)?;
    let columns = values
        .iter()
        .map(|v| {
            let column = finite(&column(Some(coords), v, "values")?, "values")?;
            same_length(locs.len(), column.len(), "values")?;
            Ok(column)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let cones: Vec<Direction> = directions
        .iter()
        .flatten()
        .map(|&(azimuth, dip)| Direction {
            azimuth,
            dip,
            tolerance,
            bandwidth,
        })
        .collect();
    let variables: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    let entries = experimental_set(
        locs.support(),
        &variables,
        &bins,
        estimator,
        &cones,
        standardize,
    )
    .map_err(err)?;
    Ok(VariogramSet {
        entries,
        names: values.iter().map(|v| v.extract().ok()).collect(),
        directions,
    })
}

/// Direct experimental variogram of each row of `realizations` at `coords`,
/// in parallel over the rows: one list per row, one entry per direction
/// (one omnidirectional when `directions` is None). Backs
/// `check_realizations`.
#[pyfunction]
#[pyo3(signature = (coords, realizations, lag, max_lag, *, directions=None, tolerance=22.5, bandwidth=None, method=None))]
#[allow(clippy::too_many_arguments)]
fn _realization_variograms(
    coords: &Bound<PyAny>,
    realizations: &Bound<PyAny>,
    lag: f64,
    max_lag: f64,
    directions: Option<Vec<(f64, f64)>>,
    tolerance: f64,
    bandwidth: Option<f64>,
    method: Option<&str>,
) -> PyResult<Vec<Vec<ExperimentalVariogram>>> {
    let bins = bins(lag, max_lag)?;
    let locs = locations(coords, method)?;
    let rows = crate::args::rows(realizations, "realizations")?;
    if rows.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("realizations must be finite"));
    }
    if let Some(row) = rows.first() {
        same_length(locs.len(), row.len(), "realizations")?;
    }
    let cones: Vec<Direction> = directions
        .iter()
        .flatten()
        .map(|&(azimuth, dip)| Direction {
            azimuth,
            dip,
            tolerance,
            bandwidth,
        })
        .collect();
    let rows: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let m = Estimator::Matheron;
    let out =
        experimental_realizations(locs.support(), &rows, &bins, m, &cones, false).map_err(err)?;
    Ok(out
        .into_iter()
        .map(|r| r.into_iter().map(ExperimentalVariogram).collect())
        .collect())
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
/// angles counter-clockwise from east). `estimator` takes the names
/// `experimental_variogram` does; γ is NaN where the correlogram is undefined.
/// `coords` and `values` are as in `experimental_variogram`.
#[pyfunction]
#[pyo3(signature = (coords, values, lag, max_lag, *, u=vec![1.0, 0.0, 0.0], v=vec![0.0, 1.0, 0.0], tolerance=22.5, steps=36, model="spherical", estimator="matheron"))]
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
    estimator: &str,
) -> PyResult<VariogramMap> {
    let (locs, values) = samples(coords, values, "values")?;
    let params = PlaneMapParams {
        bins: bins(lag, max_lag)?,
        tolerance,
        angle_steps: steps,
        model: self::model(model, None, None)?,
        estimator: self::estimator(estimator)?,
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

/// An nvar x nvar matrix of experimental variograms, one per direction in
/// each cell when `directed`; the lower triangle is dropped.
fn matrix(
    rows: &[Vec<Option<Bound<PyAny>>>],
    directed: bool,
) -> PyResult<Vec<Vec<Option<Vec<Experimental>>>>> {
    let n = rows.len();
    if n == 0 || rows.iter().any(|row| row.len() != n) {
        return Err(invalid("experimentals must be an nvar x nvar matrix"));
    }
    let cell = |c: &Bound<PyAny>| -> PyResult<Vec<Experimental>> {
        if directed {
            let many: Vec<PyRef<ExperimentalVariogram>> = c
                .extract()
                .map_err(|_| invalid("with directions, give one variogram per direction"))?;
            Ok(many.iter().map(|e| e.0.clone()).collect())
        } else {
            let one: PyRef<ExperimentalVariogram> = c
                .extract()
                .map_err(|_| invalid("without directions, give one variogram per pair"))?;
            Ok(vec![one.0.clone()])
        }
    };
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, c)| match c {
                    Some(c) if j >= i => cell(c).map(Some),
                    _ => Ok(None),
                })
                .collect()
        })
        .collect()
}

/// γ on a cube of lag vectors, and the principal axes of continuity.
///
/// Attributes
/// ----------
/// lags : ndarray
///     Cell centers along each axis, symmetric about 0.
/// gammas : ndarray
///     ``(n, n, n)`` γ indexed ``[ix, iy, iz]`` by lag vector; NaN where a cell holds no pairs.
/// counts : ndarray
///     Pair counts, shaped like `gammas`.
/// directions : ndarray
///     ``(m, 2)`` azimuth and dip of each direction read.
/// direction_ranges : ndarray
///     Range in each direction, NaN where too few pairs or γ stayed under half the sill.
/// rotation : tuple of float
///     Azimuth and dip of the major axis and rake of the semi-major, for ``Variogram(rotation=...)``.
/// axes : list of tuple
///     Azimuth and dip (dip ≥ 0) of the major, semi-major and minor axes.
/// ranges : tuple of float
///     Major, semi-major and minor ranges.
/// ratios : tuple of float
///     Semi-major and minor over major range, for ``Variogram(ratios=...)``.
#[pyclass(module = "ceres", name = "VariogramVolume", frozen)]
pub struct VariogramVolume {
    #[pyo3(get)]
    lags: Py<PyAny>,
    #[pyo3(get)]
    gammas: Py<PyAny>,
    #[pyo3(get)]
    counts: Py<PyAny>,
    #[pyo3(get)]
    directions: Py<PyAny>,
    #[pyo3(get)]
    direction_ranges: Py<PyAny>,
    #[pyo3(get)]
    rotation: (f64, f64, f64),
    #[pyo3(get)]
    axes: Vec<(f64, f64)>,
    #[pyo3(get)]
    ranges: (f64, f64, f64),
    #[pyo3(get)]
    ratios: (f64, f64),
}

/// Experimental variogram over a cube of lag vectors, and its principal axes of continuity.
///
/// Pairs are binned by lag vector into cubic cells of side `lag` out to `max_lag` on each axis. The axes are
/// read over `directions` near-uniform directions: `model` is fitted to the omnidirectional variogram, and in
/// each direction the lag where γ reaches the nugget plus half the sill traces an ellipsoid, whose axes give
/// the rotation and ratios. One structure fitted to every direction's variogram, stretched onto the major
/// axis, sets the major range.
///
/// Parameters
/// ----------
/// coords : array_like, PointSet or BlockModel
///     ``(n, 3)`` sample coordinates, or a container holding `values`.
/// values : array_like or str
///     Values, or a column name of `coords`.
/// lag : float
///     Cell size of the lag grid and lag width of each direction's variogram.
/// max_lag : float
///     Half-extent of the lag grid and longest lag read; at most 25 lags.
/// tolerance : float, default 20.0
///     Half-angle of the cone about each direction, in degrees.
/// directions : int, default 200
///     Directions read over the half-sphere; at least 9.
/// model : str, default "spherical"
///     Structure fitted to the variograms.
/// estimator : str, default "matheron"
///     As in `experimental_variogram`.
///
/// Returns
/// -------
/// VariogramVolume
///     The cube and the principal axes; `rotation` and `ratios` go straight into `Variogram`.
#[pyfunction]
#[pyo3(signature = (coords, values, lag, max_lag, *, tolerance=20.0, directions=200, model="spherical", estimator="matheron"))]
#[allow(clippy::too_many_arguments)]
fn variogram_volume(
    py: Python,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    lag: f64,
    max_lag: f64,
    tolerance: f64,
    directions: usize,
    model: &str,
    estimator: &str,
) -> PyResult<VariogramVolume> {
    let (locs, values) = samples(coords, values, "values")?;
    let params = VolumeParams {
        bins: bins(lag, max_lag)?,
        tolerance,
        directions,
        model: self::model(model, None, None)?,
        estimator: self::estimator(estimator)?,
        ..Default::default()
    };
    let v = py
        .detach(|| variogram::variogram_volume(&locs, &values, &params))
        .map_err(err)?;
    let n = v.lags.len();
    let cube = |values: Vec<f64>| {
        Array3::from_shape_vec((n, n, n), values)
            .expect("a cube of cells")
            .into_pyarray(py)
            .into_any()
            .unbind()
    };
    let a = &v.angles;
    let frame = ceres_core::rotation_matrix(a.azimuth, a.dip, a.rake);
    let axes = frame
        .row_iter()
        .map(|row| {
            let s = if row[2] > 0.0 { -1.0 } else { 1.0 };
            variogram::azimuth_dip((s * row[0], s * row[1], s * row[2]))
        })
        .collect();
    Ok(VariogramVolume {
        lags: array1(py, v.lags.clone()).into_any().unbind(),
        gammas: cube(v.gammas),
        counts: cube(v.counts.iter().map(|&c| c as f64).collect()),
        directions: array2(
            py,
            &v.directions
                .iter()
                .map(|&(az, dip)| vec![az, dip])
                .collect::<Vec<_>>(),
        )
        .into_any()
        .unbind(),
        direction_ranges: array1(py, v.ranges.iter().map(|r| r.unwrap_or(f64::NAN)).collect())
            .into_any()
            .unbind(),
        rotation: (a.azimuth, a.dip, a.rake),
        axes,
        ranges: (a.major, a.semi, a.minor),
        ratios: (a.semi / a.major, a.minor / a.major),
    })
}

/// Linear model of coregionalization: `nugget` and each structure's `sills`
/// are symmetric positive semi-definite `nvar × nvar` matrices, else
/// InvalidInput is raised. `rotation` and `ratios` are as in `Variogram`.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "Coregionalization", frozen)]
pub struct Coregionalization(pub CoreCoreg);

#[pymethods]
impl Coregionalization {
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
    #[pyo3(signature = (nugget, *, structures, rotation=(0.0, 0.0, 0.0), ratios=(1.0, 1.0)))]
    fn new(
        nugget: Vec<Vec<f64>>,
        structures: Vec<(String, f64, Vec<Vec<f64>>)>,
        rotation: (f64, f64, f64),
        ratios: (f64, f64),
    ) -> PyResult<Self> {
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
        let mut c = CoreCoreg::new(nugget, structures).map_err(err)?;
        c.anisotropy = anisotropy(rotation, ratios)?;
        Ok(Self(c))
    }

    /// Linear model of coregionalization fitted to the direct and cross
    /// experimental variograms of several variables together.
    ///
    /// The structures, their shapes and ranges, are shared by all variables;
    /// the nugget and each structure's sill matrix are kept positive
    /// semi-definite. Each pair's residuals are divided by the square root of
    /// the product of the two variables' largest direct γ, so variables in
    /// different units count alike. For given ranges the matrices are updated
    /// in turn by weighted least squares, negative eigenvalues clipped; the
    /// ranges start from `Variogram.fit` on the pooled scaled direct
    /// variograms and are refined, deterministically. With `directions`, the
    /// shared anisotropy is fitted too, starting from
    /// `Variogram.fit_directional` on the scaled direct variograms.
    ///
    /// Parameters
    /// ----------
    /// experimentals : VariogramSet or sequence of sequence of ExperimentalVariogram or None
    ///     A `VariogramSet`, whose directions are used, or an ``nvar x nvar``
    ///     matrix: entry ``[i][j]`` for ``i <= j`` is the direct
    ///     (``i == j``, required) or cross variogram of variables ``i`` and
    ///     ``j``; the lower triangle is ignored, and a missing cross pair
    ///     (None) is left to the positive semi-definite constraint. With
    ///     `directions`, each entry is a sequence of one per direction.
    /// model, weighting, ranges
    ///     As in `Variogram.fit`; ``count/gamma`` weighs a cross pair by the
    ///     model's ``sqrt(γii γjj)``.
    /// nugget : bool, default True
    ///     False fixes the nugget matrix at zero.
    /// directions : sequence of (float, float), optional
    ///     Azimuth and dip in degrees of each experimental variogram.
    /// rotation, ratios : sequence of (None, float or (float, float)), optional
    ///     Azimuth, dip and rake, and the range ratios, free, fixed or bounded
    ///     as in `Variogram.fit_directional`; need `directions`. Ranges are
    ///     then major-axis ranges.
    ///
    /// Returns
    /// -------
    /// Coregionalization
    #[staticmethod]
    #[pyo3(signature = (experimentals, model=Models::One("spherical".into()), *, weighting="count", nugget=true, ranges=None, directions=None, rotation=None, ratios=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        experimentals: &Bound<PyAny>,
        model: Models,
        weighting: &str,
        nugget: bool,
        ranges: Limits,
        directions: Option<Vec<(f64, f64)>>,
        rotation: Limits,
        ratios: Limits,
    ) -> PyResult<Self> {
        let (exps, directions) = match experimentals.cast::<VariogramSet>() {
            Ok(_) if directions.is_some() => {
                return Err(invalid("a VariogramSet carries its own directions"));
            }
            Ok(set) => (set.get().entries.clone(), set.get().directions.clone()),
            Err(_) => (
                matrix(&experimentals.extract::<Vec<_>>()?, directions.is_some())?,
                directions,
            ),
        };
        if directions.is_none() && (rotation.is_some() || ratios.is_some()) {
            return Err(invalid("rotation and ratios need directions"));
        }
        let nugget = (!nugget).then_some(Limit::Fixed(0.0));
        let spec = nested(model, nugget, None, ranges)?;
        let (rotation, ratios) = (
            per_structure(rotation, 3, "rotation")?,
            per_structure(ratios, 2, "ratios")?,
        );
        let aniso = AnisotropySpec {
            azimuth: rotation[0],
            dip: rotation[1],
            rake: rotation[2],
            semi: ratios[0],
            minor: ratios[1],
        };
        let geometry = directions.as_deref().map(|d| (d, &aniso));
        let fitted = fit_coregionalization(&exps, geometry, &spec, self::weighting(weighting)?)
            .map_err(err)?;
        Ok(Self(fitted.coregionalization))
    }

    #[getter]
    fn nvar(&self) -> usize {
        self.0.nvar
    }

    /// ``nvar x nvar`` nugget matrix.
    #[getter]
    fn nugget<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array2(py, &self.0.nugget).into_any()
    }

    /// ``(model, range, sills)`` per structure, `sills` its ``nvar x nvar``
    /// matrix.
    #[getter]
    fn structures<'py>(&self, py: Python<'py>) -> Vec<(&'static str, f64, Bound<'py, PyAny>)> {
        self.0
            .structures
            .iter()
            .map(|s| {
                (
                    model_name(s.model),
                    s.range,
                    array2(py, &s.sills).into_any(),
                )
            })
            .collect()
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
/// `categories` are integer codes from 0, or the column of `coords` holding
/// them.
#[pyfunction]
fn experimental_transiogram<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    categories: &Bound<PyAny>,
    lag: f64,
    max_lag: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    let categories = floats(
        &column(Some(coords), categories, "categories")?,
        "categories",
    )?
    .into_iter()
    .map(|c| match c >= 0.0 && c.fract() == 0.0 {
        true => Ok(c as usize),
        false => Err(invalid("categories must be non-negative integer codes")),
    })
    .collect::<PyResult<Vec<usize>>>()?;
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
#[pyo3(signature = (anamorphosis, variogram, size, *, discretization=(4, 4, 1)))]
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
#[pyo3(signature = (variogram, size, *, discretization=(4, 4, 1)))]
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
    m.add_class::<VariogramSet>()?;
    m.add_class::<VariogramMap>()?;
    m.add_class::<VariogramVolume>()?;
    m.add_class::<Coregionalization>()?;
    m.add_class::<Transiogram>()?;
    m.add_function(wrap_pyfunction!(experimental_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(experimental_variograms, m)?)?;
    m.add_function(wrap_pyfunction!(_realization_variograms, m)?)?;
    m.add_function(wrap_pyfunction!(variogram_map, m)?)?;
    m.add_function(wrap_pyfunction!(variogram_volume, m)?)?;
    m.add_function(wrap_pyfunction!(experimental_transiogram, m)?)?;
    m.add_function(wrap_pyfunction!(change_of_support, m)?)?;
    m.add_function(wrap_pyfunction!(block_correlation, m)?)?;
    Ok(())
}
