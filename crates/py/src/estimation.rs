use estimation::{
    Discretization, DriftSpec, DualKriging, Estimate, HighGrade, InterpEstimate, InterpOptions,
    Kind, Sample, Search as CoreSearch, block_krige, by_pass, estimate_many, krige, krige_bayesian,
    krige_factorial, krige_universal, leave_one_out_at,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use variogram::Variogram as CoreVariogram;

use crate::args::{self, Point, array1, distinct, finite, pick, points, same_length, triple};
use crate::containers::{PyBlockModel, PyPointSet};
use crate::invalid;
use crate::variogram::Variogram;

/// Neighbourhood: `radius` is in metres along the major axis of the
/// search ellipsoid (`rotation` azimuth, dip, rake and `ratios` semi/major,
/// minor/major), or of the variogram's anisotropy when no ellipsoid is given.
/// `high_grade` `(threshold, radius)` lets samples above `threshold` inform
/// only targets within `radius`, measured in the same ellipsoid, in
/// estimation and cross-validation alike. Estimators also take a sequence
/// of searches as passes: targets one leaves unestimated go to the next.
#[pyclass(module = "ceres", name = "Search", frozen, from_py_object)]
#[derive(Clone)]
pub struct Search(pub CoreSearch);

#[pymethods]
impl Search {
    #[new]
    #[pyo3(signature = (radius, max_samples=16, min_samples=1, octant=false, max_per_hole=None, rotation=None, ratios=None, high_grade=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        radius: f64,
        max_samples: usize,
        min_samples: usize,
        octant: bool,
        max_per_hole: Option<usize>,
        rotation: Option<(f64, f64, f64)>,
        ratios: Option<(f64, f64)>,
        high_grade: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        if radius.is_nan() || radius <= 0.0 || max_samples == 0 || min_samples > max_samples {
            return Err(invalid(
                "need radius > 0 and 1 <= min_samples <= max_samples",
            ));
        }
        if let Some((threshold, radius)) = high_grade
            && (threshold.is_nan() || radius.is_nan() || radius < 0.0)
        {
            return Err(invalid("high_grade needs a threshold and a radius >= 0"));
        }
        Ok(Self(CoreSearch {
            min_samples,
            max_samples,
            radius,
            max_per_hole,
            octant,
            anisotropy: match (rotation, ratios) {
                (None, None) => None,
                (rotation, ratios) => crate::variogram::anisotropy(
                    rotation.unwrap_or((0.0, 0.0, 0.0)),
                    ratios.unwrap_or((1.0, 1.0)),
                )?,
            },
            high_grade: high_grade.map(|(threshold, radius)| HighGrade { threshold, radius }),
        }))
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.0.radius
    }

    #[getter]
    fn max_samples(&self) -> usize {
        self.0.max_samples
    }

    #[getter]
    fn min_samples(&self) -> usize {
        self.0.min_samples
    }

    fn __repr__(&self) -> String {
        format!(
            "Search(radius={}, max_samples={}, min_samples={}, octant={})",
            self.0.radius, self.0.max_samples, self.0.min_samples, self.0.octant
        )
    }
}

enum Method {
    Kriging(Kind),
    Universal(usize),
    Factorial {
        nugget: bool,
        structures: Vec<usize>,
    },
    Block {
        size: Point,
        disc: Discretization,
    },
    Bayesian {
        degree: usize,
        mean: Vec<f64>,
        var: Vec<f64>,
    },
    InverseDistance(f64),
    Nearest,
    MovingAverage,
    MovingMedian,
    LocalLeastSquares(usize),
}

fn interp(e: estimation::Result<InterpEstimate>) -> estimation::Result<Estimate> {
    e.map(|e| Estimate {
        value: e.value,
        variance: e.stdev.map_or(f64::NAN, |s| s * s),
        n_used: e.n_used,
        weights: vec![],
        lagrange: f64::NAN,
        support_variance: f64::NAN,
    })
}

impl Method {
    fn run(
        &self,
        t: &Point,
        s: &[Sample],
        vg: Option<&CoreVariogram>,
    ) -> estimation::Result<Estimate> {
        let opts = InterpOptions {
            dmax: f64::INFINITY,
            anisotropy: vg.and_then(|v| v.anisotropy.clone()),
            min_samples: 1,
        };
        let vg = || vg.expect("kriging methods carry a variogram");
        let drift = |degree| {
            let locs: Vec<Point> = s.iter().map(|s| s.loc).collect();
            DriftSpec::polynomial(&locs, t, degree)
        };
        match self {
            Method::Kriging(kind) => krige(*kind, t, s, vg()),
            Method::Universal(degree) => krige_universal(t, s, &drift(*degree), vg(), None),
            Method::Factorial { nugget, structures } => {
                krige_factorial(t, s, vg(), *nugget, structures)
            }
            Method::Block { size, disc } => block_krige(t, size, s, disc, vg()),
            Method::Bayesian { degree, mean, var } => {
                krige_bayesian(t, s, &drift(*degree), vg(), mean, var)
            }
            Method::InverseDistance(power) => {
                interp(estimation::inverse_distance(t, s, *power, &opts))
            }
            Method::Nearest => interp(estimation::simple_interp::nearest(t, s, &opts)),
            Method::MovingAverage => interp(estimation::moving_average(t, s, &opts)),
            Method::MovingMedian => interp(estimation::moving_median(t, s, &opts)),
            Method::LocalLeastSquares(degree) => {
                interp(estimation::local_least_squares(t, s, *degree, &opts))
            }
        }
    }
}

/// Targets from a PointSet, a BlockModel (centroids) or an `(n, 2|3)` array.
pub fn targets(obj: &Bound<PyAny>) -> PyResult<Vec<Point>> {
    let rows = if let Ok(p) = obj.cast::<PyPointSet>() {
        p.get().0.coords().to_vec()
    } else if let Ok(b) = obj.cast::<PyBlockModel>() {
        b.get().0.centroids()
    } else {
        return points(obj);
    };
    Ok(rows.into_iter().map(|[x, y, z]| (x, y, z)).collect())
}

pub fn outputs<'py>(
    py: Python<'py>,
    results: &[Option<Estimate>],
    return_variance: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let value = array1(
        py,
        results
            .iter()
            .map(|e| e.as_ref().map_or(f64::NAN, |e| e.value))
            .collect(),
    );
    if !return_variance {
        return Ok(value.into_any());
    }
    let variance = array1(
        py,
        results
            .iter()
            .map(|e| e.as_ref().map_or(f64::NAN, |e| e.variance))
            .collect(),
    );
    Ok(PyTuple::new(py, [value, variance])?.into_any())
}

fn diagnostics<'py>(
    py: Python<'py>,
    results: &[Option<Estimate>],
    passes: &[Option<usize>],
) -> PyResult<Bound<'py, PyDict>> {
    let column = |f: fn(&Estimate) -> f64| {
        array1(
            py,
            results
                .iter()
                .map(|e| e.as_ref().map_or(f64::NAN, f))
                .collect(),
        )
    };
    let d = PyDict::new(py);
    d.set_item("value", column(|e| e.value))?;
    d.set_item("variance", column(|e| e.variance))?;
    d.set_item("efficiency", column(Estimate::efficiency))?;
    d.set_item("slope", column(Estimate::slope))?;
    d.set_item("n_samples", column(|e| e.n_used as f64))?;
    let pass = passes
        .iter()
        .map(|p| p.map_or(f64::NAN, |p| (p + 1) as f64));
    d.set_item("pass", array1(py, pass.collect()))?;
    Ok(d)
}

fn split(passes: Vec<Option<(usize, Estimate)>>) -> (Vec<Option<usize>>, Vec<Option<Estimate>>) {
    passes
        .into_iter()
        .map(|r| r.map_or((None, None), |(p, e)| (Some(p), Some(e))))
        .unzip()
}

/// Shared engine behind the estimator classes in `ceres.estimation`.
#[pyclass(module = "ceres", name = "_Estimator")]
pub struct Estimator {
    method: Method,
    variogram: Option<CoreVariogram>,
    search: Vec<CoreSearch>,
    samples: Option<Vec<Sample>>,
}

#[pymethods]
impl Estimator {
    #[new]
    #[pyo3(signature = (method, search, variogram=None, **options))]
    fn new(
        method: &str,
        search: &Bound<PyAny>,
        variogram: Option<Variogram>,
        options: Option<&Bound<PyDict>>,
    ) -> PyResult<Self> {
        let search: Vec<Search> = match search.extract::<Search>() {
            Ok(s) => vec![s],
            Err(_) => search
                .extract()
                .map_err(|_| invalid("search must be a Search or a sequence of Search"))?,
        };
        if search.is_empty() {
            return Err(invalid("search needs at least one Search"));
        }
        let get = |key: &str| -> PyResult<Option<Bound<PyAny>>> {
            Ok(match options {
                Some(o) => o.get_item(key)?,
                None => None,
            })
        };
        let float = |key: &str, default: f64| -> PyResult<f64> {
            get(key)?.map_or(Ok(default), |v| v.extract())
        };
        let int = |key: &str, default: usize| -> PyResult<usize> {
            get(key)?.map_or(Ok(default), |v| v.extract())
        };
        let method = match method {
            "ordinary" => Method::Kriging(Kind::Ordinary),
            "simple" => Method::Kriging(Kind::Simple {
                mean: float("mean", 0.0)?,
            }),
            "indicator" => Method::Kriging(Kind::Indicator {
                threshold: float("threshold", 0.0)?,
            }),
            "universal" => Method::Universal(int("degree", 1)?),
            "factorial" => Method::Factorial {
                nugget: get("nugget")?.map_or(Ok(false), |v| v.extract())?,
                structures: get("structures")?.map_or(Ok(vec![]), |v| v.extract())?,
            },
            "block" => {
                let size: Vec<f64> = get("size")?
                    .ok_or_else(|| invalid("block kriging needs a size"))?
                    .extract()?;
                let (nx, ny, nz): (usize, usize, usize) =
                    get("discretization")?.map_or(Ok((4, 4, 1)), |v| v.extract())?;
                Method::Block {
                    size: triple(size, 1.0, "size")?,
                    disc: Discretization { nx, ny, nz },
                }
            }
            "bayesian" => Method::Bayesian {
                degree: int("degree", 0)?,
                mean: get("prior_mean")?
                    .ok_or_else(|| invalid("bayesian kriging needs prior_mean"))?
                    .extract()?,
                var: get("prior_variance")?
                    .ok_or_else(|| invalid("bayesian kriging needs prior_variance"))?
                    .extract()?,
            },
            "inverse_distance" => Method::InverseDistance(float("power", 2.0)?),
            "nearest" => Method::Nearest,
            "moving_average" => Method::MovingAverage,
            "moving_median" => Method::MovingMedian,
            "local_least_squares" => Method::LocalLeastSquares(int("degree", 1)?),
            other => return Err(invalid(format!("unknown method {other:?}"))),
        };
        let needs_variogram = matches!(
            method,
            Method::Kriging(_)
                | Method::Universal(_)
                | Method::Factorial { .. }
                | Method::Block { .. }
                | Method::Bayesian { .. }
        );
        if needs_variogram && variogram.is_none() {
            return Err(invalid("kriging needs a variogram"));
        }
        Ok(Self {
            method,
            variogram: variogram.map(|v| v.0),
            search: search.into_iter().map(|s| s.0).collect(),
            samples: None,
        })
    }

    /// `holes` (optional) tags samples by drill hole for `max_per_hole`.
    #[pyo3(signature = (coords, values, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (locs, values) = (points(coords)?, finite(values, "values")?);
        same_length(locs.len(), values.len(), "values")?;
        let holes = args::holes(holes, locs.len())?;
        let keep = distinct(slf.py(), &locs, holes.as_ref().map(|h| &h.0[..]))?;
        let samples = keep
            .into_iter()
            .map(|i| match &holes {
                Some((_, ids)) => Sample::with_hole(locs[i], values[i], ids[i]),
                None => Sample::new(locs[i], values[i]),
            })
            .collect();
        slf.samples = Some(samples);
        Ok(slf)
    }

    /// Estimates (NaN where too few neighbours); with `return_variance`, also
    /// the kriging variance, and with `diagnostics` a dict adding kriging
    /// efficiency, slope of regression, samples used and the search pass
    /// (1-based) that filled each target. `anisotropy` (a
    /// LocalAnisotropy) gives each target its own variogram and search
    /// orientation, taken from the nearest location.
    #[pyo3(signature = (targets, return_variance=false, anisotropy=None, diagnostics=false))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        return_variance: bool,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        diagnostics: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let samples = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("estimator is not fitted; call fit first"))?;
        let targets = self::targets(targets)?;
        let vg = self.variogram.as_ref();
        let local = anisotropy.map(|field| field.at_targets(&targets));
        let base = self.variogram.clone().unwrap_or(CoreVariogram {
            nugget: 0.0,
            structures: vec![],
            anisotropy: None,
        });
        let passes = py
            .detach(|| {
                by_pass(targets.len(), &self.search, |search, remaining| {
                    let at = pick(&targets, remaining);
                    match &local {
                        None => Ok(estimate_many(&at, samples, search, vg, |t, s| {
                            self.method.run(t, s, vg)
                        })),
                        Some(local) => estimation::lva::estimate_many_local(
                            &at,
                            &local.at(&at),
                            samples,
                            search,
                            &base,
                            |t, s, v| self.method.run(t, s, Some(v)),
                        ),
                    }
                })
            })
            .map_err(invalid)?;
        let (passes, results) = split(passes);
        if diagnostics {
            return Ok(self::diagnostics(py, &results, &passes)?.into_any());
        }
        outputs(py, &results, return_variance)
    }

    /// Values of the fitted samples, after dropping shared locations.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let samples = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("estimator is not fitted; call fit first"))?;
        Ok(array1(py, samples.iter().map(|s| s.value).collect()).into_any())
    }

    /// Leave-one-out estimates and variances at the fitted samples.
    fn cross_validate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let samples = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("estimator is not fitted; call fit first"))?;
        let vg = self.variogram.as_ref();
        let passes = py
            .detach(|| {
                by_pass(samples.len(), &self.search, |search, remaining| {
                    Ok(leave_one_out_at(remaining, samples, search, vg, |t, s| {
                        self.method.run(t, s, vg)
                    }))
                })
            })
            .map_err(invalid)?;
        outputs(py, &split(passes).1, true)
    }
}

/// Global dual kriging with a polynomial drift of `degree` (no search).
#[pyclass(module = "ceres", name = "DualKriging")]
pub struct Dual {
    variogram: CoreVariogram,
    degree: usize,
    samples: Option<Vec<Sample>>,
}

#[pymethods]
impl Dual {
    #[new]
    #[pyo3(signature = (variogram, degree=0))]
    fn new(variogram: Variogram, degree: usize) -> Self {
        Self {
            variogram: variogram.0,
            degree,
            samples: None,
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (locs, values) = (points(coords)?, finite(values, "values")?);
        same_length(locs.len(), values.len(), "values")?;
        let keep = distinct(slf.py(), &locs, None)?;
        slf.samples = Some(
            keep.into_iter()
                .map(|i| Sample::new(locs[i], values[i]))
                .collect(),
        );
        Ok(slf)
    }

    fn predict<'py>(&self, py: Python<'py>, targets: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let samples = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("DualKriging is not fitted; call fit first"))?;
        let dual = DualKriging::new(samples, &self.variogram, self.degree).map_err(invalid)?;
        let values = self::targets(targets)?
            .iter()
            .map(|t| dual.estimate(t))
            .collect();
        Ok(array1(py, values).into_any())
    }
}

/// Distances and value statistics of the `k` nearest samples around targets;
/// `n_holes` counts distinct `holes` within `radius`.
#[pyfunction]
#[pyo3(signature = (targets, coords, values, k=8, radius=f64::INFINITY, variogram=None, holes=None))]
#[allow(clippy::too_many_arguments)]
fn neighborhood_stats<'py>(
    py: Python<'py>,
    targets: &Bound<PyAny>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    k: usize,
    radius: f64,
    variogram: Option<Variogram>,
    holes: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (locs, values) = (points(coords)?, finite(values, "values")?);
    same_length(locs.len(), values.len(), "values")?;
    let holes = args::holes(holes, locs.len())?;
    let samples: Vec<Sample> = (0..locs.len())
        .map(|i| match &holes {
            Some((_, ids)) => Sample::with_hole(locs[i], values[i], ids[i]),
            None => Sample::new(locs[i], values[i]),
        })
        .collect();
    let aniso = variogram.and_then(|v| v.0.anisotropy);
    let stats: Vec<_> = self::targets(targets)?
        .iter()
        .map(|t| estimation::neighborhood_stats(t, &samples, k, radius, aniso.as_ref()))
        .collect();
    let d = PyDict::new(py);
    macro_rules! column {
        ($($f:ident),*) => {$(
            d.set_item(stringify!($f), array1(py, stats.iter().map(|s| s.$f as f64).collect()))?;
        )*};
    }
    column!(
        n_within,
        n_holes,
        nearest_dist,
        mean_dist_knn,
        max_dist_knn,
        value_mean,
        value_var,
        value_min,
        value_max,
        value_idw
    );
    Ok(d)
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Search>()?;
    m.add_class::<Estimator>()?;
    m.add_class::<Dual>()?;
    m.add_function(wrap_pyfunction!(neighborhood_stats, m)?)?;
    Ok(())
}
