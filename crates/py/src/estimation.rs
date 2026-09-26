use estimation::{
    Discretization, DriftSpec, DualKriging, Estimate, HighGrade, InterpEstimate, InterpOptions,
    Kind, NeighborhoodStats, Sample, Search as CoreSearch, block_krige, by_pass, estimate_many,
    k_fold_at, krige, krige_bayesian, krige_factorial, krige_universal, leave_one_out_at,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde::{Deserialize, Serialize};
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
/// estimation and cross-validation alike. The threshold is always in data
/// units: simulators working on normal scores convert it through their
/// fitted transform, so it picks the same samples as in estimation. Estimators also take a sequence
/// of searches as passes: targets one leaves unestimated go to the next.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "Search", frozen, from_py_object)]
#[derive(Clone)]
pub struct Search(pub CoreSearch);

#[pymethods]
impl Search {
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
    fn kriging(&self) -> bool {
        !matches!(
            self,
            Method::InverseDistance(_)
                | Method::Nearest
                | Method::MovingAverage
                | Method::MovingMedian
                | Method::LocalLeastSquares(_)
        )
    }

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

type Used = (Estimate, NeighborhoodStats);

fn used(t: &Point, s: &[Sample], e: estimation::Result<Estimate>) -> estimation::Result<Used> {
    e.map(|e| {
        let near = estimation::neighborhood_stats(t, s, s.len(), f64::INFINITY, None);
        (e, near)
    })
}

fn diagnostics<'py>(
    py: Python<'py>,
    results: &[Option<(usize, Used)>],
    searches: &[CoreSearch],
) -> PyResult<Bound<'py, PyDict>> {
    let column = |f: &dyn Fn(usize, &Estimate, &NeighborhoodStats) -> f64| {
        array1(
            py,
            results
                .iter()
                .map(|r| r.as_ref().map_or(f64::NAN, |(p, (e, s))| f(*p, e, s)))
                .collect(),
        )
    };
    let d = PyDict::new(py);
    d.set_item("value", column(&|_, e, _| e.value))?;
    d.set_item("variance", column(&|_, e, _| e.variance))?;
    d.set_item("efficiency", column(&|_, e, _| e.efficiency()))?;
    d.set_item("slope", column(&|_, e, _| e.slope()))?;
    d.set_item("n_samples", column(&|_, e, _| e.n_used as f64))?;
    d.set_item("pass", column(&|p, _, _| (p + 1) as f64))?;
    d.set_item("n_holes", column(&|_, _, s| s.n_holes as f64))?;
    d.set_item("mean_distance", column(&|_, _, s| s.mean_dist_knn))?;
    d.set_item(
        "negative_weight_sum",
        column(&|_, e, _| e.negative_weight_sum()),
    )?;
    d.set_item("lagrange", column(&|_, e, _| e.lagrange))?;
    let full = |p: usize, s: &NeighborhoodStats| s.n_within >= searches[p].max_samples;
    d.set_item(
        "max_samples_reached",
        column(&|p, _, s| full(p, s) as u8 as f64),
    )?;
    Ok(d)
}

fn split<T>(passes: Vec<Option<(usize, T)>>) -> Vec<Option<T>> {
    passes.into_iter().map(|r| r.map(|(_, e)| e)).collect()
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
        if method.kriging() && variogram.is_none() {
            return Err(invalid("kriging needs a variogram"));
        }
        Ok(Self {
            method,
            variogram: variogram.map(|v| v.0),
            search: search.into_iter().map(|s| s.0).collect(),
            samples: None,
        })
    }

    /// `holes` (optional) tags samples by drill hole for `max_per_hole`;
    /// `error_variance` (optional) is each sample's measurement-error variance.
    #[pyo3(signature = (coords, values, holes=None, error_variance=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        error_variance: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (locs, values) = (points(coords)?, finite(values, "values")?);
        same_length(locs.len(), values.len(), "values")?;
        let holes = args::holes(holes, locs.len())?;
        let error = args::optional_finite(error_variance, "error_variance")?;
        if let Some(e) = &error {
            same_length(locs.len(), e.len(), "error_variance")?;
            if e.iter().any(|v| *v < 0.0) {
                return Err(invalid("error_variance must be >= 0"));
            }
            if !slf.method.kriging() {
                return Err(invalid("error_variance needs a kriging method"));
            }
        }
        let keep = distinct(slf.py(), &locs, holes.as_ref().map(|h| &h.0[..]))?;
        let samples = keep
            .into_iter()
            .map(|i| Sample {
                error_variance: error.as_ref().map_or(0.0, |e| e[i]),
                ..match &holes {
                    Some((_, ids)) => Sample::with_hole(locs[i], values[i], ids[i]),
                    None => Sample::new(locs[i], values[i]),
                }
            })
            .collect();
        slf.samples = Some(samples);
        Ok(slf)
    }

    /// Estimates (NaN where too few neighbours); with `return_variance`, also
    /// the kriging variance, and with `diagnostics` a dict adding kriging
    /// efficiency, slope of regression, samples used, the search pass
    /// (1-based) that filled each target, holes used, mean distance to the
    /// samples used, sum of negative weights, Lagrange multiplier and whether
    /// the search hit `max_samples`. `anisotropy` (a
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
                            used(t, s, self.method.run(t, s, vg))
                        })),
                        Some(local) => estimation::lva::estimate_many_local(
                            &at,
                            &local.at(&at),
                            samples,
                            search,
                            &base,
                            |t, s, v| used(t, s, self.method.run(t, s, Some(v))),
                        ),
                    }
                })
            })
            .map_err(invalid)?;
        if diagnostics {
            return Ok(self::diagnostics(py, &passes, &self.search)?.into_any());
        }
        let results: Vec<_> = split(passes).into_iter().map(|r| r.map(|r| r.0)).collect();
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

    /// Leave-one-out, or with `folds` k-fold, estimates and variances at the
    /// fitted samples; folds keep holes whole (see `k_fold_at`).
    #[pyo3(signature = (folds=None))]
    fn cross_validate<'py>(
        &self,
        py: Python<'py>,
        folds: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let samples = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("estimator is not fitted; call fit first"))?;
        let vg = self.variogram.as_ref();
        let run = |t: &Point, s: &[Sample]| self.method.run(t, s, vg);
        let passes = py
            .detach(|| {
                by_pass(
                    samples.len(),
                    &self.search,
                    |search, remaining| match folds {
                        None => Ok(leave_one_out_at(remaining, samples, search, vg, run)),
                        Some(k) => k_fold_at(k, remaining, samples, search, vg, run),
                    },
                )
            })
            .map_err(invalid)?;
        outputs(py, &split(passes), true)
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
