use estimation::{
    Discretization, DriftSpec, DualKriging, Estimate, HighGrade, InterpEstimate, InterpOptions,
    Kind, NeighborhoodStats, Sample, Search as CoreSearch, Soft, SoftPair, block_krige, by_pass,
    estimate_many, k_fold_at, krige, krige_bayesian, krige_factorial, krige_universal,
    leave_one_out_at,
};
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};
use pyo3::{IntoPyObjectExt, PyClass};
use serde::{Deserialize, Serialize};
use variogram::Variogram as CoreVariogram;

use crate::args::{
    self, Point, array1, column, distinct, finite, pick, points, same_length, triple,
};
use crate::containers::{PyBlockModel, PyPointSet};
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};
use crate::table::Table;
use crate::variogram::Variogram;

/// Neighborhood: `radius` is in meters along the major axis of the
/// search ellipsoid (`rotation` azimuth, dip, rake and `ratios` semi/major,
/// minor/major), or of the variogram's anisotropy when no ellipsoid is given.
/// `high_grade` `(threshold, radius)` lets samples above `threshold` inform
/// only targets within `radius`, measured in the same ellipsoid, in
/// estimation and cross-validation alike. The threshold is always in data
/// units: simulators compare it with the data values and the simulated
/// values of nodes, so it picks the same samples as in estimation. Estimators also take a sequence
/// of searches as passes: targets one leaves unestimated go to the next.
/// `soft` lets samples of another domain inform a target strictly within a
/// distance in the same ellipsoid: one distance for every pair of domains,
/// or a dict `{(target_domain, sample_domain): distance}`, one way; pairs not
/// listed are hard.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "Search", frozen, from_py_object)]
#[derive(Clone)]
pub struct Search {
    #[serde(flatten)]
    pub core: CoreSearch,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    soft: Option<Soft<Label>>,
}

pub use crate::args::Label;

fn label(obj: &Bound<PyAny>) -> PyResult<Label> {
    args::label(obj)?.ok_or_else(|| {
        invalid(format!(
            "domain labels must be strings, finite numbers or booleans, not {obj}"
        ))
    })
}

fn key(label: &Label) -> String {
    label.to_string()
}

/// `label` as the Python value it was given as.
pub fn py_label<'py>(py: Python<'py>, label: &Label) -> PyResult<Bound<'py, PyAny>> {
    match label {
        Label::String(s) => s.into_bound_py_any(py),
        Label::Bool(b) => b.into_bound_py_any(py),
        Label::Number(n) => match n.as_i64() {
            Some(i) => i.into_bound_py_any(py),
            None => n.as_f64().into_bound_py_any(py),
        },
        other => Err(invalid(format!("unexpected domain label {other}"))),
    }
}

/// Labels of a sequence, or of a single label repeated `n` times.
pub fn labels(obj: &Bound<PyAny>, n: Option<usize>) -> PyResult<Vec<Label>> {
    let single = obj.is_instance_of::<PyString>() || obj.try_iter().is_err();
    if single && let Some(n) = n {
        return Ok(vec![label(obj)?; n]);
    }
    let labels = obj
        .try_iter()
        .map_err(|_| invalid("domains must be a sequence of labels"))?
        .map(|item| label(&item?))
        .collect::<PyResult<Vec<_>>>()?;
    if let Some(n) = n {
        same_length(n, labels.len(), "domains")?;
    }
    Ok(labels)
}

impl Search {
    /// The parameters of a search that has no soft boundaries, for `what`.
    pub fn plain(self, what: &str) -> PyResult<CoreSearch> {
        match self.soft {
            Some(_) => Err(invalid(format!(
                "{what} does not take domains; Search.soft works with SGS, TurningBands and the kriging, \
                 inverse-distance, nearest-neighbor and interpolation estimators"
            ))),
            None => Ok(self.core),
        }
    }

    /// The core search with `soft` by domain code: the index of its labels
    /// in `domains`, the fitted ones.
    pub fn resolve(&self, domains: Option<&[Label]>) -> PyResult<CoreSearch> {
        let soft = match (&self.soft, domains) {
            (None, _) => None,
            (Some(_), None) => return Err(invalid("Search.soft needs domains at fit")),
            (Some(Soft::All(d)), Some(_)) => Some(Soft::All(*d)),
            (Some(Soft::Pairs(pairs)), Some(domains)) => {
                let code = |l: &Label| {
                    domains
                        .iter()
                        .position(|d| key(d) == key(l))
                        .map(|c| c as u32)
                        .ok_or_else(|| {
                            invalid(format!(
                                "Search.soft names domain {l}, which has no samples"
                            ))
                        })
                };
                Some(Soft::Pairs(
                    pairs
                        .iter()
                        .map(|p| {
                            Ok(SoftPair {
                                target: code(&p.target)?,
                                sample: code(&p.sample)?,
                                distance: p.distance,
                            })
                        })
                        .collect::<PyResult<_>>()?,
                ))
            }
        };
        Ok(CoreSearch {
            soft,
            ..self.core.clone()
        })
    }
}

fn soft(obj: &Bound<PyAny>) -> PyResult<Soft<Label>> {
    let distance = |d: f64| {
        if d.is_nan() || d < 0.0 {
            Err(invalid("soft distances must be >= 0"))
        } else {
            Ok(d)
        }
    };
    if let Ok(d) = obj.extract::<f64>() {
        return Ok(Soft::All(distance(d)?));
    }
    let dict = obj.cast::<PyDict>().map_err(|_| {
        invalid("soft must be a distance or a dict {(target_domain, sample_domain): distance}")
    })?;
    let mut pairs = Vec::with_capacity(dict.len());
    for (k, d) in dict.iter() {
        let (t, s): (Bound<PyAny>, Bound<PyAny>) = k
            .extract()
            .map_err(|_| invalid("soft keys must be (target_domain, sample_domain) pairs"))?;
        pairs.push(SoftPair {
            target: label(&t)?,
            sample: label(&s)?,
            distance: distance(d.extract()?)?,
        });
    }
    Ok(Soft::Pairs(pairs))
}

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
    #[pyo3(signature = (radius, max_samples=16, min_samples=1, octant=false, max_per_hole=None, rotation=None, ratios=None, high_grade=None, soft=None))]
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
        soft: Option<&Bound<PyAny>>,
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
        Ok(Self {
            soft: soft.map(self::soft).transpose()?,
            core: CoreSearch {
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
                soft: None,
            },
        })
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.core.radius
    }

    #[getter]
    fn max_samples(&self) -> usize {
        self.core.max_samples
    }

    #[getter]
    fn min_samples(&self) -> usize {
        self.core.min_samples
    }

    #[getter]
    fn octant(&self) -> bool {
        self.core.octant
    }

    #[getter]
    fn max_per_hole(&self) -> Option<usize> {
        self.core.max_per_hole
    }

    #[getter]
    fn high_grade(&self) -> Option<(f64, f64)> {
        self.core.high_grade.map(|h| (h.threshold, h.radius))
    }

    /// Azimuth, dip and rake of the search ellipsoid; None when the search
    /// follows the variogram's anisotropy.
    #[getter]
    fn rotation(&self) -> Option<(f64, f64, f64)> {
        let a = &self.core.anisotropy.as_ref()?.angles;
        Some((a.azimuth, a.dip, a.rake))
    }

    /// The soft distance, or the dict of distances by domain pair; None
    /// when domain boundaries are hard.
    #[getter]
    fn soft<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let label = |l: &Label| py_label(py, l);
        let Some(soft) = &self.soft else {
            return Ok(None);
        };
        Ok(Some(match soft {
            Soft::All(d) => d.into_bound_py_any(py)?,
            Soft::Pairs(pairs) => {
                let d = PyDict::new(py);
                for p in pairs {
                    d.set_item((label(&p.target)?, label(&p.sample)?), p.distance)?;
                }
                d.into_any()
            }
        }))
    }

    /// Semi-major/major and minor/major ratios; None as for `rotation`.
    #[getter]
    fn ratios(&self) -> Option<(f64, f64)> {
        let a = &self.core.anisotropy.as_ref()?.angles;
        Some((a.semi, a.minor))
    }

    fn __repr__(&self) -> String {
        format!(
            "Search(radius={}, max_samples={}, min_samples={}, octant={})",
            self.core.radius, self.core.max_samples, self.core.min_samples, self.core.octant
        )
    }
}

#[derive(Clone, Serialize, Deserialize)]
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

/// The estimate, statistics of the samples used and their domains.
type Used = (Estimate, NeighborhoodStats, Vec<Option<u32>>);

fn used(t: &Point, s: &[Sample], e: estimation::Result<Estimate>) -> estimation::Result<Used> {
    e.map(|e| {
        let near = estimation::neighborhood_stats(t, s, s.len(), f64::INFINITY, None);
        (e, near, s.iter().map(|s| s.domain).collect())
    })
}

fn diagnostics(
    results: &[Option<(usize, Used)>],
    searches: &[CoreSearch],
    domains: Option<&[Option<u32>]>,
) -> PyResult<Table> {
    let floats = |values: Vec<f64>| -> ArrayRef { Arc::new(Float64Array::from(values)) };
    let column = |f: &dyn Fn(usize, &Estimate, &NeighborhoodStats) -> f64| {
        floats(
            results
                .iter()
                .map(|r| r.as_ref().map_or(f64::NAN, |(p, (e, s, _))| f(*p, e, s)))
                .collect(),
        )
    };
    let other = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            r.as_ref().map_or(f64::NAN, |(_, (_, _, used))| {
                let own = domains.and_then(|d| d[i]);
                used.iter().filter(|&&d| own.is_some() && d != own).count() as f64
            })
        })
        .collect();
    let full = |p: usize, s: &NeighborhoodStats| s.n_within >= searches[p].max_samples;
    let columns = [
        ("value", column(&|_, e, _| e.value)),
        ("variance", column(&|_, e, _| e.variance)),
        ("efficiency", column(&|_, e, _| e.efficiency())),
        ("slope", column(&|_, e, _| e.slope())),
        ("n_samples", column(&|_, e, _| e.n_used as f64)),
        ("pass", column(&|p, _, _| (p + 1) as f64)),
        ("n_holes", column(&|_, _, s| s.n_holes as f64)),
        ("n_other_domain", floats(other)),
        ("mean_distance", column(&|_, _, s| s.mean_dist_knn)),
        (
            "negative_weight_sum",
            column(&|_, e, _| e.negative_weight_sum()),
        ),
        ("lagrange", column(&|_, e, _| e.lagrange)),
        ("support_variance", column(&|_, e, _| e.support_variance)),
        (
            "estimate_variance",
            column(&|_, e, _| e.estimate_variance()),
        ),
        (
            "max_samples_reached",
            column(&|p, _, s| full(p, s) as u8 as f64),
        ),
    ];
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

fn split<T>(passes: Vec<Option<(usize, T)>>) -> Vec<Option<T>> {
    passes.into_iter().map(|r| r.map(|(_, e)| e)).collect()
}

/// `x`, `y`, `z`, `value`, `hole` and `domain` codes (null when untagged)
/// and `error_variance`.
pub fn sample_columns(samples: &[Sample]) -> Columns {
    let mut columns = persist::point_columns(samples.iter().map(|s| s.loc));
    columns.push(persist::column("value", samples.iter().map(|s| s.value)));
    let holes = samples.iter().map(|s| s.hole.map(f64::from)).collect();
    columns.push(("hole".into(), holes));
    let error = samples.iter().map(|s| s.error_variance);
    columns.push(persist::column("error_variance", error));
    let domains = samples.iter().map(|s| s.domain.map(f64::from)).collect();
    columns.push(("domain".into(), domains));
    columns
}

pub fn samples_from(found: &Found) -> PyResult<Vec<Sample>> {
    let (locs, values) = (found.points()?, found.values("value")?);
    let (holes, error) = (found.optional("hole")?, found.values("error_variance")?);
    same_length(locs.len(), values.len(), "value")?;
    same_length(locs.len(), holes.len(), "hole")?;
    same_length(locs.len(), error.len(), "error_variance")?;
    let domains = found.optional("domain")?;
    same_length(locs.len(), domains.len(), "domain")?;
    (0..locs.len())
        .map(|i| {
            let hole = holes[i].map(persist::index).transpose()?;
            let domain = domains[i].map(persist::index).transpose()?;
            Ok(Sample {
                loc: locs[i],
                value: values[i],
                hole: hole.map(|h| h as u32),
                error_variance: error[i],
                domain: domain.map(|d| d as u32),
            })
        })
        .collect()
}

impl Tabular for Estimator {
    fn columns(&self) -> Option<Columns> {
        self.samples.as_deref().map(sample_columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.samples = Some(samples_from(&columns)?);
        Ok(())
    }
}

impl Tabular for Dual {
    fn columns(&self) -> Option<Columns> {
        self.samples.as_deref().map(sample_columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.samples = Some(samples_from(&columns)?);
        Ok(())
    }
}

/// A Search, or a sequence of them as passes.
pub fn searches(obj: &Bound<PyAny>) -> PyResult<Vec<Search>> {
    let search: Vec<Search> = match obj.extract::<Search>() {
        Ok(s) => vec![s],
        Err(_) => obj
            .extract()
            .map_err(|_| invalid("search must be a Search or a sequence of Search"))?,
    };
    if search.is_empty() {
        return Err(invalid("search needs at least one Search"));
    }
    Ok(search)
}

/// Shared engine behind the estimator classes in `ceres.estimation`.
#[derive(Clone, Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "_Estimator", skip_from_py_object)]
pub struct Estimator {
    method: Method,
    variogram: Option<CoreVariogram>,
    search: Vec<Search>,
    /// Labels of the fitted domains, indexed by code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domains: Option<Vec<Label>>,
    #[serde(skip)]
    samples: Option<Vec<Sample>>,
}

impl Estimator {
    fn fitted(&self) -> PyResult<&[Sample]> {
        self.samples
            .as_deref()
            .ok_or_else(|| invalid("estimator is not fitted; call fit first"))
    }

    /// The searches with soft boundaries by domain code.
    fn passes(&self) -> PyResult<Vec<CoreSearch>> {
        let domains = self.domains.as_deref();
        self.search.iter().map(|s| s.resolve(domains)).collect()
    }
}

/// The distinct labels of `obj`, one label or one per sample, and each
/// sample's code: the index of its label.
pub fn fit_codes(obj: &Bound<PyAny>, n: usize) -> PyResult<(Vec<Label>, Vec<u32>)> {
    let (mut fitted, mut code) = (vec![], std::collections::HashMap::new());
    let codes = labels(obj, Some(n))?
        .into_iter()
        .map(|l| {
            *code.entry(key(&l)).or_insert_with(|| {
                fitted.push(l);
                fitted.len() as u32 - 1
            })
        })
        .collect();
    Ok((fitted, codes))
}

/// Codes of `n` target labels in the `fitted` ones, for `method`; None for
/// a domain without samples.
pub fn codes(
    fitted: Option<&[Label]>,
    obj: Option<&Bound<PyAny>>,
    n: usize,
    method: &str,
) -> PyResult<Option<Vec<Option<u32>>>> {
    match (fitted, obj) {
        (None, None) => Ok(None),
        (Some(_), None) => Err(invalid(format!(
            "fitted with domains; {method} needs domains too"
        ))),
        (None, Some(_)) => Err(invalid(format!(
            "fitted without domains; {method} takes none"
        ))),
        (Some(fitted), Some(obj)) => {
            let code: std::collections::HashMap<String, u32> = fitted
                .iter()
                .enumerate()
                .map(|(c, l)| (key(l), c as u32))
                .collect();
            let labels = labels(obj, Some(n))?;
            Ok(Some(
                labels.iter().map(|l| code.get(&key(l)).copied()).collect(),
            ))
        }
    }
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
        let search = searches(search)?;
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
            search,
            domains: None,
            samples: None,
        })
    }

    /// `holes` (optional) tags samples by drill hole for `max_per_hole`;
    /// `error_variance` (optional) is each sample's measurement-error variance;
    /// `domains` (optional) labels each sample's domain.
    #[pyo3(signature = (coords, values, *, holes=None, error_variance=None, domains=None, domain_column=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        error_variance: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = Some(coords);
        let locs = points(coords)?;
        let values = finite(&column(data, values, "values")?, "values")?;
        same_length(locs.len(), values.len(), "values")?;
        let holes = holes.map(|h| column(data, h, "holes")).transpose()?;
        let holes = args::holes(holes.as_ref(), locs.len())?;
        let error = error_variance
            .map(|e| column(data, e, "error_variance"))
            .transpose()?;
        let error = args::optional_finite(error.as_ref(), "error_variance")?;
        if let Some(e) = &error {
            same_length(locs.len(), e.len(), "error_variance")?;
            if e.iter().any(|v| *v < 0.0) {
                return Err(invalid("error_variance must be >= 0"));
            }
            if !slf.method.kriging() {
                return Err(invalid("error_variance needs a kriging method"));
            }
        }
        let (fitted, codes) = match (domains, domain_column) {
            (None, None) => (None, None),
            _ => {
                let (fitted, codes) = args::domain_codes(data, domains, domain_column, locs.len())?;
                (Some(fitted), Some(codes))
            }
        };
        for s in &slf.search {
            s.resolve(fitted.as_deref())?;
        }
        let holes_text = holes.as_ref().map(|h| &h.0[..]);
        let keep = args::distinct_in(slf.py(), &locs, holes_text, codes.as_deref())?;
        let samples = keep
            .into_iter()
            .map(|i| Sample {
                error_variance: error.as_ref().map_or(0.0, |e| e[i]),
                domain: codes.as_ref().map(|c| c[i]),
                ..match &holes {
                    Some((_, ids)) => Sample::with_hole(locs[i], values[i], ids[i]),
                    None => Sample::new(locs[i], values[i]),
                }
            })
            .collect();
        slf.samples = Some(samples);
        slf.domains = fitted;
        Ok(slf)
    }

    /// Estimates (NaN where too few neighbors); with `return_variance`, also
    /// the kriging variance, and with `diagnostics` a Table adding kriging
    /// efficiency, slope of regression, samples used, the search pass
    /// (1-based) that filled each target, holes used, mean distance to the
    /// samples used, sum of negative weights, Lagrange multiplier and whether
    /// the search hit `max_samples`, other-domain samples used, the support
    /// variance C(v, v) and the estimator variance Var(Z*).
    /// `anisotropy` (a LocalAnisotropy) gives each target its own variogram
    /// and search orientation, taken from the nearest location. `domains`
    /// labels the targets, or is one label for all, and `domain_column` names
    /// a column of the targets holding them; a domain without samples is left
    /// unestimated.
    #[pyo3(signature = (targets, *, return_variance=false, anisotropy=None, diagnostics=false, domains=None, domain_column=None))]
    #[allow(clippy::too_many_arguments)]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        return_variance: bool,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        diagnostics: bool,
        domains: Option<Bound<'py, PyAny>>,
        domain_column: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let samples = self.fitted()?;
        let domains = match (domains, domain_column) {
            (d, None) => d,
            (None, Some(c)) => Some(args::named(Some(targets), c, "domain_column")?),
            _ => return Err(invalid("give one of domains or domain_column")),
        };
        let targets = self::targets(targets)?;
        let codes = codes(
            self.domains.as_deref(),
            domains.as_ref(),
            targets.len(),
            "predict",
        )?;
        let known: Vec<usize> = (0..targets.len())
            .filter(|&i| codes.as_ref().is_none_or(|c| c[i].is_some()))
            .collect();
        let search = self.passes()?;
        let vg = self.variogram.as_ref();
        let local = anisotropy.map(|field| field.at_targets(&targets));
        let base = self.variogram.clone().unwrap_or(CoreVariogram {
            nugget: 0.0,
            structures: vec![],
            anisotropy: None,
        });
        let found = py
            .detach(|| {
                by_pass(known.len(), &search, |search, remaining| {
                    let rows = pick(&known, remaining);
                    let at = pick(&targets, &rows);
                    let codes: Option<Vec<u32>> = codes
                        .as_ref()
                        .map(|c| rows.iter().map(|&i| c[i].expect("known")).collect());
                    let codes = codes.as_deref();
                    match &local {
                        None => Ok(estimate_many(&at, codes, samples, search, vg, |t, s| {
                            used(t, s, self.method.run(t, s, vg))
                        })),
                        Some(local) => estimation::lva::estimate_many_local(
                            &at,
                            codes,
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
        let mut passes: Vec<Option<(usize, Used)>> = (0..targets.len()).map(|_| None).collect();
        for (i, r) in known.into_iter().zip(found) {
            passes[i] = r;
        }
        if diagnostics {
            let table = self::diagnostics(&passes, &search, codes.as_deref())?;
            return table.into_bound_py_any(py);
        }
        let results: Vec<_> = split(passes).into_iter().map(|r| r.map(|r| r.0)).collect();
        outputs(py, &results, return_variance)
    }

    /// A copy with `search` (a Search or passes) in place of the searches,
    /// keeping the method, variogram, fitted samples and domains.
    fn with_search(&self, search: &Bound<PyAny>) -> PyResult<Self> {
        let search = searches(search)?;
        if self.samples.is_some() {
            for s in &search {
                s.resolve(self.domains.as_deref())?;
            }
        }
        Ok(Self {
            search,
            ..self.clone()
        })
    }

    /// A copy that estimates points: block kriging becomes ordinary kriging.
    fn _point_support(&self) -> Self {
        let method = match self.method {
            Method::Block { .. } => Method::Kriging(Kind::Ordinary),
            _ => self.method.clone(),
        };
        Self {
            method,
            ..self.clone()
        }
    }

    #[getter]
    fn variogram(&self) -> Option<Variogram> {
        self.variogram.clone().map(Variogram)
    }

    /// Domain label of each fitted sample, after dropping shared locations;
    /// None when fitted without domains.
    #[getter]
    fn _sample_domains<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        let Some(labels) = &self.domains else {
            return Ok(None);
        };
        self.fitted()?
            .iter()
            .map(|s| py_label(py, &labels[s.domain.expect("fitted with domains") as usize]))
            .collect::<PyResult<_>>()
            .map(Some)
    }

    /// Values of the fitted samples, after dropping shared locations.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let samples = self.fitted()?;
        Ok(array1(py, samples.iter().map(|s| s.value).collect()).into_any())
    }

    /// Leave-one-out, or with `folds` k-fold, estimates and variances at the
    /// fitted samples, each in its own domain; folds keep holes whole (see
    /// `k_fold_at`).
    #[pyo3(signature = (folds=None))]
    fn cross_validate<'py>(
        &self,
        py: Python<'py>,
        folds: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let samples = self.fitted()?;
        let search = self.passes()?;
        let vg = self.variogram.as_ref();
        let run = |t: &Point, s: &[Sample]| self.method.run(t, s, vg);
        let passes = py
            .detach(|| {
                by_pass(samples.len(), &search, |search, remaining| match folds {
                    None => Ok(leave_one_out_at(remaining, samples, search, vg, run)),
                    Some(k) => k_fold_at(k, remaining, samples, search, vg, run),
                })
            })
            .map_err(invalid)?;
        outputs(py, &split(passes), true)
    }

    /// Writes the estimator as class `name`: samples as columns, parameters
    /// as JSON in the file metadata.
    fn to_parquet(&self, path: PathBuf, name: &str) -> PyResult<()> {
        persist::to_parquet(name, self, &path)
    }

    #[staticmethod]
    fn from_parquet(path: PathBuf, name: &str) -> PyResult<Self> {
        persist::from_parquet(name, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<Columns>)> {
        persist::state("_Estimator", self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<Columns>) -> PyResult<Self> {
        persist::from_state("_Estimator", meta, columns)
    }
}

/// Global dual kriging with a polynomial drift of `degree` (no search).
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "DualKriging")]
pub struct Dual {
    variogram: CoreVariogram,
    degree: usize,
    #[serde(skip)]
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
        let locs = points(coords)?;
        let values = finite(&column(Some(coords), values, "values")?, "values")?;
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

    /// Writes samples as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: PathBuf) -> PyResult<()> {
        persist::to_parquet(<Self as PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: PathBuf) -> PyResult<Self> {
        persist::from_parquet(<Self as PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<Columns>)> {
        persist::state(<Self as PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<Columns>) -> PyResult<Self> {
        persist::from_state(<Self as PyClass>::NAME, meta, columns)
    }
}

/// Distances and value statistics of the `k` nearest samples around targets;
/// `n_holes` counts distinct `holes` within `radius`; `values` and `holes`
/// may name columns of `coords`.
#[pyfunction]
#[pyo3(signature = (targets, coords, values, *, k=8, radius=f64::INFINITY, variogram=None, holes=None))]
fn neighborhood_stats(
    targets: &Bound<PyAny>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    k: usize,
    radius: f64,
    variogram: Option<Variogram>,
    holes: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let data = Some(coords);
    let locs = points(coords)?;
    let values = finite(&column(data, values, "values")?, "values")?;
    same_length(locs.len(), values.len(), "values")?;
    let holes = holes.map(|h| column(data, h, "holes")).transpose()?;
    let holes = args::holes(holes.as_ref(), locs.len())?;
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
    let mut columns: Vec<(&str, ArrayRef)> = vec![];
    macro_rules! column {
        ($($f:ident),*) => {$(
            let values = stats.iter().map(|s| s.$f as f64);
            columns.push((stringify!($f), Arc::new(Float64Array::from_iter_values(values))));
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
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// Mean distance from each target to its `n` nearest drill holes, each hole
/// at its nearest sample, for classification by drill spacing.
///
/// Parameters
/// ----------
/// targets : array_like, PointSet or BlockModel
///     Locations to measure from.
/// coords : array_like
///     Sample coordinates, (n, 2) or (n, 3).
/// holes : array_like or str
///     Hole id of each sample, or the column of `coords` holding them;
///     samples of one hole count once.
/// n : int or sequence of int
///     Number of holes averaged; one per class.
/// search : Search, optional
///     Its ellipsoid sets the distance (meters along the major axis), and
///     samples beyond its radius are ignored. Isotropic and unbounded if
///     omitted.
/// domains : tuple of array_like, optional
///     ``(target_domains, sample_domains)``: each target only sees samples of
///     its own domain.
/// domain_column : str or tuple of str, optional
///     The column of `targets` and `coords` holding their domains, or one
///     name for each; instead of `domains`.
///
/// Returns
/// -------
/// ndarray
///     Distances, ``inf`` where fewer than `n` holes are in reach; shape
///     ``(targets, len(n))`` when `n` is a sequence.
#[pyfunction]
#[pyo3(signature = (targets, coords, holes, n, *, search=None, domains=None, domain_column=None))]
#[allow(clippy::too_many_arguments)]
fn hole_distance<'py>(
    py: Python<'py>,
    targets: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    holes: &Bound<'py, PyAny>,
    n: &Bound<PyAny>,
    search: Option<Search>,
    domains: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    domain_column: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let domains = match (domains, domain_column) {
        (d, None) => d,
        (None, Some(c)) => Some(args::pair(targets, coords, c, "domain_column")?),
        _ => return Err(invalid("give one of domains or domain_column")),
    };
    let locs = points(coords)?;
    let holes = column(Some(coords), holes, "holes")?;
    let (_, hole_ids) = args::holes(Some(&holes), locs.len())?.expect("given");
    let single = n.extract::<usize>().ok();
    let ns = match single {
        Some(k) => vec![k],
        None => n
            .extract::<Vec<usize>>()
            .map_err(|_| invalid("n must be a positive integer or a sequence of them"))?,
    };
    let targets = self::targets(targets)?;
    let labels = |obj: &Bound<PyAny>, len: usize| {
        args::holes(Some(obj), len)
            .map_err(|_| invalid("domains must be two 1-D sequences of labels"))
            .map(|l| l.expect("given").0)
    };
    let groups: Vec<(Vec<usize>, Vec<usize>)> = match &domains {
        None => vec![((0..targets.len()).collect(), (0..locs.len()).collect())],
        Some((at, of)) => {
            let (at, of) = (labels(at, targets.len())?, labels(of, locs.len())?);
            let mut by: std::collections::BTreeMap<&str, (Vec<usize>, Vec<usize>)> =
                Default::default();
            for (i, d) in at.iter().enumerate() {
                by.entry(d).or_default().0.push(i);
            }
            for (i, d) in of.iter().enumerate() {
                if let Some(g) = by.get_mut(d.as_str()) {
                    g.1.push(i);
                }
            }
            by.into_values().collect()
        }
    };
    let (radius, aniso) = search.map_or((f64::INFINITY, None), |s| {
        (s.core.radius, s.core.anisotropy)
    });
    let columns = py.detach(|| {
        let mut columns = vec![vec![f64::INFINITY; targets.len()]; ns.len()];
        for (at, of) in &groups {
            let t: Vec<Point> = at.iter().map(|&i| targets[i]).collect();
            let l: Vec<Point> = of.iter().map(|&i| locs[i]).collect();
            let h: Vec<u32> = of.iter().map(|&i| hole_ids[i]).collect();
            let d = estimation::hole_distance(&t, &l, &h, &ns, radius, aniso.as_ref())?;
            for (column, d) in columns.iter_mut().zip(d) {
                for (&i, d) in at.iter().zip(d) {
                    column[i] = d;
                }
            }
        }
        Ok::<_, estimation::EstimError>(columns)
    });
    let mut columns = columns.map_err(invalid)?;
    if single.is_some() {
        return Ok(array1(py, columns.remove(0)).into_any());
    }
    let rows: Vec<Vec<f64>> = (0..targets.len())
        .map(|i| columns.iter().map(|c| c[i]).collect())
        .collect();
    Ok(args::array2(py, &rows).into_any())
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hole_distance, m)?)?;
    m.add_class::<Search>()?;
    m.add_class::<Estimator>()?;
    m.add_class::<Dual>()?;
    m.add_function(wrap_pyfunction!(neighborhood_stats, m)?)?;
    Ok(())
}
