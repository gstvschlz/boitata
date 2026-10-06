use estimation::{
    Calibration, Discretization, DriftSpec, DualKriging, Estimate, HighGrade, HighGradeMode,
    InterpEstimate, InterpOptions, Kind, NeighborhoodStats, PlaneSectors, Sample,
    Search as CoreSearch, Soft, SoftPair, block_krige, by_pass, estimate_many, estimate_many_ext,
    k_fold_at, krige, krige_bayesian, krige_factorial, krige_universal, leave_one_out_at,
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

/// High-grade restriction of a Search: samples above `threshold` are left
/// out (`mode="drop"`) or capped at `threshold` (`mode="clamp"`) beyond
/// `radius` of the target. `radius` is a distance in the search ellipsoid,
/// or ranges `(major, semi, minor)` of an ellipsoid of its own, rotated by
/// `rotation` (azimuth, dip, rake).
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "HighGrade", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyHighGrade(pub HighGrade);

impl PyHighGrade {
    /// A HighGrade, or a `(threshold, radius)` tuple.
    fn of(obj: &Bound<PyAny>) -> PyResult<HighGrade> {
        if let Ok(h) = obj.cast::<PyHighGrade>() {
            return Ok(h.get().0.clone());
        }
        let (threshold, radius): (f64, f64) = obj
            .extract()
            .map_err(|_| invalid("high_grade must be a HighGrade or (threshold, radius)"))?;
        Ok(Self::new(
            threshold,
            &radius.into_bound_py_any(obj.py())?,
            None,
            "drop",
        )?
        .0)
    }
}

#[pymethods]
impl PyHighGrade {
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (threshold, radius, *, rotation=None, mode="drop"))]
    fn new(
        threshold: f64,
        radius: &Bound<PyAny>,
        rotation: Option<(f64, f64, f64)>,
        mode: &str,
    ) -> PyResult<Self> {
        let mode = match mode {
            "drop" => HighGradeMode::Drop,
            "clamp" => HighGradeMode::Clamp,
            _ => {
                return Err(invalid(format!(
                    "mode must be 'drop' or 'clamp', not {mode:?}"
                )));
            }
        };
        let (ranges, own) = match radius.extract::<f64>() {
            Ok(r) => ((r, r, r), rotation.is_some()),
            Err(_) => (
                radius
                    .extract::<(f64, f64, f64)>()
                    .map_err(|_| invalid("radius must be a distance or (major, semi, minor)"))?,
                true,
            ),
        };
        let (major, semi, minor) = ranges;
        let valid = match own {
            false => major >= 0.0,
            true => major > 0.0 && semi > 0.0 && minor > 0.0,
        };
        if threshold.is_nan() || !valid {
            return Err(invalid(
                "high_grade needs a threshold, a radius >= 0 and ranges > 0",
            ));
        }
        let anisotropy = match own {
            false => None,
            true => Some(
                variogram::Anisotropy::new(variogram::Angles {
                    azimuth: rotation.map_or(0.0, |r| r.0),
                    dip: rotation.map_or(0.0, |r| r.1),
                    rake: rotation.map_or(0.0, |r| r.2),
                    major: 1.0,
                    semi: semi / major,
                    minor: minor / major,
                })
                .map_err(|e| invalid(e.to_string()))?,
            ),
        };
        Ok(Self(HighGrade {
            threshold,
            radius: major,
            anisotropy,
            mode,
        }))
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.0.threshold
    }

    /// The distance, or the major range of the restriction's own ellipsoid.
    #[getter]
    fn radius(&self) -> f64 {
        self.0.radius
    }

    /// Ranges `(major, semi, minor)` of the restriction's own ellipsoid;
    /// None when it is measured in the search ellipsoid.
    #[getter]
    fn ranges(&self) -> Option<(f64, f64, f64)> {
        let a = &self.0.anisotropy.as_ref()?.angles;
        let r = self.0.radius;
        Some((r, r * a.semi, r * a.minor))
    }

    #[getter]
    fn rotation(&self) -> Option<(f64, f64, f64)> {
        let a = &self.0.anisotropy.as_ref()?.angles;
        Some((a.azimuth, a.dip, a.rake))
    }

    #[getter]
    fn mode(&self) -> &'static str {
        match self.0.mode {
            HighGradeMode::Drop => "drop",
            HighGradeMode::Clamp => "clamp",
        }
    }

    fn __eq__(&self, other: &Bound<PyAny>) -> bool {
        other.cast::<PyHighGrade>().is_ok_and(|o| {
            let (a, b) = (&self.0, &o.get().0);
            a.threshold == b.threshold
                && a.radius == b.radius
                && a.mode == b.mode
                && a.anisotropy.as_ref().map(|x| &x.angles)
                    == b.anisotropy.as_ref().map(|x| &x.angles)
        })
    }

    fn __repr__(&self) -> String {
        let extent = match self.ranges() {
            Some(r) => format!("{r:?}, rotation={:?}", self.rotation().unwrap_or_default()),
            None => format!("{}", self.0.radius),
        };
        format!(
            "HighGrade({}, {extent}, mode={:?})",
            self.0.threshold,
            self.mode()
        )
    }
}

/// Neighborhood: `radius` is in meters along the major axis of the
/// search ellipsoid (`rotation` azimuth, dip, rake and `ratios` semi/major,
/// minor/major), or of the variogram's anisotropy when no ellipsoid is given.
/// `octant` takes at most `max_samples / 8` samples from each octant around
/// the target, split along the ellipsoid's axes; with 2D data (one elevation)
/// the sectors are the ellipse's four quadrants, `max_samples / 4` each.
/// `sectors` instead splits the plane of the ellipsoid's major and
/// semi-major axes into that many equal angles, the first starting at the
/// major axis, and takes at most `max_per_sector` samples from each
/// (`max_samples / sectors` rounded up by default).
/// `high_grade`, a HighGrade or `(threshold, radius)`, restricts samples
/// above `threshold` to targets within `radius`, measured in the same
/// ellipsoid unless the HighGrade has its own, in estimation and
/// cross-validation alike; give each pass its own to vary it by pass. The threshold is always in data
/// units: simulators compare it with the data values and the simulated
/// values of nodes, so it picks the same samples as in estimation. Estimators also take a sequence
/// of searches as passes: targets one leaves unestimated go to the next.
/// `soft` lets samples of another domain inform a target strictly within a
/// distance in the same ellipsoid: one distance for every pair of domains,
/// or a dict `{(target_domain, sample_domain): distance}`, one way; pairs not
/// listed are hard.
/// `target_slope` or `target_efficiency` (at most one) calibrate the kriging
/// estimators per target: each takes the fewest samples from `min_samples`
/// whose slope of regression or kriging efficiency reaches it, and
/// `max_samples` where none does. With `octant` the fewer samples are taken
/// from the sectors in turn.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "Search", frozen, from_py_object)]
#[derive(Clone)]
pub struct Search {
    #[serde(flatten)]
    pub core: CoreSearch,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    soft: Option<Soft<Label>>,
    /// Length unit of the radius; None when not declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length_unit: Option<String>,
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
        self.uncalibrated(what)?;
        match self.soft {
            Some(_) => Err(invalid(format!(
                "{what} does not take domains; Search.soft works with SGS, TurningBands and the kriging, \
                 inverse-distance, nearest-neighbor and interpolation estimators"
            ))),
            None => Ok(self.core),
        }
    }

    /// An error unless the search is uncalibrated, for `what`.
    pub fn uncalibrated(&self, what: &str) -> PyResult<()> {
        match self.core.calibration {
            Some(_) => Err(invalid(format!(
                "{what} does not take target_slope or target_efficiency; they calibrate the \
                 ordinary, simple, indicator, universal, external-drift and block kriging estimators"
            ))),
            None => Ok(()),
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
    #[pyo3(signature = (radius, *, max_samples=16, min_samples=1, octant=false, sectors=None, max_per_sector=None, max_per_hole=None, rotation=None, ratios=None, high_grade=None, soft=None, target_slope=None, target_efficiency=None, length_unit=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        radius: &Bound<PyAny>,
        max_samples: usize,
        min_samples: usize,
        octant: bool,
        sectors: Option<usize>,
        max_per_sector: Option<usize>,
        max_per_hole: Option<usize>,
        rotation: Option<(f64, f64, f64)>,
        ratios: Option<(f64, f64)>,
        high_grade: Option<&Bound<PyAny>>,
        soft: Option<&Bound<PyAny>>,
        target_slope: Option<f64>,
        target_efficiency: Option<f64>,
        length_unit: Option<String>,
    ) -> PyResult<Self> {
        let (radius, unit) = crate::units::given(radius, "radius")?;
        let length_unit = crate::units::common_length([length_unit, unit])?;
        if let Some(u) = &length_unit {
            boitata_core::units::check_length(u).map_err(invalid)?;
        }
        let calibration = match (target_slope, target_efficiency) {
            (Some(_), Some(_)) => {
                return Err(invalid(
                    "give at most one of target_slope and target_efficiency",
                ));
            }
            (Some(s), None) if !s.is_finite() || s <= 0.0 => {
                return Err(invalid("target_slope must be finite and > 0"));
            }
            (None, Some(e)) if !e.is_finite() || e > 1.0 => {
                return Err(invalid("target_efficiency must be finite and <= 1"));
            }
            (Some(s), None) => Some(Calibration::Slope(s)),
            (None, Some(e)) => Some(Calibration::Efficiency(e)),
            (None, None) => None,
        };
        let high_grade = high_grade.map(PyHighGrade::of).transpose()?;
        if radius.is_nan() || radius <= 0.0 || max_samples == 0 || min_samples > max_samples {
            return Err(invalid(
                "need radius > 0 and 1 <= min_samples <= max_samples",
            ));
        }
        if max_per_sector.is_some() && sectors.is_none() {
            return Err(invalid("max_per_sector needs sectors"));
        }
        let sectors = sectors.map(|count| PlaneSectors {
            count,
            max_per_sector: max_per_sector.unwrap_or(max_samples.div_ceil(count.max(1))),
        });
        let search = Self {
            soft: soft.map(self::soft).transpose()?,
            length_unit,
            core: CoreSearch {
                min_samples,
                max_samples,
                radius,
                max_per_hole,
                octant,
                sectors,
                anisotropy: match (rotation, ratios) {
                    (None, None) => None,
                    (rotation, ratios) => crate::variogram::anisotropy(
                        rotation.unwrap_or((0.0, 0.0, 0.0)),
                        ratios.unwrap_or((1.0, 1.0)),
                    )?,
                },
                high_grade,
                soft: None,
                calibration,
            },
        };
        search
            .core
            .check_sectors()
            .map_err(|e| invalid(e.to_string()))?;
        Ok(search)
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.core.radius
    }

    /// Length unit of the radius and of the variogram's ranges used with it;
    /// None when not declared.
    #[getter]
    fn length_unit(&self) -> Option<String> {
        self.length_unit.clone()
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

    /// Number of angular sectors in the major / semi-major plane; None without.
    #[getter]
    fn sectors(&self) -> Option<usize> {
        self.core.sectors.map(|s| s.count)
    }

    #[getter]
    fn max_per_sector(&self) -> Option<usize> {
        self.core.sectors.map(|s| s.max_per_sector)
    }

    #[getter]
    fn max_per_hole(&self) -> Option<usize> {
        self.core.max_per_hole
    }

    #[getter]
    fn high_grade(&self) -> Option<PyHighGrade> {
        self.core.high_grade.clone().map(PyHighGrade)
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

    /// Slope of regression the search is calibrated to; None when it is not.
    #[getter]
    fn target_slope(&self) -> Option<f64> {
        match self.core.calibration? {
            Calibration::Slope(s) => Some(s),
            Calibration::Efficiency(_) => None,
        }
    }

    /// Kriging efficiency the search is calibrated to; None when it is not.
    #[getter]
    fn target_efficiency(&self) -> Option<f64> {
        match self.core.calibration? {
            Calibration::Efficiency(e) => Some(e),
            Calibration::Slope(_) => None,
        }
    }

    /// Semi-major/major and minor/major ratios; None as for `rotation`.
    #[getter]
    fn ratios(&self) -> Option<(f64, f64)> {
        let a = &self.core.anisotropy.as_ref()?.angles;
        Some((a.semi, a.minor))
    }

    fn __repr__(&self) -> String {
        format!(
            "Search(radius={}, max_samples={}, min_samples={}, octant={}, sectors={:?})",
            self.core.radius,
            self.core.max_samples,
            self.core.min_samples,
            self.core.octant,
            self.sectors()
        )
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum Method {
    Kriging(Kind),
    Universal(usize),
    ExternalDrift {
        degree: usize,
        drift: Vec<String>,
    },
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
            Method::ExternalDrift { .. } => Err(estimation::EstimError::InvalidParameters(
                "external-drift kriging needs covariates; only predict() supports it".into(),
            )),
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

    /// An error when a search is calibrated and this method cannot be.
    fn calibrates(&self, searches: &[Search]) -> PyResult<()> {
        for s in searches {
            match (self, s.core.calibration) {
                (_, None) => {}
                (Method::Kriging(Kind::Simple { .. }), Some(Calibration::Slope(_))) => {
                    return Err(invalid(
                        "simple kriging has a slope of regression of 1; calibrate it with target_efficiency",
                    ));
                }
                (
                    Method::Kriging(_)
                    | Method::Universal(_)
                    | Method::ExternalDrift { .. }
                    | Method::Block { .. },
                    _,
                ) => {}
                _ => s.uncalibrated("this estimator")?,
            }
        }
        Ok(())
    }

    /// How many of the samples `s`, ordered by the search, to krige `t`
    /// with under the calibration of `search`, and whether they meet it;
    /// all of them and None without calibration. `external` holds the
    /// external-drift rows of the samples and of the target.
    fn calibrated(
        &self,
        search: &CoreSearch,
        t: &Point,
        s: &[Sample],
        vg: Option<&CoreVariogram>,
        external: Option<(&[Vec<f64>], &[f64])>,
    ) -> (usize, Option<bool>) {
        let (Some(calibration), Some(vg)) = (search.calibration, vg) else {
            return (s.len(), None);
        };
        let locs = || s.iter().map(|s| s.loc).collect::<Vec<Point>>();
        let rhs = || s.iter().map(|s| vg.cov_points(&s.loc, t)).collect();
        let (rhs, drift, support) = match self {
            Method::Kriging(Kind::Simple { .. }) => (rhs(), DriftSpec::simple(s.len()), None),
            Method::Kriging(_) => (rhs(), DriftSpec::ordinary(s.len()), None),
            Method::Universal(degree) => (rhs(), DriftSpec::polynomial(&locs(), t, *degree), None),
            Method::ExternalDrift { degree, .. } => {
                let drift = DriftSpec::polynomial(&locs(), t, *degree);
                let drift = match external {
                    Some((rows, at)) => drift.with_external(rows, at),
                    None => drift,
                };
                (rhs(), drift, None)
            }
            Method::Block { size, disc } => {
                let (rhs, cbb) = estimation::block_covariances(&disc.points(t, size), s, vg);
                (rhs, DriftSpec::ordinary(s.len()), Some(cbb))
            }
            _ => return (s.len(), None),
        };
        let support = support.unwrap_or_else(|| vg.total_sill());
        let quality = estimation::prefix_quality(s, vg, &rhs, &drift, support);
        let (n, met) = estimation::calibrated_count(calibration, search.min_samples, &quality);
        (n, Some(met))
    }

    /// [`Method::run`] on the samples of `s` that the calibration of
    /// `search` keeps.
    fn estimate(
        &self,
        search: &CoreSearch,
        t: &Point,
        s: &[Sample],
        vg: Option<&CoreVariogram>,
    ) -> estimation::Result<Used> {
        let (n, met) = self.calibrated(search, t, s, vg, None);
        used(t, &s[..n], self.run(t, &s[..n], vg), met)
    }

    /// Whether the estimate is a weighted sum of the data, as `weights` needs.
    fn linear(&self) -> bool {
        !matches!(
            self,
            Method::Factorial { .. }
                | Method::Bayesian { .. }
                | Method::MovingMedian
                | Method::LocalLeastSquares(_)
        )
    }

    /// The weights of the estimate on `s`, in their order.
    fn weights(
        &self,
        t: &Point,
        s: &[Sample],
        vg: Option<&CoreVariogram>,
    ) -> estimation::Result<Vec<f64>> {
        let opts = InterpOptions {
            dmax: f64::INFINITY,
            anisotropy: vg.and_then(|v| v.anisotropy.clone()),
            min_samples: 1,
        };
        match self {
            Method::InverseDistance(power) => {
                estimation::inverse_distance_weights(t, s, *power, &opts)
            }
            Method::Nearest => {
                let mut w = vec![0.0; s.len()];
                w[estimation::nearest_index(t, s, &opts)?] = 1.0;
                Ok(w)
            }
            Method::MovingAverage => Ok(vec![1.0 / s.len() as f64; s.len()]),
            _ => self.run(t, s, vg).map(|e| e.weights),
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

/// The estimate, statistics of the samples used, their domains and whether
/// they meet the search's calibration.
type Used = (Estimate, NeighborhoodStats, Vec<Option<u32>>, Option<bool>);

fn used(
    t: &Point,
    s: &[Sample],
    e: estimation::Result<Estimate>,
    met: Option<bool>,
) -> estimation::Result<Used> {
    e.map(|e| {
        let near = estimation::neighborhood_stats(t, s, s.len(), f64::INFINITY, None);
        (e, near, s.iter().map(|s| s.domain).collect(), met)
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
                .map(|r| r.as_ref().map_or(f64::NAN, |(p, (e, s, ..))| f(*p, e, s)))
                .collect(),
        )
    };
    let other = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            r.as_ref().map_or(f64::NAN, |(_, (_, _, used, _))| {
                let own = domains.and_then(|d| d[i]);
                used.iter().filter(|&&d| own.is_some() && d != own).count() as f64
            })
        })
        .collect();
    let met = results
        .iter()
        .map(|r| match r {
            Some((_, (.., Some(met)))) => f64::from(u8::from(*met)),
            _ => f64::NAN,
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
        ("target_met", floats(met)),
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
        let mut columns = sample_columns(self.samples.as_deref()?);
        for (j, col) in self.drift_data.iter().enumerate() {
            columns.push(persist::column(&format!("drift_{j}"), col.iter().copied()));
        }
        Some(columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.samples = Some(samples_from(&columns)?);
        self.drift_data = match &self.method {
            Method::ExternalDrift { drift, .. } => (0..drift.len())
                .map(|j| columns.values(&format!("drift_{j}")))
                .collect::<PyResult<_>>()?,
            _ => vec![],
        };
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

/// [`searches`] for `what`, which takes no calibration.
pub fn plain_searches(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<Search>> {
    let search = searches(obj)?;
    for s in &search {
        s.uncalibrated(what)?;
    }
    Ok(search)
}

/// Shared engine behind the estimator classes in `boitata.estimation`.
#[derive(Clone, Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "_Estimator", skip_from_py_object)]
pub struct Estimator {
    method: Method,
    variogram: Option<CoreVariogram>,
    search: Vec<Search>,
    /// Labels of the fitted domains, indexed by code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domains: Option<Vec<Label>>,
    /// External-drift covariates, fitted with `Method::ExternalDrift`: one
    /// entry per drift variable, each a value per fitted sample. Empty for
    /// every other method.
    #[serde(default)]
    drift_data: Vec<Vec<f64>>,
    /// Units the Python layer records at fit: of the values, of the sample
    /// coordinates, and of the lengths of the search and variogram.
    #[serde(default)]
    units: (Option<String>, Option<String>, Option<String>),
    #[serde(skip)]
    samples: Option<Vec<Sample>>,
}

impl Estimator {
    fn fitted(&self) -> PyResult<&[Sample]> {
        self.samples
            .as_deref()
            .ok_or_else(|| invalid("estimator is not fitted; call fit first"))
    }

    /// Stores the samples as `fit` does; returns all locations and values and
    /// the rows kept.
    #[allow(clippy::too_many_arguments)]
    fn store(
        &mut self,
        py: Python,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        error_variance: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
    ) -> PyResult<(Vec<Point>, Vec<f64>, Vec<usize>)> {
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
            if !self.method.kriging() {
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
        for s in &self.search {
            s.resolve(fitted.as_deref())?;
        }
        let holes_text = holes.as_ref().map(|h| &h.0[..]);
        let keep = args::distinct_in(py, &locs, holes_text, codes.as_deref())?;
        self.drift_data = match &self.method {
            Method::ExternalDrift { drift, .. } => drift
                .iter()
                .map(|name| {
                    let col = args::finite(&args::named(data, name, "drift")?, "drift")?;
                    same_length(locs.len(), col.len(), "drift")?;
                    Ok(keep.iter().map(|&i| col[i]).collect())
                })
                .collect::<PyResult<_>>()?,
            _ => vec![],
        };
        let samples = keep
            .iter()
            .map(|&i| Sample {
                error_variance: error.as_ref().map_or(0.0, |e| e[i]),
                domain: codes.as_ref().map(|c| c[i]),
                ..match &holes {
                    Some((_, ids)) => Sample::with_hole(locs[i], values[i], ids[i]),
                    None => Sample::new(locs[i], values[i]),
                }
            })
            .collect();
        self.samples = Some(samples);
        self.domains = fitted;
        Ok((locs, values, keep))
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
            "external_drift" => Method::ExternalDrift {
                degree: int("degree", 0)?,
                drift: {
                    let drift = get("drift")?
                        .ok_or_else(|| invalid("external-drift kriging needs drift"))?;
                    match drift.extract::<String>() {
                        Ok(name) => vec![name],
                        Err(_) => drift.extract().map_err(|_| {
                            invalid("drift must be a column name or a sequence of names")
                        })?,
                    }
                },
            },
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
        method.calibrates(&search)?;
        Ok(Self {
            method,
            variogram: variogram.map(|v| v.0),
            search,
            domains: None,
            drift_data: vec![],
            units: Default::default(),
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
        let py = slf.py();
        slf.store(
            py,
            coords,
            values,
            holes,
            error_variance,
            domains,
            domain_column,
        )?;
        Ok(slf)
    }

    /// Weight declustering of `coords` over `targets`: see
    /// `boitata.weight_declustering`.
    fn _declustering(
        &self,
        py: Python,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        targets: &Bound<PyAny>,
    ) -> PyResult<crate::transforms::Declustering> {
        // Scope cut: declustering estimates over `targets`, which would need
        // each target's own covariate row too; not wired up yet.
        if matches!(self.method, Method::ExternalDrift { .. }) {
            return Err(invalid(
                "weight declustering is not yet supported for external-drift kriging",
            ));
        }
        if !self.method.linear() {
            return Err(invalid(
                "weight declustering needs an estimator linear in the data",
            ));
        }
        let mut fitted = Self {
            domains: None,
            ..self.clone()
        };
        let (locs, values, keep) = fitted.store(py, coords, values, None, None, None, None)?;
        let samples = fitted.fitted()?;
        let targets = self::targets(targets)?;
        let search = fitted.passes()?;
        let vg = fitted.variogram.as_ref();
        let per_target = py
            .detach(|| {
                by_pass(targets.len(), &search, |search, remaining| {
                    let at = pick(&targets, remaining);
                    Ok(estimation::weights_many(
                        &at,
                        samples,
                        search,
                        vg,
                        |t, s| {
                            let (n, _) = self.method.calibrated(search, t, s, vg, None);
                            let mut w = self.method.weights(t, &s[..n], vg)?;
                            w.resize(s.len(), 0.0);
                            Ok(w)
                        },
                    ))
                })
            })
            .map_err(invalid)?;
        let kept: Vec<f64> = samples.iter().map(|s| s.value).collect();
        let w = estimation::weight_declustering(&split(per_target), &kept).map_err(invalid)?;
        // Samples sharing a location share the weight of the one kept.
        let bits = |p: &Point| [p.0 + 0.0, p.1 + 0.0, p.2 + 0.0].map(f64::to_bits);
        let at: std::collections::HashMap<_, usize> = keep
            .iter()
            .enumerate()
            .map(|(k, &i)| (bits(&locs[i]), k))
            .collect();
        let owner: Vec<usize> = locs.iter().map(|p| at[&bits(p)]).collect();
        let mut shared = vec![0usize; keep.len()];
        owner.iter().for_each(|&k| shared[k] += 1);
        let scale = locs.len() as f64 / keep.len() as f64;
        let weights: Vec<f64> = owner
            .iter()
            .map(|&k| w.weights[k] / shared[k] as f64 * scale)
            .collect();
        let mean = weights.iter().zip(&values).map(|(w, v)| w * v).sum::<f64>() / locs.len() as f64;
        Ok(crate::transforms::declustering(
            transforms::Weights {
                weights,
                declustered_mean: mean,
                cell_size: f64::NAN,
            },
            vec![],
            vec![],
        ))
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
        // Resolve external-drift columns against the raw `targets` container,
        // before it is reduced to bare coordinates below.
        let target_drift: Vec<Vec<f64>> = match &self.method {
            Method::ExternalDrift { drift, .. } => drift
                .iter()
                .map(|name| args::finite(&args::named(Some(targets), name, "drift")?, "drift"))
                .collect::<PyResult<_>>()?,
            _ => vec![],
        };
        if anisotropy.is_some() && matches!(self.method, Method::ExternalDrift { .. }) {
            return Err(invalid(
                "external-drift kriging does not support local anisotropy",
            ));
        }
        let targets = self::targets(targets)?;
        for c in &target_drift {
            same_length(targets.len(), c.len(), "drift")?;
        }
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
        // Per-sample covariate rows, aligned with `samples`, for external-drift kriging.
        let cov: Vec<Vec<f64>> = match &self.method {
            Method::ExternalDrift { .. } => (0..samples.len())
                .map(|i| self.drift_data.iter().map(|c| c[i]).collect())
                .collect(),
            _ => vec![],
        };
        let found = py
            .detach(|| {
                by_pass(known.len(), &search, |search, remaining| {
                    let rows = pick(&known, remaining);
                    let at = pick(&targets, &rows);
                    let codes: Option<Vec<u32>> = codes
                        .as_ref()
                        .map(|c| rows.iter().map(|&i| c[i].expect("known")).collect());
                    let codes = codes.as_deref();
                    match (&local, &self.method) {
                        (None, Method::ExternalDrift { degree, .. }) => {
                            let ext: Vec<Vec<f64>> = rows
                                .iter()
                                .map(|&i| target_drift.iter().map(|c| c[i]).collect())
                                .collect();
                            Ok(estimate_many_ext(
                                &at,
                                codes,
                                &cov,
                                &ext,
                                samples,
                                search,
                                vg,
                                |t, s, cov, e| {
                                    let (n, met) =
                                        self.method.calibrated(search, t, s, vg, Some((cov, e)));
                                    let (s, cov) = (&s[..n], &cov[..n]);
                                    let locs: Vec<Point> = s.iter().map(|x| x.loc).collect();
                                    let drift = DriftSpec::polynomial(&locs, t, *degree)
                                        .with_external(cov, e);
                                    let kriged =
                                        krige_universal(t, s, &drift, vg.expect("kriging"), None);
                                    used(t, s, kriged, met)
                                },
                            ))
                        }
                        (None, _) => Ok(estimate_many(&at, codes, samples, search, vg, |t, s| {
                            self.method.estimate(search, t, s, vg)
                        })),
                        (Some(local), _) => estimation::lva::estimate_many_local(
                            &at,
                            codes,
                            &local.at(&at),
                            samples,
                            search,
                            &base,
                            |t, s, v| self.method.estimate(search, t, s, Some(v)),
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
        self.method.calibrates(&search)?;
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
        self.variogram
            .clone()
            .map(|v| Variogram(v, Default::default()))
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

    fn _set_units(&mut self, unit: Option<String>, coords: Option<String>, length: Option<String>) {
        self.units = (unit, coords, length);
    }

    #[getter]
    fn _units(&self) -> (Option<String>, Option<String>, Option<String>) {
        self.units.clone()
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
    #[pyo3(signature = (*, folds=None))]
    fn cross_validate<'py>(
        &self,
        py: Python<'py>,
        folds: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Scope cut: cross-validation re-estimates fitted samples, which
        // would need each held-out sample's own covariate row too; not
        // wired up yet.
        if matches!(self.method, Method::ExternalDrift { .. }) {
            return Err(invalid(
                "cross-validation is not yet supported for external-drift kriging",
            ));
        }
        let samples = self.fitted()?;
        let search = self.passes()?;
        let vg = self.variogram.as_ref();
        let passes = py
            .detach(|| {
                by_pass(samples.len(), &search, |search, remaining| {
                    let run = |t: &Point, s: &[Sample]| {
                        self.method.estimate(search, t, s, vg).map(|u| u.0)
                    };
                    match folds {
                        None => Ok(leave_one_out_at(remaining, samples, search, vg, run)),
                        Some(k) => k_fold_at(k, remaining, samples, search, vg, run),
                    }
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
#[pyclass(module = "boitata", name = "DualKriging")]
pub struct Dual {
    variogram: CoreVariogram,
    degree: usize,
    /// Unit of the fitted values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unit: Option<String>,
    #[serde(skip)]
    samples: Option<Vec<Sample>>,
}

#[pymethods]
impl Dual {
    #[new]
    #[pyo3(signature = (variogram, *, degree=0))]
    fn new(variogram: Variogram, degree: usize) -> Self {
        Self {
            variogram: variogram.0,
            degree,
            unit: None,
            samples: None,
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.unit = crate::units::of(values, Some(coords))?;
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
        crate::units::tag(array1(py, values).into_any(), self.unit.as_deref())
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
    m.add_class::<PyHighGrade>()?;
    m.add_class::<Estimator>()?;
    m.add_class::<Dual>()?;
    m.add_function(wrap_pyfunction!(neighborhood_stats, m)?)?;
    Ok(())
}
