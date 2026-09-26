//! Multiple indicator kriging: the conditional distribution `P(Z ≤ t | data)`
//! kriged at many thresholds, corrected for order relations, completed within
//! classes and in the tails, then summarized.
//!
//! Classes run from the lower tail bound through the thresholds to the upper
//! tail bound. Within a class the distribution follows the declustered data
//! falling in it (`Interpolation::Global`) or is uniform (`Linear`); the last
//! class may take a power or hyperbolic model instead.
//!
//! Localisation gives each panel's selective blocks the band means
//! ([`transforms::localize`]) of the panel's distribution after an affine
//! change of support, `m + √f·(z − m)`, with `f` the variance of the blocks
//! within the panel over that of the points within it.

use ceres_core::{BlockModel, block_frame};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use transforms::TransformError;
use variogram::Variogram;

use crate::Sample;
use crate::batch::{by_pass, estimate_many};
use crate::error::{EstimError, Result};
use crate::krige::{Kind, krige};
use crate::lva::{LocalAnisotropy, estimate_many_local};
use crate::search::Search;

type Point = (f64, f64, f64);

/// Distribution within a class.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Interpolation {
    /// The declustered data in the class; uniform where a class holds none.
    Global,
    /// Uniform between the class bounds.
    Linear,
}

/// Model of the class above the last threshold, up to the upper tail bound.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UpperTail {
    /// `F(z) ∝ ((z − a) / (b − a))^ω`; ω = 1 is linear.
    Power(f64),
    /// `1 − F(z) ∝ (a / z)^ω`, truncated at the upper bound.
    Hyperbolic(f64),
}

/// Parameters of multiple indicator kriging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleIndicator {
    /// Strictly increasing.
    pub thresholds: Vec<f64>,
    /// One per threshold, or one for all (median indicator kriging).
    pub variograms: Vec<Variogram>,
    /// Simple kriging with the declustered global proportions as means.
    pub simple: bool,
    /// Tail bounds; widened to cover the data and thresholds. Data min and
    /// max when `None`.
    pub tails: Option<(f64, f64)>,
    pub interpolation: Interpolation,
    pub upper_tail: Option<UpperTail>,
}

/// Declustered global distribution of the data, split into classes.
#[derive(Debug, Clone)]
pub struct Global {
    values: Vec<f64>,
    /// Prefix sums of w, w·v and w·v², weights summing to 1.
    sums: [Vec<f64>; 3],
    /// Data index range of each class.
    classes: Vec<(usize, usize)>,
    pub lower: f64,
    pub upper: f64,
    /// `F(t)` at each threshold.
    pub proportions: Vec<f64>,
}

/// Conditional distribution at one target.
#[derive(Debug, Clone)]
pub struct Conditional {
    /// Order-relation corrected `P(Z ≤ t)` at each threshold.
    pub cdf: Vec<f64>,
    /// Sum of |corrected − kriged| over the thresholds.
    pub correction: f64,
    pub mean: f64,
    pub variance: f64,
    pub probability_above: Vec<f64>,
    /// NaN where nothing lies above the cutoff.
    pub mean_above: Vec<f64>,
    pub quantile_values: Vec<f64>,
}

/// Per-target summary; per-threshold, per-cutoff and per-quantile fields are
/// indexed `[threshold, cutoff or quantile][target]`, NaN where unestimated.
#[derive(Debug, Clone, Default)]
pub struct IndicatorSummary {
    pub thresholds: Vec<f64>,
    pub cdf: Vec<Vec<f64>>,
    pub correction: Vec<f64>,
    /// E-type estimate.
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub cutoffs: Vec<f64>,
    pub probability_above: Vec<Vec<f64>>,
    pub mean_above: Vec<Vec<f64>>,
    pub quantiles: Vec<f64>,
    pub quantile_values: Vec<Vec<f64>>,
}

fn invalid(message: &str) -> EstimError {
    EstimError::InvalidParameters(message.into())
}

/// Clips to [0, 1], then averages the upward running maximum and the
/// downward running minimum; also returns Σ|corrected − raw|.
pub fn correct_order_relations(raw: &[f64]) -> (Vec<f64>, f64) {
    let clipped: Vec<f64> = raw.iter().map(|p| p.clamp(0.0, 1.0)).collect();
    let mut up = clipped.clone();
    for k in 1..up.len() {
        up[k] = up[k].max(up[k - 1]);
    }
    let mut down = clipped;
    for k in (0..down.len().saturating_sub(1)).rev() {
        down[k] = down[k].min(down[k + 1]);
    }
    let cdf: Vec<f64> = up.iter().zip(&down).map(|(u, d)| 0.5 * (u + d)).collect();
    let correction = cdf.iter().zip(raw).map(|(c, r)| (c - r).abs()).sum();
    (cdf, correction)
}

impl Global {
    pub fn new(
        values: &[f64],
        weights: Option<&[f64]>,
        thresholds: &[f64],
        tails: Option<(f64, f64)>,
    ) -> Result<Self> {
        if values.is_empty() {
            return Err(EstimError::InsufficientData("no samples".into()));
        }
        let weights = weights.map_or_else(|| vec![1.0; values.len()], <[f64]>::to_vec);
        if weights.len() != values.len() {
            return Err(invalid("need one weight per sample"));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(invalid("weights must be finite and >= 0"));
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(invalid("weights must not all be 0"));
        }
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&i, &j| values[i].total_cmp(&values[j]).then(i.cmp(&j)));
        let values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        let mut sums = [vec![0.0], vec![0.0], vec![0.0]];
        for (&i, &v) in order.iter().zip(&values) {
            let w = weights[i] / total;
            for (j, s) in sums.iter_mut().enumerate() {
                s.push(s[s.len() - 1] + w * v.powi(j as i32));
            }
        }
        let cuts: Vec<usize> = thresholds
            .iter()
            .map(|t| values.partition_point(|v| v <= t))
            .collect();
        let bounds: Vec<usize> = std::iter::once(0)
            .chain(cuts.iter().copied())
            .chain(std::iter::once(values.len()))
            .collect();
        let (first, last) = (thresholds[0], thresholds[thresholds.len() - 1]);
        let (lo, hi) = tails.unwrap_or((values[0], values[values.len() - 1]));
        Ok(Self {
            lower: lo.min(values[0]).min(first),
            upper: hi.max(values[values.len() - 1]).max(last),
            proportions: cuts.iter().map(|&c| sums[0][c]).collect(),
            classes: bounds.windows(2).map(|w| (w[0], w[1])).collect(),
            sums,
            values,
        })
    }

    fn weight(&self, (s, e): (usize, usize)) -> f64 {
        self.sums[0][e] - self.sums[0][s]
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Data(usize, usize),
    Power(f64),
    Hyperbolic(f64),
}

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

impl Shape {
    /// `E[Z^j · 1{Z > c}]` under the class distribution on `[a, b]`.
    fn above(self, global: &Global, a: f64, b: f64, c: f64, j: usize) -> f64 {
        if let Shape::Data(s, e) = self {
            let p = s + global.values[s..e].partition_point(|v| *v <= c);
            return (global.sums[j][e] - global.sums[j][p]) / global.weight((s, e));
        }
        if b <= a {
            return if a > c { a.powi(j as i32) } else { 0.0 };
        }
        match self {
            Shape::Power(w) => {
                let x0 = ((c - a) / (b - a)).clamp(0.0, 1.0);
                (0..=j)
                    .map(|i| {
                        let e = w + i as f64;
                        binomial(j, i) * a.powi((j - i) as i32) * (b - a).powi(i as i32) * w / e
                            * (1.0 - x0.powf(e))
                    })
                    .sum()
            }
            Shape::Hyperbolic(w) => {
                let (yb, yc) = (b / a, c.clamp(a, b) / a);
                let m = j as f64 - w;
                let integral = if m.abs() < 1e-12 {
                    (yb / yc).ln()
                } else {
                    (yb.powf(m) - yc.powf(m)) / m
                };
                a.powi(j as i32) * w / (1.0 - yb.powf(-w)) * integral
            }
            Shape::Data(..) => unreachable!(),
        }
    }

    /// `∫_w^1 Q(v) dv` over the class quantile function `Q`.
    fn upper_integral(self, global: &Global, a: f64, b: f64, w: f64) -> f64 {
        match self {
            Shape::Data(s, e) => {
                let weight = global.weight((s, e));
                let target = global.sums[0][s] + w * weight;
                let i = s + global.sums[0][s + 1..=e].partition_point(|&c| c <= target);
                if i >= e {
                    return 0.0;
                }
                let partial = (global.sums[0][i + 1] - target) * global.values[i];
                (partial + global.sums[1][e] - global.sums[1][i + 1]) / weight
            }
            _ if b <= a => a * (1.0 - w),
            _ => self.above(global, a, b, self.quantile(global, a, b, w), 1),
        }
    }

    /// Value at probability `u` within the class.
    fn quantile(self, global: &Global, a: f64, b: f64, u: f64) -> f64 {
        match self {
            Shape::Data(s, e) => {
                let target = global.sums[0][s] + u * global.weight((s, e));
                let i = s + global.sums[0][s + 1..=e].partition_point(|&c| c < target);
                global.values[i.min(e - 1)]
            }
            _ if b <= a => a,
            Shape::Power(w) => (a + (b - a) * u.powf(1.0 / w)).clamp(a, b),
            Shape::Hyperbolic(w) => {
                let r = (a / b).powf(w);
                (a * (1.0 - u * (1.0 - r)).powf(-1.0 / w)).clamp(a, b)
            }
        }
    }
}

impl MultipleIndicator {
    pub fn validate(&self) -> Result<()> {
        let t = &self.thresholds;
        if t.is_empty() || t.iter().any(|v| !v.is_finite()) {
            return Err(invalid("need at least one finite threshold"));
        }
        if t.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("thresholds must be strictly increasing"));
        }
        if self.variograms.len() != 1 && self.variograms.len() != t.len() {
            return Err(invalid("need one variogram, or one per threshold"));
        }
        if let Some((lo, hi)) = self.tails
            && !(lo.is_finite() && hi.is_finite() && lo <= hi)
        {
            return Err(invalid("tails must be finite with lower <= upper"));
        }
        match self.upper_tail {
            Some(UpperTail::Power(w) | UpperTail::Hyperbolic(w)) if !(w.is_finite() && w > 0.0) => {
                Err(invalid("the upper tail exponent must be > 0"))
            }
            Some(UpperTail::Hyperbolic(_)) if t[t.len() - 1] <= 0.0 => Err(invalid(
                "a hyperbolic upper tail needs a last threshold > 0",
            )),
            _ => Ok(()),
        }
    }

    /// Variogram whose anisotropy orients the search: the median threshold's.
    pub fn search_variogram(&self) -> &Variogram {
        &self.variograms[self.variograms.len() / 2]
    }

    /// Kriged `P(Z ≤ t)` at each threshold, before correction. `oriented`
    /// replaces every variogram's anisotropy.
    pub fn kriged(
        &self,
        target: &Point,
        samples: &[Sample],
        global: &Global,
        oriented: Option<&Variogram>,
    ) -> Result<Vec<f64>> {
        let kind = if self.simple {
            Kind::Simple { mean: 0.0 }
        } else {
            Kind::Ordinary
        };
        let weights = self
            .variograms
            .iter()
            .map(|vg| {
                let local;
                let vg = match oriented {
                    Some(o) => {
                        local = Variogram {
                            anisotropy: o.anisotropy.clone(),
                            ..vg.clone()
                        };
                        &local
                    }
                    None => vg,
                };
                krige(kind, target, samples, vg).map(|e| e.weights)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .thresholds
            .iter()
            .enumerate()
            .map(|(k, &t)| {
                let w = &weights[k.min(weights.len() - 1)];
                let f = global.proportions[k];
                let indicator = |s: &Sample| if s.value <= t { 1.0 } else { 0.0 };
                if self.simple {
                    f + (0..samples.len())
                        .map(|i| w[i] * (indicator(&samples[i]) - f))
                        .sum::<f64>()
                } else {
                    (0..samples.len())
                        .map(|i| w[i] * indicator(&samples[i]))
                        .sum::<f64>()
                }
            })
            .collect())
    }

    fn classes(&self, global: &Global, cdf: &[f64]) -> Vec<(f64, f64, f64, Shape)> {
        let k = self.thresholds.len();
        (0..=k)
            .map(|c| {
                let a = if c == 0 {
                    global.lower
                } else {
                    self.thresholds[c - 1]
                };
                let b = if c == k {
                    global.upper
                } else {
                    self.thresholds[c]
                };
                let lower = if c == 0 { 0.0 } else { cdf[c - 1] };
                let upper = if c == k { 1.0 } else { cdf[c] };
                let range = global.classes[c];
                let shape = match self.upper_tail {
                    Some(UpperTail::Power(w)) if c == k => Shape::Power(w),
                    Some(UpperTail::Hyperbolic(w)) if c == k => Shape::Hyperbolic(w),
                    _ if self.interpolation == Interpolation::Global
                        && global.weight(range) > 0.0 =>
                    {
                        Shape::Data(range.0, range.1)
                    }
                    _ => Shape::Power(1.0),
                };
                (a, b, upper - lower, shape)
            })
            .collect()
    }

    /// Completes the corrected `cdf` into a distribution and summarizes it.
    pub fn conditional(
        &self,
        global: &Global,
        raw: &[f64],
        cutoffs: &[f64],
        quantiles: &[f64],
    ) -> Conditional {
        let (cdf, correction) = correct_order_relations(raw);
        let classes = self.classes(global, &cdf);
        let moment = |c: f64, j: usize| -> f64 {
            classes
                .iter()
                .filter(|(.., q, _)| *q > 0.0)
                .map(|&(a, b, q, shape)| q * shape.above(global, a, b, c, j))
                .sum()
        };
        let mean = moment(f64::NEG_INFINITY, 1);
        let variance = (moment(f64::NEG_INFINITY, 2) - mean * mean).max(0.0);
        let probability_above: Vec<f64> = cutoffs.iter().map(|&c| moment(c, 0)).collect();
        let mean_above = cutoffs
            .iter()
            .zip(&probability_above)
            .map(|(&c, &p)| if p > 0.0 { moment(c, 1) / p } else { f64::NAN })
            .collect();
        let quantile_values = quantiles
            .iter()
            .map(|&u| {
                let mut below = 0.0;
                let last = classes.iter().rposition(|c| c.2 > 0.0).unwrap_or(0);
                for (i, &(a, b, q, shape)) in classes.iter().enumerate() {
                    if q > 0.0 && (below + q >= u || i == last) {
                        let within = ((u - below) / q).clamp(0.0, 1.0);
                        return shape.quantile(global, a, b, within);
                    }
                    below += q;
                }
                global.upper
            })
            .collect();
        Conditional {
            cdf,
            correction,
            mean,
            variance,
            probability_above,
            mean_above,
            quantile_values,
        }
    }

    /// Conditional distributions at `targets`, each search pass filling the
    /// targets the previous left unestimated. `local` (one entry per target)
    /// orients every variogram and the search. Identical for any number of
    /// threads.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &self,
        samples: &[Sample],
        weights: Option<&[f64]>,
        targets: &[Point],
        searches: &[Search],
        local: Option<&LocalAnisotropy>,
        cutoffs: &[f64],
        quantiles: &[f64],
    ) -> Result<IndicatorSummary> {
        self.validate()?;
        if cutoffs.iter().any(|c| !c.is_finite()) {
            return Err(invalid("cutoffs must be finite"));
        }
        if quantiles.iter().any(|q| !(0.0..=1.0).contains(q)) {
            return Err(invalid("quantiles must be in [0, 1]"));
        }
        let global = self.global(samples, weights)?;
        let results = self.at_targets(samples, &global, targets, searches, local, |raw| {
            self.conditional(&global, &raw, cutoffs, quantiles)
        })?;
        Ok(self.summary(targets.len(), results.into_iter(), cutoffs, quantiles))
    }

    fn global(&self, samples: &[Sample], weights: Option<&[f64]>) -> Result<Global> {
        let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
        Global::new(&values, weights, &self.thresholds, self.tails)
    }

    /// `finish` applied to the kriged `P(Z ≤ t)` at every target, by pass.
    fn at_targets<T: Send>(
        &self,
        samples: &[Sample],
        global: &Global,
        targets: &[Point],
        searches: &[Search],
        local: Option<&LocalAnisotropy>,
        finish: impl Fn(Vec<f64>) -> T + Sync,
    ) -> Result<Vec<Option<T>>> {
        let vg = self.search_variogram();
        let at_target = |t: &Point, s: &[Sample], o: Option<&Variogram>| {
            Ok(finish(self.kriged(t, s, global, o)?))
        };
        let passes = by_pass(targets.len(), searches, |search, remaining| {
            let at: Vec<Point> = remaining.iter().map(|&i| targets[i]).collect();
            match local {
                None => Ok(estimate_many(
                    &at,
                    None,
                    samples,
                    search,
                    Some(vg),
                    |t, s| at_target(t, s, None),
                )),
                Some(l) => {
                    estimate_many_local(&at, None, &l.at(&at), samples, search, vg, |t, s, v| {
                        at_target(t, s, Some(v))
                    })
                }
            }
        })?;
        Ok(passes.into_iter().map(|p| p.map(|p| p.1)).collect())
    }

    /// Means of `n` equal-probability bands of the conditional distribution
    /// completed from the kriged `raw`, ascending; they average to its mean.
    pub fn band_means(&self, global: &Global, raw: &[f64], n: usize) -> Vec<f64> {
        let (cdf, _) = correct_order_relations(raw);
        let classes = self.classes(global, &cdf);
        let tail = |u: f64| -> f64 {
            let mut below = 0.0;
            let mut sum = 0.0;
            for &(a, b, q, shape) in &classes {
                if q > 0.0 && u < below + q {
                    let w = ((u - below) / q).max(0.0);
                    sum += q * shape.upper_integral(global, a, b, w);
                }
                below += q;
            }
            sum
        };
        let mut integrals: Vec<f64> = (0..n).map(|i| tail(i as f64 / n as f64)).collect();
        integrals.push(0.0);
        integrals
            .windows(2)
            .map(|w| n as f64 * (w[0] - w[1]))
            .collect()
    }

    /// Localised grades of the selective blocks `smus` nested in `panels`.
    ///
    /// Each panel's point-support conditional distribution, kriged at its
    /// centroid, takes an affine change of support to the blocks,
    /// `m + √f·(z − m)`; its `n` blocks, ordered by `ranking`, get the means of
    /// its `n` equal-probability bands. `variance_factor` is `f`, in [0, 1];
    /// [`variance_factor`] of the median threshold's variogram when `None`.
    /// Null where a panel is unestimated or outside every panel. Identical for
    /// any number of threads.
    #[allow(clippy::too_many_arguments)]
    pub fn localize(
        &self,
        samples: &[Sample],
        weights: Option<&[f64]>,
        searches: &[Search],
        panels: &BlockModel,
        smus: &BlockModel,
        ranking: &[Option<f64>],
        variance_factor: Option<f64>,
    ) -> Result<Vec<Option<f64>>> {
        self.validate()?;
        let f = match variance_factor {
            Some(f) if !(0.0..=1.0).contains(&f) => {
                return Err(invalid("the variance factor must be in [0, 1]"));
            }
            Some(f) => f,
            None => self::variance_factor(self.search_variogram(), panels, smus)?,
        };
        let global = self.global(samples, weights)?;
        let centroids: Vec<Point> = panels
            .centroids()
            .into_iter()
            .map(|c| (c[0], c[1], c[2]))
            .collect();
        let raw = self.at_targets(samples, &global, &centroids, searches, None, |raw| raw)?;
        transforms::localize::localize(panels, smus, ranking, |p, n| {
            Ok(raw[p].as_ref().map(|raw| {
                let means = self.band_means(&global, raw, n);
                let m = means.iter().sum::<f64>() / n as f64;
                means.iter().map(|z| m + f.sqrt() * (z - m)).collect()
            }))
        })
        .map_err(|e| match e {
            TransformError::InvalidParameters(m) => EstimError::InvalidParameters(m),
            other => EstimError::InvalidParameters(other.to_string()),
        })
    }

    fn summary(
        &self,
        n: usize,
        results: impl Iterator<Item = Option<Conditional>>,
        cutoffs: &[f64],
        quantiles: &[f64],
    ) -> IndicatorSummary {
        let rows = |k: usize| vec![Vec::with_capacity(n); k];
        let mut out = IndicatorSummary {
            thresholds: self.thresholds.clone(),
            cdf: rows(self.thresholds.len()),
            cutoffs: cutoffs.to_vec(),
            probability_above: rows(cutoffs.len()),
            mean_above: rows(cutoffs.len()),
            quantiles: quantiles.to_vec(),
            quantile_values: rows(quantiles.len()),
            ..Default::default()
        };
        let push = |to: &mut Vec<Vec<f64>>, from: Option<&Vec<f64>>| {
            for (k, row) in to.iter_mut().enumerate() {
                row.push(from.map_or(f64::NAN, |f| f[k]));
            }
        };
        for r in results {
            let r = r.as_ref();
            out.correction.push(r.map_or(f64::NAN, |r| r.correction));
            out.mean.push(r.map_or(f64::NAN, |r| r.mean));
            out.variance.push(r.map_or(f64::NAN, |r| r.variance));
            push(&mut out.cdf, r.map(|r| &r.cdf));
            push(&mut out.probability_above, r.map(|r| &r.probability_above));
            push(&mut out.mean_above, r.map(|r| &r.mean_above));
            push(&mut out.quantile_values, r.map(|r| &r.quantile_values));
        }
        out
    }
}

/// Mean covariance, nugget excluded, within a box of `size` discretised by
/// `n` points per axis, in the frame `frame`.
fn mean_covariance(vg: &Variogram, frame: &Matrix3<f64>, size: [f64; 3], n: [usize; 3]) -> f64 {
    let mut points = vec![];
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                let at = |a: usize, i: usize| size[a] * ((i as f64 + 0.5) / n[a] as f64 - 0.5);
                let p = frame.transpose() * Vector3::new(at(0, i), at(1, j), at(2, k));
                points.push((p[0], p[1], p[2]));
            }
        }
    }
    let total: f64 = points
        .iter()
        .map(|p| {
            points
                .iter()
                .map(|q| vg.block_cov_points(p, q))
                .sum::<f64>()
        })
        .sum();
    total / (points.len() * points.len()) as f64
}

/// Variance of the selective blocks within a panel over that of the points
/// within it, `(C̄(v, v) − C̄(V, V)) / (C(0) − C̄(V, V))`, the nugget left
/// out of the block averages. Blocks are discretised by 4 points per axis,
/// panels by 4 per block up to 12 per axis; one point vertically in a grid
/// one panel high.
pub fn variance_factor(vg: &Variogram, panels: &BlockModel, smus: &BlockModel) -> Result<f64> {
    let (p, b) = (panels.geometry(), smus.geometry());
    let frame = block_frame(p.rotation);
    let flat = p.count[2] == 1;
    let per_axis = |a: usize, n: usize| if flat && a == 2 { 1 } else { n };
    let smu = mean_covariance(vg, &frame, b.size, [0, 1, 2].map(|a| per_axis(a, 4)));
    let panel = mean_covariance(
        vg,
        &frame,
        p.size,
        [0, 1, 2].map(|a| {
            per_axis(
                a,
                (4.0 * (p.size[a] / b.size[a]).round()).clamp(4.0, 12.0) as usize,
            )
        }),
    );
    let point = vg.total_sill() - panel;
    if point <= 0.0 {
        return Err(invalid("the variogram has no variance within a panel"));
    }
    Ok(((smu - panel) / point).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::Model;

    fn data(n: usize, seed: u64) -> Vec<Sample> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (x, y) = (next() * 100.0, next() * 100.0);
                let v = (-(1.0 - next()).ln()) * (1.0 + x / 100.0);
                Sample::new((x, y, 0.0), v)
            })
            .collect()
    }

    fn model(thresholds: Vec<f64>, variograms: Vec<Variogram>) -> MultipleIndicator {
        MultipleIndicator {
            thresholds,
            variograms,
            simple: false,
            tails: Some((0.0, 10.0)),
            interpolation: Interpolation::Global,
            upper_tail: None,
        }
    }

    fn search(radius: f64) -> Search {
        Search {
            min_samples: 1,
            max_samples: 12,
            radius,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
            high_grade: None,
            soft: None,
        }
    }

    fn grid() -> Vec<Point> {
        (0..400)
            .map(|i| {
                (
                    (i % 20) as f64 * 5.0 + 1.0,
                    (i / 20) as f64 * 5.0 + 1.0,
                    0.0,
                )
            })
            .collect()
    }

    #[test]
    fn corrected_distributions_are_monotone_in_unit_interval() {
        let samples = data(150, 7);
        let top = samples.iter().map(|s| s.value).fold(10.0, f64::max);
        let thresholds = vec![0.2, 0.5, 0.8, 1.2, 1.8, 2.6];
        let variograms = [10.0, 60.0, 15.0, 80.0, 5.0, 40.0]
            .map(|r| Variogram::single(Model::Gaussian, 1.0, r))
            .to_vec();
        for (upper_tail, interpolation) in [
            (None, Interpolation::Global),
            (None, Interpolation::Linear),
            (Some(UpperTail::Power(0.5)), Interpolation::Global),
            (Some(UpperTail::Hyperbolic(1.5)), Interpolation::Linear),
        ] {
            let m = MultipleIndicator {
                upper_tail,
                interpolation,
                ..model(thresholds.clone(), variograms.clone())
            };
            let quantiles = [0.0, 0.1, 0.5, 0.9, 1.0];
            let s = m
                .predict(
                    &samples,
                    None,
                    &grid(),
                    &[search(40.0)],
                    None,
                    &[1.0],
                    &quantiles,
                )
                .unwrap();
            let mut corrected = 0;
            for i in 0..grid().len() {
                let cdf: Vec<f64> = s.cdf.iter().map(|c| c[i]).collect();
                assert!(cdf.iter().all(|p| (0.0..=1.0).contains(p)), "{cdf:?}");
                assert!(cdf.windows(2).all(|w| w[0] <= w[1]), "{cdf:?}");
                let q: Vec<f64> = s.quantile_values.iter().map(|q| q[i]).collect();
                assert!(q.windows(2).all(|w| w[0] <= w[1]), "{q:?}");
                assert!(
                    q[0] >= 0.0 && q[4] <= top && s.variance[i] >= 0.0,
                    "{q:?} {} {top}",
                    s.variance[i]
                );
                assert!(s.mean[i] >= 0.0 && s.mean[i] <= top);
                corrected += (s.correction[i] > 0.0) as usize;
            }
            assert!(corrected > 0, "the check needs order-relation violations");
        }
    }

    #[test]
    fn one_threshold_equals_indicator_kriging() {
        let samples = data(80, 3);
        let vg = Variogram::single(Model::Spherical, 0.25, 30.0);
        let m = model(vec![1.0], vec![vg.clone()]);
        let s = m
            .predict(&samples, None, &grid(), &[search(25.0)], None, &[], &[])
            .unwrap();
        let ik = estimate_many(&grid(), None, &samples, &search(25.0), Some(&vg), |t, n| {
            krige(Kind::Indicator { threshold: 1.0 }, t, n, &vg)
        });
        for (p, e) in s.cdf[0].iter().zip(ik) {
            match e {
                Some(e) => assert_eq!(*p, e.value),
                None => assert!(p.is_nan()),
            }
        }
    }

    #[test]
    fn simple_form_far_from_data_gives_the_declustered_mean() {
        let samples = data(120, 11);
        let weights: Vec<f64> = (0..120).map(|i| 1.0 + (i % 5) as f64).collect();
        let total: f64 = weights.iter().sum();
        let mean: f64 = samples
            .iter()
            .zip(&weights)
            .map(|(s, w)| s.value * w)
            .sum::<f64>()
            / total;
        let m = MultipleIndicator {
            simple: true,
            ..model(
                vec![0.3, 0.7, 1.5, 2.5],
                vec![Variogram::single(Model::Spherical, 1.0, 20.0)],
            )
        };
        let far = [(1e4, 1e4, 0.0)];
        let s = m
            .predict(
                &samples,
                Some(&weights),
                &far,
                &[search(1e6)],
                None,
                &[],
                &[],
            )
            .unwrap();
        assert!((s.mean[0] - mean).abs() < 1e-10, "{} vs {mean}", s.mean[0]);
    }

    #[test]
    fn output_does_not_depend_on_thread_count() {
        let samples = data(200, 5);
        let m = model(
            vec![0.5, 1.0, 2.0],
            vec![Variogram::single(Model::Exponential, 1.0, 30.0)],
        );
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    m.predict(
                        &samples,
                        None,
                        &grid(),
                        &[search(20.0), search(80.0)],
                        None,
                        &[1.0],
                        &[0.5],
                    )
                    .unwrap()
                })
        };
        let (one, many) = (run(1), run(8));
        assert_eq!(one.mean, many.mean);
        assert_eq!(one.cdf, many.cdf);
        assert_eq!(one.quantile_values, many.quantile_values);
    }

    #[test]
    fn class_moments_match_closed_forms() {
        let global = Global::new(&[1.0, 2.0], None, &[1.5], Some((0.0, 4.0))).unwrap();
        let (a, b) = (2.0, 4.0);
        let linear = Shape::Power(1.0);
        assert!((linear.above(&global, a, b, f64::NEG_INFINITY, 1) - 3.0).abs() < 1e-12);
        assert!((linear.above(&global, a, b, 3.0, 0) - 0.5).abs() < 1e-12);
        let h = Shape::Hyperbolic(1.0);
        let expected = a * (b / a).ln() / (1.0 - a / b);
        assert!((h.above(&global, a, b, f64::NEG_INFINITY, 1) - expected).abs() < 1e-12);
        assert!((h.above(&global, a, b, f64::NEG_INFINITY, 0) - 1.0).abs() < 1e-12);
        for shape in [linear, h, Shape::Power(0.4)] {
            let median = shape.quantile(&global, a, b, 0.5);
            assert!((shape.above(&global, a, b, median, 0) - 0.5).abs() < 1e-12);
        }
    }

    fn panels(rotation: f64) -> BlockModel {
        let g = ceres_core::Geometry {
            origin: [0.0; 3],
            size: [20.0, 20.0, 1.0],
            count: [5, 5, 1],
            rotation: [rotation, 0.0, 0.0],
        };
        let batch = arrow_array::RecordBatch::try_new_with_options(
            std::sync::Arc::new(arrow_schema::Schema::empty()),
            vec![],
            &arrow_array::RecordBatchOptions::new().with_row_count(Some(g.cells() as usize)),
        )
        .unwrap();
        BlockModel::regular(g, batch).unwrap()
    }

    struct Localised {
        panel: Vec<usize>,
        ranking: Vec<Option<f64>>,
        grades: Vec<f64>,
    }

    fn localised(m: &MultipleIndicator, samples: &[Sample], f: Option<f64>) -> Localised {
        let (panels, smus) = (panels(0.0), panels(0.0).discretize([4, 4, 1]).unwrap());
        let ranking: Vec<Option<f64>> = smus
            .centroids()
            .iter()
            .map(|c| Some((c[0] * 0.37).sin() + c[1] * 0.01))
            .collect();
        let grades = m
            .localize(samples, None, &[search(60.0)], &panels, &smus, &ranking, f)
            .unwrap()
            .into_iter()
            .map(Option::unwrap)
            .collect();
        let panel = transforms::localize::nest(&panels, &smus)
            .unwrap()
            .into_iter()
            .map(Option::unwrap)
            .collect();
        Localised {
            panel,
            ranking,
            grades,
        }
    }

    impl Localised {
        /// Each panel's grades in ascending rank.
        fn ranked(&self, panel: usize) -> Vec<f64> {
            let mut rows: Vec<usize> = (0..self.panel.len())
                .filter(|&r| self.panel[r] == panel)
                .collect();
            rows.sort_by(|&i, &j| {
                self.ranking[i]
                    .unwrap()
                    .total_cmp(&self.ranking[j].unwrap())
            });
            rows.iter().map(|&r| self.grades[r]).collect()
        }
    }

    fn centroids() -> Vec<Point> {
        panels(0.0)
            .centroids()
            .iter()
            .map(|c| (c[0], c[1], c[2]))
            .collect()
    }

    fn kriged_panels(m: &MultipleIndicator, samples: &[Sample]) -> (Global, Vec<Vec<f64>>) {
        let global = m.global(samples, None).unwrap();
        let raw = m
            .at_targets(samples, &global, &centroids(), &[search(60.0)], None, |r| r)
            .unwrap();
        (global, raw.into_iter().map(Option::unwrap).collect())
    }

    #[test]
    fn localisation_without_change_of_support_gives_point_band_means() {
        let samples = data(150, 13);
        let m = model(
            vec![0.3, 0.7, 1.2, 2.0],
            vec![Variogram::single(Model::Spherical, 1.0, 40.0)],
        );
        let out = localised(&m, &samples, Some(1.0));
        let (global, raw) = kriged_panels(&m, &samples);
        for (p, raw) in raw.iter().enumerate() {
            let expected = m.band_means(&global, raw, 16);
            for (g, e) in out.ranked(p).iter().zip(&expected) {
                assert!((g - e).abs() < 1e-12, "{g} vs {e}");
            }
        }
    }

    #[test]
    fn localised_panels_keep_their_mean_rank_order_and_curve() {
        let samples = data(200, 17);
        let f = 0.45;
        for interpolation in [Interpolation::Global, Interpolation::Linear] {
            let m = MultipleIndicator {
                interpolation,
                upper_tail: Some(UpperTail::Hyperbolic(2.0)),
                ..model(
                    vec![0.3, 0.7, 1.2, 2.0],
                    vec![Variogram::single(Model::Spherical, 1.0, 40.0)],
                )
            };
            let out = localised(&m, &samples, Some(f));
            let tonnages: Vec<f64> = (1..16).map(|k| 1.0 - k as f64 / 16.0).collect();
            let s = m
                .predict(
                    &samples,
                    None,
                    &centroids(),
                    &[search(60.0)],
                    None,
                    &[],
                    &tonnages,
                )
                .unwrap();
            for p in 0..25 {
                let g = out.ranked(p);
                let mean = s.mean[p];
                assert!((g.iter().sum::<f64>() / 16.0 - mean).abs() < 1e-9);
                assert!(g.windows(2).all(|w| w[0] <= w[1] + 1e-12), "{g:?}");
                if interpolation == Interpolation::Linear {
                    let cutoffs: Vec<f64> = s.quantile_values.iter().map(|q| q[p]).collect();
                    let (global, raw) = kriged_panels(&m, &samples);
                    let point = m.conditional(&global, &raw[p], &cutoffs, &[]);
                    for k in 1..16 {
                        let top = g[16 - k..].iter().sum::<f64>() / k as f64;
                        let curve = mean + f.sqrt() * (point.mean_above[k - 1] - mean);
                        assert!((top - curve).abs() < 1e-9, "{top} vs {curve} at {k}/16");
                    }
                }
            }
        }
    }

    #[test]
    fn localisation_does_not_depend_on_thread_count() {
        let samples = data(200, 19);
        let m = model(
            vec![0.5, 1.0, 2.0],
            vec![Variogram::single(Model::Exponential, 1.0, 30.0)],
        );
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| localised(&m, &samples, None).grades)
        };
        assert_eq!(run(1), run(8));
    }

    #[test]
    fn variance_factor_falls_with_block_size_and_is_zero_for_whole_panels() {
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let p = panels(30.0);
        let f = |n: usize| variance_factor(&vg, &p, &p.discretize([n, n, 1]).unwrap()).unwrap();
        assert_eq!(f(1), 0.0);
        let (two, four) = (f(2), f(4));
        assert!(0.0 < two && two < four && four < 1.0, "{two} {four}");
    }

    #[test]
    fn order_relations_average_both_passes() {
        let (cdf, correction) = correct_order_relations(&[0.3, 0.2, 1.1]);
        assert_eq!(cdf, vec![0.25, 0.25, 1.0]);
        assert!((correction - 0.2).abs() < 1e-12);
    }
}
