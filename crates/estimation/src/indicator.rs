//! Multiple indicator kriging: the conditional distribution `P(Z ≤ t | data)`
//! kriged at many thresholds, corrected for order relations, completed within
//! classes and in the tails, then summarized.
//!
//! Classes run from the lower tail bound through the thresholds to the upper
//! tail bound. Within a class the distribution follows the declustered data
//! falling in it (`Interpolation::Global`) or is uniform (`Linear`); the last
//! class may take a power or hyperbolic model instead.

use serde::{Deserialize, Serialize};
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
        let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
        let global = Global::new(&values, weights, &self.thresholds, self.tails)?;
        let vg = self.search_variogram();
        let at_target = |t: &Point, s: &[Sample], o: Option<&Variogram>| {
            let raw = self.kriged(t, s, &global, o)?;
            Ok(self.conditional(&global, &raw, cutoffs, quantiles))
        };
        let passes = by_pass(targets.len(), searches, |search, remaining| {
            let at: Vec<Point> = remaining.iter().map(|&i| targets[i]).collect();
            match local {
                None => Ok(estimate_many(&at, samples, search, Some(vg), |t, s| {
                    at_target(t, s, None)
                })),
                Some(l) => estimate_many_local(&at, &l.at(&at), samples, search, vg, |t, s, v| {
                    at_target(t, s, Some(v))
                }),
            }
        })?;
        Ok(self.summary(
            targets.len(),
            passes.into_iter().map(|p| p.map(|p| p.1)),
            cutoffs,
            quantiles,
        ))
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
        let ik = estimate_many(&grid(), &samples, &search(25.0), Some(&vg), |t, n| {
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

    #[test]
    fn order_relations_average_both_passes() {
        let (cdf, correction) = correct_order_relations(&[0.3, 0.2, 1.1]);
        assert_eq!(cdf, vec![0.25, 0.25, 1.0]);
        assert!((correction - 0.2).abs() < 1e-12);
    }
}
