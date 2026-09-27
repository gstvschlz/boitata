//! Weighted Gaussian kernel density of one variable, a smooth reference
//! distribution for normal scores when the data are too few to define the
//! tails.
//!
//! Bounds are honored by reflecting the kernels at them; in log space the
//! kernels sit on `ln x`, so the density stays above zero with a tail
//! that is lognormal-like.

use crate::error::{Result, TransformError};
use crate::normal::phi;
use crate::normal_score::{Reference, invert};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// How the kernel width is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Bandwidth {
    /// `0.9 min(sd, IQR / 1.34) n^-1/5`, robust to a skewed or bimodal sample.
    Silverman,
    /// `1.06 sd n^-1/5`, optimal for Gaussian data.
    Scott,
    /// A given width, in log units when the density is in log space.
    Given(f64),
}

/// Fitted kernel density; `n` is the effective sample size of the weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDensity {
    centers: Vec<f64>,
    weights: Vec<f64>,
    bandwidth: f64,
    log: bool,
    /// Bounds in kernel space (log units in log space), infinite when open.
    #[serde(with = "ceres_core::nonfinite")]
    bounds: (f64, f64),
    /// Kernel mass inside the bounds, reflections included.
    mass: f64,
}

/// Weighted quantile of sorted `x` with normalized weights.
fn quantile(x: &[f64], w: &[f64], p: f64) -> f64 {
    let mut cum = 0.0;
    let mids: Vec<f64> = w
        .iter()
        .map(|wi| {
            cum += wi;
            cum - wi / 2.0
        })
        .collect();
    let k = mids.partition_point(|m| *m < p);
    match k {
        0 => x[0],
        k if k == x.len() => x[k - 1],
        k => x[k - 1] + (x[k] - x[k - 1]) * (p - mids[k - 1]) / (mids[k] - mids[k - 1]),
    }
}

impl KernelDensity {
    /// Fits to `values` with optional `weights` (e.g. declustering); `lower`
    /// and `upper` bound the density by reflection, and `log` puts the
    /// kernels on `ln x`.
    pub fn fit(
        values: &[f64],
        weights: Option<&[f64]>,
        bandwidth: Bandwidth,
        lower: Option<f64>,
        upper: Option<f64>,
        log: bool,
    ) -> Result<Self> {
        let bad = |m: &str| Err(TransformError::InvalidParameters(m.into()));
        if values.len() < 2 {
            return Err(TransformError::InsufficientData(
                "fewer than 2 values".into(),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return bad("values must be finite");
        }
        if let Some(w) = weights
            && (w.len() != values.len()
                || w.iter().any(|&x| !x.is_finite() || x < 0.0)
                || w.iter().sum::<f64>() <= 0.0)
        {
            return bad("weights must match values, be ≥ 0 and sum > 0");
        }
        let (lo, hi) = (
            lower.unwrap_or(f64::NEG_INFINITY),
            upper.unwrap_or(f64::INFINITY),
        );
        if lo.is_nan() || hi.is_nan() || lo >= hi {
            return bad("lower must be below upper");
        }
        if values.iter().any(|&v| v < lo || v > hi) {
            return bad("values must lie within lower and upper");
        }
        if log && values.iter().any(|&v| v <= 0.0) {
            return bad("log space needs positive values");
        }
        let space = |x: f64| match (log, x) {
            (true, x) if x <= 0.0 => f64::NEG_INFINITY,
            (true, x) => x.ln(),
            (false, x) => x,
        };
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let centers: Vec<f64> = order.iter().map(|&i| space(values[i])).collect();
        let total = weights.map_or(values.len() as f64, |w| w.iter().sum());
        let w: Vec<f64> = order
            .iter()
            .map(|&i| weights.map_or(1.0, |w| w[i]) / total)
            .collect();
        let mean: f64 = centers.iter().zip(&w).map(|(c, w)| c * w).sum();
        let sd = centers
            .iter()
            .zip(&w)
            .map(|(c, w)| w * (c - mean).powi(2))
            .sum::<f64>()
            .sqrt();
        let n = 1.0 / w.iter().map(|w| w * w).sum::<f64>();
        let iqr = (quantile(&centers, &w, 0.75) - quantile(&centers, &w, 0.25)) / 1.34;
        let h = match bandwidth {
            Bandwidth::Silverman if iqr > 0.0 => 0.9 * sd.min(iqr) * n.powf(-0.2),
            Bandwidth::Silverman => 0.9 * sd * n.powf(-0.2),
            Bandwidth::Scott => 1.06 * sd * n.powf(-0.2),
            Bandwidth::Given(h) => h,
        };
        if !(h.is_finite() && h > 0.0) {
            return bad("the bandwidth must be positive; are the values constant?");
        }
        let mut kde = Self {
            centers,
            weights: w,
            bandwidth: h,
            log,
            bounds: (space(lo), space(hi)),
            mass: 1.0,
        };
        let (a, b) = kde.bounds;
        kde.mass = kde.raw_cdf(b) - kde.raw_cdf(a);
        Ok(kde)
    }

    /// Kernel width, in log units in log space.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Each center with its reflections at the finite bounds.
    fn images(&self, c: f64) -> impl Iterator<Item = f64> {
        let (a, b) = self.bounds;
        [
            Some(c),
            a.is_finite().then_some(2.0 * a - c),
            b.is_finite().then_some(2.0 * b - c),
        ]
        .into_iter()
        .flatten()
    }

    fn raw_cdf(&self, t: f64) -> f64 {
        match t {
            t if t == f64::NEG_INFINITY => 0.0,
            t => self
                .centers
                .iter()
                .zip(&self.weights)
                .map(|(&c, w)| {
                    w * self
                        .images(c)
                        .map(|m| phi((t - m) / self.bandwidth))
                        .sum::<f64>()
                })
                .sum(),
        }
    }

    fn to_space(&self, x: f64) -> f64 {
        match (self.log, x) {
            (true, x) if x <= 0.0 => f64::NEG_INFINITY,
            (true, x) => x.ln(),
            (false, x) => x,
        }
    }

    /// Density at `x`, zero outside the bounds.
    pub fn pdf(&self, x: f64) -> f64 {
        let (a, b) = self.bounds;
        let t = self.to_space(x);
        if t < a || t > b || !t.is_finite() {
            return 0.0;
        }
        let h = self.bandwidth;
        let g: f64 = self
            .centers
            .iter()
            .zip(&self.weights)
            .map(|(&c, w)| {
                w * self
                    .images(c)
                    .map(|m| (-0.5 * ((t - m) / h).powi(2)).exp())
                    .sum::<f64>()
            })
            .sum::<f64>()
            / (h * (2.0 * std::f64::consts::PI).sqrt() * self.mass);
        if self.log { g / x } else { g }
    }

    /// Probability of a value at most `x`.
    pub fn cdf(&self, x: f64) -> f64 {
        let (a, b) = self.bounds;
        let t = self.to_space(x).clamp(a, b);
        ((self.raw_cdf(t) - self.raw_cdf(a)) / self.mass).clamp(0.0, 1.0)
    }

    /// `n` draws; the same `seed` gives the same draws.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut cum = 0.0;
        let cdf: Vec<f64> = self
            .weights
            .iter()
            .map(|w| {
                cum += w;
                cum
            })
            .collect();
        let (a, b) = self.bounds;
        let images = 1 + a.is_finite() as usize + b.is_finite() as usize;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u: f64 = rng.gen_range(0.0..cum);
            let i = cdf.partition_point(|c| *c <= u).min(cdf.len() - 1);
            let m = self
                .images(self.centers[i])
                .nth(rng.gen_range(0..images))
                .unwrap();
            let e: f64 = StandardNormal.sample(&mut rng);
            let t = m + self.bandwidth * e;
            if t >= a && t <= b {
                out.push(if self.log { t.exp() } else { t });
            }
        }
        out
    }
}

impl Reference for KernelDensity {
    fn quantile(&self, p: f64) -> f64 {
        let (a, b) = self.bounds;
        let span = 12.0 * self.bandwidth;
        let (first, last) = (self.centers[0], self.centers[self.centers.len() - 1]);
        let lo = (first - span).max(a);
        let hi = (last + span).min(b);
        let t = invert(
            |t| (self.raw_cdf(t) - self.raw_cdf(a)) / self.mass,
            lo,
            hi,
            p,
        );
        if self.log { t.exp() } else { t }
    }

    fn support(&self) -> (f64, f64) {
        let (a, b) = self.bounds;
        match self.log {
            true => (a.exp(), b.exp()),
            false => (a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lognormal(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let z: f64 = StandardNormal.sample(&mut rng);
                (0.8 * z).exp()
            })
            .collect()
    }

    /// Trapezoid integrals of the density and of `x` times it over `[lo, hi]`.
    fn moments(kde: &KernelDensity, lo: f64, hi: f64) -> (f64, f64) {
        let n = 200_000;
        let dx = (hi - lo) / n as f64;
        let (mut mass, mut mean) = (0.0, 0.0);
        for i in 0..=n {
            let x = lo + i as f64 * dx;
            let f = kde.pdf(x) * if i == 0 || i == n { 0.5 } else { 1.0 } * dx;
            mass += f;
            mean += x * f;
        }
        (mass, mean)
    }

    #[test]
    fn integrates_to_one_and_keeps_the_weighted_mean() {
        let values = lognormal(80, 1);
        let weights: Vec<f64> = (0..80).map(|i| 0.5 + (i % 4) as f64).collect();
        let target = values.iter().zip(&weights).map(|(v, w)| v * w).sum::<f64>()
            / weights.iter().sum::<f64>();
        for bandwidth in [
            Bandwidth::Silverman,
            Bandwidth::Scott,
            Bandwidth::Given(0.3),
        ] {
            let kde =
                KernelDensity::fit(&values, Some(&weights), bandwidth, None, None, false).unwrap();
            let (mass, mean) = moments(&kde, -10.0, 20.0);
            assert!((mass - 1.0).abs() < 1e-6, "mass {mass}");
            assert!((mean - target).abs() < 1e-6, "mean {mean} vs {target}");
            assert!((kde.cdf(20.0) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn bounds_are_respected() {
        let values = lognormal(60, 2);
        let reflected =
            KernelDensity::fit(&values, None, Bandwidth::Scott, Some(0.0), None, false).unwrap();
        let logged =
            KernelDensity::fit(&values, None, Bandwidth::Silverman, Some(0.0), None, true).unwrap();
        let capped = KernelDensity::fit(
            &values,
            None,
            Bandwidth::Scott,
            Some(0.0),
            Some(12.0),
            false,
        )
        .unwrap();
        for kde in [&reflected, &logged, &capped] {
            assert_eq!(kde.pdf(-0.1), 0.0);
            assert_eq!(kde.cdf(0.0), 0.0);
            let (mass, _) = moments(kde, 0.0, 60.0);
            assert!((mass - 1.0).abs() < 1e-4, "mass {mass}");
            let draws = kde.sample(20_000, 5);
            assert!(draws.iter().all(|&x| x >= 0.0 && x.is_finite()));
            assert_eq!(draws, kde.sample(20_000, 5));
            assert!(kde.quantile(1e-9) >= 0.0);
            // Draws follow the density.
            let q = kde.quantile(0.9);
            let above = draws.iter().filter(|&&x| x > q).count() as f64 / 20_000.0;
            assert!((above - 0.1).abs() < 0.01, "{above}");
        }
        assert!(capped.sample(5000, 1).iter().all(|&x| x <= 12.0));
        assert_eq!(capped.pdf(12.5), 0.0);
        assert_eq!(logged.support().0, 0.0);
        assert!(
            KernelDensity::fit(&values, None, Bandwidth::Scott, Some(1.0), None, false).is_err()
        );
    }

    #[test]
    fn scores_against_the_reference_round_trip() {
        let values = lognormal(40, 3);
        let kde =
            KernelDensity::fit(&values, None, Bandwidth::Silverman, Some(0.0), None, true).unwrap();
        let ns = crate::normal_score::from_reference(&values, &kde);
        for (&v, &z) in values.iter().zip(&ns.scores) {
            assert!((z - crate::normal::probit(kde.cdf(v))).abs() < 1e-3);
            assert!((ns.table.back(z) - v).abs() < 1e-6 * v.max(1.0));
        }
        assert_eq!(ns.table.tails.0, 0.0);
        assert!(ns.table.back(-8.0) >= 0.0);
    }
}
