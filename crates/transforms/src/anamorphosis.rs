//! Hermite (Gaussian) anamorphosis.
//!
//! Reframes the normal-score transform as an analytic function
//! `Z = φ(Y) = Σₙ ψₙ Hₙ(Y)` mapping a standard-normal `Y` to the data
//! distribution `Z`, expressed in the orthonormal Hermite basis
//! ([`crate::hermite`]). Unlike the raw lookup-table normal score
//! ([`crate::nscore`]), the analytic form supports:
//!
//! - change-of-support (discrete Gaussian model, [`crate::dgm`]),
//! - grade–tonnage / selectivity by closed-form integrals ([`crate::selectivity`]),
//! - uniform conditioning ([`crate::uc`]),
//! - disjunctive kriging (factor-by-factor kriging of `Hₙ(Y)`).
//!
//! Coefficients are fitted from the (optionally declustered) empirical CDF: the
//! sorted data define a step function `z(y)` over Gaussian class intervals, and
//! `ψₙ = ∫ z(y) Hₙ(y) g(y) dy` is evaluated exactly per interval with
//! [`hermite::interval_integral`].

use crate::error::{Result, TransformError};
use crate::hermite::{self, interval_integral};
use crate::normal::probit;
use serde::{Deserialize, Serialize};

/// A fitted Hermite anamorphosis `Z = φ(Y)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HermiteAnamorphosis {
    /// Hermite coefficients `ψ₀..ψ_P` (`ψ₀` is the mean of `Z`).
    pub coefficients: Vec<f64>,
    /// Lower Gaussian bound of the monotone (authorized) interval.
    pub y_min: f64,
    /// Upper Gaussian bound of the monotone (authorized) interval.
    pub y_max: f64,
    /// `φ(y_min)` — data value at the lower bound.
    pub z_min: f64,
    /// `φ(y_max)` — data value at the upper bound.
    pub z_max: f64,
}

impl HermiteAnamorphosis {
    /// Fit an anamorphosis of Hermite degree `degree` (number of terms `= degree + 1`).
    ///
    /// `weights` (optional) are declustering weights for the empirical CDF.
    pub fn fit(values: &[f64], weights: Option<&[f64]>, degree: usize) -> Result<Self> {
        let n = values.len();
        if n == 0 {
            return Err(TransformError::InsufficientData("no values".into()));
        }
        if degree == 0 {
            return Err(TransformError::InvalidParameters(
                "degree must be ≥ 1".into(),
            ));
        }
        if let Some(w) = weights
            && w.len() != n
        {
            return Err(TransformError::InvalidParameters(
                "weights length mismatch".into(),
            ));
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let w: Vec<f64> = match weights {
            Some(w) => {
                let s: f64 = w.iter().sum();
                if s <= 0.0 {
                    return Err(TransformError::InvalidParameters("weights sum ≤ 0".into()));
                }
                w.iter().map(|x| x / s).collect()
            }
            None => vec![1.0 / n as f64; n],
        };

        // Gaussian class bounds: y_i = Φ⁻¹(cumulative frequency at class edge).
        // Class i (ordered value z_(i)) spans the Gaussian interval (yb[i], yb[i+1]).
        let mut yb = Vec::with_capacity(n + 1);
        yb.push(f64::NEG_INFINITY);
        let mut cum = 0.0;
        for &idx in &order {
            cum += w[idx];
            // Clamp to avoid ±∞ at the extreme edge.
            let p = cum.min(1.0 - 1e-12);
            yb.push(if cum >= 1.0 - 1e-15 {
                f64::INFINITY
            } else {
                probit(p)
            });
        }
        *yb.last_mut().unwrap() = f64::INFINITY;

        // ψₙ = Σᵢ z_(i) · ∫_{yb[i]}^{yb[i+1]} Hₙ(y) g(y) dy.
        let mut psi = vec![0.0; degree + 1];
        for (ci, &idx) in order.iter().enumerate() {
            let z = values[idx];
            let (a, b) = (yb[ci], yb[ci + 1]);
            for (nn, p) in psi.iter_mut().enumerate() {
                *p += z * interval_integral(a, b, nn);
            }
        }

        let (y_min, y_max) = monotone_interval(&psi);
        let z_min = eval(&psi, y_min);
        let z_max = eval(&psi, y_max);

        Ok(Self {
            coefficients: psi,
            y_min,
            y_max,
            z_min,
            z_max,
        })
    }

    /// Hermite degree (highest term index `P`).
    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Mean of `Z` (`= ψ₀`).
    pub fn mean(&self) -> f64 {
        self.coefficients[0]
    }

    /// Point-support variance of `Z`: `Σ_{n≥1} ψₙ²`.
    pub fn variance(&self) -> f64 {
        self.coefficients[1..].iter().map(|p| p * p).sum()
    }

    /// Back-transform Gaussian → data: evaluate `φ(y)`, clamped to the
    /// authorized interval `[y_min, y_max]`.
    pub fn back(&self, y: f64) -> f64 {
        let yc = y.clamp(self.y_min, self.y_max);
        eval(&self.coefficients, yc)
    }

    /// Forward transform data → Gaussian: solve `φ(y) = z` by bisection over the
    /// authorized (monotone) interval. `z` is clamped to `[z_min, z_max]`.
    pub fn forward(&self, z: f64) -> f64 {
        let zc = z.clamp(self.z_min.min(self.z_max), self.z_min.max(self.z_max));
        let (mut lo, mut hi) = (self.y_min, self.y_max);
        // φ increasing on [y_min, y_max]; bisect.
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            let f = eval(&self.coefficients, mid);
            if f < zc {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-10 {
                break;
            }
        }
        0.5 * (lo + hi)
    }

    /// Block anamorphosis under the discrete Gaussian model: coefficients become
    /// `ψₙ·rⁿ` for the change-of-support coefficient `r ∈ (0, 1]`. Its variance
    /// is `Σ_{n≥1} ψₙ² r^{2n} ≤` the point variance.
    ///
    /// See [`crate::dgm`] for solving `r` from a variogram.
    pub fn block(&self, r: f64) -> HermiteAnamorphosis {
        let coeffs: Vec<f64> = self
            .coefficients
            .iter()
            .enumerate()
            .map(|(n, &p)| p * r.powi(n as i32))
            .collect();
        let (y_min, y_max) = monotone_interval(&coeffs);
        let z_min = eval(&coeffs, y_min);
        let z_max = eval(&coeffs, y_max);
        HermiteAnamorphosis {
            coefficients: coeffs,
            y_min,
            y_max,
            z_min,
            z_max,
        }
    }
}

/// Evaluate `φ(y) = Σₙ ψₙ Hₙ(y)`.
pub fn eval(psi: &[f64], y: f64) -> f64 {
    let h = hermite::polynomials(y, psi.len() - 1);
    psi.iter().zip(&h).map(|(p, hh)| p * hh).sum()
}

/// Derivative `φ'(y) = Σₙ ψₙ Hₙ'(y)`, with `Hₙ'(y) = −√n H_{n-1}(y)`.
fn eval_deriv(psi: &[f64], y: f64) -> f64 {
    let deg = psi.len() - 1;
    if deg == 0 {
        return 0.0;
    }
    let h = hermite::polynomials(y, deg);
    let mut acc = 0.0;
    for n in 1..=deg {
        acc += psi[n] * (-(n as f64).sqrt()) * h[n - 1];
    }
    acc
}

/// Widest interval around `y = 0` on which `φ` is monotone increasing,
/// scanning `[-4, 4]`. Anamorphosis inversion is only authorized here.
fn monotone_interval(psi: &[f64]) -> (f64, f64) {
    const LIMIT: f64 = 4.0;
    const STEP: f64 = 0.01;
    let mut y_min = -LIMIT;
    let mut y_max = LIMIT;
    let mut y = 0.0;
    while y > -LIMIT {
        if eval_deriv(psi, y) <= 0.0 {
            y_min = y;
            break;
        }
        y -= STEP;
    }
    y = 0.0;
    while y < LIMIT {
        if eval_deriv(psi, y) <= 0.0 {
            y_max = y;
            break;
        }
        y += STEP;
    }
    (y_min, y_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_matches_data_mean() {
        let vals: Vec<f64> = (1..=200).map(|i| i as f64 * 0.1).collect();
        let anam = HermiteAnamorphosis::fit(&vals, None, 30).unwrap();
        let data_mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            (anam.mean() - data_mean).abs() < 1e-6,
            "mean {}",
            anam.mean()
        );
    }

    #[test]
    fn variance_approximates_data_variance() {
        // Lognormal-ish sample; enough terms → anamorphosis variance ≈ data variance.
        let vals: Vec<f64> = (1..=500).map(|i| ((i as f64) / 50.0).exp()).collect();
        let anam = HermiteAnamorphosis::fit(&vals, None, 60).unwrap();
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let dv = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64;
        // Truncated expansion under-represents variance slightly; allow 12%.
        assert!(
            anam.variance() <= dv * 1.02,
            "anam var {} data var {dv}",
            anam.variance()
        );
        assert!(
            anam.variance() >= dv * 0.85,
            "anam var {} data var {dv}",
            anam.variance()
        );
    }

    #[test]
    fn round_trip_forward_back() {
        let vals: Vec<f64> = (1..=300).map(|i| ((i as f64) / 30.0).exp()).collect();
        let anam = HermiteAnamorphosis::fit(&vals, None, 50).unwrap();
        // Round-trip is only defined on the authorized (monotone) interval;
        // sample fractions of it around the median.
        let span = anam.y_max - anam.y_min;
        for f in &[0.15, 0.35, 0.5, 0.65, 0.85] {
            let y = anam.y_min + f * span;
            let z = anam.back(y);
            let y2 = anam.forward(z);
            assert!((y - y2).abs() < 0.05, "y {y} -> z {z} -> {y2}");
        }
    }

    #[test]
    fn back_is_monotone() {
        let vals: Vec<f64> = (1..=200).map(|i| (i as f64).sqrt()).collect();
        let anam = HermiteAnamorphosis::fit(&vals, None, 40).unwrap();
        let a = anam.back(-2.0);
        let b = anam.back(0.0);
        let c = anam.back(2.0);
        assert!(a < b && b < c, "not monotone: {a} {b} {c}");
    }

    #[test]
    fn block_variance_reduces() {
        let vals: Vec<f64> = (1..=300).map(|i| ((i as f64) / 30.0).exp()).collect();
        let anam = HermiteAnamorphosis::fit(&vals, None, 50).unwrap();
        let point_var = anam.variance();
        let block = anam.block(0.7);
        assert!(block.variance() < point_var, "block var not reduced");
        // r = 1 → identity.
        let same = anam.block(1.0);
        assert!((same.variance() - point_var).abs() < 1e-9);
        // Mean preserved by change of support (ψ₀ unchanged).
        assert!((block.mean() - anam.mean()).abs() < 1e-12);
    }
}
