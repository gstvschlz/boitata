//! Disjunctive Kriging (DK) via Hermite factors.
//!
//! DK estimates a *function* of the grade (the grade itself, a cutoff
//! indicator, recovered tonnage) by simple-kriging each Hermite factor
//! `Hₙ(Y)` independently and recombining. It sits between ordinary kriging
//! (only the mean) and conditional simulation (the full distribution): it gives
//! conditional expectations of any transform without simulating.
//!
//! The factors are the orthonormal Hermite polynomials of the Gaussian
//! transform `Y = φ⁻¹(Z)` ([`transforms::HermiteAnamorphosis`]). Under the
//! discrete Gaussian assumption, `Hₙ(Y(x))` has mean 0 (for `n ≥ 1`), unit
//! variance, and covariance `ρ(h)ⁿ`, where `ρ` is the correlogram of `Y`. So
//! each factor is simple-kriged (mean 0) with the power-`n` covariance, then:
//!
//! ```text
//! grade:    Ẑ(x₀)      = Σₙ ψₙ · [Hₙ(Y)]ˢᵏ(x₀)
//! tonnage:  T̂(z_c, x₀) = Σₙ φₙ(y_c) · [Hₙ(Y)]ˢᵏ(x₀),   φₙ = ∫_{y_c}^∞ Hₙ g
//! ```

use crate::error::{EstimError, Result};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use transforms::HermiteAnamorphosis;
use transforms::hermite::{polynomials, upper_tail_integral};
use variogram::Variogram;

/// A Gaussian-transformed sample: location and its normal score `Y = φ⁻¹(Z)`.
#[derive(Debug, Clone)]
pub struct GaussianSample {
    pub loc: (f64, f64, f64),
    pub y: f64,
}

/// Disjunctive kriging over a fixed point anamorphosis.
#[derive(Serialize, Deserialize)]
pub struct DisjunctiveKriging {
    anam: HermiteAnamorphosis,
}

impl DisjunctiveKriging {
    pub fn new(anam: HermiteAnamorphosis) -> Self {
        Self { anam }
    }

    /// Simple-krige every Hermite factor `H₀..H_order` at `target`.
    ///
    /// `vg_y` is the variogram of the Gaussian `Y` (any sill; normalized to the
    /// correlogram internally). Returns `factors[n] = [Hₙ(Y)]ˢᵏ(target)`, with
    /// `factors[0] = 1`.
    pub fn factors(
        &self,
        target: &(f64, f64, f64),
        samples: &[GaussianSample],
        vg_y: &Variogram,
        order: usize,
    ) -> Result<Vec<f64>> {
        let n = samples.len();
        if n == 0 {
            return Err(EstimError::InsufficientData("no samples".into()));
        }
        let sill = vg_y.total_sill();
        if sill <= 0.0 {
            return Err(EstimError::InvalidParameters(
                "Gaussian variogram sill ≤ 0".into(),
            ));
        }

        // Correlogram ρ between all sample pairs and sample↔target.
        let mut rho = DMatrix::<f64>::zeros(n, n);
        let mut rho0 = DVector::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                rho[(i, j)] = vg_y.cov_points(&samples[i].loc, &samples[j].loc) / sill;
            }
            rho0[i] = vg_y.cov_points(&samples[i].loc, target) / sill;
        }

        let hvals: Vec<Vec<f64>> = samples.iter().map(|s| polynomials(s.y, order)).collect();

        let mut factors = vec![0.0; order + 1];
        factors[0] = 1.0; // H₀ ≡ 1
        for k in 1..=order {
            // Factor-k covariance = ρᵏ (simple kriging, mean 0).
            let mut a = DMatrix::<f64>::zeros(n, n);
            let mut b = DVector::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = rho[(i, j)].powi(k as i32);
                }
                b[i] = rho0[i].powi(k as i32);
            }
            let lambda = a
                .lu()
                .solve(&b)
                .ok_or_else(|| EstimError::Singular(format!("DK factor {k} singular")))?;
            factors[k] = (0..n).map(|i| lambda[i] * hvals[i][k]).sum();
        }
        Ok(factors)
    }

    /// Recombine factors into a grade estimate `Σₙ ψₙ·factorₙ`.
    pub fn grade(&self, factors: &[f64]) -> f64 {
        self.anam
            .coefficients
            .iter()
            .zip(factors)
            .map(|(psi, f)| psi * f)
            .sum()
    }

    /// Recombine factors into recovered tonnage `P(Z ≥ cutoff)` above `cutoff`.
    pub fn tonnage(&self, factors: &[f64], cutoff: f64) -> f64 {
        let y_c = if cutoff <= self.anam.z_min {
            f64::NEG_INFINITY
        } else if cutoff >= self.anam.z_max {
            f64::INFINITY
        } else {
            self.anam.forward(cutoff)
        };
        // Indicator coefficients φₙ = ∫_{y_c}^∞ Hₙ g.
        (0..factors.len())
            .map(|n| upper_tail_integral(y_c, n) * factors[n])
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    /// Generic recombination `Σₙ coeffₙ·factorₙ` for any transform's Hermite
    /// coefficients.
    pub fn combine(&self, factors: &[f64], coeffs: &[f64]) -> f64 {
        factors.iter().zip(coeffs).map(|(f, c)| f * c).sum()
    }

    pub fn anamorphosis(&self) -> &HermiteAnamorphosis {
        &self.anam
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    // DK is stable only with enough data relative to the factor count; a dense
    // grid and a modest order reflect realistic usage.
    const ORDER: usize = 12;

    fn setup() -> (DisjunctiveKriging, Vec<GaussianSample>, Variogram) {
        let vals: Vec<f64> = (1..=300).map(|i| ((i as f64) / 40.0).exp()).collect();
        let anam = HermiteAnamorphosis::fit(&vals, None, ORDER).unwrap();
        // A 6×6 grid; grade follows a smooth ramp so interpolation is well posed.
        let mut samples = Vec::new();
        for ix in 0..6 {
            for iy in 0..6 {
                let (x, y) = (ix as f64 * 20.0, iy as f64 * 20.0);
                let z = 2.0 + 0.03 * (x + y); // 2..8.6 ramp
                samples.push(GaussianSample {
                    loc: (x, y, 0.0),
                    y: anam.forward(z),
                });
            }
        }
        let vg_y = Variogram::single(Model::Spherical, 1.0, 150.0);
        (DisjunctiveKriging::new(anam), samples, vg_y)
    }

    #[test]
    fn exact_at_data_location_zero_nugget() {
        // With a zero-nugget correlogram, DK at a datum reproduces that datum's grade.
        let (dk, samples, vg_y) = setup();
        let target = samples[1].loc;
        let factors = dk.factors(&target, &samples, &vg_y, ORDER).unwrap();
        let grade = dk.grade(&factors);
        let z1 = dk.anamorphosis().back(samples[1].y);
        assert!((grade - z1).abs() < 1e-4, "DK {grade} vs datum {z1}");
    }

    #[test]
    fn tonnage_in_unit_interval_and_monotone() {
        let (dk, samples, vg_y) = setup();
        let target = (25.0, 20.0, 0.0);
        let factors = dk.factors(&target, &samples, &vg_y, ORDER).unwrap();
        let cutoffs = [0.0, 2.0, 4.0, 6.0, 10.0, 20.0];
        let mut prev = 2.0;
        for &c in &cutoffs {
            let t = dk.tonnage(&factors, c);
            assert!(
                (0.0..=1.0).contains(&t),
                "tonnage {t} out of range at cutoff {c}"
            );
            assert!(t <= prev + 1e-6, "tonnage not monotone: {t} > {prev}");
            prev = t;
        }
    }

    #[test]
    fn grade_between_data_extremes() {
        let (dk, samples, vg_y) = setup();
        let target = (25.0, 25.0, 0.0);
        let factors = dk.factors(&target, &samples, &vg_y, ORDER).unwrap();
        let grade = dk.grade(&factors);
        // Interpolated grade should be within a sane band of the data.
        assert!(grade > 0.0 && grade < 12.0, "grade {grade}");
    }
}
