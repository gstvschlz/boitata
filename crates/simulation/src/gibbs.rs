//! Gibbs sampler for interval-truncated multivariate Gaussians.
//!
//! Draws a spatially-correlated Gaussian vector `Y ~ N(0, Σ)` in which each
//! component is constrained to an interval `[aᵢ, bᵢ]`. This is how facies /
//! categorical data and censored (inequality) data are turned into consistent
//! Gaussian values for plurigaussian simulation ([`crate::pgs`]) and truncated
//! Gaussian kriging.
//!
//! The efficient update uses the **precision matrix** `P = Σ⁻¹` (dense, computed
//! once): the full conditional is
//! `Yᵢ | Y₋ᵢ ~ N(−(1/Pᵢᵢ)·Σⱼ≠ᵢ Pᵢⱼ Yⱼ, 1/Pᵢᵢ)` truncated to `[aᵢ, bᵢ]`, sampled
//! by inverse-CDF. `Σ` is built from the (unit-sill) Gaussian variogram.

use crate::error::{Result, SimError};
use nalgebra::DMatrix;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use transforms::normal::{phi, probit};
use variogram::Variogram;

/// Gibbs sampler parameters.
#[derive(Debug, Clone)]
pub struct GibbsParams {
    /// Number of full sweeps over all sites.
    pub iterations: usize,
    /// Burn-in sweeps discarded before returning.
    pub burn_in: usize,
    pub seed: u64,
}

impl Default for GibbsParams {
    fn default() -> Self {
        Self {
            iterations: 200,
            burn_in: 50,
            seed: 1,
        }
    }
}

/// Sample a Gaussian vector at `locs`, each constrained to `bounds[i] = (lo, hi)`.
///
/// `vg` is the (unit-sill) Gaussian variogram. Returns one draw honoring both
/// the spatial correlation and every interval constraint.
pub fn gibbs(
    locs: &[(f64, f64, f64)],
    bounds: &[(f64, f64)],
    vg: &Variogram,
    params: &GibbsParams,
) -> Result<Vec<f64>> {
    let n = locs.len();
    if n == 0 {
        return Err(SimError::InsufficientData("no sites".into()));
    }
    if bounds.len() != n {
        return Err(SimError::InvalidParameters("bounds length mismatch".into()));
    }
    if bounds.iter().any(|(lo, hi)| hi < lo) {
        return Err(SimError::InvalidParameters("bound hi < lo".into()));
    }

    // Covariance (unit correlogram) and precision.
    let sill = vg.total_sill();
    let mut sigma = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            sigma[(i, j)] = vg.cov_points(&locs[i], &locs[j]) / sill;
        }
        sigma[(i, i)] += 1e-9; // jitter
    }
    let p = sigma
        .try_inverse()
        .ok_or_else(|| SimError::Estimation("covariance not invertible".into()))?;

    let mut rng = StdRng::seed_from_u64(params.seed);

    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = draw_truncated(&mut rng, 0.0, 1.0, bounds[i].0, bounds[i].1);
    }

    let sweeps = params.burn_in + params.iterations.max(1);
    for _ in 0..sweeps {
        for i in 0..n {
            let pii = p[(i, i)];
            if pii.abs() < 1e-300 {
                continue;
            }
            // Conditional mean = −(1/Pii) Σ_{j≠i} Pij Yj ; variance = 1/Pii.
            let mut acc = 0.0;
            for j in 0..n {
                if j != i {
                    acc += p[(i, j)] * y[j];
                }
            }
            let mean = -acc / pii;
            let sd = (1.0 / pii).sqrt();
            y[i] = draw_truncated(&mut rng, mean, sd, bounds[i].0, bounds[i].1);
        }
    }
    Ok(y)
}

/// Draw from `N(mean, sd²)` truncated to `[lo, hi]` by inverse-CDF.
fn draw_truncated(rng: &mut StdRng, mean: f64, sd: f64, lo: f64, hi: f64) -> f64 {
    let a = (lo - mean) / sd;
    let b = (hi - mean) / sd;
    let (pa, pb) = (phi(a), phi(b));
    if pb - pa < 1e-12 {
        // Degenerate interval: return the clamped mean.
        return mean.clamp(lo, hi);
    }
    let u: f64 = rng.gen_range(pa..pb);
    let z = probit(u);
    (mean + sd * z).clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    #[test]
    fn respects_all_bounds() {
        let locs: Vec<(f64, f64, f64)> = (0..12).map(|i| (i as f64 * 10.0, 0.0, 0.0)).collect();
        // Alternating intervals: below 0, above 0.
        let bounds: Vec<(f64, f64)> = (0..12)
            .map(|i| {
                if i % 2 == 0 {
                    (f64::NEG_INFINITY, 0.0)
                } else {
                    (0.0, f64::INFINITY)
                }
            })
            .collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let y = gibbs(&locs, &bounds, &vg, &GibbsParams::default()).unwrap();
        for (yi, (lo, hi)) in y.iter().zip(&bounds) {
            assert!(
                *yi >= *lo - 1e-9 && *yi <= *hi + 1e-9,
                "y {yi} out of [{lo},{hi}]"
            );
        }
    }

    #[test]
    fn reproducible_with_seed() {
        let locs: Vec<(f64, f64, f64)> = (0..8).map(|i| (i as f64 * 12.0, 0.0, 0.0)).collect();
        let bounds = vec![(-1.0, 1.0); 8];
        let vg = Variogram::single(Model::Exponential, 1.0, 60.0);
        let a = gibbs(&locs, &bounds, &vg, &GibbsParams::default()).unwrap();
        let b = gibbs(&locs, &bounds, &vg, &GibbsParams::default()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn positive_correlation_yields_similar_neighbors() {
        // Strong correlation + wide bounds → adjacent sites are positively correlated.
        let locs: Vec<(f64, f64, f64)> = (0..40).map(|i| (i as f64, 0.0, 0.0)).collect();
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 40];
        let vg = Variogram::single(Model::Gaussian, 1.0, 20.0);
        let y = gibbs(
            &locs,
            &bounds,
            &vg,
            &GibbsParams {
                iterations: 400,
                burn_in: 100,
                seed: 5,
            },
        )
        .unwrap();
        // Lag-1 correlation should be clearly positive.
        let m = y.iter().sum::<f64>() / y.len() as f64;
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..y.len() - 1 {
            num += (y[i] - m) * (y[i + 1] - m);
        }
        for yi in &y {
            den += (yi - m).powi(2);
        }
        assert!(num / den > 0.2, "lag-1 corr {}", num / den);
    }
}
