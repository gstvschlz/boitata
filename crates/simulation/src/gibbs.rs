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
use variogram::{Anisotropy, Variogram};

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

/// Precision of the data covariance. It depends only on the locations and the
/// variogram, so one system serves every realization and every matching field.
pub(crate) struct GibbsSystem {
    /// Row-major `P = Σ⁻¹`. `DMatrix` is column-major; the sweep reads a row
    /// at a time.
    precision: Vec<f64>,
    n: usize,
}

impl GibbsSystem {
    pub(crate) fn new(locs: &[(f64, f64, f64)], vg: &Variogram) -> Result<Self> {
        Self::assemble(locs, vg, None)
    }

    /// As [`Self::new`], with the covariance between sites `i` and `j` the
    /// average of the variogram covariances in `anisotropy[i]` and
    /// `anisotropy[j]`. The average is symmetric. It is the covariance locally
    /// varying anisotropy kriging uses for a pair when both sites share one
    /// anisotropy: that kriging reads every pair of a target in the target's
    /// anisotropy, and here each site is the target at its own location.
    /// When the average is not positive definite its eigenvalues are
    /// floored, so the sampler always has positive conditional variances.
    pub(crate) fn with_anisotropy(
        locs: &[(f64, f64, f64)],
        vg: &Variogram,
        anisotropy: &[Anisotropy],
    ) -> Result<Self> {
        if anisotropy.len() != locs.len() {
            return Err(SimError::InvalidParameters(
                "one anisotropy per site".into(),
            ));
        }
        Self::assemble(locs, vg, Some(anisotropy))
    }

    fn assemble(
        locs: &[(f64, f64, f64)],
        vg: &Variogram,
        anisotropy: Option<&[Anisotropy]>,
    ) -> Result<Self> {
        let n = locs.len();
        if n == 0 {
            return Err(SimError::InsufficientData("no sites".into()));
        }
        let sill = vg.total_sill();
        let mut sigma = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let cov = match anisotropy {
                    None => vg.cov_points(&locs[i], &locs[j]),
                    Some(a) => {
                        0.5 * (vg.cov(a[i].lag(&locs[i], &locs[j]))
                            + vg.cov(a[j].lag(&locs[i], &locs[j])))
                    }
                };
                sigma[(i, j)] = cov / sill;
            }
            sigma[(i, i)] += 1e-9; // jitter
        }
        // Averaging covariances read in different anisotropies is symmetric but
        // not always positive definite; floor the spectrum so no conditional
        // variance of the sampler turns negative.
        if anisotropy.is_some() && sigma.clone().cholesky().is_none() {
            let e = sigma.symmetric_eigen();
            let floored = e.eigenvalues.map(|l| l.max(1e-9));
            sigma = &e.eigenvectors * DMatrix::from_diagonal(&floored) * e.eigenvectors.transpose();
        }
        let p = sigma
            .try_inverse()
            .ok_or_else(|| SimError::Estimation("covariance not invertible".into()))?;
        let mut precision = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                precision[i * n + j] = p[(i, j)];
            }
        }
        Ok(Self { precision, n })
    }

    /// One draw at the sites, each constrained to `bounds[i] = (lo, hi)`.
    pub(crate) fn sample(&self, bounds: &[(f64, f64)], params: &GibbsParams) -> Result<Vec<f64>> {
        let n = self.n;
        if bounds.len() != n {
            return Err(SimError::InvalidParameters("bounds length mismatch".into()));
        }
        if bounds.iter().any(|(lo, hi)| hi < lo) {
            return Err(SimError::InvalidParameters("bound hi < lo".into()));
        }
        let mut rng = StdRng::seed_from_u64(params.seed);
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = draw_truncated(&mut rng, 0.0, 1.0, bounds[i].0, bounds[i].1);
        }
        let sweeps = params.burn_in + params.iterations.max(1);
        for _ in 0..sweeps {
            for i in 0..n {
                let row = &self.precision[i * n..(i + 1) * n];
                let pii = row[i];
                if pii.abs() < 1e-300 {
                    continue;
                }
                // Conditional mean = −(1/Pii) Σ_{j≠i} Pij Yj ; variance = 1/Pii.
                // `0..i` then `i+1..n` is the same order as skipping `j == i`.
                let mut acc = 0.0;
                for j in 0..i {
                    acc += row[j] * y[j];
                }
                for j in i + 1..n {
                    acc += row[j] * y[j];
                }
                let mean = -acc / pii;
                let sd = (1.0 / pii).sqrt();
                y[i] = draw_truncated(&mut rng, mean, sd, bounds[i].0, bounds[i].1);
            }
        }
        Ok(y)
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
    if bounds.len() != locs.len() {
        return Err(SimError::InvalidParameters("bounds length mismatch".into()));
    }
    if bounds.iter().any(|(lo, hi)| hi < lo) {
        return Err(SimError::InvalidParameters("bound hi < lo".into()));
    }
    GibbsSystem::new(locs, vg)?.sample(bounds, params)
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

    #[test]
    fn constant_anisotropy_matches_the_variogram_and_a_mix_still_draws() {
        let locs = [
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (0.0, 6.0, 0.0),
            (3.0, 5.0, 1.0),
        ];
        let same = Anisotropy::new(variogram::Angles {
            azimuth: 25.0,
            dip: 10.0,
            rake: 5.0,
            major: 1.0,
            semi: 0.4,
            minor: 0.2,
        })
        .unwrap();
        let vg = Variogram::single(Model::Exponential, 2.0, 12.0).with_anisotropy(same.clone());
        let repeated = vec![same.clone(); locs.len()];
        let params = GibbsParams {
            iterations: 40,
            burn_in: 10,
            seed: 9,
        };
        let bounds = vec![(f64::NEG_INFINITY, 0.5); locs.len()];
        let plain = GibbsSystem::new(&locs, &vg)
            .unwrap()
            .sample(&bounds, &params)
            .unwrap();
        let local = GibbsSystem::with_anisotropy(&locs, &vg, &repeated)
            .unwrap()
            .sample(&bounds, &params)
            .unwrap();
        assert_eq!(plain, local);
        let other = Anisotropy::new(variogram::Angles {
            azimuth: 110.0,
            dip: 0.0,
            rake: 40.0,
            major: 2.0,
            semi: 0.8,
            minor: 0.3,
        })
        .unwrap();
        let mut mixed = repeated;
        mixed[1] = other;
        let draw = GibbsSystem::with_anisotropy(&locs, &vg, &mixed)
            .unwrap()
            .sample(&bounds, &params)
            .unwrap();
        assert_ne!(draw, plain);
        assert!(draw.iter().all(|y| y.is_finite() && *y <= 0.5));
        assert!(GibbsSystem::with_anisotropy(&locs, &vg, &mixed[..2]).is_err());
    }

    #[test]
    fn mixed_anisotropies_keep_every_conditional_variance_positive() {
        let locs: Vec<_> = (0..40)
            .map(|i| ((i * 7 % 20) as f64, (i * 13 % 20) as f64, 0.0))
            .collect();
        let anisotropy: Vec<_> = (0..40)
            .map(|i| {
                Anisotropy::new(variogram::Angles {
                    azimuth: (i * 37 % 180) as f64,
                    dip: 0.0,
                    rake: 0.0,
                    major: 1.0,
                    semi: 0.05,
                    minor: 0.05,
                })
                .unwrap()
            })
            .collect();
        let vg = Variogram::single(Model::Gaussian, 1.0, 15.0);
        let system = GibbsSystem::with_anisotropy(&locs, &vg, &anisotropy).unwrap();
        assert!((0..40).all(|i| system.precision[i * 40 + i] > 0.0));
        let params = GibbsParams {
            iterations: 30,
            burn_in: 5,
            seed: 3,
        };
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 40];
        let draw = system.sample(&bounds, &params).unwrap();
        assert!(draw.iter().all(|y| y.is_finite()));
    }
}
