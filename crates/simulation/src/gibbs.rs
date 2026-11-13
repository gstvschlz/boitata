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
use rayon::prelude::*;
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
    /// Row-major `P = Σ⁻¹`. A sweep reads one row at a time.
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
        let inv_sill = 1.0 / vg.total_sill();
        let cov = |i: usize, j: usize| match anisotropy {
            None => vg.cov_points(&locs[i], &locs[j]),
            Some(a) => {
                0.5 * (vg.cov(a[i].lag(&locs[i], &locs[j])) + vg.cov(a[j].lag(&locs[i], &locs[j])))
            }
        };
        let mut a = vec![0.0; n * n];
        a.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..=i {
                row[j] = cov(i, j) * inv_sill;
            }
            row[i] += 1e-9;
        });
        if let Err(e) = cholesky_lower(&mut a, n) {
            if anisotropy.is_none() {
                return Err(e);
            }
            // Averaging covariances read in different anisotropies is symmetric but
            // not always positive definite; floor the spectrum so no conditional
            // variance of the sampler turns negative.
            let mut sigma = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..=i {
                    let c = cov(i, j) * inv_sill;
                    sigma[(i, j)] = c;
                    sigma[(j, i)] = c;
                }
                sigma[(i, i)] += 1e-9;
            }
            let e = sigma.symmetric_eigen();
            let floored = e.eigenvalues.map(|l| l.max(1e-9));
            let sigma =
                &e.eigenvectors * DMatrix::from_diagonal(&floored) * e.eigenvectors.transpose();
            for i in 0..n {
                for j in 0..=i {
                    a[i * n + j] = sigma[(i, j)];
                }
            }
            cholesky_lower(&mut a, n)?;
        }
        let mut columns = vec![0.0; n * n];
        columns
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(j, column)| {
                precision_column(&a, n, j, column);
            });
        let mut precision = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                precision[i * n + j] = columns[j * n + i];
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
                // Full row, then drop `Pii Yi`, so the dot stays one contiguous span.
                let acc = dot(row, &y) - pii * y[i];
                let mean = -acc / pii;
                let sd = (1.0 / pii).sqrt();
                y[i] = draw_truncated(&mut rng, mean, sd, bounds[i].0, bounds[i].1);
            }
        }
        Ok(y)
    }
}

/// Eight partial sums, then this tree. Multiply and add stay separate, and the
/// width does not change, so a seed is the same bits on every CPU.
fn reduce8(acc: [f64; 8]) -> f64 {
    let s0 = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let s1 = (acc[4] + acc[5]) + (acc[6] + acc[7]);
    s0 + s1
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0; 8];
    for (a8, b8) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        acc[0] += a8[0] * b8[0];
        acc[1] += a8[1] * b8[1];
        acc[2] += a8[2] * b8[2];
        acc[3] += a8[3] * b8[3];
        acc[4] += a8[4] * b8[4];
        acc[5] += a8[5] * b8[5];
        acc[6] += a8[6] * b8[6];
        acc[7] += a8[7] * b8[7];
    }
    let mut tail = 0.0;
    let start = a.len() / 8 * 8;
    for (&x, &y) in a[start..].iter().zip(&b[start..]) {
        tail += x * y;
    }
    reduce8(acc) + tail
}

/// In-place lower Cholesky of the lower triangle of row-major `a`.
fn cholesky_lower(a: &mut [f64], n: usize) -> Result<()> {
    for i in 0..n {
        for j in 0..i {
            let sum = dot(&a[i * n..i * n + j], &a[j * n..j * n + j]);
            a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
        }
        let sum = dot(&a[i * n..i * n + i], &a[i * n..i * n + i]);
        let d = a[i * n + i] - sum;
        if d.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(SimError::Estimation("covariance not invertible".into()));
        }
        a[i * n + i] = d.sqrt();
    }
    Ok(())
}

/// Column `j` of `L⁻ᵀ L⁻¹`, with `l` the lower Cholesky factor, row-major.
fn precision_column(l: &[f64], n: usize, j: usize, p: &mut [f64]) {
    p[j] = 1.0 / l[j * n + j];
    for i in (j + 1)..n {
        let sum = dot(&l[i * n + j..i * n + i], &p[j..i]);
        p[i] = -sum / l[i * n + i];
    }
    for i in (0..n).rev() {
        let pi = p[i] / l[i * n + i];
        p[i] = pi;
        scale_add(-pi, &l[i * n..i * n + i], &mut p[..i]);
    }
}

/// `y[i] += alpha * x[i]`. `x` and `y` do not alias. Multiply and add stay separate.
fn scale_add(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (slot, &xi) in y.iter_mut().zip(x) {
        let prod = alpha * xi;
        *slot += prod;
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
    fn precision_inverts_the_covariance() {
        let locs: Vec<(f64, f64, f64)> = (0..17)
            .map(|i| ((i % 5) as f64 * 4.0, (i / 5) as f64 * 6.0, (i % 3) as f64))
            .collect();
        let vg = Variogram::single(Model::Spherical, 1.7, 12.0);
        let system = GibbsSystem::new(&locs, &vg).unwrap();
        let n = locs.len();
        let inv_sill = 1.0 / vg.total_sill();
        let mut sigma = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                sigma[i * n + j] = vg.cov_points(&locs[i], &locs[j]) * inv_sill;
            }
            sigma[i * n + i] += 1e-9;
        }
        let mut worst = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += system.precision[i * n + k] * sigma[k * n + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((s - want).abs());
            }
        }
        assert!(worst < 1e-8, "P Σ max error {worst}");
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
