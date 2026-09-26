//! Discrete Gaussian Model (DGM) change-of-support.
//!
//! Corrects the *distribution* (not just the mean) when moving from point to
//! block support. Under DGM1, block grades are `Z_v = Σₙ ψₙ rⁿ Hₙ(Y_v)` with
//! `Y_v ~ N(0,1)` and a single change-of-support coefficient `r ∈ (0, 1]` that
//! shrinks the variance to block support.
//!
//! `r` is solved from the block-averaged Gaussian covariance `b`:
//!
//! ```text
//! b   = (1/|v|²) ∫∫ ρ_Y(x − x′) dx dx′      (average correlation of Y over v)
//! Var(Z_v) = Σ_{n≥1} ψₙ² r^{2n}  =  Σ_{n≥1} ψₙ² bⁿ   →   solve for r
//! ```
//!
//! where `ρ_Y` is the correlogram of the normal-score variogram (unit sill).
//! This is the variance-correcting replacement for arithmetic block averaging
//! ([`crate::scale`]).

use crate::anamorphosis::HermiteAnamorphosis;
use crate::error::{Result, TransformError};
use variogram::Variogram;

/// Number of discretization points per axis for block averaging.
#[derive(Debug, Clone, Copy)]
pub struct BlockDiscretization {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Default for BlockDiscretization {
    fn default() -> Self {
        Self {
            nx: 5,
            ny: 5,
            nz: 5,
        }
    }
}

impl BlockDiscretization {
    fn points(&self, center: (f64, f64, f64), size: (f64, f64, f64)) -> Vec<(f64, f64, f64)> {
        let (nx, ny, nz) = (self.nx.max(1), self.ny.max(1), self.nz.max(1));
        let dx = size.0 / nx as f64;
        let dy = size.1 / ny as f64;
        let dz = size.2 / nz as f64;
        let x0 = center.0 - size.0 / 2.0 + dx / 2.0;
        let y0 = center.1 - size.1 / 2.0 + dy / 2.0;
        let z0 = center.2 - size.2 / 2.0 + dz / 2.0;
        let mut pts = Vec::with_capacity(nx * ny * nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    pts.push((x0 + i as f64 * dx, y0 + j as f64 * dy, z0 + k as f64 * dz));
                }
            }
        }
        pts
    }
}

/// Average Gaussian correlation `b = C̄_Y(v, v) / C_Y(0)` over a block.
///
/// `vg_y` is the variogram of the normal scores (any sill; normalized here).
pub fn block_average_correlation(
    vg_y: &Variogram,
    center: (f64, f64, f64),
    size: (f64, f64, f64),
    disc: &BlockDiscretization,
) -> f64 {
    let pts = disc.points(center, size);
    let np = pts.len() as f64;
    let sill = vg_y.total_sill();
    let mut acc = 0.0;
    for p in &pts {
        for q in &pts {
            acc += vg_y.block_cov_points(p, q);
        }
    }
    (acc / (np * np) / sill).clamp(0.0, 1.0)
}

/// Solve the change-of-support coefficient `r ∈ (0, 1]` from the block-averaged
/// Gaussian correlation `b`, for a point anamorphosis.
///
/// Solves `Σ_{n≥1} ψₙ² r^{2n} = Σ_{n≥1} ψₙ² bⁿ` (monotone in `r`) by bisection.
pub fn coefficient_from_correlation(anam: &HermiteAnamorphosis, b: f64) -> f64 {
    let psi = &anam.coefficients;
    let b = b.clamp(0.0, 1.0);
    // Target block variance in Hermite space.
    let target: f64 = (1..psi.len())
        .map(|n| psi[n] * psi[n] * b.powi(n as i32))
        .sum();
    let point_var: f64 = (1..psi.len()).map(|n| psi[n] * psi[n]).sum();
    if point_var <= 0.0 {
        return 1.0;
    }
    // f(r) = Σ ψₙ² r^{2n} is increasing in r; find r with f(r) = target.
    let f = |r: f64| -> f64 {
        (1..psi.len())
            .map(|n| psi[n] * psi[n] * r.powi(2 * n as i32))
            .sum::<f64>()
    };
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    if target >= point_var {
        return 1.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Full DGM change-of-support: compute `r` from the variogram + block geometry
/// and return the block anamorphosis together with `r`.
pub fn change_of_support(
    anam: &HermiteAnamorphosis,
    vg_y: &Variogram,
    center: (f64, f64, f64),
    size: (f64, f64, f64),
    disc: &BlockDiscretization,
) -> Result<(f64, HermiteAnamorphosis)> {
    if size.0 <= 0.0 || size.1 <= 0.0 || size.2 <= 0.0 {
        return Err(TransformError::InvalidParameters(
            "block size must be positive on every axis".into(),
        ));
    }
    let b = block_average_correlation(vg_y, center, size, disc);
    let r = coefficient_from_correlation(anam, b);
    Ok((r, anam.block(r)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    fn sample_anam() -> HermiteAnamorphosis {
        let vals: Vec<f64> = (1..=400).map(|i| ((i as f64) / 40.0).exp()).collect();
        HermiteAnamorphosis::fit(&vals, None, 50).unwrap()
    }

    #[test]
    fn tiny_block_keeps_point_support() {
        // A block much smaller than the range → b ≈ 1 → r ≈ 1 → no correction.
        let anam = sample_anam();
        let vg = Variogram::single(Model::Spherical, 1.0, 1000.0);
        let (r, block) = change_of_support(
            &anam,
            &vg,
            (0.0, 0.0, 0.0),
            (0.001, 0.001, 0.001),
            &BlockDiscretization::default(),
        )
        .unwrap();
        assert!(r > 0.999, "r {r}");
        // Relative tolerance (data variance is large-magnitude here).
        assert!(
            (block.variance() - anam.variance()).abs() < 1e-3 * anam.variance(),
            "block {} vs point {}",
            block.variance(),
            anam.variance()
        );
    }

    #[test]
    fn large_block_reduces_variance() {
        // A block comparable to the range → b < 1 → r < 1 → variance reduced.
        let anam = sample_anam();
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let (r, block) = change_of_support(
            &anam,
            &vg,
            (0.0, 0.0, 0.0),
            (80.0, 80.0, 80.0),
            &BlockDiscretization::default(),
        )
        .unwrap();
        assert!(r > 0.0 && r < 1.0, "r {r}");
        assert!(block.variance() < anam.variance(), "variance not reduced");
        // Mean preserved.
        assert!((block.mean() - anam.mean()).abs() < 1e-9);
    }

    #[test]
    fn correlation_in_unit_range() {
        let vg = Variogram::single(Model::Exponential, 2.5, 100.0);
        let b = block_average_correlation(
            &vg,
            (0.0, 0.0, 0.0),
            (50.0, 50.0, 10.0),
            &BlockDiscretization::default(),
        );
        assert!(b > 0.0 && b <= 1.0, "b {b}");
    }

    #[test]
    fn nugget_averages_out_of_the_block() {
        let pure = Variogram {
            nugget: 1.0,
            structures: vec![],
            anisotropy: None,
        };
        let size = (20.0, 20.0, 5.0);
        let disc = BlockDiscretization::default();
        assert_eq!(
            block_average_correlation(&pure, (0.0, 0.0, 0.0), size, &disc),
            0.0
        );

        let (c0, c, a, l) = (0.3, 0.7, 100.0, 80.0);
        let vg = Variogram {
            nugget: c0,
            ..Variogram::single(Model::Spherical, c, a)
        };
        let disc = BlockDiscretization {
            nx: 400,
            ny: 1,
            nz: 1,
        };
        let b = block_average_correlation(&vg, (0.0, 0.0, 0.0), (l, 0.0, 0.0), &disc);
        let gamma_bar = c * (l / (2.0 * a) - l.powi(3) / (20.0 * a.powi(3)));
        assert!((b - (c - gamma_bar) / (c0 + c)).abs() < 1e-4, "b {b}");
    }
}
