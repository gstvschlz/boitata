//! Block kriging via point discretization.
//!
//! A block is discretized into an `nx×ny×nz` grid of points; block kriging averages
//! the point-to-block covariances so the estimate has proper block support (reducing
//! conditional bias relative to estimating at the block centroid).

use crate::Sample;
use crate::error::{EstimError, Result};
use crate::krige::Estimate;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use variogram::Variogram;

/// Block discretization (sub-points per axis).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Discretization {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Default for Discretization {
    fn default() -> Self {
        Self {
            nx: 3,
            ny: 3,
            nz: 3,
        }
    }
}

impl Discretization {
    pub fn count(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Discretization points centered on `center` for a block of `size` (dx, dy, dz).
    pub fn points(&self, center: &(f64, f64, f64), size: &(f64, f64, f64)) -> Vec<(f64, f64, f64)> {
        let mut pts = Vec::with_capacity(self.count());
        let dx = size.0 / self.nx as f64;
        let dy = size.1 / self.ny as f64;
        let dz = size.2 / self.nz as f64;
        let x0 = center.0 - size.0 / 2.0 + dx / 2.0;
        let y0 = center.1 - size.1 / 2.0 + dy / 2.0;
        let z0 = center.2 - size.2 / 2.0 + dz / 2.0;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    pts.push((x0 + i as f64 * dx, y0 + j as f64 * dy, z0 + k as f64 * dz));
                }
            }
        }
        pts
    }
}

/// Ordinary block kriging over a discretized block.
///
/// Only the ordinary form is provided (the common case for resource blocks); simple
/// block kriging follows the same pattern with the constraint row removed.
pub fn block_krige(
    center: &(f64, f64, f64),
    size: &(f64, f64, f64),
    samples: &[Sample],
    disc: &Discretization,
    vg: &Variogram,
) -> Result<Estimate> {
    let n = samples.len();
    if n == 0 {
        return Err(EstimError::InsufficientData("no samples".into()));
    }
    let pts = disc.points(center, size);
    let np = pts.len() as f64;

    // Point-to-block average covariance for each sample (RHS) and block-to-block
    // average covariance (for the variance term).
    let mut b = DVector::<f64>::zeros(n + 1);
    for i in 0..n {
        let avg: f64 = pts
            .iter()
            .map(|p| vg.block_cov_points(&samples[i].loc, p))
            .sum::<f64>()
            / np;
        b[i] = avg;
    }
    b[n] = 1.0;

    // Average block-to-block covariance C(B,B).
    let mut cbb = 0.0;
    for p in &pts {
        for q in &pts {
            cbb += vg.block_cov_points(p, q);
        }
    }
    cbb /= np * np;

    // Sample-to-sample covariance with the OK constraint.
    let mut a = DMatrix::<f64>::zeros(n + 1, n + 1);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = vg.cov_points(&samples[i].loc, &samples[j].loc);
        }
        a[(i, i)] += samples[i].error_variance;
        a[(i, n)] = 1.0;
        a[(n, i)] = 1.0;
    }

    let x = a
        .lu()
        .solve(&b)
        .ok_or_else(|| EstimError::Singular("block kriging matrix not invertible".into()))?;

    let weights: Vec<f64> = x.as_slice()[..n].to_vec();
    let value: f64 = (0..n).map(|i| weights[i] * samples[i].value).sum();
    let mu = x[n];
    let sum_wc: f64 = (0..n).map(|i| weights[i] * b[i]).sum();
    // Block kriging variance: C(B,B) − Σλ C(xᵢ,B) − μ.
    let variance = (cbb - sum_wc - mu).max(0.0);

    Ok(Estimate {
        value,
        variance,
        n_used: n,
        weights,
        lagrange: mu,
        support_variance: cbb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::Kind;
    use variogram::model::Model;

    #[test]
    fn discretization_centers_on_block() {
        let d = Discretization::default();
        assert_eq!(d.count(), 27);
        let pts = d.points(&(100.0, 200.0, 300.0), &(10.0, 10.0, 10.0));
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64;
        assert!((mx - 100.0).abs() < 1e-9);
    }

    #[test]
    fn block_variance_below_point_variance() {
        // Block support should reduce variance relative to a point estimate at the centroid.
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let samples = vec![
            Sample {
                loc: (0.0, 0.0, 0.0),
                value: 1.0,
                hole: None,
                error_variance: 0.0,
                domain: None,
            },
            Sample {
                loc: (60.0, 0.0, 0.0),
                value: 2.0,
                hole: None,
                error_variance: 0.0,
                domain: None,
            },
            Sample {
                loc: (0.0, 60.0, 0.0),
                value: 3.0,
                hole: None,
                error_variance: 0.0,
                domain: None,
            },
        ];
        let center = (30.0, 30.0, 0.0);
        let size = (20.0, 20.0, 20.0);
        let block = block_krige(&center, &size, &samples, &Discretization::default(), &vg).unwrap();
        let point = crate::krige::krige(Kind::Ordinary, &center, &samples, &vg).unwrap();
        assert!(block.variance <= point.variance + 1e-9);
        assert!((block.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    fn sample(loc: (f64, f64, f64), value: f64) -> Sample {
        Sample {
            loc,
            value,
            hole: None,
            error_variance: 0.0,
            domain: None,
        }
    }

    fn with_nugget(nugget: f64) -> Variogram {
        Variogram {
            nugget,
            ..Variogram::single(Model::Spherical, 1.0, 100.0)
        }
    }

    #[test]
    fn nugget_averages_out_of_the_block() {
        let (c0, c, a, l) = (0.4, 1.0, 100.0, 60.0);
        let disc = Discretization {
            nx: 400,
            ny: 1,
            nz: 1,
        };
        let samples = [
            sample((-50.0, 0.0, 0.0), 1.0),
            sample((70.0, 0.0, 0.0), 2.0),
        ];
        let vg = Variogram {
            nugget: c0,
            ..Variogram::single(Model::Spherical, c, a)
        };
        let e = block_krige(&(0.0, 0.0, 0.0), &(l, 0.0, 0.0), &samples, &disc, &vg).unwrap();
        let gamma_bar = c * (l / (2.0 * a) - l.powi(3) / (20.0 * a.powi(3)));
        assert!((e.support_variance - (c - gamma_bar)).abs() < 1e-4);

        let pure = Variogram {
            nugget: 1.0,
            structures: vec![],
            anisotropy: None,
        };
        let e = block_krige(&(0.0, 0.0, 0.0), &(l, 1.0, 1.0), &samples, &disc, &pure).unwrap();
        assert_eq!(e.support_variance, 0.0);
        assert!((e.variance - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sample_on_a_node_sees_no_nugget() {
        let disc = Discretization::default();
        let vg = with_nugget(0.5);
        let block = |x: f64| {
            let s = [sample((x, 0.0, 0.0), 1.0), sample((40.0, 25.0, 0.0), 3.0)];
            block_krige(&(0.0, 0.0, 0.0), &(30.0, 30.0, 30.0), &s, &disc, &vg).unwrap()
        };
        let (on, off) = (block(0.0), block(1e-9));
        assert!((on.weights[0] - off.weights[0]).abs() < 1e-9);
        assert!((on.variance - off.variance).abs() < 1e-9);
    }

    #[test]
    fn no_nugget_is_unchanged() {
        let vg = with_nugget(0.0);
        let pts = Discretization::default().points(&(0.0, 0.0, 0.0), &(30.0, 30.0, 30.0));
        for p in &pts {
            for q in &pts {
                assert_eq!(
                    vg.block_cov_points(p, q).to_bits(),
                    vg.cov_points(p, q).to_bits()
                );
            }
        }
    }
}
