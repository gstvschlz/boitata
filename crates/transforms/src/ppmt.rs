//! Projection Pursuit Multivariate Transform (PPMT).
//!
//! Turns a multivariate dataset into independent standard-normal variables
//! (joint Gaussianity), and back. This is the multivariate generalization of the
//! normal-score transform: it decorrelates *and* removes higher-order
//! dependence, so downstream Gaussian methods (SGS, cokriging on the factors)
//! are justified.
//!
//! Algorithm (Barnett, Manchuk & Deutsch, 2014):
//! 0. optionally normal-score each variable (weighted),
//! 1. center and sphere (whiten) the data via the covariance eigendecomposition,
//! 2. repeatedly find the most non-Gaussian 1-D projection and Gaussianize it
//!    (normal-score along that direction),
//! 3. store the direction + 1-D map at each step so the transform is invertible.
//!
//! Non-Gaussianity is scored by the FastICA negentropy proxy
//! `J ≈ m₃²/12 + (m₄−3)²/48`. Candidate directions are drawn from a deterministic
//! LCG (reproducible, no `rand` dependency).

use crate::error::{Result, TransformError};
use crate::normal::probit;
use crate::normal_score::{NormalScoreTable, transform as normal_score_table};
use crate::pca::check_rows;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// One projection-pursuit step: a unit direction plus its 1-D normal-score map.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Step {
    direction: Vec<f64>,
    /// Sorted projection values (original marginal along `direction`).
    values: Vec<f64>,
    /// Corresponding normal scores (increasing).
    scores: Vec<f64>,
}

/// A fitted PPMT: whitening transform + the ordered projection-pursuit steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ppmt {
    dim: usize,
    #[serde(default)]
    marginals: Vec<NormalScoreTable>,
    mean: DVector<f64>,
    /// Whitening matrix `W`: `z = W·(x − mean)`.
    whiten: DMatrix<f64>,
    /// De-whitening matrix `W⁻¹`.
    dewhiten: DMatrix<f64>,
    steps: Vec<Step>,
}

/// PPMT fitting parameters.
#[derive(Debug, Clone, Copy)]
pub struct PpmtParams {
    /// Number of projection-pursuit iterations.
    pub iterations: usize,
    /// Candidate directions evaluated per iteration.
    pub candidates: usize,
    /// RNG seed for the (deterministic) direction search.
    pub seed: u64,
    /// Normal-score each variable before whitening.
    pub marginal: bool,
}

impl Default for PpmtParams {
    fn default() -> Self {
        Self {
            iterations: 30,
            candidates: 60,
            seed: 1,
            marginal: true,
        }
    }
}

impl Ppmt {
    /// Fit a PPMT to `data` (`n` rows of `dim` columns); `weights` (e.g.
    /// declustering) shape the marginal normal scores.
    pub fn fit(data: &[Vec<f64>], weights: Option<&[f64]>, params: &PpmtParams) -> Result<Self> {
        let (n, dim) = (data.len(), check_rows(data, weights)?);
        if weights.is_some() && !params.marginal {
            return Err(TransformError::InvalidParameters(
                "weights need the marginal step".into(),
            ));
        }
        let mut x = DMatrix::from_row_iterator(n, dim, data.iter().flat_map(|r| r.iter().copied()));
        let mut marginals = Vec::new();
        if params.marginal {
            for k in 0..dim {
                let column: Vec<f64> = x.column(k).iter().copied().collect();
                let ns = normal_score_table(&column, weights)?;
                x.set_column(k, &DVector::from_vec(ns.scores));
                marginals.push(ns.table);
            }
        }
        let mean: DVector<f64> = x.row_mean().transpose();

        // Covariance (population) of centered data.
        let centered = {
            let mut c = x.clone();
            for mut row in c.row_iter_mut() {
                row -= mean.transpose();
            }
            c
        };
        let cov = (&centered.transpose() * &centered) / n as f64;

        let eig = nalgebra::SymmetricEigen::new(cov);
        let mut inv_sqrt = DVector::zeros(dim);
        let mut sqrt = DVector::zeros(dim);
        for i in 0..dim {
            let lam = eig.eigenvalues[i].max(1e-12);
            inv_sqrt[i] = 1.0 / lam.sqrt();
            sqrt[i] = lam.sqrt();
        }
        let v = &eig.eigenvectors;
        // W = diag(1/√λ)·Vᵀ ; W⁻¹ = V·diag(√λ)
        let whiten = DMatrix::from_diagonal(&inv_sqrt) * v.transpose();
        let dewhiten = v * DMatrix::from_diagonal(&sqrt);

        // Whitened data z (n×dim), rows.
        let mut z = &centered * whiten.transpose();

        let mut rng = Lcg::new(params.seed);
        let mut steps = Vec::with_capacity(params.iterations);
        for _ in 0..params.iterations {
            let mut best_dir = vec![0.0; dim];
            best_dir[0] = 1.0;
            let mut best_score = -1.0;
            for _ in 0..params.candidates.max(1) {
                let dir = rng.unit_vector(dim);
                let proj = project(&z, &dir);
                let j = negentropy(&proj);
                if j > best_score {
                    best_score = j;
                    best_dir = dir;
                }
            }
            let proj = project(&z, &best_dir);
            let (sorted_vals, sorted_scores, scores) = normal_score(&proj);
            // z ← z + dir·(g − p)
            for (i, (&p, &g)) in proj.iter().zip(&scores).enumerate() {
                let delta = g - p;
                for k in 0..dim {
                    z[(i, k)] += best_dir[k] * delta;
                }
            }
            steps.push(Step {
                direction: best_dir,
                values: sorted_vals,
                scores: sorted_scores,
            });
        }

        Ok(Self {
            dim,
            marginals,
            mean,
            whiten,
            dewhiten,
            steps,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Forward transform: original data → multivariate Gaussian factors.
    pub fn forward(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }
        let mut centered =
            DMatrix::from_row_iterator(n, self.dim, data.iter().flat_map(|r| r.iter().copied()));
        for (k, table) in self.marginals.iter().enumerate() {
            centered.column_mut(k).apply(|v| *v = table.forward(*v));
        }
        for mut row in centered.row_iter_mut() {
            row -= self.mean.transpose();
        }
        let mut z = &centered * self.whiten.transpose();
        for step in &self.steps {
            let dir = &step.direction;
            let proj = project(&z, dir);
            for (i, &p) in proj.iter().enumerate() {
                let g = interp(&step.values, &step.scores, p);
                let delta = g - p;
                for k in 0..self.dim {
                    z[(i, k)] += dir[k] * delta;
                }
            }
        }
        rows(&z)
    }

    /// Back-transform: Gaussian factors → original data space.
    pub fn back(&self, gauss: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = gauss.len();
        if n == 0 {
            return vec![];
        }
        let mut z =
            DMatrix::from_row_iterator(n, self.dim, gauss.iter().flat_map(|r| r.iter().copied()));
        for step in self.steps.iter().rev() {
            let dir = &step.direction;
            let proj = project(&z, dir);
            for (i, &g) in proj.iter().enumerate() {
                // invert 1-D map: gaussian score → original projection value
                let p = interp(&step.scores, &step.values, g);
                let delta = p - g;
                for k in 0..self.dim {
                    z[(i, k)] += dir[k] * delta;
                }
            }
        }
        let x = &z * self.dewhiten.transpose();
        let mut out = x;
        for mut row in out.row_iter_mut() {
            row += self.mean.transpose();
        }
        for (k, table) in self.marginals.iter().enumerate() {
            out.column_mut(k).apply(|v| *v = table.back(*v));
        }
        rows(&out)
    }
}

fn project(z: &DMatrix<f64>, dir: &[f64]) -> Vec<f64> {
    let d = DVector::from_row_slice(dir);
    (z * d).iter().copied().collect()
}

fn rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| m.row(i).iter().copied().collect())
        .collect()
}

/// FastICA negentropy proxy for a mean-0, var-1 sample.
fn negentropy(u: &[f64]) -> f64 {
    let n = u.len() as f64;
    let mean = u.iter().sum::<f64>() / n;
    let var = u.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt().max(1e-12);
    let m3 = u.iter().map(|x| ((x - mean) / sd).powi(3)).sum::<f64>() / n;
    let m4 = u.iter().map(|x| ((x - mean) / sd).powi(4)).sum::<f64>() / n;
    m3 * m3 / 12.0 + (m4 - 3.0).powi(2) / 48.0
}

/// Normal-score a projection; returns (sorted values, sorted scores, scores in
/// input order).
fn normal_score(vals: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = vals.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap());
    let mut scores = vec![0.0; n];
    let mut sorted_vals = Vec::with_capacity(n);
    let mut sorted_scores = Vec::with_capacity(n);
    for (rank, &idx) in order.iter().enumerate() {
        let p = (rank as f64 + 0.5) / n as f64;
        let z = probit(p);
        scores[idx] = z;
        sorted_vals.push(vals[idx]);
        sorted_scores.push(z);
    }
    (sorted_vals, sorted_scores, scores)
}

/// Monotone linear interpolation of `ys` over knots `xs` (both sorted ascending),
/// with linear extrapolation at the ends.
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return ys[0];
    }
    if x <= xs[0] {
        return lin(x, xs[0], xs[1], ys[0], ys[1]);
    }
    if x >= xs[n - 1] {
        return lin(x, xs[n - 2], xs[n - 1], ys[n - 2], ys[n - 1]);
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lin(x, xs[lo], xs[hi], ys[lo], ys[hi])
}

fn lin(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    if (x1 - x0).abs() < f64::MIN_POSITIVE {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Minimal linear congruential generator for reproducible direction sampling.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E3779B97F4A7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_uniform(&mut self) -> f64 {
        // Top 53 bits → (0,1).
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// A uniformly-distributed unit vector on the (dim−1)-sphere.
    fn unit_vector(&mut self, dim: usize) -> Vec<f64> {
        let mut v: Vec<f64> = (0..dim).map(|_| probit(self.next_uniform())).collect();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correlated bivariate lognormal-ish data for testing.
    fn dataset() -> Vec<Vec<f64>> {
        let mut rng = Lcg::new(7);
        (0..600)
            .map(|_| {
                let a = probit(rng.next_uniform());
                let b = probit(rng.next_uniform());
                // Correlate and make non-Gaussian.
                let x = (0.8 * a + 0.6 * b).exp();
                let y = (a - 0.3 * b) + 0.2 * (0.8 * a + 0.6 * b).powi(2);
                vec![x, y]
            })
            .collect()
    }

    #[test]
    fn forward_marginals_are_standard_normal() {
        let data = dataset();
        let ppmt = Ppmt::fit(&data, None, &PpmtParams::default()).unwrap();
        let g = ppmt.forward(&data);
        for k in 0..2 {
            let col: Vec<f64> = g.iter().map(|r| r[k]).collect();
            let n = col.len() as f64;
            let mean = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let sd = var.sqrt();
            let m4 = col.iter().map(|x| ((x - mean) / sd).powi(4)).sum::<f64>() / n;
            assert!(mean.abs() < 0.1, "col {k} mean {mean}");
            assert!((var - 1.0).abs() < 0.2, "col {k} var {var}");
            // Near-Gaussian kurtosis (3).
            assert!((m4 - 3.0).abs() < 1.0, "col {k} kurtosis {m4}");
        }
    }

    #[test]
    fn round_trips_training_data() {
        let data = dataset();
        let ppmt = Ppmt::fit(&data, None, &PpmtParams::default()).unwrap();
        let g = ppmt.forward(&data);
        let back = ppmt.back(&g);
        let mut max_err = 0.0f64;
        for (orig, rec) in data.iter().zip(&back) {
            for (a, b) in orig.iter().zip(rec) {
                max_err = max_err.max((a - b).abs());
            }
        }
        assert!(max_err < 1e-6, "round-trip error {max_err}");
    }

    #[test]
    fn decorrelates() {
        let data = dataset();
        let ppmt = Ppmt::fit(&data, None, &PpmtParams::default()).unwrap();
        let g = ppmt.forward(&data);
        let n = g.len() as f64;
        let m0 = g.iter().map(|r| r[0]).sum::<f64>() / n;
        let m1 = g.iter().map(|r| r[1]).sum::<f64>() / n;
        let cov = g.iter().map(|r| (r[0] - m0) * (r[1] - m1)).sum::<f64>() / n;
        assert!(cov.abs() < 0.15, "residual correlation {cov}");
    }
}
