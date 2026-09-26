//! Principal components and min/max autocorrelation factors (MAF).
//!
//! Both are linear maps `f = A·(x − mean)`. PCA rotates onto the eigenvectors
//! of the covariance (or correlation) matrix, so the scores are uncorrelated.
//! MAF spheres the data with PCA, then rotates onto the eigenvectors of the
//! variogram matrix Γ(h) of the sphered data at one lag: the factors are
//! uncorrelated at lag 0 and at lag h, ordered from most to least continuous.

use crate::error::{Result, TransformError};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use serde::{Deserialize, Serialize};

/// Rows `n × d`, all finite, `n ≥ 2`; returns `d`.
pub(crate) fn check_rows(data: &[Vec<f64>], weights: Option<&[f64]>) -> Result<usize> {
    if data.len() < 2 {
        return Err(TransformError::InsufficientData("need ≥ 2 rows".into()));
    }
    let dim = data[0].len();
    if dim == 0 || data.iter().any(|r| r.len() != dim) {
        return Err(TransformError::InvalidParameters(
            "ragged/empty rows".into(),
        ));
    }
    if data.iter().flatten().any(|v| !v.is_finite()) {
        return Err(TransformError::InvalidParameters("non-finite value".into()));
    }
    if let Some(w) = weights
        && (w.len() != data.len()
            || w.iter().any(|&x| !x.is_finite() || x < 0.0)
            || w.iter().sum::<f64>() <= 0.0)
    {
        return Err(TransformError::InvalidParameters(
            "weights must match rows, be ≥ 0 and sum > 0".into(),
        ));
    }
    Ok(dim)
}

/// Weighted mean and population covariance of the rows.
fn moments(data: &[Vec<f64>], weights: Option<&[f64]>) -> (DVector<f64>, DMatrix<f64>) {
    let dim = data[0].len();
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let total: f64 = (0..data.len()).map(w).sum();
    let mut mean = DVector::zeros(dim);
    for (i, r) in data.iter().enumerate() {
        mean += DVector::from_row_slice(r) * (w(i) / total);
    }
    let mut cov = DMatrix::zeros(dim, dim);
    for (i, r) in data.iter().enumerate() {
        let c = DVector::from_row_slice(r) - &mean;
        cov += &c * c.transpose() * (w(i) / total);
    }
    (mean, cov)
}

/// Eigenpairs sorted by decreasing eigenvalue; each eigenvector (column) has
/// its largest-magnitude entry positive.
fn eigen(m: DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let eig = SymmetricEigen::new(m);
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let mut vectors = eig.eigenvectors.select_columns(&order);
    for mut col in vectors.column_iter_mut() {
        let big = col
            .iter()
            .copied()
            .fold(0.0, |b: f64, v| if v.abs() > b.abs() { v } else { b });
        if big < 0.0 {
            col.neg_mut();
        }
    }
    (order.iter().map(|&i| eig.eigenvalues[i]).collect(), vectors)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Linear {
    mean: DVector<f64>,
    forward: DMatrix<f64>,
    inverse: DMatrix<f64>,
}

impl Linear {
    fn forward(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|r| {
                (&self.forward * (DVector::from_row_slice(r) - &self.mean))
                    .data
                    .into()
            })
            .collect()
    }

    fn back(&self, factors: &[Vec<f64>]) -> Vec<Vec<f64>> {
        factors
            .iter()
            .map(|r| {
                (&self.inverse * DVector::from_row_slice(r) + &self.mean)
                    .data
                    .into()
            })
            .collect()
    }
}

/// Principal component analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pca {
    map: Linear,
    eigenvalues: Vec<f64>,
    components: DMatrix<f64>,
}

impl Pca {
    /// Fits on `n` rows of `d` variables, optionally weighted; `standardize`
    /// uses the correlation instead of the covariance matrix.
    pub fn fit(data: &[Vec<f64>], weights: Option<&[f64]>, standardize: bool) -> Result<Self> {
        let dim = check_rows(data, weights)?;
        let (mean, cov) = moments(data, weights);
        let scale = DVector::from_fn(dim, |i, _| {
            if standardize {
                cov[(i, i)].sqrt().max(f64::MIN_POSITIVE)
            } else {
                1.0
            }
        });
        let corr = DMatrix::from_fn(dim, dim, |i, j| cov[(i, j)] / (scale[i] * scale[j]));
        let (eigenvalues, v) = eigen(corr);
        let forward = v.transpose() * DMatrix::from_diagonal(&scale.map(|s| 1.0 / s));
        let inverse = DMatrix::from_diagonal(&scale) * &v;
        Ok(Self {
            map: Linear {
                mean,
                forward,
                inverse,
            },
            eigenvalues,
            components: v.transpose(),
        })
    }

    /// Variance of each score, decreasing.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Unit eigenvectors as rows, in the order of [`Pca::eigenvalues`].
    pub fn components(&self) -> Vec<Vec<f64>> {
        rows(&self.components)
    }

    pub fn forward(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.map.forward(data)
    }

    pub fn back(&self, scores: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.map.back(scores)
    }
}

/// Min/max autocorrelation factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Maf {
    map: Linear,
    gammas: Vec<f64>,
}

impl Maf {
    /// Fits on `n` rows of `d` variables at `coords`, from the pairs whose
    /// separation lies within `lag ± tolerance` (omnidirectional).
    pub fn fit(
        data: &[Vec<f64>],
        coords: &[(f64, f64, f64)],
        lag: f64,
        tolerance: f64,
    ) -> Result<Self> {
        let dim = check_rows(data, None)?;
        if coords.len() != data.len() {
            return Err(TransformError::InvalidParameters(
                "coords length mismatch".into(),
            ));
        }
        if coords
            .iter()
            .any(|p| !(p.0.is_finite() && p.1.is_finite() && p.2.is_finite()))
        {
            return Err(TransformError::InvalidParameters(
                "coords must be finite".into(),
            ));
        }
        if !(lag > 0.0 && tolerance > 0.0 && lag.is_finite() && tolerance.is_finite()) {
            return Err(TransformError::InvalidParameters(
                "lag and tolerance must be > 0".into(),
            ));
        }
        let (mean, cov) = moments(data, None);
        let (lambda, v) = eigen(cov);
        if lambda[dim - 1] <= 1e-12 * lambda[0].max(f64::MIN_POSITIVE) {
            return Err(TransformError::FittingFailed("singular covariance".into()));
        }
        let root = DVector::from_vec(lambda.iter().map(|l| l.sqrt()).collect());
        let sphere = DMatrix::from_diagonal(&root.map(|r| 1.0 / r)) * v.transpose();
        let y: Vec<DVector<f64>> = data
            .iter()
            .map(|r| &sphere * (DVector::from_row_slice(r) - &mean))
            .collect();
        let mut gamma = DMatrix::zeros(dim, dim);
        let pairs = lag_pairs(coords, lag, tolerance)?;
        for &(i, j) in &pairs {
            let d = &y[i] - &y[j];
            gamma += &d * d.transpose();
        }
        let pairs = pairs.len();
        if pairs == 0 {
            return Err(TransformError::InsufficientData(
                "no pairs at this lag".into(),
            ));
        }
        let (mut gammas, mut b) = eigen(gamma / (2.0 * pairs as f64));
        gammas.reverse();
        b = b.select_columns(&(0..dim).rev().collect::<Vec<_>>());
        let forward = b.transpose() * sphere;
        let inverse = &v * DMatrix::from_diagonal(&root) * &b;
        Ok(Self {
            map: Linear {
                mean,
                forward,
                inverse,
            },
            gammas,
        })
    }

    /// Semivariogram of each factor at the fitted lag, increasing.
    pub fn gammas(&self) -> &[f64] {
        &self.gammas
    }

    pub fn forward(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.map.forward(data)
    }

    pub fn back(&self, factors: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.map.back(factors)
    }
}

/// Pairs `i < j` separated by `lag ± tolerance`, sorted by `(i, j)`.
fn lag_pairs(coords: &[(f64, f64, f64)], lag: f64, tolerance: f64) -> Result<Vec<(usize, usize)>> {
    let points: Vec<[f64; 3]> = coords.iter().map(|&(x, y, z)| [x, y, z]).collect();
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&points)
        .map_err(|e| TransformError::InvalidParameters(format!("{e:?}")))?;
    let radius2 = ((lag + tolerance) * (1.0 + 1e-9)).powi(2);
    let mut pairs = Vec::new();
    for (i, a) in coords.iter().enumerate() {
        let mut near: Vec<usize> = tree
            .query(&points[i])
            .within::<SquaredEuclidean<f64>>(radius2)
            .execute()
            .iter()
            .map(|r| r.item as usize)
            .filter(|&j| j > i)
            .collect();
        near.sort_unstable();
        pairs.extend(near.into_iter().filter_map(|j| {
            let c = coords[j];
            let h = ((a.0 - c.0).powi(2) + (a.1 - c.1).powi(2) + (a.2 - c.2).powi(2)).sqrt();
            ((h - lag).abs() <= tolerance).then_some((i, j))
        }));
    }
    Ok(pairs)
}

fn rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    m.row_iter().map(|r| r.iter().copied().collect()).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn uniforms(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            })
            .collect()
    }

    fn covariance(x: &[Vec<f64>]) -> DMatrix<f64> {
        moments(x, None).1
    }

    fn max_error(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn pca_scores_are_uncorrelated_with_eigenvalue_variances() {
        let u = uniforms(3, 1500);
        let data: Vec<Vec<f64>> = u
            .chunks(3)
            .map(|c| vec![c[0] * 4.0, c[0] + c[1], 10.0 * c[2] - c[1]])
            .collect();
        for standardize in [false, true] {
            let pca = Pca::fit(&data, None, standardize).unwrap();
            let scores = pca.forward(&data);
            let cov = covariance(&scores);
            let l = pca.eigenvalues();
            assert!(l.windows(2).all(|w| w[0] >= w[1]));
            for i in 0..3 {
                assert!((cov[(i, i)] - l[i]).abs() < 1e-9);
                for j in 0..i {
                    assert!(cov[(i, j)].abs() < 1e-9);
                }
            }
            assert!(max_error(&pca.back(&scores), &data) < 1e-9);
        }
    }

    #[test]
    fn maf_factors_are_uncorrelated_at_zero_and_lag() {
        let (nx, u) = (40, uniforms(5, 3200));
        let coords: Vec<_> = (0..nx * nx)
            .map(|i| ((i % nx) as f64, (i / nx) as f64, 0.0))
            .collect();
        let data: Vec<Vec<f64>> = coords
            .iter()
            .zip(u.chunks(2))
            .map(|(p, c)| {
                let smooth = (p.0 / 6.0).sin() + (p.1 / 7.0).cos();
                vec![smooth + 0.3 * c[0], smooth - c[1]]
            })
            .collect();
        let maf = Maf::fit(&data, &coords, 1.0, 0.01).unwrap();
        let f = maf.forward(&data);
        let cov = covariance(&f);
        assert!((cov - DMatrix::identity(2, 2)).abs().max() < 1e-9);
        let mut gamma = DMatrix::<f64>::zeros(2, 2);
        let mut pairs = 0.0;
        for i in 0..f.len() {
            for j in i + 1..f.len() {
                let (a, b) = (coords[i], coords[j]);
                if ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) - 1.0).abs() < 1e-9 {
                    let d = DVector::from_row_slice(&f[i]) - DVector::from_row_slice(&f[j]);
                    gamma += &d * d.transpose() / 2.0;
                    pairs += 1.0;
                }
            }
        }
        gamma /= pairs;
        assert!(gamma[(0, 1)].abs() < 1e-9);
        assert!((gamma[(0, 0)] - maf.gammas()[0]).abs() < 1e-9);
        assert!(maf.gammas()[0] < maf.gammas()[1]);
        assert!(max_error(&maf.back(&f), &data) < 1e-9);
    }

    #[test]
    fn lag_pairs_match_brute_force() {
        let u = uniforms(11, 1800);
        let scattered: Vec<_> = u
            .chunks(3)
            .map(|c| (c[0] * 20.0, c[1] * 20.0, c[2] * 5.0))
            .collect();
        let flat: Vec<_> = scattered.iter().map(|p| (p.0, p.1, 0.0)).collect();
        let grid: Vec<_> = (0..600)
            .map(|i| ((i % 10) as f64, (i / 10 % 10) as f64, 0.0))
            .collect();
        for (coords, lag, tolerance) in
            [(scattered, 3.0, 0.5), (flat, 2.0, 0.25), (grid, 1.0, 0.01)]
        {
            let mut brute = Vec::new();
            for (i, a) in coords.iter().enumerate() {
                for (j, c) in coords.iter().enumerate().skip(i + 1) {
                    let h =
                        ((a.0 - c.0).powi(2) + (a.1 - c.1).powi(2) + (a.2 - c.2).powi(2)).sqrt();
                    if (h - lag).abs() <= tolerance {
                        brute.push((i, j));
                    }
                }
            }
            assert!(!brute.is_empty());
            assert_eq!(lag_pairs(&coords, lag, tolerance).unwrap(), brute);
        }
    }
}
