//! Imputation of missing variables from the present ones.
//!
//! Each variable is normal-scored on its observed values; the mean and
//! covariance of the scores are fitted by expectation-maximization over all
//! rows, missing entries included. A missing score is drawn from its Gaussian
//! distribution conditional on the scores present in its row, then
//! back-transformed, so imputed values keep the histograms and the
//! correlation of the data.

use crate::error::{Result, TransformError};
use crate::normal_score::{NormalScoreTable, transform as normal_score};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// Fitted Gaussian imputation of missing (NaN) variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianImputer {
    tables: Vec<NormalScoreTable>,
    mean: DVector<f64>,
    covariance: DMatrix<f64>,
}

/// Conditional mean of the whole row (observed entries unchanged) and
/// covariance of its missing entries, whose indices are returned too.
fn conditional(
    mean: &DVector<f64>,
    cov: &DMatrix<f64>,
    z: &[f64],
) -> Result<(DVector<f64>, DMatrix<f64>, Vec<usize>)> {
    let (o, m): (Vec<usize>, Vec<usize>) = (0..z.len()).partition(|&v| z[v].is_finite());
    let mut filled = DVector::from_row_slice(z);
    if o.is_empty() {
        return Ok((mean.clone(), cov.clone(), m));
    }
    let s_oo = cov.select_rows(&o).select_columns(&o);
    let s_mo = cov.select_rows(&m).select_columns(&o);
    let chol = s_oo
        .cholesky()
        .ok_or_else(|| TransformError::InvalidParameters("score covariance is singular".into()))?;
    let resid = DVector::from_iterator(o.len(), o.iter().map(|&v| z[v] - mean[v]));
    let mu = &s_mo * chol.solve(&resid);
    for (k, &v) in m.iter().enumerate() {
        filled[v] = mean[v] + mu[k];
    }
    let c = cov.select_rows(&m).select_columns(&m) - &s_mo * chol.solve(&s_mo.transpose());
    Ok((filled, c, m))
}

impl GaussianImputer {
    /// Fits on `n` rows of `d` variables, NaN where missing; `weights`
    /// (e.g. declustering) shape the scores and the covariance.
    pub fn fit(data: &[Vec<f64>], weights: Option<&[f64]>) -> Result<Self> {
        let dim = data.first().map_or(0, Vec::len);
        if dim == 0 || data.iter().any(|r| r.len() != dim) {
            return Err(TransformError::InvalidParameters(
                "ragged/empty rows".into(),
            ));
        }
        if data.iter().flatten().any(|v| v.is_infinite()) {
            return Err(TransformError::InvalidParameters("infinite value".into()));
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
        let w = |i: usize| weights.map_or(1.0, |w| w[i]);
        let mut scores = vec![vec![f64::NAN; dim]; data.len()];
        let mut tables = Vec::with_capacity(dim);
        for v in 0..dim {
            let seen: Vec<usize> = (0..data.len()).filter(|&i| !data[i][v].is_nan()).collect();
            if seen.len() < 2 {
                return Err(TransformError::InsufficientData(format!(
                    "variable {v} has < 2 values"
                )));
            }
            let values: Vec<f64> = seen.iter().map(|&i| data[i][v]).collect();
            let ws: Vec<f64> = seen.iter().map(|&i| w(i)).collect();
            let table = normal_score(&values, Some(&ws))?.table;
            for &i in &seen {
                scores[i][v] = table.forward(data[i][v]);
            }
            tables.push(table);
        }
        let total: f64 = (0..data.len()).map(w).sum();
        let (mut mean, mut cov) = (DVector::zeros(dim), DMatrix::identity(dim, dim));
        for _ in 0..500 {
            let (mut m1, mut m2) = (DVector::zeros(dim), DMatrix::zeros(dim, dim));
            for (i, z) in scores.iter().enumerate() {
                let (x, c, missing) = conditional(&mean, &cov, z)?;
                let mut extra = DMatrix::zeros(dim, dim);
                for (a, &p) in missing.iter().enumerate() {
                    for (b, &q) in missing.iter().enumerate() {
                        extra[(p, q)] = c[(a, b)];
                    }
                }
                m2 += (&x * x.transpose() + extra) * (w(i) / total);
                m1 += x * (w(i) / total);
            }
            let next = &m2 - &m1 * m1.transpose();
            let change = (&next - &cov).amax().max((&m1 - &mean).amax());
            (mean, cov) = (m1, next);
            if change < 1e-10 {
                break;
            }
        }
        Ok(Self {
            tables,
            mean,
            covariance: cov,
        })
    }

    /// Number of variables.
    pub fn dim(&self) -> usize {
        self.tables.len()
    }

    /// Correlation matrix of the normal scores.
    pub fn correlation(&self) -> DMatrix<f64> {
        let s = self.covariance.diagonal().map(f64::sqrt);
        self.covariance.component_div(&(&s * s.transpose()))
    }

    /// `data` with every NaN replaced by a draw conditional on the values
    /// present in its row; the same `seed` gives the same draws.
    pub fn impute(&self, data: &[Vec<f64>], seed: u64) -> Result<Vec<Vec<f64>>> {
        let dim = self.dim();
        if data.iter().any(|r| r.len() != dim) {
            return Err(TransformError::InvalidParameters(format!(
                "rows must have {dim} values"
            )));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        data.iter()
            .map(|row| {
                if !row.iter().any(|v| v.is_nan()) {
                    return Ok(row.clone());
                }
                let z: Vec<f64> = row
                    .iter()
                    .zip(&self.tables)
                    .map(|(&x, t)| if x.is_nan() { x } else { t.forward(x) })
                    .collect();
                let (mu, c, missing) = conditional(&self.mean, &self.covariance, &z)?;
                let jitter = DMatrix::identity(missing.len(), missing.len()) * 1e-12;
                let l = (c + jitter)
                    .cholesky()
                    .map_or_else(|| DMatrix::zeros(missing.len(), missing.len()), |ch| ch.l());
                let e = DVector::from_fn(missing.len(), |_, _| StandardNormal.sample(&mut rng));
                let draw = l * e;
                let mut out = row.clone();
                for (k, &v) in missing.iter().enumerate() {
                    out[v] = self.tables[v].back(mu[v] + draw[k]);
                }
                Ok(out)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            c += (x - ma) * (y - mb);
            va += (x - ma).powi(2);
            vb += (y - mb).powi(2);
        }
        c / (va * vb).sqrt()
    }

    /// Lognormal pair with log correlation 0.8; the second variable is
    /// missing in 40% of rows, the first in 10% of the others.
    fn samples() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = StdRng::seed_from_u64(5);
        let full: Vec<Vec<f64>> = (0..3000)
            .map(|_| {
                let a: f64 = StandardNormal.sample(&mut rng);
                let b: f64 = StandardNormal.sample(&mut rng);
                vec![a.exp(), (0.8 * a + 0.6 * b).exp()]
            })
            .collect();
        let holed = full
            .iter()
            .enumerate()
            .map(|(i, r)| match i % 10 {
                0..=3 => vec![r[0], f64::NAN],
                4 => vec![f64::NAN, r[1]],
                _ => r.clone(),
            })
            .collect();
        (full, holed)
    }

    #[test]
    fn keeps_observed_values_and_reproduces_correlation() {
        let (_, holed) = samples();
        let imputer = GaussianImputer::fit(&holed, None).unwrap();
        assert!((imputer.correlation()[(0, 1)] - 0.8).abs() < 0.03);
        let out = imputer.impute(&holed, 3).unwrap();
        assert_eq!(out, imputer.impute(&holed, 3).unwrap());
        for (r, h) in out.iter().zip(&holed) {
            for (x, y) in r.iter().zip(h) {
                assert!(x.is_finite());
                if !y.is_nan() {
                    assert_eq!(x.to_bits(), y.to_bits());
                }
            }
        }
        let imputed: Vec<&Vec<f64>> = out
            .iter()
            .zip(&holed)
            .filter(|(_, h)| h.iter().any(|v| v.is_nan()))
            .map(|(r, _)| r)
            .collect();
        for rows in [out.iter().collect::<Vec<_>>(), imputed] {
            let a: Vec<f64> = rows.iter().map(|r| r[0].ln()).collect();
            let b: Vec<f64> = rows.iter().map(|r| r[1].ln()).collect();
            let r = corr(&a, &b);
            assert!((r - 0.8).abs() < 0.05, "correlation {r}");
            let n = b.len() as f64;
            let var =
                b.iter().map(|x| x * x).sum::<f64>() / n - (b.iter().sum::<f64>() / n).powi(2);
            assert!((var - 1.0).abs() < 0.15, "variance {var}");
        }
    }

    #[test]
    fn ties_add_no_correlation() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut detect = || {
            let x: f64 = StandardNormal.sample(&mut rng);
            x.max(0.0)
        };
        let data: Vec<Vec<f64>> = (0..2000).map(|_| vec![detect(), detect()]).collect();
        let r = GaussianImputer::fit(&data, None).unwrap().correlation()[(0, 1)];
        assert!(r.abs() < 0.05, "correlation {r}");
    }

    #[test]
    fn rejects_bad_input() {
        assert!(GaussianImputer::fit(&[vec![1.0, f64::NAN], vec![2.0, f64::NAN]], None).is_err());
        assert!(GaussianImputer::fit(&[vec![1.0, f64::INFINITY]], None).is_err());
        let (_, holed) = samples();
        let imputer = GaussianImputer::fit(&holed, None).unwrap();
        assert!(imputer.impute(&[vec![1.0]], 0).is_err());
    }
}
