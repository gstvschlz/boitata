//! Stepwise conditional transform.
//!
//! Variable 1 is normal-scored; variable k is normal-scored separately within
//! each joint class of the already transformed variables 1..k−1, each cut into
//! `classes` equal-probability Gaussian classes. Classes with fewer than
//! [`MIN_SAMPLES`] samples use the marginal table of variable k. The outputs
//! are independent standard normals up to the class resolution.

use crate::error::{Result, TransformError};
use crate::normal::phi;
use crate::normal_score::{NormalScoreTable, transform as normal_score};
use crate::pca::check_rows;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MIN_SAMPLES: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Variable {
    marginal: NormalScoreTable,
    conditional: BTreeMap<u64, NormalScoreTable>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepwiseConditional {
    classes: usize,
    variables: Vec<Variable>,
}

impl StepwiseConditional {
    pub fn fit(data: &[Vec<f64>], classes: usize) -> Result<Self> {
        let dim = check_rows(data, None)?;
        if classes < 1 {
            return Err(TransformError::InvalidParameters(
                "classes must be ≥ 1".into(),
            ));
        }
        let mut sct = Self {
            classes,
            variables: Vec::with_capacity(dim),
        };
        let mut gauss = vec![Vec::with_capacity(dim); data.len()];
        for k in 0..dim {
            let column: Vec<f64> = data.iter().map(|r| r[k]).collect();
            let marginal = normal_score(&column, None)?;
            let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
            for (i, g) in gauss.iter().enumerate() {
                groups.entry(sct.key(g)).or_default().push(i);
            }
            let mut variable = Variable {
                marginal: marginal.table,
                conditional: BTreeMap::new(),
            };
            for (key, members) in groups {
                if k == 0 || members.len() < MIN_SAMPLES {
                    for &i in &members {
                        gauss[i].push(marginal.scores[i]);
                    }
                    continue;
                }
                let values: Vec<f64> = members.iter().map(|&i| column[i]).collect();
                let local = normal_score(&values, None)?;
                for (&i, &s) in members.iter().zip(&local.scores) {
                    gauss[i].push(s);
                }
                variable.conditional.insert(key, local.table);
            }
            sct.variables.push(variable);
        }
        Ok(sct)
    }

    /// Joint class of the Gaussian values `previous`.
    fn key(&self, previous: &[f64]) -> u64 {
        previous.iter().fold(0u64, |key, &y| {
            let class = ((phi(y) * self.classes as f64) as usize).min(self.classes - 1);
            key.wrapping_mul(self.classes as u64)
                .wrapping_add(class as u64)
        })
    }

    fn table(&self, k: usize, previous: &[f64]) -> &NormalScoreTable {
        let v = &self.variables[k];
        v.conditional
            .get(&self.key(previous))
            .unwrap_or(&v.marginal)
    }

    pub fn dim(&self) -> usize {
        self.variables.len()
    }

    pub fn forward(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                let mut g = Vec::with_capacity(row.len());
                for (k, &x) in row.iter().enumerate() {
                    g.push(self.table(k, &g).forward(x));
                }
                g
            })
            .collect()
    }

    pub fn back(&self, gauss: &[Vec<f64>]) -> Vec<Vec<f64>> {
        gauss
            .iter()
            .map(|g| {
                (0..g.len())
                    .map(|k| self.table(k, &g[..k]).back(g[k]))
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normal::probit;
    use crate::pca::tests::uniforms;

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
        let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
        cov / (va * vb).sqrt()
    }

    #[test]
    fn removes_nonlinear_dependence_and_round_trips() {
        let u = uniforms(11, 6000);
        let banana: Vec<Vec<f64>> = u
            .chunks(2)
            .map(|c| {
                let x = probit(c[0]);
                vec![x, x * x + 0.3 * probit(c[1])]
            })
            .collect();
        let sct = StepwiseConditional::fit(&banana, 30).unwrap();
        let g = sct.forward(&banana);
        let (g0, g1): (Vec<f64>, Vec<f64>) = g.iter().map(|r| (r[0], r[1])).unzip();
        let squared: Vec<f64> = g0.iter().map(|x| x * x).collect();
        let raw: Vec<f64> = banana.iter().map(|r| r[1]).collect();
        assert!(corr(&squared, &raw) > 0.9);
        assert!(corr(&g0, &g1).abs() < 0.1);
        assert!(corr(&squared, &g1).abs() < 0.1);
        let n = g1.len() as f64;
        let mean = g1.iter().sum::<f64>() / n;
        let var = g1.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.1);
        let back = sct.back(&g);
        let err = back.iter().flatten().zip(banana.iter().flatten());
        assert!(err.map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) < 1e-9);
    }
}
