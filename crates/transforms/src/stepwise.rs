//! Stepwise conditional transform.
//!
//! Variable 1 is normal-scored; variable k is normal-scored separately within
//! each joint class of the already transformed variables 1..k−1, each cut into
//! `classes` equal-probability Gaussian classes. Classes with fewer than
//! [`MIN_SAMPLES`] samples use the marginal table of variable k. Tied values
//! share a class. Optional declustering weights enter every normal score. The
//! outputs are independent standard normals up to the class resolution.

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
    /// Tables by joint class, sorted by key.
    conditional: Vec<(Vec<u16>, NormalScoreTable)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepwiseConditional {
    classes: usize,
    variables: Vec<Variable>,
}

impl StepwiseConditional {
    pub fn fit(data: &[Vec<f64>], weights: Option<&[f64]>, classes: usize) -> Result<Self> {
        let dim = check_rows(data, weights)?;
        if !(1..=u16::MAX as usize).contains(&classes) {
            return Err(TransformError::InvalidParameters(format!(
                "classes must be in 1..={}",
                u16::MAX
            )));
        }
        let mut sct = Self {
            classes,
            variables: Vec::with_capacity(dim),
        };
        let mut gauss = vec![Vec::with_capacity(dim); data.len()];
        for k in 0..dim {
            let column: Vec<f64> = data.iter().map(|r| r[k]).collect();
            let marginal = normal_score(&column, weights)?;
            let mut groups: BTreeMap<Vec<u16>, Vec<usize>> = BTreeMap::new();
            for (i, g) in gauss.iter().enumerate() {
                groups.entry(sct.key(g)).or_default().push(i);
            }
            let mut variable = Variable {
                marginal: marginal.table,
                conditional: Vec::new(),
            };
            for (key, members) in groups {
                let local: Option<Vec<f64>> =
                    weights.map(|w| members.iter().map(|&i| w[i]).collect());
                if k == 0
                    || members.len() < MIN_SAMPLES
                    || local.as_ref().is_some_and(|w| w.iter().sum::<f64>() <= 0.0)
                {
                    for &i in &members {
                        gauss[i].push(variable.marginal.forward(column[i]));
                    }
                    continue;
                }
                let values: Vec<f64> = members.iter().map(|&i| column[i]).collect();
                let table = normal_score(&values, local.as_deref())?.table;
                for &i in &members {
                    gauss[i].push(table.forward(column[i]));
                }
                variable.conditional.push((key, table));
            }
            sct.variables.push(variable);
        }
        Ok(sct)
    }

    /// Joint class of the Gaussian values `previous`.
    fn key(&self, previous: &[f64]) -> Vec<u16> {
        previous
            .iter()
            .map(|&y| ((phi(y) * self.classes as f64) as usize).min(self.classes - 1) as u16)
            .collect()
    }

    fn table(&self, k: usize, previous: &[f64]) -> &NormalScoreTable {
        let v = &self.variables[k];
        let key = self.key(previous);
        v.conditional
            .binary_search_by(|(k, _)| k.cmp(&key))
            .map_or(&v.marginal, |i| &v.conditional[i].1)
    }

    /// Score of variable `previous.len()` for `value`, given the scores of the
    /// variables before it.
    pub fn forward_one(&self, previous: &[f64], value: f64) -> f64 {
        self.table(previous.len(), previous).forward(value)
    }

    /// Value of variable `previous.len()` for `score`, given the scores of the
    /// variables before it.
    pub fn back_one(&self, previous: &[f64], score: f64) -> f64 {
        self.table(previous.len(), previous).back(score)
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
        let sct = StepwiseConditional::fit(&banana, None, 30).unwrap();
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

    #[test]
    fn many_variables_round_trip() {
        let u = uniforms(5, 16 * 4000);
        let data: Vec<Vec<f64>> = u
            .chunks(16)
            .map(|c| {
                let common = probit(c[0]);
                c.iter().map(|&v| (common + probit(v)).exp()).collect()
            })
            .collect();
        let sct = StepwiseConditional::fit(&data, None, 30).unwrap();
        let json = serde_json::to_string(&sct).unwrap();
        let sct: StepwiseConditional = serde_json::from_str(&json).unwrap();
        let back = sct.back(&sct.forward(&data));
        let err = back.iter().flatten().zip(data.iter().flatten());
        assert!(err.map(|(a, b)| (a - b).abs() / b).fold(0.0, f64::max) < 1e-9);
        assert!(StepwiseConditional::fit(&data, None, 1 << 16).is_err());
    }

    #[test]
    fn weights_decluster_every_score_and_ties_share_a_class() {
        let u = uniforms(3, 2 * 2000);
        let data: Vec<Vec<f64>> = u
            .chunks(2)
            .map(|c| vec![(c[0] * 4.0).floor(), c[0] + probit(c[1])])
            .collect();
        let weights: Vec<f64> = data
            .iter()
            .map(|r| if r[1] > r[0] / 4.0 + 0.5 { 3.0 } else { 1.0 })
            .collect();
        let sct = StepwiseConditional::fit(&data, Some(&weights), 4).unwrap();
        let g = sct.forward(&data);
        for class in 0..4 {
            let members: Vec<usize> = (0..data.len())
                .filter(|&i| data[i][0] == class as f64)
                .collect();
            assert!(members.iter().all(|&i| g[i][0] == g[members[0]][0]));
            let mean = |w: &dyn Fn(usize) -> f64| {
                let total: f64 = members.iter().map(|&i| w(i)).sum();
                members.iter().map(|&i| g[i][1] * w(i)).sum::<f64>() / total
            };
            let declustered = mean(&|i| weights[i]);
            assert!(declustered.abs() < 0.05, "class {class}: {declustered}");
            assert!(mean(&|_| 1.0) < -0.2, "class {class}");
        }
        let back = sct.back(&g);
        let err = back.iter().flatten().zip(data.iter().flatten());
        assert!(err.map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) < 1e-9);
        assert!(StepwiseConditional::fit(&data, Some(&weights[1..]), 4).is_err());
    }

    #[test]
    fn joint_classes_that_overflowed_u64_differ() {
        let sct = StepwiseConditional {
            classes: 256,
            variables: Vec::new(),
        };
        let (low, next) = (-9.0, probit(1.5 / 256.0));
        let mut a = vec![low; 9];
        a[0] = next;
        let wrapped = |g: &[f64]| {
            sct.key(g)
                .iter()
                .fold(0u64, |k, &c| k.wrapping_mul(256).wrapping_add(c as u64))
        };
        assert_eq!(wrapped(&a), wrapped(&[low; 9]));
        assert_ne!(sct.key(&a), sct.key(&[low; 9]));
    }
}
