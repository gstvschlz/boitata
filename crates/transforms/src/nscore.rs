//! Normal-score transform and its back-transform.
//!
//! Maps data to standard-normal scores by matching (optionally declustered) empirical
//! quantiles to Gaussian quantiles. The stored transform table enables monotone
//! back-transformation of simulated/estimated scores, with linear tail extrapolation.

use crate::error::{Result, TransformError};
use crate::normal::probit;
use serde::{Deserialize, Serialize};

/// A monotone data↔score mapping table (sorted by data value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NscoreTable {
    /// Sorted original data values.
    pub values: Vec<f64>,
    /// Corresponding normal scores (increasing).
    pub scores: Vec<f64>,
}

/// Result of a forward transform: scores aligned with the input, plus the table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nscore {
    pub scores: Vec<f64>,
    pub table: NscoreTable,
}

/// Forward normal-score transform.
///
/// `weights` (optional) are declustering weights used to build the empirical CDF;
/// when omitted, samples are weighted equally.
pub fn transform(values: &[f64], weights: Option<&[f64]>) -> Result<Nscore> {
    let n = values.len();
    if n == 0 {
        return Err(TransformError::InsufficientData("no values".into()));
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(TransformError::InvalidParameters(
            "weights length mismatch".into(),
        ));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

    let w: Vec<f64> = match weights {
        Some(w) => {
            let s: f64 = w.iter().sum();
            if s <= 0.0 {
                return Err(TransformError::InvalidParameters("weights sum ≤ 0".into()));
            }
            w.iter().map(|x| x / s).collect()
        }
        None => vec![1.0 / n as f64; n],
    };

    // Cumulative probability at the midpoint of each ordered sample's weight.
    let mut scores = vec![0.0f64; n];
    let mut cum = 0.0;
    let mut table_vals = Vec::with_capacity(n);
    let mut table_scores = Vec::with_capacity(n);
    for &idx in &order {
        let wi = w[idx];
        let p = cum + wi / 2.0;
        cum += wi;
        let z = probit(p);
        scores[idx] = z;
        table_vals.push(values[idx]);
        table_scores.push(z);
    }

    Ok(Nscore {
        scores,
        table: NscoreTable {
            values: table_vals,
            scores: table_scores,
        },
    })
}

impl NscoreTable {
    /// Back-transform a normal score to data space (monotone interpolation).
    ///
    /// Scores beyond the tabulated range are linearly extrapolated from the two
    /// nearest table entries (clamped so the result stays finite).
    pub fn back(&self, score: f64) -> f64 {
        lookup(&self.scores, &self.values, score)
    }

    /// Normal score of a value; the inverse of [`NscoreTable::back`]. A value
    /// tied in the table gets the mean score of its ties.
    pub fn forward(&self, value: f64) -> f64 {
        let first = self.values.partition_point(|v| *v < value);
        let last = self.values.partition_point(|v| *v <= value);
        if last > first {
            return self.scores[first..last].iter().sum::<f64>() / (last - first) as f64;
        }
        lookup(&self.values, &self.scores, value)
    }
}

fn lookup(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return ys[0];
    }
    let hi = xs.partition_point(|v| *v <= x).clamp(1, n - 1);
    interp(x, xs[hi - 1], xs[hi], ys[hi - 1], ys[hi])
}

fn interp(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    if (x1 - x0).abs() < f64::MIN_POSITIVE {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_are_standard_normal_ish() {
        let vals: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let ns = transform(&vals, None).unwrap();
        let mean: f64 = ns.scores.iter().sum::<f64>() / 100.0;
        assert!(mean.abs() < 1e-6, "mean {mean}");
        // Symmetric range.
        let mn = ns.scores.iter().cloned().fold(f64::MAX, f64::min);
        let mx = ns.scores.iter().cloned().fold(f64::MIN, f64::max);
        assert!((mn + mx).abs() < 1e-6);
    }

    #[test]
    fn forward_inverts_back_and_averages_ties() {
        let ns = transform(&[0.0, 0.0, 1.0, 3.0, 7.0], None).unwrap();
        for y in [-0.3, 0.1, 0.6] {
            assert!((ns.table.forward(ns.table.back(y)) - y).abs() < 1e-12);
        }
        let tied = ns.table.forward(0.0);
        assert!((tied - (ns.scores[0] + ns.scores[1]) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn back_transform_round_trips() {
        let vals = vec![2.0, 5.0, 1.0, 9.0, 4.0, 7.0];
        let ns = transform(&vals, None).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            let back = ns.table.back(ns.scores[i]);
            assert!((back - v).abs() < 1e-6, "value {v} back {back}");
        }
    }

    #[test]
    fn monotone_back_transform() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ns = transform(&vals, None).unwrap();
        let a = ns.table.back(-3.0);
        let b = ns.table.back(0.0);
        let c = ns.table.back(3.0);
        assert!(a < b && b < c);
    }

    #[test]
    fn weighted_transform_runs() {
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![0.4, 0.3, 0.2, 0.1];
        let ns = transform(&vals, Some(&w)).unwrap();
        assert_eq!(ns.scores.len(), 4);
        // Ordering preserved: larger value → larger score.
        assert!(ns.scores[0] < ns.scores[3]);
    }
}
