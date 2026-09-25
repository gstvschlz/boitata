//! Post-processing of simulation ensembles into uncertainty summaries.
//!
//! Given many realizations over the same grid, compute per-node statistics: mean,
//! standard deviation, quantiles (e.g. P10/P50/P90), and the probability of exceeding
//! a cutoff — the core inputs to risk-qualified resource reporting.

use crate::sgs::Realization;

/// Per-node ensemble summary.
#[derive(Debug, Clone)]
pub struct NodeStats {
    pub mean: f64,
    pub std_dev: f64,
    /// Requested quantiles (same order as the `quantiles` argument).
    pub quantiles: Vec<f64>,
}

/// Collect the values at grid node `node` across all realizations.
fn column(reals: &[Realization], node: usize) -> Vec<f64> {
    reals.iter().map(|r| r.values[node]).collect()
}

/// Empirical quantile of `sorted` (ascending) using linear interpolation.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Per-node statistics across the ensemble for the requested `quantiles` (e.g. `[0.1,0.5,0.9]`).
pub fn node_stats(reals: &[Realization], quantiles: &[f64]) -> Vec<NodeStats> {
    if reals.is_empty() {
        return vec![];
    }
    let n_nodes = reals[0].values.len();
    let mut out = Vec::with_capacity(n_nodes);
    for node in 0..n_nodes {
        let mut col = column(reals, node);
        let n = col.len() as f64;
        let mean = col.iter().sum::<f64>() / n;
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let qs = quantiles
            .iter()
            .map(|&q| quantile_sorted(&col, q))
            .collect();
        out.push(NodeStats {
            mean,
            std_dev: var.sqrt(),
            quantiles: qs,
        });
    }
    out
}

/// Per-node probability that the simulated value exceeds `cutoff`.
pub fn probability_above(reals: &[Realization], cutoff: f64) -> Vec<f64> {
    if reals.is_empty() {
        return vec![];
    }
    let n_nodes = reals[0].values.len();
    let n = reals.len() as f64;
    (0..n_nodes)
        .map(|node| reals.iter().filter(|r| r.values[node] > cutoff).count() as f64 / n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ens() -> Vec<Realization> {
        vec![
            Realization {
                values: vec![1.0, 10.0],
            },
            Realization {
                values: vec![2.0, 20.0],
            },
            Realization {
                values: vec![3.0, 30.0],
            },
            Realization {
                values: vec![4.0, 40.0],
            },
        ]
    }

    #[test]
    fn mean_and_quantiles() {
        let stats = node_stats(&ens(), &[0.5]);
        assert!((stats[0].mean - 2.5).abs() < 1e-9);
        // Median of [1,2,3,4] via interpolation = 2.5.
        assert!((stats[0].quantiles[0] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn probability_above_cutoff() {
        // Node 0 values [1,2,3,4]; P(>2.5) = 2/4 = 0.5.
        let p = probability_above(&ens(), 2.5);
        assert!((p[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn quantile_endpoints() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_sorted(&s, 0.0), 1.0);
        assert_eq!(quantile_sorted(&s, 1.0), 5.0);
    }
}
