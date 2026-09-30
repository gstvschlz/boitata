//! Transiograms for categorical variables (facies / rock types).
//!
//! A transiogram `t_ij(h)` is the probability of observing category `j` at lag `h`
//! from a location in category `i`. Diagonal terms describe auto-correlation of a
//! category; off-diagonal terms describe cross-transitions. Used by indicator/
//! multiple-point methods and sequential indicator simulation of facies.

use crate::error::{Result, VarioError};
use serde::{Deserialize, Serialize};

/// Markov-type exponential transiogram model over `k` categories.
///
/// `t_ii(h) = p_i + (1 − p_i)·e^{−h/r}` and `t_ij(h) = p_j·(1 − e^{−h/r})` for `i≠j`,
/// where `p_j` are marginal proportions and `r` a correlation range. Rows sum to 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Unchecked")]
pub struct Transiogram {
    /// Marginal category proportions (must sum to ~1).
    pub proportions: Vec<f64>,
    /// Correlation range (meters).
    pub range: f64,
}

#[derive(Deserialize)]
struct Unchecked {
    proportions: Vec<f64>,
    range: f64,
}

impl TryFrom<Unchecked> for Transiogram {
    type Error = VarioError;

    fn try_from(u: Unchecked) -> Result<Self> {
        Self::new(u.proportions, u.range)
    }
}

impl Transiogram {
    pub fn new(proportions: Vec<f64>, range: f64) -> Result<Self> {
        if proportions.is_empty() {
            return Err(VarioError::InvalidParameters("no categories".into()));
        }
        if range <= 0.0 {
            return Err(VarioError::InvalidParameters(
                "range must be positive".into(),
            ));
        }
        let sum: f64 = proportions.iter().sum();
        if (sum - 1.0).abs() > 1e-3 {
            return Err(VarioError::InvalidParameters(format!(
                "proportions must sum to 1 (got {sum:.4})"
            )));
        }
        Ok(Self { proportions, range })
    }

    pub fn n_categories(&self) -> usize {
        self.proportions.len()
    }

    /// Transition probability `t_ij(h)`.
    pub fn transition(&self, i: usize, j: usize, h: f64) -> f64 {
        let decay = (-h / self.range).exp();
        if i == j {
            self.proportions[i] + (1.0 - self.proportions[i]) * decay
        } else {
            self.proportions[j] * (1.0 - decay)
        }
    }

    /// Full transition matrix at lag `h` (row-major, rows sum to 1).
    pub fn matrix(&self, h: f64) -> Vec<Vec<f64>> {
        let k = self.n_categories();
        (0..k)
            .map(|i| (0..k).map(|j| self.transition(i, j, h)).collect())
            .collect()
    }
}

/// Empirical transiogram estimated from categorical samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalTransiogram {
    pub n_categories: usize,
    pub lags: Vec<f64>,
    /// `probs[bin][i][j]` = observed P(category j at lag | category i).
    pub probs: Vec<Vec<Vec<f64>>>,
    pub counts: Vec<usize>,
}

/// Estimate an empirical transiogram from categorical samples.
///
/// `categories[k]` is the integer category label (0-based) at `locations[k]`.
pub fn empirical_transiogram(
    locations: &[(f64, f64, f64)],
    categories: &[usize],
    n_categories: usize,
    max_lag: f64,
    lag_width: f64,
) -> Result<EmpiricalTransiogram> {
    if locations.len() != categories.len() {
        return Err(VarioError::InsufficientData(
            "locations and categories length mismatch".into(),
        ));
    }
    if n_categories == 0 || lag_width <= 0.0 || max_lag <= 0.0 {
        return Err(VarioError::InvalidParameters(
            "invalid transiogram parameters".into(),
        ));
    }
    for &c in categories {
        if c >= n_categories {
            return Err(VarioError::InvalidParameters(format!(
                "category label {c} ≥ n_categories {n_categories}"
            )));
        }
    }

    let n_bins = (max_lag / lag_width).ceil().max(1.0) as usize;
    // transition_counts[bin][i][j], row_totals[bin][i]
    let mut trans = vec![vec![vec![0usize; n_categories]; n_categories]; n_bins];
    let mut row_totals = vec![vec![0usize; n_categories]; n_bins];
    let mut pair_counts = vec![0usize; n_bins];

    let n = locations.len();
    for a in 0..n {
        for b in (a + 1)..n {
            let pa = locations[a];
            let pb = locations[b];
            let dx = pb.0 - pa.0;
            let dy = pb.1 - pa.1;
            let dz = pb.2 - pa.2;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist == 0.0 || dist > max_lag {
                continue;
            }
            let bin = ((dist / lag_width).floor() as usize).min(n_bins - 1);
            let (ci, cj) = (categories[a], categories[b]);
            // Count both directions (transiogram is directional; here we symmetrize).
            trans[bin][ci][cj] += 1;
            trans[bin][cj][ci] += 1;
            row_totals[bin][ci] += 1;
            row_totals[bin][cj] += 1;
            pair_counts[bin] += 1;
        }
    }

    let mut lags = Vec::new();
    let mut probs = Vec::new();
    let mut counts = Vec::new();
    for bin in 0..n_bins {
        if pair_counts[bin] == 0 {
            continue;
        }
        let center = (bin as f64 + 0.5) * lag_width;
        let mut mat = vec![vec![0.0; n_categories]; n_categories];
        for i in 0..n_categories {
            let total = row_totals[bin][i];
            if total > 0 {
                for j in 0..n_categories {
                    mat[i][j] = trans[bin][i][j] as f64 / total as f64;
                }
            }
        }
        lags.push(center);
        probs.push(mat);
        counts.push(pair_counts[bin]);
    }

    Ok(EmpiricalTransiogram {
        n_categories,
        lags,
        probs,
        counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_is_checked() {
        let t = Transiogram::new(vec![0.5, 0.3, 0.2], 50.0).unwrap();
        let back: Transiogram = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back.proportions, t.proportions);
        assert_eq!(back.range, t.range);
        let bad = r#"{"proportions":[0.5,0.3],"range":50.0}"#;
        assert!(serde_json::from_str::<Transiogram>(bad).is_err());
    }

    #[test]
    fn rows_sum_to_one() {
        let t = Transiogram::new(vec![0.5, 0.3, 0.2], 50.0).unwrap();
        for &h in &[0.0, 10.0, 100.0, 1000.0] {
            let m = t.matrix(h);
            for row in &m {
                let s: f64 = row.iter().sum();
                assert!((s - 1.0).abs() < 1e-9, "row sum {s} at h={h}");
            }
        }
    }

    #[test]
    fn auto_transition_decays_to_proportion() {
        let t = Transiogram::new(vec![0.7, 0.3], 40.0).unwrap();
        assert!((t.transition(0, 0, 0.0) - 1.0).abs() < 1e-9); // certain at h=0
        assert!((t.transition(0, 0, 1e6) - 0.7).abs() < 1e-6); // → marginal at large h
    }

    #[test]
    fn empirical_two_facies() {
        // Alternating facies along a line.
        let locs: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64 * 10.0, 0.0, 0.0)).collect();
        let cats: Vec<usize> = (0..10).map(|i| i % 2).collect();
        let e = empirical_transiogram(&locs, &cats, 2, 50.0, 10.0).unwrap();
        assert_eq!(e.n_categories, 2);
        assert!(!e.lags.is_empty());
        // Each transition-matrix row should sum to ~1 where defined.
        for mat in &e.probs {
            for row in mat {
                let s: f64 = row.iter().sum();
                assert!((s - 1.0).abs() < 1e-9 || s == 0.0);
            }
        }
    }
}
