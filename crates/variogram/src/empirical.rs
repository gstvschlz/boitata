//! Empirical (experimental) variogram estimation.
//!
//! Supports the Matheron classical estimator and the Cressie–Hawkins robust
//! estimator, omnidirectional or directional (azimuth/dip cone with tolerance).

use crate::error::{Result, VarioError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Empirical-variogram estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Estimator {
    /// Matheron classical: `γ(h) = 1/(2N) Σ (zᵢ − zⱼ)²`.
    Matheron,
    /// Cressie–Hawkins robust estimator (down-weights outliers).
    CressieHawkins,
}

/// Optional directional constraint (a cone about a unit direction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Direction {
    /// Azimuth in degrees (from North, clockwise).
    pub azimuth: f64,
    /// Dip in degrees (positive downward).
    pub dip: f64,
    /// Angular half-width tolerance in degrees.
    pub tolerance: f64,
    /// Optional lateral bandwidth (meters). Pairs beyond this offset are rejected.
    pub bandwidth: Option<f64>,
}

impl Direction {
    fn unit(&self) -> (f64, f64, f64) {
        let az = self.azimuth.to_radians();
        let dip = self.dip.to_radians();
        // East, North, Up. Dip is positive downward.
        let cosd = dip.cos();
        (cosd * az.sin(), cosd * az.cos(), -dip.sin())
    }
}

/// Lag-binning parameters.
#[derive(Debug, Clone)]
pub struct LagBins {
    pub max_lag: f64,
    pub lag_width: f64,
}

impl Default for LagBins {
    fn default() -> Self {
        Self {
            max_lag: 1000.0,
            lag_width: 50.0,
        }
    }
}

/// Experimental variogram: lag centers, semivariances, and pair counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experimental {
    pub lags: Vec<f64>,
    pub gammas: Vec<f64>,
    pub counts: Vec<usize>,
}

/// Estimate an experimental variogram.
///
/// `O(n²)` over sample pairs; for large `n` this is the dominant cost.
pub fn experimental(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    bins: &LagBins,
    estimator: Estimator,
    direction: Option<&Direction>,
) -> Result<Experimental> {
    if locations.len() != values.len() {
        return Err(VarioError::InsufficientData(
            "locations and values length mismatch".into(),
        ));
    }
    if locations.len() < 2 {
        return Err(VarioError::InsufficientData(
            "at least 2 samples required".into(),
        ));
    }
    if bins.lag_width <= 0.0 || bins.max_lag <= 0.0 {
        return Err(VarioError::InvalidParameters(
            "lag_width and max_lag must be positive".into(),
        ));
    }

    let n_bins = (bins.max_lag / bins.lag_width).ceil() as usize;
    let n_bins = n_bins.max(1);

    let dir = direction.map(|d| (d, d.unit()));
    let cos_tol = direction.map(|d| d.tolerance.to_radians().cos());

    let n = locations.len();

    // The O(n²) pair sweep runs over a *fixed* number of chunks, never one sized
    // by the machine's core count: floating-point addition is not associative, so
    // the summation order must be identical on any machine, at any thread count.
    // Each chunk accumulates its own bins in a fixed order and the partials are
    // merged sequentially in chunk-index order below.
    //
    // Row `i` does `n - i - 1` inner iterations, so contiguous chunks of `i`
    // would carry wildly unequal work. Chunk `c` instead takes the strided rows
    // `c, c + N_CHUNKS, c + 2·N_CHUNKS, …`, giving every chunk a near-equal mix
    // of long and short rows; N_CHUNKS is comfortably above any plausible core
    // count so the pool stays fed even where the balance is imperfect.
    const N_CHUNKS: usize = 64;

    // Per-chunk (sum_sq, sum_root, counts): Matheron sum of squared diffs and
    // Cressie sum of |diff|^0.5 — both accumulated, so the estimator choice
    // stays in the finalisation loop.
    let partials: Vec<(Vec<f64>, Vec<f64>, Vec<usize>)> = (0..N_CHUNKS)
        .into_par_iter()
        .map(|chunk| {
            let mut sum_sq = vec![0.0f64; n_bins];
            let mut sum_root = vec![0.0f64; n_bins];
            let mut counts = vec![0usize; n_bins];
            for i in (chunk..n).step_by(N_CHUNKS) {
                for j in (i + 1)..n {
                    let pi = locations[i];
                    let pj = locations[j];
                    let dx = pj.0 - pi.0;
                    let dy = pj.1 - pi.1;
                    let dz = pj.2 - pi.2;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist == 0.0 || dist > bins.max_lag {
                        continue;
                    }

                    if let (Some((d, u)), Some(ct)) = (&dir, cos_tol) {
                        let inv = 1.0 / dist;
                        let proj = (dx * u.0 + dy * u.1 + dz * u.2) * inv; // cos(angle)
                        if proj.abs() < ct {
                            continue;
                        }
                        if let Some(bw) = d.bandwidth {
                            // Perpendicular offset from the direction line.
                            let along = dx * u.0 + dy * u.1 + dz * u.2;
                            let perp2 = (dx * dx + dy * dy + dz * dz) - along * along;
                            if perp2.max(0.0).sqrt() > bw {
                                continue;
                            }
                        }
                    }

                    let idx = ((dist / bins.lag_width).floor() as usize).min(n_bins - 1);
                    let diff = (values[i] - values[j]).abs();
                    sum_sq[idx] += diff * diff;
                    sum_root[idx] += diff.sqrt();
                    counts[idx] += 1;
                }
            }
            (sum_sq, sum_root, counts)
        })
        .collect();

    // Sequential merge in chunk-index order — the deterministic fold.
    let mut sum_sq = vec![0.0f64; n_bins];
    let mut sum_root = vec![0.0f64; n_bins];
    let mut counts = vec![0usize; n_bins];
    for (chunk_sq, chunk_root, chunk_counts) in &partials {
        for b in 0..n_bins {
            sum_sq[b] += chunk_sq[b];
            sum_root[b] += chunk_root[b];
            counts[b] += chunk_counts[b];
        }
    }

    let mut lags = Vec::new();
    let mut gammas = Vec::new();
    let mut out_counts = Vec::new();

    for b in 0..n_bins {
        let c = counts[b];
        if c == 0 {
            continue;
        }
        let center = (b as f64 + 0.5) * bins.lag_width;
        let gamma = match estimator {
            Estimator::Matheron => sum_sq[b] / (2.0 * c as f64),
            Estimator::CressieHawkins => {
                // γ(h) = 0.5 · (1/N Σ |Δz|^0.5)^4 / (0.457 + 0.494/N + 0.045/N²)
                let nf = c as f64;
                let mean_root = sum_root[b] / nf;
                let numer = mean_root.powi(4);
                let denom = 0.457 + 0.494 / nf + 0.045 / (nf * nf);
                0.5 * numer / denom
            }
        };
        lags.push(center);
        gammas.push(gamma);
        out_counts.push(c);
    }

    Ok(Experimental {
        lags,
        gammas,
        counts: out_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matheron_linear_trend() {
        // Points on a line with linear values; short lags → small γ, longer → larger.
        let locs: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64 * 10.0, 0.0, 0.0)).collect();
        let vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let bins = LagBins {
            max_lag: 100.0,
            lag_width: 10.0,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::Matheron, None).unwrap();
        assert!(!e.lags.is_empty());
        assert_eq!(e.lags.len(), e.gammas.len());
        // Monotone increasing for a linear field.
        for w in e.gammas.windows(2) {
            assert!(w[1] >= w[0] - 1e-9);
        }
    }

    #[test]
    fn robust_estimator_runs() {
        let locs: Vec<(f64, f64, f64)> = (0..8).map(|i| (i as f64, 0.0, 0.0)).collect();
        let vals = vec![1.0, 2.0, 1.5, 3.0, 2.5, 10.0, 2.0, 2.2]; // one outlier
        let bins = LagBins {
            max_lag: 8.0,
            lag_width: 1.0,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::CressieHawkins, None).unwrap();
        assert!(!e.gammas.is_empty());
        assert!(e.gammas.iter().all(|g| g.is_finite() && *g >= 0.0));
    }

    #[test]
    fn directional_filters_pairs() {
        // Cross pattern; East-West direction should only pick East-West pairs.
        let locs = vec![
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 20.0, 0.0),
        ];
        let vals = vec![1.0, 2.0, 5.0, 9.0];
        let bins = LagBins {
            max_lag: 30.0,
            lag_width: 10.0,
        };
        let dir = Direction {
            azimuth: 90.0, // East
            dip: 0.0,
            tolerance: 10.0,
            bandwidth: None,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::Matheron, Some(&dir)).unwrap();
        // Only the East-West pair (0,0,0)-(10,0,0) qualifies.
        let total: usize = e.counts.iter().sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn pair_counts_match_serial_expectation() {
        // 10 collinear points spaced 10 apart. A pair at distance 10k bins into
        // lag index k (a boundary distance rounds up), and there are 10 − k such
        // pairs; bin 0 stays empty, so the serial sweep yields exactly 9..=1.
        let locs: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64 * 10.0, 0.0, 0.0)).collect();
        let vals: Vec<f64> = (0..10).map(|i| (i as f64 * 0.7).sin()).collect();
        let bins = LagBins {
            max_lag: 100.0,
            lag_width: 10.0,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::Matheron, None).unwrap();
        assert_eq!(e.counts, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(
            e.lags,
            vec![15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
        );
    }

    #[test]
    fn deterministic_across_thread_counts() {
        // The audited property: bit-identical output on any thread count. Run the
        // same estimation inside pools of several sizes and require exact equality
        // for both estimators, omnidirectional and directional.
        let n = 300;
        let locs: Vec<(f64, f64, f64)> = (0..n)
            .map(|i| {
                let t = i as f64;
                ((t * 7.3) % 97.0, (t * 3.1) % 83.0, (t * 1.7) % 13.0)
            })
            .collect();
        let vals: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.37).sin() * 10.0 + (i as f64 * 0.11).cos() * 3.0)
            .collect();
        let bins = LagBins {
            max_lag: 80.0,
            lag_width: 5.0,
        };
        let dir = Direction {
            azimuth: 45.0,
            dip: 10.0,
            tolerance: 25.0,
            bandwidth: Some(30.0),
        };

        for estimator in [Estimator::Matheron, Estimator::CressieHawkins] {
            for direction in [None, Some(&dir)] {
                let reference = experimental(&locs, &vals, &bins, estimator, direction).unwrap();
                for k in [1usize, 2, 3, 7] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(k)
                        .build()
                        .unwrap();
                    let e = pool
                        .install(|| experimental(&locs, &vals, &bins, estimator, direction))
                        .unwrap();
                    assert_eq!(e.lags, reference.lags, "lags differ at {k} threads");
                    assert_eq!(e.gammas, reference.gammas, "gammas differ at {k} threads");
                    assert_eq!(e.counts, reference.counts, "counts differ at {k} threads");
                }
            }
        }
    }
}
