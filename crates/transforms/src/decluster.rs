//! Declustering.
//!
//! Clustered samples (e.g. infill drilling in high-grade zones) bias global statistics.
//! Cell declustering overlays a regular grid and weights each sample inversely to the
//! number of samples sharing its cell; polygon (nearest-neighbour) declustering weights
//! each sample by its discrete area/volume of influence. Either de-emphasizes
//! densely-sampled areas.

use crate::error::{Result, TransformError};
use std::collections::HashMap;

/// Declustering weights (normalized to sum to `n`, so the mean weight is 1).
#[derive(Debug, Clone)]
pub struct Weights {
    pub weights: Vec<f64>,
    /// Declustered weighted mean of the input values.
    pub declustered_mean: f64,
    pub cell_size: f64,
}

/// Cell-declustering weights for a given cubic `cell_size`.
///
/// An origin offset can be supplied to reduce sensitivity to grid placement (callers
/// typically average several offsets; see [`decluster_mean_over_offsets`]).
pub fn cell_weights(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    cell_size: f64,
    origin: (f64, f64, f64),
) -> Result<Weights> {
    let n = locations.len();
    if n == 0 {
        return Err(TransformError::InsufficientData("no samples".into()));
    }
    if locations.len() != values.len() {
        return Err(TransformError::InvalidParameters("length mismatch".into()));
    }
    if cell_size <= 0.0 {
        return Err(TransformError::InvalidParameters(
            "cell_size must be > 0".into(),
        ));
    }

    let cell_of = |p: &(f64, f64, f64)| -> (i64, i64, i64) {
        (
            ((p.0 - origin.0) / cell_size).floor() as i64,
            ((p.1 - origin.1) / cell_size).floor() as i64,
            ((p.2 - origin.2) / cell_size).floor() as i64,
        )
    };
    let mut counts: HashMap<(i64, i64, i64), usize> = HashMap::new();
    for p in locations {
        *counts.entry(cell_of(p)).or_insert(0) += 1;
    }
    let n_occupied = counts.len() as f64;

    // Raw weight ∝ 1 / (cell count); normalize so Σw = n.
    let raw: Vec<f64> = locations
        .iter()
        .map(|p| 1.0 / counts[&cell_of(p)] as f64)
        .collect();
    // With this scheme Σ raw = n_occupied. Scale to sum n.
    let scale = n as f64 / n_occupied;
    let weights: Vec<f64> = raw.iter().map(|w| w * scale).collect();

    let wsum: f64 = weights.iter().sum();
    let declustered_mean = weights.iter().zip(values).map(|(w, v)| w * v).sum::<f64>() / wsum;

    Ok(Weights {
        weights,
        declustered_mean,
        cell_size,
    })
}

/// Average declustered mean over several grid-origin offsets (reduces placement bias).
pub fn decluster_mean_over_offsets(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    cell_size: f64,
    n_offsets: usize,
) -> Result<f64> {
    if n_offsets == 0 {
        return Err(TransformError::InvalidParameters(
            "n_offsets must be > 0".into(),
        ));
    }
    let mut acc = 0.0;
    for k in 0..n_offsets {
        let frac = k as f64 / n_offsets as f64;
        let off = (frac * cell_size, frac * cell_size, frac * cell_size);
        acc += cell_weights(locations, values, cell_size, off)?.declustered_mean;
    }
    Ok(acc / n_offsets as f64)
}

/// Sweep a set of cell sizes and return the one whose declustered mean is minimum
/// (the standard heuristic when clustering targets high grades) along with the mean.
///
/// Set `maximize` for the opposite case (clustering in low-grade areas).
pub fn optimal_cell_size(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    candidate_sizes: &[f64],
    n_offsets: usize,
    maximize: bool,
) -> Result<(f64, f64)> {
    if candidate_sizes.is_empty() {
        return Err(TransformError::InvalidParameters(
            "no candidate sizes".into(),
        ));
    }
    let mut best_size = candidate_sizes[0];
    let mut best_mean = if maximize { f64::MIN } else { f64::MAX };
    for &cs in candidate_sizes {
        let m = decluster_mean_over_offsets(locations, values, cs, n_offsets)?;
        let better = if maximize {
            m > best_mean
        } else {
            m < best_mean
        };
        if better {
            best_mean = m;
            best_size = cs;
        }
    }
    Ok((best_size, best_mean))
}

/// Polygon (nearest-neighbour / area-of-influence) declustering weights.
///
/// Discretizes the sample bounding box into ~`target_nodes` cells, assigns each
/// cell to its nearest sample, and weights each sample by how many cells it owns
/// (a discrete Voronoi volume). Weights are normalized so `Σw = n`. `O(nodes·n)`.
pub fn polygon_weights(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    target_nodes: usize,
) -> Result<Weights> {
    let n = locations.len();
    if n == 0 {
        return Err(TransformError::InsufficientData("no samples".into()));
    }
    if locations.len() != values.len() {
        return Err(TransformError::InvalidParameters("length mismatch".into()));
    }
    let target_nodes = target_nodes.max(1);

    // Bounding box + per-axis extents.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in locations {
        let c = [p.0, p.1, p.2];
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    let extent = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let eps = 1e-9;
    let positive: Vec<usize> = (0..3).filter(|&a| extent[a] > eps).collect();

    // All samples coincident (or a single sample): equal weights.
    if positive.is_empty() {
        let weights = vec![1.0; n];
        let declustered_mean = values.iter().sum::<f64>() / n as f64;
        return Ok(Weights {
            weights,
            declustered_mean,
            cell_size: 0.0,
        });
    }

    // Common cell size so the positive-extent axes yield ~target_nodes cells.
    let volume: f64 = positive.iter().map(|&a| extent[a]).product();
    let s = (volume / target_nodes as f64).powf(1.0 / positive.len() as f64);
    let div = [0, 1, 2].map(|a| {
        if extent[a] > eps {
            (extent[a] / s).ceil().max(1.0) as usize
        } else {
            1
        }
    });
    let step = [0, 1, 2].map(|a| {
        if div[a] > 0 {
            extent[a] / div[a] as f64
        } else {
            0.0
        }
    });

    let mut counts = vec![0usize; n];
    let mut total = 0usize;
    for iz in 0..div[2] {
        let z = lo[2] + (iz as f64 + 0.5) * step[2];
        for iy in 0..div[1] {
            let y = lo[1] + (iy as f64 + 0.5) * step[1];
            for ix in 0..div[0] {
                let x = lo[0] + (ix as f64 + 0.5) * step[0];
                // Nearest sample to this node.
                let mut best = 0usize;
                let mut best_d = f64::INFINITY;
                for (i, p) in locations.iter().enumerate() {
                    let d = (p.0 - x).powi(2) + (p.1 - y).powi(2) + (p.2 - z).powi(2);
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                counts[best] += 1;
                total += 1;
            }
        }
    }

    // weight ∝ owned cells, normalized to sum n.
    let scale = n as f64 / total as f64;
    let weights: Vec<f64> = counts.iter().map(|&c| c as f64 * scale).collect();
    let wsum: f64 = weights.iter().sum();
    let declustered_mean = weights.iter().zip(values).map(|(w, v)| w * v).sum::<f64>() / wsum;

    Ok(Weights {
        weights,
        declustered_mean,
        cell_size: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clustered_high_grade_is_down_weighted() {
        // Three clustered high-grade samples + one isolated low-grade sample.
        let locs = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let vals = vec![10.0, 10.0, 10.0, 1.0];
        // Naive mean = 7.75; declustering should pull it toward the isolated low value.
        let naive = vals.iter().sum::<f64>() / 4.0;
        let w = cell_weights(&locs, &vals, 10.0, (0.0, 0.0, 0.0)).unwrap();
        assert!(
            w.declustered_mean < naive,
            "declustered {} naive {}",
            w.declustered_mean,
            naive
        );
        // Isolated sample gets a larger weight than each clustered one.
        assert!(w.weights[3] > w.weights[0]);
    }

    #[test]
    fn weights_sum_to_n() {
        let locs = vec![(0.0, 0.0, 0.0), (50.0, 0.0, 0.0), (0.0, 50.0, 0.0)];
        let vals = vec![1.0, 2.0, 3.0];
        let w = cell_weights(&locs, &vals, 20.0, (0.0, 0.0, 0.0)).unwrap();
        let s: f64 = w.weights.iter().sum();
        assert!((s - 3.0).abs() < 1e-9);
    }

    #[test]
    fn polygon_down_weights_clustered_highs() {
        // Three clustered high-grade + one isolated low-grade sample. The isolated
        // sample owns a large area of influence, so it dominates the declustered mean.
        let locs = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let vals = vec![10.0, 10.0, 10.0, 1.0];
        let naive = vals.iter().sum::<f64>() / 4.0;
        let w = polygon_weights(&locs, &vals, 10_000).unwrap();
        assert!(
            w.declustered_mean < naive,
            "declustered {}",
            w.declustered_mean
        );
        assert!(w.weights[3] > w.weights[0]); // isolated owns more area
        let s: f64 = w.weights.iter().sum();
        assert!((s - 4.0).abs() < 1e-9); // weights sum to n
    }

    #[test]
    fn polygon_coincident_samples_are_equal_weight() {
        let locs = vec![(5.0, 5.0, 5.0), (5.0, 5.0, 5.0)];
        let vals = vec![2.0, 4.0];
        let w = polygon_weights(&locs, &vals, 1000).unwrap();
        assert_eq!(w.weights, vec![1.0, 1.0]);
        assert!((w.declustered_mean - 3.0).abs() < 1e-9);
    }

    #[test]
    fn optimal_cell_size_runs() {
        let locs = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let vals = vec![10.0, 10.0, 10.0, 1.0];
        let (size, mean) =
            optimal_cell_size(&locs, &vals, &[5.0, 20.0, 50.0, 100.0], 3, false).unwrap();
        assert!(size > 0.0);
        assert!(mean.is_finite());
    }
}
