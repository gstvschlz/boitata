//! Up/downscaling between supports.
//!
//! `upscale` aggregates point (or fine-block) values into coarser blocks by averaging
//! the points falling in each block. `downscale` replicates a coarse block value onto
//! a finer sub-grid (constant refinement), preserving the mean.

use crate::error::{Result, TransformError};
use std::collections::HashMap;

/// A coarse block produced by upscaling: its center, mean value, and support count.
#[derive(Debug, Clone)]
pub struct CoarseBlock {
    pub center: (f64, f64, f64),
    pub value: f64,
    pub count: usize,
}

/// Aggregate points into a regular block grid of cubic `block_size`, averaging values.
pub fn upscale(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    block_size: (f64, f64, f64),
    origin: (f64, f64, f64),
) -> Result<Vec<CoarseBlock>> {
    if locations.len() != values.len() {
        return Err(TransformError::InvalidParameters("length mismatch".into()));
    }
    if block_size.0 <= 0.0 || block_size.1 <= 0.0 || block_size.2 <= 0.0 {
        return Err(TransformError::InvalidParameters(
            "block_size must be > 0".into(),
        ));
    }

    let mut acc: HashMap<(i64, i64, i64), (f64, usize)> = HashMap::new();
    for (p, v) in locations.iter().zip(values) {
        let key = (
            ((p.0 - origin.0) / block_size.0).floor() as i64,
            ((p.1 - origin.1) / block_size.1).floor() as i64,
            ((p.2 - origin.2) / block_size.2).floor() as i64,
        );
        let e = acc.entry(key).or_insert((0.0, 0));
        e.0 += v;
        e.1 += 1;
    }

    let mut blocks: Vec<CoarseBlock> = acc
        .into_iter()
        .map(|((i, j, k), (sum, count))| CoarseBlock {
            center: (
                origin.0 + (i as f64 + 0.5) * block_size.0,
                origin.1 + (j as f64 + 0.5) * block_size.1,
                origin.2 + (k as f64 + 0.5) * block_size.2,
            ),
            value: sum / count as f64,
            count,
        })
        .collect();
    // Deterministic order.
    blocks.sort_by(|a, b| {
        a.center
            .0
            .partial_cmp(&b.center.0)
            .unwrap()
            .then(a.center.1.partial_cmp(&b.center.1).unwrap())
            .then(a.center.2.partial_cmp(&b.center.2).unwrap())
    });
    Ok(blocks)
}

/// Split a coarse block into `nx×ny×nz` sub-block centers, all carrying `value`.
pub fn downscale(
    center: (f64, f64, f64),
    block_size: (f64, f64, f64),
    value: f64,
    refine: (usize, usize, usize),
) -> Result<Vec<((f64, f64, f64), f64)>> {
    let (nx, ny, nz) = refine;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(TransformError::InvalidParameters(
            "refine must be ≥ 1 per axis".into(),
        ));
    }
    let dx = block_size.0 / nx as f64;
    let dy = block_size.1 / ny as f64;
    let dz = block_size.2 / nz as f64;
    let x0 = center.0 - block_size.0 / 2.0 + dx / 2.0;
    let y0 = center.1 - block_size.1 / 2.0 + dy / 2.0;
    let z0 = center.2 - block_size.2 / 2.0 + dz / 2.0;

    let mut out = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                out.push((
                    (x0 + i as f64 * dx, y0 + j as f64 * dy, z0 + k as f64 * dz),
                    value,
                ));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upscale_averages_within_block() {
        let locs = vec![
            (1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0),
            (12.0, 1.0, 1.0), // different block
        ];
        let vals = vec![10.0, 20.0, 5.0];
        let blocks = upscale(&locs, &vals, (10.0, 10.0, 10.0), (0.0, 0.0, 0.0)).unwrap();
        assert_eq!(blocks.len(), 2);
        // First block averages 10 and 20 → 15.
        let first = blocks.iter().find(|b| b.count == 2).unwrap();
        assert!((first.value - 15.0).abs() < 1e-9);
    }

    #[test]
    fn downscale_preserves_value_and_count() {
        let subs = downscale((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), 7.0, (2, 2, 2)).unwrap();
        assert_eq!(subs.len(), 8);
        assert!(subs.iter().all(|(_, v)| (*v - 7.0).abs() < 1e-12));
        // Centroid of sub-blocks equals the parent center.
        let cx = subs.iter().map(|(p, _)| p.0).sum::<f64>() / 8.0;
        assert!(cx.abs() < 1e-9);
    }
}
