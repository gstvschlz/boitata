//! Localization of panel recoveries onto the selective blocks they hold.
//!
//! A panel with `n` selective blocks splits its block-grade distribution into
//! `n` equal-probability bands; the block ranked `i` (ascending) gets the mean
//! of band `i`. The blocks then average to the panel grade, and the top `k`
//! reproduce the panel's recovery at tonnage `k/n`. Each source (uniform
//! conditioning, indicator kriging, realizations) supplies the band means.

use boitata_core::{BlockModel, Layout, block_frame};
use nalgebra::Vector3;
use rayon::prelude::*;

use crate::error::{Result, TransformError};

fn invalid(message: impl Into<String>) -> TransformError {
    TransformError::InvalidParameters(message.into())
}

fn whole(x: f64) -> Option<f64> {
    let r = x.round();
    ((x - r).abs() <= 1e-6 * r.abs().max(1.0)).then_some(r)
}

/// The panel row holding each block; `None` outside every panel.
///
/// Blocks must nest in the panels: same rotation, cell sizes dividing the
/// panel sizes and a grid aligned on the panel grid. Sub-blocks are refused.
pub fn nest(panels: &BlockModel, blocks: &BlockModel) -> Result<Vec<Option<usize>>> {
    let (p, b) = (panels.geometry(), blocks.geometry());
    if matches!(blocks.layout(), Layout::SubBlocked { .. }) {
        return Err(invalid("selective blocks must not be sub-blocked"));
    }
    if (0..3).any(|a| (p.rotation[a] - b.rotation[a]).abs() > 1e-9) {
        return Err(invalid("selective blocks must share the panels' rotation"));
    }
    let offset = block_frame(p.rotation) * Vector3::from_fn(|a, _| b.origin[a] - p.origin[a]);
    for a in 0..3 {
        let ratio = p.size[a] / b.size[a];
        if !whole(ratio).is_some_and(|r| r >= 1.0) || whole(offset[a] / b.size[a]).is_none() {
            return Err(invalid(
                "selective blocks must nest in the panels: sizes dividing the panel sizes, grids aligned",
            ));
        }
    }
    Ok(blocks
        .centroids()
        .into_iter()
        .map(|c| panels.row_at(c))
        .collect())
}

/// Localized grade of every block.
///
/// `ranking` orders the blocks inside each panel, ties by row order.
/// `bands(panel, n)` gives the `n` ascending band means of a panel, or `None`
/// for an unestimated panel, whose blocks stay `None`. Blocks outside every
/// panel stay `None`; a `None` rank in an estimated panel is an error.
/// Panels run in parallel; the result does not depend on the thread count.
pub fn localize<F>(
    panels: &BlockModel,
    blocks: &BlockModel,
    ranking: &[Option<f64>],
    bands: F,
) -> Result<Vec<Option<f64>>>
where
    F: Fn(usize, usize) -> Result<Option<Vec<f64>>> + Sync,
{
    if ranking.len() != blocks.len() {
        return Err(invalid("ranking needs one value per selective block"));
    }
    if ranking.iter().flatten().any(|r| r.is_nan()) {
        return Err(invalid("ranking must not be NaN"));
    }
    let mut members = vec![vec![]; panels.len()];
    for (row, panel) in nest(panels, blocks)?.into_iter().enumerate() {
        if let Some(p) = panel {
            members[p].push(row);
        }
    }
    let assigned = members
        .into_par_iter()
        .enumerate()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(panel, mut rows)| {
            let Some(means) = bands(panel, rows.len())? else {
                return Ok(vec![]);
            };
            if means.len() != rows.len() {
                return Err(invalid("band means must number the panel's blocks"));
            }
            if rows.iter().any(|&row| ranking[row].is_none()) {
                return Err(invalid("a null rank in an estimated panel"));
            }
            rows.sort_by(|&i, &j| ranking[i].unwrap().total_cmp(&ranking[j].unwrap()));
            Ok(rows.into_iter().zip(means).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out = vec![None; blocks.len()];
    for (row, value) in assigned.into_iter().flatten() {
        out[row] = Some(value);
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use arrow_array::{RecordBatch, RecordBatchOptions};
    use boitata_core::Geometry;

    pub(crate) fn grid(origin: [f64; 3], size: f64, count: usize, rotation: f64) -> BlockModel {
        let g = Geometry {
            origin,
            size: [size, size, 1.0],
            count: [count, count, 1],
            rotation: [rotation, 0.0, 0.0],
        };
        let batch = RecordBatch::try_new_with_options(
            std::sync::Arc::new(arrow_schema::Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(g.cells() as usize)),
        )
        .unwrap();
        BlockModel::regular(g, batch).unwrap()
    }

    #[test]
    fn blocks_find_their_panel_and_ties_keep_row_order() {
        let panels = grid([0.0; 3], 20.0, 2, 30.0);
        let blocks = panels.discretize([2, 2, 1]).unwrap();
        let ranking = vec![Some(1.0); blocks.len()];
        let out = localize(&panels, &blocks, &ranking, |p, n| {
            Ok(Some((0..n).map(|i| (10 * p + i) as f64).collect()))
        })
        .unwrap();
        let panel = nest(&panels, &blocks).unwrap();
        let mut seen = [0; 4];
        for (row, v) in out.iter().enumerate() {
            let p = panel[row].unwrap();
            assert_eq!(v.unwrap(), (10 * p + seen[p]) as f64);
            seen[p] += 1;
        }
    }

    #[test]
    fn misaligned_or_rotated_blocks_are_refused() {
        let panels = grid([0.0; 3], 20.0, 2, 0.0);
        let unit =
            |b: BlockModel| localize(&panels, &b, &vec![Some(0.0); b.len()], |_, _| Ok(None));
        assert!(unit(grid([5.0, 0.0, 0.0], 10.0, 4, 0.0)).is_err());
        assert!(unit(grid([0.0; 3], 10.0, 4, 10.0)).is_err());
        assert!(unit(grid([0.0; 3], 15.0, 2, 0.0)).is_err());
        let shifted = unit(grid([-10.0, 0.0, 0.0], 10.0, 6, 0.0)).unwrap();
        assert!(shifted.iter().all(Option::is_none));
    }

    #[test]
    fn null_panels_leave_nulls_and_null_ranks_fail() {
        let panels = grid([0.0; 3], 20.0, 2, 0.0);
        let blocks = panels.discretize([2, 2, 1]).unwrap();
        let mut ranking: Vec<Option<f64>> = (0..blocks.len()).map(|i| Some(i as f64)).collect();
        let bands = |p: usize, n: usize| Ok((p != 0).then(|| vec![1.0; n]));
        let out = localize(&panels, &blocks, &ranking, bands).unwrap();
        let panel = nest(&panels, &blocks).unwrap();
        for (v, p) in out.iter().zip(&panel) {
            assert_eq!(v.is_none(), *p == Some(0));
        }
        let first_of = |q: usize| panel.iter().position(|p| *p == Some(q)).unwrap();
        ranking[first_of(0)] = None;
        assert!(localize(&panels, &blocks, &ranking, bands).is_ok());
        ranking[first_of(1)] = None;
        assert!(localize(&panels, &blocks, &ranking, bands).is_err());
    }
}
