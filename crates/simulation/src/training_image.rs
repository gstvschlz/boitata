//! Training images: a column of a block model that multiple-point simulation
//! copies patterns from.
//!
//! A categorical image holds integer codes 0 to k - 1, a continuous one float
//! values. Null cells, and the cells a masked model leaves out, never serve as
//! a pattern source. Patterns are read in cell index space (x fastest, then y,
//! then z), so the image's own origin, cell size and rotation play no part:
//! a rotated training image gives the same patterns as the unrotated one.

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use ceres_core::BlockModel;

use crate::error::{Result, SimError};
use crate::lattice::Lattice;

/// Code of a cell without data in a categorical image.
pub const NO_CODE: u8 = u8::MAX;

/// The cells of a training image, one per grid cell, x fastest.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingValues {
    /// Codes 0 to 254; [`NO_CODE`] where there is no data.
    Categorical(Vec<u8>),
    /// Values; NaN where there is no data.
    Continuous(Vec<f32>),
}

/// A training image in dense form, with the summaries simulation needs.
///
/// A position is the index of a cell, x fastest: `i + nx * (j + ny * k)`.
#[derive(Debug, Clone)]
pub struct TrainingImage {
    dims: [usize; 3],
    values: TrainingValues,
    /// Share of the valid cells in each code; empty for a continuous image.
    proportions: Vec<f64>,
    /// Smallest and largest valid value; `[0, 0]` for a categorical image.
    range: [f64; 2],
    valid: Vec<u32>,
}

impl TrainingImage {
    /// Categorical image from `column` of `model`: integer codes 0 to 254,
    /// stored as floats. The image has as many categories as its largest
    /// code plus one; codes it lacks have proportion 0.
    pub fn categorical(model: &BlockModel, column: &str) -> Result<Self> {
        let (dims, cells) = dense(model, column)?;
        let mut codes = vec![NO_CODE; cells.len()];
        for (code, value) in codes.iter_mut().zip(&cells) {
            let Some(v) = *value else { continue };
            if !(v.fract() == 0.0 && (0.0..f64::from(NO_CODE)).contains(&v)) {
                return Err(SimError::InvalidParameters(format!(
                    "training image column {column:?} holds {v}; categorical codes are integers 0 to {}",
                    NO_CODE - 1
                )));
            }
            *code = v as u8;
        }
        let valid = valid_positions(&cells, column)?;
        let k = codes
            .iter()
            .filter(|&&c| c != NO_CODE)
            .max()
            .map_or(0, |&c| usize::from(c) + 1);
        let mut proportions = vec![0.0; k];
        for &p in &valid {
            proportions[usize::from(codes[p as usize])] += 1.0;
        }
        for p in &mut proportions {
            *p /= valid.len() as f64;
        }
        Ok(Self {
            dims,
            values: TrainingValues::Categorical(codes),
            proportions,
            range: [0.0; 2],
            valid,
        })
    }

    /// Continuous image from `column` of `model`, stored as `f32`.
    pub fn continuous(model: &BlockModel, column: &str) -> Result<Self> {
        let (dims, cells) = dense(model, column)?;
        let mut range = [f64::INFINITY, f64::NEG_INFINITY];
        let mut values = vec![f32::NAN; cells.len()];
        for (value, cell) in values.iter_mut().zip(&cells) {
            let Some(v) = *cell else { continue };
            if !(v.is_finite() && (v as f32).is_finite()) {
                return Err(SimError::InvalidParameters(format!(
                    "training image column {column:?} holds {v}; values must be finite"
                )));
            }
            *value = v as f32;
            range = [
                range[0].min(f64::from(*value)),
                range[1].max(f64::from(*value)),
            ];
        }
        let valid = valid_positions(&cells, column)?;
        Ok(Self {
            dims,
            values: TrainingValues::Continuous(values),
            proportions: Vec::new(),
            range,
            valid,
        })
    }

    /// Number of cells along x, y and z.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn values(&self) -> &TrainingValues {
        &self.values
    }

    /// The codes of a categorical image.
    pub fn codes(&self) -> Option<&[u8]> {
        match &self.values {
            TrainingValues::Categorical(c) => Some(c),
            TrainingValues::Continuous(_) => None,
        }
    }

    /// The values of a continuous image.
    pub fn continuous_values(&self) -> Option<&[f32]> {
        match &self.values {
            TrainingValues::Continuous(v) => Some(v),
            TrainingValues::Categorical(_) => None,
        }
    }

    pub fn is_categorical(&self) -> bool {
        matches!(self.values, TrainingValues::Categorical(_))
    }

    /// Number of categories; 0 for a continuous image.
    pub fn n_categories(&self) -> usize {
        self.proportions.len()
    }

    /// Share of the valid cells in each code; empty for a continuous image.
    pub fn proportions(&self) -> &[f64] {
        &self.proportions
    }

    /// Smallest and largest valid value of a continuous image (of all the
    /// images after [`unify`]); `[0, 0]` for a categorical one.
    pub fn range(&self) -> [f64; 2] {
        self.range
    }

    /// Largest minus smallest value; 0 for a categorical image.
    pub fn value_range(&self) -> f64 {
        self.range[1] - self.range[0]
    }

    /// Positions of the cells with data, ascending.
    pub fn valid_positions(&self) -> &[u32] {
        &self.valid
    }

    /// Categorical image of the classes `cutoffs` cut a continuous image
    /// into: class `c` holds the values `v` with `c` cutoffs `<= v`. The
    /// image has `cutoffs.len() + 1` categories, some possibly empty.
    pub fn classes(&self, cutoffs: &[f64]) -> Result<Self> {
        let Some(values) = self.continuous_values() else {
            return Err(SimError::InvalidParameters(
                "only a continuous training image is cut into classes".into(),
            ));
        };
        if cutoffs.len() >= usize::from(NO_CODE) {
            return Err(SimError::InvalidParameters(format!(
                "{} cutoffs make more than {} classes",
                cutoffs.len(),
                NO_CODE
            )));
        }
        let k = cutoffs.len() + 1;
        let mut codes = vec![NO_CODE; values.len()];
        let mut proportions = vec![0.0; k];
        for &p in &self.valid {
            let c = class_of(cutoffs, f64::from(values[p as usize]));
            codes[p as usize] = c as u8;
            proportions[c] += 1.0 / self.valid.len() as f64;
        }
        Ok(Self {
            dims: self.dims,
            values: TrainingValues::Categorical(codes),
            proportions,
            range: [0.0; 2],
            valid: self.valid.clone(),
        })
    }
}

/// The class of `v` among classes cut by ascending `cutoffs`: how many are
/// at or below it.
pub fn class_of(cutoffs: &[f64], v: f64) -> usize {
    cutoffs.partition_point(|&t| t <= v)
}

/// Makes several training images comparable: categorical images share one
/// code set (as many categories as the largest code of any of them plus one,
/// proportion 0 where an image lacks a code), continuous images one range,
/// from the smallest to the largest value of all of them.
pub fn unify(images: &mut [TrainingImage]) -> Result<()> {
    let Some(first) = images.first() else {
        return Err(SimError::InvalidParameters(
            "give at least one training image".into(),
        ));
    };
    let categorical = first.is_categorical();
    if let Some(i) = images
        .iter()
        .position(|t| t.is_categorical() != categorical)
    {
        return Err(SimError::InvalidParameters(format!(
            "training images 0 and {i} differ in kind; give all categorical or all continuous images"
        )));
    }
    if categorical {
        let k = images.iter().map(TrainingImage::n_categories).max();
        for image in images.iter_mut() {
            image.proportions.resize(k.unwrap_or(0), 0.0);
        }
    } else {
        let lo = images
            .iter()
            .map(|t| t.range[0])
            .fold(f64::INFINITY, f64::min);
        let hi = images
            .iter()
            .map(|t| t.range[1])
            .fold(f64::NEG_INFINITY, f64::max);
        for image in images.iter_mut() {
            image.range = [lo, hi];
        }
    }
    Ok(())
}

/// The dims of `model` and `column` at every grid cell; `None` where null or
/// masked out.
fn dense(model: &BlockModel, column: &str) -> Result<([usize; 3], Vec<Option<f64>>)> {
    let lattice = Lattice::from_model(model).ok_or_else(|| {
        SimError::InvalidParameters(
            "a training image cannot be sub-blocked; regularize it first".into(),
        )
    })?;
    let geometry = lattice.geometry();
    if geometry.cells() > u64::from(u32::MAX) {
        return Err(SimError::InvalidParameters(format!(
            "a training image holds at most {} cells, got {}",
            u32::MAX,
            geometry.cells()
        )));
    }
    let array = model
        .attributes()
        .column_by_name(column)
        .ok_or_else(|| SimError::InvalidParameters(format!("no column {column:?}")))?;
    let rows: Vec<Option<f64>> = if let Some(c) = array.as_primitive_opt::<Float64Type>() {
        c.iter().collect()
    } else if let Some(c) = array.as_primitive_opt::<Float32Type>() {
        c.iter().map(|v| v.map(f64::from)).collect()
    } else {
        return Err(SimError::InvalidParameters(format!(
            "training image column {column:?} must be float"
        )));
    };
    let mut cells = vec![None; geometry.cells() as usize];
    for (node, value) in rows.into_iter().enumerate() {
        cells[lattice.cell(node) as usize] = value;
    }
    Ok((geometry.count, cells))
}

fn valid_positions(cells: &[Option<f64>], column: &str) -> Result<Vec<u32>> {
    let valid: Vec<u32> = (0..cells.len() as u32)
        .filter(|&p| cells[p as usize].is_some())
        .collect();
    if valid.is_empty() {
        return Err(SimError::InsufficientData(format!(
            "training image column {column:?} has no data"
        )));
    }
    Ok(valid)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float32Array, Float64Array, RecordBatch};
    use ceres_core::Geometry;

    use super::*;

    fn geometry(count: [usize; 3]) -> Geometry {
        Geometry {
            origin: [0.0; 3],
            size: [1.0; 3],
            count,
            rotation: [30.0, 0.0, 0.0],
        }
    }

    fn model(count: [usize; 3], values: Vec<Option<f64>>) -> BlockModel {
        let column = Arc::new(Float64Array::from(values));
        let batch = RecordBatch::try_from_iter([("v", column as _)]).unwrap();
        BlockModel::regular(geometry(count), batch).unwrap()
    }

    #[test]
    fn codes_and_proportions_skip_nulls() {
        let ti = TrainingImage::categorical(
            &model(
                [3, 2, 1],
                vec![Some(2.0), Some(0.0), None, Some(2.0), Some(2.0), Some(0.0)],
            ),
            "v",
        )
        .unwrap();
        assert_eq!(ti.dims(), [3, 2, 1]);
        assert_eq!(ti.codes().unwrap(), &[2, 0, NO_CODE, 2, 2, 0]);
        assert_eq!(ti.valid_positions(), &[0, 1, 3, 4, 5]);
        assert_eq!(ti.proportions(), &[0.4, 0.0, 0.6]);
        assert_eq!(ti.n_categories(), 3);
    }

    #[test]
    fn masked_cells_are_no_data() {
        let column = Arc::new(Float32Array::from(vec![1.5, -0.5, 4.0]));
        let batch = RecordBatch::try_from_iter([("v", column as _)]).unwrap();
        let masked = BlockModel::masked(geometry([2, 2, 1]), vec![0, 2, 3], batch).unwrap();
        let ti = TrainingImage::continuous(&masked, "v").unwrap();
        assert_eq!(ti.valid_positions(), &[0, 2, 3]);
        assert!(ti.continuous_values().unwrap()[1].is_nan());
        assert_eq!(ti.range(), [-0.5, 4.0]);
        assert_eq!(ti.value_range(), 4.5);
    }

    #[test]
    fn two_images_share_codes_or_range() {
        let a = model([2, 2, 1], vec![Some(1.0), Some(0.0), Some(1.0), None]);
        let b = model([4, 1, 1], vec![Some(3.0), Some(3.0), Some(1.0), Some(3.0)]);
        let mut images = [
            TrainingImage::categorical(&a, "v").unwrap(),
            TrainingImage::categorical(&b, "v").unwrap(),
        ];
        unify(&mut images).unwrap();
        assert_eq!(images[0].proportions(), &[1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0]);
        assert_eq!(images[1].proportions(), &[0.0, 0.25, 0.0, 0.75]);

        let mut images = [
            TrainingImage::continuous(&a, "v").unwrap(),
            TrainingImage::continuous(&b, "v").unwrap(),
        ];
        unify(&mut images).unwrap();
        assert!(images.iter().all(|t| t.range() == [0.0, 3.0]));

        let mut mixed = [
            TrainingImage::continuous(&a, "v").unwrap(),
            TrainingImage::categorical(&b, "v").unwrap(),
        ];
        assert!(unify(&mut mixed).is_err());
        assert!(unify(&mut []).is_err());
    }

    #[test]
    fn rejects_bad_codes_empty_images_and_sub_blocks() {
        let m = model([2, 1, 1], vec![Some(0.5), Some(1.0)]);
        assert!(TrainingImage::categorical(&m, "v").is_err());
        assert!(TrainingImage::continuous(&m, "v").is_ok());
        let m = model([2, 1, 1], vec![Some(255.0), Some(1.0)]);
        assert!(TrainingImage::categorical(&m, "v").is_err());
        let m = model([2, 1, 1], vec![None, None]);
        assert!(TrainingImage::continuous(&m, "v").is_err());
        assert!(TrainingImage::continuous(&m, "w").is_err());
        let batch =
            RecordBatch::try_from_iter([("v", Arc::new(Float64Array::from(vec![1.0])) as _)])
                .unwrap();
        let sub = BlockModel::subblocked(
            geometry([2, 1, 1]),
            vec![0],
            vec![[0.0, 0.0, 0.0, 0.5, 1.0, 1.0]],
            None,
            batch,
        )
        .unwrap();
        assert!(TrainingImage::categorical(&sub, "v").is_err());
    }
}
