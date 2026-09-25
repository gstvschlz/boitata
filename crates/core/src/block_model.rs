use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow_schema::Schema;
use arrow_select::filter::filter_record_batch;
use arrow_select::take::take;
use nalgebra::Vector3;

use crate::{Error, Result, block_frame, check_rows};

/// Parent grid of a block model. Cell index runs x fastest, then y, then z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    /// Corner of cell (0, 0, 0); the grid rotates about it.
    pub origin: [f64; 3],
    pub size: [f64; 3],
    pub count: [usize; 3],
    /// Azimuth, dip, rake in degrees; see [`block_frame`].
    pub rotation: [f64; 3],
}

impl Geometry {
    pub fn validate(&self) -> Result<()> {
        if self.size.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(Error::Geometry("cell sizes must be positive".into()));
        }
        if self.count.contains(&0) {
            return Err(Error::Geometry("cell counts must be positive".into()));
        }
        if self
            .origin
            .iter()
            .chain(&self.rotation)
            .any(|v| !v.is_finite())
        {
            return Err(Error::Geometry("origin and rotation must be finite".into()));
        }
        Ok(())
    }

    pub fn cells(&self) -> u64 {
        self.count.iter().map(|&n| n as u64).product()
    }

    pub fn ijk(&self, index: u64) -> [usize; 3] {
        let [nx, ny, _] = self.count.map(|n| n as u64);
        [
            (index % nx) as usize,
            (index / nx % ny) as usize,
            (index / (nx * ny)) as usize,
        ]
    }

    pub fn index(&self, [i, j, k]: [usize; 3]) -> u64 {
        let [nx, ny, _] = self.count.map(|n| n as u64);
        i as u64 + nx * (j as u64 + ny * k as u64)
    }

    pub fn centroid(&self, index: u64) -> [f64; 3] {
        let ijk = self.ijk(index);
        let local = Vector3::from_fn(|a, _| (ijk[a] as f64 + 0.5) * self.size[a]);
        let world = block_frame(self.rotation).transpose() * local;
        [0, 1, 2].map(|a| self.origin[a] + world[a])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Layout {
    /// Every parent cell, one row each, in index order.
    Regular,
    /// Only the listed parent cells; strictly increasing indices.
    Masked(Vec<u64>),
}

/// Regular grid or block model, 2D (`count[2] == 1`) or 3D, optionally rotated.
#[derive(Debug, Clone)]
pub struct BlockModel {
    geometry: Geometry,
    layout: Layout,
    attributes: RecordBatch,
    pub crs: Option<String>,
}

impl BlockModel {
    pub fn regular(geometry: Geometry, attributes: RecordBatch) -> Result<Self> {
        geometry.validate()?;
        check_rows(rows(geometry.cells())?, &attributes)?;
        Ok(Self {
            geometry,
            layout: Layout::Regular,
            attributes,
            crs: None,
        })
    }

    pub fn masked(geometry: Geometry, index: Vec<u64>, attributes: RecordBatch) -> Result<Self> {
        geometry.validate()?;
        if index.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::Geometry(
                "mask index must be strictly increasing".into(),
            ));
        }
        if index.last().is_some_and(|&i| i >= geometry.cells()) {
            return Err(Error::Geometry("mask index outside the grid".into()));
        }
        check_rows(index.len(), &attributes)?;
        Ok(Self {
            geometry,
            layout: Layout::Masked(index),
            attributes,
            crs: None,
        })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn attributes(&self) -> &RecordBatch {
        &self.attributes
    }

    pub fn len(&self) -> usize {
        self.attributes.num_rows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn parent_index(&self, row: usize) -> u64 {
        match &self.layout {
            Layout::Regular => row as u64,
            Layout::Masked(index) => index[row],
        }
    }

    pub fn centroids(&self) -> Vec<[f64; 3]> {
        (0..self.len())
            .map(|row| self.geometry.centroid(self.parent_index(row)))
            .collect()
    }

    /// Keeps the rows where `keep` is true; the result is masked.
    pub fn mask(&self, keep: &BooleanArray) -> Result<Self> {
        check_rows(keep.len(), &self.attributes)?;
        let index = (0..self.len())
            .filter(|&row| keep.is_valid(row) && keep.value(row))
            .map(|row| self.parent_index(row))
            .collect();
        let attributes = filter_record_batch(&self.attributes, keep)?;
        Ok(Self {
            layout: Layout::Masked(index),
            attributes,
            ..self.clone()
        })
    }

    /// Expands to every parent cell; absent cells are null.
    pub fn to_regular(&self) -> Result<Self> {
        let Layout::Masked(index) = &self.layout else {
            return Ok(self.clone());
        };
        let mut rows = index.iter().enumerate().peekable();
        let positions: UInt64Array = (0..self.geometry.cells())
            .map(|cell| {
                rows.next_if(|(_, i)| **i == cell)
                    .map(|(row, _)| row as u64)
            })
            .collect();
        let columns = self
            .attributes
            .columns()
            .iter()
            .map(|c| take(c, &positions, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let schema = self.attributes.schema();
        let fields = schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone().with_nullable(true));
        let options = RecordBatchOptions::new().with_row_count(Some(positions.len()));
        let attributes = RecordBatch::try_new_with_options(
            Arc::new(Schema::new(fields.collect::<Vec<_>>())),
            columns,
            &options,
        )?;
        Ok(Self {
            layout: Layout::Regular,
            attributes,
            ..self.clone()
        })
    }
}

fn rows(cells: u64) -> Result<usize> {
    usize::try_from(cells).map_err(|_| Error::Geometry("too many cells".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float64Type;
    use arrow_array::{ArrayRef, Float64Array};

    fn geometry(rotation: [f64; 3]) -> Geometry {
        Geometry {
            origin: [100.0, 200.0, 0.0],
            size: [10.0, 5.0, 2.0],
            count: [3, 2, 1],
            rotation,
        }
    }

    fn grades(values: Vec<f64>) -> RecordBatch {
        RecordBatch::try_from_iter([("au", Arc::new(Float64Array::from(values)) as ArrayRef)])
            .unwrap()
    }

    #[test]
    fn index_and_ijk_round_trip() {
        let g = geometry([0.0; 3]);
        for cell in 0..g.cells() {
            assert_eq!(g.index(g.ijk(cell)), cell);
        }
        assert_eq!(g.ijk(4), [1, 1, 0]);
    }

    #[test]
    fn centroids_rotate_about_origin() {
        assert_eq!(geometry([0.0; 3]).centroid(4), [115.0, 207.5, 1.0]);
        let c = geometry([90.0, 0.0, 0.0]).centroid(0);
        let expected = [102.5, 195.0, 1.0];
        assert!(c.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-9));
    }

    #[test]
    fn regular_requires_one_row_per_cell() {
        assert!(BlockModel::regular(geometry([0.0; 3]), grades(vec![1.0; 5])).is_err());
        assert!(BlockModel::regular(geometry([0.0; 3]), grades(vec![1.0; 6])).is_ok());
    }

    #[test]
    fn masked_rejects_unsorted_or_outside_index() {
        let g = geometry([0.0; 3]);
        assert!(BlockModel::masked(g, vec![2, 1], grades(vec![1.0, 2.0])).is_err());
        assert!(BlockModel::masked(g, vec![1, 6], grades(vec![1.0, 2.0])).is_err());
    }

    #[test]
    fn mask_then_regular_restores_values_and_nulls() {
        let m = BlockModel::regular(geometry([0.0; 3]), grades((0..6).map(f64::from).collect()))
            .unwrap();
        let keep = BooleanArray::from(vec![true, false, true, false, false, true]);
        let masked = m.mask(&keep).unwrap();
        assert_eq!(masked.layout(), &Layout::Masked(vec![0, 2, 5]));
        assert_eq!(masked.centroids()[1], m.geometry().centroid(2));

        let back = masked.to_regular().unwrap();
        let au = back.attributes().column(0).as_primitive::<Float64Type>();
        let values: Vec<_> = au.iter().collect();
        assert_eq!(values, [Some(0.0), None, Some(2.0), None, None, Some(5.0)]);
    }
}
