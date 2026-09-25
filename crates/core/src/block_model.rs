use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, RecordBatch, RecordBatchOptions, UInt64Array,
};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};
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
        self.point(index, [0.5; 3])
    }

    /// World location of fractional position `at` (0 to 1 per axis) in a cell.
    pub fn point(&self, index: u64, at: [f64; 3]) -> [f64; 3] {
        let ijk = self.ijk(index);
        let local = Vector3::from_fn(|a, _| (ijk[a] as f64 + at[a]) * self.size[a]);
        let world = block_frame(self.rotation).transpose() * local;
        [0, 1, 2].map(|a| self.origin[a] + world[a])
    }

    pub fn cell_volume(&self) -> f64 {
        self.size.iter().product()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Layout {
    /// Every parent cell, one row each, in index order.
    Regular,
    /// Only the listed parent cells; strictly increasing indices.
    Masked(Vec<u64>),
    /// Sub-blocks: each row's parent cell (non-decreasing) and its extent as
    /// fractions of that cell, `[u0, v0, w0, u1, v1, w1]`. With `grid`, every
    /// corner lies on a regular subdivision of the parent.
    SubBlocked {
        parent: Vec<u64>,
        extent: Vec<[f64; 6]>,
        grid: Option<[u32; 3]>,
    },
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

    /// Sub-blocked model; `grid` (e.g. `[4, 4, 8]`) requires every corner on
    /// that subdivision. Overlaps between sub-blocks are not checked.
    pub fn subblocked(
        geometry: Geometry,
        parent: Vec<u64>,
        extent: Vec<[f64; 6]>,
        grid: Option<[u32; 3]>,
        attributes: RecordBatch,
    ) -> Result<Self> {
        geometry.validate()?;
        if parent.len() != extent.len() {
            return Err(Error::Geometry("one extent per sub-block".into()));
        }
        if parent.windows(2).any(|w| w[0] > w[1]) {
            return Err(Error::Geometry("parent indices must be sorted".into()));
        }
        if parent.last().is_some_and(|&i| i >= geometry.cells()) {
            return Err(Error::Geometry("parent index outside the grid".into()));
        }
        let inside =
            |e: &[f64; 6]| (0..3).all(|a| 0.0 <= e[a] && e[a] < e[a + 3] && e[a + 3] <= 1.0);
        if !extent.iter().all(inside) {
            return Err(Error::Geometry(
                "extents must satisfy 0 <= min < max <= 1".into(),
            ));
        }
        if let Some(n) = grid {
            if n.contains(&0) {
                return Err(Error::Geometry("sub-grid counts must be positive".into()));
            }
            let on_grid = |e: &[f64; 6]| {
                (0..6).all(|i| {
                    let k = e[i] * n[i % 3] as f64;
                    (k - k.round()).abs() < 1e-9
                })
            };
            if !extent.iter().all(on_grid) {
                return Err(Error::Geometry("corners must lie on the sub-grid".into()));
            }
        }
        check_rows(parent.len(), &attributes)?;
        Ok(Self {
            geometry,
            layout: Layout::SubBlocked {
                parent,
                extent,
                grid,
            },
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

    /// Adds or replaces the attribute `name`.
    pub fn with_column(&self, name: &str, column: arrow_array::ArrayRef) -> Result<Self> {
        Ok(Self {
            attributes: crate::set_column(&self.attributes, name, column)?,
            ..self.clone()
        })
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
            Layout::SubBlocked { parent, .. } => parent[row],
        }
    }

    pub fn centroids(&self) -> Vec<[f64; 3]> {
        match &self.layout {
            Layout::SubBlocked { parent, extent, .. } => parent
                .iter()
                .zip(extent)
                .map(|(&p, e)| {
                    let middle = [0, 1, 2].map(|a| (e[a] + e[a + 3]) / 2.0);
                    self.geometry.point(p, middle)
                })
                .collect(),
            _ => (0..self.len())
                .map(|row| self.geometry.centroid(self.parent_index(row)))
                .collect(),
        }
    }

    /// Volume (area in 2D, with unit height) of each row.
    pub fn volumes(&self) -> Vec<f64> {
        let cell = self.geometry.cell_volume();
        match &self.layout {
            Layout::SubBlocked { extent, .. } => extent
                .iter()
                .map(|e| cell * (0..3).map(|a| e[a + 3] - e[a]).product::<f64>())
                .collect(),
            _ => vec![cell; self.len()],
        }
    }

    /// Keeps the rows where `keep` is true; a regular model becomes masked.
    pub fn mask(&self, keep: &BooleanArray) -> Result<Self> {
        check_rows(keep.len(), &self.attributes)?;
        let kept: Vec<usize> = (0..self.len())
            .filter(|&row| keep.is_valid(row) && keep.value(row))
            .collect();
        let layout = match &self.layout {
            Layout::SubBlocked {
                parent,
                extent,
                grid,
            } => Layout::SubBlocked {
                parent: kept.iter().map(|&r| parent[r]).collect(),
                extent: kept.iter().map(|&r| extent[r]).collect(),
                grid: *grid,
            },
            _ => Layout::Masked(kept.iter().map(|&r| self.parent_index(r)).collect()),
        };
        let attributes = filter_record_batch(&self.attributes, keep)?;
        Ok(Self {
            layout,
            attributes,
            ..self.clone()
        })
    }

    /// Every parent cell, one row each: absent cells are null; sub-blocks
    /// merge into their parent, numeric columns as volume-weighted means and
    /// others taking the value of the largest sub-block.
    pub fn to_regular(&self) -> Result<Self> {
        let index = match &self.layout {
            Layout::Regular => return Ok(self.clone()),
            Layout::Masked(index) => index,
            Layout::SubBlocked { .. } => return self.merge_subblocks(),
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

impl BlockModel {
    fn merge_subblocks(&self) -> Result<Self> {
        let cells = rows(self.geometry.cells())?;
        let volumes = self.volumes();
        let mut groups: Vec<Vec<usize>> = vec![vec![]; cells];
        for row in 0..self.len() {
            groups[self.parent_index(row) as usize].push(row);
        }
        let largest: UInt64Array = groups
            .iter()
            .map(|g| {
                g.iter()
                    .copied()
                    .max_by(|&a, &b| volumes[a].total_cmp(&volumes[b]))
                    .map(|r| r as u64)
            })
            .collect();
        let schema = self.attributes.schema();
        let mut fields = Vec::with_capacity(schema.fields().len());
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.capacity());
        for (field, column) in schema.fields().iter().zip(self.attributes.columns()) {
            if field.data_type().is_numeric() {
                let values = cast(column, &DataType::Float64)?;
                let values = values.as_primitive::<Float64Type>();
                let merged: Float64Array = groups
                    .iter()
                    .map(|g| {
                        let (sum, weight) = g
                            .iter()
                            .filter(|&&r| values.is_valid(r))
                            .fold((0.0, 0.0), |(s, w), &r| {
                                (s + values.value(r) * volumes[r], w + volumes[r])
                            });
                        (weight > 0.0).then(|| sum / weight)
                    })
                    .collect();
                fields.push(Field::new(field.name(), DataType::Float64, true));
                columns.push(Arc::new(merged));
            } else {
                fields.push(field.as_ref().clone().with_nullable(true));
                columns.push(take(column, &largest, None)?);
            }
        }
        let attributes = RecordBatch::try_new_with_options(
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(cells)),
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
    #[test]
    fn subblocks_locate_weigh_and_merge() {
        let g = geometry([0.0; 3]);
        let extent = vec![
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0, 0.5, 1.0],
            [0.5, 0.5, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        let model = BlockModel::subblocked(
            g,
            vec![0, 0, 0, 4],
            extent,
            Some([2, 2, 1]),
            grades(vec![1.0, 2.0, 4.0, 7.0]),
        )
        .unwrap();
        assert_eq!(model.centroids()[1], [107.5, 201.25, 1.0]);
        assert_eq!(model.volumes(), vec![50.0, 25.0, 25.0, 100.0]);
        assert!((model.volumes().iter().sum::<f64>() - 2.0 * g.cell_volume()).abs() < 1e-9);
        let regular = model.to_regular().unwrap();
        let au: Vec<_> = regular
            .attributes()
            .column(0)
            .as_primitive::<Float64Type>()
            .iter()
            .collect();
        assert_eq!(au, [Some(2.0), None, None, None, Some(7.0), None]);
        let kept = model
            .mask(&BooleanArray::from(vec![false, true, true, false]))
            .unwrap();
        assert!(
            matches!(kept.layout(), Layout::SubBlocked { parent, .. } if parent == &vec![0, 0])
        );
    }

    #[test]
    fn subblocks_are_validated() {
        let g = geometry([0.0; 3]);
        let off_grid = BlockModel::subblocked(
            g,
            vec![0],
            vec![[0.0, 0.0, 0.0, 0.3, 1.0, 1.0]],
            Some([2, 2, 1]),
            grades(vec![1.0]),
        );
        assert!(off_grid.is_err());
        let inverted = BlockModel::subblocked(
            g,
            vec![0],
            vec![[0.6, 0.0, 0.0, 0.3, 1.0, 1.0]],
            None,
            grades(vec![1.0]),
        );
        assert!(inverted.is_err());
        let unsorted = BlockModel::subblocked(
            g,
            vec![2, 1],
            vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; 2],
            None,
            grades(vec![1.0, 2.0]),
        );
        assert!(unsorted.is_err());
    }
}
