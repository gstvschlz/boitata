use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, RecordBatch, RecordBatchOptions, UInt64Array,
};
use arrow_cast::cast;
use arrow_ord::ord::make_comparator;
use arrow_schema::{DataType, Field, Schema, SortOptions};
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

    /// Cell holding world point `p` and the point's fractional position in
    /// it; the inverse of [`Geometry::point`].
    pub fn locate(&self, p: [f64; 3]) -> Option<(u64, [f64; 3])> {
        let local = block_frame(self.rotation) * Vector3::from_fn(|a, _| p[a] - self.origin[a]);
        let (mut ijk, mut at) = ([0; 3], [0.0; 3]);
        for a in 0..3 {
            let u = local[a] / self.size[a];
            let i = u.floor();
            if !(0.0..self.count[a] as f64).contains(&i) {
                return None;
            }
            (ijk[a], at[a]) = (i as usize, u - i);
        }
        Some((self.index(ijk), at))
    }

    pub fn cell_volume(&self) -> f64 {
        self.size.iter().product()
    }

    /// Smallest grid of `size` cells, rotated by `rotation`, holding every
    /// point plus `buffer` on each side along the grid axes. A non-zero
    /// `snap` puts the origin on a multiple of it in the grid frame, so grids
    /// with the same rotation and snap line up. Without `dz` the grid is 2D:
    /// one layer spanning the buffered z range.
    pub fn from_extents(
        points: &[[f64; 3]],
        size: [f64; 2],
        dz: Option<f64>,
        buffer: [f64; 3],
        rotation: [f64; 3],
        snap: [f64; 3],
    ) -> Result<Self> {
        if points.is_empty() {
            return Err(Error::Geometry("no points to size the grid from".into()));
        }
        if points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Geometry("points must be finite".into()));
        }
        if buffer
            .iter()
            .chain(&snap)
            .any(|v| !(v.is_finite() && *v >= 0.0))
        {
            return Err(Error::Geometry(
                "buffer and snap must be non-negative".into(),
            ));
        }
        let frame = block_frame(rotation);
        let local: Vec<Vector3<f64>> = points.iter().map(|p| frame * Vector3::from(*p)).collect();
        let bound = |a: usize, pick: fn(f64, f64) -> f64| {
            local.iter().map(|p| p[a]).reduce(pick).expect("points")
        };
        let rotated = rotation.iter().any(|&r| r != 0.0);
        let mut start = [0.0; 3];
        let mut end = [0.0; 3];
        for a in 0..3 {
            let (lo, hi) = (
                bound(a, f64::min) - buffer[a],
                bound(a, f64::max) + buffer[a],
            );
            let slack = if rotated && buffer[a] < 1e-9 * (1.0 + lo.abs().max(hi.abs())) {
                1e-9 * (1.0 + lo.abs().max(hi.abs()))
            } else {
                0.0
            };
            start[a] = lo - slack;
            if snap[a] > 0.0 {
                start[a] = (start[a] / snap[a]).floor() * snap[a];
            }
            end[a] = hi;
        }
        let z_span = end[2] - start[2];
        let size = [
            size[0],
            size[1],
            dz.unwrap_or(if z_span > 0.0 {
                z_span * (1.0 + 1e-9)
            } else {
                1.0
            }),
        ];
        let origin = frame.transpose() * Vector3::from(start);
        let mut geometry = Self {
            origin: origin.into(),
            size,
            count: [1; 3],
            rotation,
        };
        geometry.validate()?;
        let offsets: Vec<Vector3<f64>> = points
            .iter()
            .map(|p| frame * (Vector3::from(*p) - origin))
            .collect();
        for a in 0..if dz.is_some() { 3 } else { 2 } {
            let cells = offsets
                .iter()
                .map(|p| (p[a] / size[a]).floor() + 1.0)
                .fold((end[a] - start[a]) / size[a] * (1.0 - 1e-9), f64::max)
                .ceil();
            geometry.count[a] = cells.clamp(1.0, 1e12) as usize;
        }
        if geometry.count.iter().map(|&n| n as f64).product::<f64>() > 1e12 {
            return Err(Error::Geometry("too many cells for the extents".into()));
        }
        if points.iter().any(|&p| geometry.locate(p).is_none()) {
            return Err(Error::Geometry("points fall outside the grid".into()));
        }
        Ok(geometry)
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
    /// Unit of the coordinates, such as `m` or `ft`; None when not declared.
    pub length_unit: Option<String>,
}

impl BlockModel {
    /// The model with coordinates converted to `unit`, which becomes the length
    /// unit; `crs` names their CRS, required when they had one.
    pub fn to_length_unit(&self, unit: &str, crs: Option<String>) -> Result<Self> {
        let (k, crs) =
            crate::units::rescale(self.length_unit.as_deref(), self.crs.as_deref(), unit, crs)?;
        let mut out = self.clone();
        out.geometry
            .origin
            .iter_mut()
            .chain(&mut out.geometry.size)
            .for_each(|v| *v *= k);
        out.crs = crs;
        out.length_unit = Some(unit.into());
        Ok(out)
    }

    pub fn regular(geometry: Geometry, attributes: RecordBatch) -> Result<Self> {
        geometry.validate()?;
        check_rows(rows(geometry.cells())?, &attributes)?;
        Ok(Self {
            geometry,
            layout: Layout::Regular,
            attributes,
            crs: None,
            length_unit: None,
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
            length_unit: None,
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
            length_unit: None,
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

    /// The same blocks with other attributes, one row per block.
    pub fn with_attributes(&self, attributes: RecordBatch) -> Result<Self> {
        check_rows(self.len(), &attributes)?;
        Ok(Self {
            attributes,
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

    /// Row holding world point `p`, if any.
    pub fn row_at(&self, p: [f64; 3]) -> Option<usize> {
        let (cell, at) = self.geometry.locate(p)?;
        match &self.layout {
            Layout::Regular => Some(cell as usize),
            Layout::Masked(index) => index.binary_search(&cell).ok(),
            Layout::SubBlocked { parent, extent, .. } => (parent.partition_point(|&c| c < cell)
                ..parent.partition_point(|&c| c <= cell))
                .find(|&r| (0..3).all(|a| extent[r][a] <= at[a] && at[a] < extent[r][a + 3])),
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

    /// World-space corners of each row's box: 8 vertices. Vertex `c` (0..8)
    /// takes, for axis `a` (0..3), the row's minimum extent on that axis if
    /// bit `a` of `c` is 0, its maximum if 1.
    pub fn corners(&self) -> Vec<[[f64; 3]; 8]> {
        let whole = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        match &self.layout {
            Layout::SubBlocked { parent, extent, .. } => parent
                .iter()
                .zip(extent)
                .map(|(&p, e)| corners_of(&self.geometry, p, e))
                .collect(),
            _ => (0..self.len())
                .map(|row| corners_of(&self.geometry, self.parent_index(row), &whole))
                .collect(),
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

    /// Splits every row into `n` parts per axis, e.g. simulation nodes: a
    /// masked grid `n` times finer, or sub-blocks `n` times smaller. The only
    /// column, `block`, is each node's row in `self`.
    pub fn discretize(&self, n: [usize; 3]) -> Result<Self> {
        if n.contains(&0) {
            return Err(Error::Geometry("discretization must be positive".into()));
        }
        let parts: Vec<[usize; 3]> = (0..n[2])
            .flat_map(|k| (0..n[1]).flat_map(move |j| (0..n[0]).map(move |i| [i, j, k])))
            .collect();
        let rows = rows((self.len() as u64).saturating_mul(parts.len() as u64))?;
        let block = |rows: Vec<u64>| {
            RecordBatch::try_from_iter([("block", Arc::new(UInt64Array::from(rows)) as ArrayRef)])
        };
        if let Layout::SubBlocked {
            parent,
            extent,
            grid,
        } = &self.layout
        {
            let grid = grid
                .map(|g| {
                    (0..3)
                        .map(|a| u32::try_from(n[a]).ok().and_then(|n| g[a].checked_mul(n)))
                        .collect::<Option<Vec<_>>>()
                        .map(|g| [g[0], g[1], g[2]])
                        .ok_or_else(|| Error::Geometry("sub-grid too fine".into()))
                })
                .transpose()?;
            let (mut parents, mut extents, mut owner) = (
                Vec::with_capacity(rows),
                Vec::with_capacity(rows),
                Vec::with_capacity(rows),
            );
            for (row, (&p, e)) in parent.iter().zip(extent).enumerate() {
                let edge = |a: usize, t: usize| {
                    if t == n[a] {
                        e[a + 3]
                    } else {
                        e[a] + (e[a + 3] - e[a]) * t as f64 / n[a] as f64
                    }
                };
                for s in &parts {
                    parents.push(p);
                    extents.push([0, 1, 2, 3, 4, 5].map(|i| edge(i % 3, s[i % 3] + i / 3)));
                    owner.push(row as u64);
                }
            }
            let mut model = Self::subblocked(self.geometry, parents, extents, grid, block(owner)?)?;
            model.crs.clone_from(&self.crs);
            model.length_unit.clone_from(&self.length_unit);
            return Ok(model);
        }
        let g = &self.geometry;
        let mut count = [0; 3];
        for a in 0..3 {
            count[a] = g.count[a]
                .checked_mul(n[a])
                .ok_or_else(|| Error::Geometry("too many cells".into()))?;
        }
        count
            .iter()
            .try_fold(1u64, |c, &m| c.checked_mul(m as u64))
            .ok_or_else(|| Error::Geometry("too many cells".into()))?;
        let fine = Geometry {
            size: [0, 1, 2].map(|a| g.size[a] / n[a] as f64),
            count,
            ..*g
        };
        let mut nodes = Vec::with_capacity(rows);
        for row in 0..self.len() {
            let ijk = g.ijk(self.parent_index(row));
            nodes.extend(parts.iter().map(|s| {
                (
                    fine.index([0, 1, 2].map(|a| ijk[a] * n[a] + s[a])),
                    row as u64,
                )
            }));
        }
        nodes.sort_unstable();
        let (index, owner) = nodes.into_iter().unzip();
        let mut model = Self::masked(fine, index, block(owner)?)?;
        model.crs.clone_from(&self.crs);
        model.length_unit.clone_from(&self.length_unit);
        Ok(model)
    }

    /// Every parent cell, one row each: absent cells are null; sub-blocks
    /// merge into their parent as in [`BlockModel::regularize`], without the
    /// `fraction` column.
    pub fn to_regular(&self) -> Result<Self> {
        let index = match &self.layout {
            Layout::Regular => return Ok(self.clone()),
            Layout::Masked(index) => index,
            Layout::SubBlocked { .. } => {
                let cells = rows(self.geometry.cells())?;
                let schema = Schema::new_with_metadata(
                    Vec::<Field>::new(),
                    self.attributes.schema().metadata().clone(),
                );
                let empty = RecordBatch::try_new_with_options(
                    Arc::new(schema),
                    vec![],
                    &RecordBatchOptions::new().with_row_count(Some(cells)),
                )?;
                let target = Self::regular(self.geometry, empty)?;
                let (columns, _) = self.transfer(&target, 0.0)?;
                return target.with_columns(columns);
            }
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

    /// The rows of `target` carrying every column of `self`, weighted by the
    /// volume each row of `self` shares with them: floating-point columns as
    /// weighted means of the non-null rows, other columns as the value filling
    /// the most volume (ties to the smallest; nulls do not vote). Works both
    /// ways between regular, masked and sub-blocked models of the same
    /// rotation; the overlaps are exact boxes. `fraction` is the share of each
    /// target row covered by `self`; rows below `min_fraction` are null.
    pub fn regularize(&self, target: &BlockModel, min_fraction: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&min_fraction) {
            return Err(Error::Geometry("min_fraction must be in [0, 1]".into()));
        }
        let (mut columns, fraction) = self.transfer(target, min_fraction)?;
        columns.push(("fraction".into(), Arc::new(Float64Array::from(fraction))));
        let out = target.with_columns(columns)?;
        let mut attributes = out.attributes.clone();
        let units = crate::units::units(&self.attributes);
        let fraction = [("fraction".to_string(), "ratio".to_string())];
        for (name, unit) in units.iter().chain(&fraction) {
            attributes = crate::units::with_unit_unchecked(&attributes, name, Some(unit))?;
        }
        out.with_attributes(attributes)
    }

    fn with_columns(&self, columns: Vec<(String, ArrayRef)>) -> Result<Self> {
        let mut attributes = self.attributes.clone();
        for (name, column) in columns {
            attributes = crate::set_column(&attributes, &name, column)?;
        }
        self.with_attributes(attributes)
    }

    /// Box of each row in the local frame, `[x0, y0, z0, x1, y1, z1]`.
    fn boxes(&self) -> Vec<[f64; 6]> {
        let g = &self.geometry;
        (0..self.len())
            .map(|row| {
                let ijk = g.ijk(self.parent_index(row));
                let e = match &self.layout {
                    Layout::SubBlocked { extent, .. } => extent[row],
                    _ => [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                };
                std::array::from_fn(|i| (ijk[i % 3] as f64 + e[i]) * g.size[i % 3])
            })
            .collect()
    }

    fn rows_in(&self, cell: u64) -> std::ops::Range<usize> {
        match &self.layout {
            Layout::Regular => cell as usize..cell as usize + 1,
            Layout::Masked(index) => match index.binary_search(&cell) {
                Ok(r) => r..r + 1,
                Err(r) => r..r,
            },
            Layout::SubBlocked { parent, .. } => {
                parent.partition_point(|&c| c < cell)..parent.partition_point(|&c| c <= cell)
            }
        }
    }

    /// Rows of `self` sharing volume with each row of `target`.
    fn overlaps(&self, target: &BlockModel) -> Result<Vec<Vec<(usize, f64)>>> {
        let (s, t) = (&self.geometry, &target.geometry);
        if s.rotation != t.rotation {
            return Err(Error::Geometry("models must share the rotation".into()));
        }
        if let (Some(a), Some(b)) = (&self.crs, &target.crs)
            && a != b
        {
            return Err(Error::Geometry("models must share the CRS".into()));
        }
        crate::units::same_length_unit(
            ("source blocks", self.length_unit.as_deref()),
            ("target blocks", target.length_unit.as_deref()),
        )?;
        let shift = block_frame(s.rotation) * Vector3::from_fn(|a, _| t.origin[a] - s.origin[a]);
        let source = self.boxes();
        Ok(target
            .boxes()
            .into_iter()
            .map(|b| {
                let b: [f64; 6] = std::array::from_fn(|i| b[i] + shift[i % 3]);
                let cells = |a: usize| {
                    let lo = (b[a] / s.size[a]).floor().max(0.0) as usize;
                    lo..((b[a + 3] / s.size[a]).ceil().max(0.0) as usize).min(s.count[a])
                };
                let tiny = 1e-9 * (0..3).map(|a| b[a + 3] - b[a]).product::<f64>();
                let mut found = vec![];
                for k in cells(2) {
                    for j in cells(1) {
                        for i in cells(0) {
                            for r in self.rows_in(s.index([i, j, k])) {
                                let o = &source[r];
                                let v = (0..3)
                                    .map(|a| (b[a + 3].min(o[a + 3]) - b[a].max(o[a])).max(0.0))
                                    .product::<f64>();
                                if v > tiny {
                                    found.push((r, v));
                                }
                            }
                        }
                    }
                }
                found
            })
            .collect())
    }

    /// Columns of `self` on the rows of `target`, and the covered fraction.
    fn transfer(
        &self,
        target: &BlockModel,
        min_fraction: f64,
    ) -> Result<(Vec<(String, ArrayRef)>, Vec<f64>)> {
        let overlaps = self.overlaps(target)?;
        let fraction: Vec<f64> = overlaps
            .iter()
            .zip(target.volumes())
            .map(|(o, v)| (o.iter().map(|x| x.1).sum::<f64>() / v).min(1.0))
            .collect();
        let kept = |t: usize| fraction[t] > 0.0 && fraction[t] >= min_fraction;
        let schema = self.attributes.schema();
        let mut columns = Vec::with_capacity(schema.fields().len());
        for (field, column) in schema.fields().iter().zip(self.attributes.columns()) {
            let merged: ArrayRef = if field.data_type().is_floating() {
                let values = cast(column, &DataType::Float64)?;
                let values = values.as_primitive::<Float64Type>();
                let means: Float64Array = overlaps
                    .iter()
                    .enumerate()
                    .map(|(t, o)| {
                        let (sum, weight) = o
                            .iter()
                            .filter(|(r, _)| values.is_valid(*r))
                            .fold((0.0, 0.0), |(s, w), &(r, v)| {
                                (s + values.value(r) * v, w + v)
                            });
                        (kept(t) && weight > 0.0).then(|| sum / weight)
                    })
                    .collect();
                Arc::new(means)
            } else {
                let (rank, first) = ranks(column)?;
                let winner: UInt64Array = overlaps
                    .iter()
                    .enumerate()
                    .map(|(t, o)| {
                        let mut share: Vec<(usize, f64)> = vec![];
                        for &(r, v) in o {
                            let Some(k) = rank[r] else { continue };
                            match share.iter_mut().find(|s| s.0 == k) {
                                Some(s) => s.1 += v,
                                None => share.push((k, v)),
                            }
                        }
                        share
                            .iter()
                            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
                            .filter(|_| kept(t))
                            .map(|s| first[s.0] as u64)
                    })
                    .collect();
                take(column, &winner, None)?
            };
            columns.push((field.name().clone(), merged));
        }
        Ok((columns, fraction))
    }
}

/// Dense rank of each non-null value in sort order, and the first row of each.
fn ranks(column: &ArrayRef) -> Result<(Vec<Option<usize>>, Vec<usize>)> {
    let compare = make_comparator(column.as_ref(), column.as_ref(), SortOptions::default())?;
    let mut order: Vec<usize> = (0..column.len()).filter(|&r| column.is_valid(r)).collect();
    order.sort_by(|&a, &b| compare(a, b).then(a.cmp(&b)));
    let (mut rank, mut first) = (vec![None; column.len()], vec![]);
    for (n, &r) in order.iter().enumerate() {
        if n == 0 || compare(order[n - 1], r).is_ne() {
            first.push(r);
        }
        rank[r] = Some(first.len() - 1);
    }
    Ok((rank, first))
}

fn rows(cells: u64) -> Result<usize> {
    usize::try_from(cells).map_err(|_| Error::Geometry("too many cells".into()))
}

fn corners_of(geometry: &Geometry, index: u64, extent: &[f64; 6]) -> [[f64; 3]; 8] {
    let mut corners = [[0.0; 3]; 8];
    for (c, corner) in corners.iter_mut().enumerate() {
        let at = [0, 1, 2].map(|a| extent[a + 3 * ((c >> a) & 1)]);
        *corner = geometry.point(index, at);
    }
    corners
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

    fn scatter(n: usize, lo: [f64; 3], hi: [f64; 3]) -> Vec<[f64; 3]> {
        let mut state = 12345u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| [0, 1, 2].map(|a| lo[a] + (hi[a] - lo[a]) * next()))
            .collect()
    }

    fn far_corner(g: &Geometry) -> [f64; 3] {
        g.point(g.cells() - 1, [1.0; 3])
    }

    #[test]
    fn extents_hold_every_point_with_the_exact_buffer() {
        let points = scatter(500, [310.0, -45.0, 120.0], [870.0, 260.0, 415.0]);
        let buffer = [25.0, 10.0, 5.0];
        let g = Geometry::from_extents(
            &points,
            [20.0, 20.0],
            Some(10.0),
            buffer,
            [0.0; 3],
            [0.0; 3],
        )
        .unwrap();
        assert!(points.iter().all(|&p| g.locate(p).is_some()));
        let far = far_corner(&g);
        for a in 0..3 {
            let lo = points.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min);
            let hi = points
                .iter()
                .map(|p| p[a])
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(g.origin[a], lo - buffer[a]);
            assert!(far[a] >= hi + buffer[a] && far[a] - g.size[a] < hi + buffer[a]);
        }
    }

    #[test]
    fn snapped_origins_are_multiples_of_snap() {
        let points = scatter(200, [313.7, -44.2, 121.9], [870.0, 260.0, 415.0]);
        let snap = [50.0, 25.0, 10.0];
        let g = Geometry::from_extents(&points, [10.0, 10.0], Some(5.0), [3.0; 3], [0.0; 3], snap)
            .unwrap();
        for a in 0..3 {
            let k = g.origin[a] / snap[a];
            assert_eq!(k, k.round());
            assert!(g.origin[a] <= points.iter().map(|p| p[a]).fold(f64::INFINITY, f64::min) - 3.0);
        }
        assert!(points.iter().all(|&p| g.locate(p).is_some()));
    }

    #[test]
    fn rotated_extents_hold_rotated_points_with_the_fewest_cells() {
        let rotation = [35.0, 20.0, 10.0];
        let to_world = block_frame(rotation).transpose();
        let origin = Vector3::new(1000.0, 5000.0, 300.0);
        let points: Vec<[f64; 3]> = scatter(400, [0.5, 0.5, 0.5], [99.5, 49.5, 29.5])
            .into_iter()
            .map(|p| (origin + to_world * Vector3::from(p)).into())
            .collect();
        let g = Geometry::from_extents(
            &points,
            [10.0, 10.0],
            Some(10.0),
            [0.0; 3],
            rotation,
            [0.0; 3],
        )
        .unwrap();
        assert_eq!(g.count, [10, 5, 3]);
        assert!(points.iter().all(|&p| g.locate(p).is_some()));
        let unrotated = Geometry::from_extents(
            &points,
            [10.0, 10.0],
            Some(10.0),
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
        )
        .unwrap();
        assert!(unrotated.cells() > g.cells());
    }

    #[test]
    fn extents_rebuild_a_rotated_grid_from_its_centroids() {
        let rotation = [35.0, 20.0, 10.0];
        let to_world = block_frame(rotation).transpose();
        let origin = Vector3::new(1000.0, 5000.0, 300.0);
        let mut centroids = Vec::new();
        for k in 0..3 {
            for j in 0..4 {
                for i in 0..5 {
                    let local = Vector3::new(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5) * 10.0;
                    centroids.push((origin + to_world * local).into());
                }
            }
        }
        let g = Geometry::from_extents(
            &centroids,
            [10.0, 10.0],
            Some(10.0),
            [5.0; 3],
            rotation,
            [0.0; 3],
        )
        .unwrap();
        assert_eq!(g.count, [5, 4, 3]);
        assert!((Vector3::from(g.origin) - origin).norm() < 1e-9);
    }

    #[test]
    fn extents_without_dz_give_one_layer() {
        let flat = scatter(100, [0.0, 0.0, 0.0], [95.0, 45.0, 0.0]);
        let g =
            Geometry::from_extents(&flat, [5.0, 5.0], None, [0.0; 3], [0.0; 3], [0.0; 3]).unwrap();
        assert_eq!(g.count[2], 1);
        let deep = scatter(100, [0.0, 0.0, 100.0], [95.0, 45.0, 180.0]);
        let g =
            Geometry::from_extents(&deep, [5.0, 5.0], None, [0.0; 3], [0.0; 3], [0.0; 3]).unwrap();
        assert_eq!(g.count[2], 1);
        assert!(deep.iter().all(|&p| g.locate(p).is_some()));
        assert!(
            Geometry::from_extents(&[], [5.0, 5.0], None, [0.0; 3], [0.0; 3], [0.0; 3]).is_err()
        );
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
        let g = geometry([30.0, 20.0, 10.0]);
        for cell in 0..g.cells() {
            let (found, at) = g.locate(g.point(cell, [0.2, 0.5, 0.9])).unwrap();
            assert_eq!(found, cell);
            assert!((at[0] - 0.2).abs() < 1e-9 && (at[2] - 0.9).abs() < 1e-9);
        }
        assert!(g.locate([0.0; 3]).is_none());
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
        for (row, c) in model.centroids().into_iter().enumerate() {
            assert_eq!(model.row_at(c), Some(row));
        }
        assert_eq!(model.row_at(g.centroid(1)), None);
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

    fn check_nodes(model: &BlockModel, n: [usize; 3]) {
        let nodes = model.discretize(n).unwrap();
        let block: Vec<usize> = nodes
            .attributes()
            .column(0)
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .iter()
            .map(|&b| b as usize)
            .collect();
        assert_eq!(nodes.len(), model.len() * n.iter().product::<usize>());
        let mut volume = vec![0.0; model.len()];
        for ((c, v), &b) in nodes
            .centroids()
            .into_iter()
            .zip(nodes.volumes())
            .zip(&block)
        {
            assert_eq!(model.row_at(c), Some(b));
            volume[b] += v;
        }
        for (sum, v) in volume.iter().zip(model.volumes()) {
            assert!((sum - v).abs() < 1e-9 * v);
        }
    }

    #[test]
    fn discretized_nodes_fill_their_blocks() {
        let rotated = geometry([30.0, 20.0, 10.0]);
        let m = BlockModel::regular(rotated, grades(vec![1.0; 6])).unwrap();
        check_nodes(&m, [3, 2, 4]);
        let masked = m
            .mask(&BooleanArray::from(vec![
                false, true, true, false, false, true,
            ]))
            .unwrap();
        check_nodes(&masked, [2, 2, 2]);
        let extent = vec![
            [0.0, 0.0, 0.0, 0.3, 1.0, 1.0],
            [0.3, 0.0, 0.0, 1.0, 0.7, 1.0],
            [0.3, 0.7, 0.2, 1.0, 1.0, 0.9],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        let sub = BlockModel::subblocked(
            rotated,
            vec![0, 0, 0, 5],
            extent,
            None,
            grades(vec![1.0; 4]),
        )
        .unwrap();
        check_nodes(&sub, [3, 2, 2]);
        let on_grid = BlockModel::subblocked(
            rotated,
            vec![0, 0],
            vec![
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            Some([2, 1, 1]),
            grades(vec![1.0; 2]),
        )
        .unwrap();
        check_nodes(&on_grid, [3, 2, 1]);
        assert!(matches!(
            on_grid.discretize([3, 2, 1]).unwrap().layout(),
            Layout::SubBlocked {
                grid: Some([6, 2, 1]),
                ..
            }
        ));
        assert!(m.discretize([2, 0, 1]).is_err());
    }

    #[test]
    fn discretizing_by_one_gives_the_centroids() {
        let m = BlockModel::regular(geometry([30.0, 0.0, 0.0]), grades(vec![1.0; 6])).unwrap();
        let nodes = m.discretize([1, 1, 1]).unwrap();
        assert_eq!(nodes.centroids(), m.centroids());
        assert_eq!(nodes.geometry(), m.geometry());
    }

    fn column(model: &BlockModel, name: &str) -> Vec<Option<f64>> {
        let c = model.attributes().column_by_name(name).unwrap();
        c.as_primitive::<Float64Type>().iter().collect()
    }

    /// Volume × grade is conserved from sub-blocks to a coarser, shifted,
    /// rotated grid, and every sub-block's volume lands somewhere.
    #[test]
    fn regularizing_conserves_volume_and_metal() {
        let rotation = [30.0, 20.0, 10.0];
        let fine = Geometry {
            origin: [0.0; 3],
            size: [4.0, 4.0, 2.0],
            count: [5, 5, 5],
            rotation,
        };
        let extent: Vec<[f64; 6]> = (0..fine.cells())
            .flat_map(|_| {
                [
                    [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
                    [0.5, 0.0, 0.0, 1.0, 1.0, 0.25],
                ]
            })
            .collect();
        let parent: Vec<u64> = (0..fine.cells()).flat_map(|c| [c, c]).collect();
        let grade: Vec<f64> = (0..parent.len())
            .map(|r| (r as f64 * 0.37).sin() + 2.0)
            .collect();
        let source =
            BlockModel::subblocked(fine, parent, extent, Some([2, 1, 4]), grades(grade.clone()))
                .unwrap();
        let frame = block_frame(rotation).transpose();
        let shift = frame * Vector3::new(-3.0, -5.0, -1.0);
        let coarse = Geometry {
            origin: [shift.x, shift.y, shift.z],
            size: [10.0, 10.0, 5.0],
            count: [3, 3, 3],
            rotation,
        };
        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(27)),
        )
        .unwrap();
        let target = BlockModel::regular(coarse, empty).unwrap();
        let out = source.regularize(&target, 0.0).unwrap();
        let (au, fraction) = (column(&out, "au"), column(&out, "fraction"));
        let volume: f64 = source.volumes().iter().sum();
        let metal: f64 = source
            .volumes()
            .iter()
            .zip(&grade)
            .map(|(v, g)| v * g)
            .sum();
        let mut covered = (0.0, 0.0);
        for ((a, f), v) in au.iter().zip(&fraction).zip(out.volumes()) {
            let f = f.unwrap();
            covered.0 += v * f;
            covered.1 += v * f * a.unwrap_or(0.0);
        }
        assert!((covered.0 - volume).abs() < 1e-9 * volume);
        assert!((covered.1 - metal).abs() < 1e-9 * metal);

        let strict = source.regularize(&target, 0.5).unwrap();
        for (a, f) in column(&strict, "au").iter().zip(&fraction) {
            assert_eq!(a.is_some(), f.unwrap() >= 0.5);
        }
        let turned = BlockModel::regular(
            Geometry {
                rotation: [0.0; 3],
                ..coarse
            },
            target.attributes().clone(),
        )
        .unwrap();
        assert!(source.regularize(&turned, 0.0).is_err());
        assert!(source.regularize(&target, 1.5).is_err());
    }

    /// Other columns take the value filling the most volume; ties go to the
    /// smallest and nulls do not vote. A regular grid spreads onto sub-blocks.
    #[test]
    fn categories_take_the_volume_majority() {
        let g = geometry([0.0; 3]);
        let extent = vec![
            [0.0, 0.0, 0.0, 0.25, 1.0, 1.0],
            [0.25, 0.0, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        let rock = arrow_array::StringArray::from(vec![
            Some("b"),
            Some("b"),
            Some("a"),
            Some("z"),
            Some("c"),
            None,
            Some("d"),
        ]);
        let code = arrow_array::Int32Array::from(vec![7, 7, 3, 1, 1, 2, 2]);
        let attributes = RecordBatch::try_from_iter([
            ("rock", Arc::new(rock) as ArrayRef),
            ("code", Arc::new(code) as ArrayRef),
        ])
        .unwrap();
        let model =
            BlockModel::subblocked(g, vec![0, 0, 0, 1, 1, 2, 2], extent, None, attributes).unwrap();
        let regular = model.to_regular().unwrap();
        let rock = regular.attributes().column(0).as_string::<i32>();
        let rock: Vec<_> = rock.iter().take(3).collect();
        assert_eq!(rock, [Some("a"), Some("c"), Some("d")]);
        let code = regular
            .attributes()
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(code.values()[..3], [3, 1, 2]);
        assert_eq!(regular.attributes().num_columns(), 2);

        let whole = BlockModel::regular(g, grades((0..6).map(f64::from).collect())).unwrap();
        let spread = whole.regularize(&model, 0.0).unwrap();
        assert_eq!(
            column(&spread, "au"),
            [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0].map(Some)
        );
        assert!(column(&spread, "fraction").iter().all(|f| *f == Some(1.0)));
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

    use proptest::prelude::*;

    #[test]
    fn corners_of_an_unrotated_cell_are_its_axis_aligned_box() {
        let g = geometry([0.0; 3]);
        let model = BlockModel::regular(g, grades(vec![0.0; 6])).unwrap();
        let corners = model.corners();
        assert_eq!(corners.len(), 6);
        assert_eq!(corners[0][0], g.origin);
        let hi = [
            g.origin[0] + g.size[0],
            g.origin[1] + g.size[1],
            g.origin[2] + g.size[2],
        ];
        assert_eq!(corners[0][7], hi);
    }

    #[test]
    fn corners_of_a_masked_row_use_its_own_parent_cell() {
        let g = geometry([0.0; 3]);
        let model = BlockModel::masked(g, vec![4], grades(vec![0.0])).unwrap();
        let corners = model.corners()[0];
        assert_eq!(corners[0], g.point(4, [0.0; 3]));
        assert_eq!(corners[7], g.point(4, [1.0; 3]));
    }

    fn sub_extent() -> impl Strategy<Value = [f64; 6]> {
        (
            0.05f64..0.95,
            0.05f64..0.95,
            0.05f64..0.95,
            0.05f64..0.95,
            0.05f64..0.95,
            0.05f64..0.95,
        )
            .prop_map(|(u0, v0, w0, du, dv, dw)| {
                [
                    u0,
                    v0,
                    w0,
                    u0 + du * (1.0 - u0),
                    v0 + dv * (1.0 - v0),
                    w0 + dw * (1.0 - w0),
                ]
            })
    }

    fn box_volume(model: &BlockModel) -> f64 {
        let c = model.corners()[0];
        let edge = |i: usize| Vector3::from(c[i]) - Vector3::from(c[0]);
        edge(1).cross(&edge(2)).dot(&edge(4)).abs()
    }

    proptest! {
        #[test]
        fn corner_box_volume_matches_row_volume(
            azimuth in 0.0f64..360.0,
            dip in -90.0f64..90.0,
            rake in -180.0f64..180.0,
            extent in sub_extent(),
        ) {
            for rotation in [[0.0; 3], [azimuth, dip, rake]] {
                let g = geometry(rotation);
                let model =
                    BlockModel::subblocked(g, vec![0], vec![extent], None, grades(vec![0.0])).unwrap();
                prop_assert!((box_volume(&model) - model.volumes()[0]).abs() < 1e-6);
            }
        }
    }
}
