use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, Float64Builder, ListBuilder, StructBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray, UInt32Array};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Fields};

use crate::mesh::take_rows;
use crate::{Error, PointSet, Result, check_rows};

/// Lines and polygons: one attribute row per feature, each feature made of
/// zero or more parts, each part an open or closed run of vertices. Closed
/// parts do not repeat their first vertex. Holes and multipart features are
/// several parts of one feature; inside is decided by even-odd counting in xy
/// over the feature's closed parts.
#[derive(Debug, Clone)]
pub struct Polylines {
    vertices: Vec<[f64; 3]>,
    parts: Vec<u32>,
    features: Vec<u32>,
    closed: Vec<bool>,
    attributes: RecordBatch,
    pub crs: Option<String>,
}

fn check_offsets(offsets: &[u32], end: usize, what: &str) -> Result<()> {
    let bad = |why: &str| Err(Error::Geometry(format!("{what} offsets {why}")));
    if offsets.first() != Some(&0) {
        return bad("must start at 0");
    }
    if offsets.windows(2).any(|w| w[1] < w[0]) {
        return bad("must not decrease");
    }
    if *offsets.last().expect("non-empty") as usize != end {
        return bad(&format!("must end at {end}"));
    }
    if end > i32::MAX as usize {
        return bad("must fit in i32");
    }
    Ok(())
}

fn with_reserved(table: &RecordBatch, columns: [(&str, ArrayRef); 2]) -> Result<RecordBatch> {
    let mut table = table.clone();
    for (name, column) in columns {
        if table.schema().index_of(name).is_ok() {
            return Err(Error::Geometry(format!("attribute `{name}` is reserved")));
        }
        table = crate::set_column(&table, name, column)?;
    }
    Ok(table)
}

fn without(table: &RecordBatch, names: &[Option<&str>]) -> Result<RecordBatch> {
    let schema = table.schema();
    let keep: Vec<usize> = (0..schema.fields().len())
        .filter(|&i| !names.contains(&Some(schema.field(i).name().as_str())))
        .collect();
    Ok(table.project(&keep)?)
}

impl Polylines {
    /// `parts` offsets into `vertices` (one more than the parts), `features`
    /// offsets into the parts (one more than the features), `closed` one flag
    /// per part and `attributes` one row per feature.
    pub fn new(
        vertices: Vec<[f64; 3]>,
        parts: Vec<u32>,
        features: Vec<u32>,
        closed: Vec<bool>,
        attributes: RecordBatch,
    ) -> Result<Self> {
        check_offsets(&parts, vertices.len(), "part")?;
        check_offsets(&features, parts.len() - 1, "feature")?;
        check_rows(features.len() - 1, &attributes)?;
        if vertices.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Geometry("polyline vertices must be finite".into()));
        }
        if closed.len() != parts.len() - 1 {
            return Err(Error::Length {
                expected: parts.len() - 1,
                found: closed.len(),
            });
        }
        for (i, (w, &c)) in parts.windows(2).zip(&closed).enumerate() {
            let min = if c { 3 } else { 2 };
            if w[1] - w[0] < min {
                return Err(Error::Geometry(format!(
                    "part {i} needs at least {min} vertices"
                )));
            }
        }
        Ok(Self {
            vertices,
            parts,
            features,
            closed,
            attributes,
            crs: None,
        })
    }

    /// Number of features.
    pub fn len(&self) -> usize {
        self.features.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_parts(&self) -> usize {
        self.parts.len() - 1
    }

    pub fn vertices(&self) -> &[[f64; 3]] {
        &self.vertices
    }

    pub fn part(&self, i: usize) -> &[[f64; 3]] {
        &self.vertices[self.parts[i] as usize..self.parts[i + 1] as usize]
    }

    /// Indices of the parts of feature `f`.
    pub fn feature_parts(&self, f: usize) -> Range<usize> {
        self.features[f] as usize..self.features[f + 1] as usize
    }

    pub fn closed(&self) -> &[bool] {
        &self.closed
    }

    pub fn attributes(&self) -> &RecordBatch {
        &self.attributes
    }

    /// Adds or replaces the per-feature attribute `name`.
    pub fn with_column(&self, name: &str, column: ArrayRef) -> Result<Self> {
        Ok(Self {
            attributes: crate::set_column(&self.attributes, name, column)?,
            ..self.clone()
        })
    }

    /// `(min, max)` corners, `None` without vertices.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let first = *self.vertices.first()?;
        Some(self.vertices.iter().fold((first, first), |(lo, hi), p| {
            (
                [0, 1, 2].map(|a| lo[a].min(p[a])),
                [0, 1, 2].map(|a| hi[a].max(p[a])),
            )
        }))
    }

    /// Builds from one row per vertex. Rows are grouped by `feature`, then by
    /// `part`, in order of first appearance, keeping row order within a part;
    /// without `part` each feature is one part. Attributes come from each
    /// feature's first row; without `z`, z is 0.
    pub fn from_table(
        table: &RecordBatch,
        feature: &str,
        x: &str,
        y: &str,
        z: Option<&str>,
        part: Option<&str>,
        closed: bool,
    ) -> Result<Self> {
        if table.num_rows() > i32::MAX as usize {
            return Err(Error::Geometry("too many vertices".into()));
        }
        let points = PointSet::from_table(table, x, y, z)?;
        let keys = |name: Option<&str>| -> Result<StringArray> {
            let Some(name) = name else {
                return Ok(StringArray::new_null(table.num_rows()));
            };
            let column = points
                .attributes()
                .column_by_name(name)
                .ok_or_else(|| Error::MissingColumn(name.into()))?;
            Ok(cast(column, &DataType::Utf8)?.as_string::<i32>().clone())
        };
        let (fid, pid) = (keys(Some(feature))?, keys(part)?);
        let (mut features, mut first) = (HashMap::new(), vec![]);
        let (mut parts, mut owner, mut rows) = (HashMap::new(), vec![], vec![]);
        for (r, (f, p)) in fid.iter().zip(&pid).enumerate() {
            let f = *features.entry(f).or_insert_with(|| {
                first.push(r as u32);
                first.len() - 1
            });
            let p = *parts.entry((f, p)).or_insert_with(|| {
                owner.push(f);
                rows.push(vec![]);
                rows.len() - 1
            });
            rows[p].push(r);
        }
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&p| owner[p]);
        let (mut vertices, mut offsets) = (vec![], vec![0]);
        let mut counts = vec![0; first.len() + 1];
        for p in order {
            vertices.extend(rows[p].iter().map(|&r| points.coords()[r]));
            offsets.push(vertices.len() as u32);
            counts[owner[p] + 1] += 1;
        }
        for f in 1..counts.len() {
            counts[f] += counts[f - 1];
        }
        let attributes = take_rows(&without(points.attributes(), &[part])?, first)?;
        let closed = vec![closed; offsets.len() - 1];
        Self::new(vertices, offsets, counts, closed, attributes)
    }

    /// One row per feature: the attributes, `geometry` as a list of parts,
    /// each a list of `{x, y, z}`, and `closed` as a list of flags.
    pub fn to_table(&self) -> Result<RecordBatch> {
        let xyz: Fields = ["x", "y", "z"]
            .map(|n| Arc::new(Field::new(n, DataType::Float64, false)))
            .into();
        let vertices = StructBuilder::from_fields(xyz, self.vertices.len());
        let mut geometry = ListBuilder::new(ListBuilder::new(vertices));
        let mut closed = ListBuilder::new(BooleanBuilder::new());
        for f in 0..self.len() {
            for p in self.feature_parts(f) {
                let vertices = geometry.values().values();
                for v in self.part(p) {
                    for (a, &c) in v.iter().enumerate() {
                        let axis = vertices.field_builder::<Float64Builder>(a);
                        axis.expect("float64 axis").append_value(c);
                    }
                    vertices.append(true);
                }
                geometry.values().append(true);
                closed.values().append_value(self.closed[p]);
            }
            geometry.append(true);
            closed.append(true);
        }
        let geometry: ArrayRef = Arc::new(geometry.finish());
        with_reserved(
            &self.attributes,
            [
                ("geometry", geometry),
                ("closed", Arc::new(closed.finish())),
            ],
        )
    }

    /// Reads the form written by [`Polylines::to_table`].
    pub fn from_nested(table: &RecordBatch) -> Result<Self> {
        let bad = |what: &str| Error::Geometry(format!("`{what}` has the wrong layout"));
        let list = |name: &str| {
            let list = table
                .column_by_name(name)
                .and_then(|c| c.as_list_opt::<i32>());
            list.filter(|l| l.null_count() == 0)
                .ok_or_else(|| bad(name))
        };
        let (geometry, flags) = (list("geometry")?, list("closed")?);
        let parts = geometry.values().as_list_opt::<i32>();
        let parts = parts.ok_or_else(|| bad("geometry"))?;
        let xyz = parts
            .values()
            .as_struct_opt()
            .ok_or_else(|| bad("geometry"))?;
        let axes = ["x", "y", "z"].map(|n| {
            let axis = xyz.column_by_name(n);
            axis.and_then(|c| c.as_primitive_opt::<Float64Type>())
                .filter(|a| a.null_count() == 0)
        });
        let [Some(xs), Some(ys), Some(zs)] = axes else {
            return Err(bad("geometry"));
        };
        let closed = flags
            .values()
            .as_boolean_opt()
            .ok_or_else(|| bad("closed"))?;
        let rebase = |o: &[i32]| o.iter().map(|&i| (i - o[0]) as u32).collect::<Vec<_>>();
        let range = |o: &[i32]| o[0] as usize..o[o.len() - 1] as usize;
        let (f, c) = (geometry.value_offsets(), flags.value_offsets());
        let p = &parts.value_offsets()[f[0] as usize..=f[f.len() - 1] as usize];
        if rebase(c) != rebase(f) {
            return Err(bad("closed"));
        }
        Self::new(
            range(p)
                .map(|i| [xs.value(i), ys.value(i), zs.value(i)])
                .collect(),
            rebase(p),
            rebase(f),
            range(c).map(|i| closed.value(i)).collect(),
            without(table, &[Some("geometry"), Some("closed")])?,
        )
    }

    /// One point per vertex with `feature` and `part` indices and the
    /// feature's attributes.
    pub fn to_points(&self) -> Result<PointSet> {
        let mut feature = Vec::with_capacity(self.vertices.len());
        let mut part = Vec::with_capacity(self.vertices.len());
        for f in 0..self.len() {
            for p in self.feature_parts(f) {
                let n = self.part(p).len();
                feature.extend(std::iter::repeat_n(f as u32, n));
                part.extend(std::iter::repeat_n(p as u32, n));
            }
        }
        let table = with_reserved(
            &take_rows(&self.attributes, feature.clone())?,
            [
                ("feature", Arc::new(UInt32Array::from(feature)) as ArrayRef),
                ("part", Arc::new(UInt32Array::from(part))),
            ],
        )?;
        let mut points = PointSet::new(self.vertices.clone(), table)?;
        points.crs = self.crs.clone();
        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt32Type;

    fn rock(values: &[&str]) -> RecordBatch {
        RecordBatch::try_from_iter([(
            "rock",
            Arc::new(StringArray::from(values.to_vec())) as ArrayRef,
        )])
        .unwrap()
    }

    fn square(o: f64, s: f64) -> Vec<[f64; 3]> {
        vec![
            [o, o, 0.],
            [o + s, o, 0.],
            [o + s, o + s, 0.],
            [o, o + s, 0.],
        ]
    }

    /// A pit with a hole, an empty feature and a two-segment line.
    fn sample() -> Polylines {
        let mut v = square(0., 10.);
        v.extend(square(4., 2.));
        v.extend([[20., 0., 5.], [30., 0., 5.], [40., 5., 5.], [50., 5., 5.]]);
        Polylines::new(
            v,
            vec![0, 4, 8, 10, 12],
            vec![0, 2, 2, 4],
            vec![true, true, false, false],
            rock(&["pit", "none", "section"]),
        )
        .unwrap()
    }

    #[test]
    fn accessors_follow_the_offsets() {
        let p = sample();
        assert_eq!((p.len(), p.num_parts()), (3, 4));
        assert_eq!(p.feature_parts(0), 0..2);
        assert!(p.feature_parts(1).is_empty());
        assert_eq!(p.part(3), &[[40., 5., 5.], [50., 5., 5.]]);
        assert_eq!(p.bounds(), Some(([0.; 3], [50., 10., 5.])));
    }

    #[test]
    fn bad_input_is_rejected() {
        let v = square(0., 1.);
        let new = |v: Vec<[f64; 3]>, parts: Vec<u32>, features: Vec<u32>, closed: Vec<bool>| {
            let rows = features.len().saturating_sub(1);
            Polylines::new(v, parts, features, closed, rock(&vec!["a"; rows]))
        };
        assert!(new(v.clone(), vec![0, 4], vec![0, 1], vec![true]).is_ok());
        assert!(new(v.clone(), vec![1, 4], vec![0, 1], vec![true]).is_err());
        assert!(new(v.clone(), vec![0, 3, 2, 4], vec![0, 3], vec![false; 3]).is_err());
        assert!(new(v.clone(), vec![0, 3], vec![0, 1], vec![true]).is_err());
        assert!(new(v.clone(), vec![0, 4], vec![0, 2], vec![true]).is_err());
        assert!(new(v.clone(), vec![0, 4], vec![], vec![true]).is_err());
        assert!(new(v.clone(), vec![0, 2, 4], vec![0, 2], vec![true, false]).is_err());
        assert!(new(v.clone(), vec![0, 3, 4], vec![0, 2], vec![false; 2]).is_err());
        assert!(new(v.clone(), vec![0, 4], vec![0, 1], vec![]).is_err());
        let mut nan = v.clone();
        nan[2][1] = f64::NAN;
        assert!(new(nan, vec![0, 4], vec![0, 1], vec![true]).is_err());
        assert!(Polylines::new(v, vec![0, 4], vec![0, 1], vec![true], rock(&["a", "b"])).is_err());
    }

    #[test]
    fn nested_table_follows_the_offsets() {
        let p = sample();
        let t = p.to_table().unwrap();
        assert_eq!(t.num_rows(), 3);
        let geometry = t["geometry"].as_list::<i32>();
        assert_eq!(geometry.value_offsets(), [0, 2, 2, 4]);
        let parts = geometry.values().as_list::<i32>();
        assert_eq!(parts.value_offsets(), [0, 4, 8, 10, 12]);
        let z = parts.values().as_struct()["z"].as_primitive::<Float64Type>();
        assert_eq!(z.value(11), 5.);
        let closed = t["closed"].as_list::<i32>();
        assert_eq!(closed.value_offsets(), [0, 2, 2, 4]);
        let back = Polylines::from_nested(&t.slice(1, 2)).unwrap();
        assert_eq!(back.closed(), [false, false]);
        assert_eq!(back.vertices(), &p.vertices()[8..]);
        assert_eq!(back.attributes(), &p.attributes().slice(1, 2));
    }

    #[test]
    fn from_table_groups_long_rows() {
        let p = sample().to_points().unwrap();
        let table = p.to_table().unwrap();
        let order: Vec<u32> = (0..12).rev().collect();
        let shuffled = take_rows(&table, order).unwrap();
        let back = Polylines::from_table(
            &shuffled,
            "feature",
            "x",
            "y",
            Some("z"),
            Some("part"),
            false,
        );
        let back = back.unwrap();
        assert_eq!((back.len(), back.num_parts()), (2, 4));
        assert_eq!(back.part(0), [[50., 5., 5.], [40., 5., 5.]]);
        let hole: Vec<_> = sample().part(1).iter().rev().copied().collect();
        assert_eq!(back.part(2), hole);
        assert_eq!(back.attributes().schema().field(1).name(), "feature");
        assert_eq!(back.attributes().num_columns(), 2);
        let one = Polylines::from_table(&table, "rock", "x", "y", None, None, false).unwrap();
        assert_eq!(
            (one.len(), one.part(0).len(), one.part(1)[0][2]),
            (2, 8, 0.)
        );
    }

    #[test]
    fn to_points_repeats_feature_attributes() {
        let mut p = sample();
        p.crs = Some("EPSG:31982".into());
        let points = p.to_points().unwrap();
        assert_eq!(points.coords(), p.vertices());
        assert_eq!(points.crs, p.crs);
        let t = points.attributes();
        let rock: Vec<_> = t["rock"].as_string::<i32>().iter().flatten().collect();
        assert_eq!(rock, [vec!["pit"; 8], vec!["section"; 4]].concat());
        let ids = |name: &str| t[name].as_primitive::<UInt32Type>().values().to_vec();
        assert_eq!(ids("feature"), [vec![0; 8], vec![2; 4]].concat());
        assert_eq!(ids("part"), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]);
        let clash = p.with_column("part", Arc::new(UInt32Array::from(vec![0; 3])));
        assert!(clash.unwrap().to_points().is_err());
    }
}
