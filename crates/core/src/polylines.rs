use std::ops::Range;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};

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
        let mut table = take_rows(&self.attributes, feature.clone())?;
        for (name, values) in [("feature", feature), ("part", part)] {
            if table.schema().index_of(name).is_ok() {
                return Err(Error::Geometry(format!("attribute `{name}` is reserved")));
            }
            table = crate::set_column(&table, name, Arc::new(UInt32Array::from(values)))?;
        }
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
