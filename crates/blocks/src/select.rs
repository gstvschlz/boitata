//! Selection by imported closed strings (#349): a plan-view polygon test with
//! an optional between-two-RLs constraint.
//!
//! A "string" here is a closed polyline that came from a file (DXF, shapefile)
//! — there is deliberately no drawing tool. The test is 2-D: the
//! ring is projected to the XY (plan) plane and a point is inside when its
//! (x, y) falls inside an odd number of rings, so a ring inside another is a
//! hole, with z further constrained to the RL
//! window when one is set. This is the standard "inside the pit outline
//! between 7800 and 7850" query.

use boitata_core::Polylines;

use crate::distance::point_in_polygon;
use crate::error::{BlockModelError, Result};

/// How close the first and last vertex of an imported string must be, as a
/// fraction of the ring's own XY extent, for the ring to count as closed.
/// Relative rather than absolute so a district-scale string and a bench-scale
/// one judge closure at their own scale.
const CLOSURE_TOLERANCE_FRACTION: f64 = 1e-4;

/// Whether a polyline is closed in plan view: its first and last vertex
/// coincide (within a tolerance scaled to the ring's own extent). Rings whose
/// closure is implicit — polygon datasets never repeat the first vertex — are
/// closed by construction and skip this test.
pub fn ring_is_closed(vertices: &[[f64; 3]]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in vertices {
        min_x = min_x.min(v[0]);
        max_x = max_x.max(v[0]);
        min_y = min_y.min(v[1]);
        max_y = max_y.max(v[1]);
    }
    let extent = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
    if extent <= 0.0 {
        return false; // a degenerate (single-point) "ring" closes nothing
    }
    let first = vertices[0];
    let last = vertices[vertices.len() - 1];
    let gap = ((first[0] - last[0]).powi(2) + (first[1] - last[1]).powi(2)).sqrt();
    gap <= extent * CLOSURE_TOLERANCE_FRACTION
}

/// A plan-view polygon selector built from one or more closed rings, with an
/// optional RL (elevation) window. A point is selected when its XY projection
/// lies inside a feature, an odd number of its rings, and its z lies inside
/// the window.
#[derive(Debug, Clone)]
pub struct PolygonSelector {
    features: Vec<Vec<Vec<(f64, f64)>>>,
    lo: [f64; 2],
    hi: [f64; 2],
    /// Inclusive elevation window, when constrained.
    rl_min: Option<f64>,
    rl_max: Option<f64>,
}

impl PolygonSelector {
    /// Build a selector from rings given as 3-D polylines. `already_closed`
    /// says the rings close implicitly (a polygon dataset); otherwise each ring
    /// must explicitly return to its start in plan view, and a duplicated
    /// closing vertex is dropped before testing (the even-odd test expects the
    /// first vertex not to repeat).
    pub fn new(
        rings: &[&[[f64; 3]]],
        already_closed: bool,
        rl_min: Option<f64>,
        rl_max: Option<f64>,
    ) -> Result<Self> {
        let mut flat = Vec::new();
        for ring in rings {
            if ring.len() < 3 {
                continue; // too short to enclose anything
            }
            if !already_closed && !ring_is_closed(ring) {
                return Err(BlockModelError::InvalidGridParams(
                    "the string is not closed — its ends do not meet in plan view".into(),
                ));
            }
            let mut xy: Vec<(f64, f64)> = ring.iter().map(|v| (v[0], v[1])).collect();
            // Drop an explicit closing vertex; the test closes the ring itself.
            if xy.len() > 3 {
                let (first, last) = (xy[0], xy[xy.len() - 1]);
                if first == last {
                    xy.pop();
                }
            }
            flat.push(xy);
        }
        Self::with_features(vec![flat], rl_min, rl_max)
    }

    /// Build a selector from the closed parts of `lines`, one feature each.
    pub fn from_polylines(
        lines: &Polylines,
        rl_min: Option<f64>,
        rl_max: Option<f64>,
    ) -> Result<Self> {
        let features = (0..lines.len())
            .map(|f| {
                let ring = |r| lines.part(r).iter().map(|v| (v[0], v[1])).collect();
                lines.rings(f).map(ring).collect()
            })
            .collect();
        Self::with_features(features, rl_min, rl_max)
    }

    fn with_features(
        features: Vec<Vec<Vec<(f64, f64)>>>,
        rl_min: Option<f64>,
        rl_max: Option<f64>,
    ) -> Result<Self> {
        if let (Some(lo), Some(hi)) = (rl_min, rl_max)
            && lo > hi
        {
            return Err(BlockModelError::InvalidGridParams(format!(
                "RL window is inverted ({lo} > {hi})"
            )));
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for &(x, y) in features.iter().flatten().flatten() {
            lo = [lo[0].min(x), lo[1].min(y)];
            hi = [hi[0].max(x), hi[1].max(y)];
        }
        if lo[0] > hi[0] {
            return Err(BlockModelError::InvalidGridParams(
                "no ring with three or more vertices to select with".into(),
            ));
        }
        Ok(Self {
            features,
            lo,
            hi,
            rl_min,
            rl_max,
        })
    }

    /// `(min, max)` plan corners of the rings.
    pub fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        (self.lo, self.hi)
    }

    /// Whether `point` is inside the selection volume: in plan inside a
    /// feature, and inside the RL window when one is set.
    pub fn contains(&self, point: [f64; 3]) -> bool {
        if let Some(lo) = self.rl_min
            && point[2] < lo
        {
            return false;
        }
        if let Some(hi) = self.rl_max
            && point[2] > hi
        {
            return false;
        }
        let xy = (point[0], point[1]);
        self.features
            .iter()
            .any(|rings| rings.iter().filter(|r| point_in_polygon(xy, r)).count() % 2 == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed unit-ish square ring at elevation `z`, explicitly returning to
    /// its start.
    fn square(lo: f64, hi: f64, z: f64) -> Vec<[f64; 3]> {
        vec![
            [lo, lo, z],
            [hi, lo, z],
            [hi, hi, z],
            [lo, hi, z],
            [lo, lo, z],
        ]
    }

    #[test]
    fn selects_inside_the_ring_in_plan_view() {
        let ring = square(0.0, 10.0, 0.0);
        let sel = PolygonSelector::new(&[&ring], false, None, None).expect("selector");
        assert!(sel.contains([5.0, 5.0, 123.0])); // z unconstrained
        assert!(!sel.contains([15.0, 5.0, 0.0]));
        assert!(!sel.contains([-1.0, 5.0, 0.0]));
    }

    #[test]
    fn rl_window_constrains_elevation() {
        let ring = square(0.0, 10.0, 0.0);
        let sel =
            PolygonSelector::new(&[&ring], false, Some(100.0), Some(200.0)).expect("selector");
        assert!(sel.contains([5.0, 5.0, 150.0]));
        assert!(sel.contains([5.0, 5.0, 100.0]), "window is inclusive");
        assert!(sel.contains([5.0, 5.0, 200.0]), "window is inclusive");
        assert!(!sel.contains([5.0, 5.0, 99.9]));
        assert!(!sel.contains([5.0, 5.0, 200.1]));
        // Still outside in plan even at a valid RL.
        assert!(!sel.contains([50.0, 5.0, 150.0]));
    }

    #[test]
    fn one_sided_rl_windows_work() {
        let ring = square(0.0, 10.0, 0.0);
        let above = PolygonSelector::new(&[&ring], false, Some(100.0), None).expect("selector");
        assert!(above.contains([5.0, 5.0, 5000.0]));
        assert!(!above.contains([5.0, 5.0, 0.0]));
        let below = PolygonSelector::new(&[&ring], false, None, Some(100.0)).expect("selector");
        assert!(below.contains([5.0, 5.0, -5000.0]));
        assert!(!below.contains([5.0, 5.0, 101.0]));
    }

    #[test]
    fn inverted_rl_window_is_rejected() {
        let ring = square(0.0, 10.0, 0.0);
        let err = PolygonSelector::new(&[&ring], false, Some(200.0), Some(100.0))
            .expect_err("inverted window");
        assert!(err.to_string().contains("inverted"), "got: {err}");
    }

    #[test]
    fn an_open_string_is_rejected() {
        // Ends 5 units apart on a 10-unit ring — clearly open.
        let open: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 5.0, 0.0],
        ];
        let err = PolygonSelector::new(&[&open], false, None, None).expect_err("open string");
        assert!(err.to_string().contains("not closed"), "got: {err}");
    }

    #[test]
    fn polygon_dataset_rings_close_implicitly() {
        // A polygon dataset never repeats the first vertex; `already_closed`
        // lets it through and the test still works.
        let ring: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ];
        let sel = PolygonSelector::new(&[&ring], true, None, None).expect("selector");
        assert!(sel.contains([5.0, 5.0, 0.0]));
        assert!(!sel.contains([11.0, 5.0, 0.0]));
    }

    #[test]
    fn a_concave_ring_selects_correctly() {
        // An L-shape: the notch (7.5, 7.5) is outside even though it is inside
        // the bounding box.
        let ring: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
            [5.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let sel = PolygonSelector::new(&[&ring], false, None, None).expect("selector");
        assert!(sel.contains([2.0, 2.0, 0.0]));
        assert!(sel.contains([2.0, 8.0, 0.0]));
        assert!(!sel.contains([7.5, 7.5, 0.0]), "the notch is outside");
    }

    #[test]
    fn any_of_several_rings_selects() {
        let a = square(0.0, 10.0, 0.0);
        let b = square(100.0, 110.0, 0.0);
        let sel = PolygonSelector::new(&[&a, &b], false, None, None).expect("selector");
        assert!(sel.contains([5.0, 5.0, 0.0]));
        assert!(sel.contains([105.0, 105.0, 0.0]));
        assert!(!sel.contains([50.0, 50.0, 0.0]));
    }

    /// Theory check: an outer ring with a ring inside it selects nothing in
    /// the hole, from raw rings as from a two-part feature, and the same
    /// points come out every time.
    #[test]
    fn a_ring_inside_another_is_a_hole() {
        let (outer, inner) = (square(0.0, 10.0, 0.0), square(4.0, 6.0, 0.0));
        let raw = PolygonSelector::new(&[&outer, &inner], false, None, None).expect("selector");
        let mut vertices = outer[..4].to_vec();
        vertices.extend(&inner[..4]);
        let id: arrow_array::ArrayRef = std::sync::Arc::new(arrow_array::Int32Array::from(vec![0]));
        let table = arrow_array::RecordBatch::try_from_iter([("id", id)]).unwrap();
        let lines = Polylines::new(vertices, vec![0, 4, 8], vec![0, 2], vec![true; 2], table);
        let lines = PolygonSelector::from_polylines(&lines.unwrap(), None, None).unwrap();
        let grid: Vec<[f64; 3]> = (0..121)
            .map(|k| [(k % 11) as f64 + 0.5, (k / 11) as f64 + 0.5, 0.0])
            .collect();
        let picked = |s: &PolygonSelector| grid.iter().map(|&p| s.contains(p)).collect::<Vec<_>>();
        assert_eq!(picked(&raw), picked(&lines));
        assert_eq!(picked(&raw), picked(&raw));
        assert_eq!(picked(&raw).iter().filter(|&&b| b).count(), 100 - 4);
        assert!(raw.contains([2.0, 2.0, 0.0]));
        assert!(!raw.contains([5.0, 5.0, 0.0]));
    }

    #[test]
    fn closure_tolerance_scales_with_the_ring() {
        // A 10 km ring whose ends are 0.5 m apart — surveyed data, closed for
        // any practical purpose.
        let ring: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [10_000.0, 0.0, 0.0],
            [10_000.0, 10_000.0, 0.0],
            [0.0, 10_000.0, 0.0],
            [0.0, 0.5, 0.0],
        ];
        assert!(ring_is_closed(&ring));
        // The same half-meter gap on a 10 m ring is genuinely open.
        let small: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.5, 0.0],
        ];
        assert!(!ring_is_closed(&small));
    }
}
