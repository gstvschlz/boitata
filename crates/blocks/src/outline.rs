//! Outline polygons around scattered points, in plan or on a dipping plane.

use std::collections::{BinaryHeap, HashMap};

use delaunator::{EMPTY, Point, Triangulation, next_halfedge, triangulate};
use i_overlay::mesh::float::outline::offset::OutlineOffset;
use i_overlay::mesh::float::style::{LineJoin, OutlineStyle};

use crate::{BlockModelError, Result};

fn invalid(m: impl Into<String>) -> BlockModelError {
    BlockModelError::Outline(m.into())
}

/// How [`outline`] bounds the points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hull {
    Convex,
    /// Concave: boundary edges longer than this are eroded inward.
    Concave(f64),
}

/// Plane the points are outlined on: plan, or `(azimuth, dip)` in degrees,
/// the azimuth the bearing along strike and the dip of the plane (90 is
/// vertical), as in the section plots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Plane {
    Plan,
    Dipping { azimuth: f64, dip: f64 },
}

impl Plane {
    /// Orthonormal in-plane axes.
    fn axes(self) -> ([f64; 3], [f64; 3]) {
        match self {
            Plane::Plan => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            Plane::Dipping { azimuth, dip } => {
                let (a, d) = (azimuth.to_radians(), dip.to_radians());
                (
                    [a.sin(), a.cos(), 0.0],
                    [-d.cos() * a.cos(), d.cos() * a.sin(), d.sin()],
                )
            }
        }
    }
}

/// Closed rings outlining `points`, with vertices on `plane` through the
/// points' centroid: one ring, or several where a negative `buffer` splits it.
/// Every point lies inside or on the outline before buffering. A concave hull
/// is a chi-shape: the longest boundary edge is eroded, one triangle at a
/// time, while it exceeds the threshold and the ring stays simple. `buffer`
/// offsets the ring outward with round joins, or inward when negative.
pub fn outline(
    points: &[[f64; 3]],
    hull: Hull,
    buffer: f64,
    plane: Plane,
) -> Result<Vec<Vec<[f64; 3]>>> {
    if points.iter().flatten().any(|v| !v.is_finite()) || !buffer.is_finite() {
        return Err(invalid("points and buffer must be finite"));
    }
    let n = points.len() as f64;
    let c = [0, 1, 2].map(|k| points.iter().map(|p| p[k]).sum::<f64>() / n);
    let (u, v) = plane.axes();
    let project = |p: &[f64; 3]| {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let dot = |a: [f64; 3]| a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
        [dot(u), dot(v)]
    };
    let flat: Vec<[f64; 2]> = points.iter().map(project).collect();
    let ring = match hull {
        Hull::Convex => convex(&flat)?,
        Hull::Concave(max_edge) => concave(&flat, max_edge)?,
    };
    let rings = if buffer == 0.0 {
        vec![ring]
    } else {
        offset(&ring, buffer)?
    };
    let lift = |q: &[f64; 2]| [0, 1, 2].map(|k| c[k] + q[0] * u[k] + q[1] * v[k]);
    Ok(rings.iter().map(|r| r.iter().map(lift).collect()).collect())
}

fn triangulation(points: &[[f64; 2]]) -> Result<Triangulation> {
    let pts: Vec<Point> = points.iter().map(|p| Point { x: p[0], y: p[1] }).collect();
    let t = triangulate(&pts);
    if t.triangles.is_empty() {
        return Err(invalid("an outline needs three points not on one line"));
    }
    Ok(t)
}

/// Counter-clockwise convex hull.
fn convex(points: &[[f64; 2]]) -> Result<Vec<[f64; 2]>> {
    let t = triangulation(points)?;
    Ok(counter_clockwise(
        t.hull.iter().map(|&i| points[i]).collect(),
    ))
}

#[derive(PartialEq)]
struct Edge(f64, usize);

impl Eq for Edge {}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(other.1.cmp(&self.1))
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Chi-shape (Duckham et al., 2008): a simple polygon holding every point.
fn concave(points: &[[f64; 2]], max_edge: f64) -> Result<Vec<[f64; 2]>> {
    if max_edge.is_nan() || max_edge <= 0.0 {
        return Err(invalid("max_edge must be positive"));
    }
    let t = triangulation(points)?;
    let (tri, twin) = (&t.triangles, &t.halfedges);
    let length = |e: usize| {
        let (a, b) = (points[tri[e]], points[tri[next_halfedge(e)]]);
        (a[0] - b[0]).hypot(a[1] - b[1])
    };
    let mut alive = vec![true; tri.len() / 3];
    let mut on_boundary = vec![false; points.len()];
    let mut heap = BinaryHeap::new();
    for e in 0..tri.len() {
        if twin[e] == EMPTY {
            on_boundary[tri[e]] = true;
            heap.push(Edge(length(e), e));
        }
    }
    let boundary = |e: usize, alive: &[bool]| twin[e] == EMPTY || !alive[twin[e] / 3];
    while let Some(Edge(len, e)) = heap.pop() {
        if len <= max_edge {
            break;
        }
        let opposite = tri[next_halfedge(next_halfedge(e))];
        if !alive[e / 3] || on_boundary[opposite] {
            continue;
        }
        alive[e / 3] = false;
        on_boundary[opposite] = true;
        for f in [next_halfedge(e), next_halfedge(next_halfedge(e))] {
            if twin[f] != EMPTY && alive[twin[f] / 3] {
                heap.push(Edge(length(twin[f]), twin[f]));
            }
        }
    }
    let next: HashMap<usize, usize> = (0..tri.len())
        .filter(|&e| alive[e / 3] && boundary(e, &alive))
        .map(|e| (tri[e], tri[next_halfedge(e)]))
        .collect();
    let start = *next.keys().min().expect("a triangle survives");
    let mut ring = vec![points[start]];
    let mut at = next[&start];
    while at != start {
        ring.push(points[at]);
        at = next[&at];
    }
    Ok(counter_clockwise(ring))
}

fn counter_clockwise(mut ring: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    let n = ring.len();
    let area: f64 = (0..n)
        .map(|i| ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1])
        .sum();
    if area < 0.0 {
        ring.reverse();
    }
    ring
}

/// `ring` offset by `distance` with round joins: outer rings and any holes.
fn offset(ring: &[[f64; 2]], distance: f64) -> Result<Vec<Vec<[f64; 2]>>> {
    let style = OutlineStyle::new(distance).line_join(LineJoin::Round(0.1));
    let rings: Vec<Vec<[f64; 2]>> = ring
        .outline_as::<i64>(&style)
        .into_iter()
        .flatten()
        .collect();
    if rings.is_empty() {
        return Err(invalid(format!(
            "a buffer of {distance} erases the outline"
        )));
    }
    Ok(rings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inside_or_on(ring: &[[f64; 2]], p: [f64; 2]) -> bool {
        let n = ring.len();
        let mut odd = false;
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
            let within =
                (p[0] - a[0]) * (p[0] - b[0]) <= 1e-9 && (p[1] - a[1]) * (p[1] - b[1]) <= 1e-9;
            if cross.abs() < 1e-9 && within {
                return true;
            }
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
            {
                odd = !odd;
            }
        }
        odd
    }

    /// An L of points on a 1 m grid: the convex hull spans the notch, the
    /// concave one follows it.
    fn ell() -> Vec<[f64; 3]> {
        let mut p = vec![];
        for i in 0..10 {
            for j in 0..10 {
                if i < 3 || j < 3 {
                    p.push([i as f64, j as f64, 100.0]);
                }
            }
        }
        p
    }

    fn area(ring: &[[f64; 3]]) -> f64 {
        let n = ring.len();
        (0..n)
            .map(|i| ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1])
            .sum::<f64>()
            / 2.0
    }

    #[test]
    fn every_point_lies_inside_or_on_the_outline() {
        let points = ell();
        for hull in [Hull::Convex, Hull::Concave(1.2)] {
            let rings = outline(&points, hull, 0.0, Plane::Plan).unwrap();
            assert_eq!(rings.len(), 1);
            let ring: Vec<[f64; 2]> = rings[0].iter().map(|p| [p[0], p[1]]).collect();
            assert!(
                points.iter().all(|p| inside_or_on(&ring, [p[0], p[1]])),
                "{hull:?}"
            );
            assert!(rings[0].iter().all(|p| (p[2] - 100.0).abs() < 1e-9));
        }
        let convex = area(&outline(&points, Hull::Convex, 0.0, Plane::Plan).unwrap()[0]);
        let concave = area(&outline(&points, Hull::Concave(1.2), 0.0, Plane::Plan).unwrap()[0]);
        assert!((convex - (81.0 - 0.5 * 7.0 * 7.0)).abs() < 1e-9, "{convex}");
        assert!((concave - (81.0 - 49.0)).abs() < 1e-9, "{concave}");
    }

    #[test]
    fn random_clouds_stay_inside_their_chi_shape() {
        use boitata_core::rng::splitmix;
        let unit = |k: u64| (splitmix(k) >> 11) as f64 / (1u64 << 53) as f64;
        for seed in 0..20u64 {
            let points: Vec<[f64; 3]> = (0..300)
                .map(|i| {
                    let (r, a) = (unit(seed * 1000 + 2 * i), unit(seed * 1000 + 2 * i + 1));
                    // A crescent: dense on one side, an empty bite on the other.
                    let t = std::f64::consts::TAU * a;
                    [
                        100.0 * r.sqrt() * t.cos() + 40.0 * t.cos(),
                        100.0 * r.sqrt() * t.sin(),
                        0.0,
                    ]
                })
                .collect();
            for max_edge in [5.0, 15.0, 40.0] {
                let ring: Vec<[f64; 2]> =
                    outline(&points, Hull::Concave(max_edge), 0.0, Plane::Plan).unwrap()[0]
                        .iter()
                        .map(|p| [p[0], p[1]])
                        .collect();
                assert!(points.iter().all(|p| inside_or_on(&ring, [p[0], p[1]])));
            }
        }
    }

    #[test]
    fn buffer_grows_and_shrinks_the_ring() {
        let square: Vec<[f64; 3]> = [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [5.0, 5.0],
        ]
        .iter()
        .map(|p| [p[0], p[1], 0.0])
        .collect();
        let grown = area(&outline(&square, Hull::Convex, 1.0, Plane::Plan).unwrap()[0]);
        assert!(
            (grown - (100.0 + 40.0 + std::f64::consts::PI)).abs() < 0.05,
            "{grown}"
        );
        let shrunk = area(&outline(&square, Hull::Convex, -1.0, Plane::Plan).unwrap()[0]);
        assert!((shrunk - 64.0).abs() < 1e-3, "{shrunk}");
        assert!(outline(&square, Hull::Convex, -6.0, Plane::Plan).is_err());
    }

    #[test]
    fn a_vertical_plane_outlines_in_section() {
        // Points on the vertical plane x = 5, striking north.
        let points: Vec<[f64; 3]> = [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]]
            .iter()
            .map(|p| [5.0, p[0], p[1]])
            .collect();
        let plane = Plane::Dipping {
            azimuth: 0.0,
            dip: 90.0,
        };
        let ring = &outline(&points, Hull::Convex, 0.0, plane).unwrap()[0];
        assert_eq!(ring.len(), 4);
        assert!(ring.iter().all(|p| (p[0] - 5.0).abs() < 1e-9));
        assert!(outline(&points[..2], Hull::Convex, 0.0, Plane::Plan).is_err());
    }
}
