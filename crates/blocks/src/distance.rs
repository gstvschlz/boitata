//! Distance from points (e.g. drillhole composites) to a triangle mesh or a 2-D
//! polygon.
//!
//! Complements the point-in-solid test ([`is_inside`](crate::is_inside)) with metric
//! proximity: how far a sample or block centre is from a wireframe surface /
//! solid, or from a domain outline. Useful for domain-margin flags, halo
//! selection, and distance-to-vein attributes.
//!
//! - Mesh distance uses the exact closest-point-on-triangle test (Ericson,
//!   *Real-Time Collision Detection*), minimized over all triangles.
//! - Signed distance is negative inside the solid (via the winding number).
//! - Polygon distance is the planar distance to the closed ring, signed
//!   negative inside (even-odd rule).

use crate::{BlockModelError, Result};
use ceres_core::Mesh;
use nalgebra::Vector3;

fn v(p: &(f64, f64, f64)) -> Vector3<f64> {
    Vector3::new(p.0, p.1, p.2)
}

/// Squared distance from point `p` to triangle `abc` (Ericson's region test).
fn point_triangle_dist2(p: Vector3<f64>, a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.norm_squared(); // vertex A
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.norm_squared(); // vertex B
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let w = d1 / (d1 - d3);
        return (ap - ab * w).norm_squared(); // edge AB
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.norm_squared(); // vertex C
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (ap - ac * w).norm_squared(); // edge AC
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).norm_squared(); // edge BC
    }
    // Interior: project onto the triangle plane via barycentric coords.
    let denom = 1.0 / (va + vb + vc);
    let bw = vb * denom;
    let cw = vc * denom;
    (p - (a + ab * bw + ac * cw)).norm_squared()
}

/// Unsigned distance from `point` to the nearest point on the mesh surface.
pub fn distance_to(mesh: &Mesh, point: &(f64, f64, f64)) -> Result<f64> {
    if mesh.triangles().is_empty() {
        return Err(BlockModelError::InvalidMesh("mesh has no triangles".into()));
    }
    let p = v(point);
    Ok((0..mesh.triangles().len())
        .map(|t| {
            let [a, b, c] = mesh.corners(t).map(Vector3::from);
            point_triangle_dist2(p, a, b, c)
        })
        .fold(f64::INFINITY, f64::min)
        .sqrt())
}

/// Signed distance to a closed mesh: negative inside, positive outside.
pub fn signed_distance_to(mesh: &Mesh, point: &(f64, f64, f64)) -> Result<f64> {
    let d = distance_to(mesh, point)?;
    Ok(if crate::is_inside(mesh, point)? {
        -d
    } else {
        d
    })
}

/// Distance from a 2-D point to a segment `[a, b]`.
fn point_segment_dist_2d(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let (apx, apy) = (p.0 - a.0, p.1 - a.1);
    let len2 = abx * abx + aby * aby;
    let t = if len2 <= 0.0 {
        0.0
    } else {
        ((apx * abx + apy * aby) / len2).clamp(0.0, 1.0)
    };
    let (cx, cy) = (a.0 + t * abx, a.1 + t * aby);
    ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt()
}

/// Even-odd point-in-polygon test for a closed ring `polygon` (last vertex need
/// not repeat the first).
pub fn point_in_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if ((yi > point.1) != (yj > point.1))
            && (point.0 < (xj - xi) * (point.1 - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Unsigned distance from a 2-D `point` to the boundary of a closed `polygon`.
pub fn polygon_distance(point: (f64, f64), polygon: &[(f64, f64)]) -> Result<f64> {
    let n = polygon.len();
    if n < 2 {
        return Err(BlockModelError::InvalidMesh(
            "polygon needs ≥ 2 vertices".into(),
        ));
    }
    let mut best = f64::INFINITY;
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        best = best.min(point_segment_dist_2d(point, a, b));
    }
    Ok(best)
}

/// Signed distance to a closed `polygon`: negative inside, positive outside.
pub fn polygon_signed_distance(point: (f64, f64), polygon: &[(f64, f64)]) -> Result<f64> {
    let d = polygon_distance(point, polygon)?;
    if point_in_polygon(point, polygon) {
        Ok(-d)
    } else {
        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit tetrahedron mesh (closed solid).
    fn tetra() -> Mesh {
        Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
        .unwrap()
    }

    #[test]
    fn distance_to_triangle_face() {
        let m = tetra();
        // Point straight below the base (z = -2) → distance 2 to the z=0 face.
        let d = distance_to(&m, &(0.2, 0.2, -2.0)).unwrap();
        assert!((d - 2.0).abs() < 1e-9, "distance {d}");
    }

    #[test]
    fn distance_to_vertex() {
        let m = tetra();
        // Far along +x from vertex (1,0,0).
        let d = distance_to(&m, &(4.0, 0.0, 0.0)).unwrap();
        assert!((d - 3.0).abs() < 1e-9, "distance {d}");
    }

    #[test]
    fn signed_distance_inside_negative() {
        let m = tetra();
        let inside = (0.1, 0.1, 0.1);
        let sd = signed_distance_to(&m, &inside).unwrap();
        assert!(sd < 0.0, "signed distance {sd} should be negative inside");
        let outside = signed_distance_to(&m, &(2.0, 2.0, 2.0)).unwrap();
        assert!(outside > 0.0);
    }

    #[test]
    fn polygon_distance_and_containment() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        // Outside, 3 to the left of the edge x=0.
        assert!((polygon_distance((-3.0, 5.0), &square).unwrap() - 3.0).abs() < 1e-9);
        assert!(point_in_polygon((5.0, 5.0), &square));
        assert!(!point_in_polygon((15.0, 5.0), &square));
        // Inside: signed distance negative, magnitude = distance to nearest edge (2).
        let sd = polygon_signed_distance((2.0, 5.0), &square).unwrap();
        assert!((sd + 2.0).abs() < 1e-9, "signed {sd}");
    }
}
