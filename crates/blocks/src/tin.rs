//! Triangulated irregular networks: surfaces through scattered points.

use boitata_core::Mesh;
use delaunator::{EMPTY, Point, next_halfedge, triangulate};

use crate::{BlockModelError, Result, Surface};

fn plan(points: &[[f64; 3]]) -> Result<Vec<Point>> {
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(BlockModelError::InvalidMesh("points must be finite".into()));
    }
    Ok(points.iter().map(|p| Point { x: p[0], y: p[1] }).collect())
}

/// Delaunay triangulation of `points` in plan, lifted to their elevations,
/// facing up. Points sharing a plan location after the first are left out of
/// the triangles.
pub fn tin(points: &[[f64; 3]]) -> Result<Mesh> {
    let t = triangulate(&plan(points)?);
    if t.triangles.is_empty() {
        return Err(BlockModelError::InvalidMesh(
            "a surface needs three points not on one line in plan".into(),
        ));
    }
    let triangles = t
        .triangles
        .chunks(3)
        .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
        .collect();
    Ok(Mesh::new(points.to_vec(), triangles)?)
}

/// Each point's elevation minus the surface through the others at its plan
/// location: the leave-one-out residual of [`tin`]. The point's neighbors
/// are re-triangulated without it; a point on the hull, outside them, takes
/// their inverse-distance mean instead. A point left out of the triangles
/// as a plan duplicate is compared with the full surface.
pub fn tin_residuals(points: &[[f64; 3]]) -> Result<Vec<f64>> {
    let full = tin(points)?;
    let surface = Surface::new(&full)?;
    let t = triangulate(&plan(points)?);
    let mut ring: Vec<Vec<usize>> = vec![vec![]; points.len()];
    for e in 0..t.triangles.len() {
        let (a, b) = (t.triangles[e], t.triangles[next_halfedge(e)]);
        ring[a].push(b);
        if t.halfedges[e] == EMPTY {
            ring[b].push(a);
        }
    }
    Ok(points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if ring[i].is_empty() {
                return surface.elevation(p[0], p[1]).map_or(f64::NAN, |e| p[2] - e);
            }
            p[2] - without(points, &ring[i], p)
        })
        .collect())
}

/// Elevation at `p` from the surface through `around` alone.
fn without(points: &[[f64; 3]], around: &[usize], p: &[f64; 3]) -> f64 {
    let local: Vec<[f64; 3]> = around.iter().map(|&j| points[j]).collect();
    let inside = tin(&local)
        .ok()
        .and_then(|m| Surface::new(&m).ok())
        .and_then(|s| s.elevation(p[0], p[1]));
    inside.unwrap_or_else(|| extrapolate(&local, p))
}

/// Elevation at `p` from the plane through `local` fitted by least squares
/// with inverse-square-distance weights; their weighted mean where they lie
/// on a line.
fn extrapolate(local: &[[f64; 3]], p: &[f64; 3]) -> f64 {
    let w: Vec<f64> = local
        .iter()
        .map(|q| 1.0 / ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)))
        .collect();
    let (mut a, mut b) = (
        nalgebra::Matrix3::<f64>::zeros(),
        nalgebra::Vector3::<f64>::zeros(),
    );
    for (q, w) in local.iter().zip(&w) {
        let row = nalgebra::Vector3::new(1.0, q[0] - p[0], q[1] - p[1]);
        a += *w * row * row.transpose();
        b += *w * q[2] * row;
    }
    let mean = local.iter().zip(&w).map(|(q, w)| w * q[2]).sum::<f64>() / w.iter().sum::<f64>();
    match a.lu().solve(&b) {
        Some(x) if x[0].is_finite() && local.len() >= 3 => x[0],
        _ => mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(f: impl Fn(f64, f64) -> f64) -> Vec<[f64; 3]> {
        let mut p = vec![];
        for i in 0..8 {
            for j in 0..8 {
                // Jitter keeps the triangulation free of cocircular ties.
                let (x, y) = (
                    i as f64 * 10.0 + (j % 3) as f64,
                    j as f64 * 10.0 + (i % 2) as f64,
                );
                p.push([x, y, f(x, y)]);
            }
        }
        p
    }

    /// Theory check: the TIN is exact at the points, and a plane is
    /// reproduced everywhere, so leave-one-out residuals vanish inside.
    #[test]
    fn tin_honors_points_and_planes() {
        let plane = |x: f64, y: f64| 100.0 + 0.3 * x - 0.2 * y;
        let points = grid(plane);
        let mesh = tin(&points).unwrap();
        let s = Surface::new(&mesh).unwrap();
        for p in &points {
            assert!((s.elevation(p[0], p[1]).unwrap() - p[2]).abs() < 1e-9);
        }
        assert!((s.elevation(33.0, 41.0).unwrap() - plane(33.0, 41.0)).abs() < 1e-9);
        let r = tin_residuals(&points).unwrap();
        assert!(r.iter().all(|r| r.abs() < 1e-6), "{r:?}");
    }

    #[test]
    fn a_spike_has_the_largest_residual() {
        let mut points = grid(|x, y| 50.0 + 0.1 * x + 0.05 * y);
        points[27][2] += 30.0;
        let r = tin_residuals(&points).unwrap();
        let worst = (0..r.len())
            .max_by(|&a, &b| r[a].abs().total_cmp(&r[b].abs()))
            .unwrap();
        assert_eq!(worst, 27);
        assert!((r[27] - 30.0).abs() < 1e-9);
        let mut twin = points.clone();
        twin.push([points[27][0], points[27][1], points[27][2] - 30.0]);
        let r = tin_residuals(&twin).unwrap();
        assert!((r[64] + 30.0).abs() < 1e-9 && r.len() == 65);
        assert!(tin(&[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]).is_err());
    }
}
