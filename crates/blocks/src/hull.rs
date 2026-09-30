use std::collections::HashSet;

use boitata_core::Mesh;
use nalgebra::Vector3;

use crate::{BlockModelError, Result};

/// Convex hull of `points` as a closed mesh with outward-facing triangles.
/// Errors when the points are fewer than four, coplanar or not finite.
pub fn convex_hull(points: &[[f64; 3]]) -> Result<Mesh> {
    let invalid = |m: &str| BlockModelError::InvalidMesh(m.to_string());
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("points must be finite"));
    }
    let p: Vec<Vector3<f64>> = points.iter().map(|&v| Vector3::from(v)).collect();
    let farthest = |score: &dyn Fn(&Vector3<f64>) -> f64| {
        (0..p.len()).max_by(|&i, &j| score(&p[i]).total_cmp(&score(&p[j])))
    };
    let a = farthest(&|v| -v.x).ok_or_else(|| invalid("no points"))?;
    let b = farthest(&|v| (v - p[a]).norm()).unwrap();
    let ab = p[b] - p[a];
    let scale = ab.norm();
    let eps = 1e-9 * scale;
    let c = farthest(&|v| ab.cross(&(v - p[a])).norm()).unwrap();
    let normal = ab.cross(&(p[c] - p[a]));
    let d = farthest(&|v| normal.dot(&(v - p[a])).abs()).unwrap();
    if normal.norm() <= eps * scale || normal.normalize().dot(&(p[d] - p[a])).abs() <= eps {
        return Err(invalid("points are collinear or coplanar"));
    }

    let plane = |f: &[usize; 3]| {
        let n = (p[f[1]] - p[f[0]]).cross(&(p[f[2]] - p[f[0]])).normalize();
        (n, n.dot(&p[f[0]]))
    };
    let mut faces: Vec<[usize; 3]> = vec![[a, b, c], [a, c, d], [a, d, b], [b, d, c]];
    let inside = (p[a] + p[b] + p[c] + p[d]) / 4.0;
    for f in &mut faces {
        let (n, o) = plane(f);
        if n.dot(&inside) > o {
            f.swap(1, 2);
        }
    }
    let mut planes: Vec<_> = faces.iter().map(plane).collect();

    for (i, q) in p.iter().enumerate() {
        let visible: Vec<bool> = planes.iter().map(|(n, o)| n.dot(q) - o > eps).collect();
        if !visible.contains(&true) {
            continue;
        }
        let edges: Vec<(usize, usize)> = faces
            .iter()
            .zip(&visible)
            .filter(|(_, v)| **v)
            .flat_map(|(f, _)| [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])
            .collect();
        let lookup: HashSet<_> = edges.iter().copied().collect();
        let mut kept: Vec<[usize; 3]> = faces
            .iter()
            .zip(&visible)
            .filter(|(_, v)| !**v)
            .map(|(f, _)| *f)
            .collect();
        kept.extend(
            edges
                .iter()
                .filter(|(u, v)| !lookup.contains(&(*v, *u)))
                .map(|&(u, v)| [u, v, i]),
        );
        faces = kept;
        planes = faces.iter().map(plane).collect();
    }

    let mut used: Vec<usize> = faces.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let vertices = used.iter().map(|&i| points[i]).collect();
    let triangles = faces
        .iter()
        .map(|f| f.map(|i| used.binary_search(&i).unwrap() as u32))
        .collect();
    Mesh::new(vertices, triangles).map_err(|e| BlockModelError::InvalidMesh(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_a_cube_with_inner_points_is_the_cube() {
        let mut points = vec![];
        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    points.push([x, y, z]);
                }
            }
        }
        points.extend([[0.5, 0.5, 0.5], [0.2, 0.7, 0.4], [0.5, 0.5, 1.0]]);
        let hull = convex_hull(&points).unwrap();
        assert!(hull.is_closed());
        assert_eq!(hull.vertices().len(), 8);
        assert!((hull.volume().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn every_point_is_inside_or_on_the_hull() {
        let points: Vec<[f64; 3]> = (0..500)
            .map(|i| {
                let t = i as f64;
                [
                    (t * 0.37).sin() * 10.0,
                    (t * 0.91).cos() * 5.0,
                    (t * 0.13).sin() * t / 50.0,
                ]
            })
            .collect();
        let hull = convex_hull(&points).unwrap();
        assert!(hull.is_closed());
        for t in 0..hull.triangles().len() {
            let [a, b, c] = hull.corners(t).map(Vector3::from);
            let n = (b - a).cross(&(c - a)).normalize();
            assert!(
                points
                    .iter()
                    .all(|q| n.dot(&(Vector3::from(*q) - a)) < 1e-9)
            );
        }
    }

    #[test]
    fn coplanar_points_are_rejected() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert!(convex_hull(&points).is_err());
    }
}
