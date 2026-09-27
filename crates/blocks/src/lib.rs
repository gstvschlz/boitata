//! Domain assignment, point-in-solid testing, polygon selection and block shells.
//!
//! Domain assignment with multiple strategies,
//! and point-in-solid testing using generalized winding number for robust mesh handling.

mod classes;
mod distance;
mod error;
mod grid;
mod hull;
mod select;
mod shell;
mod solid;
mod subblock;
pub use classes::smooth_classes;
pub use distance::{
    Surface, distance_to, point_in_polygon, polygon_distance, polygon_signed_distance,
    signed_distance_to, vertical_distance,
};
pub use error::{BlockModelError, Result};
pub use grid::grid_surface;
pub use hull::convex_hull;
pub use select::{PolygonSelector, ring_is_closed};
pub use shell::{
    BlockSubset, Orientation, ShellBlock, ShellFilter, ShellLimits, ShellMesh, Slab,
    estimate_shell_faces, extract_shell, infer_orientation,
};
pub use solid::{Aabb, BlockDomainRule, BlockSolid, SolidTester};
pub use subblock::{Domain, Region, proportions, subblock};

use ceres_core::{Mesh, signed_solid_angle};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Domain assignment method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainMethod {
    Nearest,
    MajorityVote,
    PointInSolid,
}

/// Domain assignment result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAssignment {
    pub domain: String,
    pub confidence: f64,
}

/// Assign domain to block.
pub fn assign_domain(
    block_center: &(f64, f64, f64),
    samples: &[((f64, f64, f64), String)],
    method: DomainMethod,
    mesh: Option<&Mesh>,
) -> Result<DomainAssignment> {
    match method {
        DomainMethod::Nearest => assign_nearest(block_center, samples),
        DomainMethod::MajorityVote => assign_majority_vote(block_center, samples),
        DomainMethod::PointInSolid => {
            if let Some(m) = mesh {
                assign_point_in_solid(block_center, m)
            } else {
                Err(BlockModelError::DomainAssignmentFailed(
                    "Wireframe required for point-in-solid".to_string(),
                ))
            }
        }
    }
}

/// Assign to nearest sample's domain.
fn assign_nearest(
    block_center: &(f64, f64, f64),
    samples: &[((f64, f64, f64), String)],
) -> Result<DomainAssignment> {
    if samples.is_empty() {
        return Err(BlockModelError::DomainAssignmentFailed(
            "No samples available".to_string(),
        ));
    }

    let mut nearest = 0;
    let mut min_dist = f64::INFINITY;

    for (i, (loc, _)) in samples.iter().enumerate() {
        let dx = block_center.0 - loc.0;
        let dy = block_center.1 - loc.1;
        let dz = block_center.2 - loc.2;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist < min_dist {
            min_dist = dist;
            nearest = i;
        }
    }

    Ok(DomainAssignment {
        domain: samples[nearest].1.clone(),
        confidence: 1.0,
    })
}

/// Assign by majority vote among K nearest samples.
fn assign_majority_vote(
    block_center: &(f64, f64, f64),
    samples: &[((f64, f64, f64), String)],
) -> Result<DomainAssignment> {
    if samples.is_empty() {
        return Err(BlockModelError::DomainAssignmentFailed(
            "No samples available".to_string(),
        ));
    }

    let k = 5.min(samples.len());

    let mut distances: Vec<(usize, f64)> = samples
        .iter()
        .enumerate()
        .map(|(i, (loc, _))| {
            let dx = block_center.0 - loc.0;
            let dy = block_center.1 - loc.1;
            let dz = block_center.2 - loc.2;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            (i, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut votes: HashMap<String, usize> = HashMap::new();
    for (i, _) in distances.iter().take(k) {
        let domain = &samples[*i].1;
        *votes.entry(domain.clone()).or_insert(0) += 1;
    }

    let (domain, count) = votes
        .iter()
        .max_by_key(|&(_, &v)| v)
        .ok_or_else(|| BlockModelError::DomainAssignmentFailed("Voting failed".to_string()))?;

    Ok(DomainAssignment {
        domain: domain.clone(),
        confidence: *count as f64 / k as f64,
    })
}

/// Assign using point-in-solid test (generalized winding number).
fn assign_point_in_solid(block_center: &(f64, f64, f64), mesh: &Mesh) -> Result<DomainAssignment> {
    let inside = is_inside(mesh, block_center)?;

    Ok(DomainAssignment {
        domain: if inside {
            "inside".to_string()
        } else {
            "outside".to_string()
        },
        confidence: 1.0,
    })
}

/// Error unless `mesh` is closed.
pub fn require_closed(mesh: &Mesh) -> Result<()> {
    if mesh.is_closed() {
        Ok(())
    } else {
        Err(BlockModelError::InvalidMesh(
            "mesh is not closed; this needs a solid".to_string(),
        ))
    }
}

/// Generalized winding number: the signed solid angles the triangles subtend
/// at `point`, over 4π. ±1 inside a closed solid, 0 outside, fractional near
/// an open surface.
pub fn winding_number(mesh: &Mesh, point: &(f64, f64, f64)) -> f64 {
    let p = Vector3::new(point.0, point.1, point.2);
    let total: f64 = (0..mesh.triangles().len())
        .map(|t| {
            let [a, b, c] = mesh.corners(t).map(|v| Vector3::from(v) - p);
            signed_solid_angle(&a, &b, &c)
        })
        .sum();
    total / (4.0 * std::f64::consts::PI)
}

/// Whether `point` is inside a closed mesh. For more than a handful of points
/// build a [`SolidTester`] instead.
pub fn is_inside(mesh: &Mesh, point: &(f64, f64, f64)) -> Result<bool> {
    require_closed(mesh)?;
    Ok(winding_number(mesh, point).abs() > 0.5)
}

/// Inside test on a summed winding angle (radians, not yet divided by 4π).
///
/// Compares the *magnitude*: a mesh whose triangles wind inward — common in
/// imported OBJ/DXF solids, where orientation is whatever the exporter felt
/// like — sums to -4π rather than +4π, and is no less inside for it.
fn is_inside_winding(winding: f64) -> bool {
    (winding / (4.0 * std::f64::consts::PI)).abs() > 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_domain_assignment() {
        let samples = vec![
            ((0.0, 0.0, 0.0), "granite".to_string()),
            ((100.0, 0.0, 0.0), "basalt".to_string()),
        ];

        let block = (5.0, 5.0, 0.0);
        let result = assign_domain(&block, &samples, DomainMethod::Nearest, None);

        assert!(result.is_ok());
        let assignment = result.unwrap();
        assert_eq!(assignment.domain, "granite");
    }

    fn unit_cube() -> Mesh {
        solid::tests::cube(0.0, 1.0)
    }

    #[test]
    fn test_open_mesh_is_not_a_solid() {
        let square = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
        )
        .unwrap();
        assert!(is_inside(&square, &(0.1, 0.1, 0.1)).is_err());
        assert!(winding_number(&square, &(0.1, 0.1, 0.1)).abs() > 0.0);
    }

    #[test]
    fn test_is_inside_cube() {
        let mesh = unit_cube();
        assert!(is_inside(&mesh, &(0.5, 0.5, 0.5)).expect("valid mesh"));
        assert!(!is_inside(&mesh, &(5.0, 0.5, 0.5)).expect("valid mesh"));
    }

    /// Regression: summing *unsigned* solid angles pushes the total towards 2π
    /// just outside a face, so a shell of outside points read as inside. The
    /// sign is what makes the sum a winding number.
    #[test]
    fn test_is_inside_has_no_halo_just_outside_a_face() {
        let mesh = unit_cube();
        assert!(!is_inside(&mesh, &(0.5, 0.5, 1.05)).expect("valid mesh"));
        assert!(!is_inside(&mesh, &(0.5, 0.5, -0.05)).expect("valid mesh"));
        assert!(!is_inside(&mesh, &(-0.05, 0.5, 0.5)).expect("valid mesh"));
    }

    /// point_in_solid domain assignment rides on the same test.
    #[test]
    fn test_point_in_solid_domain_assignment() {
        let mesh = unit_cube();
        let inside = assign_domain(
            &(0.5, 0.5, 0.5),
            &[],
            DomainMethod::PointInSolid,
            Some(&mesh),
        )
        .expect("assigns");
        assert_eq!(inside.domain, "inside");
        let outside = assign_domain(
            &(0.5, 0.5, 2.0),
            &[],
            DomainMethod::PointInSolid,
            Some(&mesh),
        )
        .expect("assigns");
        assert_eq!(outside.domain, "outside");
    }
}
