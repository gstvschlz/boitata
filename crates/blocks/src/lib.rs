//! Block model management: grids, domain assignment, and point-in-solid testing.
//!
//! Handles regular rotated block grids, domain assignment with multiple strategies,
//! and point-in-solid testing using generalized winding number for robust mesh handling.

mod distance;
mod error;
mod select;
mod shell;
mod solid;
pub use distance::{point_in_polygon, polygon_distance, polygon_signed_distance};
pub use error::{BlockModelError, Result};
pub use select::{PolygonSelector, ring_is_closed};
pub use shell::{
    BlockSubset, Orientation, ShellBlock, ShellFilter, ShellLimits, ShellMesh, Slab,
    estimate_shell_faces, extract_shell, infer_orientation,
};
pub use solid::{Aabb, BlockDomainRule, BlockSolid, SolidTester};

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Block grid parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockGridParams {
    /// Origin (east, north, elev)
    pub origin: (f64, f64, f64),
    /// Block dimensions (m)
    pub block_size: (f64, f64, f64),
    /// Grid extent (nx, ny, nz blocks)
    pub grid_extent: (usize, usize, usize),
    /// Rotation angles in degrees (azimuth, dip, pitch)
    pub rotation: Option<(f64, f64, f64)>,
}

impl BlockGridParams {
    pub fn validate(&self) -> Result<()> {
        if self.block_size.0 <= 0.0 || self.block_size.1 <= 0.0 || self.block_size.2 <= 0.0 {
            return Err(BlockModelError::InvalidGridParams(
                "Block dimensions must be positive".to_string(),
            ));
        }

        if self.grid_extent.0 == 0 || self.grid_extent.1 == 0 || self.grid_extent.2 == 0 {
            return Err(BlockModelError::InvalidGridParams(
                "Grid extent must be positive".to_string(),
            ));
        }

        Ok(())
    }

    pub fn total_blocks(&self) -> usize {
        self.grid_extent.0 * self.grid_extent.1 * self.grid_extent.2
    }
}

/// Regular block model grid.
pub struct BlockGrid {
    pub params: BlockGridParams,
    rotation_matrix: Option<Matrix3<f64>>,
}

impl BlockGrid {
    /// Create a new block grid.
    pub fn new(params: BlockGridParams) -> Result<Self> {
        params.validate()?;

        let rotation_matrix = params.rotation.map(|(az, dip, pitch)| {
            compute_rotation_matrix(az.to_radians(), dip.to_radians(), pitch.to_radians())
        });

        Ok(BlockGrid {
            params,
            rotation_matrix,
        })
    }

    /// Get block center location (east, north, elev).
    pub fn block_center(&self, ix: usize, iy: usize, iz: usize) -> Result<(f64, f64, f64)> {
        if ix >= self.params.grid_extent.0
            || iy >= self.params.grid_extent.1
            || iz >= self.params.grid_extent.2
        {
            return Err(BlockModelError::InvalidGridParams(
                "Block indices out of bounds".to_string(),
            ));
        }

        let mut center = Vector3::new(
            self.params.origin.0 + (ix as f64 + 0.5) * self.params.block_size.0,
            self.params.origin.1 + (iy as f64 + 0.5) * self.params.block_size.1,
            self.params.origin.2 + (iz as f64 + 0.5) * self.params.block_size.2,
        );

        if let Some(rot) = &self.rotation_matrix {
            center = rot * center;
        }

        Ok((center.x, center.y, center.z))
    }

    /// Get all block centers.
    pub fn all_block_centers(&self) -> Result<Vec<(f64, f64, f64)>> {
        let mut centers = vec![];

        for ix in 0..self.params.grid_extent.0 {
            for iy in 0..self.params.grid_extent.1 {
                for iz in 0..self.params.grid_extent.2 {
                    centers.push(self.block_center(ix, iy, iz)?);
                }
            }
        }

        Ok(centers)
    }

    /// Get block corners (min and max for axis-aligned, or 8 corners).
    pub fn block_bounds(
        &self,
        ix: usize,
        iy: usize,
        iz: usize,
    ) -> Result<((f64, f64, f64), (f64, f64, f64))> {
        let center = self.block_center(ix, iy, iz)?;
        let half_size = (
            self.params.block_size.0 / 2.0,
            self.params.block_size.1 / 2.0,
            self.params.block_size.2 / 2.0,
        );

        Ok((
            (
                center.0 - half_size.0,
                center.1 - half_size.1,
                center.2 - half_size.2,
            ),
            (
                center.0 + half_size.0,
                center.1 + half_size.1,
                center.2 + half_size.2,
            ),
        ))
    }
}

/// Compute rotation matrix from Euler angles (same as variography).
fn compute_rotation_matrix(azimuth: f64, dip: f64, pitch: f64) -> Matrix3<f64> {
    let ca = azimuth.cos();
    let sa = azimuth.sin();
    let r1 = Matrix3::new(ca, sa, 0.0, -sa, ca, 0.0, 0.0, 0.0, 1.0);

    let cd = dip.cos();
    let sd = dip.sin();
    let r2 = Matrix3::new(cd, 0.0, -sd, 0.0, 1.0, 0.0, sd, 0.0, cd);

    let cp = pitch.cos();
    let sp = pitch.sin();
    let r3 = Matrix3::new(1.0, 0.0, 0.0, 0.0, cp, sp, 0.0, -sp, cp);

    r3 * r2 * r1
}

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
    let inside = mesh.is_inside(block_center)?;

    Ok(DomainAssignment {
        domain: if inside {
            "inside".to_string()
        } else {
            "outside".to_string()
        },
        confidence: 1.0,
    })
}

/// Triangle mesh for point-in-solid testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: Vec<(f64, f64, f64)>,
    pub triangles: Vec<(usize, usize, usize)>,
}

impl Mesh {
    /// Validate mesh.
    pub fn validate(&self) -> Result<()> {
        if self.vertices.is_empty() {
            return Err(BlockModelError::InvalidMesh("No vertices".to_string()));
        }

        if self.triangles.is_empty() {
            return Err(BlockModelError::InvalidMesh("No triangles".to_string()));
        }

        for (v0, v1, v2) in &self.triangles {
            if *v0 >= self.vertices.len()
                || *v1 >= self.vertices.len()
                || *v2 >= self.vertices.len()
            {
                return Err(BlockModelError::InvalidMesh(
                    "Triangle index out of bounds".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Test if point is inside mesh using the generalized winding number.
    ///
    /// # Algorithm
    /// Sum the *signed* solid angle each triangle subtends at the point and
    /// divide by 4π. For a closed mesh the signed angles cancel to 0 outside
    /// the solid and sum to ±4π inside, so |winding| > 0.5 means inside.
    ///
    /// This is robust to non-watertight meshes unlike ray casting.
    ///
    /// Validates on every call, and re-reads the vertex list per triangle. For
    /// more than a handful of points build a [`SolidTester`] instead, which
    /// does both once up front.
    pub fn is_inside(&self, point: &(f64, f64, f64)) -> Result<bool> {
        self.validate()?;

        let p = Vector3::new(point.0, point.1, point.2);
        let mut winding = 0.0;

        for (v0_idx, v1_idx, v2_idx) in &self.triangles {
            let v0 = Vector3::new(
                self.vertices[*v0_idx].0,
                self.vertices[*v0_idx].1,
                self.vertices[*v0_idx].2,
            );
            let v1 = Vector3::new(
                self.vertices[*v1_idx].0,
                self.vertices[*v1_idx].1,
                self.vertices[*v1_idx].2,
            );
            let v2 = Vector3::new(
                self.vertices[*v2_idx].0,
                self.vertices[*v2_idx].1,
                self.vertices[*v2_idx].2,
            );

            winding += signed_solid_angle(&(v0 - p), &(v1 - p), &(v2 - p));
        }

        Ok(is_inside_winding(winding))
    }
}

/// Signed solid angle subtended by a triangle at the origin (van Oosterom &
/// Strackee, IEEE Trans. Biomed. Eng.).
///
/// The *sign* is what makes a sum of these a winding number — it encodes which
/// face of the triangle the origin sees. Taking `.abs()` of the numerator
/// instead turns every contribution positive, so the sum approaches 2π just
/// outside any face and the point reads as inside: a false-"inside" halo around
/// the whole surface.
///
/// `atan2`, not `atan(num / denom)`: the denominator goes negative for solid
/// angles past π, and only atan2 puts those in the right quadrant.
fn signed_solid_angle(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> f64 {
    let num = a.dot(&(b.cross(c)));
    let denom = a.norm() * b.norm() * c.norm()
        + a.dot(b) * c.norm()
        + b.dot(c) * a.norm()
        + c.dot(a) * b.norm();

    // atan2(0, 0) is 0, which is the right contribution for a degenerate
    // triangle or a point sitting exactly on a vertex.
    2.0 * num.atan2(denom)
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
    fn test_block_grid_creation() {
        let params = BlockGridParams {
            origin: (0.0, 0.0, 0.0),
            block_size: (10.0, 10.0, 10.0),
            grid_extent: (5, 5, 5),
            rotation: None,
        };

        let grid = BlockGrid::new(params).unwrap();
        assert_eq!(grid.params.total_blocks(), 125);
    }

    #[test]
    fn test_block_center() {
        let params = BlockGridParams {
            origin: (0.0, 0.0, 0.0),
            block_size: (10.0, 10.0, 10.0),
            grid_extent: (3, 3, 3),
            rotation: None,
        };

        let grid = BlockGrid::new(params).unwrap();
        let center = grid.block_center(0, 0, 0).unwrap();

        assert!((center.0 - 5.0).abs() < 0.01);
        assert!((center.1 - 5.0).abs() < 0.01);
        assert!((center.2 - 5.0).abs() < 0.01);
    }

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

    #[test]
    fn test_mesh_validation() {
        let mesh = Mesh {
            vertices: vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            triangles: vec![(0, 1, 2)],
        };

        assert!(mesh.validate().is_ok());
    }

    /// Unit cube at the origin, triangles wound counter-clockwise from outside.
    fn unit_cube() -> Mesh {
        Mesh {
            vertices: vec![
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            ],
            triangles: vec![
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (2, 3, 7),
                (2, 7, 6),
                (0, 4, 7),
                (0, 7, 3),
                (1, 2, 6),
                (1, 6, 5),
            ],
        }
    }

    #[test]
    fn test_is_inside_cube() {
        let mesh = unit_cube();
        assert!(mesh.is_inside(&(0.5, 0.5, 0.5)).expect("valid mesh"));
        assert!(!mesh.is_inside(&(5.0, 0.5, 0.5)).expect("valid mesh"));
    }

    /// Regression: summing *unsigned* solid angles pushes the total towards 2π
    /// just outside a face, so a shell of outside points read as inside. The
    /// sign is what makes the sum a winding number.
    #[test]
    fn test_is_inside_has_no_halo_just_outside_a_face() {
        let mesh = unit_cube();
        assert!(!mesh.is_inside(&(0.5, 0.5, 1.05)).expect("valid mesh"));
        assert!(!mesh.is_inside(&(0.5, 0.5, -0.05)).expect("valid mesh"));
        assert!(!mesh.is_inside(&(-0.05, 0.5, 0.5)).expect("valid mesh"));
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

    #[test]
    fn test_mesh_invalid_index() {
        let mesh = Mesh {
            vertices: vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            triangles: vec![(0, 1, 999)],
        };

        assert!(mesh.validate().is_err());
    }
}
