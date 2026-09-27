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
    Surface, TriangleTree, distance_to, point_in_polygon, polygon_distance,
    polygon_signed_distance, signed_distance_to, vertical_distance,
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
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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

/// Domain of each target.
///
/// `Nearest` takes the label of the nearest sample; `MajorityVote` the most
/// frequent label among the `n` nearest samples, ties going to the label of
/// the nearest sample. Samples at equal distance rank by label. For both,
/// confidence is the share of the `n` nearest samples carrying the chosen
/// label. `PointInSolid` gives `"inside"` or `"outside"` of the closed `mesh`,
/// with confidence 1.
pub fn assign_domain(
    targets: &[(f64, f64, f64)],
    samples: &[((f64, f64, f64), String)],
    method: DomainMethod,
    n: usize,
    mesh: Option<&Mesh>,
) -> Result<Vec<DomainAssignment>> {
    match method {
        DomainMethod::PointInSolid => {
            let mesh = mesh.ok_or_else(|| {
                BlockModelError::DomainAssignmentFailed(
                    "Wireframe required for point-in-solid".to_string(),
                )
            })?;
            require_closed(mesh)?;
            Ok(targets
                .par_iter()
                .map(|t| assign_point_in_solid(t, mesh))
                .collect())
        }
        _ if samples.is_empty() => Err(BlockModelError::DomainAssignmentFailed(
            "No samples available".to_string(),
        )),
        _ if n == 0 => Err(BlockModelError::DomainAssignmentFailed(
            "n must be at least 1".to_string(),
        )),
        _ => Ok(targets
            .par_iter()
            .map(|t| assign_from_samples(t, samples, method, n))
            .collect()),
    }
}

fn assign_from_samples(
    target: &(f64, f64, f64),
    samples: &[((f64, f64, f64), String)],
    method: DomainMethod,
    n: usize,
) -> DomainAssignment {
    let k = n.min(samples.len());
    let mut ranked: Vec<(f64, &str)> = samples
        .iter()
        .map(|(p, label)| {
            let d = (target.0 - p.0).powi(2) + (target.1 - p.1).powi(2) + (target.2 - p.2).powi(2);
            (d, label.as_str())
        })
        .collect();
    let order = |a: &(f64, &str), b: &(f64, &str)| a.0.total_cmp(&b.0).then(a.1.cmp(b.1));
    if k < ranked.len() {
        ranked.select_nth_unstable_by(k - 1, order);
        ranked.truncate(k);
    }
    ranked.sort_unstable_by(order);
    // label -> (votes, rank of its nearest sample)
    let mut votes: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (rank, (_, label)) in ranked.iter().enumerate() {
        votes.entry(label).or_insert((0, rank)).0 += 1;
    }
    let (domain, (count, _)) = match method {
        DomainMethod::MajorityVote => votes
            .into_iter()
            .min_by_key(|&(_, (count, rank))| (std::cmp::Reverse(count), rank))
            .expect("at least one sample"),
        _ => (ranked[0].1, votes[ranked[0].1]),
    };
    DomainAssignment {
        domain: domain.to_string(),
        confidence: count as f64 / k as f64,
    }
}

fn assign_point_in_solid(target: &(f64, f64, f64), mesh: &Mesh) -> DomainAssignment {
    let inside = winding_number(mesh, target).abs() > 0.5;
    DomainAssignment {
        domain: if inside { "inside" } else { "outside" }.to_string(),
        confidence: 1.0,
    }
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

    fn labeled(points: &[(f64, &str)]) -> Vec<((f64, f64, f64), String)> {
        points
            .iter()
            .map(|&(x, label)| ((x, 0.0, 0.0), label.to_string()))
            .collect()
    }

    fn assign(
        samples: &[((f64, f64, f64), String)],
        method: DomainMethod,
        n: usize,
    ) -> (String, f64) {
        let out = assign_domain(&[(0.0, 0.0, 0.0)], samples, method, n, None).unwrap();
        (out[0].domain.clone(), out[0].confidence)
    }

    #[test]
    fn test_nearest_and_majority_confidence() {
        let samples = labeled(&[
            (1.0, "z"),
            (2.0, "a"),
            (3.0, "a"),
            (4.0, "z"),
            (5.0, "c"),
            (6.0, "a"),
        ]);
        // Five nearest: z a a z c; the z/a tie goes to z, the nearest.
        let z = ("z".to_string(), 0.4);
        assert_eq!(assign(&samples, DomainMethod::MajorityVote, 5), z);
        assert_eq!(assign(&samples, DomainMethod::Nearest, 5), z);
        // Three nearest: z a a.
        let (label, share) = assign(&samples, DomainMethod::MajorityVote, 3);
        assert_eq!((label.as_str(), share), ("a", 2.0 / 3.0));
        let (label, share) = assign(&samples, DomainMethod::Nearest, 3);
        assert_eq!((label.as_str(), share), ("z", 1.0 / 3.0));
        // More neighbors than samples uses them all.
        let (label, share) = assign(&samples, DomainMethod::MajorityVote, 50);
        assert_eq!((label.as_str(), share), ("a", 0.5));
        assert!(
            assign_domain(&[(0.0, 0.0, 0.0)], &samples, DomainMethod::Nearest, 0, None).is_err()
        );
    }

    #[test]
    fn test_equidistant_samples_rank_by_label() {
        for samples in [
            labeled(&[(-1.0, "b"), (1.0, "a")]),
            labeled(&[(1.0, "a"), (-1.0, "b")]),
        ] {
            let (label, share) = assign(&samples, DomainMethod::Nearest, 1);
            assert_eq!((label.as_str(), share), ("a", 1.0));
            let (label, share) = assign(&samples, DomainMethod::MajorityVote, 2);
            assert_eq!((label.as_str(), share), ("a", 0.5));
        }
    }

    #[test]
    fn test_domains_do_not_depend_on_runs_threads_or_sample_order() {
        let names = ["ox", "tr", "fr"];
        let samples: Vec<_> = (0..60)
            .map(|i| {
                let p = ((i % 10) as f64 * 2.0, (i / 10) as f64 * 2.0, 0.0);
                (p, names[i % 3].to_string())
            })
            .collect();
        let reversed: Vec<_> = samples.iter().rev().cloned().collect();
        let targets: Vec<_> = (0..400)
            .map(|i| ((i % 20) as f64, (i / 20) as f64 * 0.5, 0.0))
            .collect();
        let run = |threads: usize, samples: &[((f64, f64, f64), String)]| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| assign_domain(&targets, samples, DomainMethod::MajorityVote, 4, None))
                .unwrap()
                .into_iter()
                .map(|a| (a.domain, a.confidence.to_bits()))
                .collect::<Vec<_>>()
        };
        let reference = run(1, &samples);
        let ties = reference
            .iter()
            .filter(|(_, c)| f64::from_bits(*c) == 0.5)
            .count();
        assert!(ties > 0);
        for _ in 0..5 {
            assert_eq!(run(8, &samples), reference);
        }
        assert_eq!(run(8, &reversed), reference);
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
        let out = assign_domain(
            &[(0.5, 0.5, 0.5), (0.5, 0.5, 2.0)],
            &[],
            DomainMethod::PointInSolid,
            5,
            Some(&mesh),
        )
        .expect("assigns");
        assert_eq!(out[0].domain, "inside");
        assert_eq!(out[1].domain, "outside");
    }
}
