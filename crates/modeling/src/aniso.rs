//! Anisotropy as a coordinate transform.
//!
//! [`variogram::Anisotropy`] exposes the reduced lag as a *distance function*
//! (`lag(p, q)`), which is all kriging needs. An RBF instead wants the reduced
//! *coordinates*: the kernel matrix, the polynomial drift and the conditioning
//! rescale all operate on points, not pairs. Same rotation and range
//! scaling, applied once per point — `transform_matches_variogram_lag` in the
//! tests pins the two to each other.

use nalgebra::{Matrix3, Vector3};
use variogram::{Angles, aniso::rotation_matrix};

use crate::error::{ModelError, Result};

/// Rotation + range scaling, world coordinates → reduced (isotropic) space.
#[derive(Debug, Clone)]
pub struct AnisoTransform {
    rotation: Matrix3<f64>,
    ranges: [f64; 3],
}

impl AnisoTransform {
    pub fn new(angles: &Angles) -> Result<Self> {
        angles
            .validate()
            .map_err(|e| ModelError::InvalidParameter(e.to_string()))?;
        Ok(Self {
            rotation: rotation_matrix(angles.azimuth, angles.dip, angles.rake),
            ranges: [angles.major, angles.semi, angles.minor],
        })
    }

    /// The transform as a matrix, so a *direction* can be carried through it.
    ///
    /// [`AnisoTransform::apply`] moves points; a structural reading is a
    /// direction, and directions do not transform the same way. A gradient
    /// constraint `∇f·n = v` in world space becomes `∇g·(M n) = v` in reduced
    /// space, which needs `M` itself rather than the map it induces on points.
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::from_diagonal(&Vector3::new(
            1.0 / self.ranges[0],
            1.0 / self.ranges[1],
            1.0 / self.ranges[2],
        )) * self.rotation
    }

    /// Reduced coordinates of `p`; Euclidean distance between two results
    /// equals `variogram::Anisotropy::lag` on the same pair.
    pub fn apply(&self, p: &[f64; 3]) -> [f64; 3] {
        let r = self.rotation * Vector3::new(p[0], p[1], p[2]);
        [
            r.x / self.ranges[0],
            r.y / self.ranges[1],
            r.z / self.ranges[2],
        ]
    }
}

/// Apply an optional transform, or pass the point through unchanged.
pub fn reduce(transform: Option<&AnisoTransform>, p: &[f64; 3]) -> [f64; 3] {
    match transform {
        Some(t) => t.apply(p),
        None => *p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::Anisotropy;

    fn angles() -> Angles {
        Angles {
            azimuth: 35.0,
            dip: 20.0,
            rake: 10.0,
            major: 120.0,
            semi: 60.0,
            minor: 15.0,
        }
    }

    #[test]
    fn transform_matches_variogram_lag() {
        let t = AnisoTransform::new(&angles()).unwrap();
        let a = Anisotropy::new(angles()).unwrap();
        for (p, q) in [
            ([0.0, 0.0, 0.0], [30.0, 40.0, 0.0]),
            ([10.0, -5.0, 3.0], [-20.0, 7.0, 55.0]),
            ([1e5, 2e5, 300.0], [1e5 + 12.0, 2e5 - 8.0, 340.0]),
        ] {
            let tp = t.apply(&p);
            let tq = t.apply(&q);
            let d = ((tq[0] - tp[0]).powi(2) + (tq[1] - tp[1]).powi(2) + (tq[2] - tp[2]).powi(2))
                .sqrt();
            let lag = a.lag(&(p[0], p[1], p[2]), &(q[0], q[1], q[2]));
            assert!((d - lag).abs() < 1e-9, "{d} vs {lag}");
        }
    }

    #[test]
    fn rejects_nonpositive_range() {
        let bad = Angles {
            minor: 0.0,
            ..angles()
        };
        assert!(AnisoTransform::new(&bad).is_err());
    }
}
