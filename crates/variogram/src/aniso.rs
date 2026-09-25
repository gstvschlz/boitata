//! Anisotropy: 3D rotation and anisotropic lag distance.

use crate::error::{Result, VarioError};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

pub use ceres_core::rotation_matrix;

/// Azimuth/dip/rake angles and anisotropic ranges.
///
/// Angles (degrees):
/// - `azimuth`: direction of the major axis (0–360°, from North, clockwise)
/// - `dip`: inclination of the major axis (−90°…+90°, positive plunging down)
/// - `rake`: rotation about the major axis (0–360°)
///
/// Ranges (meters): `major` ≥ `semi` ≥ `minor` (not enforced, but conventional).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Angles {
    pub azimuth: f64,
    pub dip: f64,
    #[serde(alias = "pitch")]
    pub rake: f64,
    pub major: f64,
    pub semi: f64,
    pub minor: f64,
}

impl Angles {
    pub fn validate(&self) -> Result<()> {
        if self.major <= 0.0 || self.semi <= 0.0 || self.minor <= 0.0 {
            return Err(VarioError::InvalidAnisotropy(
                "all ranges must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// Anisotropy transform: rotation matrix plus range scaling. Serialized as its
/// [`Angles`]; the rotation is rebuilt on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Angles", into = "Angles")]
pub struct Anisotropy {
    pub angles: Angles,
    rotation: Matrix3<f64>,
}

impl TryFrom<Angles> for Anisotropy {
    type Error = VarioError;

    fn try_from(angles: Angles) -> Result<Self> {
        Self::new(angles)
    }
}

impl From<Anisotropy> for Angles {
    fn from(a: Anisotropy) -> Self {
        a.angles
    }
}

impl Anisotropy {
    /// Build from angles/ranges.
    pub fn new(angles: Angles) -> Result<Self> {
        angles.validate()?;
        let rotation = rotation_matrix(angles.azimuth, angles.dip, angles.rake);
        Ok(Self { angles, rotation })
    }

    /// The linear map from world coordinates to variogram space: rotation
    /// followed by range scaling, so `lag(p, q) == (matrix() * (q − p)).norm()`.
    ///
    /// [`Anisotropy::lag`] is all an estimator needs, but a *derivative*
    /// observation has to be carried through the same map — a dip measured in
    /// world coordinates is a different direction once the axes are squeezed —
    /// and that needs the matrix itself, not the distance it induces.
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::from_diagonal(&Vector3::new(
            1.0 / self.angles.major,
            1.0 / self.angles.semi,
            1.0 / self.angles.minor,
        )) * self.rotation
    }

    /// Anisotropic (reduced) lag distance between two points.
    ///
    /// Rotates the separation vector into variogram space and scales each axis by
    /// its range, so an isotropic model applied to this distance yields the desired
    /// ellipsoidal anisotropy.
    pub fn lag(&self, p: &(f64, f64, f64), q: &(f64, f64, f64)) -> f64 {
        let v = Vector3::new(q.0 - p.0, q.1 - p.1, q.2 - p.2);
        let r = self.rotation * v;
        let s = Vector3::new(
            r.x / self.angles.major,
            r.y / self.angles.semi,
            r.z / self.angles.minor,
        );
        s.norm()
    }
}

/// Plain Euclidean distance (isotropic).
pub fn euclidean(p: &(f64, f64, f64), q: &(f64, f64, f64)) -> f64 {
    let dx = q.0 - p.0;
    let dy = q.1 - p.1;
    let dz = q.2 - p.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_lag_equals_euclidean_scaled() {
        let a = Anisotropy::new(Angles {
            azimuth: 0.0,
            dip: 0.0,
            rake: 0.0,
            major: 100.0,
            semi: 100.0,
            minor: 100.0,
        })
        .unwrap();
        let p = (0.0, 0.0, 0.0);
        let q = (30.0, 40.0, 0.0);
        // Euclidean 50, scaled by 100 → 0.5
        assert!((a.lag(&p, &q) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn anisotropy_shortens_minor_axis() {
        // Major along East (azimuth 90°); minor range small → vertical separation looks large.
        let a = Anisotropy::new(Angles {
            azimuth: 90.0,
            dip: 0.0,
            rake: 0.0,
            major: 100.0,
            semi: 100.0,
            minor: 10.0,
        })
        .unwrap();
        let along = a.lag(&(0.0, 0.0, 0.0), &(0.0, 0.0, 20.0)); // vertical
        assert!(along > 1.0); // 20 / 10 = 2 reduced units
    }

    #[test]
    fn major_axis_matches_experimental_direction() {
        for (az, dip) in [(0.0, 0.0), (90.0, 0.0), (135.0, 30.0), (300.0, -45.0)] {
            let major = rotation_matrix(az, dip, 17.0).row(0).transpose();
            let (x, y, z) = crate::surface::unit_vector(az, dip);
            assert!((major - Vector3::new(x, y, z)).norm() < 1e-12);
        }
    }

    #[test]
    fn json_round_trip_keeps_the_rotation() {
        let a = Anisotropy::new(Angles {
            azimuth: 30.0,
            dip: 10.0,
            rake: 5.0,
            major: 1.0,
            semi: 0.5,
            minor: 0.2,
        })
        .unwrap();
        let b: Anisotropy = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        let (p, q) = ((0.0, 0.0, 0.0), (3.0, 4.0, 1.0));
        assert!((a.lag(&p, &q) - b.lag(&p, &q)).abs() < 1e-12);
    }

    #[test]
    fn reads_the_old_pitch_name() {
        let a: Angles = serde_json::from_str(
            r#"{"azimuth":10,"dip":5,"pitch":20,"major":1,"semi":0.5,"minor":0.5}"#,
        )
        .unwrap();
        assert_eq!(a.rake, 20.0);
    }

    #[test]
    fn rejects_nonpositive_range() {
        let bad = Angles {
            azimuth: 0.0,
            dip: 0.0,
            rake: 0.0,
            major: 0.0,
            semi: 100.0,
            minor: 100.0,
        };
        assert!(Anisotropy::new(bad).is_err());
    }
}
