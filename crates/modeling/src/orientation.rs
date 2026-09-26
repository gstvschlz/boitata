//! Structural readings as unit vectors.
//!
//! A dip / dip-direction pair and a plunge / trend pair are the two ways field
//! geology records an orientation, and neither says anything about the value of
//! the implicit field — they constrain its *gradient*. A planar structure fixes
//! the direction the field changes fastest in (its normal); a lineation fixes a
//! direction the field does not change along at all.
//!
//! Angles follow the convention the rest of the stack already uses for
//! directional data — the one `drillholes::desurvey` works in: coordinates are
//! (East, North, Up), azimuths run clockwise from North, and dip and plunge are
//! measured downwards from horizontal.

use crate::error::{ModelError, Result};

/// A planar structural reading: bedding, foliation, a vein wall, a contact
/// measured in the pit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// Degrees below horizontal, 0–90.
    pub dip: f64,
    /// Azimuth of the down-dip direction, degrees clockwise from North.
    pub dip_direction: f64,
}

/// A linear structural reading: a fold axis, a mineral lineation, the local
/// trend of a vein.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lineation {
    /// Degrees below horizontal, 0–90.
    pub plunge: f64,
    /// Azimuth of the down-plunge direction, degrees clockwise from North.
    pub trend: f64,
}

impl Plane {
    /// Upward unit normal.
    ///
    /// A horizontal bed gives `(0, 0, 1)`; a vertical plane striking N–S and
    /// "dipping" 90° toward 090° gives `(1, 0, 0)`. The sign convention (up
    /// rather than down) is arbitrary but must be consistent, because it is
    /// what decides which side of the modeled contact the field increases on.
    pub fn normal(&self) -> Result<[f64; 3]> {
        if !(0.0..=90.0).contains(&self.dip) || !self.dip.is_finite() {
            return Err(ModelError::InvalidParameter(format!(
                "dip must be between 0 and 90 degrees, got {}",
                self.dip
            )));
        }
        if !self.dip_direction.is_finite() {
            return Err(ModelError::InvalidParameter(
                "dip direction must be a finite angle".into(),
            ));
        }
        let (d, a) = (self.dip.to_radians(), self.dip_direction.to_radians());
        Ok([d.sin() * a.sin(), d.sin() * a.cos(), d.cos()])
    }

    /// Two orthonormal directions lying *in* the plane, so a reading can be
    /// expressed as "the field does not change along these" as well as "it
    /// changes across the normal".
    ///
    /// The pair is the down-dip vector and the strike vector; together with the
    /// normal they form a right-handed frame.
    pub fn tangents(&self) -> Result<([f64; 3], [f64; 3])> {
        let (d, a) = (self.dip.to_radians(), self.dip_direction.to_radians());
        self.normal()?;
        // Down-dip: horizontal along the dip direction, descending by the dip.
        let down_dip = [d.cos() * a.sin(), d.cos() * a.cos(), -d.sin()];
        // Strike: horizontal, 90° anticlockwise of the dip direction.
        let strike = [a.cos(), -a.sin(), 0.0];
        Ok((down_dip, strike))
    }
}

impl Lineation {
    /// Down-plunge unit vector. A horizontal lineation trending due North is
    /// `(0, 1, 0)`; one plunging vertically is `(0, 0, -1)`.
    pub fn direction(&self) -> Result<[f64; 3]> {
        if !(0.0..=90.0).contains(&self.plunge) || !self.plunge.is_finite() {
            return Err(ModelError::InvalidParameter(format!(
                "plunge must be between 0 and 90 degrees, got {}",
                self.plunge
            )));
        }
        if !self.trend.is_finite() {
            return Err(ModelError::InvalidParameter(
                "trend must be a finite angle".into(),
            ));
        }
        let (p, t) = (self.plunge.to_radians(), self.trend.to_radians());
        Ok([p.cos() * t.sin(), p.cos() * t.cos(), -p.sin()])
    }
}

/// Euclidean norm of a 3-vector.
pub fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Scale a vector to unit length, or fail if it has none to scale.
pub fn unit(v: &[f64; 3]) -> Result<[f64; 3]> {
    let n = norm(v);
    if !n.is_finite() || n <= 0.0 {
        return Err(ModelError::InvalidParameter(
            "direction vector has zero length".into(),
        ));
    }
    Ok([v[0] / n, v[1] / n, v[2] / n])
}

pub fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64; 3], b: &[f64; 3]) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < 1e-12)
    }

    #[test]
    fn a_horizontal_bed_has_a_vertical_normal() {
        let n = Plane {
            dip: 0.0,
            dip_direction: 137.0,
        }
        .normal()
        .unwrap();
        assert!(close(&n, &[0.0, 0.0, 1.0]), "{n:?}");
    }

    #[test]
    fn a_vertical_plane_dipping_east_has_an_easterly_normal() {
        let n = Plane {
            dip: 90.0,
            dip_direction: 90.0,
        }
        .normal()
        .unwrap();
        assert!(close(&n, &[1.0, 0.0, 0.0]), "{n:?}");
    }

    /// Dip direction is an azimuth clockwise from North, the same convention
    /// `drillholes::desurvey` uses — 000° must lean the normal towards North,
    /// not towards East.
    #[test]
    fn dip_direction_is_an_azimuth_from_north() {
        let n = Plane {
            dip: 45.0,
            dip_direction: 0.0,
        }
        .normal()
        .unwrap();
        assert!(n[1] > 0.5, "normal should lean North, got {n:?}");
        assert!(
            n[0].abs() < 1e-12,
            "normal should have no easting, got {n:?}"
        );
    }

    #[test]
    fn normals_and_lineations_are_unit_length() {
        for dip in [0.0, 12.0, 45.0, 78.0, 90.0] {
            for azm in [0.0, 47.0, 145.0, 260.0, 359.0] {
                let n = Plane {
                    dip,
                    dip_direction: azm,
                }
                .normal()
                .unwrap();
                assert!((norm(&n) - 1.0).abs() < 1e-12);
                let l = Lineation {
                    plunge: dip,
                    trend: azm,
                }
                .direction()
                .unwrap();
                assert!((norm(&l) - 1.0).abs() < 1e-12);
            }
        }
    }

    /// The whole point of the tangents: they span the plane, so the normal is
    /// orthogonal to both and the three together are a frame.
    #[test]
    fn tangents_are_orthogonal_to_the_normal_and_to_each_other() {
        for dip in [0.0, 20.0, 55.0, 90.0] {
            for azm in [0.0, 90.0, 210.0, 300.0] {
                let plane = Plane {
                    dip,
                    dip_direction: azm,
                };
                let n = plane.normal().unwrap();
                let (a, b) = plane.tangents().unwrap();
                assert!((norm(&a) - 1.0).abs() < 1e-12);
                assert!((norm(&b) - 1.0).abs() < 1e-12);
                assert!(dot(&n, &a).abs() < 1e-12, "dip {dip} azm {azm}: {a:?}");
                assert!(dot(&n, &b).abs() < 1e-12, "dip {dip} azm {azm}: {b:?}");
                assert!(dot(&a, &b).abs() < 1e-12, "dip {dip} azm {azm}");
            }
        }
    }

    /// A lineation lying in a plane is a tangent of it — the down-dip vector is
    /// exactly the lineation that plunges at the dip along the dip direction.
    #[test]
    fn the_down_dip_tangent_is_the_lineation_plunging_at_the_dip() {
        let plane = Plane {
            dip: 35.0,
            dip_direction: 115.0,
        };
        let (down_dip, _) = plane.tangents().unwrap();
        let lin = Lineation {
            plunge: 35.0,
            trend: 115.0,
        }
        .direction()
        .unwrap();
        assert!(close(&down_dip, &lin), "{down_dip:?} vs {lin:?}");
    }

    #[test]
    fn rejects_angles_outside_the_convention() {
        assert!(
            Plane {
                dip: -1.0,
                dip_direction: 0.0
            }
            .normal()
            .is_err()
        );
        assert!(
            Plane {
                dip: 91.0,
                dip_direction: 0.0
            }
            .normal()
            .is_err()
        );
        assert!(
            Lineation {
                plunge: 120.0,
                trend: 0.0
            }
            .direction()
            .is_err()
        );
        assert!(
            Plane {
                dip: f64::NAN,
                dip_direction: 0.0
            }
            .normal()
            .is_err()
        );
    }

    #[test]
    fn unit_rejects_a_zero_vector() {
        assert!(unit(&[0.0, 0.0, 0.0]).is_err());
        let u = unit(&[3.0, 0.0, 4.0]).unwrap();
        assert!(close(&u, &[0.6, 0.0, 0.8]), "{u:?}");
    }
}
