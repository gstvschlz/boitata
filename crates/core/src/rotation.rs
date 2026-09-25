use nalgebra::Matrix3;

/// World (east, north, up) to anisotropy axes (major, semi-major, minor).
///
/// Degrees. Azimuth of the major axis clockwise from north, dip positive
/// down, rake about the major axis. At zero rotation the major axis points
/// north, the semi-major axis west and the minor axis up.
pub fn rotation_matrix(azimuth: f64, dip: f64, rake: f64) -> Matrix3<f64> {
    let (sa, ca) = azimuth.to_radians().sin_cos();
    let (sd, cd) = dip.to_radians().sin_cos();
    let (sr, cr) = rake.to_radians().sin_cos();
    let r1 = Matrix3::new(sa, ca, 0.0, -ca, sa, 0.0, 0.0, 0.0, 1.0);
    let r2 = Matrix3::new(cd, 0.0, -sd, 0.0, 1.0, 0.0, sd, 0.0, cd);
    let r3 = Matrix3::new(1.0, 0.0, 0.0, 0.0, cr, sr, 0.0, -sr, cr);
    r3 * r2 * r1
}

/// Azimuth, dip and rake of orthonormal `major` and `semi` axes (the minor axis
/// completes a right-handed frame); the inverse of [`rotation_matrix`].
pub fn angles_from_axes(major: [f64; 3], semi: [f64; 3]) -> [f64; 3] {
    let dip = (-major[2]).clamp(-1.0, 1.0).asin();
    let azimuth = major[0].atan2(major[1]);
    let (sa, ca) = azimuth.sin_cos();
    let (sd, cd) = dip.sin_cos();
    let across = [-ca, sa, 0.0];
    let below = [sd * sa, sd * ca, cd];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let rake = dot(semi, below).atan2(dot(semi, across));
    [
        azimuth.to_degrees().rem_euclid(360.0),
        dip.to_degrees(),
        rake.to_degrees().rem_euclid(360.0),
    ]
}

/// World to block-model axes (x, y, z) for `[azimuth, dip, rake]`.
///
/// Same angles as [`rotation_matrix`], applied to the y axis: identity at
/// zero rotation, y along the azimuth.
pub fn block_frame(rotation: [f64; 3]) -> Matrix3<f64> {
    let r = rotation_matrix(rotation[0], rotation[1], rotation[2]);
    Matrix3::from_rows(&[-r.row(1), r.row(0).into_owned(), r.row(2).into_owned()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn close(a: Vector3<f64>, b: [f64; 3]) -> bool {
        (a - Vector3::from(b)).norm() < 1e-12
    }

    #[test]
    fn azimuth_is_clockwise_from_north() {
        assert!(close(
            rotation_matrix(0.0, 0.0, 0.0).row(0).transpose(),
            [0.0, 1.0, 0.0]
        ));
        assert!(close(
            rotation_matrix(90.0, 0.0, 0.0).row(0).transpose(),
            [1.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn dip_is_positive_down() {
        let major = rotation_matrix(0.0, 30.0, 0.0).row(0).transpose();
        assert!(close(major, [0.0, 30f64.to_radians().cos(), -0.5]));
    }

    #[test]
    fn angles_from_axes_inverts_the_rotation() {
        for (a, d, r) in [
            (0.0, 0.0, 0.0),
            (37.0, -12.0, 81.0),
            (300.0, 60.0, 15.0),
            (120.0, 5.0, 350.0),
        ] {
            let m = rotation_matrix(a, d, r);
            let row = |i: usize| [m[(i, 0)], m[(i, 1)], m[(i, 2)]];
            let back = angles_from_axes(row(0), row(1));
            let m2 = rotation_matrix(back[0], back[1], back[2]);
            assert!((m - m2).norm() < 1e-9, "{a} {d} {r} -> {back:?}");
        }
    }

    #[test]
    fn rotations_are_proper() {
        for (a, d, r) in [(0.0, 0.0, 0.0), (37.0, -12.0, 81.0), (300.0, 60.0, 15.0)] {
            let m = rotation_matrix(a, d, r);
            assert!((m * m.transpose() - Matrix3::identity()).norm() < 1e-12);
            assert!((m.determinant() - 1.0).abs() < 1e-12);
            assert!((block_frame([a, d, r]).determinant() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn block_frame_is_identity_unrotated_and_turns_y_to_azimuth() {
        assert!((block_frame([0.0; 3]) - Matrix3::identity()).norm() < 1e-12);
        let y = block_frame([90.0, 0.0, 0.0]).row(1).transpose();
        assert!(close(y, [1.0, 0.0, 0.0]));
    }
}
