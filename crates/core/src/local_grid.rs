use serde::{Deserialize, Serialize};

use crate::{BlockModel, Error, Mesh, PointSet, Polylines, Result};

/// Similarity transform from a local mine grid to projected coordinates:
/// `projected = shift + scale * R * local` in xy and `z + z_shift`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalGrid {
    pub scale: f64,
    /// Azimuth of local grid north in projected coordinates, degrees
    /// clockwise.
    pub rotation: f64,
    pub shift: [f64; 2],
    pub z_shift: f64,
    /// CRS of the projected coordinates.
    pub crs: Option<String>,
}

impl LocalGrid {
    /// Least-squares fit to control point pairs; with `z`, the z offset is
    /// the mean difference, otherwise 0.
    pub fn fit(
        local: &[[f64; 3]],
        projected: &[[f64; 3]],
        z: bool,
        crs: Option<String>,
    ) -> Result<Self> {
        if local.len() != projected.len() {
            return Err(Error::Length {
                expected: local.len(),
                found: projected.len(),
            });
        }
        if local
            .iter()
            .chain(projected)
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err(Error::Geometry("control points must be finite".into()));
        }
        let n = local.len() as f64;
        let mean = |p: &[[f64; 3]]| [0, 1, 2].map(|a| p.iter().map(|q| q[a]).sum::<f64>() / n);
        let (l, p) = (mean(local), mean(projected));
        let (mut a, mut b, mut ss) = (0.0, 0.0, 0.0);
        for (u, v) in local.iter().zip(projected) {
            let (ux, uy) = (u[0] - l[0], u[1] - l[1]);
            let (vx, vy) = (v[0] - p[0], v[1] - p[1]);
            a += ux * vx + uy * vy;
            b += ux * vy - uy * vx;
            ss += ux * ux + uy * uy;
        }
        if local.len() < 2 || ss == 0.0 {
            return Err(Error::Geometry(
                "need at least two control points at distinct local positions".into(),
            ));
        }
        let (a, b) = (a / ss, b / ss);
        Ok(Self {
            scale: a.hypot(b),
            rotation: (-b.atan2(a).to_degrees()).rem_euclid(360.0),
            shift: [p[0] - (a * l[0] - b * l[1]), p[1] - (b * l[0] + a * l[1])],
            z_shift: if z { p[2] - l[2] } else { 0.0 },
            crs,
        })
    }

    /// `[[a, -b], [b, a]]`, the scaled rotation of the xy plane.
    fn ab(&self) -> (f64, f64) {
        let (s, c) = (-self.rotation.to_radians()).sin_cos();
        (self.scale * c, self.scale * s)
    }

    pub fn forward(&self, q: [f64; 3]) -> [f64; 3] {
        let (a, b) = self.ab();
        [
            self.shift[0] + a * q[0] - b * q[1],
            self.shift[1] + b * q[0] + a * q[1],
            q[2] + self.z_shift,
        ]
    }

    pub fn backward(&self, q: [f64; 3]) -> [f64; 3] {
        let (a, b) = self.ab();
        let k = a * a + b * b;
        let (x, y) = (q[0] - self.shift[0], q[1] - self.shift[1]);
        [
            (a * x + b * y) / k,
            (a * y - b * x) / k,
            q[2] - self.z_shift,
        ]
    }

    /// `projected - forward(local)` per control point.
    pub fn residuals(&self, local: &[[f64; 3]], projected: &[[f64; 3]]) -> Vec<[f64; 3]> {
        local
            .iter()
            .zip(projected)
            .map(|(&u, v)| {
                let f = self.forward(u);
                [0, 1, 2].map(|a| v[a] - f[a])
            })
            .collect()
    }

    /// Errors when coordinates already in this grid's CRS are given to
    /// [`LocalGrid::apply`] or foreign ones to [`LocalGrid::inverse`].
    pub fn check(&self, crs: Option<&str>, forward: bool) -> Result<()> {
        match (crs, self.crs.as_deref()) {
            (Some(c), Some(own))
                if forward && crate::crs::crs_key(c) == crate::crs::crs_key(own) =>
            {
                Err(Error::Geometry(format!(
                    "the coordinates are already in {own}"
                )))
            }
            (Some(c), Some(own))
                if !forward && crate::crs::crs_key(c) != crate::crs::crs_key(own) =>
            {
                Err(Error::Geometry(format!(
                    "the coordinates are in {c}, not {own}"
                )))
            }
            _ => Ok(()),
        }
    }

    fn crs_after(&self, forward: bool) -> Option<String> {
        if forward { self.crs.clone() } else { None }
    }

    pub fn points(&self, points: &PointSet, forward: bool) -> Result<PointSet> {
        self.check(points.crs.as_deref(), forward)?;
        let mut out = points.map_coords(|q| self.map(q, forward));
        out.crs = self.crs_after(forward);
        Ok(out)
    }

    pub fn polylines(&self, lines: &Polylines, forward: bool) -> Result<Polylines> {
        self.check(lines.crs.as_deref(), forward)?;
        let mut out = lines.map_coords(|q| self.map(q, forward));
        out.crs = self.crs_after(forward);
        Ok(out)
    }

    pub fn mesh(&self, mesh: &Mesh, forward: bool) -> Result<Mesh> {
        self.check(mesh.crs.as_deref(), forward)?;
        let mut out = mesh.map_coords(|q| self.map(q, forward));
        out.crs = self.crs_after(forward);
        Ok(out)
    }

    /// The model moved to the other frame: origin mapped, azimuth turned and
    /// horizontal block sizes scaled. A dipping or raked model only takes a
    /// grid of scale 1, as horizontal scaling would shear its blocks.
    pub fn block_model(&self, model: &BlockModel, forward: bool) -> Result<BlockModel> {
        self.check(model.crs.as_deref(), forward)?;
        let mut g = *model.geometry();
        if (g.rotation[1] != 0.0 || g.rotation[2] != 0.0) && (self.scale - 1.0).abs() > 1e-12 {
            return Err(Error::Geometry(
                "a dipping or raked block model needs a local grid of scale 1".into(),
            ));
        }
        let (k, turn) = if forward {
            (self.scale, self.rotation)
        } else {
            (1.0 / self.scale, -self.rotation)
        };
        g.origin = self.map(g.origin, forward);
        g.size[0] *= k;
        g.size[1] *= k;
        g.rotation[0] = (g.rotation[0] + turn).rem_euclid(360.0);
        let mut out = model.with_geometry(g)?;
        out.crs = self.crs_after(forward);
        Ok(out)
    }

    pub fn map(&self, q: [f64; 3], forward: bool) -> [f64; 3] {
        if forward {
            self.forward(q)
        } else {
            self.backward(q)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Geometry;
    use arrow_array::RecordBatch;

    fn grid(rotation: f64) -> LocalGrid {
        LocalGrid {
            scale: 0.9996,
            rotation,
            shift: [512_000.0, 7_400_000.0],
            z_shift: 12.5,
            crs: Some("EPSG:31982".into()),
        }
    }

    fn controls() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1000.0, 0.0, 10.0],
            [0.0, 2000.0, -5.0],
            [750.0, 1300.0, 40.0],
        ]
    }

    #[test]
    fn fit_recovers_an_exact_similarity_at_any_angle() {
        for rotation in [0.0, 17.0, 120.0, 180.0, 250.0, 359.0] {
            let truth = grid(rotation);
            let local = controls();
            let projected: Vec<_> = local.iter().map(|&q| truth.forward(q)).collect();
            let fit = LocalGrid::fit(&local, &projected, true, truth.crs.clone()).unwrap();
            assert!((fit.scale - truth.scale).abs() < 1e-12);
            let turn = (fit.rotation - rotation + 180.0).rem_euclid(360.0) - 180.0;
            assert!(turn.abs() < 1e-9, "{rotation}: {}", fit.rotation);
            for r in fit.residuals(&local, &projected).iter().flatten() {
                assert!(r.abs() < 1e-6);
            }
            for q in local {
                let back = fit.backward(fit.forward(q));
                assert!((0..3).all(|a| (back[a] - q[a]).abs() < 1e-9));
            }
        }
    }

    #[test]
    fn local_north_points_along_the_rotation_azimuth() {
        let g = LocalGrid {
            scale: 1.0,
            rotation: 90.0,
            shift: [0.0, 0.0],
            z_shift: 0.0,
            crs: None,
        };
        let p = g.forward([0.0, 1.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 1e-12 && p[1].abs() < 1e-12);
    }

    #[test]
    fn block_model_centroids_follow_the_points() {
        let g = grid(120.0);
        let geometry = Geometry {
            origin: [100.0, 200.0, 50.0],
            size: [10.0, 5.0, 2.0],
            count: [3, 4, 2],
            rotation: [30.0, 0.0, 0.0],
        };
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(24));
        let batch = RecordBatch::try_new_with_options(
            std::sync::Arc::new(arrow_schema::Schema::empty()),
            vec![],
            &options,
        )
        .unwrap();
        let model = BlockModel::regular(geometry, batch).unwrap();
        let moved = g.block_model(&model, true).unwrap();
        assert_eq!(moved.crs, g.crs);
        for i in 0..geometry.cells() {
            let want = g.forward(geometry.centroid(i));
            let got = moved.geometry().centroid(i);
            assert!((0..3).all(|a| (want[a] - got[a]).abs() < 1e-6));
        }
        let back = g.block_model(&moved, false).unwrap();
        let b = back.geometry();
        assert!((0..3).all(|a| (b.origin[a] - geometry.origin[a]).abs() < 1e-6));
        assert!((0..3).all(|a| (b.size[a] - geometry.size[a]).abs() < 1e-9));
        assert!((b.rotation[0] - 30.0).abs() < 1e-9);
        assert!(back.crs.is_none());
    }

    #[test]
    fn rejects_degenerate_controls_and_a_second_apply() {
        let p = [[1.0, 1.0, 0.0]; 3];
        assert!(LocalGrid::fit(&p, &p, false, None).is_err());
        let g = grid(10.0);
        assert!(g.check(Some("EPSG:31982"), true).is_err());
        assert!(g.check(Some("EPSG:4326"), false).is_err());
        assert!(g.check(None, true).is_ok());
    }

    #[test]
    fn serializes_to_json() {
        let g = grid(42.0);
        let text = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<LocalGrid>(&text).unwrap(), g);
    }
}
