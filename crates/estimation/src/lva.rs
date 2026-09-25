//! Locally varying anisotropy: orientation fields derived from a gridded
//! attribute, a point cloud or a mesh, and estimation that uses the local
//! anisotropy of each target for both its variogram and its search.

use ceres_core::{Geometry, angles_from_axes, block_frame};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use variogram::{Angles, Anisotropy, Variogram};

use crate::Sample;
use crate::error::{EstimError, Result};
use crate::krige::Estimate;
use crate::search::{Search, SearchTree};

type Point = (f64, f64, f64);

/// Orientation and range ratios at a set of locations.
#[derive(Debug, Clone)]
pub struct LocalAnisotropy {
    pub coords: Vec<Point>,
    /// Azimuth, dip and rake in degrees.
    pub angles: Vec<[f64; 3]>,
    /// Semi-major/major and minor/major range ratios, in (0, 1].
    pub ratios: Vec<[f64; 2]>,
}

/// Which in-plane direction of a mesh becomes the major axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshMajor {
    Dip,
    Strike,
}

fn invalid(message: &str) -> EstimError {
    EstimError::InvalidParameters(message.into())
}

fn column(m: &Matrix3<f64>, i: usize) -> Vector3<f64> {
    m.column(i).into_owned()
}

/// Right-handed frame from a major and an approximate semi-major direction.
fn frame(major: Vector3<f64>, semi: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let mut major = major.normalize();
    if major.z > 0.0 {
        major = -major;
    }
    let semi = (semi - major * major.dot(&semi)).normalize();
    (major, semi)
}

fn ratio(small: f64, large: f64) -> f64 {
    if large <= 0.0 {
        1.0
    } else {
        (small.max(0.0) / large).sqrt().clamp(0.05, 1.0)
    }
}

impl LocalAnisotropy {
    pub fn new(coords: Vec<Point>, angles: Vec<[f64; 3]>, ratios: Vec<[f64; 2]>) -> Result<Self> {
        if coords.len() != angles.len() || coords.len() != ratios.len() {
            return Err(invalid("coords, angles and ratios need equal lengths"));
        }
        if ratios.iter().flatten().any(|r| !(*r > 0.0 && *r <= 1.0)) {
            return Err(invalid("ratios must be in (0, 1]"));
        }
        Ok(Self {
            coords,
            angles,
            ratios,
        })
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Anisotropy at location `i`, with ranges as ratios of the major range.
    pub fn anisotropy(&self, i: usize) -> Anisotropy {
        let [azimuth, dip, pitch] = self.angles[i];
        let [semi, minor] = self.ratios[i];
        Anisotropy::new(Angles {
            azimuth,
            dip,
            pitch,
            major: 1.0,
            semi,
            minor,
        })
        .expect("ratios are positive")
    }

    fn from_axes(coords: Vec<Point>, axes: Vec<(Vector3<f64>, Vector3<f64>, [f64; 2])>) -> Self {
        let (angles, ratios) = axes
            .into_iter()
            .map(|(major, semi, ratios)| {
                let (major, semi) = frame(major, semi);
                (angles_from_axes(major.into(), semi.into()), ratios)
            })
            .unzip();
        Self {
            coords,
            angles,
            ratios,
        }
    }

    /// Orientations from symmetric tensors whose eigenvector of largest
    /// eigenvalue is the direction of fastest change (the minor axis).
    /// Without `ratios`, ratios follow the eigenvalues.
    pub fn from_tensors(
        coords: Vec<Point>,
        tensors: &[Matrix3<f64>],
        ratios: Option<[f64; 2]>,
    ) -> Self {
        let axes = tensors
            .iter()
            .map(|t| {
                let eig = SymmetricEigen::new(*t);
                let mut order = [0, 1, 2];
                order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
                let [lo, mid, hi] = order.map(|i| eig.eigenvalues[i]);
                let r = ratios.unwrap_or([ratio(lo, mid), ratio(lo, hi)]);
                (
                    column(&eig.eigenvectors, order[0]),
                    column(&eig.eigenvectors, order[1]),
                    r,
                )
            })
            .collect();
        Self::from_axes(coords, axes)
    }

    /// Orientations from the gradient of a regular grid attribute (`NaN` is
    /// missing): the structure tensor of central differences, summed over a
    /// window of `window` cells each side, has its least-change direction as
    /// the major axis. A 2D grid keeps its axes horizontal.
    pub fn from_grid(
        geometry: &Geometry,
        values: &[f64],
        window: usize,
        ratios: Option<[f64; 2]>,
    ) -> Result<Self> {
        let cells = geometry.cells() as usize;
        if values.len() != cells {
            return Err(invalid("one value per grid cell"));
        }
        let [nx, ny, nz] = geometry.count;
        let at = |i: isize, j: isize, k: isize| -> Option<f64> {
            if i < 0 || j < 0 || k < 0 || i >= nx as isize || j >= ny as isize || k >= nz as isize {
                return None;
            }
            let v = values[geometry.index([i as usize, j as usize, k as usize]) as usize];
            v.is_finite().then_some(v)
        };
        let to_world = block_frame(geometry.rotation).transpose();
        let gradients: Vec<Option<Vector3<f64>>> = (0..cells as u64)
            .into_par_iter()
            .map(|c| {
                let [i, j, k] = geometry.ijk(c).map(|v| v as isize);
                at(i, j, k)?;
                let mut g = Vector3::zeros();
                for (axis, step) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)].iter().enumerate() {
                    let (di, dj, dk) = *step;
                    let ahead = at(i + di, j + dj, k + dk);
                    let behind = at(i - di, j - dj, k - dk);
                    let here = at(i, j, k)?;
                    let h = geometry.size[axis];
                    g[axis] = match (ahead, behind) {
                        (Some(a), Some(b)) => (a - b) / (2.0 * h),
                        (Some(a), None) => (a - here) / h,
                        (None, Some(b)) => (here - b) / h,
                        (None, None) => 0.0,
                    };
                }
                Some(to_world * g)
            })
            .collect();
        let w = window as isize;
        let flat = nz == 1;
        let tensors: Vec<Matrix3<f64>> = (0..cells as u64)
            .into_par_iter()
            .map(|c| {
                let [i, j, k] = geometry.ijk(c).map(|v| v as isize);
                let mut t = Matrix3::zeros();
                for dk in -w..=w {
                    for dj in -w..=w {
                        for di in -w..=w {
                            let (a, b, d) = (i + di, j + dj, k + dk);
                            if a < 0
                                || b < 0
                                || d < 0
                                || a >= nx as isize
                                || b >= ny as isize
                                || d >= nz as isize
                            {
                                continue;
                            }
                            let cell =
                                geometry.index([a as usize, b as usize, d as usize]) as usize;
                            if let Some(g) = gradients[cell] {
                                t += g * g.transpose();
                            }
                        }
                    }
                }
                if flat {
                    t[(2, 2)] += 10.0 * t.trace() + f64::MIN_POSITIVE;
                }
                t
            })
            .collect();
        let coords = (0..cells as u64)
            .map(|c| {
                let p = geometry.centroid(c);
                (p[0], p[1], p[2])
            })
            .collect();
        let mut local = Self::from_tensors(coords, &tensors, ratios);
        if flat && ratios.is_none() {
            for r in &mut local.ratios {
                r[1] = r[0];
            }
        }
        Ok(local)
    }

    /// Orientations of a point cloud: principal axes of each point's `k`
    /// nearest neighbours (largest spread is the major axis).
    pub fn from_points(coords: &[Point], k: usize, ratios: Option<[f64; 2]>) -> Result<Self> {
        if coords.len() < 3 || k < 3 {
            return Err(invalid("need at least 3 points and k >= 3"));
        }
        let samples: Vec<Sample> = coords.iter().map(|&p| Sample::new(p, 0.0)).collect();
        let search = Search {
            min_samples: 1,
            max_samples: k.min(coords.len()),
            ..Default::default()
        };
        let tree = SearchTree::new(&samples, &search, None);
        let axes = coords
            .par_iter()
            .map(|p| {
                let near = tree.neighbors(p).unwrap_or_default();
                let pts: Vec<Vector3<f64>> = near
                    .iter()
                    .map(|&i| Vector3::new(coords[i].0, coords[i].1, coords[i].2))
                    .collect();
                let mean = pts.iter().sum::<Vector3<f64>>() / pts.len() as f64;
                let cov = pts
                    .iter()
                    .map(|q| (q - mean) * (q - mean).transpose())
                    .sum::<Matrix3<f64>>();
                let eig = SymmetricEigen::new(cov);
                let mut order = [0, 1, 2];
                order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
                let [hi, mid, lo] = order.map(|i| eig.eigenvalues[i]);
                let r = ratios.unwrap_or([ratio(mid, hi), ratio(lo, hi)]);
                (
                    column(&eig.eigenvectors, order[0]),
                    column(&eig.eigenvectors, order[1]),
                    r,
                )
            })
            .collect();
        Ok(Self::from_axes(coords.to_vec(), axes))
    }

    /// Orientations at `targets` from the nearest triangle of a mesh: its
    /// normal is the minor axis, and `major` picks dip direction or strike.
    pub fn from_mesh(
        vertices: &[Point],
        triangles: &[(usize, usize, usize)],
        targets: &[Point],
        major: MeshMajor,
        ratios: [f64; 2],
    ) -> Result<Self> {
        if triangles.is_empty() {
            return Err(invalid("mesh has no triangles"));
        }
        let v = |i: usize| Vector3::new(vertices[i].0, vertices[i].1, vertices[i].2);
        let mut normals = Vec::with_capacity(triangles.len());
        let mut centres = Vec::with_capacity(triangles.len());
        for &(a, b, c) in triangles {
            if a.max(b).max(c) >= vertices.len() {
                return Err(invalid("triangle index outside the vertices"));
            }
            let n = (v(b) - v(a)).cross(&(v(c) - v(a)));
            let centre = (v(a) + v(b) + v(c)) / 3.0;
            normals.push(if n.z < 0.0 { -n } else { n });
            centres.push(Sample::new((centre.x, centre.y, centre.z), 0.0));
        }
        let tree = SearchTree::new(
            &centres,
            &Search {
                min_samples: 1,
                max_samples: 1,
                ..Default::default()
            },
            None,
        );
        let up = Vector3::z();
        let axes = targets
            .par_iter()
            .map(|t| {
                let nearest = tree.neighbors(t).map(|n| n[0]).unwrap_or(0);
                let n = normals[nearest].normalize();
                let strike = up.cross(&n);
                let strike = if strike.norm() < 1e-9 {
                    Vector3::y()
                } else {
                    strike.normalize()
                };
                let down_dip = n.cross(&strike);
                match major {
                    MeshMajor::Dip => (down_dip, strike, ratios),
                    MeshMajor::Strike => (strike, down_dip, ratios),
                }
            })
            .collect();
        Ok(Self::from_axes(targets.to_vec(), axes))
    }

    /// Averages, over all locations within `radius`, the direction tensors of
    /// the major and semi-major axes and the ratios.
    pub fn smooth(&self, radius: f64) -> Self {
        let axes: Vec<(Matrix3<f64>, Matrix3<f64>)> = (0..self.len())
            .map(|i| {
                let [a, d, r] = self.angles[i];
                let m = ceres_core::rotation_matrix(a, d, r);
                let (major, semi) = (m.row(0).transpose(), m.row(1).transpose());
                (major * major.transpose(), semi * semi.transpose())
            })
            .collect();
        let samples: Vec<Sample> = self.coords.iter().map(|&p| Sample::new(p, 0.0)).collect();
        let search = Search {
            min_samples: 1,
            max_samples: self.len(),
            radius,
            ..Default::default()
        };
        let tree = SearchTree::new(&samples, &search, None);
        let top = |t: &Matrix3<f64>| {
            let eig = SymmetricEigen::new(*t);
            let i = eig.eigenvalues.imax();
            column(&eig.eigenvectors, i)
        };
        let averaged = self
            .coords
            .par_iter()
            .map(|p| {
                let near = tree.neighbors(p).unwrap_or_default();
                let major: Matrix3<f64> = near.iter().map(|&i| axes[i].0).sum();
                let semi: Matrix3<f64> = near.iter().map(|&i| axes[i].1).sum();
                let n = near.len().max(1) as f64;
                let ratios = near.iter().fold([0.0, 0.0], |acc, &i| {
                    [
                        acc[0] + self.ratios[i][0] / n,
                        acc[1] + self.ratios[i][1] / n,
                    ]
                });
                (top(&major), top(&semi), ratios)
            })
            .collect();
        Self::from_axes(self.coords.clone(), averaged)
    }

    /// The anisotropy of the nearest location, at each target.
    pub fn at(&self, targets: &[Point]) -> Self {
        let samples: Vec<Sample> = self.coords.iter().map(|&p| Sample::new(p, 0.0)).collect();
        let search = Search {
            min_samples: 1,
            max_samples: 1,
            ..Default::default()
        };
        let tree = SearchTree::new(&samples, &search, None);
        let nearest: Vec<usize> = targets
            .par_iter()
            .map(|t| tree.neighbors(t).map(|n| n[0]).unwrap_or(0))
            .collect();
        Self {
            coords: targets.to_vec(),
            angles: nearest.iter().map(|&i| self.angles[i]).collect(),
            ratios: nearest.iter().map(|&i| self.ratios[i]).collect(),
        }
    }
}

/// As [`crate::estimate_many`], with target `i` using `local[i]` for its
/// variogram and its search ellipsoid (`search.radius` along the major axis).
pub fn estimate_many_local<F>(
    targets: &[Point],
    local: &LocalAnisotropy,
    samples: &[Sample],
    search: &Search,
    vg: &Variogram,
    estimator: F,
) -> Result<Vec<Option<Estimate>>>
where
    F: Fn(&Point, &[Sample], &Variogram) -> Result<Estimate> + Sync,
{
    if local.len() != targets.len() {
        return Err(invalid("one local anisotropy per target"));
    }
    let isotropic = Search {
        anisotropy: None,
        ..search.clone()
    };
    let tree = SearchTree::new(samples, &isotropic, None);
    Ok(targets
        .par_iter()
        .enumerate()
        .map(|(i, target)| {
            let aniso = local.anisotropy(i);
            let chosen = tree.neighbors_within(target, &aniso).ok()?;
            let selected: Vec<Sample> = chosen.iter().map(|&k| samples[k].clone()).collect();
            let vg = Variogram {
                anisotropy: Some(aniso),
                ..vg.clone()
            };
            estimator(target, &selected, &vg).ok()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::{Kind, krige};
    use variogram::Model;

    fn close(a: f64, b: f64) -> bool {
        let d = (a - b).rem_euclid(180.0);
        d.min(180.0 - d) < 1.0
    }

    #[test]
    fn grid_gradient_finds_the_layering() {
        let geometry = Geometry {
            origin: [0.0; 3],
            size: [1.0; 3],
            count: [40, 40, 1],
            rotation: [0.0; 3],
        };
        let values: Vec<f64> = (0..1600)
            .map(|c| {
                let [i, j, _] = geometry.ijk(c);
                let (x, y) = (i as f64, j as f64);
                (0.3 * (x * 30f64.to_radians().cos() - y * 30f64.to_radians().sin())).sin()
            })
            .collect();
        let lva = LocalAnisotropy::from_grid(&geometry, &values, 2, None).unwrap();
        let [azimuth, dip, _] = lva.angles[820];
        assert!(close(azimuth, 30.0), "azimuth {azimuth}");
        assert!(dip.abs() < 1e-6);
    }

    #[test]
    fn points_along_a_line_point_along_it() {
        let coords: Vec<Point> = (0..50)
            .map(|i| (i as f64, i as f64, 0.1 * ((i * 7) % 3) as f64))
            .collect();
        let lva = LocalAnisotropy::from_points(&coords, 10, None).unwrap();
        assert!(close(lva.angles[25][0], 45.0));
        assert!(lva.ratios[25][0] < 0.2);
    }

    #[test]
    fn mesh_dip_and_strike() {
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (10.0, 0.0, -10.0),
            (10.0, 10.0, -10.0),
        ];
        let triangles = vec![(0, 1, 2), (1, 3, 2)];
        let targets = vec![(5.0, 5.0, -5.0)];
        let dip =
            LocalAnisotropy::from_mesh(&vertices, &triangles, &targets, MeshMajor::Dip, [1.0, 0.1])
                .unwrap();
        assert!(close(dip.angles[0][0], 90.0) && (dip.angles[0][1] - 45.0).abs() < 1e-6);
        let strike = LocalAnisotropy::from_mesh(
            &vertices,
            &triangles,
            &targets,
            MeshMajor::Strike,
            [1.0, 0.1],
        )
        .unwrap();
        assert!(close(strike.angles[0][0], 0.0) && strike.angles[0][1].abs() < 1e-6);
    }

    #[test]
    fn smoothing_keeps_a_uniform_field() {
        let coords: Vec<Point> = (0..30).map(|i| (i as f64, 0.0, 0.0)).collect();
        let lva = LocalAnisotropy::new(
            coords.clone(),
            vec![[60.0, 10.0, 20.0]; 30],
            vec![[0.5, 0.2]; 30],
        )
        .unwrap();
        let smooth = lva.smooth(5.0);
        let (a, b) = (lva.anisotropy(10).matrix(), smooth.anisotropy(10).matrix());
        assert!((a.transpose() * a - b.transpose() * b).norm() < 1e-9);
    }

    #[test]
    fn smoothing_2d_fields_keeps_axes_horizontal() {
        let coords: Vec<Point> = (0..30).map(|i| (i as f64, 0.0, 0.0)).collect();
        let angles = (0..30).map(|i| [20.0 + i as f64, 0.0, 0.0]).collect();
        let lva = LocalAnisotropy::new(coords, angles, vec![[0.3, 1.0]; 30]).unwrap();
        let smooth = lva.smooth(3.0);
        for (i, a) in smooth.angles.iter().enumerate() {
            assert!(a[1].abs() < 1e-6, "dip {}", a[1]);
            assert!(
                close(a[0], 20.0 + i as f64) || i < 3 || i > 26,
                "azimuth {} at {i}",
                a[0]
            );
        }
    }

    #[test]
    fn local_search_matches_the_rotated_tree() {
        let samples: Vec<Sample> = (0..400)
            .map(|i| {
                let j = (i as f64 * 0.618).fract();
                Sample::new(
                    (
                        (i * 37 % 101) as f64 + j,
                        (i * 53 % 97) as f64 + 0.3 * j,
                        0.0,
                    ),
                    0.0,
                )
            })
            .collect();
        let local = LocalAnisotropy::new(
            vec![(0.0, 0.0, 0.0)],
            vec![[30.0, 0.0, 0.0]],
            vec![[0.4, 1.0]],
        )
        .unwrap();
        let aniso = local.anisotropy(0);
        let search = Search {
            min_samples: 1,
            max_samples: 12,
            radius: 30.0,
            ..Default::default()
        };
        let iso = SearchTree::new(&samples, &search, None);
        let rotated = SearchTree::new(
            &samples,
            &Search {
                anisotropy: Some(aniso.clone()),
                ..search.clone()
            },
            None,
        );
        for t in [(50.3, 40.1, 0.0), (10.7, 80.2, 0.0), (90.0, 5.5, 0.0)] {
            assert_eq!(
                iso.neighbors_within(&t, &aniso).unwrap(),
                rotated.neighbors(&t).unwrap()
            );
        }
    }

    #[test]
    fn local_kriging_with_a_constant_field_is_global_kriging() {
        let samples: Vec<Sample> = (0..300)
            .map(|i| {
                let (x, y) = ((i * 37 % 101) as f64, (i * 53 % 97) as f64);
                Sample::new((x, y, 0.0), (x / 10.0).sin() + y / 50.0)
            })
            .collect();
        let targets: Vec<Point> = (0..100)
            .map(|i| ((i % 10) as f64 * 9.5, (i / 10) as f64 * 9.5, 0.0))
            .collect();
        let local = LocalAnisotropy::new(
            targets.clone(),
            vec![[30.0, 0.0, 0.0]; 100],
            vec![[0.4, 1.0]; 100],
        )
        .unwrap();
        let search = Search {
            min_samples: 1,
            max_samples: 16,
            radius: 60.0,
            ..Default::default()
        };
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let ours = estimate_many_local(&targets, &local, &samples, &search, &vg, |t, s, v| {
            krige(Kind::Ordinary, t, s, v)
        })
        .unwrap();
        let global = Variogram {
            anisotropy: Some(local.anisotropy(0)),
            ..vg.clone()
        };
        let reference = crate::estimate_many(&targets, &samples, &search, Some(&global), |t, s| {
            krige(Kind::Ordinary, t, s, &global)
        });
        for (a, b) in ours.iter().zip(&reference) {
            assert!((a.as_ref().unwrap().value - b.as_ref().unwrap().value).abs() < 1e-9);
        }
    }
}
