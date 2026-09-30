//! Locally varying anisotropy: orientation fields derived from a gridded
//! attribute, a point cloud or a mesh, and estimation that uses the local
//! anisotropy of each target for both its variogram and its search.

use boitata_core::{Geometry, angles_from_axes, block_frame};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use variogram::{Angles, Anisotropy, Variogram};

use crate::Sample;
use crate::error::{EstimError, Result};
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
    /// Multiplier of every variogram range and of the search radius, > 0.
    pub scales: Vec<f64>,
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
            scales: vec![1.0; coords.len()],
            coords,
            angles,
            ratios,
        })
    }

    /// This field with range multipliers `scales`, one per location.
    pub fn with_scales(mut self, scales: Vec<f64>) -> Result<Self> {
        if scales.len() != self.len() {
            return Err(invalid("one scale per location"));
        }
        if scales.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(invalid("scales must be positive"));
        }
        self.scales = scales;
        Ok(self)
    }

    /// World-to-(major, semi-major, minor) rotation at each location.
    pub fn rotations(&self) -> Vec<Matrix3<f64>> {
        self.angles
            .iter()
            .map(|&[a, d, r]| boitata_core::rotation_matrix(a, d, r))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Anisotropy at location `i`: major axis `scales[i]`, the others that
    /// times their ratios, so every variogram range is multiplied by the scale.
    pub fn anisotropy(&self, i: usize) -> Anisotropy {
        let [azimuth, dip, rake] = self.angles[i];
        let [semi, minor] = self.ratios[i];
        let s = self.scales[i];
        Anisotropy::new(Angles {
            azimuth,
            dip,
            rake,
            major: s,
            semi: s * semi,
            minor: s * minor,
        })
        .expect("ratios and scales are positive")
    }

    fn from_axes(coords: Vec<Point>, axes: Vec<(Vector3<f64>, Vector3<f64>, [f64; 2])>) -> Self {
        let (angles, ratios): (Vec<_>, Vec<_>) = axes
            .into_iter()
            .map(|(major, semi, ratios)| {
                let (major, semi) = frame(major, semi);
                (angles_from_axes(major.into(), semi.into()), ratios)
            })
            .unzip();
        Self {
            scales: vec![1.0; coords.len()],
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
    /// nearest neighbors (largest spread is the major axis).
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
        let mut centers = Vec::with_capacity(triangles.len());
        for &(a, b, c) in triangles {
            if a.max(b).max(c) >= vertices.len() {
                return Err(invalid("triangle index outside the vertices"));
            }
            let n = (v(b) - v(a)).cross(&(v(c) - v(a)));
            let center = (v(a) + v(b) + v(c)) / 3.0;
            normals.push(if n.z < 0.0 { -n } else { n });
            centers.push(Sample::new((center.x, center.y, center.z), 0.0));
        }
        let tree = SearchTree::new(
            &centers,
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
    /// the major and semi-major axes, the ratios and the scales.
    pub fn smooth(&self, radius: f64) -> Self {
        let axes: Vec<(Matrix3<f64>, Matrix3<f64>)> = (0..self.len())
            .map(|i| {
                let [a, d, r] = self.angles[i];
                let m = boitata_core::rotation_matrix(a, d, r);
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
        let (averaged, scales): (Vec<_>, Vec<_>) = self
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
                let scale = near.iter().map(|&i| self.scales[i] / n).sum::<f64>();
                let ratios = ratios.map(|r| r.min(1.0));
                ((top(&major), top(&semi), ratios), scale)
            })
            .unzip();
        Self {
            scales,
            ..Self::from_axes(self.coords.clone(), averaged)
        }
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
            scales: nearest.iter().map(|&i| self.scales[i]).collect(),
        }
    }
}

/// Settings of [`local_parameters`].
#[derive(Debug, Clone)]
pub struct WindowFit {
    /// Radius of the moving window around each node.
    pub window: f64,
    /// Lag-bin width and largest pair distance.
    pub lag: f64,
    pub max_lag: f64,
    /// Direction sectors over 180 degrees in the major/semi-major plane.
    pub sectors: usize,
    /// Fewest pairs in a window for a fit; fewer keep the base parameters.
    pub min_pairs: usize,
}

/// Experimental variogram of one window: per (sector, lag) bin the pair
/// count, the squared-difference sum, the mean (major, semi) separation,
/// folded onto one side, and the mean squared minor separation; and the
/// variance of the window's values.
struct Bins {
    count: Vec<f64>,
    sq: Vec<f64>,
    along: Vec<[f64; 2]>,
    across: Vec<f64>,
    variance: f64,
}

const SCALES: (f64, f64) = (0.1, 10.0);

impl Bins {
    fn gather(
        near: &[usize],
        coords: &[Point],
        values: &[f64],
        frames: &[Matrix3<f64>],
        params: &WindowFit,
    ) -> Self {
        let lags = ((params.max_lag / params.lag).ceil() as usize).max(1);
        let n = params.sectors * lags;
        let mut b = Self {
            count: vec![0.0; n],
            sq: vec![0.0; n],
            along: vec![[0.0; 2]; n],
            across: vec![0.0; n],
            variance: 0.0,
        };
        let k = near.len().max(1) as f64;
        let mean = near.iter().map(|&i| values[i]).sum::<f64>() / k;
        b.variance = near
            .iter()
            .map(|&i| (values[i] - mean).powi(2))
            .sum::<f64>()
            / k;
        let width = std::f64::consts::PI / params.sectors as f64;
        for (k, &i) in near.iter().enumerate() {
            let p = coords[i];
            for &j in &near[k + 1..] {
                let q = coords[j];
                let d = Vector3::new(q.0 - p.0, q.1 - p.1, q.2 - p.2);
                let dist = d.norm();
                if dist == 0.0 || dist > params.max_lag {
                    continue;
                }
                let mut c = frames[i] * d;
                if (-c.y).atan2(c.x) < 0.0 {
                    c = -c;
                }
                let psi = (-c.y).atan2(c.x).rem_euclid(std::f64::consts::PI);
                let sector = ((psi / width) as usize).min(params.sectors - 1);
                let lag = ((dist / params.lag) as usize).min(lags - 1);
                let at = sector * lags + lag;
                b.count[at] += 1.0;
                b.sq[at] += (values[i] - values[j]).powi(2);
                b.along[at][0] += c.x;
                b.along[at][1] += c.y;
                b.across[at] += c.z * c.z;
            }
        }
        for at in 0..n {
            if b.count[at] > 0.0 {
                let w = b.count[at];
                b.along[at] = b.along[at].map(|v| v / w);
                b.across[at] /= w;
                b.sq[at] /= 2.0 * w;
            }
        }
        b
    }

    fn pairs(&self) -> usize {
        self.count.iter().sum::<f64>() as usize
    }

    /// Weighted squared misfit of the shape of `vg`, with the window's
    /// variance as sill, turned by `theta` (radians) in the plane, with
    /// semi-major ratio `ratio`, minor ratio `minor` and ranges times `scale`.
    fn misfit(&self, vg: &Variogram, theta: f64, ratio: f64, minor: f64, scale: f64) -> f64 {
        let (s, c) = theta.sin_cos();
        let sill = self.variance / vg.total_sill();
        let mut sum = 0.0;
        for at in 0..self.count.len() {
            let w = self.count[at];
            if w == 0.0 {
                continue;
            }
            let [m, t] = self.along[at];
            let a = m * c - t * s;
            let b = m * s + t * c;
            let h = ((a / scale).powi(2)
                + (b / (scale * ratio)).powi(2)
                + self.across[at] / (scale * minor).powi(2))
            .sqrt();
            let g = sill * vg.gamma(h);
            sum += w * (self.sq[at] / g - 1.0).powi(2);
        }
        sum
    }

    /// Best (theta, ratio, scale): a coarse grid, then a pattern search with
    /// halving steps. `turn` false keeps theta at 0.
    fn fit(&self, vg: &Variogram, minor: f64, turn: bool) -> (f64, f64, f64) {
        let f = |p: [f64; 3]| self.misfit(vg, p[0].to_radians(), p[1].exp(), minor, p[2].exp());
        let thetas: Vec<f64> = match turn {
            true => (0..18).map(|k| k as f64 * 10.0).collect(),
            false => vec![0.0],
        };
        let (lo_r, lo_s, hi_s) = (0.05f64.ln(), SCALES.0.ln(), SCALES.1.ln());
        let mut best = ([0.0, 0.0, 0.0], f64::INFINITY);
        for &t in &thetas {
            for r in 0..10 {
                for k in 0..17 {
                    let p = [
                        t,
                        lo_r * (1.0 - r as f64 / 9.0),
                        lo_s + (hi_s - lo_s) * k as f64 / 16.0,
                    ];
                    let v = f(p);
                    if v < best.1 {
                        best = (p, v);
                    }
                }
            }
        }
        let mut step = [
            if turn { 5.0 } else { 0.0 },
            -lo_r / 18.0,
            (hi_s - lo_s) / 32.0,
        ];
        for _ in 0..8 {
            for axis in 0..3 {
                if step[axis] == 0.0 {
                    continue;
                }
                for sign in [-1.0, 1.0] {
                    let mut p = best.0;
                    p[axis] += sign * step[axis];
                    p[1] = p[1].clamp(lo_r, 0.0);
                    p[2] = p[2].clamp(lo_s, hi_s);
                    let v = f(p);
                    if v < best.1 {
                        best = (p, v);
                    }
                }
            }
            step = step.map(|s| s / 2.0);
        }
        let [t, r, s] = best.0;
        (t.rem_euclid(180.0), r.exp(), s.exp())
    }
}

/// Moving-window fits of locally varying variogram parameters at `nodes`.
///
/// The pairs of samples within `params.window` of a node are binned by
/// distance and by direction in the plane of the major and semi-major axes
/// of a base frame, their separations read in the base frame of the tail.
/// The shape of `vg` (nugget and structures), rescaled to the variance of
/// the window's values, is fitted to these bins by least squares weighted by
/// the pair counts: without `field`
/// the base frame is the anisotropy of `vg` and the fit finds the in-plane
/// rotation, the semi-major ratio and a scale of every range; with `field`
/// the base frame at each sample and node is that of its nearest location
/// in `field`, the angles and the minor ratio stay, and the fit finds the
/// semi-major ratio and the scale. A window with fewer than
/// `params.min_pairs` pairs keeps the base parameters and scale 1. Scales
/// are within [0.1, 10], ratios within [0.05, 1].
pub fn local_parameters(
    nodes: &[Point],
    coords: &[Point],
    values: &[f64],
    vg: &Variogram,
    field: Option<&LocalAnisotropy>,
    params: &WindowFit,
) -> Result<LocalAnisotropy> {
    if coords.len() != values.len() {
        return Err(invalid("one value per sample"));
    }
    if !(params.window > 0.0 && params.lag > 0.0 && params.max_lag > 0.0) {
        return Err(invalid("window, lag and max_lag must be positive"));
    }
    if params.sectors == 0 {
        return Err(invalid("sectors must be at least 1"));
    }
    if vg.total_sill() <= 0.0 || !vg.is_stationary() {
        return Err(invalid("the variogram needs a positive, finite sill"));
    }
    if field.is_some_and(|f| f.is_empty()) {
        return Err(invalid("the anisotropy field is empty"));
    }
    let shape = Variogram {
        anisotropy: None,
        ..vg.clone()
    };
    let base = match &vg.anisotropy {
        Some(a) => {
            let g = &a.angles;
            let angles = [g.azimuth, g.dip, g.rake];
            let ratios = [g.semi / g.major, g.minor / g.major];
            LocalAnisotropy::new(vec![(0.0, 0.0, 0.0)], vec![angles], vec![ratios])?
        }
        None => LocalAnisotropy::new(vec![(0.0, 0.0, 0.0)], vec![[0.0; 3]], vec![[1.0; 2]])?,
    };
    let (at_samples, at_nodes) = match field {
        Some(f) => (f.at(coords), f.at(nodes)),
        None => (base.at(coords), base.at(nodes)),
    };
    let frames = at_samples.rotations();
    let samples: Vec<Sample> = coords.iter().map(|&p| Sample::new(p, 0.0)).collect();
    let search = Search {
        min_samples: 1,
        max_samples: coords.len().max(1),
        radius: params.window,
        ..Default::default()
    };
    let tree = SearchTree::new(&samples, &search, None);
    let node_frames = at_nodes.rotations();
    let (axes, scales): (Vec<_>, Vec<_>) = nodes
        .par_iter()
        .enumerate()
        .map(|(n, p)| {
            let [ratio, minor] = at_nodes.ratios[n];
            let mut near = tree.neighbors(p).unwrap_or_default();
            near.sort_unstable();
            let bins = Bins::gather(&near, coords, values, &frames, params);
            let (theta, ratio, scale) = if bins.pairs() < params.min_pairs {
                (0.0, ratio, 1.0)
            } else {
                bins.fit(&shape, minor, field.is_none())
            };
            let (s, c) = theta.to_radians().sin_cos();
            let back = node_frames[n].transpose();
            let major = back * Vector3::new(c, -s, 0.0);
            let semi = back * Vector3::new(s, c, 0.0);
            ((major, semi, [ratio, minor]), scale)
        })
        .unzip();
    LocalAnisotropy::from_axes(nodes.to_vec(), axes).with_scales(scales)
}

/// As [`crate::estimate_many`], with target `i` using `local[i]` for its
/// variogram and its search ellipsoid (`search.radius` along the major axis),
/// and `domains` as there.
pub fn estimate_many_local<F, T>(
    targets: &[Point],
    domains: Option<&[u32]>,
    local: &LocalAnisotropy,
    samples: &[Sample],
    search: &Search,
    vg: &Variogram,
    estimator: F,
) -> Result<Vec<Option<T>>>
where
    F: Fn(&Point, &[Sample], &Variogram) -> Result<T> + Sync,
    T: Send,
{
    estimate_many_local_with(
        targets, domains, local, samples, search, vg, estimator, None,
    )
}

/// As [`estimate_many_local`], ticking `progress` for each target estimated.
#[allow(clippy::too_many_arguments)]
pub fn estimate_many_local_with<F, T>(
    targets: &[Point],
    domains: Option<&[u32]>,
    local: &LocalAnisotropy,
    samples: &[Sample],
    search: &Search,
    vg: &Variogram,
    estimator: F,
    progress: Option<&boitata_core::Progress>,
) -> Result<Vec<Option<T>>>
where
    F: Fn(&Point, &[Sample], &Variogram) -> Result<T> + Sync,
    T: Send,
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
            let chosen = tree
                .neighbors_within(target, domains.map(|d| d[i]), &aniso)
                .ok()?;
            let selected = tree.take(target, Some(&aniso), &chosen, samples);
            let vg = Variogram {
                anisotropy: Some(aniso),
                ..vg.clone()
            };
            let result = estimator(target, &selected, &vg).ok();
            if let (Some(p), Some(_)) = (progress, &result) {
                p.inc();
            }
            result
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
                close(a[0], 20.0 + i as f64) || !(3..=26).contains(&i),
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
                iso.neighbors_within(&t, None, &aniso).unwrap(),
                rotated.neighbors(&t).unwrap()
            );
        }
    }

    /// Uniforms in [0, 1) from a seed.
    fn uniforms(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Unit-sill field with a Gaussian covariance of practical ranges `major`
    /// and `major * ratio`, major axis at `azimuth`: a sum of random waves.
    fn waves(seed: u64, azimuth: f64, major: f64, ratio: f64) -> impl Fn(&Point) -> f64 {
        let mut u = uniforms(seed);
        let mut normal = move || {
            let (a, b) = (u().max(1e-300), u());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        };
        let back = boitata_core::rotation_matrix(azimuth, 0.0, 0.0).transpose();
        let k = 400;
        let waves: Vec<(Vector3<f64>, f64)> = (0..k)
            .map(|_| {
                let w = Vector3::new(
                    normal() * 6f64.sqrt() / major,
                    normal() * 6f64.sqrt() / (major * ratio),
                    0.0,
                );
                (back * w, normal() * 1e3)
            })
            .collect();
        move |p| {
            let x = Vector3::new(p.0, p.1, p.2);
            waves
                .iter()
                .map(|(w, f)| (w.dot(&x) + f).cos())
                .sum::<f64>()
                * (2.0 / k as f64).sqrt()
        }
    }

    /// Azimuth 30, ranges 36 and 9 west of x = 60; azimuth 120, ranges 12
    /// and 4 east of it.
    fn two_regions(n: usize) -> (Vec<Point>, Vec<f64>) {
        let (west, east) = (waves(1, 30.0, 20.0, 0.25), waves(2, 120.0, 8.0, 0.4));
        let mut u = uniforms(3);
        let coords: Vec<Point> = (0..n).map(|_| (120.0 * u(), 80.0 * u(), 0.0)).collect();
        let values = coords
            .iter()
            .map(|p| if p.0 < 60.0 { west(p) } else { east(p) })
            .collect();
        (coords, values)
    }

    fn gaussian(range: f64) -> Variogram {
        Variogram {
            nugget: 0.01,
            ..Variogram::single(Model::Gaussian, 0.99, range)
        }
    }

    #[test]
    fn window_fits_find_each_region() {
        let (coords, values) = two_regions(1500);
        let params = WindowFit {
            window: 25.0,
            lag: 2.0,
            max_lag: 15.0,
            sectors: 8,
            min_pairs: 100,
        };
        let nodes = vec![(28.0, 40.0, 0.0), (92.0, 40.0, 0.0)];
        let local =
            local_parameters(&nodes, &coords, &values, &gaussian(12.0), None, &params).unwrap();
        let expected = [(30.0, 0.25, 20.0 / 12.0), (120.0, 0.4, 8.0 / 12.0)];
        for (i, (azimuth, ratio, scale)) in expected.into_iter().enumerate() {
            let d = (local.angles[i][0] - azimuth).rem_euclid(180.0);
            assert!(d.min(180.0 - d) < 12.0, "azimuth {:?}", local.angles[i]);
            assert!(
                (local.ratios[i][0] - ratio).abs() < 0.12,
                "ratio {:?}",
                local.ratios[i]
            );
            assert!(
                (local.scales[i] / scale - 1.0).abs() < 0.3,
                "scale {}",
                local.scales[i]
            );
        }
    }

    #[test]
    fn window_fits_do_not_depend_on_the_thread_count() {
        let (coords, values) = two_regions(800);
        let nodes: Vec<Point> = (0..24)
            .map(|i| {
                (
                    5.0 + 20.0 * (i % 6) as f64,
                    10.0 + 20.0 * (i / 6) as f64,
                    0.0,
                )
            })
            .collect();
        let params = WindowFit {
            window: 25.0,
            lag: 2.0,
            max_lag: 15.0,
            sectors: 8,
            min_pairs: 100,
        };
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let l =
                        local_parameters(&nodes, &coords, &values, &gaussian(12.0), None, &params)
                            .unwrap();
                    (l.angles, l.ratios, l.scales)
                })
        };
        assert_eq!(run(1), run(8));
    }

    #[test]
    fn local_parameters_beat_global_kriging_on_held_out_samples() {
        let (coords, values) = two_regions(1500);
        let samples: Vec<Sample> = coords[..1200]
            .iter()
            .zip(&values)
            .map(|(&p, &v)| Sample::new(p, v))
            .collect();
        let (targets, truth) = (&coords[1200..], &values[1200..]);
        let nodes: Vec<Point> = (0..48)
            .map(|i| {
                (
                    5.0 + 10.0 * (i % 12) as f64,
                    10.0 + 20.0 * (i / 12) as f64,
                    0.0,
                )
            })
            .collect();
        let params = WindowFit {
            window: 25.0,
            lag: 2.0,
            max_lag: 15.0,
            sectors: 8,
            min_pairs: 100,
        };
        let vg = gaussian(12.0);
        let train: Vec<Point> = samples.iter().map(|s| s.loc).collect();
        let train_values: Vec<f64> = samples.iter().map(|s| s.value).collect();
        let local = local_parameters(&nodes, &train, &train_values, &vg, None, &params)
            .unwrap()
            .smooth(12.0);
        let search = Search {
            min_samples: 1,
            max_samples: 16,
            radius: 40.0,
            ..Default::default()
        };
        let ok = |t: &Point, s: &[Sample], v: &Variogram| krige(Kind::Ordinary, t, s, v);
        let ours = estimate_many_local(
            targets,
            None,
            &local.at(targets),
            &samples,
            &search,
            &vg,
            ok,
        )
        .unwrap();
        let global_search = Search {
            radius: 60.0,
            ..search.clone()
        };
        let global = crate::estimate_many(targets, None, &samples, &global_search, None, |t, s| {
            krige(Kind::Ordinary, t, s, &vg)
        });
        let mse = |e: &[Option<crate::Estimate>]| {
            e.iter()
                .zip(truth)
                .map(|(e, v)| (e.as_ref().unwrap().value - v).powi(2))
                .sum::<f64>()
                / truth.len() as f64
        };
        let (ours, global) = (mse(&ours), mse(&global));
        assert!(ours < 0.8 * global, "local {ours} vs global {global}");
    }

    #[test]
    fn a_constant_scale_is_a_global_range_and_radius() {
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
        .unwrap()
        .with_scales(vec![1.5; 100])
        .unwrap();
        let search = Search {
            min_samples: 1,
            max_samples: 16,
            radius: 40.0,
            ..Default::default()
        };
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let ours =
            estimate_many_local(&targets, None, &local, &samples, &search, &vg, |t, s, v| {
                krige(Kind::Ordinary, t, s, v)
            })
            .unwrap();
        let global = Variogram {
            anisotropy: Some(
                Anisotropy::new(Angles {
                    azimuth: 30.0,
                    dip: 0.0,
                    rake: 0.0,
                    major: 1.0,
                    semi: 0.4,
                    minor: 1.0,
                })
                .unwrap(),
            ),
            ..Variogram::single(Model::Spherical, 1.0, 75.0)
        };
        let wide = Search {
            radius: 60.0,
            ..search.clone()
        };
        let reference =
            crate::estimate_many(&targets, None, &samples, &wide, Some(&global), |t, s| {
                krige(Kind::Ordinary, t, s, &global)
            });
        for (a, b) in ours.iter().zip(&reference) {
            assert!((a.as_ref().unwrap().value - b.as_ref().unwrap().value).abs() < 1e-9);
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
        let ours =
            estimate_many_local(&targets, None, &local, &samples, &search, &vg, |t, s, v| {
                krige(Kind::Ordinary, t, s, v)
            })
            .unwrap();
        let global = Variogram {
            anisotropy: Some(local.anisotropy(0)),
            ..vg.clone()
        };
        let reference =
            crate::estimate_many(&targets, None, &samples, &search, Some(&global), |t, s| {
                krige(Kind::Ordinary, t, s, &global)
            });
        for (a, b) in ours.iter().zip(&reference) {
            assert!((a.as_ref().unwrap().value - b.as_ref().unwrap().value).abs() < 1e-9);
        }
    }
}
