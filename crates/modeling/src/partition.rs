//! Partition-of-unity RBF: the dense solve of [`crate::rbf`] split into
//! overlapping local solves and blended back into one field.
//!
//! The dense system costs `O(n³)` to factor and `O(n²)` to store, which ends
//! it somewhere past a few thousand rows. Here the constraint locations, in the
//! same reduced and normalized space the dense fit uses, are cut by an octree
//! until the ball around each cell holds at most about `patch` rows. Each leaf
//! gets a ball of radius [`OVERLAP`] × its half-diagonal, so neighbouring balls
//! overlap, and a dense [`Rbf`] fitted to every value, boundary and derivative
//! row inside it. The field is the Shepard blend
//!
//! ```text
//! f(p) = Σ wᵢ(p) fᵢ(p) / Σ wᵢ(p),   wᵢ(p) = ψ(‖q − cᵢ‖ / Rᵢ)
//! ```
//!
//! with `ψ(t) = (1 − t)⁴(4t + 1)` the Wendland C² function, zero from `t = 1`
//! on, so a point only sums the few patches whose balls contain it. The
//! gradient is the product rule on that quotient, weights included, so it is
//! the exact derivative of the blended value.
//!
//! Three invariants hold the construction together:
//!
//! - **coverage**: the leaves tile the root cube `[−1, 1]³`, which holds every
//!   normalized constraint location, and each leaf's ball strictly contains its
//!   cell, so `Σ wᵢ > 0` everywhere in the cube. Weights are taken at the point
//!   clamped to the cube, so outside it each patch extrapolates its own fit
//!   under the blend of the nearest face and the field stays continuous.
//! - **exactness**: a patch fits every row strictly inside its ball, and its
//!   weight is zero outside it, so at a value constraint every patch with a
//!   non-zero weight returns that value and so does the blend.
//! - **enough data**: a ball holding fewer than half of `patch` value rows (or
//!   no sample, only contact picks) grows to its nearest ones, so no local
//!   system is short of rows for its drift or blind to which side is which.
//!
//! Patches are fitted in parallel, each into its own slot, and a point sums its
//! patches in index order, so the field does not depend on the thread count.
//!
//! References: Wendland (2002), "Fast evaluation of radial basis functions:
//! methods based on partition of unity", in *Approximation Theory X*; Ohtake,
//! Belyaev, Alexa, Turk & Seidel (2003), "Multi-level partition of unity
//! implicits", *ACM Transactions on Graphics* 22(3); Carr, Beatson, Cherrie,
//! Mitchell, Fright, McCallum & Evans (2001), "Reconstruction and
//! representation of 3D objects with radial basis functions", *SIGGRAPH*.

use std::collections::BTreeMap;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::aniso::{AnisoTransform, reduce};
use crate::constraint::{ConstraintSet, ValueKind};
use crate::error::{ModelError, Result};
use crate::rbf::{Kernel, Rbf, RbfSpec, basis_len, normalization, normalize, value_scale};

/// Constraint rows up to which `patch: None` solves densely: a 2000-row LU
/// takes about a second and 32 MB, and the cubic cost makes the next doubling
/// eight times that.
pub const DENSE_ROWS: usize = 2000;
/// Rows per patch when `patch: None` picks the partition of unity.
pub const DEFAULT_PATCH: usize = 400;
/// Smallest `patch` accepted, so the octree cannot chase single points.
pub const MIN_PATCH: usize = 10;
/// Patch radius over the half-diagonal of its cell. Above 1, so a cell sits
/// strictly inside its own ball — the coverage invariant.
pub const OVERLAP: f64 = 1.25;
const MAX_DEPTH: usize = 12;

/// The field `engine="rbf"` fits: one dense solve, or a partition of unity.
#[derive(Debug)]
pub enum RbfField {
    Dense(Rbf),
    Partition(PartitionRbf),
}

impl RbfField {
    /// Fit `set`, choosing the solver from `spec.patch`.
    pub fn fit(set: &ConstraintSet, spec: &RbfSpec) -> Result<Self> {
        let rows = set.values.len() + set.derivatives.len();
        match spec.patch {
            Some(0) => Rbf::fit_constraints(set, spec).map(Self::Dense),
            None if rows <= DENSE_ROWS => Rbf::fit_constraints(set, spec).map(Self::Dense),
            None => PartitionRbf::fit(set, spec, DEFAULT_PATCH).map(Self::Partition),
            Some(n) => PartitionRbf::fit(set, spec, n).map(Self::Partition),
        }
    }

    pub fn value(&self, p: &[f64; 3]) -> f64 {
        match self {
            Self::Dense(f) => f.value(p),
            Self::Partition(f) => f.value(p),
        }
    }

    pub fn gradient(&self, p: &[f64; 3]) -> [f64; 3] {
        match self {
            Self::Dense(f) => f.gradient(p),
            Self::Partition(f) => f.gradient(p),
        }
    }

    /// Local solves behind the field; 1 for the dense one.
    pub fn patch_count(&self) -> usize {
        match self {
            Self::Dense(_) => 1,
            Self::Partition(f) => f.patch_count(),
        }
    }
}

#[derive(Debug)]
struct Patch {
    center: [f64; 3],
    radius: f64,
    fit: Rbf,
}

/// Patch centers whose radii share a power of two, searched with the largest of
/// them, so one grown patch does not widen every query.
#[derive(Debug)]
struct Band {
    radius: f64,
    tree: ImmutableKdTree<f64, 3>,
    members: Vec<usize>,
}

/// A partition-of-unity RBF field. See the module docs.
#[derive(Debug)]
pub struct PartitionRbf {
    transform: Option<AnisoTransform>,
    origin: [f64; 3],
    scale: f64,
    /// World → normalized linear map, as in [`Rbf`].
    jacobian: Matrix3<f64>,
    patches: Vec<Patch>,
    bands: Vec<Band>,
}

fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn tree(points: &[[f64; 3]]) -> Result<Option<ImmutableKdTree<f64, 3>>> {
    if points.is_empty() {
        return Ok(None);
    }
    ImmutableKdTree::new_from_slice(points)
        .map(Some)
        .map_err(|e| ModelError::InvalidParameter(format!("{e:?}")))
}

/// Indices of `points` strictly within `radius` of `center`, ascending.
fn within(
    tree: &Option<ImmutableKdTree<f64, 3>>,
    points: &[[f64; 3]],
    center: &[f64; 3],
    radius: f64,
) -> Vec<usize> {
    let Some(tree) = tree else {
        return Vec::new();
    };
    let r2 = radius * radius;
    let mut found: Vec<usize> = tree
        .query(center)
        .within::<SquaredEuclidean<f64>>(r2)
        .execute()
        .iter()
        .map(|n| n.item as usize)
        .filter(|&i| dist2(&points[i], center) < r2)
        .collect();
    found.sort_unstable();
    found
}

/// Distance to the `k`-th nearest of `points`, nudged up so that point falls
/// strictly inside a ball of that radius.
fn reach(tree: &Option<ImmutableKdTree<f64, 3>>, center: &[f64; 3], k: usize) -> f64 {
    let Some(tree) = tree else {
        return 0.0;
    };
    let Some(k) = std::num::NonZero::new(k) else {
        return 0.0;
    };
    tree.query(center)
        .nearest_n::<SquaredEuclidean<f64>>(k)
        .execute()
        .last()
        .map_or(0.0, |n| n.distance.sqrt() * (1.0 + 1e-9))
}

/// `φ(r / s) = φ(r) / sᵏ`: how the kernel matrix scales with the normalization.
/// The thin-plate kernel is homogeneous only up to a quadratic term, so its
/// degree is approximate.
fn homogeneity(kernel: Kernel) -> i32 {
    match kernel {
        Kernel::Biharmonic => 1,
        Kernel::ThinPlate => 2,
        Kernel::Triharmonic => 3,
    }
}

/// `ψ(t) = (1 − t)⁴(4t + 1)` and `ψ'(t)/t = −20(1 − t)³`, the factor of the
/// separation vector in the weight's gradient.
fn wendland(t: f64) -> (f64, f64) {
    let s = 1.0 - t;
    (s.powi(4) * (4.0 * t + 1.0), -20.0 * s.powi(3))
}

impl PartitionRbf {
    /// Fit `set` with local solves of about `patch` rows each.
    pub fn fit(set: &ConstraintSet, spec: &RbfSpec, patch: usize) -> Result<Self> {
        set.validate()?;
        if patch < MIN_PATCH {
            return Err(ModelError::InvalidParameter(format!(
                "patch must be 0 (dense) or at least {MIN_PATCH}, got {patch}"
            )));
        }
        let transform = spec
            .anisotropy
            .as_ref()
            .map(AnisoTransform::new)
            .transpose()?;
        let reduced_values: Vec<[f64; 3]> = set
            .values
            .iter()
            .map(|v| reduce(transform.as_ref(), &v.at))
            .collect();
        let reduced_derivs: Vec<[f64; 3]> = set
            .derivatives
            .iter()
            .map(|d| reduce(transform.as_ref(), &d.at))
            .collect();
        let (origin, scale) = normalization(&reduced_values, &reduced_derivs);
        let values: Vec<[f64; 3]> = reduced_values
            .iter()
            .map(|r| normalize(r, &origin, scale))
            .collect();
        let derivs: Vec<[f64; 3]> = reduced_derivs
            .iter()
            .map(|r| normalize(r, &origin, scale))
            .collect();
        let samples: Vec<[f64; 3]> = set
            .values
            .iter()
            .zip(&values)
            .filter(|(v, _)| v.kind == ValueKind::Sample)
            .map(|(_, q)| *q)
            .collect();
        let value_tree = tree(&values)?;
        let deriv_tree = tree(&derivs)?;
        let sample_tree = tree(&samples)?;

        let degree = spec.drift_degree.max(spec.kernel.min_drift_degree()).min(1);
        let fewest = (patch / 2).max(basis_len(degree) + 1).min(values.len());
        let reach_of = |h: f64| OVERLAP * 3f64.sqrt() * h;
        let rows_in = |c: &[f64; 3], r: f64| {
            within(&value_tree, &values, c, r).len() + within(&deriv_tree, &derivs, c, r).len()
        };

        // Depth-first, children in a fixed order: the patch list, and so the
        // summation order at every point, is a function of the data alone.
        let mut cells = vec![([0.0; 3], 1.0, 0)];
        let mut leaves = Vec::new();
        while let Some((c, h, depth)) = cells.pop() {
            if depth < MAX_DEPTH && rows_in(&c, reach_of(h)) > patch {
                for octant in (0..8).rev() {
                    let child = [0, 1, 2].map(|a| {
                        c[a] + if octant >> a & 1 == 1 {
                            h / 2.0
                        } else {
                            -h / 2.0
                        }
                    });
                    cells.push((child, h / 2.0, depth + 1));
                }
            } else {
                leaves.push((c, h));
            }
        }

        let spread = value_scale(&set.values.iter().map(|v| v.value).collect::<Vec<_>>());
        let fits: Vec<Result<Patch>> = leaves
            .par_iter()
            .map(|&(center, h)| {
                let mut radius = reach_of(h);
                if within(&value_tree, &values, &center, radius).len() < fewest {
                    radius = radius.max(reach(&value_tree, &center, fewest));
                }
                if sample_tree.is_some()
                    && within(&sample_tree, &samples, &center, radius).is_empty()
                {
                    radius = radius.max(reach(&sample_tree, &center, 1));
                }
                let mut local = ConstraintSet::new();
                let value_rows = within(&value_tree, &values, &center, radius);
                let deriv_rows = within(&deriv_tree, &derivs, &center, radius);
                local.values = value_rows.iter().map(|&i| set.values[i]).collect();
                local.derivatives = deriv_rows.iter().map(|&i| set.derivatives[i]).collect();

                // The local solve normalizes to its own extent and value spread;
                // rescaling the ridges keeps them what they are in the whole
                // model. Every block of the system scales by the same factor.
                let (_, local_extent) = normalization(
                    &value_rows
                        .iter()
                        .map(|&i| reduced_values[i])
                        .collect::<Vec<_>>(),
                    &deriv_rows
                        .iter()
                        .map(|&i| reduced_derivs[i])
                        .collect::<Vec<_>>(),
                );
                let local_spread =
                    value_scale(&local.values.iter().map(|v| v.value).collect::<Vec<_>>());
                let factor =
                    (scale / local_extent).powi(homogeneity(spec.kernel)) * spread / local_spread;
                let local_spec = RbfSpec {
                    smoothing: spec.smoothing * factor,
                    boundary_tolerance: spec.boundary_tolerance * factor,
                    gradient_nugget: spec.gradient_nugget * factor,
                    ..spec.clone()
                };
                let fit = Rbf::fit_constraints(&local, &local_spec)?;
                Ok(Patch {
                    center,
                    radius,
                    fit,
                })
            })
            .collect();
        let patches = fits.into_iter().collect::<Result<Vec<_>>>()?;

        let mut grouped: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        for (i, p) in patches.iter().enumerate() {
            grouped
                .entry(p.radius.log2().ceil() as i32)
                .or_default()
                .push(i);
        }
        let bands = grouped
            .into_values()
            .map(|members| {
                let centers: Vec<[f64; 3]> = members.iter().map(|&i| patches[i].center).collect();
                Ok(Band {
                    radius: members
                        .iter()
                        .map(|&i| patches[i].radius)
                        .fold(0.0, f64::max),
                    tree: tree(&centers)?.expect("a band has members"),
                    members,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let jacobian = match transform.as_ref() {
            Some(t) => t.matrix() / scale,
            None => Matrix3::identity() / scale,
        };
        Ok(Self {
            transform,
            origin,
            scale,
            jacobian,
            patches,
            bands,
        })
    }

    /// Patches with a non-zero weight at `p`, in index order, with the weight
    /// and its gradient in normalized coordinates.
    fn weights(&self, p: &[f64; 3]) -> Vec<(usize, f64, Vector3<f64>)> {
        let q = normalize(
            &reduce(self.transform.as_ref(), p),
            &self.origin,
            self.scale,
        );
        let c = q.map(|x| x.clamp(-1.0, 1.0));
        let mut found = Vec::new();
        for band in &self.bands {
            for n in band
                .tree
                .query(&c)
                .within::<SquaredEuclidean<f64>>(band.radius * band.radius)
                .execute()
            {
                let i = band.members[n.item as usize];
                let patch = &self.patches[i];
                let d2 = dist2(&c, &patch.center);
                let r2 = patch.radius * patch.radius;
                if d2 < r2 {
                    let (w, dw) = wendland(d2.sqrt() / patch.radius);
                    let mut g = Vector3::zeros();
                    for a in 0..3 {
                        // Clamped coordinates do not move with the point.
                        if q[a] == c[a] {
                            g[a] = dw * (c[a] - patch.center[a]) / r2;
                        }
                    }
                    found.push((i, w, g));
                }
            }
        }
        found.sort_unstable_by_key(|&(i, _, _)| i);
        found
    }

    /// Field value at a world-space point.
    pub fn value(&self, p: &[f64; 3]) -> f64 {
        let mut sum = 0.0;
        let mut total = 0.0;
        for (i, w, _) in self.weights(p) {
            sum += w * self.patches[i].fit.value(p);
            total += w;
        }
        sum / total
    }

    /// Field gradient at a world-space point, in field units per meter.
    pub fn gradient(&self, p: &[f64; 3]) -> [f64; 3] {
        let weights = self.weights(p);
        let values: Vec<f64> = weights
            .iter()
            .map(|&(i, _, _)| self.patches[i].fit.value(p))
            .collect();
        let total: f64 = weights.iter().map(|&(_, w, _)| w).sum();
        let f = weights
            .iter()
            .zip(&values)
            .map(|(&(_, w, _), v)| w * v)
            .sum::<f64>()
            / total;
        let mut local = Vector3::zeros();
        let mut blend = Vector3::zeros();
        for (&(i, w, dw), v) in weights.iter().zip(&values) {
            let g = self.patches[i].fit.gradient(p);
            local += Vector3::new(g[0], g[1], g[2]) * w;
            blend += dw * (v - f);
        }
        let world = (local + self.jacobian.transpose() * blend) / total;
        [world.x, world.y, world.z]
    }

    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn smooth(p: &[f64; 3]) -> f64 {
        (p[0] / 40.0).sin() * (p[1] / 55.0).cos() + 0.02 * p[2] + 0.3 * (p[2] / 30.0).sin()
    }

    fn cloud(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                [
                    rng.gen_range(0.0..200.0),
                    rng.gen_range(0.0..200.0),
                    rng.gen_range(0.0..100.0),
                ]
            })
            .collect()
    }

    fn samples(points: &[[f64; 3]]) -> ConstraintSet {
        let mut set = ConstraintSet::new();
        for p in points {
            set.push_sample(*p, smooth(p));
        }
        set
    }

    fn spec(kernel: Kernel, patch: Option<usize>) -> RbfSpec {
        RbfSpec {
            kernel,
            patch,
            ..RbfSpec::default()
        }
    }

    fn held_out() -> Vec<[f64; 3]> {
        let mut grid = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                for k in 0..6 {
                    grid.push([
                        10.0 + i as f64 * 16.0,
                        10.0 + j as f64 * 16.0,
                        8.0 + k as f64 * 16.0,
                    ]);
                }
            }
        }
        grid
    }

    /// Relative RMS of `a` against `b` over `grid`: the miss over the spread of
    /// `b`.
    fn relative(
        grid: &[[f64; 3]],
        a: impl Fn(&[f64; 3]) -> f64,
        b: impl Fn(&[f64; 3]) -> f64,
    ) -> f64 {
        let reference: Vec<f64> = grid.iter().map(&b).collect();
        let mean = reference.iter().sum::<f64>() / reference.len() as f64;
        let spread = reference.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
        let miss = grid
            .iter()
            .zip(&reference)
            .map(|(p, r)| (a(p) - r).powi(2))
            .sum::<f64>();
        (miss / spread).sqrt()
    }

    /// The theory check: on 1000 points of a smooth field, patches of 250 rows
    /// reproduce the dense solve on a held-out grid, interpolate the data, and
    /// stay within 30 % of the dense solve's own error from the true field.
    /// (1000 rather than more because the dense reference is an unoptimized LU
    /// in a debug build; 2000 points with patches of 300 come closer still.)
    ///
    /// The biharmonic's dense solve leans on data far away, which no patch
    /// sees, so it gets the looser tolerance: half a percent of the field's
    /// spread, below its own interpolation error.
    #[test]
    fn matches_the_dense_solve_and_interpolates_the_data() {
        let points = cloud(1000, 11);
        let set = samples(&points);
        let grid = held_out();
        for (kernel, tolerance) in [(Kernel::Biharmonic, 5e-3), (Kernel::Triharmonic, 1e-3)] {
            let dense = RbfField::fit(&set, &spec(kernel, Some(0))).unwrap();
            let pu = RbfField::fit(&set, &spec(kernel, Some(250))).unwrap();
            assert!(matches!(dense, RbfField::Dense(_)));
            assert!(pu.patch_count() > 8, "{} patches", pu.patch_count());

            let against_dense = relative(&grid, |p| pu.value(p), |p| dense.value(p));
            let dense_error = relative(&grid, |p| dense.value(p), smooth);
            let pu_error = relative(&grid, |p| pu.value(p), smooth);
            assert!(
                against_dense < tolerance,
                "{kernel:?}: relative RMS {against_dense:.2e} against the dense solve"
            );
            assert!(
                pu_error < 1.3 * dense_error,
                "{kernel:?}: error {pu_error:.2e} against dense {dense_error:.2e}"
            );

            let worst = points
                .iter()
                .map(|p| (pu.value(p) - smooth(p)).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-6, "{kernel:?}: misses the data by {worst:.2e}");
        }
    }

    #[test]
    fn the_gradient_is_the_derivative_of_the_blend() {
        let points = cloud(1500, 3);
        let pu =
            PartitionRbf::fit(&samples(&points), &spec(Kernel::Triharmonic, None), 120).unwrap();
        let eps = 1e-4;
        // The last probe lies outside the root cube, where the weights are
        // frozen along the clamped axes.
        for probe in [
            [50.0, 120.0, 40.0],
            [101.0, 99.0, 51.0],
            [260.0, -30.0, 50.0],
        ] {
            let g = pu.gradient(&probe);
            for a in 0..3 {
                let mut up = probe;
                let mut dn = probe;
                up[a] += eps;
                dn[a] -= eps;
                let fd = (pu.value(&up) - pu.value(&dn)) / (2.0 * eps);
                assert!(
                    (g[a] - fd).abs() < 1e-5 * (1.0 + fd.abs()),
                    "component {a} at {probe:?}: {} vs fd {fd}",
                    g[a]
                );
            }
        }
    }

    #[test]
    fn covers_points_far_outside_the_data() {
        let points = cloud(1500, 5);
        let pu =
            PartitionRbf::fit(&samples(&points), &spec(Kernel::Biharmonic, None), 100).unwrap();
        for probe in [[-1e4, 50.0, 50.0], [100.0, 100.0, 5e3], [1e5, 1e5, -1e5]] {
            assert!(pu.value(&probe).is_finite(), "{probe:?}");
        }
    }

    #[test]
    fn does_not_depend_on_the_thread_count() {
        let points = cloud(3000, 7);
        let set = samples(&points);
        let grid = held_out();
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let pu = RbfField::fit(&set, &spec(Kernel::Biharmonic, Some(200))).unwrap();
                    grid.par_iter()
                        .map(|p| (pu.value(p), pu.gradient(p)))
                        .collect::<Vec<_>>()
                })
        };
        let one = run(1);
        assert_eq!(one, run(4));
        assert_eq!(one, run(7));
    }

    #[test]
    fn small_models_stay_dense_and_large_ones_do_not() {
        let small = samples(&cloud(300, 1));
        let field = RbfField::fit(&small, &RbfSpec::default()).unwrap();
        let dense = Rbf::fit_constraints(&small, &RbfSpec::default()).unwrap();
        assert!(matches!(field, RbfField::Dense(_)));
        let probe = [33.0, 44.0, 55.0];
        assert_eq!(field.value(&probe).to_bits(), dense.value(&probe).to_bits());

        let large = samples(&cloud(DENSE_ROWS + 1, 1));
        assert!(matches!(
            RbfField::fit(&large, &RbfSpec::default()).unwrap(),
            RbfField::Partition(_)
        ));
        assert!(PartitionRbf::fit(&large, &RbfSpec::default(), 3).is_err());
    }

    /// Patches see structural rows too: a fold sampled by contacts and dips
    /// keeps both under the partition.
    #[test]
    fn honors_contacts_and_dips_inside_patches() {
        let mut rng = StdRng::seed_from_u64(9);
        let surface = |x: f64| 50.0 + 10.0 * (x / 30.0).sin();
        let mut set = ConstraintSet::new();
        for _ in 0..600 {
            let (x, y) = (rng.gen_range(0.0..200.0), rng.gen_range(0.0..200.0));
            set.push_boundary([x, y, surface(x)], 0.0);
            set.push_sample([x, y, surface(x) + 15.0], 1.0);
            set.push_sample([x, y, surface(x) - 15.0], -1.0);
        }
        for _ in 0..100 {
            let x: f64 = rng.gen_range(0.0..200.0);
            let slope = 10.0 / 30.0 * (x / 30.0).cos();
            let y = rng.gen_range(0.0..200.0);
            set.push_derivative([x, y, surface(x)], [1.0, 0.0, slope], 0.0)
                .unwrap();
        }
        let pu = PartitionRbf::fit(&set, &spec(Kernel::Triharmonic, None), 250).unwrap();
        assert!(pu.patch_count() > 4);
        for v in &set.values {
            assert!((pu.value(&v.at) - v.value).abs() < 1e-6, "{:?}", v.at);
        }
        let probe = [100.0, 100.0, surface(100.0)];
        let above = [100.0, 100.0, surface(100.0) + 5.0];
        assert!(pu.value(&probe).abs() < 0.05 && pu.value(&above) > 0.0);
    }

    /// The scale smoke test behind #554: run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn fits_fifty_thousand_points_in_seconds() {
        let points = cloud(50_000, 2);
        let set = samples(&points);
        let start = std::time::Instant::now();
        let field = RbfField::fit(&set, &RbfSpec::default()).unwrap();
        let fitted = start.elapsed();
        let grid = held_out();
        let start = std::time::Instant::now();
        let rms = (grid
            .par_iter()
            .map(|p| (field.value(p) - smooth(p)).powi(2))
            .sum::<f64>()
            / grid.len() as f64)
            .sqrt();
        println!(
            "50k points: {} patches, fit {fitted:.2?}, {} evaluations {:.2?}, RMS error {rms:.2e}",
            field.patch_count(),
            grid.len(),
            start.elapsed()
        );
        assert!(fitted.as_secs() < 60);
    }
}
