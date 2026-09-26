//! Radial basis function interpolation of a scalar field, with derivative data.
//!
//! Fits `f(p) = Σ wᵢ φ(‖p − xᵢ‖) + Σ vⱼ ψⱼ(p) + Σ cₗ pₗ(p)` through value and
//! *derivative* observations by solving the saddle-point system
//!
//! ```text
//! ┌                      ┐ ┌   ┐   ┌   ┐
//! │ K + λI   Kᵍ     P    │ │ w │   │ f │
//! │ Kᵍᵀ      G + νI  Pᵍ  │ │ v │ = │ g │
//! │ Pᵀ       Pᵍᵀ     0   │ │ c │   │ 0 │
//! └                      ┘ └   ┘   └   ┘
//! ```
//!
//! where `K` is the kernel matrix, `P` the polynomial drift basis, `λ` a ridge
//! that turns the exact interpolant into an approximating spline (a nugget, in
//! kriging terms), and the `ᵍ` blocks carry the derivative observations.
//!
//! This is Hermite–Birkhoff interpolation, the RBF face of what the
//! geostatistical literature calls co-kriging with derivative data — the
//! potential-field formulation of Lajaunie, Courrioux & Manuel (1997),
//! "Foliation fields and 3D cartography in geology", *Mathematical Geology*
//! 29(4), and Chapter 5 of Chilès & Delfiner. A structural reading says nothing
//! about the field's value and everything about its gradient, so it enters as a
//! row on `∇f` rather than on `f`; `crate::constraint` is where a dip becomes
//! one.
//!
//! Every kernel here is only *conditionally* positive definite, so the
//! polynomial block is not optional decoration — it is what makes the system
//! solvable. [`Kernel::min_drift_degree`] is the order each one needs.
//!
//! Points are reduced through the anisotropy transform and then centered and
//! scaled to unit extent before anything is assembled: mine-grid coordinates
//! are ~1e5 with kernel values ~1e2, and the drift block of the raw system is
//! ill-conditioned enough to lose most of the mantissa. Directions are carried
//! through the *same* map — but as directions, via its matrix rather than its
//! action on points, which is what [`AnisoTransform::matrix`] exists for.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use variogram::Angles;

use crate::aniso::{AnisoTransform, reduce};
use crate::constraint::{ConstraintSet, ValueKind};
use crate::error::{ModelError, Result};

/// Radial kernel. All three are parameter-free — there is no shape constant to
/// tune, which is why the shape-parameter families (multiquadric, Gaussian)
/// are left out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// `φ(r) = r` — the "linear" kernel; the standard first choice for ore-body
    /// boundaries because it extrapolates flat rather than blowing up.
    Biharmonic,
    /// `φ(r) = r³` — smoother, follows trends further from the data. The only
    /// one of the three that admits derivative data; see
    /// [`Kernel::supports_derivatives`].
    Triharmonic,
    /// `φ(r) = r² ln r` — the 2D-plate spline; flattest of the three.
    ThinPlate,
}

impl Kernel {
    pub fn phi(self, r: f64) -> f64 {
        match self {
            Kernel::Biharmonic => r,
            Kernel::Triharmonic => r * r * r,
            Kernel::ThinPlate => {
                if r <= 0.0 {
                    0.0
                } else {
                    r * r * r.ln()
                }
            }
        }
    }

    /// `φ'(r)`.
    pub fn dphi(self, r: f64) -> f64 {
        match self {
            Kernel::Biharmonic => 1.0,
            Kernel::Triharmonic => 3.0 * r * r,
            Kernel::ThinPlate => {
                if r <= 0.0 {
                    0.0
                } else {
                    r * (2.0 * r.ln() + 1.0)
                }
            }
        }
    }

    /// `φ''(r)`.
    pub fn d2phi(self, r: f64) -> f64 {
        match self {
            Kernel::Biharmonic => 0.0,
            Kernel::Triharmonic => 6.0 * r,
            Kernel::ThinPlate => {
                if r <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    2.0 * r.ln() + 3.0
                }
            }
        }
    }

    /// `φ'(r)/r`, the coefficient of the separation vector in `∇φ`.
    ///
    /// Split out because it is the quantity that actually appears — and because
    /// its limit at the origin is the whole question of whether a kernel can
    /// take derivative data at all.
    pub fn dphi_over_r(self, r: f64) -> f64 {
        match self {
            Kernel::Triharmonic => 3.0 * r,
            _ => {
                if r <= 0.0 {
                    f64::INFINITY
                } else {
                    self.dphi(r) / r
                }
            }
        }
    }

    /// Hessian of `φ(‖d‖)` with respect to the separation vector `d`:
    ///
    /// ```text
    /// ∇²φ = φ''(r)·d̂d̂ᵀ + (φ'(r)/r)·(I − d̂d̂ᵀ)
    /// ```
    ///
    /// `None` when the value at `d = 0` would be infinite, which is the case
    /// for every kernel except the triharmonic. That is not an evaluation
    /// nuisance: a field drawn from a `φ(r) = r` kernel is continuous but
    /// nowhere differentiable, so its gradient covariance genuinely does not
    /// exist at zero separation. See [`Kernel::supports_derivatives`].
    pub fn hessian(self, d: &[f64; 3], r: f64) -> Option<Matrix3<f64>> {
        if r <= 0.0 {
            // The limit is finite only where φ' vanishes to second order.
            return match self {
                Kernel::Triharmonic => Some(Matrix3::zeros()),
                _ => None,
            };
        }
        let v = Vector3::new(d[0], d[1], d[2]) / r;
        let outer = v * v.transpose();
        Some(outer * self.d2phi(r) + (Matrix3::identity() - outer) * self.dphi_over_r(r))
    }

    /// Whether this kernel admits gradient observations.
    ///
    /// The gradient–gradient block needs `∇²φ` at zero separation on its
    /// diagonal — a structural reading against itself. Only the triharmonic has
    /// a finite value there (zero, since `φ'(r)/r = 3r → 0`); the biharmonic
    /// and thin-plate kernels diverge, and no amount of regularization makes a
    /// dip meaningful on a field that has no derivative.
    pub fn supports_derivatives(self) -> bool {
        matches!(self, Kernel::Triharmonic)
    }

    /// Lowest polynomial drift degree that makes the kernel's system solvable.
    pub fn min_drift_degree(self) -> usize {
        match self {
            Kernel::Biharmonic => 0,
            Kernel::Triharmonic | Kernel::ThinPlate => 1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Kernel::Biharmonic => "biharmonic",
            Kernel::Triharmonic => "triharmonic",
            Kernel::ThinPlate => "thin-plate",
        }
    }
}

/// How to fit the field. `drift_degree` is raised to the kernel's minimum when
/// it is set lower.
#[derive(Debug, Clone)]
pub struct RbfSpec {
    pub kernel: Kernel,
    /// 0 = constant drift, 1 = linear (a planar trend through the deposit).
    pub drift_degree: usize,
    /// Ridge term, relative to the value scale. 0 interpolates the samples
    /// exactly; larger values smooth through conflicting nearby data.
    pub smoothing: f64,
    /// Ridge on boundary picks specifically, relative to the value scale.
    ///
    /// A contact pick is a human judgment about where a boundary runs, and two
    /// picks a meter apart on opposite sides of the truth are a contradiction
    /// the exact interpolant has to spike to satisfy. A little slack here keeps
    /// a slightly inconsistent set of picks from making the system singular.
    pub boundary_tolerance: f64,
    /// Ridge on derivative rows, relative to the value scale per unit length.
    ///
    /// The triharmonic's gradient–gradient self-term is exactly zero, so the
    /// block has a zero diagonal; that is not fatal on its own but leaves the
    /// system leaning entirely on the value rows for its pivots. A small
    /// positive value here also absorbs genuinely conflicting readings — two
    /// dips a few centimeters apart that disagree.
    pub gradient_nugget: f64,
    pub anisotropy: Option<Angles>,
}

impl Default for RbfSpec {
    fn default() -> Self {
        Self {
            kernel: Kernel::Biharmonic,
            drift_degree: 1,
            smoothing: 0.0,
            boundary_tolerance: 0.0,
            gradient_nugget: 0.0,
            anisotropy: None,
        }
    }
}

/// A fitted RBF field. Evaluate with [`Rbf::value`], differentiate with
/// [`Rbf::gradient`].
#[derive(Debug, Clone)]
pub struct Rbf {
    kernel: Kernel,
    transform: Option<AnisoTransform>,
    /// Centroid of the reduced constraint locations, subtracted before scaling.
    origin: [f64; 3],
    /// Half-extent of the reduced locations; coordinates are divided by it.
    scale: f64,
    /// World → normalized linear map, for carrying directions and for pushing a
    /// normalized gradient back out to world units.
    jacobian: Matrix3<f64>,
    /// Value-constraint centers in normalized space.
    centers: Vec<[f64; 3]>,
    weights: Vec<f64>,
    /// Derivative-constraint centers in normalized space, with the direction
    /// each row was posed along — already carried through `jacobian`, so these
    /// are not unit vectors.
    derivative_centers: Vec<[f64; 3]>,
    derivative_dirs: Vec<[f64; 3]>,
    derivative_weights: Vec<f64>,
    drift: Vec<f64>,
    drift_degree: usize,
}

/// Polynomial basis at `p`: `[1]` for degree 0, `[1, x, y, z]` for degree 1.
fn basis(p: &[f64; 3], degree: usize) -> Vec<f64> {
    if degree == 0 {
        vec![1.0]
    } else {
        vec![1.0, p[0], p[1], p[2]]
    }
}

/// Directional derivative of [`basis`] along `m`. The constant term drops out,
/// which is exactly why a derivative row says nothing about the field's level.
fn dbasis(m: &[f64; 3], degree: usize) -> Vec<f64> {
    if degree == 0 {
        vec![0.0]
    } else {
        vec![0.0, m[0], m[1], m[2]]
    }
}

fn basis_len(degree: usize) -> usize {
    if degree == 0 { 1 } else { 4 }
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Rbf {
    /// Fit through `points` / `values` (equal length, at least one point).
    ///
    /// The value-only entry point, kept because most callers have no structural
    /// data; it is [`Rbf::fit_constraints`] over plain samples.
    pub fn fit(points: &[[f64; 3]], values: &[f64], spec: &RbfSpec) -> Result<Self> {
        if points.len() != values.len() {
            return Err(ModelError::InvalidParameter(
                "points and values differ in length".into(),
            ));
        }
        let mut set = ConstraintSet::new();
        for (p, v) in points.iter().zip(values) {
            set.push_sample(*p, *v);
        }
        Self::fit_constraints(&set, spec)
    }

    /// Fit a field honoring values, boundary picks and derivative rows at once.
    pub fn fit_constraints(set: &ConstraintSet, spec: &RbfSpec) -> Result<Self> {
        set.validate()?;
        if spec.smoothing < 0.0 || spec.boundary_tolerance < 0.0 || spec.gradient_nugget < 0.0 {
            return Err(ModelError::InvalidParameter(
                "smoothing, boundary tolerance and gradient nugget must not be negative".into(),
            ));
        }
        if !set.derivatives.is_empty() && !spec.kernel.supports_derivatives() {
            return Err(ModelError::InvalidParameter(format!(
                "the {} kernel cannot take structural data: a field with a φ(r) = {} kernel is \
                 not differentiable, so its gradient at a point does not exist. Use the \
                 triharmonic (r³) kernel for dip and foliation constraints.",
                spec.kernel.label(),
                match spec.kernel {
                    Kernel::Biharmonic => "r",
                    Kernel::ThinPlate => "r² ln r",
                    Kernel::Triharmonic => "r³",
                }
            )));
        }

        let n = set.values.len();
        let ng = set.derivatives.len();
        let degree = spec.drift_degree.max(spec.kernel.min_drift_degree()).min(1);
        let p = basis_len(degree);
        if n + ng <= p {
            return Err(ModelError::InsufficientData(format!(
                "{} kernel with degree-{degree} drift needs more than {p} constraints, got {}",
                spec.kernel.label(),
                n + ng
            )));
        }

        let transform = match &spec.anisotropy {
            Some(a) => Some(AnisoTransform::new(a)?),
            None => None,
        };
        // Every constrained location, structural readings included: the
        // normalization has to cover the whole system, not just its value rows.
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

        let centers: Vec<[f64; 3]> = reduced_values
            .iter()
            .map(|r| normalize(r, &origin, scale))
            .collect();
        let derivative_centers: Vec<[f64; 3]> = reduced_derivs
            .iter()
            .map(|r| normalize(r, &origin, scale))
            .collect();

        // World → normalized. Points go through `reduce` then the shift and
        // scale; the linear part of that is what a *direction* transforms by.
        let jacobian = match transform.as_ref() {
            Some(t) => t.matrix() / scale,
            None => Matrix3::identity() / scale,
        };
        // ∇ₚf·n = v becomes ∇_q g·(J n) = v, so the row direction in normalized
        // space is `J n` — not a unit vector, and deliberately so: its length
        // carries the change of units.
        let derivative_dirs: Vec<[f64; 3]> = set
            .derivatives
            .iter()
            .map(|d| {
                let m = jacobian * Vector3::new(d.direction[0], d.direction[1], d.direction[2]);
                [m.x, m.y, m.z]
            })
            .collect();

        // Ridge on the value scale, so the tolerances read the same whatever
        // the units of the modeled variable are.
        let value_scale = value_scale(&set.values.iter().map(|v| v.value).collect::<Vec<_>>());
        let lambda = spec.smoothing * value_scale;
        let boundary_lambda = spec.boundary_tolerance * value_scale;
        let nu = spec.gradient_nugget * value_scale;

        let dim = n + ng + p;
        let mut a = DMatrix::<f64>::zeros(dim, dim);
        let mut rhs = DVector::<f64>::zeros(dim);

        // Value–value block, the value drift block, and the value right-hand side.
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = spec.kernel.phi(distance(&centers[i], &centers[j]));
            }
            a[(i, i)] += match set.values[i].kind {
                ValueKind::Sample => lambda,
                ValueKind::Boundary => boundary_lambda,
            };
            let f = basis(&centers[i], degree);
            for (l, fl) in f.iter().enumerate() {
                a[(i, n + ng + l)] = *fl;
                a[(n + ng + l, i)] = *fl;
            }
            rhs[i] = set.values[i].value;
        }

        // Value–derivative blocks. `Kᵍ[i][j] = φ'(r)/r · mⱼ·(yⱼ − xᵢ)` is
        // symmetric between the two orderings, which is what keeps the whole
        // system symmetric.
        for j in 0..ng {
            for i in 0..n {
                let d = sub(&derivative_centers[j], &centers[i]);
                let r = distance(&derivative_centers[j], &centers[i]);
                let entry = if r > 0.0 {
                    spec.kernel.dphi_over_r(r) * dot3(&derivative_dirs[j], &d)
                } else {
                    // A structural reading exactly on a sample: the separation
                    // vector is zero, so the directional term is too.
                    0.0
                };
                a[(i, n + j)] = entry;
                a[(n + j, i)] = entry;
            }
        }

        // Derivative–derivative block, the derivative drift block, and the
        // derivative right-hand side.
        for j in 0..ng {
            for k in 0..ng {
                let d = sub(&derivative_centers[j], &derivative_centers[k]);
                let r = distance(&derivative_centers[j], &derivative_centers[k]);
                let h = spec.kernel.hessian(&d, r).ok_or_else(|| {
                    ModelError::Singular(format!(
                        "two structural readings coincide and the {} kernel has no finite \
                         gradient covariance there",
                        spec.kernel.label()
                    ))
                })?;
                let mj = Vector3::new(
                    derivative_dirs[j][0],
                    derivative_dirs[j][1],
                    derivative_dirs[j][2],
                );
                let mk = Vector3::new(
                    derivative_dirs[k][0],
                    derivative_dirs[k][1],
                    derivative_dirs[k][2],
                );
                a[(n + j, n + k)] = -(mj.transpose() * h * mk)[(0, 0)];
            }
            a[(n + j, n + j)] += nu;
            let g = dbasis(&derivative_dirs[j], degree);
            for (l, gl) in g.iter().enumerate() {
                a[(n + j, n + ng + l)] = *gl;
                a[(n + ng + l, n + j)] = *gl;
            }
            rhs[n + j] = set.derivatives[j].value;
        }

        let x = a
            .lu()
            .solve(&rhs)
            .ok_or_else(|| ModelError::Singular("RBF system not invertible".into()))?;
        let solution = x.as_slice();
        if solution.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::Singular(
                "RBF solve produced non-finite coefficients".into(),
            ));
        }
        Ok(Self {
            kernel: spec.kernel,
            transform,
            origin,
            scale,
            jacobian,
            centers,
            weights: solution[..n].to_vec(),
            derivative_centers,
            derivative_dirs,
            derivative_weights: solution[n..n + ng].to_vec(),
            drift: solution[n + ng..].to_vec(),
            drift_degree: degree,
        })
    }

    /// Normalized coordinates of a world point.
    fn to_normalized(&self, p: &[f64; 3]) -> [f64; 3] {
        normalize(
            &reduce(self.transform.as_ref(), p),
            &self.origin,
            self.scale,
        )
    }

    /// Field value at a world-space point.
    pub fn value(&self, p: &[f64; 3]) -> f64 {
        let q = self.to_normalized(p);
        let mut sum = 0.0;
        for (c, w) in self.centers.iter().zip(&self.weights) {
            sum += w * self.kernel.phi(distance(c, &q));
        }
        // ψⱼ(q) = mⱼ·∇_y φ(‖q − y‖) at y = yⱼ.
        for ((y, m), v) in self
            .derivative_centers
            .iter()
            .zip(&self.derivative_dirs)
            .zip(&self.derivative_weights)
        {
            let r = distance(y, &q);
            if r > 0.0 {
                sum += v * self.kernel.dphi_over_r(r) * dot3(m, &sub(y, &q));
            }
        }
        let f = basis(&q, self.drift_degree);
        for (fl, c) in f.iter().zip(&self.drift) {
            sum += fl * c;
        }
        sum
    }

    /// Field gradient at a world-space point, in field units per meter.
    ///
    /// Analytic, not differenced — and the thing a structural constraint is
    /// posed against, so `gradient_honors_a_structural_reading` uses it to
    /// check the fit did what it was told.
    pub fn gradient(&self, p: &[f64; 3]) -> [f64; 3] {
        let q = self.to_normalized(p);
        let mut g = Vector3::zeros();
        for (c, w) in self.centers.iter().zip(&self.weights) {
            let r = distance(c, &q);
            if r > 0.0 {
                let d = sub(&q, c);
                g += Vector3::new(d[0], d[1], d[2]) * (w * self.kernel.dphi_over_r(r));
            }
        }
        // ∇_q ψⱼ(q) = −∇²φ(q − yⱼ) mⱼ.
        for ((y, m), v) in self
            .derivative_centers
            .iter()
            .zip(&self.derivative_dirs)
            .zip(&self.derivative_weights)
        {
            let d = sub(&q, y);
            let r = distance(&q, y);
            if let Some(h) = self.kernel.hessian(&d, r) {
                g -= h * Vector3::new(m[0], m[1], m[2]) * *v;
            }
        }
        if self.drift_degree >= 1 && self.drift.len() >= 4 {
            g += Vector3::new(self.drift[1], self.drift[2], self.drift[3]);
        }
        // ∇ₚf = Jᵀ ∇_q g — the transpose, because the chain rule on a gradient
        // runs the other way from the chain rule on a point.
        let world = self.jacobian.transpose() * g;
        [world.x, world.y, world.z]
    }

    pub fn sample_count(&self) -> usize {
        self.centers.len()
    }

    pub fn derivative_count(&self) -> usize {
        self.derivative_centers.len()
    }
}

/// Centroid and half-extent of every constrained location. A degenerate set
/// (all coincident, or a single point) gets scale 1 so normalization is a no-op
/// rather than a division by zero.
fn normalization(values: &[[f64; 3]], derivatives: &[[f64; 3]]) -> ([f64; 3], f64) {
    let n = (values.len() + derivatives.len()) as f64;
    let mut origin = [0.0; 3];
    for p in values.iter().chain(derivatives) {
        for k in 0..3 {
            origin[k] += p[k] / n;
        }
    }
    let mut extent: f64 = 0.0;
    for p in values.iter().chain(derivatives) {
        for k in 0..3 {
            extent = extent.max((p[k] - origin[k]).abs());
        }
    }
    let scale = if extent > 0.0 { extent } else { 1.0 };
    (origin, scale)
}

fn normalize(p: &[f64; 3], origin: &[f64; 3], scale: f64) -> [f64; 3] {
    [
        (p[0] - origin[0]) / scale,
        (p[1] - origin[1]) / scale,
        (p[2] - origin[2]) / scale,
    ]
}

/// Spread of the values, used to put the tolerances on a unit-free scale. Falls
/// back to 1 for a constant field.
fn value_scale(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    if var > 0.0 { var.sqrt() } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PlaneEncoding;
    use crate::orientation::{Plane, unit};

    /// Corners + center of a cube, enough to pin a linear drift.
    fn cube() -> Vec<[f64; 3]> {
        let mut pts = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    pts.push([i as f64 * 10.0, j as f64 * 10.0, k as f64 * 10.0]);
                }
            }
        }
        pts.push([5.0, 5.0, 5.0]);
        pts
    }

    #[test]
    fn interpolates_its_samples_exactly() {
        let pts = cube();
        let values: Vec<f64> = pts
            .iter()
            .map(|p| p[0] * 0.3 + p[1] * p[1] * 0.01)
            .collect();
        for kernel in [Kernel::Biharmonic, Kernel::Triharmonic, Kernel::ThinPlate] {
            let rbf = Rbf::fit(
                &pts,
                &values,
                &RbfSpec {
                    kernel,
                    ..Default::default()
                },
            )
            .unwrap();
            for (p, v) in pts.iter().zip(&values) {
                assert!(
                    (rbf.value(p) - v).abs() < 1e-6,
                    "{kernel:?}: {} vs {v}",
                    rbf.value(p)
                );
            }
        }
    }

    #[test]
    fn linear_drift_reproduces_a_planar_field_away_from_the_data() {
        // f = 2x - y + 3 is exactly the degree-1 drift, so the kernel weights
        // collapse to ~0 and the fit must extrapolate the plane.
        let pts = cube();
        let values: Vec<f64> = pts.iter().map(|p| 2.0 * p[0] - p[1] + 3.0).collect();
        let rbf = Rbf::fit(&pts, &values, &RbfSpec::default()).unwrap();
        for probe in [[40.0, -20.0, 5.0], [-15.0, 30.0, 25.0]] {
            let want = 2.0 * probe[0] - probe[1] + 3.0;
            assert!(
                (rbf.value(&probe) - want).abs() < 1e-4,
                "{} vs {want}",
                rbf.value(&probe)
            );
        }
    }

    #[test]
    fn smoothing_relaxes_the_exact_interpolation() {
        let pts = cube();
        // Two nearly-coincident points disagreeing — the classic case for a
        // nugget: the exact interpolant spikes between them.
        let mut pts = pts;
        pts.push([5.0, 5.0, 5.2]);
        let mut values: Vec<f64> = pts.iter().map(|p| p[0] * 0.1).collect();
        let last = values.len() - 1;
        values[last] += 5.0;

        let exact = Rbf::fit(&pts, &values, &RbfSpec::default()).unwrap();
        let smooth = Rbf::fit(
            &pts,
            &values,
            &RbfSpec {
                smoothing: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        let residual = |r: &Rbf| (r.value(&pts[last]) - values[last]).abs();
        assert!(residual(&exact) < 1e-6);
        assert!(residual(&smooth) > residual(&exact));
    }

    #[test]
    fn survives_mine_grid_coordinates() {
        // Same geometry shifted to a real easting/northing: the normalization
        // is the only thing keeping this solvable.
        let shift = [498_000.0, 7_212_000.0, 1_450.0];
        let pts: Vec<[f64; 3]> = cube()
            .iter()
            .map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
            .collect();
        let values: Vec<f64> = pts.iter().map(|p| (p[2] - shift[2]) * 0.7).collect();
        let rbf = Rbf::fit(&pts, &values, &RbfSpec::default()).unwrap();
        for (p, v) in pts.iter().zip(&values) {
            assert!((rbf.value(p) - v).abs() < 1e-6);
        }
    }

    #[test]
    fn anisotropy_is_exactly_an_isotropic_fit_of_the_reduced_coordinates() {
        // The whole contract of the anisotropy option: fitting with ranges R
        // must equal fitting isotropically on coordinates already reduced by
        // R. Pins the transform, the normalization that follows it, and the
        // evaluation path in one assertion.
        let angles = Angles {
            azimuth: 30.0,
            dip: 15.0,
            rake: 40.0,
            major: 100.0,
            semi: 40.0,
            minor: 10.0,
        };
        let pts = cube();
        let values: Vec<f64> = pts.iter().map(|p| p[0] * 0.2 - p[2] * 0.5 + 1.0).collect();

        let aniso = Rbf::fit(
            &pts,
            &values,
            &RbfSpec {
                anisotropy: Some(angles.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        let t = crate::aniso::AnisoTransform::new(&angles).unwrap();
        let reduced: Vec<[f64; 3]> = pts.iter().map(|p| t.apply(p)).collect();
        let iso = Rbf::fit(&reduced, &values, &RbfSpec::default()).unwrap();

        for probe in [[5.0, 5.0, 5.0], [30.0, -12.0, 4.0], [-8.0, 22.0, 17.0]] {
            let a = aniso.value(&probe);
            let b = iso.value(&t.apply(&probe));
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn rejects_too_few_samples_for_the_drift() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let values = vec![1.0, 2.0];
        assert!(Rbf::fit(&pts, &values, &RbfSpec::default()).is_err());
    }

    #[test]
    fn rejects_mismatched_input_lengths() {
        assert!(Rbf::fit(&cube(), &[1.0], &RbfSpec::default()).is_err());
    }

    // kernel derivatives

    /// The acceptance criterion: closed-form first and second derivatives,
    /// checked against central differences of `phi` itself.
    #[test]
    fn kernel_derivatives_match_finite_differences() {
        // A second difference loses ~ε_machine·|φ|/h² to cancellation, so the
        // curvature check needs a much wider step than the slope check — at
        // r = 9 the triharmonic's φ is ~730 and h = 1e-6 leaves nothing.
        let h1 = 1e-6;
        let h2 = 1e-3;
        for kernel in [Kernel::Biharmonic, Kernel::Triharmonic, Kernel::ThinPlate] {
            for &r in &[0.05_f64, 0.3, 1.0, 2.7, 9.0] {
                let d1 = (kernel.phi(r + h1) - kernel.phi(r - h1)) / (2.0 * h1);
                let d2 =
                    (kernel.phi(r + h2) - 2.0 * kernel.phi(r) + kernel.phi(r - h2)) / (h2 * h2);
                assert!(
                    (kernel.dphi(r) - d1).abs() < 1e-5 * (1.0 + d1.abs()),
                    "{kernel:?} φ'({r}): {} vs fd {d1}",
                    kernel.dphi(r)
                );
                assert!(
                    (kernel.d2phi(r) - d2).abs() < 1e-4 * (1.0 + d2.abs()),
                    "{kernel:?} φ''({r}): {} vs fd {d2}",
                    kernel.d2phi(r)
                );
            }
        }
    }

    /// The Hessian block is the delicate one, so difference the *gradient*
    /// (itself closed-form) rather than the value twice.
    #[test]
    fn kernel_hessian_matches_finite_differences_of_the_gradient() {
        let eps = 1e-6;
        let grad = |k: Kernel, d: &[f64; 3]| {
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let c = k.dphi_over_r(r);
            [c * d[0], c * d[1], c * d[2]]
        };
        for kernel in [Kernel::Biharmonic, Kernel::Triharmonic, Kernel::ThinPlate] {
            let probes: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.3, -0.7, 1.2], [2.0, 2.0, -1.0]];
            for d in probes {
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let h = kernel.hessian(&d, r).unwrap();
                for a in 0..3 {
                    let mut up = d;
                    let mut dn = d;
                    up[a] += eps;
                    dn[a] -= eps;
                    let gu = grad(kernel, &up);
                    let gd = grad(kernel, &dn);
                    for b in 0..3 {
                        let fd = (gu[b] - gd[b]) / (2.0 * eps);
                        assert!(
                            (h[(b, a)] - fd).abs() < 1e-4,
                            "{kernel:?} H[{b}][{a}] at {d:?}: {} vs fd {fd}",
                            h[(b, a)]
                        );
                    }
                }
            }
        }
    }

    /// Only the triharmonic has a finite gradient covariance at zero
    /// separation, and that is exactly which kernels may take structural data.
    #[test]
    fn only_the_triharmonic_admits_derivative_data() {
        assert!(Kernel::Triharmonic.supports_derivatives());
        assert!(!Kernel::Biharmonic.supports_derivatives());
        assert!(!Kernel::ThinPlate.supports_derivatives());

        assert_eq!(
            Kernel::Triharmonic.hessian(&[0.0; 3], 0.0),
            Some(Matrix3::zeros())
        );
        assert!(Kernel::Biharmonic.hessian(&[0.0; 3], 0.0).is_none());
        assert!(Kernel::ThinPlate.hessian(&[0.0; 3], 0.0).is_none());
    }

    #[test]
    fn structural_data_on_an_unsupported_kernel_says_which_one_to_use() {
        let mut set = ConstraintSet::new();
        for p in cube() {
            set.push_sample(p, p[2] * 0.1);
        }
        set.push_derivative([5.0, 5.0, 5.0], [0.0, 0.0, 1.0], 1.0)
            .unwrap();
        let err = Rbf::fit_constraints(
            &set,
            &RbfSpec {
                kernel: Kernel::Biharmonic,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("triharmonic"), "unhelpful message: {err}");
    }

    // the fitted field

    fn triharmonic() -> RbfSpec {
        RbfSpec {
            kernel: Kernel::Triharmonic,
            ..Default::default()
        }
    }

    /// The analytic gradient must agree with differencing the analytic value —
    /// the two halves of the evaluator are derived separately and this is what
    /// catches a sign slip in either.
    #[test]
    fn gradient_matches_finite_differences_of_the_value() {
        let mut set = ConstraintSet::new();
        for p in cube() {
            set.push_sample(p, p[0] * 0.4 - p[1] * 0.2 + p[2] * p[2] * 0.01);
        }
        set.push_plane(
            [5.0, 5.0, 5.0],
            Plane {
                dip: 30.0,
                dip_direction: 120.0,
            },
            PlaneEncoding::Both,
            0.5,
        )
        .unwrap();
        let rbf = Rbf::fit_constraints(&set, &triharmonic()).unwrap();

        let eps = 1e-4;
        for probe in [[3.0, 7.0, 2.0], [8.0, 2.0, 9.0], [-4.0, 12.0, 5.0]] {
            let g = rbf.gradient(&probe);
            for a in 0..3 {
                let mut up = probe;
                let mut dn = probe;
                up[a] += eps;
                dn[a] -= eps;
                let fd = (rbf.value(&up) - rbf.value(&dn)) / (2.0 * eps);
                assert!(
                    (g[a] - fd).abs() < 1e-4 * (1.0 + fd.abs()),
                    "component {a} at {probe:?}: {} vs fd {fd}",
                    g[a]
                );
            }
        }
    }

    /// The gradient chain rule has to survive anisotropy too — that is where a
    /// transpose is easiest to get wrong, because `J` is no longer a multiple
    /// of the identity.
    #[test]
    fn gradient_matches_finite_differences_under_anisotropy() {
        let angles = Angles {
            azimuth: 55.0,
            dip: 20.0,
            rake: 10.0,
            major: 120.0,
            semi: 60.0,
            minor: 20.0,
        };
        let mut set = ConstraintSet::new();
        for p in cube() {
            set.push_sample(p, p[0] * 0.3 + p[2] * 0.2);
        }
        set.push_derivative([5.0, 5.0, 5.0], [0.0, 0.0, 1.0], 0.4)
            .unwrap();
        let rbf = Rbf::fit_constraints(
            &set,
            &RbfSpec {
                anisotropy: Some(angles),
                ..triharmonic()
            },
        )
        .unwrap();

        let eps = 1e-4;
        let probe = [4.0, 6.0, 3.0];
        let g = rbf.gradient(&probe);
        for a in 0..3 {
            let mut up = probe;
            let mut dn = probe;
            up[a] += eps;
            dn[a] -= eps;
            let fd = (rbf.value(&up) - rbf.value(&dn)) / (2.0 * eps);
            assert!(
                (g[a] - fd).abs() < 1e-3 * (1.0 + fd.abs()),
                "component {a}: {} vs fd {fd}",
                g[a]
            );
        }
    }

    /// A boundary pick is a known zero of the field, and the fit must put it
    /// there — not near there.
    #[test]
    fn boundary_points_land_exactly_on_the_isosurface() {
        let mut set = ConstraintSet::new();
        set.push_sample([0.0, 0.0, 0.0], 1.0);
        set.push_sample([0.0, 0.0, 20.0], -1.0);
        set.push_sample([20.0, 0.0, 0.0], 1.0);
        set.push_sample([20.0, 0.0, 20.0], -1.0);
        set.push_sample([0.0, 20.0, 0.0], 1.0);
        set.push_sample([0.0, 20.0, 20.0], -1.0);
        for at in [[5.0, 5.0, 9.0], [15.0, 12.0, 11.0], [2.0, 18.0, 10.5]] {
            set.push_boundary(at, 0.0);
        }
        let rbf = Rbf::fit_constraints(&set, &triharmonic()).unwrap();
        for v in set.values.iter().filter(|v| v.kind == ValueKind::Boundary) {
            assert!(
                rbf.value(&v.at).abs() < 1e-6,
                "boundary pick at {:?} sits at {}",
                v.at,
                rbf.value(&v.at)
            );
        }
    }

    /// The tolerance exists so contradictory picks relax instead of making the
    /// system spike; with it, the fit no longer honors them exactly.
    #[test]
    fn the_boundary_tolerance_relaxes_conflicting_picks() {
        let mut set = ConstraintSet::new();
        set.push_sample([0.0, 0.0, 0.0], 1.0);
        set.push_sample([0.0, 0.0, 20.0], -1.0);
        set.push_sample([20.0, 0.0, 0.0], 1.0);
        set.push_sample([20.0, 0.0, 20.0], -1.0);
        set.push_sample([0.0, 20.0, 0.0], 1.0);
        // Two picks 10 cm apart that cannot both be the contact of a field
        // that also has to pass through the samples above.
        set.push_boundary([10.0, 10.0, 10.0], 0.0);
        set.push_boundary([10.0, 10.0, 10.1], 0.0);
        set.push_sample([10.0, 10.0, 10.05], 0.9);

        let exact = Rbf::fit_constraints(&set, &triharmonic()).unwrap();
        let slack = Rbf::fit_constraints(
            &set,
            &RbfSpec {
                boundary_tolerance: 0.5,
                ..triharmonic()
            },
        )
        .unwrap();
        let miss = |r: &Rbf| r.value(&[10.0, 10.0, 10.0]).abs();
        assert!(miss(&exact) < 1e-6, "exact fit missed by {}", miss(&exact));
        assert!(
            miss(&slack) > miss(&exact),
            "tolerance changed nothing: {} vs {}",
            miss(&slack),
            miss(&exact)
        );
    }

    /// The point of the whole file: a structural reading must actually steer
    /// the field's gradient at the place it was taken.
    #[test]
    fn gradient_honors_a_structural_reading() {
        let plane = Plane {
            dip: 40.0,
            dip_direction: 90.0,
        };
        let at = [10.0, 10.0, 10.0];
        let mut set = ConstraintSet::new();
        set.push_sample([0.0, 0.0, 0.0], 1.0);
        set.push_sample([20.0, 20.0, 20.0], -1.0);
        set.push_sample([0.0, 20.0, 0.0], 1.0);
        set.push_sample([20.0, 0.0, 20.0], -1.0);
        set.push_plane(at, plane, PlaneEncoding::Tangents, 1.0)
            .unwrap();
        let rbf = Rbf::fit_constraints(&set, &triharmonic()).unwrap();

        // The tangent rows say the field is flat along the structure.
        let g = rbf.gradient(&at);
        let (down_dip, strike) = plane.tangents().unwrap();
        assert!(
            dot3(&g, &down_dip).abs() < 1e-6,
            "field changes down-dip: {}",
            dot3(&g, &down_dip)
        );
        assert!(
            dot3(&g, &strike).abs() < 1e-6,
            "field changes along strike: {}",
            dot3(&g, &strike)
        );
        // Which leaves the gradient parallel to the normal, up to sign.
        let n = plane.normal().unwrap();
        let gu = unit(&g).unwrap();
        assert!(
            (dot3(&gu, &n).abs() - 1.0).abs() < 1e-6,
            "gradient {gu:?} is not along the normal {n:?}"
        );
    }

    /// A normal-encoded reading fixes the *rate* as well as the direction.
    #[test]
    fn a_normal_constraint_fixes_the_rate_of_change_across_the_structure() {
        let plane = Plane {
            dip: 25.0,
            dip_direction: 200.0,
        };
        let at = [10.0, 10.0, 10.0];
        let mut set = ConstraintSet::new();
        for p in cube() {
            set.push_sample([p[0] * 2.0, p[1] * 2.0, p[2] * 2.0], p[2] * 0.05);
        }
        set.push_plane(at, plane, PlaneEncoding::Normal, 0.3)
            .unwrap();
        let rbf = Rbf::fit_constraints(&set, &triharmonic()).unwrap();
        let g = rbf.gradient(&at);
        let n = plane.normal().unwrap();
        assert!(
            (dot3(&g, &n) - 0.3).abs() < 1e-6,
            "rate across the structure is {}, want 0.3",
            dot3(&g, &n)
        );
    }

    /// The quality claim behind #318: orientations make the surface follow the
    /// geology *between* the data. A value-only fit through two contacts either
    /// side of a dipping bed interpolates something close to flat; adding the
    /// dips has to tilt it.
    #[test]
    fn a_dip_constraint_changes_a_fit_no_value_data_reproduces() {
        // Two holes, each with an inside sample below and an outside above, so
        // a value-only fit has no way to know the contact is not horizontal.
        let mut base = ConstraintSet::new();
        for x in [0.0, 100.0] {
            base.push_sample([x, 0.0, 0.0], 1.0);
            base.push_sample([x, 0.0, 40.0], -1.0);
            base.push_sample([x, 100.0, 0.0], 1.0);
            base.push_sample([x, 100.0, 40.0], -1.0);
        }
        let flat = Rbf::fit_constraints(&base, &triharmonic()).unwrap();

        // The same data plus a 45° dip towards the east at both holes.
        let plane = Plane {
            dip: 45.0,
            dip_direction: 90.0,
        };
        let mut dipped = base.clone();
        for x in [0.0, 100.0] {
            dipped
                .push_plane([x, 50.0, 20.0], plane, PlaneEncoding::Tangents, 1.0)
                .unwrap();
        }
        let tilted = Rbf::fit_constraints(&dipped, &triharmonic()).unwrap();

        // Between the holes the flat fit's gradient is essentially vertical;
        // the dipped one has to lean.
        let probe = [50.0, 50.0, 20.0];
        let easting_share = |r: &Rbf| {
            let g = unit(&r.gradient(&probe)).unwrap();
            g[0].abs()
        };
        // The value data is symmetric in x, so a value-only fit's gradient has
        // *exactly* no easting component — the flat contact is forced, not
        // merely likely, which is what makes this a fair comparison.
        assert!(
            easting_share(&flat) < 1e-9,
            "value-only fit already leans: {}",
            easting_share(&flat)
        );
        // The dips break that symmetry from 50 m away. They do not rotate the
        // field all the way to the reading's own 45° (which would be ~0.707):
        // the eight value rows still pull towards flat, and a fit that ignored
        // them to honor two orientations would be the worse answer. Moving
        // from nothing to a third of the way is the constraint doing its job.
        assert!(
            easting_share(&tilted) > 0.25,
            "dips did not tilt the field: {}",
            easting_share(&tilted)
        );
    }

    /// The gradient nugget is a ridge, so it must relax a contradictory pair of
    /// readings rather than being ignored.
    #[test]
    fn the_gradient_nugget_relaxes_conflicting_readings() {
        let at = [10.0, 10.0, 10.0];
        let mut set = ConstraintSet::new();
        for p in cube() {
            set.push_sample([p[0] * 2.0, p[1] * 2.0, p[2] * 2.0], p[2] * 0.05);
        }
        // Two readings 5 cm apart demanding different rates along the same
        // direction — a straight contradiction.
        set.push_derivative(at, [0.0, 0.0, 1.0], 0.3).unwrap();
        set.push_derivative([10.0, 10.0, 10.05], [0.0, 0.0, 1.0], -0.3)
            .unwrap();

        let exact = Rbf::fit_constraints(&set, &triharmonic()).unwrap();
        let slack = Rbf::fit_constraints(
            &set,
            &RbfSpec {
                gradient_nugget: 1.0,
                ..triharmonic()
            },
        )
        .unwrap();
        let miss = |r: &Rbf| (r.gradient(&at)[2] - 0.3).abs();
        assert!(miss(&slack) > miss(&exact));
    }

    /// A fit with structural data but nothing to set its level must be refused
    /// before it reaches the solver.
    #[test]
    fn a_structural_only_fit_is_refused() {
        let mut set = ConstraintSet::new();
        for i in 0..6 {
            set.push_plane(
                [i as f64 * 10.0, 0.0, 0.0],
                Plane {
                    dip: 30.0,
                    dip_direction: 45.0,
                },
                PlaneEncoding::Tangents,
                1.0,
            )
            .unwrap();
        }
        assert!(Rbf::fit_constraints(&set, &triharmonic()).is_err());
    }

    /// Adding structural data must not disturb the value rows: the samples are
    /// still interpolated exactly.
    #[test]
    fn structural_data_does_not_break_exact_interpolation_of_the_samples() {
        let pts = cube();
        let values: Vec<f64> = pts.iter().map(|p| p[0] * 0.2 - p[1] * 0.1).collect();
        let mut set = ConstraintSet::new();
        for (p, v) in pts.iter().zip(&values) {
            set.push_sample(*p, *v);
        }
        set.push_plane(
            [5.0, 5.0, 5.0],
            Plane {
                dip: 60.0,
                dip_direction: 15.0,
            },
            PlaneEncoding::Tangents,
            1.0,
        )
        .unwrap();
        let rbf = Rbf::fit_constraints(&set, &triharmonic()).unwrap();
        for (p, v) in pts.iter().zip(&values) {
            assert!(
                (rbf.value(p) - v).abs() < 1e-6,
                "{} vs {v} at {p:?}",
                rbf.value(p)
            );
        }
        assert_eq!(rbf.sample_count(), pts.len());
        assert_eq!(rbf.derivative_count(), 2);
    }
}
