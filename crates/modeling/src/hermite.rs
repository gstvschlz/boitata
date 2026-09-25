//! Dual kriging with derivative data — the potential-field formulation.
//!
//! `estimation::DualKriging` factorizes one system from value data and then
//! evaluates cheaply everywhere. This is the same construction with structural
//! observations added as extra rows, which is co-kriging between a variable and
//! its own gradient: Lajaunie, Courrioux & Manuel (1997), "Foliation fields and
//! 3D cartography in geology", *Mathematical Geology* 29(4), and Chapter 5 of
//! Chilès & Delfiner, *Geostatistics: Modeling Spatial Uncertainty*.
//!
//! ```text
//! ┌                       ┐ ┌   ┐   ┌   ┐
//! │ C_ff    C_fg     P_f  │ │ d │   │ f │
//! │ C_gfᵀ   C_gg+ν   P_g  │ │ e │ = │ g │
//! │ P_fᵀ    P_gᵀ     0    │ │ c │   │ 0 │
//! └                       ┘ └   ┘   └   ┘
//! ```
//!
//! The derivative blocks are the covariance differentiated once (value against
//! gradient) and twice (gradient against gradient). Where the RBF face of this
//! in [`crate::rbf`] has to live with a zero gradient–gradient diagonal, here
//! that diagonal is `γ''(0)‖m‖² > 0` — a real, positive variance for a gradient
//! observation — which is why the covariance formulation is the classical
//! choice for structural data.
//!
//! Not every variogram admits it. A model whose shape leaves the origin with a
//! finite slope (spherical, exponential, Matérn ν = ½) describes a field that
//! is continuous but nowhere differentiable: its gradient does not exist, so a
//! dip measured on it constrains nothing. `variogram::is_differentiable` is the
//! test, and [`HermiteKriging::new`] refuses the rest by name.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use variogram::Variogram;

use crate::constraint::{ConstraintSet, ValueKind};
use crate::error::{ModelError, Result};

/// A fitted potential field. Evaluate with [`HermiteKriging::value`].
#[derive(Debug, Clone)]
pub struct HermiteKriging {
    vg: Variogram,
    /// World → variogram space, so `lag(p, q) = ‖A(p − q)‖`. Identity when the
    /// variogram is isotropic.
    a: Matrix3<f64>,
    value_locs: Vec<[f64; 3]>,
    d: Vec<f64>,
    derivative_locs: Vec<[f64; 3]>,
    /// World-space unit directions, as posed. The anisotropy is applied inside
    /// the covariance rather than baked in here, because unlike the RBF this
    /// engine works in world coordinates throughout.
    derivative_dirs: Vec<[f64; 3]>,
    e: Vec<f64>,
    drift: Vec<f64>,
    drift_degree: usize,
    /// The drift basis alone is evaluated on centered, unit-scaled coordinates:
    /// the covariance needs world metres (its ranges are in them), but a raw
    /// mine-grid easting in the drift block costs most of the mantissa.
    drift_origin: [f64; 3],
    drift_scale: f64,
}

/// Options that are not carried by the variogram itself.
#[derive(Debug, Clone)]
pub struct HermiteSpec {
    /// Polynomial drift order: 0 is ordinary kriging, 1 adds a linear trend.
    pub drift_degree: usize,
    /// Ridge on boundary picks, relative to the total sill.
    pub boundary_tolerance: f64,
    /// Ridge on derivative rows, relative to the total sill — the nugget of a
    /// structural reading.
    pub gradient_nugget: f64,
}

impl Default for HermiteSpec {
    fn default() -> Self {
        Self {
            drift_degree: 0,
            boundary_tolerance: 0.0,
            gradient_nugget: 0.0,
        }
    }
}

fn basis(p: &[f64; 3], degree: usize) -> Vec<f64> {
    if degree == 0 {
        vec![1.0]
    } else {
        vec![1.0, p[0], p[1], p[2]]
    }
}

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

impl HermiteKriging {
    /// Fit a field honouring values, boundary picks and derivative rows.
    ///
    /// With no derivative rows this is numerically `estimation::DualKriging`,
    /// which `matches_plain_dual_kriging_without_derivatives` pins.
    pub fn new(set: &ConstraintSet, vg: &Variogram, spec: &HermiteSpec) -> Result<Self> {
        set.validate()?;
        if spec.boundary_tolerance < 0.0 || spec.gradient_nugget < 0.0 {
            return Err(ModelError::InvalidParameter(
                "boundary tolerance and gradient nugget must not be negative".into(),
            ));
        }
        if !set.derivatives.is_empty() && !vg.is_differentiable() {
            let names: Vec<String> = vg
                .non_differentiable_models()
                .iter()
                .map(|m| format!("{m:?}"))
                .collect();
            let detail = if names.is_empty() {
                "a pure-nugget variogram has no spatial structure to have a gradient".to_string()
            } else {
                format!(
                    "the {} structure(s) leave the origin with a finite slope, so a field with \
                     this variogram is continuous but nowhere differentiable and its gradient \
                     does not exist",
                    names.join(", ")
                )
            };
            return Err(ModelError::InvalidParameter(format!(
                "this variogram cannot take structural data: {detail}. Refit with a Gaussian, \
                 cubic or Matérn (ν ≥ 3/2) model, or use the RBF engine's triharmonic kernel."
            )));
        }

        let a = match &vg.anisotropy {
            Some(an) => an.matrix(),
            None => Matrix3::identity(),
        };

        let n = set.values.len();
        let ng = set.derivatives.len();
        let degree = spec.drift_degree.min(1);
        let p = basis_len(degree);
        if n + ng <= p {
            return Err(ModelError::InsufficientData(format!(
                "degree-{degree} drift needs more than {p} constraints, got {}",
                n + ng
            )));
        }

        let value_locs: Vec<[f64; 3]> = set.values.iter().map(|v| v.at).collect();
        let derivative_locs: Vec<[f64; 3]> = set.derivatives.iter().map(|d| d.at).collect();
        let derivative_dirs: Vec<[f64; 3]> = set.derivatives.iter().map(|d| d.direction).collect();
        let (drift_origin, drift_scale) = drift_normalization(&value_locs, &derivative_locs);

        let sill = vg.total_sill().max(f64::MIN_POSITIVE);
        let boundary_ridge = spec.boundary_tolerance * sill;
        let nu = spec.gradient_nugget * sill;

        let dim = n + ng + p;
        let mut m = DMatrix::<f64>::zeros(dim, dim);
        let mut rhs = DVector::<f64>::zeros(dim);

        let scaled = |q: &[f64; 3]| normalize(q, &drift_origin, drift_scale);

        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = cov(vg, &value_locs[i], &value_locs[j]);
            }
            if set.values[i].kind == ValueKind::Boundary {
                m[(i, i)] += boundary_ridge;
            }
            let f = basis(&scaled(&value_locs[i]), degree);
            for (l, fl) in f.iter().enumerate() {
                m[(i, n + ng + l)] = *fl;
                m[(n + ng + l, i)] = *fl;
            }
            rhs[i] = set.values[i].value;
        }

        for j in 0..ng {
            for i in 0..n {
                let entry = cov_d(
                    vg,
                    &a,
                    &value_locs[i],
                    &derivative_locs[j],
                    &derivative_dirs[j],
                );
                m[(i, n + j)] = entry;
                m[(n + j, i)] = entry;
            }
        }

        for j in 0..ng {
            for k in 0..ng {
                m[(n + j, n + k)] = cov_dd(
                    vg,
                    &a,
                    &derivative_locs[j],
                    &derivative_dirs[j],
                    &derivative_locs[k],
                    &derivative_dirs[k],
                );
            }
            m[(n + j, n + j)] += nu;
            // The drift basis is evaluated on scaled coordinates, so its
            // derivative picks up the same 1/scale.
            let dir = derivative_dirs[j];
            let scaled_dir = [
                dir[0] / drift_scale,
                dir[1] / drift_scale,
                dir[2] / drift_scale,
            ];
            let g = dbasis(&scaled_dir, degree);
            for (l, gl) in g.iter().enumerate() {
                m[(n + j, n + ng + l)] = *gl;
                m[(n + ng + l, n + j)] = *gl;
            }
            rhs[n + j] = set.derivatives[j].value;
        }

        let x = m
            .lu()
            .solve(&rhs)
            .ok_or_else(|| ModelError::Singular("potential-field system not invertible".into()))?;
        if x.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(ModelError::Singular(
                "potential-field solve produced non-finite coefficients".into(),
            ));
        }
        Ok(Self {
            vg: vg.clone(),
            a,
            value_locs,
            d: x.as_slice()[..n].to_vec(),
            derivative_locs,
            derivative_dirs,
            e: x.as_slice()[n..n + ng].to_vec(),
            drift: x.as_slice()[n + ng..].to_vec(),
            drift_degree: degree,
            drift_origin,
            drift_scale,
        })
    }

    /// Field value at a world-space point.
    pub fn value(&self, p: &[f64; 3]) -> f64 {
        let mut sum = 0.0;
        for (x, d) in self.value_locs.iter().zip(&self.d) {
            sum += d * cov(&self.vg, p, x);
        }
        for ((y, m), e) in self
            .derivative_locs
            .iter()
            .zip(&self.derivative_dirs)
            .zip(&self.e)
        {
            sum += e * cov_d(&self.vg, &self.a, p, y, m);
        }
        let f = basis(
            &normalize(p, &self.drift_origin, self.drift_scale),
            self.drift_degree,
        );
        for (fl, c) in f.iter().zip(&self.drift) {
            sum += fl * c;
        }
        sum
    }

    /// Field gradient at a world-space point, in field units per metre.
    pub fn gradient(&self, p: &[f64; 3]) -> [f64; 3] {
        let mut g = Vector3::zeros();
        for (x, d) in self.value_locs.iter().zip(&self.d) {
            g += grad_cov(&self.vg, &self.a, p, x) * *d;
        }
        for ((y, m), e) in self
            .derivative_locs
            .iter()
            .zip(&self.derivative_dirs)
            .zip(&self.e)
        {
            // ∇ₚ[m·∇_q C] = −Aᵀ ∇²C₀(A(p − y)) A m.
            let w = [p[0] - y[0], p[1] - y[1], p[2] - y[2]];
            let u = self.a * Vector3::new(w[0], w[1], w[2]);
            let h = u.norm();
            let hess = cov_hessian(&self.vg, &u, h);
            let mv = self.a * Vector3::new(m[0], m[1], m[2]);
            g -= self.a.transpose() * (hess * mv) * *e;
        }
        if self.drift_degree >= 1 && self.drift.len() >= 4 {
            g += Vector3::new(self.drift[1], self.drift[2], self.drift[3]) / self.drift_scale;
        }
        [g.x, g.y, g.z]
    }

    pub fn sample_count(&self) -> usize {
        self.value_locs.len()
    }

    pub fn derivative_count(&self) -> usize {
        self.derivative_locs.len()
    }
}

fn cov(vg: &Variogram, p: &[f64; 3], q: &[f64; 3]) -> f64 {
    vg.cov_points(&(p[0], p[1], p[2]), &(q[0], q[1], q[2]))
}

/// `m · ∇_q C(p, q)` — a value row against a derivative row.
///
/// `∇_q C = −C₀'(h)·Aᵀu/h` with `u = A(p − q)`, so the entry is
/// `−C₀'(h)/h · (Am)·u`.
fn cov_d(vg: &Variogram, a: &Matrix3<f64>, p: &[f64; 3], q: &[f64; 3], m: &[f64; 3]) -> f64 {
    let w = Vector3::new(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    let u = a * w;
    let h = u.norm();
    if h <= 0.0 {
        // A structural reading exactly on a sample: the separation vector
        // vanishes, and with it the directional term.
        return 0.0;
    }
    let mv = a * Vector3::new(m[0], m[1], m[2]);
    -vg.cov_d1(h) / h * mv.dot(&u)
}

/// `mⱼ ∇ₚ∇_qᵀC mₖ` — a derivative row against a derivative row.
fn cov_dd(
    vg: &Variogram,
    a: &Matrix3<f64>,
    pj: &[f64; 3],
    mj: &[f64; 3],
    pk: &[f64; 3],
    mk: &[f64; 3],
) -> f64 {
    let w = Vector3::new(pj[0] - pk[0], pj[1] - pk[1], pj[2] - pk[2]);
    let u = a * w;
    let h = u.norm();
    let hess = cov_hessian(vg, &u, h);
    let vj = a * Vector3::new(mj[0], mj[1], mj[2]);
    let vk = a * Vector3::new(mk[0], mk[1], mk[2]);
    -(vj.transpose() * hess * vk)[(0, 0)]
}

/// `∇²C₀` at separation `u` in variogram space.
///
/// At zero lag both terms tend to `C₀''(0)·I = −γ''(0)·I`, which is finite
/// precisely for the models [`variogram::is_differentiable`] admits — and is
/// what puts a positive number on the gradient–gradient diagonal.
fn cov_hessian(vg: &Variogram, u: &Vector3<f64>, h: f64) -> Matrix3<f64> {
    if h <= 0.0 {
        return Matrix3::identity() * vg.cov_d2(0.0);
    }
    let unit = u / h;
    let outer = unit * unit.transpose();
    outer * vg.cov_d2(h) + (Matrix3::identity() - outer) * (vg.cov_d1(h) / h)
}

/// `∇ₚ C(p, q) = C₀'(h)·Aᵀu/h`.
fn grad_cov(vg: &Variogram, a: &Matrix3<f64>, p: &[f64; 3], q: &[f64; 3]) -> Vector3<f64> {
    let w = Vector3::new(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    let u = a * w;
    let h = u.norm();
    if h <= 0.0 {
        return Vector3::zeros();
    }
    a.transpose() * u * (vg.cov_d1(h) / h)
}

/// Centroid and half-extent of every constrained location, for the drift block
/// only. Degenerate sets get scale 1 rather than a division by zero.
fn drift_normalization(values: &[[f64; 3]], derivatives: &[[f64; 3]]) -> ([f64; 3], f64) {
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
    (origin, if extent > 0.0 { extent } else { 1.0 })
}

fn normalize(p: &[f64; 3], origin: &[f64; 3], scale: f64) -> [f64; 3] {
    [
        (p[0] - origin[0]) / scale,
        (p[1] - origin[1]) / scale,
        (p[2] - origin[2]) / scale,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PlaneEncoding;
    use crate::orientation::{Plane, unit};
    use estimation::{DualKriging, Sample};
    use variogram::{Model, Variogram};

    fn smooth_vg() -> Variogram {
        Variogram::single(Model::Cubic, 1.0, 60.0)
    }

    fn cube(step: f64) -> Vec<[f64; 3]> {
        let mut pts = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    pts.push([i as f64 * step, j as f64 * step, k as f64 * step]);
                }
            }
        }
        pts.push([step * 0.5, step * 0.5, step * 0.5]);
        pts
    }

    fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// With no derivative rows this must be the dual kriging the `estimation`
    /// crate already ships — same predictor, not merely a similar one.
    #[test]
    fn matches_plain_dual_kriging_without_derivatives() {
        let pts = cube(20.0);
        let vg = smooth_vg();
        let mut set = ConstraintSet::new();
        let mut samples = Vec::new();
        for p in &pts {
            let v = p[0] * 0.2 - p[2] * 0.1;
            set.push_sample(*p, v);
            samples.push(Sample::new((p[0], p[1], p[2]), v));
        }
        let hermite = HermiteKriging::new(&set, &vg, &HermiteSpec::default()).unwrap();
        let dual = DualKriging::new(&samples, &vg, 0).unwrap();
        for probe in [[5.0, 5.0, 5.0], [17.0, 3.0, 11.0], [-6.0, 24.0, 8.0]] {
            let a = hermite.value(&probe);
            let b = dual.estimate(&(probe[0], probe[1], probe[2]));
            assert!((a - b).abs() < 1e-8, "{a} vs {b} at {probe:?}");
        }
    }

    /// A model that is not mean-square differentiable has no gradient to
    /// constrain, and the refusal has to say which structure is at fault.
    #[test]
    fn a_non_differentiable_variogram_is_refused_by_name() {
        let mut set = ConstraintSet::new();
        for p in cube(20.0) {
            set.push_sample(p, p[2] * 0.1);
        }
        set.push_derivative([10.0, 10.0, 10.0], [0.0, 0.0, 1.0], 0.1)
            .unwrap();
        let err = HermiteKriging::new(
            &set,
            &Variogram::single(Model::Spherical, 1.0, 60.0),
            &HermiteSpec::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("Spherical"), "unhelpful message: {err}");
        assert!(err.contains("cubic"), "no remedy offered: {err}");

        // The same data on a cubic model is fine.
        assert!(HermiteKriging::new(&set, &smooth_vg(), &HermiteSpec::default()).is_ok());
    }

    /// Value rows with no nugget are honoured exactly.
    #[test]
    fn interpolates_its_samples() {
        let pts = cube(20.0);
        let mut set = ConstraintSet::new();
        for p in &pts {
            set.push_sample(*p, p[0] * 0.3 + p[1] * 0.1);
        }
        let k = HermiteKriging::new(&set, &smooth_vg(), &HermiteSpec::default()).unwrap();
        for p in &pts {
            let want = p[0] * 0.3 + p[1] * 0.1;
            assert!((k.value(p) - want).abs() < 1e-6, "{} vs {want}", k.value(p));
        }
    }

    #[test]
    fn boundary_points_land_on_the_isosurface() {
        let mut set = ConstraintSet::new();
        set.push_sample([0.0, 0.0, 0.0], 1.0);
        set.push_sample([0.0, 0.0, 20.0], -1.0);
        set.push_sample([20.0, 0.0, 0.0], 1.0);
        set.push_sample([20.0, 0.0, 20.0], -1.0);
        set.push_sample([0.0, 20.0, 0.0], 1.0);
        for at in [[5.0, 5.0, 9.0], [15.0, 12.0, 11.0]] {
            set.push_boundary(at, 0.0);
        }
        let k = HermiteKriging::new(&set, &smooth_vg(), &HermiteSpec::default()).unwrap();
        for v in set.values.iter().filter(|v| v.kind == ValueKind::Boundary) {
            assert!(
                k.value(&v.at).abs() < 1e-6,
                "pick sits at {}",
                k.value(&v.at)
            );
        }
    }

    /// The analytic gradient against central differences of the value — the
    /// same check the RBF engine gets, because the two derivations are
    /// independent.
    #[test]
    fn gradient_matches_finite_differences() {
        let mut set = ConstraintSet::new();
        for p in cube(20.0) {
            set.push_sample(p, p[0] * 0.2 - p[1] * 0.1 + p[2] * 0.05);
        }
        set.push_plane(
            [10.0, 10.0, 10.0],
            Plane {
                dip: 35.0,
                dip_direction: 130.0,
            },
            PlaneEncoding::Both,
            0.05,
        )
        .unwrap();
        let k = HermiteKriging::new(
            &set,
            &smooth_vg(),
            &HermiteSpec {
                drift_degree: 1,
                ..Default::default()
            },
        )
        .unwrap();

        let eps = 1e-3;
        for probe in [[6.0, 13.0, 4.0], [15.0, 5.0, 16.0], [2.0, 2.0, 12.0]] {
            let g = k.gradient(&probe);
            for a in 0..3 {
                let mut up = probe;
                let mut dn = probe;
                up[a] += eps;
                dn[a] -= eps;
                let fd = (k.value(&up) - k.value(&dn)) / (2.0 * eps);
                assert!(
                    (g[a] - fd).abs() < 1e-4 * (1.0 + fd.abs()),
                    "component {a} at {probe:?}: {} vs fd {fd}",
                    g[a]
                );
            }
        }
    }

    /// The gradient chain rule under an anisotropic variogram, where `A` is no
    /// longer the identity and a stray transpose would go unnoticed.
    #[test]
    fn gradient_matches_finite_differences_under_anisotropy() {
        use variogram::{Angles, Anisotropy};
        let vg = Variogram::single(Model::Cubic, 1.0, 60.0).with_anisotropy(
            Anisotropy::new(Angles {
                azimuth: 40.0,
                dip: 10.0,
                pitch: 25.0,
                major: 80.0,
                semi: 40.0,
                minor: 15.0,
            })
            .unwrap(),
        );
        let mut set = ConstraintSet::new();
        for p in cube(20.0) {
            set.push_sample(p, p[0] * 0.2 + p[2] * 0.1);
        }
        set.push_derivative([10.0, 10.0, 10.0], [0.0, 0.0, 1.0], 0.05)
            .unwrap();
        let k = HermiteKriging::new(&set, &vg, &HermiteSpec::default()).unwrap();

        let eps = 1e-3;
        let probe = [7.0, 9.0, 6.0];
        let g = k.gradient(&probe);
        for a in 0..3 {
            let mut up = probe;
            let mut dn = probe;
            up[a] += eps;
            dn[a] -= eps;
            let fd = (k.value(&up) - k.value(&dn)) / (2.0 * eps);
            assert!(
                (g[a] - fd).abs() < 1e-3 * (1.0 + fd.abs()),
                "component {a}: {} vs fd {fd}",
                g[a]
            );
        }
    }

    /// The gradient–gradient diagonal is a real positive variance here, unlike
    /// the RBF's zero — that is the reason this engine exists for structural
    /// data, so pin it.
    #[test]
    fn the_gradient_self_covariance_is_positive() {
        let vg = smooth_vg();
        let a = Matrix3::identity();
        let at = [0.0; 3];
        let dir = [0.0, 0.0, 1.0];
        let self_cov = cov_dd(&vg, &a, &at, &dir, &at, &dir);
        assert!(self_cov > 0.0, "gradient self-covariance is {self_cov}");
        // And it is γ''(0), the curvature of the variogram at the origin.
        assert!((self_cov - vg.gamma_d2(0.0)).abs() < 1e-9);
    }

    #[test]
    fn a_structural_reading_steers_the_gradient() {
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
        let k = HermiteKriging::new(&set, &smooth_vg(), &HermiteSpec::default()).unwrap();

        let g = k.gradient(&at);
        let (down_dip, strike) = plane.tangents().unwrap();
        assert!(dot3(&g, &down_dip).abs() < 1e-6, "{}", dot3(&g, &down_dip));
        assert!(dot3(&g, &strike).abs() < 1e-6, "{}", dot3(&g, &strike));
        let n = plane.normal().unwrap();
        let gu = unit(&g).unwrap();
        assert!((dot3(&gu, &n).abs() - 1.0).abs() < 1e-6, "{gu:?} vs {n:?}");
    }

    #[test]
    fn a_normal_constraint_fixes_the_rate_across_the_structure() {
        let plane = Plane {
            dip: 20.0,
            dip_direction: 310.0,
        };
        let at = [10.0, 10.0, 10.0];
        let mut set = ConstraintSet::new();
        for p in cube(20.0) {
            set.push_sample(p, p[2] * 0.02);
        }
        set.push_plane(at, plane, PlaneEncoding::Normal, 0.04)
            .unwrap();
        let k = HermiteKriging::new(&set, &smooth_vg(), &HermiteSpec::default()).unwrap();
        let g = k.gradient(&at);
        let n = plane.normal().unwrap();
        assert!((dot3(&g, &n) - 0.04).abs() < 1e-6, "{}", dot3(&g, &n));
    }

    #[test]
    fn a_structural_only_system_is_refused() {
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
        assert!(HermiteKriging::new(&set, &smooth_vg(), &HermiteSpec::default()).is_err());
    }

    #[test]
    fn the_gradient_nugget_relaxes_conflicting_readings() {
        let at = [10.0, 10.0, 10.0];
        let mut set = ConstraintSet::new();
        for p in cube(20.0) {
            set.push_sample(p, p[2] * 0.02);
        }
        set.push_derivative(at, [0.0, 0.0, 1.0], 0.05).unwrap();
        set.push_derivative([10.0, 10.0, 10.05], [0.0, 0.0, 1.0], -0.05)
            .unwrap();

        let exact = HermiteKriging::new(&set, &smooth_vg(), &HermiteSpec::default()).unwrap();
        let slack = HermiteKriging::new(
            &set,
            &smooth_vg(),
            &HermiteSpec {
                gradient_nugget: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        let miss = |k: &HermiteKriging| (k.gradient(&at)[2] - 0.05).abs();
        assert!(miss(&slack) > miss(&exact));
    }

    /// Mine-grid coordinates: the drift block is conditioned separately from
    /// the covariance, and this is the test that pins that split.
    #[test]
    fn survives_mine_grid_coordinates() {
        let shift = [498_000.0, 7_212_000.0, 1_450.0];
        let mut set = ConstraintSet::new();
        for p in cube(20.0) {
            let q = [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
            set.push_sample(q, p[2] * 0.05);
        }
        let k = HermiteKriging::new(
            &set,
            &smooth_vg(),
            &HermiteSpec {
                drift_degree: 1,
                ..Default::default()
            },
        )
        .unwrap();
        for p in cube(20.0) {
            let q = [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
            assert!((k.value(&q) - p[2] * 0.05).abs() < 1e-6, "{}", k.value(&q));
        }
    }
}
