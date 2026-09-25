//! Theoretical variogram models.
//!
//! Each model defines a structural shape function `g(h) ∈ [0, 1]` (0 at the origin,
//! rising to 1 at/above the range), scaled by a partial `sill`. A composite variogram
//! sums structures plus a `nugget` discontinuity.

use serde::{Deserialize, Serialize};

/// Variogram model kind (structural shape).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Model {
    Spherical,
    Exponential,
    Gaussian,
    Cubic,
    PentaSpherical,
    Circular,
    /// Hole-effect model (oscillates; shape may exceed 1).
    SineHole,
    /// Matérn model. `order` (ν) supported at half-integers 0.5, 1.5, 2.5;
    /// other values snap to the nearest of these (see module note).
    Matern {
        order: f64,
    },
    /// Power model (non-stationary): `g(h) = (h/range)^exponent`, unbounded.
    /// Here `range` acts as the base length and `sill` as the scaling factor.
    Power {
        exponent: f64,
    },
}

/// A single variogram structure: a shape scaled by a partial sill over a range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Structure {
    pub model: Model,
    /// Partial sill (variance contribution of this structure).
    pub sill: f64,
    /// Practical range (correlation length). For `Power`, the base length.
    pub range: f64,
}

impl Structure {
    pub fn new(model: Model, sill: f64, range: f64) -> Self {
        Self { model, sill, range }
    }

    /// Structural semivariance contribution at lag `h` (already scaled by sill).
    pub fn semivariance(&self, h: f64) -> f64 {
        self.sill * shape(self.model, h, self.range)
    }

    /// Whether this structure is second-order stationary (has a finite sill).
    pub fn is_stationary(&self) -> bool {
        !matches!(self.model, Model::Power { .. })
    }
}

/// Normalized structural shape `g(h)`.
///
/// Returns 0 at the origin and (for bounded models) 1 at/above the range.
/// `SineHole` may exceed 1; `Power` is unbounded and grows without limit.
pub fn shape(model: Model, h: f64, range: f64) -> f64 {
    if !h.is_finite() {
        return 0.0;
    }
    if h <= 0.0 {
        return 0.0;
    }
    let r = range.max(f64::MIN_POSITIVE);
    let t = h / r;

    match model {
        Model::Spherical => {
            if t >= 1.0 {
                1.0
            } else {
                1.5 * t - 0.5 * t.powi(3)
            }
        }
        Model::Exponential => 1.0 - (-3.0 * t).exp(),
        Model::Gaussian => 1.0 - (-3.0 * t * t).exp(),
        Model::Cubic => {
            if t >= 1.0 {
                1.0
            } else {
                7.0 * t.powi(2) - (35.0 / 4.0) * t.powi(3) + (7.0 / 2.0) * t.powi(5)
                    - (3.0 / 4.0) * t.powi(7)
            }
        }
        Model::PentaSpherical => {
            if t >= 1.0 {
                1.0
            } else {
                (15.0 / 8.0) * t - (5.0 / 4.0) * t.powi(3) + (3.0 / 8.0) * t.powi(5)
            }
        }
        Model::Circular => {
            if t >= 1.0 {
                1.0
            } else {
                1.0 - (2.0 / std::f64::consts::PI) * t.acos()
                    + (2.0 * t / std::f64::consts::PI) * (1.0 - t * t).sqrt()
            }
        }
        Model::SineHole => {
            let c = std::f64::consts::PI * t;
            1.0 - c.sin() / c
        }
        Model::Matern { order } => matern_shape(order, t),
        Model::Power { exponent } => t.powf(exponent),
    }
}

/// The half-integer Matérn orders with exact closed forms. Any other `order`
/// snaps to the nearest of these — see [`MaternOrder::nearest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaternOrder {
    Half,
    ThreeHalves,
    FiveHalves,
}

impl MaternOrder {
    fn nearest(order: f64) -> Self {
        if (order - 0.5).abs() < 0.25 {
            MaternOrder::Half
        } else if (order - 1.5).abs() < 0.5 {
            MaternOrder::ThreeHalves
        } else {
            MaternOrder::FiveHalves
        }
    }

    fn nu(self) -> f64 {
        match self {
            MaternOrder::Half => 0.5,
            MaternOrder::ThreeHalves => 1.5,
            MaternOrder::FiveHalves => 2.5,
        }
    }

    /// Scaling of the normalized lag: `δ = √(2ν)·3·t`.
    fn delta_scale(self) -> f64 {
        (2.0 * self.nu()).sqrt() * 3.0
    }
}

/// Matérn shape at normalized lag `t = h/range`
/// with scaling (`δ = √(2ν)·3·t`). Exact closed forms at half-integer orders; other orders
/// snap to the nearest of {0.5, 1.5, 2.5}.
fn matern_shape(order: f64, t: f64) -> f64 {
    let nu = MaternOrder::nearest(order);
    let delta = nu.delta_scale() * t;
    match nu {
        MaternOrder::Half => 1.0 - (-delta).exp(),
        MaternOrder::ThreeHalves => 1.0 - (1.0 + delta) * (-delta).exp(),
        MaternOrder::FiveHalves => 1.0 - (1.0 + delta + delta * delta / 3.0) * (-delta).exp(),
    }
}

/// First derivative of the normalized shape with respect to the lag `h`,
/// `dg/dh`. Zero at and beyond the range for the bounded models.
///
/// Together with [`shape_d2`] this is what a derivative observation needs: a
/// gradient datum constrains `∇C`, and a gradient–gradient pair constrains
/// `∇²C` (see `modeling::hermite`).
pub fn shape_d1(model: Model, h: f64, range: f64) -> f64 {
    if !h.is_finite() || h < 0.0 {
        return 0.0;
    }
    let r = range.max(f64::MIN_POSITIVE);
    let t = h / r;
    d_shape(model, t).0 / r
}

/// Second derivative of the normalized shape with respect to the lag `h`,
/// `d²g/dh²`.
pub fn shape_d2(model: Model, h: f64, range: f64) -> f64 {
    if !h.is_finite() || h < 0.0 {
        return 0.0;
    }
    let r = range.max(f64::MIN_POSITIVE);
    let t = h / r;
    d_shape(model, t).1 / (r * r)
}

/// `(dG/dt, d²G/dt²)` of the normalized shape at normalized lag `t`.
///
/// Kept as one function because every branch computes the two together, and
/// because the `t → 0` limits are the delicate part: for the smooth models the
/// first derivative vanishes there and the second tends to a finite constant,
/// and that pair is exactly what [`is_differentiable`] certifies.
fn d_shape(model: Model, t: f64) -> (f64, f64) {
    match model {
        Model::Spherical => {
            if t >= 1.0 {
                (0.0, 0.0)
            } else {
                (1.5 - 1.5 * t * t, -3.0 * t)
            }
        }
        Model::Exponential => {
            let e = (-3.0 * t).exp();
            (3.0 * e, -9.0 * e)
        }
        Model::Gaussian => {
            let e = (-3.0 * t * t).exp();
            (6.0 * t * e, (6.0 - 36.0 * t * t) * e)
        }
        Model::Cubic => {
            if t >= 1.0 {
                (0.0, 0.0)
            } else {
                (
                    14.0 * t - (105.0 / 4.0) * t.powi(2) + (35.0 / 2.0) * t.powi(4)
                        - (21.0 / 4.0) * t.powi(6),
                    14.0 - (105.0 / 2.0) * t + 70.0 * t.powi(3) - (63.0 / 2.0) * t.powi(5),
                )
            }
        }
        Model::PentaSpherical => {
            if t >= 1.0 {
                (0.0, 0.0)
            } else {
                (
                    15.0 / 8.0 - (15.0 / 4.0) * t * t + (15.0 / 8.0) * t.powi(4),
                    -(15.0 / 2.0) * t + (15.0 / 2.0) * t.powi(3),
                )
            }
        }
        Model::Circular => {
            if t >= 1.0 {
                (0.0, 0.0)
            } else {
                // The algebra collapses: g'(t) = (4/π)·√(1 − t²).
                let s = (1.0 - t * t).max(0.0).sqrt();
                let d1 = (4.0 / std::f64::consts::PI) * s;
                let d2 = if s > 0.0 {
                    -(4.0 / std::f64::consts::PI) * t / s
                } else {
                    f64::NEG_INFINITY
                };
                (d1, d2)
            }
        }
        Model::SineHole => {
            let c = std::f64::consts::PI * t;
            let pi = std::f64::consts::PI;
            if c.abs() < 1e-4 {
                // sin c − c cos c = c³/3 + O(c⁵); the quotients below are 0/0
                // at the origin, so take the series instead.
                (pi * c / 3.0, pi * pi / 3.0)
            } else {
                let (sin, cos) = (c.sin(), c.cos());
                let u = (sin - c * cos) / (c * c);
                let du = (c * c * sin - 2.0 * sin + 2.0 * c * cos) / c.powi(3);
                (pi * u, pi * pi * du)
            }
        }
        Model::Matern { order } => matern_d_shape(order, t),
        Model::Power { exponent } => {
            let e = exponent;
            if t <= 0.0 {
                // t^(e−1) and t^(e−2) diverge below e = 1 and e = 2; the
                // origin is exactly where the power model is not smooth.
                let d1 = if e > 1.0 { 0.0 } else { f64::INFINITY };
                let d2 = if e > 2.0 {
                    0.0
                } else if e == 2.0 {
                    2.0
                } else {
                    f64::INFINITY
                };
                (d1, d2)
            } else {
                (e * t.powf(e - 1.0), e * (e - 1.0) * t.powf(e - 2.0))
            }
        }
    }
}

/// `(dG/dt, d²G/dt²)` for the Matérn shape, snapping `order` to the same
/// half-integers [`matern_shape`] supports.
fn matern_d_shape(order: f64, t: f64) -> (f64, f64) {
    let nu = MaternOrder::nearest(order);
    let k = nu.delta_scale();
    let delta = k * t;
    let e = (-delta).exp();
    match nu {
        // ν = 1/2 is the exponential model, derivatives included.
        MaternOrder::Half => (k * e, -k * k * e),
        MaternOrder::ThreeHalves => (k * delta * e, k * k * (1.0 - delta) * e),
        MaternOrder::FiveHalves => (
            k * (delta / 3.0) * (1.0 + delta) * e,
            k * k * (1.0 + delta - delta * delta) * e / 3.0,
        ),
    }
}

/// Whether a random field with this model is mean-square differentiable — the
/// condition for a *gradient* observation to mean anything.
///
/// The test is `g'(0) = 0`: a shape that leaves the origin with a finite slope
/// (spherical, exponential, Matérn ν = ½) describes a field whose realizations
/// are continuous but nowhere differentiable, so a dip reading on it constrains
/// a derivative that does not exist. The gradient–gradient covariance of those
/// models is genuinely `+∞` at zero lag, not merely awkward to evaluate.
pub fn is_differentiable(model: Model) -> bool {
    match model {
        Model::Gaussian | Model::Cubic | Model::SineHole => true,
        Model::Matern { order } => order >= 1.0,
        Model::Power { exponent } => exponent >= 2.0,
        Model::Spherical | Model::Exponential | Model::PentaSpherical | Model::Circular => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_bounds() {
        assert_eq!(shape(Model::Spherical, 0.0, 100.0), 0.0);
        assert!((shape(Model::Spherical, 100.0, 100.0) - 1.0).abs() < 1e-12);
        assert_eq!(shape(Model::Spherical, 200.0, 100.0), 1.0);
        let mid = shape(Model::Spherical, 50.0, 100.0);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn cubic_reaches_sill_at_range() {
        assert!((shape(Model::Cubic, 100.0, 100.0) - 1.0).abs() < 1e-9);
        assert_eq!(shape(Model::Cubic, 150.0, 100.0), 1.0);
    }

    #[test]
    fn pentaspherical_monotone() {
        let a = shape(Model::PentaSpherical, 25.0, 100.0);
        let b = shape(Model::PentaSpherical, 75.0, 100.0);
        assert!(a < b && b < 1.0);
    }

    #[test]
    fn circular_valid_range() {
        let v = shape(Model::Circular, 50.0, 100.0);
        assert!(v > 0.0 && v < 1.0);
        assert!((shape(Model::Circular, 100.0, 100.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn matern_half_integer_reduces_to_exponential() {
        // ν=0.5 must equal the exponential model exactly.
        for &h in &[10.0, 55.0, 130.0] {
            let m = shape(Model::Matern { order: 0.5 }, h, 100.0);
            let e = shape(Model::Exponential, h, 100.0);
            assert!((m - e).abs() < 1e-12, "matern0.5 {} vs exp {}", m, e);
        }
    }

    #[test]
    fn matern_smoother_orders_increase_correlation() {
        // Higher ν → smoother → lower semivariance at short lag.
        let g05 = shape(Model::Matern { order: 0.5 }, 20.0, 100.0);
        let g25 = shape(Model::Matern { order: 2.5 }, 20.0, 100.0);
        assert!(g25 < g05);
    }

    #[test]
    fn power_is_unbounded() {
        let a = shape(Model::Power { exponent: 1.5 }, 100.0, 100.0);
        let b = shape(Model::Power { exponent: 1.5 }, 200.0, 100.0);
        assert!(b > a);
        assert!((a - 1.0).abs() < 1e-12); // (100/100)^1.5 = 1
    }

    #[test]
    fn sinehole_hole_effect() {
        // Sine-hole can exceed 1 (oscillation) at some lags.
        let vals: Vec<f64> = (1..40)
            .map(|i| shape(Model::SineHole, i as f64 * 10.0, 100.0))
            .collect();
        assert!(vals.iter().cloned().fold(f64::MIN, f64::max) > 1.0);
    }

    /// Central differences of `shape` must reproduce `shape_d1`/`shape_d2` for
    /// every model, away from the origin and away from the range where the
    /// bounded shapes kink.
    #[test]
    fn derivatives_match_finite_differences() {
        let models = [
            Model::Spherical,
            Model::Exponential,
            Model::Gaussian,
            Model::Cubic,
            Model::PentaSpherical,
            Model::Circular,
            Model::SineHole,
            Model::Matern { order: 0.5 },
            Model::Matern { order: 1.5 },
            Model::Matern { order: 2.5 },
            Model::Power { exponent: 1.5 },
        ];
        let range: f64 = 100.0;
        let eps = 1e-4;
        for model in models {
            for &h in &[7.0, 23.0, 61.0, 140.0] {
                // Skip lags straddling the range: the bounded shapes are only
                // C¹ there, so a central difference is meaningless.
                if (h - range).abs() < 1.0 {
                    continue;
                }
                let d1 =
                    (shape(model, h + eps, range) - shape(model, h - eps, range)) / (2.0 * eps);
                let d2 = (shape(model, h + eps, range) - 2.0 * shape(model, h, range)
                    + shape(model, h - eps, range))
                    / (eps * eps);
                let a1 = shape_d1(model, h, range);
                let a2 = shape_d2(model, h, range);
                assert!(
                    (d1 - a1).abs() < 1e-6,
                    "{model:?} at h={h}: d1 {a1} vs fd {d1}"
                );
                assert!(
                    (d2 - a2).abs() < 1e-3,
                    "{model:?} at h={h}: d2 {a2} vs fd {d2}"
                );
            }
        }
    }

    /// The differentiability predicate has to agree with the shapes themselves:
    /// a model is differentiable exactly when it leaves the origin flat.
    #[test]
    fn differentiability_matches_the_slope_at_the_origin() {
        let models = [
            Model::Spherical,
            Model::Exponential,
            Model::Gaussian,
            Model::Cubic,
            Model::PentaSpherical,
            Model::Circular,
            Model::SineHole,
            Model::Matern { order: 0.5 },
            Model::Matern { order: 1.5 },
            Model::Matern { order: 2.5 },
        ];
        for model in models {
            let slope_at_origin = shape_d1(model, 0.0, 100.0);
            let smooth = slope_at_origin.abs() < 1e-12;
            assert_eq!(
                smooth,
                is_differentiable(model),
                "{model:?}: slope {slope_at_origin} vs is_differentiable {}",
                is_differentiable(model)
            );
        }
    }

    /// The smooth models must have a finite curvature at zero lag — that
    /// number *is* the gradient–gradient covariance a structural reading
    /// leans on, so an infinity here is the whole failure mode.
    #[test]
    fn smooth_models_have_finite_curvature_at_zero_lag() {
        for model in [
            Model::Gaussian,
            Model::Cubic,
            Model::SineHole,
            Model::Matern { order: 1.5 },
            Model::Matern { order: 2.5 },
        ] {
            let c = shape_d2(model, 0.0, 100.0);
            assert!(c.is_finite() && c > 0.0, "{model:?}: curvature {c}");
            // And it must be the limit of the curvature just off the origin.
            let near = shape_d2(model, 1e-3, 100.0);
            assert!((c - near).abs() / c < 1e-2, "{model:?}: {c} vs {near}");
        }
    }

    /// ν = 1/2 is the exponential model; its derivatives must match too, not
    /// just its values.
    #[test]
    fn matern_half_order_derivatives_match_the_exponential() {
        for &h in &[5.0, 40.0, 120.0] {
            let m = shape_d1(Model::Matern { order: 0.5 }, h, 100.0);
            let e = shape_d1(Model::Exponential, h, 100.0);
            assert!((m - e).abs() < 1e-12);
        }
    }
}
