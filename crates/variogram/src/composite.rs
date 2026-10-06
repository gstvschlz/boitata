//! Composite variogram: nugget + nested structures, with optional anisotropy.

use crate::aniso::{Anisotropy, euclidean};
use crate::model::{Model, Structure};
use serde::{Deserialize, Serialize};

/// A fitted/parametric variogram: a nugget plus one or more nested structures.
///
/// Semivariance is `γ(h) = nugget·[h>0] + Σ sillᵢ·gᵢ(h)`, and the pseudo-covariance
/// used by kriging is `C(h) = C(0) − γ(h)` with `C(0) = nugget + Σ sillᵢ` (total sill).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variogram {
    pub nugget: f64,
    pub structures: Vec<Structure>,
    #[serde(default)]
    pub anisotropy: Option<Anisotropy>,
}

impl Variogram {
    /// Single-structure variogram with no nugget.
    pub fn single(model: Model, sill: f64, range: f64) -> Self {
        Self {
            nugget: 0.0,
            structures: vec![Structure::new(model, sill, range)],
            anisotropy: None,
        }
    }

    /// Attach an anisotropy transform (consumes and returns self).
    pub fn with_anisotropy(mut self, aniso: Anisotropy) -> Self {
        self.anisotropy = Some(aniso);
        self
    }

    /// Total sill `C(0) = nugget + Σ partial sills`.
    pub fn total_sill(&self) -> f64 {
        self.nugget + self.structures.iter().map(|s| s.sill).sum::<f64>()
    }

    /// The same shape scaled so the total sill is `sill`, e.g. 1 for normal scores.
    pub fn standardized(&self, sill: f64) -> Self {
        let k = sill / self.total_sill();
        let mut out = self.clone();
        out.nugget *= k;
        out.structures.iter_mut().for_each(|s| s.sill *= k);
        out
    }

    /// Whether every structure is second-order stationary (finite sill).
    pub fn is_stationary(&self) -> bool {
        self.structures.iter().all(|s| s.is_stationary())
    }

    /// Semivariance at scalar lag `h`.
    pub fn gamma(&self, h: f64) -> f64 {
        let structural: f64 = self.structures.iter().map(|s| s.semivariance(h)).sum();
        if h > 0.0 {
            self.nugget + structural
        } else {
            0.0
        }
    }

    /// Pseudo-covariance at scalar lag `h`: `C(h) = C(0) − γ(h)`.
    pub fn cov(&self, h: f64) -> f64 {
        if h <= 0.0 {
            self.total_sill()
        } else {
            self.total_sill() - self.gamma(h)
        }
    }

    /// Reduced lag distance between two points, applying anisotropy if present.
    pub fn lag(&self, p: &(f64, f64, f64), q: &(f64, f64, f64)) -> f64 {
        match &self.anisotropy {
            Some(a) => a.lag(p, q),
            None => euclidean(p, q),
        }
    }

    /// Semivariance between two 3D points (honors anisotropy).
    pub fn gamma_points(&self, p: &(f64, f64, f64), q: &(f64, f64, f64)) -> f64 {
        self.gamma(self.lag(p, q))
    }

    /// Pseudo-covariance between two 3D points (honors anisotropy).
    pub fn cov_points(&self, p: &(f64, f64, f64), q: &(f64, f64, f64)) -> f64 {
        self.cov(self.lag(p, q))
    }

    /// Covariance between discretization points of a block: the nugget has zero
    /// range, so it averages out over a volume and is left out even where the
    /// points coincide.
    pub fn block_cov_points(&self, p: &(f64, f64, f64), q: &(f64, f64, f64)) -> f64 {
        let h = self.lag(p, q);
        if h <= 0.0 {
            self.total_sill() - self.nugget
        } else {
            self.cov(h)
        }
    }

    /// `dγ/dh` at scalar lag `h`, summed over the structures.
    ///
    /// The nugget is deliberately absent. It is a discontinuity at the origin,
    /// not a slope: it contributes nothing to `γ'` for `h > 0` and is not
    /// differentiable at `h = 0`. Derivative observations therefore see the
    /// structures only, which is also the right physics — a nugget is sampling
    /// and assay error, and error does not have a dip.
    pub fn gamma_d1(&self, h: f64) -> f64 {
        self.structures
            .iter()
            .map(|s| s.sill * crate::model::shape_d1(s.model, h, s.range))
            .sum()
    }

    /// `d²γ/dh²` at scalar lag `h`, summed over the structures. See
    /// [`Variogram::gamma_d1`] for why the nugget is excluded.
    pub fn gamma_d2(&self, h: f64) -> f64 {
        self.structures
            .iter()
            .map(|s| s.sill * crate::model::shape_d2(s.model, h, s.range))
            .sum()
    }

    /// `dC/dh = −dγ/dh`.
    pub fn cov_d1(&self, h: f64) -> f64 {
        -self.gamma_d1(h)
    }

    /// `d²C/dh² = −d²γ/dh²`.
    pub fn cov_d2(&self, h: f64) -> f64 {
        -self.gamma_d2(h)
    }

    /// Whether every structure describes a mean-square differentiable field,
    /// so a gradient observation on it is meaningful.
    ///
    /// A variogram with no structures at all (pure nugget) is not: there is
    /// nothing left to have a gradient.
    pub fn is_differentiable(&self) -> bool {
        !self.structures.is_empty()
            && self
                .structures
                .iter()
                .all(|s| crate::model::is_differentiable(s.model))
    }

    /// The structures whose shape blocks differentiation, for an error message
    /// that names them instead of saying "unsupported".
    pub fn non_differentiable_models(&self) -> Vec<Model> {
        self.structures
            .iter()
            .filter(|s| !crate::model::is_differentiable(s.model))
            .map(|s| s.model)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nugget_discontinuity() {
        let v = Variogram {
            nugget: 0.2,
            structures: vec![Structure::new(Model::Spherical, 0.8, 100.0)],
            anisotropy: None,
        };
        assert_eq!(v.gamma(0.0), 0.0); // exactly at origin
        assert!((v.gamma(1e-9) - 0.2).abs() < 0.01); // jump to nugget just off origin
        assert!((v.total_sill() - 1.0).abs() < 1e-12);
        assert!((v.gamma(1000.0) - 1.0).abs() < 1e-9); // reaches total sill
    }

    /// Theory check: standardizing keeps γ(h) / total sill at every lag.
    #[test]
    fn standardized_keeps_the_shape() {
        let v = Variogram {
            nugget: 3.0,
            structures: vec![
                Structure::new(Model::Spherical, 5.0, 100.0),
                Structure::new(Model::Exponential, 2.0, 400.0),
            ],
            anisotropy: None,
        };
        let s = v.standardized(1.0);
        assert!((s.total_sill() - 1.0).abs() < 1e-12);
        for h in [1.0, 50.0, 150.0, 1000.0] {
            assert!((s.gamma(h) - v.gamma(h) / v.total_sill()).abs() < 1e-12);
        }
    }

    #[test]
    fn covariance_complements_semivariance() {
        let v = Variogram::single(Model::Exponential, 1.0, 100.0);
        for &h in &[5.0, 40.0, 90.0, 300.0] {
            assert!((v.cov(h) - (v.total_sill() - v.gamma(h))).abs() < 1e-12);
        }
        assert!((v.cov(0.0) - v.total_sill()).abs() < 1e-12);
    }

    #[test]
    fn nested_structures_sum() {
        let v = Variogram {
            nugget: 0.1,
            structures: vec![
                Structure::new(Model::Spherical, 0.5, 100.0),
                Structure::new(Model::Exponential, 0.4, 400.0),
            ],
            anisotropy: None,
        };
        assert!((v.total_sill() - 1.0).abs() < 1e-12);
        let g = v.gamma(50.0);
        assert!(g > 0.1 && g < 1.0);
    }
}
