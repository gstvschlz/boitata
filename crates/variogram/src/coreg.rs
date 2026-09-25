//! Linear Model of Coregionalization (LMC) for multivariate geostatistics.
//!
//! Extends the single-variable [`crate::Variogram`] to `nvar` correlated
//! variables. Each nested structure carries an `nvar × nvar` **coregionalization
//! matrix** `Bₛ` (positive semi-definite for a valid model) instead of a scalar
//! sill, and the cross-covariance is
//!
//! ```text
//! C_ij(h) = N_ij·[h = 0] + Σₛ Bₛ[i][j]·(1 − gₛ(h))
//! ```
//!
//! where `gₛ` is the normalized structural shape ([`crate::model::shape`]) and
//! `N` is the (PSD) nugget matrix. This is the model layer that cokriging
//! ([`estimation`]) consumes.

use crate::aniso::{Anisotropy, euclidean};
use crate::model::{Model, shape};
use serde::{Deserialize, Serialize};

/// One nested structure of an LMC: a shape/range plus its coregionalization
/// (sill) matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoregStructure {
    pub model: Model,
    pub range: f64,
    /// `nvar × nvar` coregionalization matrix `Bₛ` (should be symmetric PSD).
    pub sills: Vec<Vec<f64>>,
}

/// A linear model of coregionalization: nugget matrix + nested structures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coregionalization {
    pub nvar: usize,
    /// `nvar × nvar` nugget matrix.
    pub nugget: Vec<Vec<f64>>,
    pub structures: Vec<CoregStructure>,
    #[serde(default)]
    pub anisotropy: Option<Anisotropy>,
}

impl Coregionalization {
    /// Build a bivariate/multivariate model from a nugget matrix and structures.
    pub fn new(nugget: Vec<Vec<f64>>, structures: Vec<CoregStructure>) -> Self {
        let nvar = nugget.len();
        Self {
            nvar,
            nugget,
            structures,
            anisotropy: None,
        }
    }

    pub fn with_anisotropy(mut self, a: Anisotropy) -> Self {
        self.anisotropy = Some(a);
        self
    }

    /// Reduced lag between two points (honors anisotropy).
    pub fn lag(&self, p: &(f64, f64, f64), q: &(f64, f64, f64)) -> f64 {
        match &self.anisotropy {
            Some(a) => a.lag(p, q),
            None => euclidean(p, q),
        }
    }

    /// Cross-sill `C_ij(0) = N_ij + Σₛ Bₛ[i][j]`.
    pub fn sill(&self, i: usize, j: usize) -> f64 {
        self.nugget[i][j] + self.structures.iter().map(|s| s.sills[i][j]).sum::<f64>()
    }

    /// Cross-covariance `C_ij(h)` between variables `i` and `j` at points `p`, `q`.
    pub fn cross_cov(&self, i: usize, j: usize, p: &(f64, f64, f64), q: &(f64, f64, f64)) -> f64 {
        let h = self.lag(p, q);
        let mut c = 0.0;
        if h <= 0.0 {
            c += self.nugget[i][j];
        }
        for s in &self.structures {
            c += s.sills[i][j] * (1.0 - shape(s.model, h, s.range));
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple bivariate proportional-covariance model.
    fn model() -> Coregionalization {
        Coregionalization::new(
            vec![vec![0.1, 0.02], vec![0.02, 0.2]],
            vec![CoregStructure {
                model: Model::Spherical,
                range: 100.0,
                sills: vec![vec![0.9, 0.4], vec![0.4, 0.8]],
            }],
        )
    }

    #[test]
    fn symmetric_cross_covariance() {
        let m = model();
        let p = (0.0, 0.0, 0.0);
        let q = (30.0, 40.0, 0.0);
        assert!((m.cross_cov(0, 1, &p, &q) - m.cross_cov(1, 0, &q, &p)).abs() < 1e-12);
        // Diagonal sill matches the single-variable convention.
        assert!((m.sill(0, 0) - 1.0).abs() < 1e-12);
        assert!((m.sill(1, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn covariance_decays_to_zero() {
        let m = model();
        let p = (0.0, 0.0, 0.0);
        let near = m.cross_cov(0, 0, &p, &(10.0, 0.0, 0.0));
        let far = m.cross_cov(0, 0, &p, &(500.0, 0.0, 0.0));
        assert!(near > far, "near {near} far {far}");
        assert!(far.abs() < 1e-9, "beyond range should vanish, got {far}");
    }

    #[test]
    fn autocovariance_at_zero_is_sill() {
        let m = model();
        let p = (5.0, 5.0, 5.0);
        assert!((m.cross_cov(0, 0, &p, &p) - m.sill(0, 0)).abs() < 1e-12);
    }
}
