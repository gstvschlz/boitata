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
use crate::error::{Result, VarioError};
use crate::model::{Model, shape};
use nalgebra::DMatrix;
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
#[serde(try_from = "Unchecked")]
pub struct Coregionalization {
    pub nvar: usize,
    /// `nvar × nvar` nugget matrix.
    pub nugget: Vec<Vec<f64>>,
    pub structures: Vec<CoregStructure>,
    #[serde(default)]
    pub anisotropy: Option<Anisotropy>,
}

#[derive(Deserialize)]
struct Unchecked {
    nvar: usize,
    nugget: Vec<Vec<f64>>,
    structures: Vec<CoregStructure>,
    #[serde(default)]
    anisotropy: Option<Anisotropy>,
}

impl TryFrom<Unchecked> for Coregionalization {
    type Error = VarioError;

    fn try_from(u: Unchecked) -> Result<Self> {
        if u.nvar != u.nugget.len() {
            return Err(VarioError::InvalidParameters(format!(
                "nvar {} but a {}-row nugget",
                u.nvar,
                u.nugget.len()
            )));
        }
        let mut c = Self::new(u.nugget, u.structures)?;
        c.anisotropy = u.anisotropy;
        Ok(c)
    }
}

/// Rejects a matrix that is not square, symmetric and positive semi-definite,
/// up to round-off relative to its largest entry or eigenvalue.
fn check_psd(name: &str, m: &[Vec<f64>], n: usize) -> Result<()> {
    let bad = |why: String| {
        Err(VarioError::InvalidParameters(format!(
            "{name} {why}; use Coregionalization.fit for a valid model"
        )))
    };
    if m.len() != n || m.iter().any(|r| r.len() != n) {
        return bad(format!("is not {n} x {n}"));
    }
    let a = DMatrix::from_fn(n, n, |i, j| m[i][j]);
    if !a.iter().all(|v| v.is_finite()) {
        return bad("has a non-finite entry".into());
    }
    let tol = 1e-10 * a.amax();
    if (0..n).any(|i| (0..i).any(|j| (a[(i, j)] - a[(j, i)]).abs() > tol)) {
        return bad("is not symmetric".into());
    }
    if n == 0 {
        return Ok(());
    }
    let eig = a.symmetric_eigenvalues();
    let min = eig.min();
    if min < -1e-10 * eig.amax() {
        return bad(format!(
            "is not positive semi-definite: smallest eigenvalue {min:.3e}"
        ));
    }
    Ok(())
}

impl Coregionalization {
    /// Build a bivariate/multivariate model from a nugget matrix and structures.
    ///
    /// # Errors
    /// [`VarioError::InvalidParameters`] when the nugget or a structure's sill
    /// matrix is not `nvar × nvar`, symmetric and positive semi-definite.
    pub fn new(nugget: Vec<Vec<f64>>, structures: Vec<CoregStructure>) -> Result<Self> {
        let nvar = nugget.len();
        check_psd("nugget", &nugget, nvar)?;
        for (k, s) in structures.iter().enumerate() {
            check_psd(&format!("structure {k} sills"), &s.sills, nvar)?;
        }
        Ok(Self {
            nvar,
            nugget,
            structures,
            anisotropy: None,
        })
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
        .unwrap()
    }

    fn with_sills(sills: Vec<Vec<f64>>) -> Result<Coregionalization> {
        let n = sills.len();
        Coregionalization::new(
            vec![vec![0.0; n]; n],
            vec![CoregStructure {
                model: Model::Spherical,
                range: 50.0,
                sills,
            }],
        )
    }

    #[test]
    fn psd_matrices_accepted() {
        assert!(with_sills(vec![vec![1.0, 1.0], vec![1.0, 1.0]]).is_ok());
        assert!(with_sills(vec![vec![0.0, 0.0], vec![0.0, 0.0]]).is_ok());
        let v = [0.7, -0.3, 0.5];
        let rank_one = (0..3)
            .map(|i| (0..3).map(|j| v[i] * v[j]).collect())
            .collect();
        assert!(with_sills(rank_one).is_ok());
    }

    #[test]
    fn negative_eigenvalue_rejected_with_message() {
        let e = with_sills(vec![vec![1.0, 1.2], vec![1.2, 1.0]]).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("structure 0 sills"), "{msg}");
        assert!(msg.contains("-2.000e-1"), "{msg}");
        assert!(msg.contains("Coregionalization.fit"), "{msg}");
        let e = Coregionalization::new(vec![vec![0.1, 0.2], vec![0.2, 0.1]], vec![]).unwrap_err();
        assert!(e.to_string().contains("nugget"), "{e}");
    }

    #[test]
    fn symmetry_checked_up_to_round_off() {
        assert!(with_sills(vec![vec![1.0, 0.5], vec![0.4, 1.0]]).is_err());
        assert!(with_sills(vec![vec![1.0, 0.5 + 1e-14], vec![0.5, 1.0]]).is_ok());
        assert!(with_sills(vec![vec![1.0, 1.0 + 1e-14], vec![1.0, 1.0]]).is_ok());
        assert!(with_sills(vec![vec![1.0, 0.5]]).is_err());
    }

    #[test]
    fn deserializing_validates() {
        let json = serde_json::to_string(&model()).unwrap();
        assert!(serde_json::from_str::<Coregionalization>(&json).is_ok());
        let bad = json.replace("0.4", "1.4");
        let e = serde_json::from_str::<Coregionalization>(&bad).unwrap_err();
        assert!(e.to_string().contains("positive semi-definite"), "{e}");
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
