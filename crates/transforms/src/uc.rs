//! Uniform Conditioning (UC) and Localized Uniform Conditioning (LUC).
//!
//! Estimates recoverable resources at *selective mining unit* (SMU) support
//! inside a larger panel, conditioned only on the panel's (ordinary-kriged)
//! grade. This is the core recoverable-reserves deliverable: it answers "how
//! much ore and metal can be selected at SMU scale within this panel, at a given
//! cutoff", without simulating.
//!
//! ## Model
//!
//! With a point Hermite anamorphosis `Z = φ(Y)` ([`crate::anamorphosis`]) and
//! change-of-support coefficients `r_smu ≥ r_panel` (SMUs are more selective
//! than panels), the SMU and panel Gaussians `Y_v`, `Y_V` are bivariate normal
//! with correlation `s = r_panel / r_smu` (from Cartier's relation
//! `E[Z_v | Z_V] = Z_V`). Conditioning on the panel value gives
//! `Y_v | Y_V ~ N(s·Y_V, 1 − s²)`, so recovery above a cutoff `z_c` is:
//!
//! ```text
//! Y_V = φ_panel⁻¹(Z_V*)          (panel grade → panel Gaussian)
//! y_c = φ_smu⁻¹(z_c)             (cutoff → SMU Gaussian)
//! u_c = (y_c − s·Y_V) / √(1−s²)
//! T   = 1 − Φ(u_c)                                       (SMU ore fraction in panel)
//! Q   = Σₙ (ψₙ r_smuⁿ) · ∫_{u_c}^∞ Hₙ(s·Y_V + √(1−s²)·u) g(u) du
//! ```
//!
//! The integral is evaluated exactly by expanding `Hₙ(a + c·u)` in the Hermite
//! basis of `u` (a three-term recurrence) and reusing the closed-form tail
//! integrals [`crate::hermite::upper_tail_integral`].

use crate::anamorphosis::HermiteAnamorphosis;
use crate::error::{Result, TransformError};
use crate::hermite::upper_tail_integral;
use crate::normal::probit;
use crate::selectivity::Recovery;

/// A UC configuration: a point anamorphosis plus SMU and panel change-of-support
/// coefficients.
#[derive(Debug, Clone)]
pub struct UniformConditioning {
    point: HermiteAnamorphosis,
    smu: HermiteAnamorphosis,
    panel: HermiteAnamorphosis,
    /// Correlation `s = r_panel / r_smu ∈ [0, 1)`.
    s: f64,
}

impl UniformConditioning {
    /// Build from a point anamorphosis and the two change-of-support coefficients.
    ///
    /// Requires `0 < r_panel ≤ r_smu ≤ 1` (panels are less selective than SMUs).
    pub fn new(point: HermiteAnamorphosis, r_smu: f64, r_panel: f64) -> Result<Self> {
        if !(r_panel > 0.0 && r_panel <= r_smu && r_smu <= 1.0 + 1e-12) {
            return Err(TransformError::InvalidParameters(
                "require 0 < r_panel ≤ r_smu ≤ 1".into(),
            ));
        }
        let s = (r_panel / r_smu).min(1.0 - 1e-9);
        let smu = point.block(r_smu);
        let panel = point.block(r_panel);
        Ok(Self {
            point,
            smu,
            panel,
            s,
        })
    }

    /// Recovered SMU quantities within a panel of estimated grade `panel_grade`,
    /// at each `cutoff`.
    pub fn panel_recovery(&self, panel_grade: f64, cutoffs: &[f64]) -> Vec<Recovery> {
        let y_v = self.panel.forward(panel_grade);
        let c = (1.0 - self.s * self.s).sqrt();
        let a = self.s * y_v;
        let degree = self.smu.coefficients.len() - 1;
        // Coefficient vectors of Hₙ(a + c·u) in the Hermite basis of u.
        let pcoeffs = shift_scale_hermite(a, c, degree);

        cutoffs
            .iter()
            .map(|&z_c| {
                let y_c = if z_c <= self.smu.z_min {
                    f64::NEG_INFINITY
                } else if z_c >= self.smu.z_max {
                    f64::INFINITY
                } else {
                    self.smu.forward(z_c)
                };
                let u_c = (y_c - a) / c;
                // Iₙ = Σ_k Pₙ[k] · tail(u_c, k).
                let integ = |n: usize| -> f64 {
                    pcoeffs[n]
                        .iter()
                        .enumerate()
                        .map(|(k, &pk)| pk * upper_tail_integral(u_c, k))
                        .sum::<f64>()
                };
                let tonnage = integ(0); // = 1 − Φ(u_c)
                let metal: f64 = self
                    .smu
                    .coefficients
                    .iter()
                    .enumerate()
                    .map(|(n, &psi)| psi * integ(n))
                    .sum();
                let mean_grade = if tonnage > 1e-12 {
                    metal / tonnage
                } else {
                    0.0
                };
                let benefit = metal - z_c * tonnage;
                Recovery {
                    cutoff: z_c,
                    tonnage,
                    metal,
                    mean_grade,
                    benefit,
                }
            })
            .collect()
    }

    /// Localized UC: assign a single grade to each of `n_smu` SMUs in a panel,
    /// ordered by ascending rank (e.g. by a direct SMU kriging estimate).
    ///
    /// SMU at rank `i` (0-based) receives the conditional-quantile grade at
    /// cumulative probability `(i + 0.5)/n_smu`, so the assembled grades honor
    /// the panel's UC grade–tonnage curve exactly.
    pub fn localized_grades(&self, panel_grade: f64, n_smu: usize) -> Vec<f64> {
        if n_smu == 0 {
            return vec![];
        }
        let y_v = self.panel.forward(panel_grade);
        let c = (1.0 - self.s * self.s).sqrt();
        (0..n_smu)
            .map(|i| {
                let p = (i as f64 + 0.5) / n_smu as f64;
                let y_v_smu = self.s * y_v + c * probit(p);
                self.smu.back(y_v_smu)
            })
            .collect()
    }

    /// The point anamorphosis this UC was built from.
    pub fn point_anamorphosis(&self) -> &HermiteAnamorphosis {
        &self.point
    }
}

/// Coefficient vectors of `Hₙ(a + c·u)` expressed in the Hermite basis of `u`,
/// for `n = 0..=degree`. `out[n][k]` is the coefficient of `Hₖ(u)`.
///
/// Built from the three-term recurrence
/// `√(n+1)·H_{n+1}(x) = −x·Hₙ(x) − √n·H_{n-1}(x)` with `x = a + c·u`, and the
/// multiplication rule `u·Hₖ(u) = −√(k+1)·H_{k+1}(u) − √k·H_{k-1}(u)`.
fn shift_scale_hermite(a: f64, c: f64, degree: usize) -> Vec<Vec<f64>> {
    let size = degree + 1;
    let mut p: Vec<Vec<f64>> = Vec::with_capacity(size);
    // P₀ = 1 = H₀(u).
    let mut p0 = vec![0.0; size];
    p0[0] = 1.0;
    p.push(p0);
    if degree == 0 {
        return p;
    }
    // P₁ = H₁(a + c·u) = −(a + c·u) = −a·H₀ + c·H₁  (since u = −H₁).
    let mut p1 = vec![0.0; size];
    p1[0] = -a;
    if size > 1 {
        p1[1] = c;
    }
    p.push(p1);

    for n in 1..degree {
        let nf = n as f64;
        // u · Pₙ  in the Hermite basis of u.
        let mut u_pn = vec![0.0; size];
        for k in 0..size {
            let v = p[n][k];
            if v == 0.0 {
                continue;
            }
            // u·Hₖ = −√(k+1)·H_{k+1} − √k·H_{k-1}
            if k + 1 < size {
                u_pn[k + 1] += -((k + 1) as f64).sqrt() * v;
            }
            if k >= 1 {
                u_pn[k - 1] += -(k as f64).sqrt() * v;
            }
        }
        // √(n+1)·P_{n+1} = −a·Pₙ − c·(u·Pₙ) − √n·P_{n-1}
        let mut pn1 = vec![0.0; size];
        for k in 0..size {
            pn1[k] = -a * p[n][k] - c * u_pn[k] - nf.sqrt() * p[n - 1][k];
            pn1[k] /= (nf + 1.0).sqrt();
        }
        p.push(pn1);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::Variogram;
    use variogram::model::Model;

    fn point_anam() -> HermiteAnamorphosis {
        let vals: Vec<f64> = (1..=500).map(|i| ((i as f64) / 50.0).exp()).collect();
        HermiteAnamorphosis::fit(&vals, None, 40).unwrap()
    }

    fn uc() -> UniformConditioning {
        // Realistic supports via DGM would give r; here pick r_smu > r_panel directly.
        UniformConditioning::new(point_anam(), 0.85, 0.65).unwrap()
    }

    #[test]
    fn cartier_relation_metal_equals_panel_grade() {
        // At a cutoff below everything, T→1 and Q→panel grade (Cartier's relation).
        let uc = uc();
        let panel_grade = uc.point_anamorphosis().mean() * 1.3;
        let rec = uc.panel_recovery(panel_grade, &[uc.point_anamorphosis().z_min - 5.0]);
        assert!((rec[0].tonnage - 1.0).abs() < 1e-3, "T {}", rec[0].tonnage);
        assert!(
            (rec[0].metal - panel_grade).abs() < 0.05 * panel_grade.abs().max(1.0),
            "Q {} vs panel {}",
            rec[0].metal,
            panel_grade
        );
    }

    #[test]
    fn tonnage_monotone_decreasing() {
        let uc = uc();
        let cutoffs: Vec<f64> = (0..25).map(|i| i as f64 * 0.5).collect();
        let rec = uc.panel_recovery(uc.point_anamorphosis().mean(), &cutoffs);
        for w in rec.windows(2) {
            assert!(w[0].tonnage >= w[1].tonnage - 1e-9, "tonnage not monotone");
            assert!(w[0].tonnage >= -1e-9 && w[0].tonnage <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn richer_panel_recovers_more() {
        // A higher-grade panel recovers more metal at any positive cutoff.
        let uc = uc();
        let cut = vec![uc.point_anamorphosis().mean()];
        let poor = uc.panel_recovery(uc.point_anamorphosis().mean() * 0.6, &cut);
        let rich = uc.panel_recovery(uc.point_anamorphosis().mean() * 1.8, &cut);
        assert!(
            rich[0].metal > poor[0].metal,
            "rich {} poor {}",
            rich[0].metal,
            poor[0].metal
        );
        assert!(rich[0].tonnage >= poor[0].tonnage - 1e-9);
    }

    #[test]
    fn localized_grades_average_to_panel_grade() {
        // The mean of assigned SMU grades reproduces the panel grade (Cartier).
        let uc = uc();
        let panel_grade = uc.point_anamorphosis().mean() * 1.1;
        let grades = uc.localized_grades(panel_grade, 400);
        let mean = grades.iter().sum::<f64>() / grades.len() as f64;
        assert!(
            (mean - panel_grade).abs() < 0.03 * panel_grade,
            "mean {mean} vs panel {panel_grade}"
        );
        // Grades are sorted ascending by construction.
        assert!(grades.windows(2).all(|w| w[0] <= w[1] + 1e-9));
    }

    #[test]
    fn dgm_derived_supports_are_ordered() {
        // Sanity: DGM gives r_smu ≥ r_panel for a smaller SMU than panel.
        use crate::dgm::{BlockDiscretization, change_of_support};
        let anam = point_anam();
        let vg = Variogram::single(Model::Spherical, 1.0, 120.0);
        let (r_smu, _) = change_of_support(
            &anam,
            &vg,
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 10.0),
            &BlockDiscretization::default(),
        )
        .unwrap();
        let (r_panel, _) = change_of_support(
            &anam,
            &vg,
            (0.0, 0.0, 0.0),
            (40.0, 40.0, 40.0),
            &BlockDiscretization::default(),
        )
        .unwrap();
        assert!(r_smu >= r_panel, "r_smu {r_smu} r_panel {r_panel}");
        assert!(UniformConditioning::new(anam, r_smu, r_panel).is_ok());
    }
}
