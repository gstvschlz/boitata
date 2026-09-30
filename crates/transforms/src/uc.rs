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
//!
//! `r_panel` is the coefficient of the panel *estimates*: fixed, or per panel
//! from its kriging, solving `Σ ψₙ² r^{2n} = Var(Z_V*)` with
//! `Var(Z_V*) = C(V, V) − σ²_K − 2μ` (the information effect).
//!
//! Localization ([`crate::localize`]) gives the panel's block ranked `i` of `n`
//! the mean of the `i`-th equal-probability band of `u`.

use std::borrow::Cow;

use boitata_core::BlockModel;
use rayon::prelude::*;

use crate::anamorphosis::{HermiteAnamorphosis, eval, eval_deriv};
use crate::dgm::coefficient_from_variance;
use crate::error::{Result, TransformError};
use crate::hermite::{gaussian_pdf, polynomials};
use crate::normal::{phi, probit};
use crate::selectivity::Recovery;
use serde::{Deserialize, Serialize};

/// A UC configuration: a point anamorphosis, the SMU change-of-support
/// coefficient and, unless each panel brings its own, the panel one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformConditioning {
    point: HermiteAnamorphosis,
    smu: HermiteAnamorphosis,
    /// Anamorphosis of the panel estimates; `None` when each panel's comes
    /// from its estimate variance.
    panel: Option<HermiteAnamorphosis>,
    /// Correlation `s = r_panel / r_smu ∈ [0, 1)` of a fixed `panel`.
    s: f64,
    #[serde(default)]
    r_smu: f64,
}

fn invalid(message: impl Into<String>) -> TransformError {
    TransformError::InvalidParameters(message.into())
}

/// `φ⁻¹(z)` polished by Newton steps, `z` clamped to the anamorphosis range.
fn gaussian(anam: &HermiteAnamorphosis, z: f64) -> f64 {
    let z = z.clamp(anam.z_min.min(anam.z_max), anam.z_min.max(anam.z_max));
    let mut y = anam.forward(z);
    for _ in 0..3 {
        let slope = eval_deriv(&anam.coefficients, y);
        if slope <= 0.0 {
            break;
        }
        y = (y - (eval(&anam.coefficients, y) - z) / slope).clamp(anam.y_min, anam.y_max);
    }
    y
}

/// `∫_u^∞ Hₖ(t) g(t) dt` for `k = 0..=degree`.
fn tails(u: f64, degree: usize) -> Vec<f64> {
    if u == f64::INFINITY {
        return vec![0.0; degree + 1];
    }
    let mut out = vec![0.0; degree + 1];
    out[0] = 1.0 - phi(u);
    if u.is_finite() {
        let h = polynomials(u, degree);
        let g = gaussian_pdf(u);
        for k in 1..=degree {
            out[k] = -g * h[k - 1] / (k as f64).sqrt();
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One panel: its grade, and its estimate variance when r_panel is not fixed.
type Panel = (f64, Option<f64>);

fn panel_at(
    grade: &[Option<f64>],
    estimate_variance: Option<&[Option<f64>]>,
    p: usize,
) -> Result<Option<Panel>> {
    let Some(g) = grade[p] else {
        return Ok(None);
    };
    let v = match estimate_variance {
        None => None,
        Some(v) => {
            Some(v[p].ok_or_else(|| invalid("an estimated panel has no estimate variance"))?)
        }
    };
    Ok(Some((g, v)))
}

fn check_lengths(
    grade: &[Option<f64>],
    estimate_variance: Option<&[Option<f64>]>,
    panels: usize,
) -> Result<()> {
    if grade.len() != panels || estimate_variance.is_some_and(|v| v.len() != panels) {
        return Err(invalid("panel columns need one value per panel"));
    }
    Ok(())
}

impl UniformConditioning {
    /// Build from a point anamorphosis and the two change-of-support coefficients.
    ///
    /// Requires `0 < r_panel ≤ r_smu ≤ 1` (panels are less selective than SMUs).
    pub fn new(point: HermiteAnamorphosis, r_smu: f64, r_panel: f64) -> Result<Self> {
        if !(r_panel > 0.0 && r_panel <= r_smu && r_smu <= 1.0 + 1e-12) {
            return Err(invalid("require 0 < r_panel ≤ r_smu ≤ 1"));
        }
        let s = (r_panel / r_smu).min(1.0 - 1e-9);
        Ok(Self {
            smu: point.block(r_smu),
            panel: Some(point.block(s * r_smu)),
            point,
            s,
            r_smu,
        })
    }

    /// Each panel's coefficient comes from its estimate variance
    /// `Var(Z_V*) = C(V, V) − σ²_K − 2μ`, given to every panel method; the
    /// kriging variogram's sill should be the anamorphosis variance.
    /// Requires `0 < r_smu ≤ 1`.
    pub fn per_panel(point: HermiteAnamorphosis, r_smu: f64) -> Result<Self> {
        if !(r_smu > 0.0 && r_smu <= 1.0 + 1e-12) {
            return Err(invalid("require 0 < r_smu ≤ 1"));
        }
        Ok(Self {
            smu: point.block(r_smu),
            panel: None,
            point,
            s: 0.0,
            r_smu,
        })
    }

    /// `r_panel / r_smu` and the panel anamorphosis for one panel.
    fn panel(&self, estimate_variance: Option<f64>) -> Result<(f64, Cow<'_, HermiteAnamorphosis>)> {
        match (&self.panel, estimate_variance) {
            (Some(panel), None) => Ok((self.s, Cow::Borrowed(panel))),
            (None, Some(v)) => {
                if !(v.is_finite() && v > 0.0) {
                    return Err(invalid("estimate variances must be positive"));
                }
                let r = coefficient_from_variance(&self.point, v);
                let s = (r / self.r_smu).clamp(1e-6, 1.0 - 1e-9);
                Ok((s, Cow::Owned(self.point.block(s * self.r_smu))))
            }
            (Some(_), Some(_)) => Err(invalid(
                "r_panel is fixed; estimate variances apply without it",
            )),
            (None, None) => Err(invalid(
                "without r_panel, every panel needs its estimate variance",
            )),
        }
    }

    /// The SMU grade given the panel, `φ_smu(a + c·u)` with `u ~ N(0, 1)`,
    /// as coefficients in the Hermite basis of `u`, and `(a, c)`.
    fn conditional(&self, (grade, variance): Panel) -> Result<(Vec<f64>, f64, f64)> {
        if !grade.is_finite() {
            return Err(invalid("panel grades must be finite"));
        }
        let (s, panel) = self.panel(variance)?;
        let a = s * gaussian(&panel, grade);
        let c = (1.0 - s * s).sqrt();
        let p = shift_scale_hermite(a, c, self.smu.degree());
        let e = (0..=self.smu.degree())
            .map(|k| {
                self.smu
                    .coefficients
                    .iter()
                    .zip(&p)
                    .map(|(w, pn)| w * pn[k])
                    .sum()
            })
            .collect();
        Ok((e, a, c))
    }

    fn recoveries(&self, panel: Panel, cutoffs: &[f64]) -> Result<Vec<Recovery>> {
        let (e, a, c) = self.conditional(panel)?;
        Ok(cutoffs
            .iter()
            .map(|&z_c| {
                let y_c = if z_c <= self.smu.z_min {
                    f64::NEG_INFINITY
                } else if z_c >= self.smu.z_max {
                    f64::INFINITY
                } else {
                    gaussian(&self.smu, z_c)
                };
                let t = tails((y_c - a) / c, self.smu.degree());
                let (tonnage, metal) = (t[0], dot(&e, &t));
                Recovery {
                    cutoff: z_c,
                    tonnage,
                    metal,
                    mean_grade: if tonnage > 1e-12 {
                        metal / tonnage
                    } else {
                        f64::NAN
                    },
                    benefit: metal - z_c * tonnage,
                }
            })
            .collect())
    }

    /// Recovered SMU quantities within a panel of estimated grade
    /// `panel_grade`, at each `cutoff`. `estimate_variance` is the panel's
    /// `Var(Z_V*)` when built by [`Self::per_panel`], else `None`.
    pub fn panel_recovery(
        &self,
        panel_grade: f64,
        estimate_variance: Option<f64>,
        cutoffs: &[f64],
    ) -> Result<Vec<Recovery>> {
        self.recoveries((panel_grade, estimate_variance), cutoffs)
    }

    /// [`Self::panel_recovery`] of every panel, `None` where the grade is.
    pub fn grade_tonnage(
        &self,
        grade: &[Option<f64>],
        estimate_variance: Option<&[Option<f64>]>,
        cutoffs: &[f64],
    ) -> Result<Vec<Option<Vec<Recovery>>>> {
        check_lengths(grade, estimate_variance, grade.len())?;
        (0..grade.len())
            .into_par_iter()
            .map(|p| {
                panel_at(grade, estimate_variance, p)?
                    .map(|panel| self.recoveries(panel, cutoffs))
                    .transpose()
            })
            .collect()
    }

    /// Grades of `n_smu` SMUs in a panel, ascending: the means of `n_smu`
    /// equal-probability bands of the panel's SMU distribution. They average
    /// to the panel grade, and the top `k` recover what the panel's
    /// grade–tonnage curve gives at tonnage `k / n_smu`.
    pub fn band_means(
        &self,
        panel_grade: f64,
        estimate_variance: Option<f64>,
        n_smu: usize,
    ) -> Result<Vec<f64>> {
        let (e, _, _) = self.conditional((panel_grade, estimate_variance))?;
        let n = n_smu as f64;
        let bounds: Vec<Vec<f64>> = (0..=n_smu)
            .map(|i| {
                let u = match i {
                    0 => f64::NEG_INFINITY,
                    i if i == n_smu => f64::INFINITY,
                    i => probit(i as f64 / n),
                };
                tails(u, self.smu.degree())
            })
            .collect();
        Ok(bounds
            .windows(2)
            .map(|w| n * (dot(&e, &w[0]) - dot(&e, &w[1])))
            .collect())
    }

    /// Localized SMU grades of `blocks`, nested in `panels`
    /// ([`crate::localize::localize`]): `grade` (and `estimate_variance` for
    /// [`Self::per_panel`]) per panel row, `ranking` per block.
    pub fn localize(
        &self,
        panels: &BlockModel,
        grade: &[Option<f64>],
        estimate_variance: Option<&[Option<f64>]>,
        blocks: &BlockModel,
        ranking: &[Option<f64>],
    ) -> Result<Vec<Option<f64>>> {
        check_lengths(grade, estimate_variance, panels.len())?;
        crate::localize::localize(panels, blocks, ranking, |p, n| {
            panel_at(grade, estimate_variance, p)?
                .map(|(g, v)| self.band_means(g, v, n))
                .transpose()
        })
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

    fn recovery(uc: &UniformConditioning, grade: f64, cutoffs: &[f64]) -> Vec<Recovery> {
        uc.panel_recovery(grade, None, cutoffs).unwrap()
    }

    #[test]
    fn cartier_relation_metal_equals_panel_grade() {
        // At a cutoff below everything, T→1 and Q→panel grade (Cartier's relation).
        let uc = uc();
        let panel_grade = uc.point_anamorphosis().mean() * 1.3;
        let rec = recovery(&uc, panel_grade, &[uc.point_anamorphosis().z_min - 5.0]);
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
        let rec = recovery(&uc, uc.point_anamorphosis().mean(), &cutoffs);
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
        let poor = recovery(&uc, uc.point_anamorphosis().mean() * 0.6, &cut);
        let rich = recovery(&uc, uc.point_anamorphosis().mean() * 1.8, &cut);
        assert!(
            rich[0].metal > poor[0].metal,
            "rich {} poor {}",
            rich[0].metal,
            poor[0].metal
        );
        assert!(rich[0].tonnage >= poor[0].tonnage - 1e-9);
    }

    #[test]
    fn band_means_average_to_the_panel_grade_and_rise() {
        let uc = uc();
        for f in [0.4, 1.0, 1.1, 2.5] {
            let grade = uc.point_anamorphosis().mean() * f;
            for n in [1, 7, 25, 400] {
                let means = uc.band_means(grade, None, n).unwrap();
                let mean = means.iter().sum::<f64>() / n as f64;
                assert!((mean - grade).abs() < 1e-9 * grade, "{mean} vs {grade}");
                assert!(means.windows(2).all(|w| w[0] <= w[1]), "{means:?}");
            }
        }
    }

    #[test]
    fn band_means_reproduce_the_panel_curve_at_k_over_n() {
        let uc = uc();
        let grade = uc.point_anamorphosis().mean() * 1.2;
        let n = 25;
        let means = uc.band_means(grade, None, n).unwrap();
        let (_, a, c) = uc.conditional((grade, None)).unwrap();
        for k in 1..n {
            let y = a + c * probit(1.0 - k as f64 / n as f64);
            if !(uc.smu.y_min < y && y < uc.smu.y_max) {
                continue;
            }
            let rec = recovery(&uc, grade, &[uc.smu.back(y)])[0];
            let top = means[n - k..].iter().sum::<f64>() / n as f64;
            assert!((rec.tonnage - k as f64 / n as f64).abs() < 1e-8, "k {k}");
            assert!(
                (rec.metal - top).abs() < 1e-9 * top,
                "k {k}: {} vs {top}",
                rec.metal
            );
        }
    }

    #[test]
    fn many_blocks_converge_to_the_panel_curve() {
        let uc = uc();
        let grade = uc.point_anamorphosis().mean() * 0.9;
        let n = 20_000;
        let means = uc.band_means(grade, None, n).unwrap();
        let cutoffs: Vec<f64> = (1..12).map(|i| i as f64).collect();
        for rec in recovery(&uc, grade, &cutoffs) {
            let above: Vec<f64> = means.iter().copied().filter(|&m| m > rec.cutoff).collect();
            let tonnage = above.len() as f64 / n as f64;
            let metal = above.iter().sum::<f64>() / n as f64;
            assert!((tonnage - rec.tonnage).abs() < 2e-4, "{rec:?} vs {tonnage}");
            assert!(
                (metal - rec.metal).abs() < 2e-4 * grade,
                "{rec:?} vs {metal}"
            );
        }
    }

    #[test]
    fn estimate_variance_gives_the_matching_fixed_coefficient() {
        let point = point_anam();
        let r_panel = 0.65;
        let variance: f64 = point.block(r_panel).variance();
        let fixed = UniformConditioning::new(point.clone(), 0.85, r_panel).unwrap();
        let each = UniformConditioning::per_panel(point, 0.85).unwrap();
        let grade = fixed.point_anamorphosis().mean() * 1.4;
        let cutoffs = [1.0, 4.0, 9.0];
        let a = fixed.panel_recovery(grade, None, &cutoffs).unwrap();
        let b = each
            .panel_recovery(grade, Some(variance), &cutoffs)
            .unwrap();
        for (a, b) in a.iter().zip(&b) {
            assert!((a.tonnage - b.tonnage).abs() < 1e-8, "{a:?} {b:?}");
            assert!((a.metal - b.metal).abs() < 1e-8 * grade, "{a:?} {b:?}");
        }
        assert!(
            fixed
                .panel_recovery(grade, Some(variance), &cutoffs)
                .is_err()
        );
        assert!(each.panel_recovery(grade, None, &cutoffs).is_err());
        assert!(each.panel_recovery(grade, Some(0.0), &cutoffs).is_err());
    }

    #[test]
    fn a_smoother_panel_estimate_spreads_its_blocks_more() {
        let each = UniformConditioning::per_panel(point_anam(), 0.85).unwrap();
        let grade = each.point_anamorphosis().mean();
        let var = each.point_anamorphosis().variance();
        let spread = |v: f64| {
            let m = each.band_means(grade, Some(v), 50).unwrap();
            m[49] - m[0]
        };
        assert!(spread(0.2 * var) > spread(0.5 * var));
    }

    #[test]
    fn localize_is_thread_count_independent() {
        let panels = crate::localize::tests::grid([0.0; 3], 50.0, 6, 20.0);
        let blocks = panels.discretize([5, 5, 1]).unwrap();
        let uc = uc();
        let mean = uc.point_anamorphosis().mean();
        let grade: Vec<Option<f64>> = (0..panels.len())
            .map(|p| (p != 3).then_some(mean * (0.5 + 0.1 * p as f64)))
            .collect();
        let ranking: Vec<Option<f64>> = (0..blocks.len())
            .map(|i| Some(((i * 7919) % 13) as f64))
            .collect();
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| uc.localize(&panels, &grade, None, &blocks, &ranking))
                .unwrap()
        };
        let one = run(1);
        assert_eq!(one, run(4));
        let nest = crate::localize::nest(&panels, &blocks).unwrap();
        for (p, g) in grade.iter().enumerate() {
            let mine: Vec<f64> = (0..blocks.len())
                .filter(|&b| nest[b] == Some(p))
                .filter_map(|b| one[b])
                .collect();
            match g {
                None => assert!(mine.is_empty()),
                Some(g) => {
                    assert_eq!(mine.len(), 25);
                    let mean = mine.iter().sum::<f64>() / 25.0;
                    assert!((mean - g).abs() < 1e-9 * g);
                }
            }
        }
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
