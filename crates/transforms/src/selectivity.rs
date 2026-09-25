//! Selectivity: grade–tonnage and recovered-quantity curves.
//!
//! Given a Hermite anamorphosis `Z = φ(Y)` ([`crate::anamorphosis`]) — point or
//! block support — the recovery functions above a cutoff `z_c` are closed-form
//! Hermite integrals. With `y_c = φ⁻¹(z_c)`:
//!
//! ```text
//! T(z_c) = P(Z ≥ z_c)        = 1 − Φ(y_c)                       (tonnage / ore fraction)
//! Q(z_c) = E[Z · 1_{Z≥z_c}]  = Σₙ ψₙ ∫_{y_c}^∞ Hₙ(y) g(y) dy    (metal)
//! m(z_c) = Q / T                                                (mean grade of ore)
//! B(z_c) = Q − z_c · T                                          (conventional benefit)
//! ```

use crate::anamorphosis::HermiteAnamorphosis;
use crate::hermite::upper_tail_integral;

/// Recovery at one cutoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recovery {
    pub cutoff: f64,
    /// Ore fraction `T(z_c) ∈ [0, 1]` (proportion of tonnage above cutoff).
    pub tonnage: f64,
    /// Recovered metal `Q(z_c)` (mean of `Z` over the whole domain, restricted
    /// to `Z ≥ z_c`).
    pub metal: f64,
    /// Mean grade of the recovered ore `m(z_c) = Q/T` (`NaN`-safe: `0` when `T=0`).
    pub mean_grade: f64,
    /// Conventional benefit `B(z_c) = Q − z_c·T`.
    pub benefit: f64,
}

/// Recovered quantities above a single `cutoff` for the given anamorphosis.
pub fn recovery(anam: &HermiteAnamorphosis, cutoff: f64) -> Recovery {
    // A cutoff outside the authorized interval means all (or none) of the
    // material is above it: map to ∓∞ so the tail integrals resolve exactly.
    let y_c = if cutoff <= anam.z_min {
        f64::NEG_INFINITY
    } else if cutoff >= anam.z_max {
        f64::INFINITY
    } else {
        anam.forward(cutoff)
    };
    let tonnage = upper_tail_integral(y_c, 0);
    let metal: f64 = anam
        .coefficients
        .iter()
        .enumerate()
        .map(|(n, psi)| psi * upper_tail_integral(y_c, n))
        .sum();
    let mean_grade = if tonnage > 1e-12 {
        metal / tonnage
    } else {
        0.0
    };
    let benefit = metal - cutoff * tonnage;
    Recovery {
        cutoff,
        tonnage,
        metal,
        mean_grade,
        benefit,
    }
}

/// Grade–tonnage curve over a list of cutoffs.
pub fn grade_tonnage(anam: &HermiteAnamorphosis, cutoffs: &[f64]) -> Vec<Recovery> {
    cutoffs.iter().map(|&c| recovery(anam, c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anam() -> HermiteAnamorphosis {
        let vals: Vec<f64> = (1..=500).map(|i| ((i as f64) / 50.0).exp()).collect();
        HermiteAnamorphosis::fit(&vals, None, 60).unwrap()
    }

    #[test]
    fn zero_cutoff_recovers_everything() {
        let a = anam();
        // Cutoff below the minimum → all tonnage, metal = mean grade.
        let r = recovery(&a, a.z_min - 1.0);
        assert!((r.tonnage - 1.0).abs() < 1e-3, "T {}", r.tonnage);
        assert!(
            (r.metal - a.mean()).abs() < 1e-2,
            "Q {} mean {}",
            r.metal,
            a.mean()
        );
    }

    #[test]
    fn tonnage_decreases_with_cutoff() {
        let a = anam();
        let cutoffs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let gt = grade_tonnage(&a, &cutoffs);
        for w in gt.windows(2) {
            assert!(w[0].tonnage >= w[1].tonnage - 1e-9, "tonnage not monotone");
            // Mean grade of ore rises (or holds) as cutoff increases.
            assert!(
                w[1].mean_grade >= w[0].mean_grade - 1e-6,
                "mean grade not rising"
            );
        }
    }

    #[test]
    fn metal_never_exceeds_mean() {
        let a = anam();
        for c in 0..15 {
            let r = recovery(&a, c as f64);
            assert!(
                r.metal <= a.mean() + 1e-6,
                "metal {} > mean {}",
                r.metal,
                a.mean()
            );
            assert!(r.metal >= -1e-6, "metal negative {}", r.metal);
            assert!(r.tonnage >= -1e-9 && r.tonnage <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn benefit_peaks_and_grade_above_cutoff() {
        let a = anam();
        for c in 1..12 {
            let r = recovery(&a, c as f64);
            if r.tonnage > 1e-3 {
                // Mean grade of recovered ore must exceed the cutoff.
                assert!(
                    r.mean_grade >= c as f64 - 1e-6,
                    "mean {} < cutoff {c}",
                    r.mean_grade
                );
            }
        }
    }
}
