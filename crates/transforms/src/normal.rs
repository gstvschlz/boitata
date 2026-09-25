//! Standard-normal helpers: CDF (`Φ`) and inverse CDF / quantile (`Φ⁻¹`).
//!
//! Used by the normal-score transform and by sequential simulation. `probit` uses
//! Acklam's rational approximation (abs error ≈ 1e-9); `phi` is exact to double
//! precision through `libm::erfc`.

/// Standard-normal CDF `Φ(x) = P(Z ≤ x)`.
pub fn phi(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Complementary error function.
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// Inverse standard-normal CDF (quantile function), Acklam's algorithm.
///
/// `p` is clamped to the open interval `(0, 1)`.
pub fn probit(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p = p.clamp(1e-15, 1.0 - 1e-15);
    let plow = 0.02425;
    let phigh = 1.0 - plow;

    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_is_accurate_in_the_tails() {
        assert!((phi(-3.0) / 1.349_898_031_630_096e-3 - 1.0).abs() < 1e-12);
        assert!((phi(-6.0) / 9.865_876_450_377_016e-10 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn phi_known_values() {
        assert!((phi(0.0) - 0.5).abs() < 1e-6);
        assert!((phi(1.0) - 0.8413447).abs() < 1e-5);
        assert!((phi(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn probit_inverts_phi() {
        for &z in &[-2.0, -0.5, 0.0, 0.7, 1.5, 2.3] {
            let p = phi(z);
            let zz = probit(p);
            assert!((z - zz).abs() < 1e-4, "z {z} -> p {p} -> {zz}");
        }
    }

    #[test]
    fn probit_quantiles() {
        assert!((probit(0.5)).abs() < 1e-6);
        assert!((probit(0.975) - 1.959964).abs() < 1e-4);
    }
}
