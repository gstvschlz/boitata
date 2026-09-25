//! Normalized Hermite polynomials and Gauss–Hermite quadrature.
//!
//! These are the orthonormal basis of the Gaussian anamorphosis (see
//! [`crate::anamorphosis`]). We use the geostatistics convention (Chilès &
//! Delfiner), where the normalized Hermite polynomials `Hₙ`
//! satisfy `E[Hₙ(Y) Hₘ(Y)] = δₙₘ` for `Y ~ N(0, 1)`:
//!
//! ```text
//! H₀(y) = 1
//! H₁(y) = −y
//! Hₙ₊₁(y) = −y·Hₙ(y)/√(n+1) − √(n/(n+1))·Hₙ₋₁(y)      (n ≥ 1)
//! ```
//!
//! The sign convention (`H₁ = −y`) is the one used in the geostatistics literature. Orthonormality is verified by test
//! rather than assumed.

use crate::normal::phi;

/// Evaluate the normalized Hermite polynomials `H₀..H_{degree}` at `y`.
///
/// Returns a vector of length `degree + 1` with `out[n] = Hₙ(y)`.
pub fn polynomials(y: f64, degree: usize) -> Vec<f64> {
    let mut h = vec![0.0; degree + 1];
    h[0] = 1.0;
    if degree == 0 {
        return h;
    }
    h[1] = -y;
    for n in 1..degree {
        let nf = n as f64;
        // H_{n+1} = −y·Hₙ/√(n+1) − √(n/(n+1))·H_{n-1}
        h[n + 1] = -y * h[n] / (nf + 1.0).sqrt() - (nf / (nf + 1.0)).sqrt() * h[n - 1];
    }
    h
}

/// Evaluate a single normalized Hermite polynomial `Hₙ(y)`.
pub fn polynomial(y: f64, n: usize) -> f64 {
    polynomials(y, n)[n]
}

/// Integral `∫ₐ^∞ Hₙ(y) g(y) dy`, where `g` is the standard-normal pdf.
///
/// From `(g·Hₙ)' = √(n+1)·g·H_{n+1}` (with `H₁ = −y`), the closed form for
/// `n ≥ 1` is `−g(a)·H_{n-1}(a)/√n`. For `n = 0` it is the upper tail
/// `1 − Φ(a)`. These are the building blocks of grade–tonnage curves.
pub fn upper_tail_integral(a: f64, n: usize) -> f64 {
    if n == 0 {
        1.0 - phi(a)
    } else {
        -pdf_times_h(a, n - 1) / (n as f64).sqrt()
    }
}

/// Integral `∫_a^b Hₙ(y) g(y) dy` over a (possibly infinite) interval.
pub fn interval_integral(a: f64, b: f64, n: usize) -> f64 {
    if n == 0 {
        phi(b) - phi(a)
    } else {
        // ∫_a^b Hₙ g = [g(b)H_{n-1}(b) − g(a)H_{n-1}(a)] / √n
        (pdf_times_h(b, n - 1) - pdf_times_h(a, n - 1)) / (n as f64).sqrt()
    }
}

/// `g(x)·H_k(x)`, returning `0` at infinite `x` (the pdf decays faster than the
/// polynomial grows, so the product vanishes — avoids `0·∞ = NaN`).
fn pdf_times_h(x: f64, k: usize) -> f64 {
    if x.is_infinite() {
        0.0
    } else {
        gaussian_pdf(x) * polynomial(x, k)
    }
}

/// Standard-normal probability density `g(x) = exp(−x²/2)/√(2π)`.
pub fn gaussian_pdf(x: f64) -> f64 {
    use std::f64::consts::PI;
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Gauss–Hermite quadrature nodes and weights for the *probabilists'* measure
/// `g(y) dy` (i.e. `∫ f(y) g(y) dy ≈ Σ wᵢ f(xᵢ)` with `Y ~ N(0,1)`).
///
/// Nodes are the roots of the (physicists') Hermite polynomial found via the
/// Golub–Welsch eigenvalue method on the Jacobi matrix, then rescaled by `√2`
/// to the probabilists' measure. Used to compute anamorphosis expectations and
/// to project arbitrary functions onto the Hermite basis.
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Golub–Welsch for physicists' Hermite: symmetric tridiagonal Jacobi matrix
    // with zero diagonal and off-diagonal β_k = √(k/2), k = 1..n-1.
    // Eigenvalues → nodes; (first eigenvector component)² · μ₀ → weights, μ₀ = √π.
    use nalgebra::DMatrix;
    if n == 0 {
        return (vec![], vec![]);
    }
    let mut j = DMatrix::<f64>::zeros(n, n);
    for k in 1..n {
        let beta = (k as f64 / 2.0).sqrt();
        j[(k, k - 1)] = beta;
        j[(k - 1, k)] = beta;
    }
    let eig = nalgebra::SymmetricEigen::new(j);
    // Physicists' nodes xᵢ (eigenvalues) rescaled to the probabilists' measure
    // by y = √2·x. Golub–Welsch weight ∝ (first eigenvector component)²; for the
    // N(0,1) measure the weights must sum to 1, and vᵢ² already sums to 1.
    let nodes: Vec<f64> = (0..n)
        .map(|i| std::f64::consts::SQRT_2 * eig.eigenvalues[i])
        .collect();
    let mut w: Vec<f64> = (0..n).map(|i| eig.eigenvectors[(0, i)].powi(2)).collect();
    let wsum: f64 = w.iter().sum();
    for wi in &mut w {
        *wi /= wsum;
    }
    // Sort by node for determinism.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| nodes[a].partial_cmp(&nodes[b]).unwrap());
    let sn: Vec<f64> = idx.iter().map(|&i| nodes[i]).collect();
    let sw: Vec<f64> = idx.iter().map(|&i| w[i]).collect();
    (sn, sw)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Orthonormality `E[Hₙ Hₘ] = δₙₘ` verified by high-order quadrature.
    #[test]
    fn hermite_orthonormal() {
        let (nodes, weights) = gauss_hermite(40);
        let deg = 6;
        for n in 0..=deg {
            for m in 0..=deg {
                let mut acc = 0.0;
                for (x, w) in nodes.iter().zip(&weights) {
                    let h = polynomials(*x, deg);
                    acc += w * h[n] * h[m];
                }
                let expected = if n == m { 1.0 } else { 0.0 };
                assert!(
                    (acc - expected).abs() < 1e-9,
                    "E[H{n} H{m}] = {acc}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn quadrature_integrates_moments() {
        // E[Y²] = 1, E[Y⁴] = 3, E[Y] = 0 under N(0,1).
        let (nodes, weights) = gauss_hermite(20);
        let m1: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x).sum();
        let m2: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
        let m4: f64 = nodes.iter().zip(&weights).map(|(x, w)| w * x.powi(4)).sum();
        assert!(m1.abs() < 1e-12, "E[Y]={m1}");
        assert!((m2 - 1.0).abs() < 1e-12, "E[Y²]={m2}");
        assert!((m4 - 3.0).abs() < 1e-10, "E[Y⁴]={m4}");
    }

    #[test]
    fn upper_tail_matches_quadrature() {
        // ∫_a^∞ Hₙ g dy via closed form vs fine quadrature over a truncated grid.
        let a = 0.4;
        for n in 1..=4 {
            let closed = upper_tail_integral(a, n);
            // Numerical: trapezoid over [a, 10].
            let steps = 200_000;
            let hi = 10.0;
            let dx = (hi - a) / steps as f64;
            let mut num = 0.0;
            for k in 0..=steps {
                let y = a + k as f64 * dx;
                let f = polynomial(y, n) * gaussian_pdf(y);
                let wgt = if k == 0 || k == steps { 0.5 } else { 1.0 };
                num += wgt * f * dx;
            }
            assert!(
                (closed - num).abs() < 1e-6,
                "n={n}: closed {closed} vs num {num}"
            );
        }
    }

    #[test]
    fn first_polynomials_known() {
        // H₀=1, H₁=−y, H₂=(y²−1)/√2, H₃=−(y³−3y)/√6
        let y = 1.3;
        let h = polynomials(y, 3);
        assert!((h[0] - 1.0).abs() < 1e-12);
        assert!((h[1] + y).abs() < 1e-12);
        assert!((h[2] - (y * y - 1.0) / 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((h[3] + (y.powi(3) - 3.0 * y) / 6.0_f64.sqrt()).abs() < 1e-12);
    }
}
