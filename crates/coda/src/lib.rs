//! Compositional data analysis (CoDa).
//!
//! Assay grades (and other parts-of-a-whole data) live on the simplex: only ratios
//! carry information, so ordinary statistics/kriging on raw parts are biased. This
//! crate provides the closure operation and the ALR/CLR/ILR log-ratio transforms
//! (with inverses) that map compositions to real coordinates where standard
//! geostatistics is valid, plus the Aitchison distance.
//!
//! References: Aitchison (1986); Egozcue et al. (2003), "Isometric logratio
//! transformations for compositional data analysis" (mirrors `CoDa.jl`).

pub mod error;

pub use error::{CodaError, Result};

/// Close a composition to sum to `total` (default use `total = 1`).
///
/// Non-positive parts are rejected (log-ratios require strictly positive values).
pub fn closure(parts: &[f64], total: f64) -> Result<Vec<f64>> {
    if parts.is_empty() {
        return Err(CodaError::InvalidComposition("empty composition".into()));
    }
    if parts.iter().any(|&x| x <= 0.0 || !x.is_finite()) {
        return Err(CodaError::InvalidComposition(
            "all parts must be positive and finite".into(),
        ));
    }
    let s: f64 = parts.iter().sum();
    Ok(parts.iter().map(|x| x / s * total).collect())
}

/// Geometric mean of a strictly positive vector.
fn geometric_mean(x: &[f64]) -> f64 {
    let ln_sum: f64 = x.iter().map(|v| v.ln()).sum();
    (ln_sum / x.len() as f64).exp()
}

/// Centered log-ratio: `clr(x)_i = ln(x_i / g(x))`. The result sums to zero.
pub fn clr(parts: &[f64]) -> Result<Vec<f64>> {
    let x = closure(parts, 1.0)?;
    let g = geometric_mean(&x);
    Ok(x.iter().map(|v| (v / g).ln()).collect())
}

/// Inverse CLR: `clr⁻¹(y)_i = softmax(y)`, closed to sum 1.
pub fn clr_inv(coords: &[f64]) -> Result<Vec<f64>> {
    if coords.is_empty() {
        return Err(CodaError::InvalidParameters("empty coordinates".into()));
    }
    let m = coords.iter().cloned().fold(f64::MIN, f64::max);
    let exps: Vec<f64> = coords.iter().map(|c| (c - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    Ok(exps.iter().map(|e| e / s).collect())
}

/// Additive log-ratio with the last part as reference:
/// `alr(x)_i = ln(x_i / x_D)` for `i = 1..D-1`.
pub fn alr(parts: &[f64]) -> Result<Vec<f64>> {
    let x = closure(parts, 1.0)?;
    let d = x.len();
    if d < 2 {
        return Err(CodaError::InvalidComposition("need ≥ 2 parts".into()));
    }
    let xd = x[d - 1];
    Ok((0..d - 1).map(|i| (x[i] / xd).ln()).collect())
}

/// Inverse ALR: recover the `D`-part composition (closed to sum 1) from `D-1` coords.
pub fn alr_inv(coords: &[f64]) -> Result<Vec<f64>> {
    let m = coords.iter().cloned().fold(0.0f64, f64::max);
    let exps: Vec<f64> = coords.iter().map(|c| (c - m).exp()).collect();
    let ref_part = (-m).exp(); // exp(0 - m) for the reference part
    let s: f64 = exps.iter().sum::<f64>() + ref_part;
    let mut out: Vec<f64> = exps.iter().map(|e| e / s).collect();
    out.push(ref_part / s);
    Ok(out)
}

/// Isometric log-ratio (Egozcue default basis), mapping `D` parts to `D-1` orthonormal
/// coordinates: `ilr(x)_i = √(i/(i+1)) · ln( g(x₁..xᵢ) / x_{i+1} )`, `i = 1..D-1`.
pub fn ilr(parts: &[f64]) -> Result<Vec<f64>> {
    let x = closure(parts, 1.0)?;
    let d = x.len();
    if d < 2 {
        return Err(CodaError::InvalidComposition("need ≥ 2 parts".into()));
    }
    let mut out = Vec::with_capacity(d - 1);
    for i in 1..d {
        let g = geometric_mean(&x[..i]);
        let coord = ((i as f64) / (i as f64 + 1.0)).sqrt() * (g / x[i]).ln();
        out.push(coord);
    }
    Ok(out)
}

/// Inverse ILR: recover the `D`-part composition from `D-1` orthonormal coordinates,
/// by reconstructing CLR coordinates from the Egozcue basis and applying `clr_inv`.
pub fn ilr_inv(coords: &[f64]) -> Result<Vec<f64>> {
    let dm1 = coords.len();
    if dm1 == 0 {
        return Err(CodaError::InvalidParameters("empty coordinates".into()));
    }
    let d = dm1 + 1;
    // Build the (D × D-1) Egozcue basis and map coords back to CLR: clr = V · coords.
    let mut clr_vec = vec![0.0f64; d];
    for (i, &c) in coords.iter().enumerate() {
        // Basis vector ψ_i (1-based index ii).
        let ii = i + 1;
        let norm = ((ii as f64) / (ii as f64 + 1.0)).sqrt();
        let upper = norm / ii as f64; // weight on each of the first ii parts
        for k in 0..ii {
            clr_vec[k] += c * upper;
        }
        clr_vec[ii] += c * (-norm);
    }
    clr_inv(&clr_vec)
}

/// Aitchison distance between two compositions (Euclidean distance in CLR space).
pub fn aitchison_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(CodaError::InvalidParameters("length mismatch".into()));
    }
    let ca = clr(a)?;
    let cb = clr(b)?;
    Ok(ca
        .iter()
        .zip(&cb)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn closure_sums_to_one() {
        let c = closure(&[2.0, 3.0, 5.0], 1.0).unwrap();
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn clr_sums_to_zero() {
        let y = clr(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(y.iter().sum::<f64>().abs() < 1e-12);
    }

    #[test]
    fn clr_round_trip() {
        let x = closure(&[1.0, 4.0, 2.0, 8.0], 1.0).unwrap();
        let back = clr_inv(&clr(&x).unwrap()).unwrap();
        assert!(approx(&x, &back, 1e-10), "{x:?} vs {back:?}");
    }

    #[test]
    fn alr_round_trip() {
        let x = closure(&[3.0, 1.0, 6.0], 1.0).unwrap();
        let back = alr_inv(&alr(&x).unwrap()).unwrap();
        assert!(approx(&x, &back, 1e-10), "{x:?} vs {back:?}");
    }

    #[test]
    fn ilr_round_trip() {
        let x = closure(&[2.0, 5.0, 1.0, 3.0, 4.0], 1.0).unwrap();
        let coords = ilr(&x).unwrap();
        assert_eq!(coords.len(), 4);
        let back = ilr_inv(&coords).unwrap();
        assert!(approx(&x, &back, 1e-9), "{x:?} vs {back:?}");
    }

    #[test]
    fn ilr_is_isometric_with_aitchison() {
        // Euclidean distance in ILR space equals the Aitchison distance.
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 1.0, 2.0, 1.0];
        let ia = ilr(&a).unwrap();
        let ib = ilr(&b).unwrap();
        let ilr_dist = ia
            .iter()
            .zip(&ib)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        let aitch = aitchison_distance(&a, &b).unwrap();
        assert!(
            (ilr_dist - aitch).abs() < 1e-9,
            "ilr {ilr_dist} aitch {aitch}"
        );
    }

    #[test]
    fn scale_invariance() {
        // Closure makes the transform scale-invariant: [1,2,3] ≡ [2,4,6].
        let c1 = clr(&[1.0, 2.0, 3.0]).unwrap();
        let c2 = clr(&[2.0, 4.0, 6.0]).unwrap();
        assert!(approx(&c1, &c2, 1e-12));
    }
}
