//! Compositional data analysis (CoDa).
//!
//! Assay grades (and other parts-of-a-whole data) live on the simplex: only ratios
//! carry information, so ordinary statistics/kriging on raw parts are biased. This
//! crate provides the closure operation and the ALR/CLR/ILR log-ratio transforms
//! (with inverses) that map compositions to real coordinates where standard
//! geostatistics is valid, plus the Aitchison distance.
//!
//! References: Aitchison (1986); Egozcue et al. (2003), "Isometric logratio
//! transformations for compositional data analysis".

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

/// Additive log-ratio against part `reference`: `ln(x_i / x_r)` for `i ≠ r`.
pub fn alr_with(parts: &[f64], reference: usize) -> Result<Vec<f64>> {
    let x = closure(parts, 1.0)?;
    if x.len() < 2 || reference >= x.len() {
        return Err(CodaError::InvalidParameters(format!(
            "reference {reference} is not one of {} parts",
            x.len()
        )));
    }
    let r = x[reference];
    Ok((0..x.len())
        .filter(|&i| i != reference)
        .map(|i| (x[i] / r).ln())
        .collect())
}

/// Inverse of [`alr_with`]: the closed composition, part `reference` restored.
pub fn alr_with_inv(coords: &[f64], reference: usize) -> Result<Vec<f64>> {
    if reference > coords.len() {
        return Err(CodaError::InvalidParameters(format!(
            "reference {reference} is not one of {} parts",
            coords.len() + 1
        )));
    }
    let mut clr = coords.to_vec();
    clr.insert(reference, 0.0);
    clr_inv(&clr)
}

/// Orthonormal clr basis, one row of `D` weights per balance, from a
/// sequential binary partition: `D - 1` rows of signs, `1` for the parts in
/// the numerator, `-1` in the denominator and `0` outside the balance.
pub fn partition_basis(signs: &[Vec<i8>]) -> Result<Vec<Vec<f64>>> {
    let d = signs.first().map_or(0, Vec::len);
    if d < 2 || signs.len() != d - 1 || signs.iter().any(|r| r.len() != d) {
        return Err(CodaError::InvalidParameters(
            "a partition of D parts has D - 1 rows of D signs".into(),
        ));
    }
    let basis: Vec<Vec<f64>> = signs
        .iter()
        .map(|row| {
            let r = row.iter().filter(|&&s| s == 1).count() as f64;
            let s = row.iter().filter(|&&s| s == -1).count() as f64;
            if r == 0.0 || s == 0.0 || row.iter().any(|v| !(-1..=1).contains(v)) {
                return Err(CodaError::InvalidParameters(
                    "each balance needs signs 1, -1 or 0, with at least one 1 and one -1".into(),
                ));
            }
            let k = (r * s / (r + s)).sqrt();
            Ok(row
                .iter()
                .map(|&v| match v {
                    1 => k / r,
                    -1 => -k / s,
                    _ => 0.0,
                })
                .collect())
        })
        .collect::<Result<_>>()?;
    for (i, a) in basis.iter().enumerate() {
        for b in &basis[..i] {
            if dot(a, b).abs() > 1e-9 {
                return Err(CodaError::InvalidParameters(
                    "the signs are not a sequential binary partition: balances overlap".into(),
                ));
            }
        }
    }
    Ok(basis)
}

/// Isometric log-ratio coordinates on `basis` (see [`partition_basis`]).
pub fn ilr_with(parts: &[f64], basis: &[Vec<f64>]) -> Result<Vec<f64>> {
    if basis.iter().any(|b| b.len() != parts.len()) {
        return Err(CodaError::InvalidParameters(format!(
            "the basis has {} parts, the composition {}",
            basis.first().map_or(0, Vec::len),
            parts.len()
        )));
    }
    let y = clr(parts)?;
    Ok(basis.iter().map(|b| dot(&y, b)).collect())
}

/// Inverse of [`ilr_with`].
pub fn ilr_with_inv(coords: &[f64], basis: &[Vec<f64>]) -> Result<Vec<f64>> {
    let d = basis.first().map_or(0, Vec::len);
    if coords.len() != basis.len() || d == 0 {
        return Err(CodaError::InvalidParameters(format!(
            "expected {} coordinates, got {}",
            basis.len(),
            coords.len()
        )));
    }
    let y: Vec<f64> = (0..d)
        .map(|j| coords.iter().zip(basis).map(|(c, b)| c * b[j]).sum())
        .collect();
    clr_inv(&y)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

/// Multiplicative replacement of parts below their detection limit: each
/// becomes `fraction × limit`, and the parts above their limit shrink by one
/// common factor so the row keeps its total and their ratios. A part at or
/// above its limit counts as detected; zeros are below any positive limit.
pub fn replace_below_detection(parts: &[f64], limits: &[f64], fraction: f64) -> Result<Vec<f64>> {
    if limits.len() != parts.len() {
        return Err(CodaError::InvalidParameters(format!(
            "{} detection limits for {} parts",
            limits.len(),
            parts.len()
        )));
    }
    if !(fraction > 0.0 && fraction < 1.0) || limits.iter().any(|l| l.is_nan() || *l < 0.0) {
        return Err(CodaError::InvalidParameters(
            "fraction must be in (0, 1) and limits non-negative".into(),
        ));
    }
    if parts.iter().any(|&x| x < 0.0 || !x.is_finite()) {
        return Err(CodaError::InvalidComposition(
            "parts must be non-negative and finite".into(),
        ));
    }
    let below: Vec<bool> = parts.iter().zip(limits).map(|(x, l)| x < l).collect();
    let total: f64 = parts.iter().sum();
    let added: f64 = limits
        .iter()
        .zip(&below)
        .filter(|(_, b)| **b)
        .map(|(l, _)| fraction * l)
        .sum();
    let kept: f64 = parts
        .iter()
        .zip(&below)
        .filter(|(_, b)| !**b)
        .map(|(x, _)| x)
        .sum();
    if added >= total || kept <= 0.0 {
        return Err(CodaError::InvalidComposition(
            "the replacements would exceed the row total".into(),
        ));
    }
    let scale = (total - added) / kept;
    Ok(parts
        .iter()
        .zip(limits)
        .zip(&below)
        .map(|((x, l), b)| if *b { fraction * l } else { x * scale })
        .collect())
}

/// Perturbation, the simplex sum: the closed product of `a` and `b`.
pub fn perturbation(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    same_parts(a, b)?;
    closure(
        &a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<_>>(),
        1.0,
    )
}

/// Powering, the simplex scaling: the closed `x` to the power `alpha`.
pub fn powering(parts: &[f64], alpha: f64) -> Result<Vec<f64>> {
    let x = closure(parts, 1.0)?;
    closure(&x.iter().map(|v| v.powf(alpha)).collect::<Vec<_>>(), 1.0)
}

/// Aitchison inner product: the dot product of the clr coordinates.
pub fn aitchison_inner_product(a: &[f64], b: &[f64]) -> Result<f64> {
    same_parts(a, b)?;
    Ok(dot(&clr(a)?, &clr(b)?))
}

/// Aitchison norm: the length of the clr coordinates.
pub fn aitchison_norm(parts: &[f64]) -> Result<f64> {
    let y = clr(parts)?;
    Ok(dot(&y, &y).sqrt())
}

fn same_parts(a: &[f64], b: &[f64]) -> Result<()> {
    if a.len() == b.len() {
        Ok(())
    } else {
        Err(CodaError::InvalidParameters(format!(
            "{} parts against {}",
            a.len(),
            b.len()
        )))
    }
}

/// Logs of the closed rows, and the normalized weights.
fn logs(rows: &[Vec<f64>], weights: Option<&[f64]>) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let d = rows.first().map_or(0, Vec::len);
    if d < 2 || rows.iter().any(|r| r.len() != d) {
        return Err(CodaError::InvalidComposition(
            "need rows of at least 2 parts each".into(),
        ));
    }
    let w = match weights {
        Some(w) if w.len() != rows.len() => {
            return Err(CodaError::InvalidParameters(format!(
                "{} weights for {} rows",
                w.len(),
                rows.len()
            )));
        }
        Some(w) if w.iter().any(|v| v.is_nan() || *v < 0.0) || w.iter().sum::<f64>() <= 0.0 => {
            return Err(CodaError::InvalidParameters(
                "weights must be non-negative with a positive sum".into(),
            ));
        }
        Some(w) => w.to_vec(),
        None => vec![1.0; rows.len()],
    };
    let s: f64 = w.iter().sum();
    let logs = rows
        .iter()
        .map(|r| Ok(closure(r, 1.0)?.iter().map(|v| v.ln()).collect()))
        .collect::<Result<_>>()?;
    Ok((logs, w.iter().map(|v| v / s).collect()))
}

/// Center of a set of compositions: the closed, weighted geometric mean of
/// each part.
pub fn center(rows: &[Vec<f64>], weights: Option<&[f64]>) -> Result<Vec<f64>> {
    let (logs, w) = logs(rows, weights)?;
    let d = logs[0].len();
    let mean: Vec<f64> = (0..d)
        .map(|j| {
            logs.iter()
                .zip(&w)
                .map(|(l, w)| w * l[j])
                .sum::<f64>()
                .exp()
        })
        .collect();
    closure(&mean, 1.0)
}

/// Variation matrix: the weighted variance of `ln(x_i / x_j)` for each pair
/// of parts, over the sum of weights.
pub fn variation_matrix(rows: &[Vec<f64>], weights: Option<&[f64]>) -> Result<Vec<Vec<f64>>> {
    let (logs, w) = logs(rows, weights)?;
    let d = logs[0].len();
    let variance = |i: usize, j: usize| {
        let mean: f64 = logs.iter().zip(&w).map(|(l, w)| w * (l[i] - l[j])).sum();
        logs.iter()
            .zip(&w)
            .map(|(l, w)| w * (l[i] - l[j] - mean).powi(2))
            .sum::<f64>()
    };
    Ok((0..d)
        .map(|i| (0..d).map(|j| variance(i, j)).collect())
        .collect())
}

/// Total variance: the sum of the variation matrix over `2D`, equal to the
/// summed variances of the clr coordinates.
pub fn total_variance(rows: &[Vec<f64>], weights: Option<&[f64]>) -> Result<f64> {
    let t = variation_matrix(rows, weights)?;
    Ok(t.iter().flatten().sum::<f64>() / (2.0 * t.len() as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_keeps_the_total_and_the_detected_ratios() {
        let x = [0.0, 2.0, 0.003, 5.0, 3.0];
        let limits = [0.01, 0.01, 0.005, 0.01, 0.01];
        let r = replace_below_detection(&x, &limits, 0.65).unwrap();
        assert!((r.iter().sum::<f64>() - 10.003).abs() < 1e-12);
        assert!((r[0] - 0.0065).abs() < 1e-15 && (r[2] - 0.00325).abs() < 1e-15);
        assert!((r[1] / r[3] - 0.4).abs() < 1e-12 && (r[3] / r[4] - 5.0 / 3.0).abs() < 1e-12);
        assert_eq!(
            replace_below_detection(&[1.0, 2.0], &[0.1, 0.1], 0.65).unwrap(),
            vec![1.0, 2.0]
        );
        assert!(replace_below_detection(&[0.0, 0.001], &[1.0, 1.0], 0.65).is_err());
    }

    #[test]
    fn clr_turns_simplex_operations_into_vector_operations() {
        let (a, b) = ([1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 1.0]);
        let (ca, cb) = (clr(&a).unwrap(), clr(&b).unwrap());
        let sum: Vec<f64> = ca.iter().zip(&cb).map(|(x, y)| x + y).collect();
        assert!(approx(
            &clr(&perturbation(&a, &b).unwrap()).unwrap(),
            &sum,
            1e-12
        ));
        let scaled: Vec<f64> = ca.iter().map(|x| 2.5 * x).collect();
        assert!(approx(
            &clr(&powering(&a, 2.5).unwrap()).unwrap(),
            &scaled,
            1e-12
        ));
        assert!((aitchison_inner_product(&a, &b).unwrap() - dot(&ca, &cb)).abs() < 1e-12);
        let norm =
            aitchison_norm(&perturbation(&a, &powering(&b, -1.0).unwrap()).unwrap()).unwrap();
        assert!((norm - aitchison_distance(&a, &b).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn statistics_of_compositions() {
        let rows = vec![
            vec![1.0, 2.0, 7.0],
            vec![2.0, 2.0, 6.0],
            vec![4.0, 1.0, 5.0],
        ];
        let c = center(&rows, None).unwrap();
        let g = |j: usize| rows.iter().map(|r| r[j] / 10.0).product::<f64>().cbrt();
        let s = g(0) + g(1) + g(2);
        assert!(approx(&c, &[g(0) / s, g(1) / s, g(2) / s], 1e-12));
        let t = variation_matrix(&rows, None).unwrap();
        assert!(t[0][0] == 0.0 && (t[0][2] - t[2][0]).abs() < 1e-15);
        let clrs: Vec<Vec<f64>> = rows.iter().map(|r| clr(r).unwrap()).collect();
        let clr_variance: f64 = (0..3)
            .map(|j| {
                let m = clrs.iter().map(|r| r[j]).sum::<f64>() / 3.0;
                clrs.iter().map(|r| (r[j] - m).powi(2)).sum::<f64>() / 3.0
            })
            .sum();
        assert!((total_variance(&rows, None).unwrap() - clr_variance).abs() < 1e-12);
        let doubled = center(&rows, Some(&[2.0, 2.0, 2.0])).unwrap();
        assert!(approx(&doubled, &c, 1e-15));
        let first = center(&rows, Some(&[1.0, 0.0, 0.0])).unwrap();
        assert!(approx(&first, &closure(&rows[0], 1.0).unwrap(), 1e-12));
    }

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
    fn alr_with_any_reference_round_trips() {
        let x = closure(&[3.0, 1.0, 6.0, 2.0], 1.0).unwrap();
        for r in 0..4 {
            let y = alr_with(&x, r).unwrap();
            assert!(approx(&alr_with_inv(&y, r).unwrap(), &x, 1e-12));
        }
        assert!(approx(&alr_with(&x, 3).unwrap(), &alr(&x).unwrap(), 1e-12));
        assert!(alr_with(&x, 4).is_err());
    }

    #[test]
    fn partition_balances_are_orthonormal_and_invert() {
        let signs = vec![vec![1, 1, -1, -1], vec![1, -1, 0, 0], vec![0, 0, 1, -1]];
        let basis = partition_basis(&signs).unwrap();
        for (i, a) in basis.iter().enumerate() {
            for (j, b) in basis.iter().enumerate() {
                assert!((dot(a, b) - f64::from(u8::from(i == j))).abs() < 1e-12);
            }
            assert!(a.iter().sum::<f64>().abs() < 1e-12);
        }
        let x = closure(&[2.0, 5.0, 1.0, 3.0], 1.0).unwrap();
        let y = ilr_with(&x, &basis).unwrap();
        assert!(approx(&ilr_with_inv(&y, &basis).unwrap(), &x, 1e-12));
        // The first balance is the log-ratio of the geometric means, scaled.
        let g = |a: f64, b: f64| (a * b).sqrt();
        let first = (2.0f64 * 2.0 / 4.0).sqrt() * (g(x[0], x[1]) / g(x[2], x[3])).ln();
        assert!((y[0] - first).abs() < 1e-12);
    }

    #[test]
    fn partition_rejects_overlapping_or_short_signs() {
        assert!(partition_basis(&[vec![1, -1, 0], vec![1, 0, -1]]).is_err());
        assert!(partition_basis(&[vec![1, 1, 0], vec![1, -1, 0]]).is_err());
        assert!(partition_basis(&[vec![1, -1, 0]]).is_err());
    }

    #[test]
    fn scale_invariance() {
        // Closure makes the transform scale-invariant: [1,2,3] ≡ [2,4,6].
        let c1 = clr(&[1.0, 2.0, 3.0]).unwrap();
        let c2 = clr(&[2.0, 4.0, 6.0]).unwrap();
        assert!(approx(&c1, &c2, 1e-12));
    }
}
