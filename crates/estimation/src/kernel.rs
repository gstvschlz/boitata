//! Kriging of up to [`MAX`] samples on the stack: Cholesky of the sample
//! covariances in the variogram's frame, and the ordinary-kriging constraint
//! by its Schur complement.

use crate::Sample;
use crate::error::{EstimError, Result};
use crate::krige::{Estimate, Kind};
use nalgebra::Vector3;
use variogram::Variogram;

pub(crate) const MAX: usize = 64;

/// Packed lower triangle of an `MAX × MAX` matrix, row after row.
const TRI: usize = MAX * (MAX + 1) / 2;

/// Pivots at or below this fraction of the sill make the system singular.
const PIVOT: f64 = 1e-12;

/// Kriging on the stack, or `None` when the system is beyond it (no or more
/// than [`MAX`] samples, a variogram without a sill), which
/// [`crate::krige::krige_lu`] then takes. A covariance matrix that is not
/// positive definite (duplicates without nugget, a model not valid in 3D) is
/// [`EstimError::Singular`].
pub(crate) fn krige(
    kind: Kind,
    target: &(f64, f64, f64),
    samples: &[Sample],
    vg: &Variogram,
) -> Option<Result<Estimate>> {
    let n = samples.len();
    if n == 0 || n > MAX || !vg.is_stationary() {
        return None;
    }
    // Samples relative to the target, in the variogram's frame, so every
    // lag is a Euclidean norm and large coordinates lose no precision.
    let frame = vg.anisotropy.as_ref().map(|a| a.matrix());
    let mut p = [[0.0f64; 3]; MAX];
    for (q, s) in p.iter_mut().zip(samples) {
        let d = Vector3::new(s.loc.0 - target.0, s.loc.1 - target.1, s.loc.2 - target.2);
        let d = frame.map_or(d, |m| m * d);
        *q = [d.x, d.y, d.z];
    }
    let norm = |a: &[f64; 3], b: &[f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let c0 = vg.total_sill();
    let mut l = [0.0f64; TRI];
    let mut c = [0.0f64; MAX];
    for i in 0..n {
        let row = i * (i + 1) / 2;
        for j in 0..i {
            l[row + j] = vg.cov(norm(&p[i], &p[j]));
        }
        l[row + i] = c0 + samples[i].error_variance;
        c[i] = vg.cov(norm(&p[i], &[0.0; 3]));
    }
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    for j in 0..n {
        let rj = j * (j + 1) / 2;
        let d = l[rj + j] - dot(&l[rj..rj + j], &l[rj..rj + j]);
        if !(d > PIVOT * c0) {
            return Some(Err(EstimError::Singular(format!(
                "kriging matrix is not positive definite at sample {j} of {n} (duplicate locations without a nugget, or a model not valid in 3D)"
            ))));
        }
        let d = d.sqrt();
        l[rj + j] = d;
        for i in j + 1..n {
            let ri = i * (i + 1) / 2;
            l[ri + j] = (l[ri + j] - dot(&l[ri..ri + j], &l[rj..rj + j])) / d;
        }
    }
    let solve = |v: &mut [f64; MAX]| {
        for i in 0..n {
            let ri = i * (i + 1) / 2;
            v[i] = (v[i] - dot(&l[ri..ri + i], &v[..i])) / l[ri + i];
        }
        for i in (0..n).rev() {
            let mut s = v[i];
            for k in i + 1..n {
                s -= l[k * (k + 1) / 2 + i] * v[k];
            }
            v[i] = s / l[i * (i + 1) / 2 + i];
        }
    };
    let mut a = c;
    solve(&mut a);
    let ordinary = !matches!(kind, Kind::Simple { .. });
    let mut b = [0.0f64; MAX];
    let mu = if ordinary {
        b[..n].fill(1.0);
        solve(&mut b);
        (a[..n].iter().sum::<f64>() - 1.0) / b[..n].iter().sum::<f64>()
    } else {
        0.0
    };
    let weights: Vec<f64> = (0..n).map(|i| a[i] - mu * b[i]).collect();
    let value = match kind {
        Kind::Simple { mean } => {
            mean + weights
                .iter()
                .zip(samples)
                .map(|(w, s)| w * (s.value - mean))
                .sum::<f64>()
        }
        Kind::Indicator { threshold } => weights
            .iter()
            .zip(samples)
            .map(|(w, s)| w * f64::from(u8::from(s.value <= threshold)))
            .sum::<f64>()
            .clamp(0.0, 1.0),
        Kind::Ordinary => weights.iter().zip(samples).map(|(w, s)| w * s.value).sum(),
    };
    let sum_wc: f64 = weights.iter().zip(&c).map(|(w, c)| w * c).sum();
    Some(Ok(Estimate {
        value,
        variance: (c0 - sum_wc - mu).max(0.0),
        n_used: n,
        weights,
        lagrange: mu,
        support_variance: c0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::krige_lu;
    use variogram::aniso::{Angles, Anisotropy};
    use variogram::model::{Model, Structure};

    fn unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn variogram() -> Variogram {
        Variogram {
            nugget: 0.1,
            structures: vec![
                Structure::new(Model::Spherical, 0.6, 80.0),
                Structure::new(Model::Exponential, 0.3, 200.0),
            ],
            anisotropy: None,
        }
        .with_anisotropy(
            Anisotropy::new(Angles {
                azimuth: 30.0,
                dip: 10.0,
                rake: 0.0,
                major: 1.0,
                semi: 0.5,
                minor: 0.25,
            })
            .unwrap(),
        )
    }

    /// `n` samples scattered around a target at UTM-sized coordinates.
    fn system(state: &mut u64, n: usize) -> ((f64, f64, f64), Vec<Sample>) {
        let origin = (650_000.0, 7_400_000.0, 1_200.0);
        let samples = (0..n)
            .map(|_| {
                let p = (
                    origin.0 + unit(state) * 200.0,
                    origin.1 + unit(state) * 200.0,
                    origin.2 + unit(state) * 50.0,
                );
                Sample::new(p, unit(state) * 3.0)
            })
            .collect();
        let target = (origin.0 + 100.0, origin.1 + 100.0, origin.2 + 25.0);
        (target, samples)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn matches_lu() {
        let vg = variogram();
        let mut state = 7;
        for trial in 0..300 {
            let n = 1 + trial % 40;
            let (target, samples) = system(&mut state, n);
            for kind in [
                Kind::Ordinary,
                Kind::Simple { mean: 1.2 },
                Kind::Indicator { threshold: 1.5 },
            ] {
                let fast = krige(kind, &target, &samples, &vg).unwrap().unwrap();
                let slow = krige_lu(kind, &target, &samples, &vg).unwrap();
                assert!(close(fast.value, slow.value), "value {trial} {kind:?}");
                assert!(close(fast.variance, slow.variance), "variance {trial} {kind:?}");
                assert!(close(fast.lagrange, slow.lagrange), "lagrange {trial} {kind:?}");
                for (a, b) in fast.weights.iter().zip(&slow.weights) {
                    assert!(close(*a, *b), "weight {trial} {kind:?}");
                }
            }
        }
    }

    #[test]
    fn ordinary_weights_sum_to_one() {
        let vg = variogram();
        let mut state = 11;
        for n in [1, 5, 32, MAX] {
            let (target, samples) = system(&mut state, n);
            let e = krige(Kind::Ordinary, &target, &samples, &vg).unwrap().unwrap();
            assert!((e.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn exact_at_a_datum_without_nugget() {
        let vg = Variogram {
            nugget: 0.0,
            ..variogram()
        };
        let mut state = 3;
        let (_, samples) = system(&mut state, 20);
        let e = krige(Kind::Ordinary, &samples[4].loc, &samples, &vg)
            .unwrap()
            .unwrap();
        assert!((e.value - samples[4].value).abs() < 1e-9);
        assert!(e.variance < 1e-9);
    }

    #[test]
    fn duplicate_locations_are_singular() {
        let vg = Variogram {
            nugget: 0.0,
            ..variogram()
        };
        let mut state = 5;
        let (target, mut samples) = system(&mut state, 6);
        samples.push(samples[0].clone());
        assert!(matches!(
            krige(Kind::Ordinary, &target, &samples, &vg),
            Some(Err(EstimError::Singular(_)))
        ));
        assert!(matches!(
            crate::krige::krige(Kind::Ordinary, &target, &samples, &vg),
            Err(EstimError::Singular(_))
        ));
    }

    #[test]
    fn falls_back_beyond_max_and_without_a_sill() {
        let mut state = 9;
        let (target, samples) = system(&mut state, MAX + 1);
        assert!(krige(Kind::Ordinary, &target, &samples, &variogram()).is_none());
        let power = Variogram::single(Model::Power { exponent: 1.5 }, 1.0, 100.0);
        assert!(krige(Kind::Ordinary, &target, &samples[..8], &power).is_none());
        assert!(krige(Kind::Ordinary, &target, &[], &variogram()).is_none());
    }

    #[test]
    fn measurement_error_enters_the_diagonal() {
        let vg = variogram();
        let mut state = 13;
        let (target, mut samples) = system(&mut state, 10);
        samples[2].error_variance = 0.4;
        let fast = krige(Kind::Ordinary, &target, &samples, &vg).unwrap().unwrap();
        let slow = krige_lu(Kind::Ordinary, &target, &samples, &vg).unwrap();
        assert!(close(fast.value, slow.value) && close(fast.variance, slow.variance));
    }
}
