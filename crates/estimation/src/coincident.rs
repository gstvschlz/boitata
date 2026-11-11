//! Coincident-sample guard: samples closer than [`EPS`] krige as one datum,
//! with the mean value and the variance of the mean, and share its weight
//! equally. It runs before any solver, so whether a system with near-duplicate
//! samples is solved does not depend on its size.

use crate::Sample;
use crate::krige::Estimate;
use variogram::Variogram;

/// Euclidean distance (m) within which samples are one datum.
pub(crate) const EPS: f64 = 1e-6;

fn close(p: &(f64, f64, f64), q: &(f64, f64, f64)) -> bool {
    (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2) + (p.2 - q.2).powi(2) <= EPS * EPS
}

pub(crate) struct Clusters {
    /// Cluster of each sample, numbered in order of their first member.
    of: Vec<usize>,
    /// Members of each cluster, in sample order.
    members: Vec<Vec<usize>>,
}

impl Clusters {
    /// The clusters of `samples`, or `None` when no two coincide. Each sample
    /// joins the first cluster whose first member is within [`EPS`].
    pub(crate) fn find(samples: &[Sample]) -> Option<Self> {
        let any = samples
            .iter()
            .enumerate()
            .any(|(i, s)| samples[..i].iter().any(|t| close(&s.loc, &t.loc)));
        if !any {
            return None;
        }
        let mut of = Vec::with_capacity(samples.len());
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            match members
                .iter()
                .position(|m| close(&samples[m[0]].loc, &s.loc))
            {
                Some(c) => {
                    of.push(c);
                    members[c].push(i);
                }
                None => {
                    of.push(members.len());
                    members.push(vec![i]);
                }
            }
        }
        Some(Self { of, members })
    }

    /// One sample per cluster at its first member, with the mean value and
    /// an error variance that makes its diagonal entry the variance of the
    /// members' mean.
    pub(crate) fn merge(&self, samples: &[Sample], vg: &Variogram) -> Vec<Sample> {
        let c0 = vg.total_sill();
        self.members
            .iter()
            .map(|m| {
                let first = &samples[m[0]];
                if m.len() == 1 {
                    return first.clone();
                }
                let k = m.len() as f64;
                let mut var = 0.0;
                for &i in m {
                    var += samples[i].error_variance;
                    for &j in m {
                        var += vg.cov_points(&samples[i].loc, &samples[j].loc);
                    }
                }
                Sample {
                    value: m.iter().map(|&i| samples[i].value).sum::<f64>() / k,
                    error_variance: var / (k * k) - c0,
                    ..first.clone()
                }
            })
            .collect()
    }

    /// The mean of `rows` over each cluster.
    pub(crate) fn rows(&self, rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.members
            .iter()
            .map(|m| {
                let mut mean = rows[m[0]].clone();
                for &i in &m[1..] {
                    mean.iter_mut().zip(&rows[i]).for_each(|(a, b)| *a += b);
                }
                mean.iter_mut().for_each(|a| *a /= m.len() as f64);
                mean
            })
            .collect()
    }

    /// `e`, solved on the merged samples, back on the samples: each cluster's
    /// weight shared equally among its members and the value recomputed from
    /// those weights by `value`.
    pub(crate) fn spread(&self, mut e: Estimate, value: impl FnOnce(&[f64]) -> f64) -> Estimate {
        e.weights = self
            .of
            .iter()
            .map(|&c| e.weights[c] / self.members[c].len() as f64)
            .collect();
        e.value = value(&e.weights);
        e.n_used = self.of.len();
        e
    }
}

#[cfg(test)]
mod tests {
    use crate::Sample;
    use crate::block::block_krige_points;
    use crate::krige::{Kind, krige};
    use crate::kriging_algebra::{
        DriftSpec, DualKriging, krige_bayesian_polynomial, krige_factorial, krige_universal,
    };
    use variogram::Variogram;
    use variogram::model::Model;

    /// A near-coincident pair with values 1 and 5 at (10, 10), then lattice
    /// samples up to `n`.
    fn pair_and_lattice(n: usize, gap: f64) -> Vec<Sample> {
        let mut s = vec![
            Sample::new((10.0, 10.0, 0.0), 1.0),
            Sample::new((10.0 + gap, 10.0, 0.0), 5.0),
        ];
        let mut k = 0;
        while s.len() < n {
            let (x, y) = ((k % 9) as f64 * 7.0 - 20.0, (k / 9) as f64 * 7.0 - 20.0);
            if (x - 10.0).abs() + (y - 10.0).abs() > 1.0 {
                s.push(Sample::new((x, y, 0.0), 3.0 + (k as f64 * 0.3).sin()));
            }
            k += 1;
        }
        s
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn near_coincident_pair_outcome_independent_of_n() {
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let t = (12.0, 11.0, 0.0);
        for gap in [0.0, 1e-12, 1e-7, 1e-6] {
            for kind in [
                Kind::Ordinary,
                Kind::Simple { mean: 3.0 },
                Kind::Indicator { threshold: 2.0 },
            ] {
                let a = krige(kind, &t, &pair_and_lattice(64, gap), &vg).unwrap();
                let b = krige(kind, &t, &pair_and_lattice(65, gap), &vg).unwrap();
                for e in [&a, &b] {
                    assert_eq!(e.weights[0], e.weights[1], "{gap} {kind:?}");
                    assert!(e.value.is_finite() && e.variance.is_finite());
                }
                if !matches!(kind, Kind::Simple { .. }) {
                    assert!((a.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                }
                assert_eq!((a.n_used, b.n_used), (64, 65));
            }
        }
    }

    #[test]
    fn duplicates_regularize_with_equal_weights_in_every_solver() {
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let mut s = pair_and_lattice(12, 0.0);
        let t = (12.0, 11.0, 0.0);
        let pts = [(11.0, 10.5, 0.0), (13.0, 11.5, 0.0)];
        let point = krige(Kind::Ordinary, &t, &s, &vg).unwrap();
        let block = block_krige_points(Kind::Ordinary, &pts, &s, &vg).unwrap();
        // A linear drift in x and y (the samples lie in z = 0).
        let linear = |p: &(f64, f64, f64)| vec![1.0, p.0 - t.0, p.1 - t.1];
        let drift = DriftSpec {
            data: s.iter().map(|s| linear(&s.loc)).collect(),
            target: linear(&t),
        };
        let universal = krige_universal(&t, &s, &drift, &vg, None).unwrap();
        for e in [&point, &block, &universal] {
            assert_eq!(e.weights[0], e.weights[1]);
            assert!(e.weights.iter().all(|w| w.is_finite()));
            assert!((e.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(e.value.is_finite() && e.variance >= 0.0);
        }
        let ordinary = DriftSpec::ordinary(s.len());
        let as_ok = krige_universal(&t, &s, &ordinary, &vg, None).unwrap();
        assert!(close(as_ok.value, point.value) && close(as_ok.variance, point.variance));
        s[1].loc.0 += 1e-9;
        let shifted = krige(Kind::Ordinary, &t, &s, &vg).unwrap();
        assert!(close(shifted.value, point.value));
    }

    #[test]
    fn duplicates_regularize_in_dual_bayesian_and_factorial_kriging() {
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let s = pair_and_lattice(12, 1e-9);
        let t = (12.0, 11.0, 0.0);
        let point = krige(Kind::Ordinary, &t, &s, &vg).unwrap();
        let dual = DualKriging::new(&s, &vg, 0).unwrap();
        assert!(close(dual.estimate(&t), point.value));
        let bayes = krige_bayesian_polynomial(&t, &s, 0, &vg, &[3.0], &[1e12]).unwrap();
        let factorial = krige_factorial(&t, &s, &vg, true, &[0]).unwrap();
        for e in [&bayes, &factorial] {
            assert_eq!(e.weights[0], e.weights[1]);
            assert!(e.weights.iter().all(|w| w.is_finite()));
            assert!(e.value.is_finite() && e.variance.is_finite());
        }
        assert!((bayes.value - point.value).abs() < 1e-6);
        let mut one = s.clone();
        one.remove(1);
        one[0].value = 3.0;
        let single = krige_factorial(&t, &one, &vg, true, &[0]).unwrap();
        assert!(close(factorial.value, single.value));
        assert!(close(2.0 * factorial.weights[0], single.weights[0]));
    }

    #[test]
    fn a_datum_repeated_krige_as_its_mean_with_the_variance_of_the_mean() {
        let vg = Variogram {
            nugget: 0.2,
            ..Variogram::single(Model::Exponential, 0.8, 40.0)
        };
        let others = pair_and_lattice(20, 0.0).split_off(2);
        let t = (12.0, 11.0, 0.0);
        let with = |datum: Vec<Sample>| {
            let all: Vec<Sample> = datum.into_iter().chain(others.clone()).collect();
            krige(Kind::Ordinary, &t, &all, &vg).unwrap()
        };
        let at = |value: f64, error_variance: f64| Sample {
            error_variance,
            ..Sample::new((10.0, 10.0, 0.0), value)
        };
        let three = with(vec![at(1.0, 0.3), at(2.0, 0.3), at(6.0, 0.3)]);
        let one = with(vec![at(3.0, 0.1)]);
        assert!(close(three.value, one.value));
        assert!(close(three.variance, one.variance));
        assert!(close(3.0 * three.weights[0], one.weights[0]));
        assert!(close(three.lagrange, one.lagrange));
    }

    #[test]
    fn indicator_twins_average_their_indicators() {
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let s = pair_and_lattice(10, 0.0);
        let at = s[0].loc;
        let e = krige(Kind::Indicator { threshold: 2.0 }, &at, &s, &vg).unwrap();
        assert!(close(e.value, 0.5), "{}", e.value);
        assert!(e.variance < 1e-9);
    }
}
