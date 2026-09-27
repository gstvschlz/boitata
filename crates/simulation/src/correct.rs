//! Correction of realizations to a target distribution by rank-preserving
//! quantile mapping.

use std::collections::BTreeMap;

use rayon::prelude::*;
use transforms::Reference;

use crate::error::{Result, SimError};

fn invalid<T>(message: &str) -> Result<T> {
    Err(SimError::InvalidParameters(message.into()))
}

/// The weighted empirical distribution of data: quantiles interpolate
/// linearly between the midpoints of the sorted values' cumulative weights,
/// and stop at the smallest and largest value.
#[derive(Debug, Clone)]
pub struct Empirical {
    values: Vec<f64>,
    mids: Vec<f64>,
}

impl Empirical {
    pub fn new(values: &[f64], weights: Option<&[f64]>) -> Result<Self> {
        if values.is_empty() {
            return invalid("the reference needs values");
        }
        if values.iter().any(|v| !v.is_finite()) {
            return invalid("reference values must be finite");
        }
        let w = weights.map_or_else(|| vec![1.0; values.len()], <[f64]>::to_vec);
        if w.len() != values.len() {
            return invalid("one weight per reference value");
        }
        if w.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return invalid("weights must be finite and non-negative");
        }
        let total: f64 = w.iter().sum();
        if total <= 0.0 {
            return invalid("weights must not sum to 0");
        }
        let mut order: Vec<usize> = (0..values.len()).filter(|&i| w[i] > 0.0).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let mut cum = 0.0;
        let mids = order
            .iter()
            .map(|&i| {
                cum += w[i];
                (cum - w[i] / 2.0) / total
            })
            .collect();
        Ok(Self {
            values: order.iter().map(|&i| values[i]).collect(),
            mids,
        })
    }
}

impl Reference for Empirical {
    fn quantile(&self, p: f64) -> f64 {
        let (x, m) = (&self.values, &self.mids);
        match m.partition_point(|m| *m < p) {
            0 => x[0],
            k if k == x.len() => x[k - 1],
            k => x[k - 1] + (x[k] - x[k - 1]) * (p - m[k - 1]) / (m[k] - m[k - 1]),
        }
    }

    fn support(&self) -> (f64, f64) {
        (self.values[0], self.values[self.values.len() - 1])
    }
}

/// `realizations[r]` mapped rank for rank onto the quantiles of `reference`,
/// then moved `strength` of the way from the original values to the mapped
/// ones: the value ranked `i` of `m` goes to the quantile at `(i + 0.5) / m`,
/// ties ranked by position. With `strength` 1 each realization takes the
/// reference's distribution exactly; 0 leaves it unchanged. NaN values stay
/// NaN and are not ranked. With `only`, the other realizations are returned
/// unchanged. Realizations are corrected in parallel, identically on any
/// number of threads.
pub fn correct_distribution(
    realizations: &[Vec<f64>],
    reference: &dyn Reference,
    strength: f64,
    only: Option<&[usize]>,
) -> Result<Vec<Vec<f64>>> {
    if !(0.0..=1.0).contains(&strength) {
        return invalid("strength must be in [0, 1]");
    }
    if realizations.iter().flatten().any(|v| v.is_infinite()) {
        return invalid("realizations must be finite or NaN");
    }
    let mut chosen = vec![only.is_none(); realizations.len()];
    for &r in only.unwrap_or_default() {
        match chosen.get_mut(r) {
            Some(c) => *c = true,
            None => return invalid("realization index out of range"),
        }
    }
    let count = |r: &Vec<f64>| r.iter().filter(|v| !v.is_nan()).count();
    let mut quantiles = BTreeMap::new();
    for (r, c) in realizations.iter().zip(&chosen) {
        if *c {
            quantiles.entry(count(r)).or_insert_with(Vec::new);
        }
    }
    quantiles.par_iter_mut().for_each(|(&m, q)| {
        *q = (0..m)
            .into_par_iter()
            .map(|i| reference.quantile((i as f64 + 0.5) / m as f64))
            .collect();
    });
    Ok(realizations
        .par_iter()
        .zip(&chosen)
        .map(|(r, c)| {
            if !*c {
                return r.clone();
            }
            let mut order: Vec<usize> = (0..r.len()).filter(|&i| !r[i].is_nan()).collect();
            order.sort_by(|&a, &b| r[a].total_cmp(&r[b]).then(a.cmp(&b)));
            let q = &quantiles[&order.len()];
            let mut out = r.clone();
            for (rank, &i) in order.iter().enumerate() {
                out[i] = r[i] + strength * (q[rank] - r[i]);
            }
            out
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn field(n: usize, reals: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..reals)
            .map(|_| (0..n).map(|_| rng.gen_range(0.0..2.0)).collect())
            .collect()
    }

    #[test]
    fn full_strength_reproduces_the_target_exactly_and_keeps_ranks() {
        let reals = field(300, 4, 1);
        let target: Vec<f64> = field(300, 1, 2)[0].iter().map(|v| v * v * 5.0).collect();
        let reference = Empirical::new(&target, None).unwrap();
        let out = correct_distribution(&reals, &reference, 1.0, None).unwrap();
        let mut sorted_target = target.clone();
        sorted_target.sort_by(f64::total_cmp);
        for (r, o) in reals.iter().zip(&out) {
            let mut s = o.clone();
            s.sort_by(f64::total_cmp);
            for (a, b) in s.iter().zip(&sorted_target) {
                assert!((a - b).abs() < 1e-9);
            }
            for i in 0..r.len() {
                for j in 0..r.len() {
                    if r[i] < r[j] {
                        assert!(o[i] <= o[j]);
                    }
                }
            }
        }
    }

    #[test]
    fn weighted_reference_quantiles_are_matched() {
        let reals = field(1000, 2, 3);
        let target = [1.0, 2.0, 3.0, 4.0];
        let weights = [3.0, 1.0, 1.0, 3.0];
        let reference = Empirical::new(&target, Some(&weights)).unwrap();
        let out = correct_distribution(&reals, &reference, 1.0, None).unwrap();
        let mean = out[0].iter().sum::<f64>() / 1000.0;
        assert!((mean - 2.5).abs() < 1e-9);
        assert_eq!(reference.quantile(0.5), 2.5);
    }

    #[test]
    fn strength_zero_is_the_identity_and_partial_is_linear() {
        let reals = field(100, 3, 4);
        let reference = Empirical::new(&field(50, 1, 5)[0], None).unwrap();
        assert_eq!(
            correct_distribution(&reals, &reference, 0.0, None).unwrap(),
            reals
        );
        let full = correct_distribution(&reals, &reference, 1.0, None).unwrap();
        let half = correct_distribution(&reals, &reference, 0.5, None).unwrap();
        for ((r, f), h) in reals.iter().zip(&full).zip(&half) {
            for i in 0..r.len() {
                assert!((h[i] - 0.5 * (r[i] + f[i])).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn only_selected_realizations_change_nan_is_kept_and_threads_do_not_matter() {
        let mut reals = field(200, 5, 6);
        reals[1][7] = f64::NAN;
        let reference = Empirical::new(&field(80, 1, 7)[0], None).unwrap();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| correct_distribution(&reals, &reference, 0.8, Some(&[1, 3])))
                .unwrap()
        };
        let (one, many) = (run(1), run(4));
        let bits =
            |v: &Vec<Vec<f64>>| -> Vec<u64> { v.iter().flatten().map(|x| x.to_bits()).collect() };
        assert_eq!(bits(&one), bits(&many));
        assert_eq!(one[0], reals[0]);
        assert_eq!(one[4], reals[4]);
        assert!(one[1][7].is_nan());
        assert_ne!(one[3], reals[3]);
    }

    #[test]
    fn bad_input_is_an_error() {
        let reference = Empirical::new(&[1.0, 2.0], None).unwrap();
        assert!(correct_distribution(&[vec![1.0]], &reference, 1.5, None).is_err());
        assert!(correct_distribution(&[vec![1.0]], &reference, 1.0, Some(&[1])).is_err());
        assert!(Empirical::new(&[1.0, f64::NAN], None).is_err());
        assert!(Empirical::new(&[1.0], Some(&[0.0])).is_err());
    }
}
