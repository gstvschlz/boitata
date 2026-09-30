//! Estimation over many targets.

use boitata_core::Progress;
use rayon::prelude::*;
use variogram::Variogram;

use crate::error::EstimError;
use crate::search::{Search, SearchTree};
use crate::{Result, Sample};

/// Selects the neighbors of every target and applies `estimator` to them, in
/// parallel. A target with too few neighbors, or whose system fails, gives
/// `None`. The output follows `targets` and does not depend on the number of
/// threads. `domains`, one code per target, confines each target to the
/// samples of its domain, and of others within [`Search::soft`].
pub fn estimate_many<F, T>(
    targets: &[(f64, f64, f64)],
    domains: Option<&[u32]>,
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Vec<Option<T>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<T> + Sync,
    T: Send,
{
    estimate_many_with(targets, domains, samples, search, vg, estimator, None)
}

/// As [`estimate_many`], ticking `progress` for each target estimated. Targets
/// left for a later search pass are not counted, so a caller running passes
/// tops the counter up once they are done.
pub fn estimate_many_with<F, T>(
    targets: &[(f64, f64, f64)],
    domains: Option<&[u32]>,
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
    progress: Option<&Progress>,
) -> Vec<Option<T>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<T> + Sync,
    T: Send,
{
    let tree = SearchTree::new(samples, search, vg);
    targets
        .par_iter()
        .enumerate()
        .map(|(i, target)| {
            let chosen = tree.neighbors_in(target, domains.map(|d| d[i])).ok()?;
            let chosen = tree.balanced(target, None, chosen);
            let selected = tree.take(target, None, &chosen, samples);
            let result = estimator(target, &selected).ok();
            tick(progress, &result);
            result
        })
        .collect()
}

fn tick<T>(progress: Option<&Progress>, result: &Option<T>) {
    if let (Some(p), Some(_)) = (progress, result) {
        p.inc();
    }
}

/// As [`estimate_many`], but for an estimator that also needs external-drift
/// covariates: `covariates` has one row per sample, aligned with `samples`,
/// and `ext` one row per target, aligned with `targets`. Each target's
/// estimator call receives the covariate rows of its chosen neighbors,
/// gathered in the same order as the samples it receives.
#[allow(clippy::too_many_arguments)]
pub fn estimate_many_ext<F, T>(
    targets: &[(f64, f64, f64)],
    domains: Option<&[u32]>,
    covariates: &[Vec<f64>],
    ext: &[Vec<f64>],
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
    progress: Option<&Progress>,
) -> Vec<Option<T>>
where
    F: Fn(&(f64, f64, f64), &[Sample], &[Vec<f64>], &[f64]) -> Result<T> + Sync,
    T: Send,
{
    let tree = SearchTree::new(samples, search, vg);
    targets
        .par_iter()
        .enumerate()
        .map(|(i, target)| {
            let chosen = tree.neighbors_in(target, domains.map(|d| d[i])).ok()?;
            let chosen = tree.balanced(target, None, chosen);
            let selected = tree.take(target, None, &chosen, samples);
            let cov: Vec<Vec<f64>> = chosen.iter().map(|&j| covariates[j].clone()).collect();
            let result = estimator(target, &selected, &cov, &ext[i]).ok();
            tick(progress, &result);
            result
        })
        .collect()
}

/// Samples used for every target and the `weights` the estimator gives them,
/// in the order of the samples it receives, in parallel; as [`estimate_many`].
pub fn weights_many<F>(
    targets: &[(f64, f64, f64)],
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    weights: F,
) -> Vec<Option<(Vec<usize>, Vec<f64>)>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<Vec<f64>> + Sync,
{
    let tree = SearchTree::new(samples, search, vg);
    targets
        .par_iter()
        .map(|target| {
            let chosen = tree.balanced(target, None, tree.neighbors_in(target, None).ok()?);
            let selected = tree.take(target, None, &chosen, samples);
            Some((chosen, weights(target, &selected).ok()?))
        })
        .collect()
}

/// Declustering weights from estimation weights: each sample's weight is the
/// sum of the weights it receives over targets covering the domain, as given
/// by [`weights_many`], scaled to sum to the number of samples. Unestimated
/// targets add nothing; the sum is taken in target order.
pub fn weight_declustering(
    per_target: &[Option<(Vec<usize>, Vec<f64>)>],
    values: &[f64],
) -> Result<transforms::Weights> {
    let mut weights = vec![0.0; values.len()];
    for (used, w) in per_target.iter().flatten() {
        for (&i, &w) in used.iter().zip(w) {
            weights[i] += w;
        }
    }
    let total: f64 = weights.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return Err(EstimError::InsufficientData(
            "no target was estimated".into(),
        ));
    }
    let scale = values.len() as f64 / total;
    weights.iter_mut().for_each(|w| *w *= scale);
    let declustered_mean =
        weights.iter().zip(values).map(|(w, v)| w * v).sum::<f64>() / values.len() as f64;
    Ok(transforms::Weights {
        weights,
        declustered_mean,
        cell_size: f64::NAN,
    })
}

/// Runs `pass(search, remaining)` for each search in turn on the indices of
/// the `n` targets that earlier passes left unestimated; `pass` returns one
/// estimate per remaining index. Each estimate comes with the index of the
/// pass that made it.
pub fn by_pass<F, T>(n: usize, searches: &[Search], mut pass: F) -> Result<Vec<Option<(usize, T)>>>
where
    F: FnMut(&Search, &[usize]) -> Result<Vec<Option<T>>>,
{
    let mut out: Vec<_> = (0..n).map(|_| None).collect();
    for (p, search) in searches.iter().enumerate() {
        let remaining: Vec<usize> = (0..n).filter(|&i| out[i].is_none()).collect();
        if remaining.is_empty() {
            break;
        }
        for (i, e) in remaining.iter().zip(pass(search, &remaining)?) {
            out[*i] = e.map(|e| (p, e));
        }
    }
    Ok(out)
}

/// Estimates every sample from its neighbors with the sample itself left out.
pub fn leave_one_out_many<F, T>(
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Vec<Option<T>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<T> + Sync,
    T: Send,
{
    let all: Vec<usize> = (0..samples.len()).collect();
    leave_one_out_at(&all, samples, search, vg, estimator)
}

/// As [`leave_one_out_many`] for the samples at indices `which` only.
pub fn leave_one_out_at<F, T>(
    which: &[usize],
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Vec<Option<T>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<T> + Sync,
    T: Send,
{
    let wider = Search {
        max_samples: search.max_samples + 1,
        ..search.clone()
    };
    let tree = SearchTree::new(samples, &wider, vg);
    which
        .par_iter()
        .map(|&i| {
            let target = &samples[i].loc;
            let mut chosen = tree
                .neighbors_around(target, samples[i].domain, Some(i))
                .unwrap_or_default();
            chosen.retain(|&j| j != i);
            chosen.truncate(search.max_samples);
            if chosen.len() < search.min_samples.max(1) {
                return None;
            }
            let chosen = tree.balanced(target, None, chosen);
            let selected = tree.take(target, None, &chosen, samples);
            estimator(target, &selected).ok()
        })
        .collect()
}

/// K-fold cross-validation: estimates the samples at indices `which` from
/// the samples outside their fold. Holes stay whole: the `j`-th of the
/// sorted hole ids goes to fold `j % k`, and an untagged sample `i` to fold
/// `i % k`. Without holes and with `k` equal to the number of samples this
/// is leave-one-out.
pub fn k_fold_at<F, T>(
    k: usize,
    which: &[usize],
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Result<Vec<Option<T>>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<T> + Sync,
    T: Send,
{
    if k < 2 {
        return Err(EstimError::InvalidParameters("k must be ≥ 2".into()));
    }
    let mut holes: Vec<u32> = samples.iter().filter_map(|s| s.hole).collect();
    holes.sort_unstable();
    holes.dedup();
    let of: Vec<usize> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| s.hole.map_or(i, |h| holes.partition_point(|&x| x < h)) % k)
        .collect();
    let mut out: Vec<Option<T>> = which.iter().map(|_| None).collect();
    for fold in 0..k {
        let (test, held): (Vec<usize>, Vec<&Sample>) = which
            .iter()
            .enumerate()
            .filter(|(_, i)| of[**i] == fold)
            .map(|(p, &i)| (p, &samples[i]))
            .unzip();
        if test.is_empty() {
            continue;
        }
        let at: Vec<_> = held.iter().map(|s| s.loc).collect();
        let domains: Option<Vec<u32>> = held.iter().map(|s| s.domain).collect();
        let train: Vec<Sample> = (0..samples.len())
            .filter(|&i| of[i] != fold)
            .map(|i| samples[i].clone())
            .collect();
        let estimates = estimate_many(&at, domains.as_deref(), &train, search, vg, &estimator);
        for (p, e) in test.into_iter().zip(estimates) {
            out[p] = e;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::{Estimate, Kind, krige};
    use crate::kriging_algebra::krige_calibrated;
    use crate::search::{Calibration, HighGrade, HighGradeMode, Soft, SoftPair};
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Point = (f64, f64, f64);
    use variogram::Model;

    fn samples() -> Vec<Sample> {
        (0..200)
            .map(|i| {
                let (x, y) = ((i * 37 % 101) as f64, (i * 53 % 97) as f64);
                Sample::new((x, y, 0.0), (x / 10.0).sin() + y / 50.0)
            })
            .collect()
    }

    fn run(threads: usize) -> Vec<Option<Estimate>> {
        let samples = samples();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let search = Search {
            min_samples: 4,
            max_samples: 12,
            radius: 30.0,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
            high_grade: None,
            soft: None,
            calibration: None,
        };
        let targets: Vec<_> = (0..400)
            .map(|i| ((i % 20) as f64 * 5.0, (i / 20) as f64 * 5.0, 0.0))
            .collect();
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                estimate_many(&targets, None, &samples, &search, Some(&vg), |t, s| {
                    krige(Kind::Ordinary, t, s, &vg)
                })
            })
    }

    #[test]
    fn results_do_not_depend_on_thread_count() {
        let (one, many) = (run(1), run(8));
        assert_eq!(one.len(), many.len());
        for (a, b) in one.iter().zip(&many) {
            assert_eq!(
                a.as_ref().map(|e| e.value.to_bits()),
                b.as_ref().map(|e| e.value.to_bits())
            );
        }
    }

    #[test]
    fn progress_counts_estimated_targets_and_leaves_results_unchanged() {
        let samples = samples();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let search = Search {
            min_samples: 4,
            max_samples: 12,
            radius: 30.0,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
            high_grade: None,
            soft: None,
            calibration: None,
        };
        let targets: Vec<_> = (0..50).map(|i| (i as f64 * 5.0, 0.0, 0.0)).collect();
        let at = |t: &Point, s: &[Sample]| krige(Kind::Ordinary, t, s, &vg);
        let plain = estimate_many(&targets, None, &samples, &search, Some(&vg), at);
        let counter = Progress::new(Some(50));
        let counted = estimate_many_with(
            &targets,
            None,
            &samples,
            &search,
            Some(&vg),
            at,
            Some(&counter),
        );
        let bits = |v: &[Option<Estimate>]| -> Vec<_> {
            v.iter()
                .map(|e| e.as_ref().map(|e| e.value.to_bits()))
                .collect()
        };
        assert_eq!(bits(&plain), bits(&counted));
        let estimated = plain.iter().flatten().count() as u64;
        assert!(estimated < 50);
        assert_eq!(counter.snapshot().0, estimated);
    }

    #[test]
    fn leave_one_out_matches_the_ordinary_kriging_version() {
        let samples = samples();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let search = Search {
            min_samples: 2,
            max_samples: 10,
            radius: 30.0,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
            high_grade: None,
            soft: None,
            calibration: None,
        };
        let ours = leave_one_out_many(&samples, &search, Some(&vg), |t, s| {
            krige(Kind::Ordinary, t, s, &vg)
        });
        let reference = crate::validate::leave_one_out(&samples, &vg, &search).unwrap();
        let ours: Vec<f64> = ours.into_iter().flatten().map(|e| e.value).collect();
        assert_eq!(ours.len(), reference.records.len());
        for (a, b) in ours.iter().zip(&reference.records) {
            assert!((a - b.estimate).abs() < 1e-9);
        }
    }

    #[test]
    fn k_fold_with_one_sample_per_fold_is_leave_one_out() {
        let samples = samples();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let (search, all) = (search(4, 30.0), (0..samples.len()).collect::<Vec<_>>());
        let loo = leave_one_out_many(&samples, &search, Some(&vg), krige_with(&vg));
        let folds = k_fold_at(
            samples.len(),
            &all,
            &samples,
            &search,
            Some(&vg),
            krige_with(&vg),
        );
        let folds = folds.unwrap();
        assert!(loo.iter().filter(|e| e.is_some()).count() > 150);
        for (a, b) in loo.iter().zip(&folds) {
            assert_eq!(bits(a), bits(b));
        }
        let reference = crate::validate::leave_one_out(&samples, &vg, &search).unwrap();
        let k_fold = crate::validate::k_fold(&samples, &vg, &search, samples.len()).unwrap();
        for (a, b) in reference.records.iter().zip(&k_fold.records) {
            assert_eq!(a.estimate.to_bits(), b.estimate.to_bits());
        }
        assert!(k_fold_at(1, &all, &samples, &search, Some(&vg), krige_with(&vg)).is_err());
    }

    #[test]
    fn k_fold_never_splits_a_hole() {
        let samples: Vec<Sample> = samples()
            .into_iter()
            .enumerate()
            .map(|(i, s)| match i % 7 {
                0 => s,
                _ => Sample::with_hole(s.loc, s.value, 1000 - (i / 5) as u32),
            })
            .collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let (search, all) = (search(2, 60.0), (0..samples.len()).collect::<Vec<_>>());
        let hole_at = |t: &Point| samples.iter().find(|s| s.loc == *t).unwrap().hole;
        let used = AtomicUsize::new(0);
        let check = |t: &Point, s: &[Sample]| {
            if let Some(h) = hole_at(t) {
                assert!(s.iter().all(|x| x.hole != Some(h)));
                used.fetch_add(1, Ordering::Relaxed);
            }
            krige(Kind::Ordinary, t, s, &vg)
        };
        for k in [2, 5, samples.len()] {
            let out = k_fold_at(k, &all, &samples, &search, Some(&vg), check).unwrap();
            assert!(out.iter().all(Option::is_some));
        }
        assert!(used.into_inner() > 0);
    }

    #[test]
    fn k_fold_with_measurement_error_does_not_depend_on_thread_count() {
        let samples: Vec<Sample> = samples()
            .into_iter()
            .enumerate()
            .map(|(i, s)| Sample {
                error_variance: (i % 3) as f64 * 0.1,
                ..s
            })
            .collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let (search, all) = (search(4, 30.0), (0..samples.len()).collect::<Vec<_>>());
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    k_fold_at(5, &all, &samples, &search, Some(&vg), krige_with(&vg)).unwrap()
                })
        };
        let (one, many) = (run(1), run(8));
        assert!(one.iter().all(Option::is_some));
        for (a, b) in one.iter().zip(&many) {
            assert_eq!(bits(a), bits(b));
        }
    }

    #[test]
    fn kriging_honors_the_data() {
        let samples = samples();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let search = Search {
            min_samples: 1,
            max_samples: 12,
            radius: 30.0,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
            high_grade: None,
            soft: None,
            calibration: None,
        };
        let targets: Vec<_> = samples.iter().map(|s| s.loc).collect();
        let out = estimate_many(&targets, None, &samples, &search, Some(&vg), |t, s| {
            krige(Kind::Ordinary, t, s, &vg)
        });
        for (e, s) in out.iter().zip(&samples) {
            let e = e.as_ref().unwrap();
            assert!((e.value - s.value).abs() < 1e-8);
            assert!(e.variance.abs() < 1e-8);
        }
    }

    fn bits(e: &Option<Estimate>) -> Option<u64> {
        e.as_ref().map(|e| e.value.to_bits())
    }

    fn krige_with(vg: &Variogram) -> impl Fn(&Point, &[Sample]) -> Result<Estimate> + Sync {
        |t, s| krige(Kind::Ordinary, t, s, vg)
    }

    fn many(targets: &[Point], search: &Search) -> Vec<Option<Estimate>> {
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        estimate_many(
            targets,
            None,
            &samples(),
            search,
            Some(&vg),
            krige_with(&vg),
        )
    }

    fn passes(targets: &[Point], searches: &[Search]) -> Vec<Option<(usize, Estimate)>> {
        let (samples, vg) = (samples(), Variogram::single(Model::Spherical, 1.0, 40.0));
        by_pass(targets.len(), searches, |search, remaining| {
            let at: Vec<Point> = remaining.iter().map(|&i| targets[i]).collect();
            Ok(estimate_many(
                &at,
                None,
                &samples,
                search,
                Some(&vg),
                krige_with(&vg),
            ))
        })
        .unwrap()
    }

    fn grid() -> Vec<Point> {
        (0..900)
            .map(|i| ((i % 30) as f64 * 5.0, (i / 30) as f64 * 5.0, 0.0))
            .collect()
    }

    fn search(min_samples: usize, radius: f64) -> Search {
        Search {
            min_samples,
            max_samples: 12,
            radius,
            ..Default::default()
        }
    }

    #[test]
    fn a_single_pass_is_estimate_many() {
        let (targets, s) = (grid(), search(4, 30.0));
        let one = passes(&targets, std::slice::from_ref(&s));
        for (p, e) in one.into_iter().zip(&many(&targets, &s)) {
            assert_eq!(p.as_ref().map(|p| p.0), e.as_ref().map(|_| 0));
            assert_eq!(bits(&p.map(|p| p.1)), bits(e));
        }
    }

    #[test]
    fn later_passes_fill_only_what_earlier_ones_left() {
        let targets = grid();
        let (tight, wide) = (search(6, 10.0), search(2, 60.0));
        let (first, second) = (many(&targets, &tight), many(&targets, &wide));
        let mut filled = [0, 0];
        for ((p, a), b) in passes(&targets, &[tight, wide])
            .into_iter()
            .zip(&first)
            .zip(&second)
        {
            let (expected, pass) = if a.is_some() { (a, 0) } else { (b, 1) };
            assert_eq!(p.as_ref().map(|p| p.0), expected.as_ref().map(|_| pass));
            assert_eq!(bits(&p.map(|p| p.1)), bits(expected));
            filled[pass] += expected.is_some() as usize;
        }
        assert!(filled[0] > 0 && filled[1] > 0, "{filled:?}");
    }

    /// 400 samples over 200 × 200 of a Gaussian field with variogram `vg`.
    fn gaussian(vg: &Variogram) -> Vec<Sample> {
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let locs: Vec<Point> = (0..400)
            .map(|_| (next() * 200.0, next() * 200.0, 0.0))
            .collect();
        let cov = nalgebra::DMatrix::from_fn(400, 400, |i, j| vg.cov_points(&locs[i], &locs[j]));
        let normals = nalgebra::DVector::from_fn(400, |_, _| {
            let (u, v) = (next().max(1e-300), next());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        });
        let field = cov.cholesky().unwrap().l() * normals;
        locs.iter()
            .zip(&field)
            .map(|(&l, &z)| Sample::new(l, z))
            .collect()
    }

    #[test]
    fn a_higher_calibration_takes_more_samples_and_smooths() {
        let vg = Variogram {
            nugget: 0.2,
            ..Variogram::single(Model::Spherical, 0.8, 60.0)
        };
        let samples = gaussian(&vg);
        let targets: Vec<Point> = (0..900)
            .map(|i| {
                (
                    3.0 + (i % 30) as f64 * 6.5,
                    3.0 + (i / 30) as f64 * 6.5,
                    0.0,
                )
            })
            .collect();
        let run = |calibration: Option<Calibration>, octant: bool, threads: usize| {
            let search = Search {
                min_samples: 2,
                max_samples: 24,
                octant,
                calibration,
                ..Default::default()
            };
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    estimate_many(&targets, None, &samples, &search, Some(&vg), |t, s| {
                        match calibration {
                            Some(c) => krige_calibrated(Kind::Ordinary, t, s, &vg, c, 2),
                            None => Ok((krige(Kind::Ordinary, t, s, &vg)?, true)),
                        }
                    })
                })
                .into_iter()
                .map(Option::unwrap)
                .collect::<Vec<_>>()
        };
        let summary = |out: &[(Estimate, bool)]| {
            let n = out.len() as f64;
            let used = out.iter().map(|e| e.0.n_used as f64).sum::<f64>() / n;
            let mean = out.iter().map(|e| e.0.value).sum::<f64>() / n;
            let spread = out.iter().map(|e| (e.0.value - mean).powi(2)).sum::<f64>() / n;
            (used, spread)
        };
        let values = |out: &[(Estimate, bool)]| -> Vec<(u64, bool)> {
            out.iter().map(|e| (e.0.value.to_bits(), e.1)).collect()
        };
        for octant in [false, true] {
            let runs: Vec<(f64, f64)> = [0.5, 0.7, 0.85]
                .into_iter()
                .map(|s| summary(&run(Some(Calibration::Slope(s)), octant, 4)))
                .collect();
            for pair in runs.windows(2) {
                assert!(pair[1].0 > pair[0].0 && pair[1].1 < pair[0].1, "{runs:?}");
            }
            let c = Some(Calibration::Efficiency(0.6));
            assert_eq!(values(&run(c, octant, 1)), values(&run(c, octant, 8)));
            let out = run(c, octant, 3);
            assert!(out.iter().any(|e| e.1) && out.iter().any(|e| !e.1));
            assert!(octant || out.iter().all(|e| e.1 || e.0.n_used == 24));
        }
        let plain = values(&run(None, true, 4));
        let direct = estimate_many(
            &targets,
            None,
            &samples,
            &Search {
                min_samples: 2,
                max_samples: 24,
                octant: true,
                ..Default::default()
            },
            Some(&vg),
            |t, s| krige(Kind::Ordinary, t, s, &vg),
        );
        let direct: Vec<(u64, bool)> = direct
            .into_iter()
            .map(|e| (e.unwrap().value.to_bits(), true))
            .collect();
        assert_eq!(plain, direct);
    }

    #[test]
    fn calibrated_octant_searches_take_the_sectors_in_turn() {
        let samples: Vec<Sample> = (0..40)
            .map(|i| {
                let (r, a) = (1.0 + i as f64, i as f64 * 0.37);
                Sample::new((r * a.cos(), r * a.sin(), 0.0), 0.0)
            })
            .collect();
        let octant = Search {
            min_samples: 1,
            max_samples: 16,
            octant: true,
            ..Default::default()
        };
        let calibrated = Search {
            calibration: Some(Calibration::Slope(0.9)),
            ..octant.clone()
        };
        let t = (0.0, 0.0, 0.0);
        let plain = SearchTree::new(&samples, &octant, None);
        let chosen = plain.neighbors(&t).unwrap();
        assert_eq!(plain.balanced(&t, None, chosen.clone()), chosen);
        let tree = SearchTree::new(&samples, &calibrated, None);
        let turns = tree.balanced(&t, None, tree.neighbors(&t).unwrap());
        let mut sorted = turns.clone();
        sorted.sort_unstable();
        let mut expected = chosen.clone();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
        let quadrant = |i: usize| {
            let (x, y, _) = samples[i].loc;
            ((x >= 0.0) as usize) << 1 | (y >= 0.0) as usize
        };
        let mut first: Vec<usize> = turns[..4].iter().map(|&i| quadrant(i)).collect();
        first.sort_unstable();
        assert_eq!(first, [0, 1, 2, 3]);
    }

    #[test]
    fn more_samples_raise_the_slope_and_smooth_the_blocks() {
        let vg = Variogram {
            nugget: 0.2,
            ..Variogram::single(Model::Spherical, 0.8, 60.0)
        };
        let samples = gaussian(&vg);
        let blocks: Vec<Point> = (0..400)
            .map(|i| {
                (
                    5.0 + (i % 20) as f64 * 10.0,
                    5.0 + (i / 20) as f64 * 10.0,
                    0.0,
                )
            })
            .collect();
        let disc = crate::Discretization {
            nx: 3,
            ny: 3,
            nz: 1,
        };
        let scores = |max_samples| {
            let search = Search {
                min_samples: 1,
                max_samples,
                ..Default::default()
            };
            let out: Vec<Estimate> =
                estimate_many(&blocks, None, &samples, &search, Some(&vg), |t, s| {
                    crate::block_krige(t, &(10.0, 10.0, 0.0), s, &disc, &vg)
                })
                .into_iter()
                .map(Option::unwrap)
                .collect();
            let n = out.len() as f64;
            let slope = out.iter().map(Estimate::slope).sum::<f64>() / n;
            let mean = out.iter().map(|e| e.value).sum::<f64>() / n;
            let spread = out.iter().map(|e| (e.value - mean).powi(2)).sum::<f64>() / n;
            (slope, spread / out[0].support_variance)
        };
        let runs: Vec<(f64, f64)> = [2, 4, 8, 16].into_iter().map(scores).collect();
        for pair in runs.windows(2) {
            assert!(pair[1].0 > pair[0].0, "{runs:?}");
            assert!(pair[1].1 < pair[0].1, "{runs:?}");
        }
    }

    #[test]
    fn a_high_grade_radius_beyond_the_search_changes_nothing() {
        let (samples, targets, plain) = (samples(), grid(), search(4, 30.0));
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let loo = |s: &Search| leave_one_out_many(&samples, s, Some(&vg), krige_with(&vg));
        for radius in [30.0, 100.0] {
            let restricted = Search {
                high_grade: Some(HighGrade::new(1.0, radius)),
                ..plain.clone()
            };
            let a = many(&targets, &plain).into_iter().chain(loo(&plain));
            let b = many(&targets, &restricted)
                .into_iter()
                .chain(loo(&restricted));
            for (a, b) in a.zip(b) {
                assert_eq!(bits(&a), bits(&b));
            }
        }
    }

    #[test]
    fn a_high_grade_sample_is_never_used_beyond_its_radius() {
        let (samples, targets, plain) = (samples(), grid(), search(4, 30.0));
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let (threshold, radius) = (1.0, 8.0);
        let restricted = Search {
            high_grade: Some(HighGrade::new(threshold, radius)),
            ..plain.clone()
        };
        let used = AtomicUsize::new(0);
        let check = |t: &Point, s: &[Sample]| {
            for x in s.iter().filter(|x| x.value > threshold) {
                assert!(variogram::aniso::euclidean(t, &x.loc) <= radius);
                used.fetch_add(1, Ordering::Relaxed);
            }
            krige(Kind::Ordinary, t, s, &vg)
        };
        let out = estimate_many(&targets, None, &samples, &restricted, Some(&vg), check);
        let cv = leave_one_out_many(&samples, &restricted, Some(&vg), check);
        assert!(used.into_inner() > 0);
        let changed = |a: &[Option<Estimate>], b: &[Option<Estimate>]| {
            a.iter().zip(b).any(|(a, b)| bits(a) != bits(b))
        };
        assert!(changed(&out, &many(&targets, &plain)));
        let plain_cv = leave_one_out_many(&samples, &plain, Some(&vg), krige_with(&vg));
        assert!(changed(&cv, &plain_cv));
    }

    fn restricted(mode: HighGradeMode, anisotropy: Option<variogram::Anisotropy>) -> Search {
        Search {
            high_grade: Some(HighGrade {
                anisotropy,
                mode,
                ..HighGrade::new(1.0, 8.3)
            }),
            ..search(4, 30.0)
        }
    }

    #[test]
    fn a_spherical_high_grade_ellipsoid_is_the_scalar_radius() {
        let sphere = variogram::Anisotropy::new(variogram::Angles {
            azimuth: 30.0,
            dip: 20.0,
            rake: 10.0,
            major: 1.0,
            semi: 1.0,
            minor: 1.0,
        })
        .unwrap();
        let (samples, targets) = (samples(), grid());
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let loo = |s: &Search| leave_one_out_many(&samples, s, Some(&vg), krige_with(&vg));
        for mode in [HighGradeMode::Drop, HighGradeMode::Clamp] {
            let (a, b) = (
                restricted(mode, None),
                restricted(mode, Some(sphere.clone())),
            );
            let x = many(&targets, &a).into_iter().chain(loo(&a));
            let y = many(&targets, &b).into_iter().chain(loo(&b));
            for (x, y) in x.zip(y) {
                assert_eq!(bits(&x), bits(&y));
            }
        }
    }

    #[test]
    fn a_high_grade_ellipsoid_restricts_along_its_axes() {
        let east = variogram::Anisotropy::new(variogram::Angles {
            azimuth: 90.0,
            dip: 0.0,
            rake: 0.0,
            major: 1.0,
            semi: 0.25,
            minor: 0.25,
        })
        .unwrap();
        let samples = vec![
            Sample::new((10.0, 0.0, 0.0), 5.0),
            Sample::new((0.0, 10.0, 0.0), 5.0),
            Sample::new((-3.0, -3.0, 0.0), 0.0),
        ];
        let s = Search {
            high_grade: Some(HighGrade {
                anisotropy: Some(east),
                ..HighGrade::new(1.0, 20.0)
            }),
            ..search(1, 100.0)
        };
        let got = crate::search::neighbors(&(0.0, 0.0, 0.0), &samples, &s, None).unwrap();
        assert_eq!(got, vec![2, 0]);
    }

    #[test]
    fn clamping_lies_between_dropping_and_no_restriction() {
        let (samples, threshold, radius) = (samples(), 1.0, 0.0);
        let targets: Vec<Point> = grid().iter().map(|p| (p.0 + 0.5, p.1 + 0.5, 0.0)).collect();
        let all = Search {
            min_samples: 1,
            max_samples: samples.len(),
            ..Default::default()
        };
        let with = |mode| Search {
            high_grade: Some(HighGrade {
                mode,
                ..HighGrade::new(threshold, radius)
            }),
            ..all.clone()
        };
        let checked = |t: &Point, s: &[Sample]| {
            for x in s.iter().filter(|x| x.value > threshold) {
                assert!(variogram::aniso::euclidean(t, &x.loc) <= radius);
            }
            crate::idw::idw(t, s, 2.0)
        };
        let idw = |s: &Search| estimate_many(&targets, None, &samples, s, None, checked);
        let open = estimate_many(&targets, None, &samples, &all, None, |t, s| {
            crate::idw::idw(t, s, 2.0)
        });
        let (dropped, clamped) = (
            idw(&with(HighGradeMode::Drop)),
            idw(&with(HighGradeMode::Clamp)),
        );
        let mut strict = 0;
        for ((d, c), u) in dropped.iter().zip(&clamped).zip(&open) {
            let (d, c, u) = (d.unwrap(), c.unwrap(), u.unwrap());
            assert!(d <= c + 1e-12 && c <= u + 1e-12, "{d} {c} {u}");
            strict += usize::from(d < c && c < u);
        }
        assert!(strict > 0);
        let cv = leave_one_out_many(&samples, &with(HighGradeMode::Clamp), None, checked);
        assert!(cv.iter().all(Option::is_some));
    }

    #[test]
    fn clamping_does_not_depend_on_thread_count() {
        let (targets, s) = (grid(), restricted(HighGradeMode::Clamp, None));
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads);
            let out = pool.build().unwrap().install(|| many(&targets, &s));
            out.iter().map(bits).collect::<Vec<_>>()
        };
        let one = run(1);
        assert_eq!(one, run(8));
        let dropped: Vec<_> = many(&targets, &restricted(HighGradeMode::Drop, None))
            .iter()
            .map(bits)
            .collect();
        assert_ne!(one, dropped);
    }

    fn domain_of(p: &Point) -> u32 {
        match p.0 + 0.3 * p.1 {
            v if v < 45.0 => 0,
            v if v < 80.0 => 1,
            _ => 2,
        }
    }

    fn zoned() -> Vec<Sample> {
        samples()
            .into_iter()
            .map(|s| Sample {
                domain: Some(domain_of(&s.loc)),
                ..s
            })
            .collect()
    }

    fn soft(soft: Soft) -> Search {
        Search {
            soft: Some(soft),
            ..search(2, 30.0)
        }
    }

    fn zoned_many<T: Send>(
        search: &Search,
        estimator: impl Fn(&Point, &[Sample]) -> Result<T> + Sync,
    ) -> Vec<Option<T>> {
        let (targets, vg) = (grid(), Variogram::single(Model::Spherical, 1.0, 40.0));
        let domains: Vec<u32> = targets.iter().map(domain_of).collect();
        estimate_many(
            &targets,
            Some(&domains),
            &zoned(),
            search,
            Some(&vg),
            estimator,
        )
    }

    fn zoned_krige(search: &Search) -> Vec<Option<Estimate>> {
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        zoned_many(search, krige_with(&vg))
    }

    fn zoned_loo(search: &Search) -> Vec<Option<Estimate>> {
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        leave_one_out_many(&zoned(), search, Some(&vg), krige_with(&vg))
    }

    fn same(a: &[Option<Estimate>], b: &[Option<Estimate>]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| bits(a) == bits(b))
    }

    #[test]
    fn hard_domains_are_separate_estimations() {
        let (samples, targets, plain) = (zoned(), grid(), search(2, 30.0));
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let (hard, hard_cv) = (zoned_krige(&plain), zoned_loo(&plain));
        for d in 0..3 {
            let own: Vec<Sample> = samples
                .iter()
                .filter(|s| s.domain == Some(d))
                .cloned()
                .collect();
            let at: Vec<usize> = (0..targets.len())
                .filter(|&i| domain_of(&targets[i]) == d)
                .collect();
            let points: Vec<Point> = at.iter().map(|&i| targets[i]).collect();
            let alone = estimate_many(&points, None, &own, &plain, Some(&vg), krige_with(&vg));
            let joint: Vec<_> = at.iter().map(|&i| hard[i].clone()).collect();
            assert!(same(&alone, &joint));
            let cv = leave_one_out_many(&own, &plain, Some(&vg), krige_with(&vg));
            let joint: Vec<_> = (0..samples.len())
                .filter(|&i| samples[i].domain == Some(d))
                .map(|i| hard_cv[i].clone())
                .collect();
            assert!(same(&cv, &joint));
        }
        assert!(hard.iter().filter(|e| e.is_some()).count() > 300);
    }

    #[test]
    fn a_zero_soft_distance_is_hard() {
        let plain = search(2, 30.0);
        let zero = [
            Soft::All(0.0),
            Soft::Pairs(vec![SoftPair {
                target: 0,
                sample: 1,
                distance: 0.0,
            }]),
        ];
        for z in zero {
            assert!(same(&zoned_krige(&soft(z.clone())), &zoned_krige(&plain)));
            assert!(same(&zoned_loo(&soft(z)), &zoned_loo(&plain)));
        }
    }

    #[test]
    fn an_infinite_soft_distance_pools_the_domains() {
        let pooled = soft(Soft::All(f64::INFINITY));
        let plain = search(2, 30.0);
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let (targets, samples) = (grid(), samples());
        let free = estimate_many(&targets, None, &samples, &plain, Some(&vg), krige_with(&vg));
        assert!(same(&zoned_krige(&pooled), &free));
        let cv = leave_one_out_many(&samples, &plain, Some(&vg), krige_with(&vg));
        assert!(same(&zoned_loo(&pooled), &cv));
    }

    #[test]
    fn other_domain_samples_lie_within_the_soft_distance() {
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let one_way = Soft::Pairs(vec![SoftPair {
            target: 1,
            sample: 0,
            distance: 12.0,
        }]);
        for (rule, limit) in [(Soft::All(8.0), [8.0; 2]), (one_way, [12.0, 0.0])] {
            let used = AtomicUsize::new(0);
            let check = |t: &Point, s: &[Sample]| {
                for x in s.iter().filter(|x| x.domain != Some(domain_of(t))) {
                    let d = variogram::aniso::euclidean(t, &x.loc);
                    let allowed = match (domain_of(t), x.domain) {
                        (1, Some(0)) => limit[0],
                        _ => limit[1],
                    };
                    assert!(d < allowed, "{d} from domain {:?}", x.domain);
                    used.fetch_add(1, Ordering::Relaxed);
                }
                krige(Kind::Ordinary, t, s, &vg)
            };
            zoned_many(&soft(rule.clone()), check);
            leave_one_out_many(&zoned(), &soft(rule), Some(&vg), check);
            assert!(used.into_inner() > 0);
        }
    }

    #[test]
    fn other_domain_samples_used_never_decrease_as_the_soft_distance_grows() {
        let counts = |d: f64| {
            zoned_many(&soft(Soft::All(d)), |t, s| {
                let other = s.iter().filter(|x| x.domain != Some(domain_of(t)));
                Ok((other.count(), s.len()))
            })
        };
        let mut before = counts(0.0);
        for d in [2.0, 5.0, 10.0, 20.0, 40.0, f64::INFINITY] {
            let now = counts(d);
            for (a, b) in before.iter().zip(&now) {
                if let Some(a) = a {
                    let b = b.expect("more samples, still estimated");
                    assert!(b.0 >= a.0 && b.1 >= a.1);
                }
            }
            before = now;
        }
        assert!(before.iter().flatten().any(|c| c.0 > 0));
    }

    #[test]
    fn a_shared_contact_location_keeps_one_sample_of_the_target_domain() {
        let twin = (50.0, 50.0, 0.0);
        let mut samples = zoned();
        for (value, domain) in [(5.0, 1), (-5.0, 0)] {
            samples.push(Sample {
                domain: Some(domain),
                ..Sample::new(twin, value)
            });
        }
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let rule = soft(Soft::All(f64::INFINITY));
        let targets = [(51.0, 50.0, 0.0), (49.0, 51.0, 0.0), twin];
        let seen = AtomicUsize::new(0);
        let domain_at = |t: &Point| if t.0 > 50.0 { 1 } else { 0 };
        let check = |t: &Point, s: &[Sample]| {
            let at: Vec<_> = s.iter().filter(|x| x.loc == twin).collect();
            assert_eq!(at.len(), 1);
            assert_eq!(at[0].domain, Some(domain_at(t)));
            seen.fetch_add(1, Ordering::Relaxed);
            krige(Kind::Ordinary, t, s, &vg)
        };
        let domains: Vec<u32> = targets.iter().map(domain_at).collect();
        let out = estimate_many(&targets, Some(&domains), &samples, &rule, Some(&vg), check);
        assert!(
            out.iter()
                .all(|e| e.as_ref().is_some_and(|e| e.value.is_finite()))
        );
        let local = crate::lva::LocalAnisotropy::new(
            targets.to_vec(),
            vec![[0.0; 3]; 3],
            vec![[1.0; 2]; 3],
        )
        .unwrap();
        let lva = crate::lva::estimate_many_local(
            &targets,
            Some(&domains),
            &local,
            &samples,
            &rule,
            &vg,
            |t, s, v| check(t, s).and_then(|_| krige(Kind::Ordinary, t, s, v)),
        )
        .unwrap();
        assert!(lva.iter().all(Option::is_some));
        let n = samples.len();
        let cv = leave_one_out_at(&[n - 2, n - 1], &samples, &rule, Some(&vg), |t, s| {
            assert_eq!(s.iter().filter(|x| x.loc == twin).count(), 1);
            krige(Kind::Ordinary, t, s, &vg)
        });
        assert!(
            cv.iter()
                .all(|e| e.as_ref().is_some_and(|e| e.value.is_finite()))
        );
        assert_eq!(seen.into_inner(), 6);
        for (t, d) in targets.iter().zip(&domains) {
            let scan = crate::search::neighbors_in(t, Some(*d), &samples, &rule, Some(&vg));
            let tree = SearchTree::new(&samples, &rule, Some(&vg)).neighbors_in(t, Some(*d));
            assert_eq!(scan.unwrap(), tree.unwrap());
        }
    }

    #[test]
    fn soft_boundaries_do_not_depend_on_thread_count() {
        let rule = soft(Soft::All(10.0));
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
                    let all: Vec<usize> = (0..200).collect();
                    let folds = k_fold_at(5, &all, &zoned(), &rule, Some(&vg), krige_with(&vg));
                    (zoned_krige(&rule), zoned_loo(&rule), folds.unwrap())
                })
        };
        let (one, many) = (run(1), run(8));
        assert!(same(&one.0, &many.0) && same(&one.1, &many.1) && same(&one.2, &many.2));
    }

    fn field(p: &Point) -> f64 {
        (p.0 / 15.0).sin() + p.1 / 40.0
    }

    /// A 10 x 10 grid at spacing 10, plus 60 samples clustered where the
    /// field is high.
    fn preferential() -> Vec<Sample> {
        let mut locs: Vec<Point> = (0..100)
            .map(|i| {
                (
                    (i % 10) as f64 * 10.0 + 5.0,
                    (i / 10) as f64 * 10.0 + 5.0,
                    0.0,
                )
            })
            .collect();
        locs.extend((0..60).map(|i| {
            let (a, r) = (i as f64 * 2.4, 1.0 + (i as f64).sqrt() * 1.5);
            (23.0 + r * a.cos(), 85.0 + r * a.sin(), 0.0)
        }));
        locs.iter().map(|p| Sample::new(*p, field(p))).collect()
    }

    fn cover() -> Vec<Point> {
        (0..10_000)
            .map(|i| ((i % 100) as f64 + 0.5, (i / 100) as f64 + 0.5, 0.0))
            .collect()
    }

    fn declustered(samples: &[Sample], ordinary: bool) -> transforms::Weights {
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let search = Search {
            min_samples: 1,
            max_samples: 16,
            radius: 50.0,
            ..Default::default()
        };
        let opts = crate::InterpOptions::default();
        let per = weights_many(&cover(), samples, &search, Some(&vg), |t, s| {
            if ordinary {
                krige(Kind::Ordinary, t, s, &vg).map(|e| e.weights)
            } else {
                crate::inverse_distance_weights(t, s, 2.0, &opts)
            }
        });
        let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
        weight_declustering(&per, &values).unwrap()
    }

    #[test]
    fn weight_declustering_is_uniform_on_a_regular_grid() {
        let grid: Vec<Sample> = preferential().into_iter().take(100).collect();
        for ordinary in [false, true] {
            let w = declustered(&grid, ordinary).weights;
            assert!((w.iter().sum::<f64>() - 100.0).abs() < 1e-9);
            assert!(w.iter().all(|w| (w - 1.0).abs() < 0.15), "{w:?}");
        }
    }

    #[test]
    fn weight_declustering_lowers_clustered_samples_and_the_bias() {
        let samples = preferential();
        let truth = cover().iter().map(field).sum::<f64>() / 10_000.0;
        let naive = samples.iter().map(|s| s.value).sum::<f64>() / samples.len() as f64;
        for ordinary in [false, true] {
            let w = declustered(&samples, ordinary);
            assert!((w.weights.iter().sum::<f64>() - 160.0).abs() < 1e-9);
            let (grid, cluster) = w.weights.split_at(100);
            let mean = |w: &[f64]| w.iter().sum::<f64>() / w.len() as f64;
            assert!(mean(cluster) < 0.5 * mean(grid));
            assert!((w.declustered_mean - truth).abs() < 0.25 * (naive - truth).abs());
        }
    }

    #[test]
    fn weight_declustering_does_not_depend_on_thread_count() {
        let samples = preferential();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| declustered(&samples, true).weights)
        };
        let (one, many) = (run(1), run(8));
        assert!(
            one.iter()
                .zip(&many)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(weight_declustering(&[None], &[1.0]).is_err());
    }
}
