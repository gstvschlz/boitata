//! Estimation over many targets.

use rayon::prelude::*;
use variogram::Variogram;

use crate::error::EstimError;
use crate::krige::Estimate;
use crate::search::{Search, SearchTree};
use crate::{Result, Sample};

/// Selects the neighbours of every target and applies `estimator` to them, in
/// parallel. A target with too few neighbours, or whose system fails, gives
/// `None`. The output follows `targets` and does not depend on the number of
/// threads.
pub fn estimate_many<F, T>(
    targets: &[(f64, f64, f64)],
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Vec<Option<T>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<T> + Sync,
    T: Send,
{
    let tree = SearchTree::new(samples, search, vg);
    targets
        .par_iter()
        .map(|target| {
            let chosen = tree.neighbors(target).ok()?;
            let selected: Vec<Sample> = chosen.iter().map(|&i| samples[i].clone()).collect();
            estimator(target, &selected).ok()
        })
        .collect()
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

/// Estimates every sample from its neighbours with the sample itself left out.
pub fn leave_one_out_many<F>(
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Vec<Option<Estimate>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<Estimate> + Sync,
{
    let all: Vec<usize> = (0..samples.len()).collect();
    leave_one_out_at(&all, samples, search, vg, estimator)
}

/// As [`leave_one_out_many`] for the samples at indices `which` only.
pub fn leave_one_out_at<F>(
    which: &[usize],
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Vec<Option<Estimate>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<Estimate> + Sync,
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
            let mut chosen = tree.neighbors(target).unwrap_or_default();
            chosen.retain(|&j| j != i);
            chosen.truncate(search.max_samples);
            if chosen.len() < search.min_samples.max(1) {
                return None;
            }
            let selected: Vec<Sample> = chosen.iter().map(|&j| samples[j].clone()).collect();
            estimator(target, &selected).ok()
        })
        .collect()
}

/// K-fold cross-validation: estimates the samples at indices `which` from
/// the samples outside their fold, sample `i` being in fold `i % k`. With
/// `k` equal to the number of samples this is leave-one-out.
pub fn k_fold_at<F>(
    k: usize,
    which: &[usize],
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Result<Vec<Option<Estimate>>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<Estimate> + Sync,
{
    if k < 2 {
        return Err(EstimError::InvalidParameters("k must be ≥ 2".into()));
    }
    let mut out = vec![None; which.len()];
    for fold in 0..k {
        let (test, at): (Vec<usize>, Vec<_>) = which
            .iter()
            .enumerate()
            .filter(|(_, i)| *i % k == fold)
            .map(|(p, &i)| (p, samples[i].loc))
            .unzip();
        if test.is_empty() {
            continue;
        }
        let train: Vec<Sample> = (0..samples.len())
            .filter(|i| i % k != fold)
            .map(|i| samples[i].clone())
            .collect();
        for (p, e) in test
            .into_iter()
            .zip(estimate_many(&at, &train, search, vg, &estimator))
        {
            out[p] = e;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::{Kind, krige};
    use crate::search::HighGrade;
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
        };
        let targets: Vec<_> = (0..400)
            .map(|i| ((i % 20) as f64 * 5.0, (i / 20) as f64 * 5.0, 0.0))
            .collect();
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                estimate_many(&targets, &samples, &search, Some(&vg), |t, s| {
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
    fn kriging_honours_the_data() {
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
        };
        let targets: Vec<_> = samples.iter().map(|s| s.loc).collect();
        let out = estimate_many(&targets, &samples, &search, Some(&vg), |t, s| {
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
        estimate_many(targets, &samples(), search, Some(&vg), krige_with(&vg))
    }

    fn passes(targets: &[Point], searches: &[Search]) -> Vec<Option<(usize, Estimate)>> {
        let (samples, vg) = (samples(), Variogram::single(Model::Spherical, 1.0, 40.0));
        by_pass(targets.len(), searches, |search, remaining| {
            let at: Vec<Point> = remaining.iter().map(|&i| targets[i]).collect();
            Ok(estimate_many(
                &at,
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

    #[test]
    fn a_high_grade_radius_beyond_the_search_changes_nothing() {
        let (samples, targets, plain) = (samples(), grid(), search(4, 30.0));
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let loo = |s: &Search| leave_one_out_many(&samples, s, Some(&vg), krige_with(&vg));
        for radius in [30.0, 100.0] {
            let restricted = Search {
                high_grade: Some(HighGrade {
                    threshold: 1.0,
                    radius,
                }),
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
            high_grade: Some(HighGrade { threshold, radius }),
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
        let out = estimate_many(&targets, &samples, &restricted, Some(&vg), check);
        let cv = leave_one_out_many(&samples, &restricted, Some(&vg), check);
        assert!(used.into_inner() > 0);
        let changed = |a: &[Option<Estimate>], b: &[Option<Estimate>]| {
            a.iter().zip(b).any(|(a, b)| bits(a) != bits(b))
        };
        assert!(changed(&out, &many(&targets, &plain)));
        let plain_cv = leave_one_out_many(&samples, &plain, Some(&vg), krige_with(&vg));
        assert!(changed(&cv, &plain_cv));
    }
}
