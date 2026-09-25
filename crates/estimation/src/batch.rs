//! Estimation over many targets.

use rayon::prelude::*;
use variogram::Variogram;

use crate::krige::Estimate;
use crate::search::{Search, neighbors};
use crate::{Result, Sample};

/// Selects the neighbours of every target and applies `estimator` to them, in
/// parallel. A target with too few neighbours, or whose system fails, gives
/// `None`. The output follows `targets` and does not depend on the number of
/// threads.
pub fn estimate_many<F>(
    targets: &[(f64, f64, f64)],
    samples: &[Sample],
    search: &Search,
    vg: Option<&Variogram>,
    estimator: F,
) -> Vec<Option<Estimate>>
where
    F: Fn(&(f64, f64, f64), &[Sample]) -> Result<Estimate> + Sync,
{
    targets
        .par_iter()
        .map(|target| {
            let chosen = neighbors(target, samples, search, vg).ok()?;
            let selected: Vec<Sample> = chosen.iter().map(|&i| samples[i].clone()).collect();
            estimator(target, &selected).ok()
        })
        .collect()
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
    let wider = Search {
        max_samples: search.max_samples + 1,
        ..search.clone()
    };
    (0..samples.len())
        .into_par_iter()
        .map(|i| {
            let target = &samples[i].loc;
            let mut chosen = neighbors(target, samples, &wider, vg).unwrap_or_default();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::{Kind, krige};
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
    fn kriging_honours_the_data() {
        let samples = samples();
        let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
        let search = Search {
            min_samples: 1,
            max_samples: 12,
            radius: 30.0,
            max_per_hole: None,
            octant: false,
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
}
