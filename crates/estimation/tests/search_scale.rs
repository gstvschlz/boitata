//! Searches among 504 100 composites of 5041 drill holes checked against
//! brute force at lattice targets, where exact distance ties are common.
//! Run: mise exec -- cargo test --release -p estimation --test search_scale -- --ignored

use estimation::Sample;
use estimation::search::{Search, SearchTree};
use rayon::prelude::*;

fn hash(mut x: u64) -> f64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

fn composites() -> Vec<Sample> {
    let mut out = vec![];
    for h in 0..5041u64 {
        let (x0, y0) = ((h % 71) as f64 * 30.0, (h / 71) as f64 * 30.0);
        let (dx, dy) = (hash(h) - 0.5, hash(h + 9999) - 0.5);
        for k in 0..100 {
            let d = k as f64 * 4.0 + 2.0;
            out.push(Sample::with_hole(
                (x0 + dx * 0.3 * d, y0 + dy * 0.3 * d, 400.0 - d),
                hash(h * 1000 + k),
                h as u32,
            ));
        }
    }
    out
}

/// Every sample by (squared distance, index) through the greedy rules of an
/// isotropic search.
fn brute(t: &(f64, f64, f64), samples: &[Sample], s: &Search) -> Option<Vec<usize>> {
    let d2 = |p: &(f64, f64, f64)| (p.0 - t.0).powi(2) + (p.1 - t.1).powi(2) + (p.2 - t.2).powi(2);
    let mut all: Vec<(f64, usize)> = samples
        .iter()
        .enumerate()
        .map(|(i, x)| (d2(&x.loc), i))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let (mut chosen, mut per_hole, mut per_octant) =
        (vec![], std::collections::HashMap::new(), [0usize; 8]);
    let cap = if s.octant {
        s.max_samples.div_ceil(8)
    } else {
        usize::MAX
    };
    for (d, i) in all {
        if d > s.radius * s.radius || chosen.len() >= s.max_samples {
            break;
        }
        let h = samples[i].hole.unwrap();
        if s.max_per_hole
            .is_some_and(|m| per_hole.get(&h).copied().unwrap_or(0) >= m)
        {
            continue;
        }
        let p = samples[i].loc;
        let o =
            (((p.0 >= t.0) as usize) << 2) | (((p.1 >= t.1) as usize) << 1) | (p.2 >= t.2) as usize;
        if s.octant {
            if per_octant[o] >= cap {
                continue;
            }
            per_octant[o] += 1;
        }
        *per_hole.entry(h).or_insert(0) += 1;
        chosen.push(i);
    }
    (chosen.len() >= s.min_samples).then_some(chosen)
}

#[test]
#[ignore = "about a minute in release"]
fn lattice_targets_among_drill_holes_match_brute_force() {
    let samples = composites();
    let base = Search {
        min_samples: 4,
        max_samples: 32,
        radius: 200.0,
        ..Default::default()
    };
    let cases = [
        base.clone(),
        Search {
            max_per_hole: Some(4),
            ..base.clone()
        },
        Search {
            max_per_hole: Some(4),
            octant: true,
            ..base.clone()
        },
    ];
    // Every 53rd of a 200 x 200 x 25 lattice of 5 m blocks, including the
    // three targets where the k-d tree dropped a neighbor.
    let mut targets: Vec<(f64, f64, f64)> = (0..1_000_000usize)
        .step_by(53)
        .map(|i| {
            (
                300.0 + (i % 200) as f64 * 5.0,
                300.0 + (i / 200 % 200) as f64 * 5.0,
                50.0 + (i / 40_000) as f64 * 5.0,
            )
        })
        .collect();
    targets.extend([
        (300.0, 1255.0, 205.0),
        (590.0, 820.0, 215.0),
        (575.0, 830.0, 210.0),
    ]);
    for search in &cases {
        let tree = SearchTree::new(&samples, search, None);
        let bad: Vec<_> = targets
            .par_iter()
            .filter(|t| tree.neighbors(t).ok() != brute(t, &samples, search))
            .collect();
        assert!(
            bad.is_empty(),
            "{search:?}: {} of {} differ, first {:?}",
            bad.len(),
            targets.len(),
            bad[0]
        );
    }
}
