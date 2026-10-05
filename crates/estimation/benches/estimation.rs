use criterion::{Criterion, criterion_group, criterion_main};
use estimation::Sample;
use estimation::search::{Search, SearchTree, neighbors};
use std::hint::black_box;

fn cloud(n: usize, seed: u64) -> Vec<(f64, f64, f64)> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| (next() * 2000.0, next() * 2000.0, next() * 400.0))
        .collect()
}

fn bench(c: &mut Criterion) {
    let samples: Vec<Sample> = cloud(20_000, 1)
        .into_iter()
        .map(|p| Sample::new(p, 0.0))
        .collect();
    let targets = cloud(200, 2);
    let search = Search {
        min_samples: 1,
        max_samples: 24,
        radius: 150.0,
        ..Default::default()
    };
    let tree = SearchTree::new(&samples, &search, None);
    let mut group = c.benchmark_group("200 queries in 20 000 samples");
    group.bench_function("scan", |b| {
        b.iter(|| {
            for t in &targets {
                black_box(neighbors(t, &samples, &search, None).ok());
            }
        })
    });
    group.bench_function("tree", |b| {
        b.iter(|| {
            for t in &targets {
                black_box(tree.neighbors(t).ok());
            }
        })
    });
    group.finish();
}

/// Sequential simulation's pattern on one level of a grid: query a node, then add it.
fn growth(c: &mut Criterion) {
    let samples: Vec<Sample> = cloud(300, 5)
        .into_iter()
        .map(|p| Sample::new(p, 0.0))
        .collect();
    let level: Vec<Sample> = (0..10_000)
        .map(|i| {
            let j = i * 7919 % 10_000;
            Sample::new(
                ((j % 100) as f64 * 20.0, (j / 100) as f64 * 20.0, 200.0),
                0.0,
            )
        })
        .collect();
    let search = Search {
        min_samples: 1,
        max_samples: 16,
        radius: 300.0,
        ..Default::default()
    };
    let mut group = c.benchmark_group("10 000 nodes on one level among 300 samples");
    group.sample_size(10);
    group.bench_function("query and add", |b| {
        b.iter(|| {
            let mut tree = SearchTree::new(&samples, &search, None);
            for s in &level {
                black_box(tree.neighbors(&s.loc).ok());
                tree.add(s);
            }
        })
    });
    group.finish();
}

fn kriging(c: &mut Criterion) {
    let samples: Vec<Sample> = cloud(5_000, 3)
        .into_iter()
        .map(|p| Sample::new(p, (p.0 / 100.0).sin() + p.2 / 400.0))
        .collect();
    let targets = cloud(10_000, 4);
    let search = Search {
        min_samples: 4,
        max_samples: 24,
        radius: 300.0,
        ..Default::default()
    };
    let vg = variogram::Variogram::single(variogram::Model::Spherical, 1.0, 300.0);
    c.bench_function(
        "ordinary kriging of 10 000 targets from 5 000 samples",
        |b| {
            b.iter(|| {
                black_box(estimation::estimate_many(
                    &targets,
                    None,
                    &samples,
                    &search,
                    Some(&vg),
                    |t, s| estimation::krige(estimation::Kind::Ordinary, t, s, &vg),
                ))
            })
        },
    );
}

/// The solve alone: 20 000 ordinary-kriging systems of 32 samples, one thread.
fn systems(c: &mut Criterion) {
    let samples: Vec<Sample> = cloud(5_000, 3)
        .into_iter()
        .map(|p| Sample::new(p, p.0 / 1000.0))
        .collect();
    let targets = cloud(20_000, 4);
    let search = Search {
        min_samples: 32,
        max_samples: 32,
        radius: f64::INFINITY,
        ..Default::default()
    };
    let vg = variogram::Variogram::single(variogram::Model::Spherical, 1.0, 300.0);
    let tree = SearchTree::new(&samples, &search, Some(&vg));
    let sets: Vec<Vec<Sample>> = targets
        .iter()
        .map(|t| tree.take(t, None, &tree.neighbors(t).unwrap(), &samples))
        .collect();
    c.bench_function("20 000 ordinary-kriging systems of 32 samples", |b| {
        b.iter(|| {
            for (t, s) in targets.iter().zip(&sets) {
                black_box(estimation::krige(estimation::Kind::Ordinary, t, s, &vg).ok());
            }
        })
    });
}

/// 40 x 40 holes 30 m apart, 100 composites of 4 m each, slightly inclined.
fn drill_holes() -> Vec<Sample> {
    let mut state = 7u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = vec![];
    for h in 0..1600u32 {
        let (x0, y0) = ((h % 40) as f64 * 30.0, (h / 40) as f64 * 30.0);
        let (dx, dy) = (next() - 0.5, next() - 0.5);
        for k in 0..100 {
            let d = k as f64 * 4.0 + 2.0;
            let loc = (x0 + dx * 0.3 * d, y0 + dy * 0.3 * d, 400.0 - d);
            out.push(Sample::with_hole(loc, next(), h));
        }
    }
    out
}

/// Block centroids 5 m apart, as a block model queries them.
fn constrained(c: &mut Criterion) {
    let samples = drill_holes();
    let targets: Vec<(f64, f64, f64)> = (0..20_000)
        .map(|i| {
            (
                300.0 + (i % 40) as f64 * 5.0,
                300.0 + (i / 40 % 25) as f64 * 5.0,
                50.0 + (i / 1000) as f64 * 5.0,
            )
        })
        .collect();
    let base = Search {
        min_samples: 4,
        max_samples: 32,
        radius: 200.0,
        ..Default::default()
    };
    let cases = [
        ("plain", base.clone()),
        (
            "max_per_hole 4",
            Search {
                max_per_hole: Some(4),
                ..base.clone()
            },
        ),
        (
            "octant + max_per_hole 4",
            Search {
                max_per_hole: Some(4),
                octant: true,
                ..base.clone()
            },
        ),
    ];
    let mut group = c.benchmark_group("20 000 queries among drill holes");
    group.sample_size(10);
    for (label, search) in cases {
        let tree = SearchTree::new(&samples, &search, None);
        group.bench_function(label, |b| {
            b.iter(|| {
                for t in &targets {
                    black_box(tree.neighbors(t).ok());
                }
            })
        });
    }
    group.finish();
}

/// 10 000 blocks of 10 m, 2 025 candidate holes 11 m apart, 100 drilled.
fn plan(c: &mut Criterion) {
    use estimation::plan::{Candidate, Constraints, Kriging, Objective, Plan};
    let hole = |x: f64, y: f64| -> Vec<(f64, f64, f64)> {
        (0..8).map(|k| (x, y, -2.5 - 5.0 * k as f64)).collect()
    };
    let data: Vec<Sample> = (0..100u32)
        .flat_map(|h| {
            let (x, y) = ((h % 10) as f64 * 50.0 + 20.0, (h / 10) as f64 * 50.0 + 20.0);
            hole(x, y)
                .into_iter()
                .map(move |p| Sample::with_hole(p, 0.0, h))
        })
        .collect();
    let candidates: Vec<Candidate> = (0..2025)
        .map(|i| {
            let (x, y) = ((i % 45) as f64 * 11.0 + 6.0, (i / 45) as f64 * 11.0 + 6.0);
            Candidate {
                composites: hole(x, y),
                collar: (x, y, 0.0),
                cost: 40.0,
                excluded: false,
            }
        })
        .collect();
    let targets: Vec<(f64, f64, f64)> = (0..10_000)
        .map(|i| {
            let (x, y, z) = (i % 50, i / 50 % 50, i / 2500);
            (
                5.0 + 10.0 * x as f64,
                5.0 + 10.0 * y as f64,
                -5.0 - 10.0 * z as f64,
            )
        })
        .collect();
    let kriging = Kriging {
        kind: estimation::Kind::Ordinary,
        block: Some((
            (10.0, 10.0, 10.0),
            estimation::Discretization {
                nx: 2,
                ny: 2,
                nz: 1,
            },
        )),
        variogram: variogram::Variogram::single(variogram::Model::Spherical, 1.0, 80.0),
        passes: vec![Search {
            min_samples: 4,
            max_samples: 16,
            radius: 40.0,
            ..Default::default()
        }],
    };
    let new = || {
        Plan::new(
            kriging.clone(),
            data.clone(),
            candidates.clone(),
            targets.clone(),
            None,
            Objective::Variance,
            Constraints::default(),
        )
        .unwrap()
    };
    let mut group = c.benchmark_group("plan of 2 025 holes over 10 000 blocks");
    group.sample_size(10);
    group.bench_function("gains", |b| {
        b.iter_batched(
            new,
            |mut p| black_box(p.gains(&[], None).unwrap()),
            criterion::BatchSize::LargeInput,
        )
    });
    group.bench_function("gains after one more hole", |b| {
        b.iter_batched(
            || {
                let mut p = new();
                p.gains(&[], None).unwrap();
                p
            },
            |mut p| black_box(p.gains(&[1000], None).unwrap()),
            criterion::BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench, growth, kriging, systems, constrained, plan);
criterion_main!(benches);
