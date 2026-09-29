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

criterion_group!(benches, bench, growth, kriging, systems);
criterion_main!(benches);
