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
                    &samples,
                    &search,
                    Some(&vg),
                    |t, s| estimation::krige(estimation::Kind::Ordinary, t, s, &vg),
                ))
            })
        },
    );
}

criterion_group!(benches, bench, kriging);
criterion_main!(benches);
