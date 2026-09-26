use criterion::{Criterion, criterion_group, criterion_main};
use estimation::Search;
use rand::SeedableRng;
use rand::rngs::StdRng;
use simulation::{
    Bands, SgsParams, TurningBandsEnsemble, TurningBandsParams, bounds, sgs, turning_bands,
};
use std::hint::black_box;
use variogram::{Model, Variogram};

fn inputs() -> (Vec<(f64, f64, f64)>, Vec<f64>, Vec<(f64, f64, f64)>) {
    let data: Vec<_> = (0..500)
        .map(|i| {
            (
                ((i * 37) % 211) as f64 * 0.95 + 0.3,
                ((i * 53) % 197) as f64 + 0.7,
                0.0,
            )
        })
        .collect();
    let values = data
        .iter()
        .map(|p| (p.0 / 30.0).sin() + p.1 / 200.0)
        .collect();
    let grid = (0..10_000)
        .map(|i| ((i % 100) as f64 * 2.0, (i / 100) as f64 * 2.0, 0.0))
        .collect();
    (data, values, grid)
}

fn bench(c: &mut Criterion) {
    let (data, values, grid) = inputs();
    let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
    let mut group = c.benchmark_group("one realization of 10 000 nodes");
    group.sample_size(10);
    group.bench_function("SGS", |b| {
        let params = SgsParams {
            search: vec![Search {
                min_samples: 1,
                max_samples: 24,
                radius: 60.0,
                ..Default::default()
            }],
            seed: 1,
        };
        b.iter(|| black_box(sgs(&data, &values, None, None, &grid, &vg, &params, None).unwrap()))
    });
    group.bench_function("turning bands", |b| {
        let params = TurningBandsParams::default();
        b.iter(|| {
            black_box(turning_bands(&data, &values, None, None, &grid, &vg, &params).unwrap())
        })
    });
    group.finish();
}

/// One SGS realization of 100 000 nodes on 1 and 8 threads.
fn threads(c: &mut Criterion) {
    let (data, values, _) = inputs();
    let grid: Vec<_> = (0..100_000)
        .map(|i| ((i % 400) as f64 * 0.5, (i / 400) as f64 * 0.8, 0.0))
        .collect();
    let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
    let params = SgsParams {
        search: vec![Search {
            min_samples: 1,
            max_samples: 24,
            radius: 60.0,
            ..Default::default()
        }],
        seed: 1,
    };
    let mut group = c.benchmark_group("SGS, 100 000 nodes");
    group.sample_size(10);
    for n in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        group.bench_function(format!("{n} threads"), |b| {
            b.iter(|| {
                pool.install(|| {
                    black_box(sgs(&data, &values, None, None, &grid, &vg, &params, None).unwrap())
                })
            })
        });
    }
    group.finish();
}

/// Turning bands on a million nodes, split into its phases.
fn phases(c: &mut Criterion) {
    let (data, values, _) = inputs();
    let grid: Vec<_> = (0..1_000_000)
        .map(|i| ((i % 1000) as f64 * 0.2, (i / 1000) as f64 * 0.2, 0.0))
        .collect();
    let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
    let params = TurningBandsParams::default();
    let (lo, hi) = bounds(&grid);
    let mut group = c.benchmark_group("turning bands, 1 000 000 nodes");
    group.sample_size(10);
    group.bench_function("simulate bands", |b| {
        b.iter(|| {
            black_box(Bands::new(
                lo,
                hi,
                &vg,
                &params,
                &mut StdRng::seed_from_u64(1),
            ))
        })
    });
    let bands = Bands::new(lo, hi, &vg, &params, &mut StdRng::seed_from_u64(1));
    group.bench_function("evaluate field", |b| {
        b.iter(|| black_box(bands.field(&grid)))
    });
    let ensemble = TurningBandsEnsemble::new(
        &data, &values, None, None, None, None, lo, hi, &vg, &params, 1,
    )
    .unwrap();
    group.bench_function("field and conditioning", |b| {
        b.iter(|| black_box(ensemble.realization(0, &grid, None, None).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, bench, threads, phases);
criterion_main!(benches);
