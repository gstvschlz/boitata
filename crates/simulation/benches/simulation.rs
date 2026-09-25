use criterion::{Criterion, criterion_group, criterion_main};
use estimation::Search;
use simulation::{SgsParams, TurningBandsParams, sgs, turning_bands};
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
            search: Search {
                min_samples: 1,
                max_samples: 24,
                radius: 60.0,
                ..Default::default()
            },
            seed: 1,
        };
        b.iter(|| black_box(sgs(&data, &values, None, &grid, &vg, &params, None).unwrap()))
    });
    group.bench_function("turning bands", |b| {
        let params = TurningBandsParams::default();
        b.iter(|| black_box(turning_bands(&data, &values, None, &grid, &vg, &params).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
