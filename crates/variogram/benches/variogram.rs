use boitata_core::Geometry;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use variogram::{Direction, Estimator, LagBins, Support, experimental};

fn bench(c: &mut Criterion) {
    let geometry = Geometry {
        origin: [0.0; 3],
        size: [1.0; 3],
        count: [100, 100, 1],
        rotation: [0.0; 3],
    };
    let cells: Vec<u64> = (0..geometry.cells()).collect();
    let centers: Vec<(f64, f64, f64)> = cells
        .iter()
        .map(|&c| {
            let [x, y, z] = geometry.centroid(c);
            (x, y, z)
        })
        .collect();
    let values: Vec<f64> = centers
        .iter()
        .map(|p| (p.0 / 7.0).sin() + (p.1 / 11.0).cos())
        .collect();
    let bins = LagBins {
        max_lag: 25.0,
        lag_width: 1.0,
    };
    let east = Direction {
        azimuth: 90.0,
        dip: 0.0,
        tolerance: 5.0,
        bandwidth: None,
    };
    for (name, direction) in [("omnidirectional", None), ("along x", Some(&east))] {
        let mut group = c.benchmark_group(format!("variogram of a 100 x 100 grid, {name}"));
        group.sample_size(10);
        let m = Estimator::Matheron;
        group.bench_function("pairs", |b| {
            b.iter(|| experimental(black_box(&centers), &values, &bins, m, direction, false))
        });
        group.bench_function("grid", |b| {
            let on = Support::Grid(&geometry, &cells);
            b.iter(|| experimental(black_box(on), &values, &bins, m, direction, false))
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
