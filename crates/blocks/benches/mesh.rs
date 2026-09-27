use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use blocks::{Domain, Region, SolidTester, TriangleTree, convex_hull, distance_to, subblock};
use ceres_core::{BlockModel, Geometry, Mesh};
use criterion::{Criterion, criterion_group, criterion_main};

/// Closed lens: the hull of `n` points on an ellipsoid centered at height `z`.
fn lens(n: usize, radii: [f64; 3], z: f64) -> Mesh {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let points: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let w = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - w * w).sqrt();
            let t = golden * i as f64;
            [
                radii[0] * r * t.cos(),
                radii[1] * r * t.sin(),
                z + radii[2] * w,
            ]
        })
        .collect();
    convex_hull(&points).unwrap()
}

fn grid(count: [usize; 3], rotation: [f64; 3]) -> BlockModel {
    let g = Geometry {
        origin: [-100.0, -60.0, -40.0],
        size: [
            200.0 / count[0] as f64,
            120.0 / count[1] as f64,
            80.0 / count[2] as f64,
        ],
        count,
        rotation,
    };
    let value = Float64Array::from(vec![0.0; g.cells() as usize]);
    let attributes = RecordBatch::try_from_iter([("v", Arc::new(value) as ArrayRef)]).unwrap();
    BlockModel::regular(g, attributes).unwrap()
}

fn bench(c: &mut Criterion) {
    let domains: Vec<Domain> = [-20.0, 0.0, 20.0]
        .iter()
        .enumerate()
        .map(|(i, &z)| Domain {
            region: Region::Inside(SolidTester::new(&lens(1000, [90.0, 50.0, 8.0], z)).unwrap()),
            label: format!("lens{i}"),
        })
        .collect();
    let mut group = c.benchmark_group("3 lenses of 2000 triangles, 20 000 parents");
    group.sample_size(10);
    for (name, rotation) in [("flat", [0.0; 3]), ("rotated", [30.0, 20.0, 10.0])] {
        let model = grid([40, 25, 20], rotation);
        for n in [2u32, 4] {
            group.bench_function(format!("{name} {n}^3"), |b| {
                b.iter(|| black_box(subblock(&model, &domains, [n; 3], "domain", Some("waste"))))
            });
        }
    }
    group.finish();
}

fn distance(c: &mut Criterion) {
    let mesh = lens(1000, [90.0, 50.0, 8.0], 0.0);
    let points: Vec<[f64; 3]> = (0..2000)
        .map(|i| {
            let t = i as f64;
            [
                (t * 0.618).fract() * 200.0 - 100.0,
                (t * 0.414).fract() * 120.0 - 60.0,
                (t * 0.732).fract() * 80.0 - 40.0,
            ]
        })
        .collect();
    let tree = TriangleTree::new(&mesh).unwrap();
    let mut group = c.benchmark_group("distance of 2000 points to 2000 triangles");
    group.bench_function("scan", |b| {
        b.iter(|| {
            for p in &points {
                black_box(distance_to(&mesh, &(p[0], p[1], p[2])).unwrap());
            }
        })
    });
    group.bench_function("tree", |b| {
        b.iter(|| {
            for p in &points {
                black_box(tree.distance(*p));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench, distance);
criterion_main!(benches);
