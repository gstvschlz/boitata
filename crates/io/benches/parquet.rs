use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, RecordBatch};
use ceres_core::{BlockModel, Geometry};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn noise(i: u64) -> f32 {
    let mut x = i.wrapping_mul(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

/// 2 000 000 cells with 15 float32 grades, as a simulation writes them.
fn bench(c: &mut Criterion) {
    let count = [200usize, 200, 50];
    let n = count.iter().product::<usize>();
    let columns: Vec<(String, ArrayRef)> = (0..15u64)
        .map(|j| {
            let values = (0..n as u64).map(|i| 2.0 + (i as f32 / 900.0).sin() + noise(i * 16 + j));
            (
                format!("v{j}"),
                Arc::new(Float32Array::from_iter_values(values)) as ArrayRef,
            )
        })
        .collect();
    let geometry = Geometry {
        origin: [0.0; 3],
        size: [5.0; 3],
        count,
        rotation: [0.0; 3],
    };
    let model =
        BlockModel::regular(geometry, RecordBatch::try_from_iter(columns).unwrap()).unwrap();
    let path = std::env::temp_dir().join(format!("ceres-bench-{}.parquet", std::process::id()));
    let mut group = c.benchmark_group("parquet");
    group.sample_size(10);
    group.throughput(Throughput::Elements((n * 15) as u64));
    group.bench_function("block model, 2M cells x 15 f32", |b| {
        b.iter(|| ceres_io::write_block_model(&path, &model).unwrap())
    });
    group.finish();
    println!(
        "file size {} MB",
        std::fs::metadata(&path).unwrap().len() / 1_000_000
    );
    std::fs::remove_file(path).unwrap();
}

criterion_group!(benches, bench);
criterion_main!(benches);
