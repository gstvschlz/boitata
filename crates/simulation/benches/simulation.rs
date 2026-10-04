use criterion::{Criterion, criterion_group, criterion_main};
use estimation::Search;
use rand::SeedableRng;
use rand::rngs::StdRng;
use simulation::{
    Bands, SgsParams, TurningBandsEnsemble, TurningBandsParams, bounds, dss_in, sgs, turning_bands,
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
    group.bench_function("DSS", |b| {
        let params = SgsParams {
            search: vec![Search {
                min_samples: 1,
                max_samples: 24,
                radius: 60.0,
                ..Default::default()
            }],
            seed: 1,
        };
        b.iter(|| {
            black_box(dss_in(&data, &values, None, None, None, &grid, &vg, &params, None).unwrap())
        })
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

/// Turning bands over a 20 km plateau: 3000 data, range 300 m, 30
/// realizations, summarized at 20 000 nodes.
fn wide(c: &mut Criterion) {
    let data: Vec<_> = (0..3000)
        .map(|i| {
            (
                ((i * 7919) % 20_000) as f64,
                ((i * 104_729) % 8000) as f64,
                ((i * 31) % 200) as f64,
            )
        })
        .collect();
    let values: Vec<f64> = data
        .iter()
        .map(|p| (p.0 / 900.0).sin() + p.1 / 8000.0 + 2.0)
        .collect();
    let grid: Vec<_> = (0..20_000)
        .map(|i| ((i % 200) as f64 * 100.0, (i / 200) as f64 * 80.0, 100.0))
        .collect();
    let vg = Variogram::single(Model::Spherical, 1.0, 300.0);
    let params = TurningBandsParams::default();
    let (lo, hi) = bounds(&grid);
    let mut group = c.benchmark_group("turning bands, 20 km extent");
    group.sample_size(10);
    group.bench_function("30 realizations", |b| {
        b.iter(|| {
            let e = TurningBandsEnsemble::new(
                &data, &values, None, None, None, None, lo, hi, &vg, &params, 30,
            )
            .unwrap();
            black_box(
                e.summary(&grid, None, None, &Default::default(), None)
                    .unwrap(),
            )
        })
    });
    group.finish();
}

/// 50 realizations of 100 000 nodes from 5000 data: conditioning shared by
/// the realizations.
fn many(c: &mut Criterion) {
    let data: Vec<_> = (0..5000)
        .map(|i| {
            (
                ((i * 7919) % 2003) as f64,
                ((i * 104_729) % 1999) as f64,
                ((i * 31) % 211) as f64,
            )
        })
        .collect();
    let values: Vec<f64> = data
        .iter()
        .map(|p| (p.0 / 300.0).sin() + p.1 / 2000.0)
        .collect();
    let grid: Vec<_> = (0..100_000)
        .map(|i| {
            (
                (i % 100) as f64 * 20.0,
                ((i / 100) % 100) as f64 * 20.0,
                (i / 10_000) as f64 * 20.0,
            )
        })
        .collect();
    let vg = Variogram::single(Model::Spherical, 1.0, 200.0);
    let params = TurningBandsParams::default();
    let (lo, hi) = bounds(&grid);
    let e = TurningBandsEnsemble::new(
        &data, &values, None, None, None, None, lo, hi, &vg, &params, 50,
    )
    .unwrap();
    let mut group = c.benchmark_group("turning bands, 50 realizations of 100 000 nodes");
    group.sample_size(10);
    group.bench_function("summary", |b| {
        b.iter(|| {
            black_box(
                e.summary(&grid, None, None, &Default::default(), None)
                    .unwrap(),
            )
        })
    });
    group.finish();
}

/// SGS on 1 000 000 lattice nodes: 16 realizations along a shared path
/// against one along a random path.
fn shared(c: &mut Criterion) {
    let (data, values, _) = inputs();
    let geometry = boitata_core::Geometry {
        origin: [0.0; 3],
        size: [0.2, 0.2, 1.0],
        count: [1000, 1000, 1],
        rotation: [0.0; 3],
    };
    let lattice = simulation::Lattice::regular(geometry);
    let grid: Vec<_> = (0..lattice.len()).map(|m| lattice.location(m)).collect();
    let vg = Variogram::single(Model::Spherical, 1.0, 40.0);
    let search = vec![Search {
        min_samples: 1,
        max_samples: 24,
        radius: 60.0,
        ..Default::default()
    }];
    let mut group = c.benchmark_group("SGS, 1 000 000 nodes");
    group.sample_size(10);
    group.bench_function("shared path, 16 realizations", |b| {
        let shared = simulation::SharedSgs {
            lattice: &lattice,
            search: &search,
            levels: None,
            seed: 1,
        };
        b.iter(|| {
            black_box(
                simulation::sgs_shared(
                    &data,
                    &values,
                    None,
                    None,
                    None,
                    None,
                    &vg,
                    &shared,
                    0..16,
                    None,
                )
                .unwrap(),
            )
        })
    });
    group.bench_function("random path, 1 realization", |b| {
        let params = SgsParams {
            search: search.clone(),
            seed: 1,
        };
        b.iter(|| black_box(sgs(&data, &values, None, None, &grid, &vg, &params, None).unwrap()))
    });
    group.finish();
}

criterion_group!(
    benches, bench, threads, phases, wide, many, shared, quilting
);
criterion_main!(benches);

/// One patch's costs by direct scan and by FFT: channels in 2D comparing the
/// overlap alone and with a secondary variable over the whole patch, and
/// layers in 3D comparing the overlap.
fn quilting(c: &mut Criterion) {
    use simulation::{CostFft, CostScratch, Term, cost_map};
    let channels: Vec<f32> = (0..250 * 250)
        .map(|i| {
            let (x, y) = ((i % 250) as f32, (i / 250) as f32);
            f32::from(u8::from((y - 6.0 * (x / 7.0).sin()).rem_euclid(16.0) < 5.0))
        })
        .collect();
    let layers: Vec<f32> = (0..64 * 64 * 32)
        .map(|i| (((i % 64) + 2 * (i / 4096)) / 4 % 2) as f32)
        .collect();
    let secondary: Vec<f32> = channels.iter().map(|v| 3.0 * v + 0.5).collect();
    let cases: [(&str, &[f32], [usize; 3], [usize; 3], [usize; 3], bool); 3] = [
        (
            "2D overlap",
            &channels,
            [250, 250, 1],
            [40, 40, 1],
            [6, 6, 0],
            false,
        ),
        (
            "2D overlap and secondary",
            &channels,
            [250, 250, 1],
            [40, 40, 1],
            [6, 6, 0],
            true,
        ),
        (
            "3D overlap",
            &layers,
            [64, 64, 32],
            [20, 20, 10],
            [3, 3, 1],
            false,
        ),
    ];
    let mut group = c.benchmark_group("image quilting, one patch's costs");
    group.sample_size(10);
    for (name, image, dims, patch, overlap, with_secondary) in cases {
        let cells: usize = patch.iter().product();
        let in_overlap = |c: usize| {
            let p = [
                c % patch[0],
                c / patch[0] % patch[1],
                c / (patch[0] * patch[1]),
            ];
            (0..3).any(|a| p[a] < overlap[a])
        };
        let n = (0..cells).filter(|&c| in_overlap(c)).count() as f64;
        let template: Vec<f32> = (0..cells)
            .map(|c| {
                if in_overlap(c) {
                    image[c % patch[0] + dims[0] * (c / patch[0] % patch[1])]
                } else {
                    f32::NAN
                }
            })
            .collect();
        let weights: Vec<f64> = (0..cells)
            .map(|c| if in_overlap(c) { 1.0 / n } else { 0.0 })
            .collect();
        let full: Vec<f32> = (0..cells).map(|c| (c % 3) as f32).collect();
        let all = vec![1.0 / cells as f64; cells];
        let mut terms = vec![Term {
            image,
            categorical: true,
            inv_range: 0.0,
            template: &template,
            weights: &weights,
        }];
        let mut images = vec![0];
        if with_secondary {
            terms.push(Term {
                image: &secondary,
                categorical: false,
                inv_range: 1.0 / 3.0,
                template: &full,
                weights: &all,
            });
            images.push(1);
        }
        let fft = CostFft::new(
            dims,
            patch,
            &[(image, true), (&secondary, false)][..images.len()],
        );
        let mut scratch = CostScratch::default();
        group.bench_function(format!("{name}, direct"), |b| {
            b.iter(|| black_box(cost_map(dims, patch, &terms)))
        });
        group.bench_function(format!("{name}, FFT"), |b| {
            b.iter(|| black_box(fft.costs(&terms, &images, &mut scratch)))
        });
    }
    group.finish();
}
