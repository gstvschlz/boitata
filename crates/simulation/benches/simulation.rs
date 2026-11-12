use boitata_core::Geometry;
use boitata_core::rng::realization_seed;
use criterion::profiler::Profiler;
use criterion::{Criterion, criterion_group};
use estimation::Search;
use estimation::lva::LocalAnisotropy;
use rand::SeedableRng;
use rand::rngs::StdRng;
use simulation::pgs::LatentFields;
use simulation::{
    Bands, ContinuousOptions, Decorrelation, Dss, Hierarchy, Lattice, PgsParams, SgsParams,
    SharedSgs, SisParams, Trend, TruncationRule, TurningBandsEnsemble, TurningBandsParams, bounds,
    dss_in, multivariate, sgs, sgs_shared, sis, turning_bands,
};
use std::collections::HashMap;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;
use transforms::Pca;
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
            .unwrap()
        })
    });
    let bands = Bands::new(lo, hi, &vg, &params, &mut StdRng::seed_from_u64(1)).unwrap();
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
            black_box(e.summary(&grid, None, None, &Default::default()).unwrap())
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
        b.iter(|| black_box(e.summary(&grid, None, None, &Default::default()).unwrap()))
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

/// Counts for one function. `file:line` is the leaf line with the most samples.
struct Leaf {
    samples: u64,
    file: String,
    line: u32,
    at_line: u64,
}

impl Leaf {
    fn add(&mut self, samples: u64, file: String, line: u32) {
        self.samples += samples;
        let hotter = samples > self.at_line
            || (samples == self.at_line
                && (line < self.line || (line == self.line && file < self.file)));
        if hotter {
            self.file = file;
            self.line = line;
            self.at_line = samples;
        }
    }
}

/// `frames[0]` is the interrupted function. Its first symbol is the innermost inline.
fn leaf_of(frames: &pprof::Frames) -> (String, String, u32) {
    match frames.frames.first().and_then(|frame| frame.first()) {
        Some(symbol) => (
            symbol.name(),
            symbol.filename().into_owned(),
            symbol.lineno(),
        ),
        None => ("unknown".to_owned(), "unknown".to_owned(), 0),
    }
}

fn write_top(dir: &Path, report: &pprof::Report) {
    let mut counts: HashMap<String, Leaf> = HashMap::new();
    let mut total = 0u64;
    for (frames, count) in &report.data {
        let Ok(count) = u64::try_from(*count) else {
            continue;
        };
        if count == 0 {
            continue;
        }
        total += count;
        let (name, file, line) = leaf_of(frames);
        match counts.entry(name) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(Leaf {
                    samples: count,
                    file,
                    line,
                    at_line: count,
                });
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                slot.get_mut().add(count, file, line);
            }
        }
    }
    let mut rows: Vec<_> = counts.into_iter().collect();
    rows.sort_by(|a, b| b.1.samples.cmp(&a.1.samples).then_with(|| a.0.cmp(&b.0)));
    rows.truncate(15);
    let mut lines = vec![format!("total samples {total}")];
    if total > 0 {
        for (name, row) in rows {
            let pct = 100.0 * row.samples as f64 / total as f64;
            lines.push(format!(
                "{pct:6.2}%  {count:8}  {name}  {file}:{line}",
                count = row.samples,
                file = row.file,
                line = row.line,
            ));
        }
    }
    std::fs::create_dir_all(dir).expect("profile directory");
    std::fs::write(dir.join("top.txt"), lines.join("\n") + "\n").expect("top.txt");
}

struct SelfTime {
    guard: Option<pprof::ProfilerGuard<'static>>,
}

impl SelfTime {
    fn new() -> Self {
        Self { guard: None }
    }
}

impl Profiler for SelfTime {
    fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
        self.guard = Some(pprof::ProfilerGuard::new(997).expect("start profiler"));
    }

    fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
        let guard = self.guard.take().expect("profiler was started");
        let report = guard.report().build().expect("profile report");
        write_top(benchmark_dir, &report);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(SelfTime::new());
    targets = bench, threads, phases, wide, many, shared, quilting, scenarios
}

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

const REALIZATIONS: usize = 10;
const SEED: u64 = 1;
const NODES: usize = 100 * 100 * 10;
const DATA: usize = 2_000;
const TREND_CLASSES: usize = 5;
const SCENARIOS: &[&str] = &[
    "sgs-shared",
    "sgs-random",
    "sgs-lva",
    "sgs-trend",
    "dss",
    "dss-lva",
    "tb",
    "tb-trend",
    "multivariate-pca",
    "sis",
    "sis-local-proportions",
    "pgs",
    "pgs-local-proportions",
];

fn neighborhood() -> Search {
    Search {
        min_samples: 8,
        max_samples: 24,
        radius: 150.0,
        ..Default::default()
    }
}

fn spherical(nugget: f64, sill: f64, range: f64) -> Variogram {
    Variogram {
        nugget,
        ..Variogram::single(Model::Spherical, sill, range)
    }
}

fn sample(i: usize) -> (f64, f64, f64) {
    (
        ((i * 7919) % 10_000) as f64 * 0.1,
        ((i * 104_729) % 10_000) as f64 * 0.1,
        ((i * 31) % 500) as f64 * 0.1,
    )
}

fn trend_of(p: &(f64, f64, f64)) -> f64 {
    0.15 + 0.70 * p.0 / 1_000.0 + 0.25 * p.1 / 1_000.0 + 0.10 * p.2 / 50.0
}

fn spatial(p: &(f64, f64, f64)) -> f64 {
    (p.0 / 220.0).sin() + 0.65 * (p.1 / 160.0).cos() + 0.35 * p.2 / 50.0
}

fn grade(p: &(f64, f64, f64)) -> f64 {
    (1.2 * trend_of(p) + 0.35 * spatial(p)).exp()
}

fn grade_b(p: &(f64, f64, f64)) -> f64 {
    (0.85 * trend_of(p) + 0.30 * spatial(p) + 0.25 * (p.0 / 140.0).cos()).exp()
}

/// Azimuth in degrees, rotating smoothly from west to east.
fn azimuth_of(p: &(f64, f64, f64)) -> f64 {
    15.0 + 100.0 * p.0 / 1_000.0 + 18.0 * (p.1 / 1_000.0 - 0.5) + 6.0 * (p.2 / 50.0 - 0.5)
}

fn proportions_of(p: &(f64, f64, f64)) -> [f64; 3] {
    let u = p.0 / 1_000.0;
    let v = p.1 / 1_000.0;
    let w = p.2 / 50.0;
    let a = 0.46 - 0.12 * u + 0.04 * w;
    let b = 0.28 + 0.10 * v;
    [a, b, 1.0 - a - b]
}

fn category_of(i: usize, p: &[f64; 3]) -> usize {
    let u = (i.wrapping_mul(1_103_515_245).wrapping_add(12_345) % 10_000) as f64 / 10_000.0;
    if u < p[0] {
        0
    } else if u < p[0] + p[1] {
        1
    } else {
        2
    }
}

fn proportion_rows(points: &[(f64, f64, f64)]) -> Vec<Vec<f64>> {
    points.iter().map(|p| proportions_of(p).to_vec()).collect()
}

fn mix_nodes(values: &[f64]) -> u64 {
    assert_eq!(values.len(), NODES, "scenario grid");
    let mut h = 0u64;
    for v in values {
        h = h
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(v.to_bits());
    }
    h
}

fn mix_categories(values: &[usize]) -> u64 {
    assert_eq!(values.len(), NODES, "scenario grid");
    let mut h = 0u64;
    for &v in values {
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(v as u64);
    }
    h
}

struct World {
    lattice: Lattice,
    grid: Vec<(f64, f64, f64)>,
    data: Vec<(f64, f64, f64)>,
    values: Vec<f64>,
    values_b: Vec<f64>,
    trend_data: Vec<f64>,
    trend_nodes: Vec<f64>,
    anisotropy: LocalAnisotropy,
    categories: Vec<usize>,
    prop_data: Vec<Vec<f64>>,
    prop_nodes: Vec<Vec<f64>>,
    search: Search,
    variogram: Variogram,
    indicators: Vec<Variogram>,
    latent: Vec<Variogram>,
    rule: TruncationRule,
    hierarchy: Hierarchy,
}

impl World {
    fn prepare() -> Self {
        let lattice = Lattice::regular(Geometry {
            origin: [0.0; 3],
            size: [10.0, 10.0, 5.0],
            count: [100, 100, 10],
            rotation: [0.0; 3],
        });
        assert_eq!(lattice.len(), NODES);
        let grid: Vec<_> = (0..lattice.len()).map(|m| lattice.location(m)).collect();
        let data: Vec<_> = (0..DATA).map(sample).collect();
        let values: Vec<_> = data.iter().map(grade).collect();
        let values_b: Vec<_> = data.iter().map(grade_b).collect();
        let trend_data: Vec<_> = data.iter().map(trend_of).collect();
        let trend_nodes: Vec<_> = grid.iter().map(trend_of).collect();
        let azimuths: Vec<[f64; 3]> = grid.iter().map(|p| [azimuth_of(p), 0.0, 0.0]).collect();
        let span = azimuths.iter().fold((f64::MAX, f64::MIN), |(lo, hi), a| {
            (lo.min(a[0]), hi.max(a[0]))
        });
        assert!(span.1 - span.0 > 60.0, "azimuth field does not rotate");
        let anisotropy = LocalAnisotropy::new(grid.clone(), azimuths, vec![[0.55, 0.30]; NODES])
            .expect("local anisotropy");
        let prop_data = proportion_rows(&data);
        let prop_nodes = proportion_rows(&grid);
        assert!(prop_data.iter().chain(&prop_nodes).all(|row| {
            let sum: f64 = row.iter().sum();
            (sum - 1.0).abs() < 1e-9 && row.iter().all(|p| *p > 0.0)
        }));
        let categories: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(i, p)| category_of(i, &proportions_of(p)))
            .collect();
        assert!(
            (0..3).all(|c| categories.contains(&c)),
            "synthetic data is missing a category"
        );
        let hierarchy = Hierarchy::Split {
            field: 0,
            children: vec![
                Hierarchy::Facies(0),
                Hierarchy::Split {
                    field: 1,
                    children: vec![Hierarchy::Facies(1), Hierarchy::Facies(2)],
                },
            ],
        };
        hierarchy.validate(3, 2).expect("truncation rule");
        let rule = hierarchy.rule(&[0.42, 0.33, 0.25]);
        Self {
            lattice,
            grid,
            data,
            values,
            values_b,
            trend_data,
            trend_nodes,
            anisotropy,
            categories,
            prop_data,
            prop_nodes,
            search: neighborhood(),
            variogram: spherical(0.08, 0.92, 80.0),
            indicators: vec![
                spherical(0.05, 0.95, 100.0),
                spherical(0.05, 0.95, 70.0),
                spherical(0.05, 0.95, 45.0),
            ],
            latent: vec![spherical(0.05, 0.95, 90.0), spherical(0.05, 0.95, 50.0)],
            rule,
            hierarchy,
        }
    }

    fn sgs_params(&self, k: usize) -> SgsParams {
        SgsParams {
            search: vec![self.search.clone()],
            seed: realization_seed(SEED, k as u64),
        }
    }

    /// Shared path. A shared path refuses local anisotropy, so `sgs-lva` does not use this.
    fn run_sgs_shared(&self, trend: bool) -> u64 {
        let search = [self.search.clone()];
        let shared = SharedSgs {
            lattice: &self.lattice,
            search: &search,
            levels: None,
            seed: SEED,
        };
        let trend = trend.then_some(Trend {
            data: self.trend_data.as_slice(),
            nodes: self.trend_nodes.as_slice(),
            classes: TREND_CLASSES,
        });
        let batch = sgs_shared(
            &self.data,
            &self.values,
            None,
            None,
            None,
            trend,
            &self.variogram,
            &shared,
            0..REALIZATIONS,
            None,
        )
        .unwrap();
        let mut h = 0u64;
        for k in 0..REALIZATIONS {
            h = h.wrapping_add(mix_nodes(&batch.realization(k)));
        }
        h
    }

    fn run_sgs(&self, lva: bool) -> u64 {
        let local = lva.then_some(&self.anisotropy);
        let mut h = 0u64;
        for k in 0..REALIZATIONS {
            let values = sgs(
                &self.data,
                &self.values,
                None,
                None,
                &self.grid,
                &self.variogram,
                &self.sgs_params(k),
                local,
            )
            .unwrap()
            .values;
            h = h.wrapping_add(mix_nodes(&values));
        }
        h
    }

    fn run_dss(&self, lva: bool) -> u64 {
        let local = lva.then_some(&self.anisotropy);
        let fitted = Dss::new(&self.data, &self.values, None, None, None, &self.variogram).unwrap();
        let mut h = 0u64;
        for k in 0..REALIZATIONS {
            let values = fitted
                .simulate(None, &self.grid, &self.sgs_params(k), local)
                .unwrap()
                .values;
            h = h.wrapping_add(mix_nodes(&values));
        }
        h
    }

    fn run_tb(&self, trend: bool) -> u64 {
        let (lo, hi) = bounds(&self.grid);
        let data_trend = trend.then_some((self.trend_data.as_slice(), TREND_CLASSES));
        let params = TurningBandsParams {
            search: self.search.clone(),
            seed: SEED,
            ..Default::default()
        };
        let ensemble = TurningBandsEnsemble::new(
            &self.data,
            &self.values,
            None,
            None,
            None,
            data_trend,
            lo,
            hi,
            &self.variogram,
            &params,
            REALIZATIONS,
        )
        .unwrap();
        let at_nodes = trend.then_some(self.trend_nodes.as_slice());
        let mut h = 0u64;
        for values in ensemble
            .realizations(0..REALIZATIONS, &self.grid, |_| None, at_nodes)
            .unwrap()
        {
            h = h.wrapping_add(mix_nodes(&values));
        }
        h
    }

    fn run_multivariate(&self) -> u64 {
        let rows: Vec<Vec<f64>> = (0..self.data.len())
            .map(|i| vec![self.values[i], self.values_b[i]])
            .collect();
        let transform = Decorrelation::Pca(Pca::fit(&rows, None, true).unwrap());
        let factors = transform.forward(&rows);
        let summaries = multivariate(
            REALIZATIONS,
            SEED,
            2,
            &transform,
            &ContinuousOptions::default(),
            None,
            |_, j, seed| {
                let column: Vec<f64> = factors.iter().map(|row| row[j]).collect();
                Ok(sgs(
                    &self.data,
                    &column,
                    None,
                    None,
                    &self.grid,
                    &self.variogram,
                    &SgsParams {
                        search: vec![self.search.clone()],
                        seed,
                    },
                    None,
                )?
                .values)
            },
        )
        .unwrap();
        assert_eq!(summaries.len(), 2);
        let mut h = 0u64;
        for summary in &summaries {
            assert_eq!(summary.n, REALIZATIONS);
            h = h.wrapping_add(mix_nodes(&summary.mean));
        }
        h
    }

    fn run_sis(&self, local: bool) -> u64 {
        let proportions = local.then_some((self.prop_data.as_slice(), self.prop_nodes.as_slice()));
        let mut h = 0u64;
        for k in 0..REALIZATIONS {
            let categories = sis(
                &self.data,
                &self.categories,
                None,
                &self.grid,
                3,
                &self.indicators,
                &SisParams {
                    search: self.search.clone(),
                    seed: realization_seed(SEED, k as u64),
                },
                proportions,
            )
            .unwrap()
            .categories;
            h = h.wrapping_add(mix_categories(&categories));
        }
        h
    }

    fn run_pgs(&self, local: bool) -> u64 {
        let bands = TurningBandsParams {
            search: self.search.clone(),
            seed: SEED,
            ..Default::default()
        };
        let prepared = LatentFields::new(&self.data, &self.grid, &self.latent, &bands).unwrap();
        let mut h = 0u64;
        for k in 0..REALIZATIONS {
            let params = PgsParams {
                seed: realization_seed(SEED, k as u64),
                ..Default::default()
            };
            let categories = if local {
                prepared
                    .categories_local(
                        &self.categories,
                        &self.grid,
                        &self.hierarchy,
                        &self.prop_data,
                        &self.prop_nodes,
                        &params,
                    )
                    .unwrap()
            } else {
                prepared
                    .categories(&self.categories, &self.grid, &self.rule, &params)
                    .unwrap()
            };
            h = h.wrapping_add(mix_categories(&categories));
        }
        h
    }

    fn run(&self, name: &str) -> u64 {
        match name {
            "sgs-shared" => self.run_sgs_shared(false),
            "sgs-random" => self.run_sgs(false),
            "sgs-lva" => self.run_sgs(true),
            "sgs-trend" => self.run_sgs_shared(true),
            "dss" => self.run_dss(false),
            "dss-lva" => self.run_dss(true),
            "tb" => self.run_tb(false),
            "tb-trend" => self.run_tb(true),
            "multivariate-pca" => self.run_multivariate(),
            "sis" => self.run_sis(false),
            "sis-local-proportions" => self.run_sis(true),
            "pgs" => self.run_pgs(false),
            "pgs-local-proportions" => self.run_pgs(true),
            other => panic!("unknown scenario {other}"),
        }
    }
}

/// Ten realizations on 100 000 nodes from 2 000 seeded samples.
///
/// One iteration is the whole run: the fit the public entry point does for
/// those realizations, then all ten. A shared path refuses local anisotropy,
/// so `sgs-lva` is the random path. `sgs-trend` stays on the shared path.
fn scenarios(c: &mut Criterion) {
    let world = World::prepare();
    let mut group = c.benchmark_group("scenario");
    group.sample_size(10);
    for name in SCENARIOS {
        let name = *name;
        group.bench_function(name, |b| b.iter(|| black_box(world.run(name))));
    }
    group.finish();
}

fn selected(name: &str, filter: Option<&str>) -> bool {
    match filter {
        None | Some("") | Some("scenario") => true,
        Some(filter) => name.contains(filter),
    }
}

fn run_once(filter: Option<&str>) {
    let world = World::prepare();
    for name in SCENARIOS {
        if !selected(name, filter) {
            continue;
        }
        let start = Instant::now();
        let checksum = world.run(name);
        println!(
            "{name} {:.3}s checksum {checksum:016x}",
            start.elapsed().as_secs_f64()
        );
    }
}

fn main() {
    let mut args = std::env::args();
    let _program = args.next();
    if args.next().as_deref() == Some("--once") {
        run_once(args.next().as_deref());
        return;
    }
    benches();
}
