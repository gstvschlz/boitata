//! Consistency of a training image with the hard data, after Boisvert, Pyrcz
//! and Deutsch (2007): whether the patterns the drill holes show occur in the
//! image as often as they would in holes drilled through the image itself.
//!
//! 1. The hard data are placed in the cells of the simulation grid, one per
//!    cell (the one nearest the cell center).
//! 2. A string is a run of consecutive informed cells along one grid axis (z
//!    for vertical holes) at one column; a data event, each run of
//!    `pattern_length` consecutive cells of a string.
//! 3. The frequencies of the data events are compared with those of every run
//!    of `pattern_length` cells along the same axis of the image, by their
//!    Jensen-Shannon divergence (base 2: 0 for the same frequencies, 1 when no
//!    pattern is shared).
//! 4. Even data drawn from the image never match its frequencies exactly. So
//!    `n_samples` times, holes of the lengths of the strings are drilled
//!    through the image at random columns and depths and compared with it the
//!    same way. The p-value is the share of these reference divergences at
//!    least as large as the data's: a small p-value means holes like the data
//!    seldom come from the image.
//!
//! Categorical images are compared on their codes, continuous ones on
//! `n_classes` classes cut at the quantiles of the hard data. The image is
//! read in cell index space, one image cell per grid cell, as in a simulation.

use boitata_core::Geometry;
use boitata_core::rng::realization_seed;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::error::{Result, SimError};
use crate::post::quantile_sorted;
use crate::training_image::{NO_CODE, TrainingImage, TrainingValues};

/// Class of a cell without data.
const NO_CLASS: u8 = NO_CODE;
const AXES: [&str; 3] = ["x", "y", "z"];

/// Options of [`consistency`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConsistencyParams {
    /// Grid axis the holes follow: 0 (x), 1 (y) or 2 (z).
    pub axis: usize,
    /// Cells in a data event, at least 2.
    pub pattern_length: usize,
    /// Classes a continuous image is cut into, 2 to 254; unused for a
    /// categorical one.
    pub n_classes: usize,
    /// Reference holes drawn from the image for the p-value.
    pub n_samples: usize,
    pub seed: u64,
}

impl Default for ConsistencyParams {
    fn default() -> Self {
        Self {
            axis: 2,
            pattern_length: 4,
            n_classes: 4,
            n_samples: 200,
            seed: 0,
        }
    }
}

/// How well a training image agrees with the hard data.
#[derive(Debug, Clone, PartialEq)]
pub struct Consistency {
    /// Jensen-Shannon divergence of the data events from the image, 0 to 1.
    pub distance: f64,
    /// `(1 + reference divergences >= distance) / (1 + reference draws)`.
    pub p_value: f64,
    /// Divergence of each reference draw from the image, ascending.
    pub reference: Vec<f64>,
    /// Share of the data events whose pattern never occurs in the image.
    pub unseen: f64,
    pub n_events: usize,
    /// Grid cells holding hard data.
    pub n_cells: usize,
    /// Share of each code in the image; empty for a continuous image.
    pub proportions: Vec<f64>,
    /// Share of each code in the hard-data cells; empty for a continuous image.
    pub data_proportions: Vec<f64>,
}

/// Compares `image` with the hard data `values` at `coords` placed on
/// `grid`; see the [module docs](self). Data with a NaN value or outside the
/// grid are left out. Categorical data are codes of the image. The same seed
/// gives the same result on any number of threads.
pub fn consistency(
    image: &TrainingImage,
    grid: &Geometry,
    coords: &[[f64; 3]],
    values: &[f64],
    params: &ConsistencyParams,
) -> Result<Consistency> {
    let &ConsistencyParams {
        axis,
        pattern_length: length,
        n_classes,
        n_samples,
        seed,
    } = params;
    check(image, grid, coords, values, params)?;

    let cells = nearest_per_cell(grid, coords, values);
    let (cells, n_symbols, edges) = match image.values() {
        TrainingValues::Categorical(_) => {
            let k = image.n_categories();
            let classes = cells
                .iter()
                .map(|&(cell, v)| match v {
                    v if v.fract() == 0.0 && (0.0..k as f64).contains(&v) => Ok((cell, v as u8)),
                    v => Err(SimError::InvalidParameters(format!(
                        "hard data code {v} is not a code of the training image (0 to {})",
                        k - 1
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            (classes, k, Vec::new())
        }
        TrainingValues::Continuous(_) => {
            let mut sorted: Vec<f64> = cells.iter().map(|c| c.1).collect();
            sorted.sort_by(f64::total_cmp);
            let edges: Vec<f64> = (1..n_classes)
                .map(|c| quantile_sorted(&sorted, c as f64 / n_classes as f64))
                .collect();
            let classes = cells
                .iter()
                .map(|&(c, v)| (c, class_of(v, &edges)))
                .collect();
            (classes, n_classes, edges)
        }
    };
    (n_symbols as u64)
        .checked_pow(length as u32)
        .filter(|&n| n < 1 << 63)
        .ok_or_else(|| {
            SimError::InvalidParameters(format!(
                "{n_symbols} classes and pattern_length {length} give more patterns than can be counted; lower pattern_length"
            ))
        })?;

    let strings = strings(grid, &cells, axis);
    let data = Histogram::new(
        strings
            .iter()
            .flat_map(|s| events(s, length, n_symbols))
            .collect(),
    );
    if data.total == 0 {
        return Err(SimError::InsufficientData(format!(
            "no hole has {length} consecutive informed cells along {}; lower pattern_length, or match the grid cell size along {0} to the sample spacing",
            AXES[axis]
        )));
    }

    let classes: Vec<u8> = match image.values() {
        TrainingValues::Categorical(codes) => codes.clone(),
        TrainingValues::Continuous(v) => v.iter().map(|&v| class_of(v.into(), &edges)).collect(),
    };
    let columns = Columns::new(image.dims(), axis);
    let ti = Histogram::new(columns.all_events(&classes, length, n_symbols));
    let distance = js_divergence(&data, &ti);
    let lengths: Vec<usize> = strings.iter().map(Vec::len).collect();
    let mut reference: Vec<f64> = (0..n_samples)
        .into_par_iter()
        .map(|sample| {
            let mut rng = StdRng::seed_from_u64(realization_seed(seed, sample as u64));
            let ids = columns.drill(&classes, &lengths, length, n_symbols, &mut rng);
            js_divergence(&Histogram::new(ids), &ti)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|d| !d.is_nan())
        .collect();
    reference.sort_by(f64::total_cmp);
    let larger = reference.iter().filter(|&&d| d >= distance).count();

    let data_proportions = if image.is_categorical() {
        let mut shares = vec![0.0; n_symbols];
        for &(_, c) in &cells {
            shares[usize::from(c)] += 1.0 / cells.len() as f64;
        }
        shares
    } else {
        Vec::new()
    };
    Ok(Consistency {
        distance,
        p_value: (1 + larger) as f64 / (1 + reference.len()) as f64,
        reference,
        unseen: data.share_absent_from(&ti),
        n_events: data.total as usize,
        n_cells: cells.len(),
        proportions: image.proportions().to_vec(),
        data_proportions,
    })
}

fn check(
    image: &TrainingImage,
    grid: &Geometry,
    coords: &[[f64; 3]],
    values: &[f64],
    params: &ConsistencyParams,
) -> Result<()> {
    let invalid = |m: String| Err(SimError::InvalidParameters(m));
    let &ConsistencyParams {
        axis,
        pattern_length,
        n_classes,
        n_samples,
        ..
    } = params;
    if coords.len() != values.len() {
        return invalid(format!(
            "{} coordinates but {} values",
            coords.len(),
            values.len()
        ));
    }
    if axis > 2 {
        return invalid(format!("axis must be 0 (x), 1 (y) or 2 (z), got {axis}"));
    }
    if pattern_length < 2 {
        return invalid(format!(
            "pattern_length must be at least 2, got {pattern_length}"
        ));
    }
    if !image.is_categorical() && !(2..=254).contains(&n_classes) {
        return invalid(format!("n_classes must be 2 to 254, got {n_classes}"));
    }
    if n_samples == 0 {
        return invalid("n_samples must be at least 1".into());
    }
    for (what, along) in [
        ("the grid", grid.count[axis]),
        ("the training image", image.dims()[axis]),
    ] {
        if along < pattern_length {
            return invalid(format!(
                "{what} has {along} cells along {}, fewer than pattern_length {pattern_length}; choose the axis the holes follow (z for vertical holes, y on a 2D grid) or lower pattern_length",
                AXES[axis]
            ));
        }
    }
    Ok(())
}

/// `(cell, value)` of the datum nearest each informed cell center, by cell;
/// ties go to the earlier datum.
fn nearest_per_cell(grid: &Geometry, coords: &[[f64; 3]], values: &[f64]) -> Vec<(u64, f64)> {
    let mut placed: Vec<(u64, f64, usize)> = coords
        .iter()
        .zip(values)
        .enumerate()
        .filter(|(_, (_, v))| !v.is_nan())
        .filter_map(|(k, (&p, _))| {
            let (cell, at) = grid.locate(p)?;
            Some((cell, at.iter().map(|a| (a - 0.5).powi(2)).sum(), k))
        })
        .collect();
    placed.sort_by(|a, b| {
        (a.0, a.1, a.2)
            .partial_cmp(&(b.0, b.1, b.2))
            .expect("finite")
    });
    placed.dedup_by_key(|p| p.0);
    placed.into_iter().map(|(c, _, k)| (c, values[k])).collect()
}

/// Class of a continuous `value`: the number of `edges` at or below it.
fn class_of(value: f64, edges: &[f64]) -> u8 {
    if value.is_nan() {
        NO_CLASS
    } else {
        edges.partition_point(|&e| e <= value) as u8
    }
}

/// Runs of cells consecutive along `axis` at one column, each as its classes
/// in order along the axis.
fn strings(grid: &Geometry, cells: &[(u64, u8)], axis: usize) -> Vec<Vec<u8>> {
    let (b, c) = other_axes(axis);
    let mut keyed: Vec<([usize; 2], usize, u8)> = cells
        .iter()
        .map(|&(cell, class)| {
            let ijk = grid.ijk(cell);
            ([ijk[b], ijk[c]], ijk[axis], class)
        })
        .collect();
    keyed.sort_unstable();
    let mut strings: Vec<Vec<u8>> = Vec::new();
    let mut previous = None;
    for (column, at, class) in keyed {
        match strings.last_mut() {
            Some(s) if previous == Some((column, at.wrapping_sub(1))) => s.push(class),
            _ => strings.push(vec![class]),
        }
        previous = Some((column, at));
    }
    strings
}

/// Pattern id of each run of `length` cells of `classes` without a no-data
/// cell: the base-`n_symbols` number whose digits are its classes.
fn events(classes: &[u8], length: usize, n_symbols: usize) -> impl Iterator<Item = u64> + '_ {
    classes
        .windows(length)
        .filter(|w| !w.contains(&NO_CLASS))
        .map(move |w| {
            w.iter()
                .fold(0, |id, &c| id * n_symbols as u64 + u64::from(c))
        })
}

fn other_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// The columns of a training image along one axis.
struct Columns {
    along: usize,
    stride: usize,
    across: [usize; 2],
    across_stride: [usize; 2],
}

impl Columns {
    fn new(dims: [usize; 3], axis: usize) -> Self {
        let strides = [1, dims[0], dims[0] * dims[1]];
        let (b, c) = other_axes(axis);
        Self {
            along: dims[axis],
            stride: strides[axis],
            across: [dims[b], dims[c]],
            across_stride: [strides[b], strides[c]],
        }
    }

    fn read(&self, classes: &[u8], [u, v]: [usize; 2], from: usize, n: usize) -> Vec<u8> {
        let base = u * self.across_stride[0] + v * self.across_stride[1];
        (from..from + n)
            .map(|t| classes[base + t * self.stride])
            .collect()
    }

    fn all_events(&self, classes: &[u8], length: usize, n_symbols: usize) -> Vec<u64> {
        let mut ids = Vec::new();
        for v in 0..self.across[1] {
            for u in 0..self.across[0] {
                let column = self.read(classes, [u, v], 0, self.along);
                ids.extend(events(&column, length, n_symbols));
            }
        }
        ids
    }

    /// Pattern ids of holes of `lengths` cells, each at a random column and
    /// depth; a hole longer than a column is cut into column-long pieces.
    fn drill(
        &self,
        classes: &[u8],
        lengths: &[usize],
        length: usize,
        n_symbols: usize,
        rng: &mut StdRng,
    ) -> Vec<u64> {
        let mut ids = Vec::new();
        for &total in lengths {
            let mut left = total;
            while left > 0 {
                let n = left.min(self.along);
                left -= n;
                if n < length {
                    continue;
                }
                let u = rng.gen_range(0..self.across[0]);
                let v = rng.gen_range(0..self.across[1]);
                let from = rng.gen_range(0..=self.along - n);
                ids.extend(events(
                    &self.read(classes, [u, v], from, n),
                    length,
                    n_symbols,
                ));
            }
        }
        ids
    }
}

/// Counts of pattern ids, sorted by id.
struct Histogram {
    counts: Vec<(u64, u64)>,
    total: u64,
}

impl Histogram {
    fn new(mut ids: Vec<u64>) -> Self {
        ids.sort_unstable();
        let mut counts: Vec<(u64, u64)> = Vec::new();
        for id in ids {
            match counts.last_mut() {
                Some((last, n)) if *last == id => *n += 1,
                _ => counts.push((id, 1)),
            }
        }
        let total = counts.iter().map(|c| c.1).sum();
        Self { counts, total }
    }

    /// Counts of both histograms for every id in either, in id order.
    fn merged(&self, other: &Histogram) -> Vec<(u64, u64)> {
        let (a, b) = (&self.counts, &other.counts);
        let (mut i, mut j) = (0, 0);
        let mut out = Vec::with_capacity(a.len() + b.len());
        while i < a.len() || j < b.len() {
            match (a.get(i), b.get(j)) {
                (Some(x), Some(y)) if x.0 == y.0 => {
                    out.push((x.1, y.1));
                    (i, j) = (i + 1, j + 1);
                }
                (Some(x), Some(y)) if x.0 < y.0 => {
                    out.push((x.1, 0));
                    i += 1;
                }
                (Some(x), None) => {
                    out.push((x.1, 0));
                    i += 1;
                }
                (_, Some(y)) => {
                    out.push((0, y.1));
                    j += 1;
                }
                (None, None) => unreachable!(),
            }
        }
        out
    }

    /// Share of this histogram's counts whose id `other` lacks.
    fn share_absent_from(&self, other: &Histogram) -> f64 {
        let absent: u64 = self
            .merged(other)
            .iter()
            .filter(|c| c.1 == 0)
            .map(|c| c.0)
            .sum();
        absent as f64 / self.total as f64
    }
}

/// Jensen-Shannon divergence of two histograms, base 2, 0 to 1; NaN if
/// either is empty.
fn js_divergence(p: &Histogram, q: &Histogram) -> f64 {
    if p.total == 0 || q.total == 0 {
        return f64::NAN;
    }
    let term = |x: f64, m: f64| if x > 0.0 { x * (x / m).log2() } else { 0.0 };
    let sum: f64 = p
        .merged(q)
        .iter()
        .map(|&(a, b)| {
            let (a, b) = (a as f64 / p.total as f64, b as f64 / q.total as f64);
            let m = 0.5 * (a + b);
            term(a, m) + term(b, m)
        })
        .sum();
    (0.5 * sum).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, RecordBatch};
    use boitata_core::BlockModel;

    use super::*;
    use crate::objects::{ObjectSet, Param, Shape, object_training_image};
    use rand::seq::SliceRandom;

    const SIDE: usize = 100;

    fn grid() -> Geometry {
        Geometry {
            origin: [0.0; 3],
            size: [1.0; 3],
            count: [SIDE, SIDE, 1],
            rotation: [0.0; 3],
        }
    }

    /// Sinuous channels on 30 % of the image, running along x; along y when
    /// `rotated`.
    fn channels(rotated: bool) -> Vec<f64> {
        let set = ObjectSet {
            code: 1,
            proportion: 0.3,
            shape: Shape::Channel {
                width: Param::Uniform(4.0, 10.0),
                thickness: Param::Fixed(1.0),
                azimuth: Param::Fixed(if rotated { 0.0 } else { 90.0 }),
                wavelength: Param::Fixed(60.0),
                amplitude: Param::Fixed(6.0),
            },
        };
        let model = object_training_image(grid(), &[set], 0, 7, "facies").unwrap();
        let ti = TrainingImage::categorical(&model, "facies").unwrap();
        ti.codes().unwrap().iter().map(|&c| f64::from(c)).collect()
    }

    fn image(codes: Vec<f64>) -> TrainingImage {
        let column = Arc::new(Float64Array::from(codes));
        let batch = RecordBatch::try_from_iter([("facies", column as _)]).unwrap();
        TrainingImage::categorical(&BlockModel::regular(grid(), batch).unwrap(), "facies").unwrap()
    }

    /// 25 holes along y across the whole image, at distinct columns drawn
    /// with `seed`, every cell sampled.
    fn holes(codes: &[f64], seed: u64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut columns: Vec<usize> = (0..SIDE).collect();
        columns.shuffle(&mut StdRng::seed_from_u64(seed));
        columns[..25]
            .iter()
            .flat_map(|&i| (0..SIDE).map(move |j| (i, j)))
            .map(|(i, j)| ([i as f64 + 0.5, j as f64 + 0.5, 0.5], codes[i + SIDE * j]))
            .unzip()
    }

    fn params() -> ConsistencyParams {
        ConsistencyParams {
            axis: 1,
            seed: 1,
            ..Default::default()
        }
    }

    /// Holes drilled through an image are consistent with it: their p-values
    /// spread over (0, 1), seldom below 0.05. Against the image turned 90°
    /// they are always below.
    #[test]
    fn data_from_the_image_are_consistent_and_from_a_rotated_one_are_not() {
        let codes = channels(false);
        let (own, rotated) = (image(codes.clone()), image(channels(true)));
        let runs: Vec<(Consistency, Consistency)> = (0..20)
            .map(|seed| {
                let (coords, values) = holes(&codes, seed);
                let run = |ti| consistency(ti, &grid(), &coords, &values, &params()).unwrap();
                (run(&own), run(&rotated))
            })
            .collect();
        let p_own: Vec<f64> = runs.iter().map(|r| r.0.p_value).collect();
        let mean = p_own.iter().sum::<f64>() / 20.0;
        assert!(mean > 0.3, "{p_own:?}");
        assert!(
            p_own.iter().filter(|&&p| p < 0.05).count() <= 2,
            "{p_own:?}"
        );
        for (a, b) in &runs {
            assert!(b.p_value < 0.025 && b.distance > 10.0 * a.distance, "{b:?}");
        }
        let (a, _) = &runs[0];
        assert_eq!(
            (a.n_cells, a.n_events, a.reference.len()),
            (2500, 25 * 97, 200)
        );
        assert!((a.data_proportions.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(a.proportions.len(), 2);
    }

    #[test]
    fn same_result_on_any_number_of_threads() {
        let codes = channels(false);
        let (coords, values) = holes(&codes, 0);
        let ti = image(codes);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| consistency(&ti, &grid(), &coords, &values, &params()).unwrap())
        };
        assert_eq!(run(1), run(4));
        assert_eq!(run(4), run(4));
    }

    #[test]
    fn continuous_images_use_classes_of_the_data() {
        let noisy = |codes: Vec<f64>| -> Vec<f64> {
            codes
                .iter()
                .enumerate()
                .map(|(k, c)| 10.0 * c + (realization_seed(3, k as u64) % 1000) as f64 / 1000.0)
                .collect()
        };
        let continuous = |values: Vec<f64>| {
            let column = Arc::new(Float64Array::from(values));
            let batch = RecordBatch::try_from_iter([("v", column as _)]).unwrap();
            TrainingImage::continuous(&BlockModel::regular(grid(), batch).unwrap(), "v").unwrap()
        };
        let own = noisy(channels(false));
        let (coords, values) = holes(&own, 0);
        let p = ConsistencyParams {
            n_classes: 3,
            ..params()
        };
        let a = consistency(&continuous(own), &grid(), &coords, &values, &p).unwrap();
        let b = consistency(
            &continuous(noisy(channels(true))),
            &grid(),
            &coords,
            &values,
            &p,
        )
        .unwrap();
        assert!(a.proportions.is_empty() && a.data_proportions.is_empty());
        assert!(a.distance < b.distance);
    }

    #[test]
    fn errors_say_what_to_change() {
        let codes = channels(false);
        let ti = image(codes.clone());
        let (coords, values) = holes(&codes, 0);
        let fails = |p: ConsistencyParams, values: &[f64]| {
            consistency(&ti, &grid(), &coords, values, &p)
                .unwrap_err()
                .to_string()
        };
        assert!(fails(Default::default(), &values).contains("cells along z"));
        let p = || params();
        assert!(
            fails(
                ConsistencyParams {
                    pattern_length: 1,
                    ..p()
                },
                &values
            )
            .contains("at least 2")
        );
        assert!(fails(ConsistencyParams { axis: 3, ..p() }, &values).contains("axis"));
        assert!(
            fails(
                ConsistencyParams {
                    n_samples: 0,
                    ..p()
                },
                &values
            )
            .contains("n_samples")
        );
        let mut bad = values.clone();
        bad[0] = 2.0;
        assert!(fails(p(), &bad).contains("not a code"));
        let every_other: Vec<f64> = values
            .iter()
            .enumerate()
            .map(|(k, &v)| if k % 2 == 0 { v } else { f64::NAN })
            .collect();
        assert!(fails(p(), &every_other).contains("consecutive"));
    }

    #[test]
    fn js_is_zero_for_identical_and_one_for_disjoint() {
        let a = Histogram::new(vec![1, 1, 2, 3]);
        let b = Histogram::new(vec![3, 2, 1, 1]);
        let c = Histogram::new(vec![7, 8]);
        assert_eq!(js_divergence(&a, &b), 0.0);
        assert!((js_divergence(&a, &c) - 1.0).abs() < 1e-12);
        assert!(js_divergence(&a, &Histogram::new(vec![])).is_nan());
        assert_eq!(a.share_absent_from(&Histogram::new(vec![1])), 0.5);
    }

    #[test]
    fn strings_split_at_gaps_and_columns() {
        let g = Geometry {
            count: [2, 1, 5],
            ..grid()
        };
        let cell = |x: u64, z: u64| x + 2 * z;
        let cells = [
            (cell(0, 0), 0),
            (cell(0, 1), 1),
            (cell(1, 2), 1),
            (cell(0, 3), 1),
            (cell(0, 4), 0),
        ];
        assert_eq!(
            strings(&g, &cells, 2),
            vec![vec![0, 1], vec![1, 0], vec![1]]
        );
    }

    #[test]
    fn events_skip_no_data_and_one_datum_per_cell() {
        let ids: Vec<u64> = events(&[1, 0, NO_CLASS, 1, 1, 0], 2, 2).collect();
        assert_eq!(ids, vec![0b10, 0b11, 0b10]);
        let placed = nearest_per_cell(
            &grid(),
            &[
                [0.1, 0.1, 0.0],
                [0.5, 0.45, 0.0],
                [0.6, 0.5, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            &[1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(placed, vec![(0, 2.0)]);
        assert_eq!(class_of(5.0, &[1.0, 5.0]), 2);
    }
}
