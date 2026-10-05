//! Exploratory data analysis on raw columns: NaN values are skipped, weights are
//! declustering weights.

use std::collections::{BTreeMap, HashMap};

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;
use thiserror::Error;
pub use variogram::Direction;

#[derive(Error, Debug)]
pub enum EdaError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error(transparent)]
    Transform(#[from] transforms::TransformError),
}

pub type Result<T> = std::result::Result<T, EdaError>;

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(EdaError::InvalidInput(message.into()))
}

fn check(n: usize, values: &[f64], weights: Option<&[f64]>) -> Result<()> {
    if values.len() != n {
        return invalid(format!("expected {n} values, got {}", values.len()));
    }
    if values.iter().any(|v| v.is_infinite()) {
        return invalid("values must be finite or NaN");
    }
    if let Some(w) = weights {
        if w.len() != n {
            return invalid(format!("expected {n} weights, got {}", w.len()));
        }
        if w.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return invalid("weights must be finite and >= 0");
        }
    }
    Ok(())
}

/// Non-NaN values and their weights (1 without `weights`).
fn valid(values: &[f64], weights: Option<&[f64]>) -> Result<(Vec<f64>, Vec<f64>)> {
    check(values.len(), values, weights)?;
    let (v, w): (Vec<f64>, Vec<f64>) = values
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, &v)| (v, weights.map_or(1.0, |w| w[i])))
        .unzip();
    if w.iter().sum::<f64>() <= 0.0 {
        return invalid("no valid values with positive weight");
    }
    Ok((v, w))
}

/// Weighted quantiles: sorted values sit at the mid-point of their cumulative
/// weight, linear in between and constant beyond the extremes.
pub fn quantiles(
    values: &[f64],
    weights: Option<&[f64]>,
    probabilities: &[f64],
) -> Result<Vec<f64>> {
    let (v, w) = valid(values, weights)?;
    quantiles_of(&v, &w, probabilities)
}

fn quantiles_of(v: &[f64], w: &[f64], probabilities: &[f64]) -> Result<Vec<f64>> {
    if probabilities.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return invalid("probabilities must be in [0, 1]");
    }
    let mut order: Vec<usize> = (0..v.len()).filter(|&i| w[i] > 0.0).collect();
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let total: f64 = w.iter().sum();
    let (mut xs, mut ps, mut cum) = (Vec::new(), Vec::new(), 0.0);
    for &i in &order {
        ps.push((cum + w[i] / 2.0) / total);
        cum += w[i];
        xs.push(v[i]);
    }
    Ok(probabilities
        .iter()
        .map(|&p| match ps.partition_point(|&q| q < p) {
            0 => xs[0],
            k if k == xs.len() => xs[k - 1],
            k => xs[k - 1] + (p - ps[k - 1]) / (ps[k] - ps[k - 1]) * (xs[k] - xs[k - 1]),
        })
        .collect())
}

#[derive(Debug, Clone)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub std: f64,
    pub cv: f64,
    pub min: f64,
    pub max: f64,
    pub quantiles: Vec<f64>,
    /// Sum of weights.
    pub weight: f64,
}

/// Weighted summary statistics and quantiles at `probabilities`.
pub fn describe(values: &[f64], weights: Option<&[f64]>, probabilities: &[f64]) -> Result<Summary> {
    let (v, w) = valid(values, weights)?;
    let (mean, variance) = transforms::weighted_mean_variance(&v, Some(&w))?;
    let std = variance.sqrt();
    Ok(Summary {
        n: v.len(),
        mean,
        variance,
        std,
        cv: std / mean,
        min: v.iter().copied().fold(f64::INFINITY, f64::min),
        max: v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        quantiles: quantiles_of(&v, &w, probabilities)?,
        weight: w.iter().sum(),
    })
}

/// [`describe`] per category, ascending, then over all values (`None`);
/// categories without valid values are left out.
pub fn describe_by(
    values: &[f64],
    categories: &[u32],
    weights: Option<&[f64]>,
    probabilities: &[f64],
) -> Result<Vec<(Option<u32>, Summary)>> {
    check(categories.len(), values, weights)?;
    let mut groups: BTreeMap<u32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (i, (&v, &c)) in values.iter().zip(categories).enumerate() {
        if !v.is_nan() {
            let g = groups.entry(c).or_default();
            g.0.push(v);
            g.1.push(weights.map_or(1.0, |w| w[i]));
        }
    }
    let mut rows = groups
        .into_iter()
        .map(|(c, (v, w))| Ok((Some(c), describe(&v, Some(&w), probabilities)?)))
        .collect::<Result<Vec<_>>>()?;
    rows.push((None, describe(values, weights, probabilities)?));
    Ok(rows)
}

/// Totals above one cutoff.
#[derive(Debug, Clone)]
pub struct Tonnage {
    pub cutoff: f64,
    /// Sum of weight × density of values at or above `cutoff`.
    pub tonnage: f64,
    /// Weighted mean of those values; NaN when `tonnage` is 0.
    pub mean_grade: f64,
    /// `tonnage × mean_grade`.
    pub metal: f64,
}

/// Tonnage and metal at or above each cutoff, per category (ascending, every
/// code present) then over all (`None`); NaN values are skipped.
fn curves(
    values: &[f64],
    categories: Option<&[u32]>,
    tonnes: Option<&[f64]>,
    cutoffs: &[f64],
) -> Vec<(Option<u32>, Vec<Tonnage>)> {
    let zero = || vec![(0.0, 0.0); cutoffs.len()];
    let (mut by, mut all) = (BTreeMap::new(), zero());
    for (i, &v) in values.iter().enumerate() {
        let mut group = categories.map(|c| by.entry(c[i]).or_insert_with(zero));
        let t = tonnes.map_or(1.0, |t| t[i]);
        for (k, _) in cutoffs.iter().enumerate().filter(|(_, c)| v >= **c) {
            all[k] = (all[k].0 + t, all[k].1 + t * v);
            if let Some(g) = &mut group {
                g[k] = (g[k].0 + t, g[k].1 + t * v);
            }
        }
    }
    let finish = |sums: Vec<(f64, f64)>| {
        cutoffs
            .iter()
            .zip(sums)
            .map(|(&cutoff, (tonnage, metal))| Tonnage {
                cutoff,
                tonnage,
                mean_grade: if tonnage > 0.0 {
                    metal / tonnage
                } else {
                    f64::NAN
                },
                metal,
            })
            .collect()
    };
    by.into_iter()
        .map(|(c, s)| (Some(c), finish(s)))
        .chain([(None, finish(all))])
        .collect()
}

/// Checked `weights × density`, either alone, or `None`.
fn tonnes(
    values: &[f64],
    weights: Option<&[f64]>,
    density: Option<&[f64]>,
    cutoffs: &[f64],
) -> Result<Option<Vec<f64>>> {
    check(values.len(), values, weights)?;
    check(values.len(), values, density)?;
    if cutoffs.iter().any(|c| c.is_nan()) {
        return invalid("cutoffs must not be NaN");
    }
    let t = match (weights, density) {
        (Some(w), Some(d)) => Some(w.iter().zip(d).map(|(w, d)| w * d).collect()),
        (w, d) => w.or(d).map(<[f64]>::to_vec),
    };
    valid(values, t.as_deref())?;
    Ok(t)
}

/// Grade–tonnage curve of the data: each value stands for weight (e.g. volume)
/// × density tonnes, 1 by default.
pub fn grade_tonnage(
    values: &[f64],
    weights: Option<&[f64]>,
    density: Option<&[f64]>,
    cutoffs: &[f64],
) -> Result<Vec<Tonnage>> {
    let t = tonnes(values, weights, density, cutoffs)?;
    let mut rows = curves(values, None, t.as_deref(), cutoffs);
    Ok(rows.pop().expect("all").1)
}

/// [`grade_tonnage`] per category, ascending, then over all values (`None`).
pub fn grade_tonnage_by(
    values: &[f64],
    categories: &[u32],
    weights: Option<&[f64]>,
    density: Option<&[f64]>,
    cutoffs: &[f64],
) -> Result<Vec<(Option<u32>, Vec<Tonnage>)>> {
    check(categories.len(), values, None)?;
    let t = tonnes(values, weights, density, cutoffs)?;
    Ok(curves(values, Some(categories), t.as_deref(), cutoffs))
}

/// One row of [`compare_models`].
#[derive(Debug, Clone)]
pub struct Comparison {
    /// Index into the models.
    pub model: usize,
    pub category: Option<u32>,
    pub tonnage: Tonnage,
    /// `tonnage / reference tonnage - 1`.
    pub tonnage_diff: f64,
    /// As `tonnage_diff`, for the mean grade.
    pub grade_diff: f64,
    /// As `tonnage_diff`, for the metal.
    pub metal_diff: f64,
}

/// Grade–tonnage of several models on the same blocks, each block `tonnes`
/// (1 by default), per category then over all as in [`grade_tonnage_by`],
/// against `models[reference]`; rows by category, cutoff, then model.
pub fn compare_models(
    models: &[&[f64]],
    categories: Option<&[u32]>,
    tonnes: Option<&[f64]>,
    cutoffs: &[f64],
    reference: usize,
) -> Result<Vec<Comparison>> {
    if reference >= models.len() {
        return invalid("reference must be one of the models");
    }
    let n = models[0].len();
    if models.iter().any(|m| m.len() != n) {
        return invalid("models must have the same number of blocks");
    }
    if categories.is_some_and(|c| c.len() != n) {
        return invalid(format!("expected {n} categories"));
    }
    let tables = models
        .iter()
        .map(|m| {
            self::tonnes(m, tonnes, None, cutoffs)?;
            Ok(curves(m, categories, tonnes, cutoffs))
        })
        .collect::<Result<Vec<_>>>()?;
    let diff = |a: f64, b: f64| a / b - 1.0;
    let mut out = Vec::new();
    for (row, (category, reference)) in tables[reference].iter().enumerate() {
        for (k, r) in reference.iter().enumerate() {
            for (model, table) in tables.iter().enumerate() {
                let t = table[row].1[k].clone();
                out.push(Comparison {
                    model,
                    category: *category,
                    tonnage_diff: diff(t.tonnage, r.tonnage),
                    grade_diff: diff(t.mean_grade, r.mean_grade),
                    metal_diff: diff(t.metal, r.metal),
                    tonnage: t,
                });
            }
        }
    }
    Ok(out)
}

/// Mean and count per bin; `centers` ascending, empty bins omitted.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub centers: Vec<f64>,
    pub mean: Vec<f64>,
    pub count: Vec<usize>,
    /// Swaths only: sum of weight × density per bin.
    pub tonnage: Vec<f64>,
    /// Swaths only: sum of weight × density × value per bin.
    pub metal: Vec<f64>,
}

fn profile(bins: BTreeMap<i64, (f64, f64, usize)>, width: f64) -> Profile {
    let mut p = Profile::default();
    for (k, (sw, swz, n)) in bins {
        p.centers.push((k as f64 + 0.5) * width);
        p.mean.push(swz / sw);
        p.count.push(n);
    }
    p
}

pub enum Along {
    /// Horizontal, degrees clockwise from north.
    Azimuth(f64),
    /// 0, 1 or 2 for x, y or z.
    Axis(usize),
}

/// Weighted mean per slice of `width` along a direction, with tonnage and
/// metal as in [`grade_tonnage`]; slices start at coordinate 0 so profiles of
/// different data line up.
pub fn swath(
    coords: &[[f64; 3]],
    values: &[f64],
    weights: Option<&[f64]>,
    density: Option<&[f64]>,
    width: f64,
    along: Along,
) -> Result<Profile> {
    check(coords.len(), values, weights)?;
    check(coords.len(), values, density)?;
    if width.is_nan() || width <= 0.0 {
        return invalid("width must be positive");
    }
    let u = match along {
        Along::Azimuth(a) => [a.to_radians().sin(), a.to_radians().cos(), 0.0],
        Along::Axis(k) if k < 3 => {
            let mut u = [0.0; 3];
            u[k] = 1.0;
            u
        }
        Along::Axis(_) => return invalid("axis must be x, y or z"),
    };
    let (mut bins, mut totals) = (BTreeMap::new(), BTreeMap::new());
    for (i, (p, &z)) in coords.iter().zip(values).enumerate() {
        if z.is_nan() {
            continue;
        }
        let w = weights.map_or(1.0, |w| w[i]);
        let k = ((p[0] * u[0] + p[1] * u[1] + p[2] * u[2]) / width).floor() as i64;
        let b: &mut (f64, f64, usize) = bins.entry(k).or_default();
        b.0 += w;
        b.1 += w * z;
        b.2 += 1;
        let t = w * density.map_or(1.0, |d| d[i]);
        let m: &mut (f64, f64) = totals.entry(k).or_default();
        m.0 += t;
        m.1 += t * z;
    }
    let mut p = profile(bins, width);
    (p.tonnage, p.metal) = totals.into_values().unzip();
    Ok(p)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Mean value against signed distance to the contact between domains `inside`
/// and `outside`: each sample's distance to the nearest sample of the other
/// domain in the same hole, negative inside. Bins are `[k·bin, (k+1)·bin)`
/// covering exactly `[-max_distance, max_distance]`: the outermost one is
/// closed and ends at `max_distance`, so it may be narrower than `bin`.
#[allow(clippy::too_many_arguments)]
pub fn contact(
    coords: &[[f64; 3]],
    values: &[f64],
    domains: &[u32],
    holes: &[u32],
    inside: u32,
    outside: u32,
    max_distance: f64,
    bin: f64,
) -> Result<Profile> {
    let n = coords.len();
    check(n, values, None)?;
    if domains.len() != n || holes.len() != n {
        return invalid(format!("domains and holes need {n} values"));
    }
    if !(bin > 0.0 && max_distance > 0.0) {
        return invalid("bin and max_distance must be positive");
    }
    if inside == outside {
        return invalid("inside and outside must differ");
    }
    let mut by_hole: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for i in (0..n).filter(|&i| domains[i] == inside || domains[i] == outside) {
        by_hole.entry(holes[i]).or_default().push(i);
    }
    let last = ((max_distance / bin).ceil() as i64).max(1) - 1;
    let mut bins = BTreeMap::new();
    for members in by_hole.values() {
        for &i in members.iter().filter(|&&i| !values[i].is_nan()) {
            let d = members
                .iter()
                .filter(|&&j| domains[j] != domains[i])
                .map(|&j| distance(coords[i], coords[j]))
                .fold(f64::INFINITY, f64::min);
            if d > max_distance {
                continue;
            }
            let k = ((d / bin).floor() as i64).min(last);
            let k = if domains[i] == inside { -k - 1 } else { k };
            let b: &mut (f64, f64, usize) = bins.entry(k).or_default();
            b.0 += 1.0;
            b.1 += values[i];
            b.2 += 1;
        }
    }
    let mut p = profile(bins, bin);
    let edge = last as f64 * bin;
    for c in p.centers.iter_mut().filter(|c| c.abs() > edge) {
        *c = c.signum() * (edge + max_distance) / 2.0;
    }
    Ok(p)
}

/// Which restriction of the samples a row of [`soft_boundary`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Only the `target` domain.
    Hard,
    /// `Hard` plus samples of other domains within `buffer` of it.
    Soft,
}

/// Result of [`soft_boundary`].
#[derive(Debug, Clone)]
pub struct SoftBoundary {
    /// One per sample of the other domains, in their original order: true
    /// where within `buffer` of the nearest `target` sample.
    pub added: Vec<bool>,
    /// `[(Hard, ...), (Soft, ...)]`.
    pub rows: Vec<(Boundary, Summary)>,
}

/// Statistics of the `target` domain alone (`Hard`) against also folding in
/// samples of other domains within `buffer` of their nearest `target` sample
/// (`Soft`): how much a search allowed to cross the boundary would draw in.
pub fn soft_boundary(
    coords: &[[f64; 3]],
    values: &[f64],
    domains: &[u32],
    weights: Option<&[f64]>,
    target: u32,
    buffer: f64,
    probabilities: &[f64],
) -> Result<SoftBoundary> {
    let n = coords.len();
    check(n, values, weights)?;
    if domains.len() != n {
        return invalid(format!("expected {n} domains, got {}", domains.len()));
    }
    if buffer.is_nan() || buffer <= 0.0 {
        return invalid("buffer must be positive");
    }
    if !domains.contains(&target) {
        return invalid("target must be a domain present in domains");
    }
    let points: Vec<[f64; 3]> = (0..n)
        .filter(|&i| domains[i] == target)
        .map(|i| coords[i])
        .collect();
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&points)
        .map_err(|e| EdaError::InvalidInput(format!("{e:?}")))?;
    let other: Vec<usize> = (0..n).filter(|&i| domains[i] != target).collect();
    let added: Vec<bool> = other
        .par_iter()
        .map(|&i| {
            tree.query(&coords[i])
                .nearest_n::<SquaredEuclidean<f64>>(std::num::NonZero::<usize>::MIN)
                .execute()
                .first()
                .is_some_and(|r| r.distance.sqrt() <= buffer)
        })
        .collect();
    let mut in_soft = vec![false; n];
    other
        .iter()
        .zip(&added)
        .filter(|&(_, &a)| a)
        .for_each(|(&i, _)| in_soft[i] = true);
    let subset = |extra: bool| -> Result<Summary> {
        let rows: Vec<usize> = (0..n)
            .filter(|&i| domains[i] == target || (extra && in_soft[i]))
            .collect();
        let v: Vec<f64> = rows.iter().map(|&i| values[i]).collect();
        let w = weights.map(|w| rows.iter().map(|&i| w[i]).collect::<Vec<_>>());
        describe(&v, w.as_deref(), probabilities)
    };
    Ok(SoftBoundary {
        added,
        rows: vec![
            (Boundary::Hard, subset(false)?),
            (Boundary::Soft, subset(true)?),
        ],
    })
}

pub const CAP_PROBABILITIES: [f64; 6] = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999];

#[derive(Debug, Clone)]
pub struct Cap {
    pub cap: f64,
    /// Weighted fraction of values above `cap`.
    pub fraction: f64,
    /// `1 - capped mean / mean`.
    pub metal_removed: f64,
    pub mean: f64,
    pub cv: f64,
}

/// Effect of each cap (default: quantiles at [`CAP_PROBABILITIES`]).
pub fn capping(values: &[f64], weights: Option<&[f64]>, caps: Option<&[f64]>) -> Result<Vec<Cap>> {
    let (v, w) = valid(values, weights)?;
    let caps = match caps {
        Some(c) => c.to_vec(),
        None => quantiles_of(&v, &w, &CAP_PROBABILITIES)?,
    };
    let (mean, _) = transforms::weighted_mean_variance(&v, Some(&w))?;
    let total: f64 = w.iter().sum();
    caps.into_iter()
        .map(|cap| {
            let capped: Vec<f64> = v.iter().map(|x| x.min(cap)).collect();
            let (m, var) = transforms::weighted_mean_variance(&capped, Some(&w))?;
            let above: f64 = v
                .iter()
                .zip(&w)
                .filter(|(x, _)| **x > cap)
                .map(|(_, w)| w)
                .sum();
            Ok(Cap {
                cap,
                fraction: above / total,
                metal_removed: 1.0 - m / mean,
                mean: m,
                cv: var.sqrt() / m,
            })
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct CapReport {
    /// Infinite when the category is not capped, NaN on the all-data row.
    pub cap: f64,
    pub before: Summary,
    pub after: Summary,
    /// Number of values above the cap.
    pub capped: usize,
    /// Sum of `(value - cap) × weight` above the cap.
    pub metal_removed: f64,
}

/// Statistics before and after capping each category at `caps[category]`
/// (infinite: not capped), rows as in [`describe_by`].
pub fn capping_report(
    values: &[f64],
    categories: &[u32],
    weights: Option<&[f64]>,
    caps: &[f64],
) -> Result<Vec<(Option<u32>, CapReport)>> {
    check(categories.len(), values, weights)?;
    if caps.iter().any(|c| c.is_nan()) {
        return invalid("caps must not be NaN");
    }
    if categories.iter().any(|&c| c as usize >= caps.len()) {
        return invalid(format!(
            "expected a cap for each of {} categories",
            caps.len()
        ));
    }
    let capped = cap_values(values, categories, caps);
    let before = describe_by(values, categories, weights, &[])?;
    let after = describe_by(&capped, categories, weights, &[])?;
    Ok(before
        .into_iter()
        .zip(after)
        .map(|((c, before), (_, after))| {
            let (mut n, mut metal) = (0, 0.0);
            for (i, (v, x)) in values.iter().zip(&capped).enumerate() {
                if v > x && c.is_none_or(|c| categories[i] == c) {
                    n += 1;
                    metal += (v - x) * weights.map_or(1.0, |w| w[i]);
                }
            }
            let cap = c.map_or(f64::NAN, |c| caps[c as usize]);
            (
                c,
                CapReport {
                    cap,
                    before,
                    after,
                    capped: n,
                    metal_removed: metal,
                },
            )
        })
        .collect())
}

/// `values` clipped to `caps[category]`; NaN stays NaN.
pub fn cap_values(values: &[f64], categories: &[u32], caps: &[f64]) -> Vec<f64> {
    values
        .iter()
        .zip(categories)
        .map(|(&v, &c)| {
            if v > caps[c as usize] {
                caps[c as usize]
            } else {
                v
            }
        })
        .collect()
}

/// How a cap is chosen from the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CapRule {
    /// The weighted quantile at this probability.
    Quantile(f64),
    /// The cap removing this fraction of the metal, `1 - capped mean / mean`.
    MetalRemoved(f64),
    /// The largest cap whose capped coefficient of variation is at most this.
    Cv(f64),
}

/// Cap of `values` chosen by `rule`; the metal and CV rules bisect between
/// the smallest and largest value, evaluating each cap with [`capping`].
pub fn choose_cap(values: &[f64], weights: Option<&[f64]>, rule: CapRule) -> Result<f64> {
    let (v, w) = valid(values, weights)?;
    let effect = |cap: f64| -> Result<Cap> { Ok(capping(&v, Some(&w), Some(&[cap]))?.remove(0)) };
    let (mut lo, mut hi) = v
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let (target, metal) = match rule {
        CapRule::Quantile(p) => return Ok(quantiles_of(&v, &w, &[p])?[0]),
        CapRule::MetalRemoved(t) if (0.0..1.0).contains(&t) => (t, true),
        CapRule::Cv(t) if t > 0.0 => (t, false),
        _ => return invalid("metal_removed must be in [0, 1) and cv > 0"),
    };
    if !metal && effect(hi)?.cv <= target {
        return Ok(hi);
    }
    loop {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        let e = effect(mid)?;
        if (metal && e.metal_removed <= target) || (!metal && e.cv > target) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(if metal { hi } else { lo })
}

/// Cap per category (`0..=max`) of `values` chosen by `rule` within the
/// category; infinite for a category without valid values.
pub fn fit_caps(
    values: &[f64],
    categories: &[u32],
    weights: Option<&[f64]>,
    rule: CapRule,
) -> Result<Vec<f64>> {
    check(categories.len(), values, weights)?;
    let n = categories.iter().max().map_or(0, |&c| c + 1);
    (0..n)
        .map(|c| {
            let rows: Vec<usize> = (0..values.len()).filter(|&i| categories[i] == c).collect();
            let v: Vec<f64> = rows.iter().map(|&i| values[i]).collect();
            let w = weights.map(|w| rows.iter().map(|&i| w[i]).collect::<Vec<_>>());
            if v.iter().all(|x| x.is_nan()) {
                return Ok(f64::INFINITY);
            }
            choose_cap(&v, w.as_deref(), rule)
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Naive,
    Declustered,
    Model,
    Reference,
}

/// One row of [`validate_model`].
#[derive(Debug, Clone)]
pub struct Validation {
    pub domain: Option<u32>,
    pub source: Source,
    pub summary: Summary,
    /// Sum of tonnes of the valid blocks; NaN for data.
    pub tonnage: f64,
    /// `mean / data mean - 1`, against the declustered data when weighted.
    pub mean_diff: f64,
    /// `variance / data variance`, as `mean_diff`.
    pub variance_ratio: f64,
}

/// Block model statistics against the data, per domain (ascending) then over
/// all (`None`): data naive, declustered with `weights`, the model and an
/// optional `reference` on the same blocks (e.g. the truth), blocks weighted by
/// `tonnes`. `domains` codes the model's blocks, then the data.
pub fn validate_model(
    model: &[f64],
    data: &[f64],
    weights: Option<&[f64]>,
    domains: Option<(&[u32], &[u32])>,
    tonnes: Option<&[f64]>,
    reference: Option<&[f64]>,
    probabilities: &[f64],
) -> Result<Vec<Validation>> {
    let zeros = (vec![0; model.len()], vec![0; data.len()]);
    let (dm, dd) = domains.unwrap_or((&zeros.0, &zeros.1));
    let mut sources = vec![(Source::Naive, describe_by(data, dd, None, probabilities)?)];
    if weights.is_some() {
        let rows = describe_by(data, dd, weights, probabilities)?;
        sources.push((Source::Declustered, rows));
    }
    for (source, values) in [(Source::Model, Some(model)), (Source::Reference, reference)] {
        if let Some(v) = values {
            sources.push((source, describe_by(v, dm, tonnes, probabilities)?));
        }
    }
    let mut keys: Vec<Option<u32>> = sources
        .iter()
        .flat_map(|(_, rows)| rows.iter().map(|r| r.0))
        .filter(|k| domains.is_some() || k.is_none())
        .collect();
    keys.sort_by_key(|k| (k.is_none(), *k));
    keys.dedup();
    let mut out = Vec::new();
    for key in keys {
        let found: Vec<(Source, &Summary)> = sources
            .iter()
            .filter_map(|(s, rows)| rows.iter().find(|r| r.0 == key).map(|r| (*s, &r.1)))
            .collect();
        let (mean, variance) = found
            .iter()
            .rfind(|(s, _)| matches!(s, Source::Naive | Source::Declustered))
            .map_or((f64::NAN, f64::NAN), |(_, s)| (s.mean, s.variance));
        for (source, summary) in found {
            let tonnage = if matches!(source, Source::Model | Source::Reference) {
                summary.weight
            } else {
                f64::NAN
            };
            out.push(Validation {
                domain: key,
                source,
                summary: summary.clone(),
                tonnage,
                mean_diff: summary.mean / mean - 1.0,
                variance_ratio: summary.variance / variance,
            });
        }
    }
    Ok(out)
}

/// Weighted covariance, over the sum of weights as [`describe`]'s variance,
/// and correlation of `x` and `y`.
fn moments(x: &[f64], y: &[f64], w: &[f64]) -> (f64, f64) {
    let sw: f64 = w.iter().sum();
    if x.is_empty() || sw <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let mx = x.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / sw;
    let my = y.iter().zip(w).map(|(y, w)| y * w).sum::<f64>() / sw;
    let (mut cxy, mut cxx, mut cyy) = (0.0, 0.0, 0.0);
    for ((x, y), w) in x.iter().zip(y).zip(w) {
        cxy += w * (x - mx) * (y - my);
        cxx += w * (x - mx) * (x - mx);
        cyy += w * (y - my) * (y - my);
    }
    let r = if x.len() < 2 {
        f64::NAN
    } else {
        cxy / (cxx * cyy).sqrt()
    };
    (cxy / sw, r)
}

fn pearson(x: &[f64], y: &[f64], w: &[f64]) -> f64 {
    moments(x, y, w).1
}

/// Head (`values` at `x + h`), tail (`other`, else `values`, at `x`) and their
/// correlation for every ordered pair with `|h|` within `lag ± tolerance`
/// and, given `direction`, `h` within its cone.
pub fn h_scatter(
    coords: &[[f64; 3]],
    values: &[f64],
    other: Option<&[f64]>,
    lag: f64,
    tolerance: f64,
    direction: Option<&Direction>,
) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let n = coords.len();
    check(n, values, None)?;
    let tails = other.unwrap_or(values);
    check(n, tails, None)?;
    if !(lag > 0.0 && tolerance >= 0.0) {
        return invalid("lag must be positive and tolerance >= 0");
    }
    let cone = direction.map(|d| (d.unit(), d.tolerance.to_radians().cos()));
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| coords[a][0].total_cmp(&coords[b][0]));
    let xs: Vec<f64> = order.iter().map(|&i| coords[i][0]).collect();
    let reach = lag + tolerance;
    let pairs: Vec<(f64, f64)> = order
        .par_iter()
        .flat_map_iter(|&i| {
            let c = coords[i];
            let from = xs.partition_point(|&x| x < c[0] - reach);
            let to = xs.partition_point(|&x| x <= c[0] + reach);
            order[from..to].iter().filter_map(move |&j| {
                let h = [
                    coords[j][0] - c[0],
                    coords[j][1] - c[1],
                    coords[j][2] - c[2],
                ];
                let d = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
                if i == j || d == 0.0 || (d - lag).abs() > tolerance {
                    return None;
                }
                if let Some((u, cos)) = cone
                    && (h[0] * u.0 + h[1] * u.1 + h[2] * u.2) / d < cos
                {
                    return None;
                }
                let (head, tail) = (values[j], tails[i]);
                (!head.is_nan() && !tail.is_nan()).then_some((head, tail))
            })
        })
        .collect();
    let (head, tail): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    let r = pearson(&head, &tail, &vec![1.0; head.len()]);
    Ok((head, tail, r))
}

#[derive(Debug, Clone, Copy)]
pub enum Method {
    Pearson,
    /// Pearson on mid-point cumulative weights, ties averaged.
    Spearman,
    /// Weighted covariance, the variances on the diagonal.
    Covariance,
}

fn ranks(x: &[f64], w: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut out = vec![0.0; x.len()];
    let (mut cum, mut start) = (0.0, 0);
    while start < order.len() {
        let end = start + order[start..].partition_point(|&i| x[i] == x[order[start]]);
        let group: f64 = order[start..end].iter().map(|&i| w[i]).sum();
        order[start..end]
            .iter()
            .for_each(|&i| out[i] = cum + group / 2.0);
        cum += group;
        start = end;
    }
    out
}

/// Weighted correlation (or covariance) matrix of `columns`, each pair over
/// rows where both are not NaN.
pub fn correlation(
    columns: &[Vec<f64>],
    weights: Option<&[f64]>,
    method: Method,
) -> Result<Vec<Vec<f64>>> {
    let n = columns.first().map_or(0, Vec::len);
    for c in columns {
        check(n, c, weights)?;
    }
    let d = columns.len();
    let mut out = vec![vec![f64::NAN; d]; d];
    for a in 0..d {
        for b in a..d {
            let rows: Vec<usize> = (0..n)
                .filter(|&i| !columns[a][i].is_nan() && !columns[b][i].is_nan())
                .collect();
            let w: Vec<f64> = rows
                .iter()
                .map(|&i| weights.map_or(1.0, |w| w[i]))
                .collect();
            let mut x: Vec<f64> = rows.iter().map(|&i| columns[a][i]).collect();
            let mut y: Vec<f64> = rows.iter().map(|&i| columns[b][i]).collect();
            if let Method::Spearman = method {
                x = ranks(&x, &w);
                y = ranks(&y, &w);
            }
            let (cov, r) = moments(&x, &y, &w);
            out[a][b] = if let Method::Covariance = method {
                cov
            } else {
                r
            };
            out[b][a] = out[a][b];
        }
    }
    Ok(out)
}

/// Groups of samples at most `tolerance` apart, each in row order and ordered
/// by first row; samples alone are left out. Grouping is transitive: samples
/// further apart share a group when a chain of close samples links them. A
/// tolerance of 0 groups samples at exactly the same location.
pub fn duplicates(coords: &[[f64; 3]], tolerance: f64) -> Result<Vec<Vec<usize>>> {
    if tolerance.is_nan() || tolerance < 0.0 {
        return invalid("tolerance must be >= 0");
    }
    let cell = |p: &[f64; 3]| {
        p.map(|v| {
            if tolerance == 0.0 {
                (v + 0.0).to_bits() as i64
            } else {
                (v / tolerance).floor() as i64
            }
        })
    };
    let mut cells: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, p) in coords.iter().enumerate() {
        cells.entry(cell(p)).or_default().push(i);
    }
    let reach = if tolerance == 0.0 { 0 } else { 1 };
    let mut parent: Vec<usize> = (0..coords.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (i, p) in coords.iter().enumerate() {
        let c = cell(p);
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    let near = [
                        c[0].saturating_add(dx),
                        c[1].saturating_add(dy),
                        c[2].saturating_add(dz),
                    ];
                    for &j in cells.get(&near).into_iter().flatten().filter(|&&j| j < i) {
                        let d2: f64 = (0..3).map(|k| (p[k] - coords[j][k]).powi(2)).sum();
                        if d2 <= tolerance * tolerance {
                            let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                            parent[a.max(b)] = a.min(b);
                        }
                    }
                }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..coords.len() {
        groups.entry(root(&mut parent, i)).or_default().push(i);
    }
    Ok(groups.into_values().filter(|g| g.len() > 1).collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Merge {
    /// Weighted mean of the group's values.
    Mean,
    /// The first row's value.
    First,
    /// The group's largest value.
    Max,
}

/// Values left after merging each of `groups` (as from [`duplicates`]) into
/// its first row, in row order. Mean and max skip NaN; mean is weighted by
/// `weights`, by count without.
pub fn merge_duplicates(
    values: &[f64],
    groups: &[Vec<usize>],
    merge: Merge,
    weights: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let n = values.len();
    check(n, values, weights)?;
    let weight = |i: usize| weights.map_or(1.0, |w| w[i]);
    let mut out = values.to_vec();
    let mut dropped = vec![false; n];
    for g in groups {
        if g.is_empty() || g.iter().any(|&i| i >= n || dropped[i]) {
            return invalid("groups must be disjoint, non-empty and within the rows");
        }
        let valid = || g.iter().copied().filter(|&i| !values[i].is_nan());
        out[g[0]] = match merge {
            Merge::First => values[g[0]],
            Merge::Max => valid().map(|i| values[i]).fold(f64::NAN, f64::max),
            Merge::Mean => {
                valid().map(|i| weight(i) * values[i]).sum::<f64>()
                    / valid().map(weight).sum::<f64>()
            }
        };
        g[1..].iter().for_each(|&i| dropped[i] = true);
    }
    Ok((0..n).filter(|&i| !dropped[i]).map(|i| out[i]).collect())
}

/// Pairs `(i, j, distance)` of `a[i]` and its nearest `b[j]` at most
/// `max_distance` away, sorted by `i`, e.g. twin holes or two drilling types.
/// Samples with a NaN in `values` are left out, and so are pairs within one
/// hole given `holes` codes shared by `a` and `b`. With `unique`, each sample
/// pairs at most once, the closest pairs first, ties by `(i, j)`; without,
/// a sample of `b` may be the nearest of several of `a`.
pub fn pairs(
    a: &[[f64; 3]],
    b: &[[f64; 3]],
    max_distance: f64,
    values: Option<(&[f64], &[f64])>,
    holes: Option<(&[u32], &[u32])>,
    unique: bool,
) -> Result<Vec<(usize, usize, f64)>> {
    if !(max_distance >= 0.0 && max_distance.is_finite()) {
        return invalid("max_distance must be finite and >= 0");
    }
    if let Some((va, vb)) = values {
        check(a.len(), va, None)?;
        check(b.len(), vb, None)?;
    }
    if let Some((ha, hb)) = holes
        && (ha.len() != a.len() || hb.len() != b.len())
    {
        return invalid("holes need one code per sample of a and of b");
    }
    let valid = |v: Option<&[f64]>, i: usize| v.is_none_or(|v| !v[i].is_nan());
    let rows: Vec<usize> = (0..b.len())
        .filter(|&j| valid(values.map(|v| v.1), j))
        .collect();
    if rows.is_empty() {
        return Ok(vec![]);
    }
    let points: Vec<[f64; 3]> = rows.iter().map(|&j| b[j]).collect();
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&points)
        .map_err(|e| EdaError::InvalidInput(format!("{e:?}")))?;
    let radius2 = (max_distance * (1.0 + 1e-9)).powi(2);
    let near: Vec<Vec<(f64, usize, usize)>> = (0..a.len())
        .into_par_iter()
        .map(|i| {
            if !valid(values.map(|v| v.0), i) {
                return vec![];
            }
            let mut near: Vec<(f64, usize, usize)> = tree
                .query(&a[i])
                .within::<SquaredEuclidean<f64>>(radius2)
                .execute()
                .iter()
                .map(|r| rows[r.item as usize])
                .filter(|&j| holes.is_none_or(|(ha, hb)| ha[i] != hb[j]))
                .map(|j| (distance(a[i], b[j]), i, j))
                .filter(|p| p.0 <= max_distance)
                .collect();
            near.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.2.cmp(&q.2)));
            near
        })
        .collect();
    let mut out: Vec<(f64, usize, usize)> = if unique {
        let mut all: Vec<(f64, usize, usize)> = near.into_iter().flatten().collect();
        all.sort_by(|p, q| p.0.total_cmp(&q.0).then((p.1, p.2).cmp(&(q.1, q.2))));
        let (mut used_a, mut used_b) = (vec![false; a.len()], vec![false; b.len()]);
        all.into_iter()
            .filter(|&(_, i, j)| {
                let free = !used_a[i] && !used_b[j];
                if free {
                    (used_a[i], used_b[j]) = (true, true);
                }
                free
            })
            .collect()
    } else {
        near.into_iter()
            .filter_map(|n| n.first().copied())
            .collect()
    };
    out.sort_by_key(|p| p.1);
    Ok(out.into_iter().map(|(d, i, j)| (i, j, d)).collect())
}

/// How [`spacing`] measures the drilling around a target.
#[derive(Debug, Clone, Copy)]
pub enum Spacing<'a> {
    /// Composites of length `composite_length` inside the ellipsoid of
    /// `radius` in the metric of `anisotropy` (a sphere without it).
    Volume {
        anisotropy: Option<&'a variogram::Anisotropy>,
        radius: f64,
        composite_length: f64,
    },
    /// Plan distances to the `n`th and `n + 1`th nearest samples, per `n`.
    Plan(&'a [usize]),
}

/// Volume of the ellipsoid of `radius` in the metric of `anisotropy`.
pub fn ellipsoid_volume(anisotropy: Option<&variogram::Anisotropy>, radius: f64) -> f64 {
    let axes = anisotropy.map_or(1.0, |a| a.angles.major * a.angles.semi * a.angles.minor);
    4.0 / 3.0 * std::f64::consts::PI * radius.powi(3) * axes
}

/// Equivalent data spacing at each target (Cabral Pinto & Deutsch, 2017).
/// [`Spacing::Volume`] gives `sqrt(V / (c n))`, with `n` the samples inside
/// the ellipsoid of volume `V` around the target and `c` the composite length.
/// [`Spacing::Plan`] gives the mean over `n` of `sqrt(pi r^2 / n)`, with `r`
/// the mean plan distance to the `n`th and `n + 1`th nearest samples, or to
/// holes with `holes` (one code per sample, a hole at its nearest sample).
/// NaN where the ellipsoid is empty or fewer than `n + 1` are found.
pub fn spacing(
    data: &[[f64; 3]],
    targets: &[[f64; 3]],
    how: Spacing,
    holes: Option<&[u32]>,
) -> Result<Vec<f64>> {
    if data.iter().chain(targets).flatten().any(|v| !v.is_finite()) {
        return invalid("coordinates must be finite");
    }
    if holes.is_some_and(|h| h.len() != data.len()) {
        return invalid("holes need one code per sample");
    }
    let origin = data.first().copied().unwrap_or_default();
    match how {
        Spacing::Volume {
            anisotropy,
            radius,
            composite_length,
        } => {
            if !(radius > 0.0 && radius.is_finite()) {
                return invalid("radius must be finite and > 0");
            }
            if !(composite_length > 0.0 && composite_length.is_finite()) {
                return invalid("composite_length must be finite and > 0");
            }
            if holes.is_some() {
                return invalid("holes apply to the plan form only");
            }
            if data.is_empty() {
                return Ok(vec![f64::NAN; targets.len()]);
            }
            let frame = anisotropy.map(variogram::Anisotropy::matrix);
            let local = |p: &[f64; 3]| {
                let d = [0, 1, 2].map(|k| p[k] - origin[k]);
                frame.map_or(d, |m| {
                    [0, 1, 2].map(|r| (0..3).map(|c| m[(r, c)] * d[c]).sum())
                })
            };
            let points: Vec<[f64; 3]> = data.iter().map(local).collect();
            let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&points)
                .map_err(|e| EdaError::InvalidInput(format!("{e:?}")))?;
            let volume = ellipsoid_volume(anisotropy, radius);
            Ok(targets
                .par_iter()
                .map(|t| {
                    let n = tree
                        .query(&local(t))
                        .within::<SquaredEuclidean<f64>>(radius * radius)
                        .execute()
                        .len();
                    if n == 0 {
                        f64::NAN
                    } else {
                        (volume / (composite_length * n as f64)).sqrt()
                    }
                })
                .collect())
        }
        Spacing::Plan(ns) => {
            if ns.is_empty() || ns.contains(&0) {
                return invalid("n must be at least 1");
            }
            let want = ns.iter().max().expect("not empty") + 1;
            if data.is_empty() {
                return Ok(vec![f64::NAN; targets.len()]);
            }
            let flat = |p: &[f64; 3]| [p[0] - origin[0], p[1] - origin[1], 0.0];
            let points: Vec<[f64; 3]> = data.iter().map(flat).collect();
            let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&points)
                .map_err(|e| EdaError::InvalidInput(format!("{e:?}")))?;
            Ok(targets
                .par_iter()
                .map(|t| {
                    let d = nearest(&tree, &flat(t), want, holes, data.len());
                    if d.len() < want {
                        return f64::NAN;
                    }
                    let total: f64 = ns
                        .iter()
                        .map(|&n| {
                            let r = (d[n - 1] + d[n]) / 2.0;
                            (std::f64::consts::PI * r * r / n as f64).sqrt()
                        })
                        .sum();
                    total / ns.len() as f64
                })
                .collect())
        }
    }
}

/// Sorted distances to the `want` nearest of `total` samples, or with `holes`
/// to the nearest sample of each of the `want` nearest holes; fewer if short.
fn nearest(
    tree: &ImmutableKdTree<f64, 3>,
    at: &[f64; 3],
    want: usize,
    holes: Option<&[u32]>,
    total: usize,
) -> Vec<f64> {
    let mut k = want.min(total);
    loop {
        let found = tree
            .query(at)
            .nearest_n::<SquaredEuclidean<f64>>(std::num::NonZero::new(k).expect("data not empty"))
            .execute();
        let Some(holes) = holes else {
            return found.iter().map(|r| r.distance.sqrt()).collect();
        };
        let mut seen = std::collections::HashSet::new();
        let d: Vec<f64> = found
            .iter()
            .filter(|r| seen.insert(holes[r.item as usize]))
            .map(|r| r.distance.sqrt())
            .take(want)
            .collect();
        if d.len() == want || k == total {
            return d;
        }
        k = (2 * k).min(total);
    }
}

/// Paired values in one bin `[from, to)` of pairing distance, the last bin
/// closed; means and bias are NaN when the bin is empty.
#[derive(Debug, Clone)]
pub struct Bias {
    pub from: f64,
    pub to: f64,
    pub n: usize,
    pub mean_a: f64,
    pub mean_b: f64,
    /// `mean_b / mean_a - 1`.
    pub bias: f64,
}

/// Means of paired values `a` and `b` and their relative bias per bin of
/// `distance` between consecutive `edges`; pairs with a NaN are skipped.
pub fn paired_bias(distance: &[f64], a: &[f64], b: &[f64], edges: &[f64]) -> Result<Vec<Bias>> {
    let n = distance.len();
    check(n, a, None)?;
    check(n, b, None)?;
    if edges.len() < 2 || !edges.iter().all(|e| e.is_finite()) || !edges.is_sorted_by(|x, y| x < y)
    {
        return invalid("edges must be at least 2 finite increasing values");
    }
    let bins = edges.len() - 1;
    let mut sums = vec![(0usize, 0.0, 0.0); bins];
    for i in 0..n {
        let d = distance[i];
        if a[i].is_nan() || b[i].is_nan() || !(edges[0] <= d && d <= edges[bins]) {
            continue;
        }
        let k = (edges.partition_point(|&e| e <= d) - 1).min(bins - 1);
        sums[k].0 += 1;
        sums[k].1 += a[i];
        sums[k].2 += b[i];
    }
    Ok(sums
        .into_iter()
        .enumerate()
        .map(|(k, (n, sa, sb))| {
            let (mean_a, mean_b) = (sa / n as f64, sb / n as f64);
            Bias {
                from: edges[k],
                to: edges[k + 1],
                n,
                mean_a,
                mean_b,
                bias: mean_b / mean_a - 1.0,
            }
        })
        .collect())
}

/// Spacing bins of [`uncertainty_curve`].
#[derive(Debug, Clone)]
pub enum Bins {
    /// Freedman–Diaconis width over the spacing range.
    Auto,
    /// Equal-width bins over the spacing range.
    Count(usize),
    Edges(Vec<f64>),
}

/// One spacing bin `[from, to)` (the last closed) of [`uncertainty_curve`].
#[derive(Debug, Clone)]
pub struct CurveBin {
    /// Mean spacing in the bin.
    pub spacing: f64,
    pub n: usize,
    /// Uncertainty quantiles, non-decreasing from bin to bin.
    pub quantiles: Vec<f64>,
    /// Share of uncertainty at or below the threshold.
    pub share: f64,
}

fn edges_of(x: &[f64], bins: &Bins) -> Result<Vec<f64>> {
    let (lo, hi) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let hi = if hi > lo { hi } else { lo + 1.0 };
    let count = match bins {
        Bins::Edges(e) => return Ok(e.clone()),
        Bins::Count(0) => return invalid("bins must be positive"),
        Bins::Count(k) => *k,
        Bins::Auto => {
            let q = quantiles_of(x, &vec![1.0; x.len()], &[0.25, 0.75])?;
            let width = 2.0 * (q[1] - q[0]) / (x.len() as f64).cbrt();
            if width > 0.0 {
                ((hi - lo) / width).ceil().clamp(1.0, x.len() as f64) as usize
            } else {
                1
            }
        }
    };
    Ok((0..=count)
        .map(|i| lo + (hi - lo) * i as f64 / count as f64)
        .collect())
}

/// Uncertainty quantiles at `probabilities` and the share at or below
/// `threshold` per bin of `spacing`; pairs with a non-finite value are skipped,
/// bins with fewer than `min_count` pairs dropped, and each quantile is made
/// non-decreasing in spacing by a running maximum.
pub fn uncertainty_curve(
    spacing: &[f64],
    uncertainty: &[f64],
    bins: &Bins,
    probabilities: &[f64],
    threshold: f64,
    min_count: usize,
) -> Result<Vec<CurveBin>> {
    if spacing.len() != uncertainty.len() {
        return invalid(format!(
            "expected {} uncertainty values, got {}",
            spacing.len(),
            uncertainty.len()
        ));
    }
    if !threshold.is_finite() {
        return invalid("threshold must be finite");
    }
    if probabilities.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return invalid("probabilities must be in [0, 1]");
    }
    let (x, y): (Vec<f64>, Vec<f64>) = spacing
        .iter()
        .zip(uncertainty)
        .filter(|(s, u)| s.is_finite() && u.is_finite())
        .unzip();
    if x.is_empty() {
        return invalid("no finite spacing and uncertainty pairs");
    }
    let edges = edges_of(&x, bins)?;
    if edges.len() < 2 || !edges.iter().all(|e| e.is_finite()) || !edges.is_sorted_by(|a, b| a < b)
    {
        return invalid("edges must be at least 2 finite increasing values");
    }
    let last = edges.len() - 2;
    let mut members = vec![Vec::new(); last + 1];
    for (i, &s) in x.iter().enumerate() {
        if edges[0] <= s && s <= edges[last + 1] {
            members[(edges.partition_point(|&e| e <= s) - 1).min(last)].push(i);
        }
    }
    let mut curve = Vec::new();
    let mut top = vec![f64::NEG_INFINITY; probabilities.len()];
    for m in members.into_iter().filter(|m| m.len() >= min_count.max(1)) {
        let n = m.len();
        let u: Vec<f64> = m.iter().map(|&i| y[i]).collect();
        let mut quantiles = quantiles_of(&u, &vec![1.0; n], probabilities)?;
        for (q, t) in quantiles.iter_mut().zip(&mut top) {
            *t = t.max(*q);
            *q = *t;
        }
        curve.push(CurveBin {
            spacing: m.iter().map(|&i| x[i]).sum::<f64>() / n as f64,
            n,
            quantiles,
            share: u.iter().filter(|&&v| v <= threshold).count() as f64 / n as f64,
        });
    }
    Ok(curve)
}

/// Largest spacing whose `values` still meet `threshold`: linear between the
/// two points around the first rise above it; NaN when the first point is
/// already above. When the curve never rises above, the last spacing and
/// `true`, as the required spacing lies beyond the tested range. Points with
/// a NaN are skipped.
pub fn required_spacing(spacing: &[f64], values: &[f64], threshold: f64) -> (f64, bool) {
    let points: Vec<(f64, f64)> = spacing
        .iter()
        .zip(values)
        .filter(|(s, v)| !s.is_nan() && !v.is_nan())
        .map(|(&s, &v)| (s, v))
        .collect();
    let crossing = match points.iter().position(|&(_, v)| v > threshold) {
        None => return points.last().map_or((f64::NAN, false), |p| (p.0, true)),
        Some(0) => f64::NAN,
        Some(j) => {
            let ((x0, y0), (x1, y1)) = (points[j - 1], points[j]);
            x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)
        }
    };
    (crossing, false)
}

/// One cell of [`domain_change`]: the blocks of class `from` in the first
/// model and `to` in the second.
#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub from: usize,
    pub to: usize,
    pub tonnage: f64,
    /// Sum of tonnage × grade; 0 without grades.
    pub metal: f64,
    /// `metal` over the tonnage of blocks with a grade; NaN when none has one.
    pub mean_grade: f64,
}

/// Cross-tabulation of two classifications of the same blocks into `k`
/// classes: the tonnage (weights, 1 by default) and metal moving from each
/// class of `before` to each of `after`, `k × k` cells by `from` then `to`.
/// Blocks unclassed (`None`) in either model are left out; NaN grades add
/// tonnage but no metal.
pub fn domain_change(
    before: &[Option<u32>],
    after: &[Option<u32>],
    k: usize,
    tonnes: Option<&[f64]>,
    grades: Option<&[f64]>,
) -> Result<Vec<Change>> {
    let n = before.len();
    if after.len() != n {
        return invalid(format!("expected {n} classes in both models"));
    }
    if grades.is_some_and(|g| g.len() != n) {
        return invalid(format!("expected {n} grades"));
    }
    check(n, grades.unwrap_or(&vec![0.0; n]), tonnes)?;
    if before
        .iter()
        .chain(after)
        .flatten()
        .any(|&c| c as usize >= k)
    {
        return invalid(format!("classes must be codes 0 to {}", k.max(1) - 1));
    }
    let mut cells = vec![[0.0; 3]; k * k];
    for i in 0..n {
        let (Some(a), Some(b)) = (before[i], after[i]) else {
            continue;
        };
        let t = tonnes.map_or(1.0, |t| t[i]);
        let cell = &mut cells[a as usize * k + b as usize];
        cell[0] += t;
        if let Some(g) = grades.map(|g| g[i]).filter(|g| !g.is_nan()) {
            cell[1] += t * g;
            cell[2] += t;
        }
    }
    Ok(cells
        .into_iter()
        .enumerate()
        .map(|(c, [tonnage, metal, graded])| Change {
            from: c / k,
            to: c % k,
            tonnage,
            metal,
            mean_grade: if graded > 0.0 {
                metal / graded
            } else {
                f64::NAN
            },
        })
        .collect())
}

/// One cell of [`transition_matrix`]: `count` samples of class `from` with a
/// sample of class `to` at `lag ± tolerance` deeper in the same hole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    pub from: u32,
    pub to: u32,
    pub count: u64,
}

/// Along-hole category transition matrix: samples are grouped by hole and
/// sorted by `depth`, then for every sample the classes of samples `lag ±
/// tolerance` deeper in the same hole are tallied, `from` (the shallower
/// sample) then `to` (the deeper one), into a dense `k × k` matrix.
pub fn transition_matrix(
    depth: &[f64],
    categories: &[u32],
    holes: &[u32],
    k: usize,
    lag: f64,
    tolerance: f64,
) -> Result<Vec<Transition>> {
    let n = depth.len();
    if categories.len() != n || holes.len() != n {
        return invalid(format!("expected {n} categories and holes"));
    }
    if k == 0 {
        return invalid("k must be positive");
    }
    if !(lag > 0.0 && tolerance >= 0.0) {
        return invalid("lag must be positive and tolerance >= 0");
    }
    if depth.iter().any(|d| !d.is_finite()) {
        return invalid("depth must be finite");
    }
    if categories.iter().any(|&c| c as usize >= k) {
        return invalid(format!("classes must be codes 0 to {}", k - 1));
    }
    let mut by_hole: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        by_hole.entry(holes[i]).or_default().push(i);
    }
    let mut counts = vec![0u64; k * k];
    for members in by_hole.values_mut() {
        members.sort_by(|&a, &b| depth[a].total_cmp(&depth[b]));
        let ds: Vec<f64> = members.iter().map(|&i| depth[i]).collect();
        for &i in members.iter() {
            let lo = ds.partition_point(|&d| d < depth[i] + lag - tolerance);
            let hi = ds.partition_point(|&d| d <= depth[i] + lag + tolerance);
            for &j in &members[lo..hi] {
                if j == i {
                    continue;
                }
                counts[categories[i] as usize * k + categories[j] as usize] += 1;
            }
        }
    }
    Ok(counts
        .into_iter()
        .enumerate()
        .map(|(c, count)| Transition {
            from: (c / k) as u32,
            to: (c % k) as u32,
            count,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_matrix_counts_and_margins() {
        let depth: Vec<f64> = (0..10).map(f64::from).collect();
        let categories: Vec<u32> = vec![0, 0, 1, 1, 2, 2, 0, 0, 1, 1];
        let holes = vec![0u32; 10];
        let t = transition_matrix(&depth, &categories, &holes, 3, 1.0, 0.0).unwrap();
        assert_eq!(t.len(), 9);
        let count = |from: u32, to: u32| {
            t.iter()
                .find(|x| x.from == from && x.to == to)
                .unwrap()
                .count
        };
        assert_eq!(count(0, 0), 2);
        assert_eq!(count(0, 1), 2);
        assert_eq!(count(0, 2), 0);
        assert_eq!(count(1, 0), 0);
        assert_eq!(count(1, 1), 2);
        assert_eq!(count(1, 2), 1);
        assert_eq!(count(2, 0), 1);
        assert_eq!(count(2, 1), 0);
        assert_eq!(count(2, 2), 1);
        // Row `c` sums to how often `c` occurs with a deeper neighbor in reach
        // (every occurrence but the last sample of the hole); columns, with a
        // shallower one (every occurrence but the first).
        for c in 0..3u32 {
            let occurrences = categories.iter().filter(|&&x| x == c).count() as u64;
            let row: u64 = t.iter().filter(|x| x.from == c).map(|x| x.count).sum();
            assert_eq!(
                row,
                occurrences - u64::from(*categories.last().unwrap() == c)
            );
            let col: u64 = t.iter().filter(|x| x.to == c).map(|x| x.count).sum();
            assert_eq!(col, occurrences - u64::from(categories[0] == c));
        }
    }

    #[test]
    fn transition_matrix_rejects_bad_input() {
        assert!(transition_matrix(&[0.0, 1.0], &[0, 1], &[0, 0], 0, 1.0, 0.0).is_err());
        assert!(transition_matrix(&[0.0, 1.0], &[0, 1], &[0, 0], 2, 0.0, 0.0).is_err());
        assert!(transition_matrix(&[0.0, 1.0], &[0, 1], &[0, 0], 2, 1.0, -0.1).is_err());
        assert!(transition_matrix(&[0.0, 1.0], &[0, 2], &[0, 0], 2, 1.0, 0.0).is_err());
        assert!(transition_matrix(&[0.0], &[0, 1], &[0, 0], 2, 1.0, 0.0).is_err());
    }

    #[test]
    fn domain_change_margins_diagonal_and_metal() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(9);
        let n = 500;
        let before: Vec<Option<u32>> = (0..n).map(|_| Some(rng.gen_range(0..3))).collect();
        let after: Vec<Option<u32>> = before
            .iter()
            .map(|b| match rng.gen_bool(0.2) {
                true => Some(rng.gen_range(0..3)),
                false => *b,
            })
            .collect();
        let tonnes: Vec<f64> = (0..n).map(|_| rng.gen_range(1.0..3.0)).collect();
        let grades: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..5.0)).collect();
        let cells = domain_change(&before, &after, 3, Some(&tonnes), Some(&grades)).unwrap();
        assert_eq!(cells.len(), 9);
        let class = |m: &[Option<u32>], c: u32| -> f64 {
            (0..n).filter(|&i| m[i] == Some(c)).map(|i| tonnes[i]).sum()
        };
        for c in 0..3 {
            let row: f64 = cells
                .iter()
                .filter(|x| x.from == c)
                .map(|x| x.tonnage)
                .sum();
            let col: f64 = cells.iter().filter(|x| x.to == c).map(|x| x.tonnage).sum();
            assert!(close(row, class(&before, c as u32)));
            assert!(close(col, class(&after, c as u32)));
        }
        let unchanged: f64 = (0..n)
            .filter(|&i| before[i] == after[i])
            .map(|i| tonnes[i])
            .sum();
        let diagonal: f64 = cells
            .iter()
            .filter(|x| x.from == x.to)
            .map(|x| x.tonnage)
            .sum();
        assert!(close(unchanged, diagonal));
        let metal: f64 = tonnes.iter().zip(&grades).map(|(t, g)| t * g).sum();
        assert!(close(cells.iter().map(|x| x.metal).sum(), metal));
        let c = &cells[4];
        assert!(close(c.mean_grade * c.tonnage, c.metal));
    }

    #[test]
    fn domain_change_skips_unclassed_and_rejects_bad_codes() {
        let cells = domain_change(&[Some(0), None], &[Some(1), Some(1)], 2, None, None).unwrap();
        assert_eq!(cells[1].tonnage, 1.0);
        assert!(cells[1].mean_grade.is_nan());
        assert_eq!(cells.iter().map(|c| c.tonnage).sum::<f64>(), 1.0);
        assert!(domain_change(&[Some(2)], &[Some(0)], 2, None, None).is_err());
        assert!(domain_change(&[Some(0)], &[], 2, None, None).is_err());
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn weight_acts_as_duplication() {
        let p = [0.0, 0.5, 1.0];
        let weighted =
            describe(&[3.0, 1.0, f64::NAN, 7.0], Some(&[2.0, 1.0, 5.0, 1.0]), &p).unwrap();
        let repeated = describe(&[3.0, 3.0, 1.0, 7.0], None, &p).unwrap();
        assert_eq!(weighted.n, 3);
        assert!(close(weighted.mean, repeated.mean) && close(weighted.variance, repeated.variance));
        for (a, b) in weighted.quantiles.iter().zip(&repeated.quantiles) {
            assert!(close(*a, *b));
        }
        let (m, v) = transforms::weighted_mean_variance(&[3.0, 3.0, 1.0, 7.0], None).unwrap();
        assert!(close(weighted.mean, m) && close(weighted.variance, v));
    }

    #[test]
    fn quantiles_are_mid_point() {
        let v: Vec<f64> = (1..=10).map(f64::from).collect();
        let q = quantiles(&v, None, &[0.0, 0.1, 0.5, 1.0]).unwrap();
        assert_eq!(q, vec![1.0, 1.5, 5.5, 10.0]);
    }

    #[test]
    fn swath_follows_a_trend() {
        let coords: Vec<[f64; 3]> = (0..100).map(|i| [i as f64 + 0.5, 3.0, 0.0]).collect();
        let values: Vec<f64> = coords.iter().map(|c| c[0]).collect();
        let s = swath(&coords, &values, None, None, 10.0, Along::Azimuth(90.0)).unwrap();
        assert_eq!(s.count, vec![10; 10]);
        for (c, m) in s.centers.iter().zip(&s.mean) {
            assert!(close(*c, *m));
        }
        let y = swath(&coords, &values, None, None, 10.0, Along::Axis(1)).unwrap();
        assert_eq!((y.centers, y.count), (vec![5.0], vec![100]));
    }

    #[test]
    fn contact_splits_sides() {
        let coords: Vec<[f64; 3]> = (0..40).map(|i| [0.0, 0.0, (i % 20) as f64]).collect();
        let domains: Vec<u32> = (0..40)
            .map(|i| u32::from(i % 20 >= 10) + 2 * u32::from(i >= 30))
            .collect();
        let holes: Vec<u32> = (0..40).map(|i| i / 20).collect();
        let values: Vec<f64> = domains.iter().map(|&d| 1.0 + 4.0 * f64::from(d)).collect();
        let c = contact(&coords, &values, &domains, &holes, 0, 1, 100.0, 2.0).unwrap();
        for (d, m) in c.centers.iter().zip(&c.mean) {
            assert_eq!(*m, if *d < 0.0 { 1.0 } else { 5.0 });
        }
        assert_eq!(c.count.iter().sum::<usize>(), 20);
        assert_eq!(c.centers.first(), Some(&-11.0));
    }

    #[test]
    fn contact_bins_stop_at_max_distance() {
        let coords: Vec<[f64; 3]> = (0..20).map(|i| [0.0, 0.0, f64::from(i)]).collect();
        let domains: Vec<u32> = (0..20).map(|i| u32::from(i >= 10)).collect();
        let values = vec![1.0; 20];
        let c = contact(&coords, &values, &domains, &[0; 20], 0, 1, 8.0, 1.0).unwrap();
        assert_eq!(c.count.iter().sum::<usize>(), 16);
        assert_eq!((c.centers[0], c.count[0]), (-7.5, 2));
        assert_eq!(
            (*c.centers.last().unwrap(), *c.count.last().unwrap()),
            (7.5, 2)
        );
        let c = contact(&coords, &values, &domains, &[0; 20], 0, 1, 6.5, 1.0).unwrap();
        assert_eq!(c.count.iter().sum::<usize>(), 12);
        assert_eq!((c.centers[0], *c.centers.last().unwrap()), (-6.25, 6.25));
    }

    #[test]
    fn soft_boundary_folds_in_nearby_other_domain_samples() {
        let coords: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ];
        let domains = vec![0u32, 0, 1, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let r = soft_boundary(&coords, &values, &domains, None, 0, 6.0, &[]).unwrap();
        assert_eq!(r.added, vec![true, false]);
        let (hard, soft) = (&r.rows[0].1, &r.rows[1].1);
        assert_eq!(r.rows.len(), 2);
        assert_eq!(hard.n, 2);
        assert_eq!(soft.n, hard.n + r.added.iter().filter(|&&a| a).count());
        assert!(close(soft.mean, (1.0 + 2.0 + 3.0) / 3.0));
        assert!(soft_boundary(&coords, &values, &domains, None, 0, 0.0, &[]).is_err());
        assert!(soft_boundary(&coords, &values, &domains, None, 9, 6.0, &[]).is_err());
    }

    #[test]
    fn capping_at_the_maximum_removes_nothing() {
        let v = [1.0, 2.0, 3.0, 10.0];
        let caps = capping(&v, None, Some(&[10.0, 3.0])).unwrap();
        assert!(close(caps[0].metal_removed, 0.0) && close(caps[0].fraction, 0.0));
        assert!(close(caps[1].metal_removed, 1.0 - 9.0 / 16.0) && close(caps[1].fraction, 0.25));
        assert_eq!(
            capping(&v, None, None).unwrap().len(),
            CAP_PROBABILITIES.len()
        );
    }

    fn skewed() -> (Vec<f64>, Vec<u32>, Vec<f64>) {
        let v: Vec<f64> = (0..60)
            .map(|i| {
                if i == 7 {
                    f64::NAN
                } else {
                    ((i * 37 % 60) as f64 / 12.0).exp()
                }
            })
            .collect();
        let c: Vec<u32> = (0..60).map(|i| (i % 3) as u32).collect();
        let w: Vec<f64> = (0..60).map(|i| 0.5 + (i % 5) as f64).collect();
        (v, c, w)
    }

    #[test]
    fn describe_by_ends_with_all_data() {
        let (v, c, w) = skewed();
        let p = [0.1, 0.5, 0.9];
        let rows = describe_by(&v, &c, Some(&w), &p).unwrap();
        assert_eq!(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            [Some(0), Some(1), Some(2), None]
        );
        let (all, whole) = (&rows[3].1, describe(&v, Some(&w), &p).unwrap());
        assert_eq!(all.n, whole.n);
        assert_eq!(
            (all.mean, all.variance, &all.quantiles),
            (whole.mean, whole.variance, &whole.quantiles)
        );
        assert_eq!(rows[..3].iter().map(|r| r.1.n).sum::<usize>(), whole.n);
    }

    #[test]
    fn grade_tonnage_from_data() {
        let (v, _, w) = skewed();
        let d = vec![2.5; v.len()];
        let mut cutoffs = vec![f64::NEG_INFINITY];
        cutoffs.extend((0..20).map(|i| i as f64 * 8.0));
        let gt = grade_tonnage(&v, Some(&w), Some(&d), &cutoffs).unwrap();
        let s = describe(&v, Some(&w), &[]).unwrap();
        let total: f64 = v
            .iter()
            .zip(&w)
            .filter(|(v, _)| !v.is_nan())
            .map(|(_, w)| 2.5 * w)
            .sum();
        assert!(close(gt[0].tonnage, total) && close(gt[0].mean_grade, s.mean));
        for r in &gt {
            assert!(r.tonnage == 0.0 || close(r.metal, r.tonnage * r.mean_grade));
        }
        assert!(gt.windows(2).all(|p| p[1].tonnage <= p[0].tonnage));
        assert!(gt.last().unwrap().mean_grade.is_nan());
        assert!(grade_tonnage(&v, None, None, &[f64::NAN]).is_err());
    }

    #[test]
    fn compare_models_balances_metal() {
        let (v, c, w) = skewed();
        let scaled: Vec<f64> = v.iter().map(|x| 1.2 * x).collect();
        let cutoffs = [f64::NEG_INFINITY, 2.0, 10.0];
        let rows = compare_models(&[&v, &scaled], Some(&c), Some(&w), &cutoffs, 0).unwrap();
        assert_eq!(rows.len(), 4 * 3 * 2);
        let by = grade_tonnage_by(&v, &c, Some(&w), None, &cutoffs).unwrap();
        for (i, r) in rows.iter().enumerate() {
            let (row, k) = (i / 6, i / 2 % 3);
            assert_eq!((r.model, r.category), (i % 2, by[row].0));
            if r.model == 0 {
                assert_eq!(r.tonnage.metal, by[row].1[k].metal);
                assert_eq!(
                    (r.tonnage_diff, r.grade_diff, r.metal_diff),
                    (0.0, 0.0, 0.0)
                );
            } else if k == 0 {
                assert!(close(r.tonnage_diff, 0.0) && close(r.metal_diff, 0.2));
            }
        }
        for k in 0..3 {
            let sum = |f: fn(&Tonnage) -> f64| by[..3].iter().map(|r| f(&r.1[k])).sum::<f64>();
            assert!(close(sum(|t| t.tonnage), by[3].1[k].tonnage));
            assert!(close(sum(|t| t.metal), by[3].1[k].metal));
        }
        let all = grade_tonnage(&v, Some(&w), None, &cutoffs).unwrap();
        assert!(close(all[1].metal, by[3].1[1].metal));
        assert!(compare_models(&[&v, &v[1..]], None, None, &cutoffs, 0).is_err());
        assert!(compare_models(&[&v], None, None, &cutoffs, 1).is_err());
    }

    #[test]
    fn swath_balances_grade_tonnage() {
        let (v, _, w) = skewed();
        let coords: Vec<[f64; 3]> = (0..60).map(|i| [0.0, 0.0, i as f64]).collect();
        let d: Vec<f64> = (0..60).map(|i| 2.0 + (i % 4) as f64 / 10.0).collect();
        let s = swath(&coords, &v, Some(&w), Some(&d), 7.0, Along::Axis(2)).unwrap();
        let gt = &grade_tonnage(&v, Some(&w), Some(&d), &[f64::NEG_INFINITY]).unwrap()[0];
        assert!(close(s.tonnage.iter().sum(), gt.tonnage));
        assert!(close(s.metal.iter().sum(), gt.metal));
    }

    #[test]
    fn model_equal_to_data_validates() {
        let (v, c, w) = skewed();
        let rows =
            validate_model(&v, &v, Some(&w), Some((&c, &c)), Some(&w), Some(&v), &[0.5]).unwrap();
        assert_eq!(rows.len(), 4 * 4);
        assert_eq!(rows.last().unwrap().domain, None);
        for r in rows.iter().filter(|r| r.source != Source::Naive) {
            assert!(close(r.mean_diff, 0.0) && close(r.variance_ratio, 1.0));
            assert!(r.source == Source::Declustered || close(r.tonnage, r.summary.weight));
        }
        let all = validate_model(&v, &v, None, None, None, None, &[]).unwrap();
        assert_eq!(
            all.iter().map(|r| r.source).collect::<Vec<_>>(),
            [Source::Naive, Source::Model]
        );
        assert!(close(all[1].tonnage, 59.0) && close(all[1].mean_diff, 0.0));
    }

    #[test]
    fn capping_report_metal_removed() {
        let (v, c, w) = skewed();
        let caps = [20.0, f64::INFINITY, 5.0];
        let rows = capping_report(&v, &c, Some(&w), &caps).unwrap();
        let mut total = 0.0;
        for (k, r) in rows[..3].iter().enumerate() {
            let removed: f64 = (0..v.len())
                .filter(|&i| c[i] == k as u32 && v[i] > caps[k])
                .map(|i| (v[i] - caps[k]) * w[i])
                .sum();
            assert!(close(r.1.metal_removed, removed));
            let sw: f64 = (0..v.len())
                .filter(|&i| c[i] == k as u32 && !v[i].is_nan())
                .map(|i| w[i])
                .sum();
            assert!(close(r.1.before.mean - r.1.after.mean, removed / sw));
            total += removed;
        }
        assert!(close(rows[3].1.metal_removed, total) && rows[3].1.cap.is_nan());
        assert_eq!(rows[1].1.capped, 0);
        assert_eq!(rows[2].1.after.max, 5.0);
        assert!(capping_report(&v, &c, None, &caps[..2]).is_err());
    }

    #[test]
    fn fitted_caps_follow_their_rule() {
        let (v, c, w) = skewed();
        let q = fit_caps(&v, &c, Some(&w), CapRule::Quantile(0.9)).unwrap();
        for k in 0..3u32 {
            let (vk, wk): (Vec<f64>, Vec<f64>) = (0..v.len())
                .filter(|&i| c[i] == k)
                .map(|i| (v[i], w[i]))
                .unzip();
            assert_eq!(q[k as usize], quantiles(&vk, Some(&wk), &[0.9]).unwrap()[0]);
            let m = choose_cap(&vk, Some(&wk), CapRule::MetalRemoved(0.05)).unwrap();
            let e = &capping(&vk, Some(&wk), Some(&[m])).unwrap()[0];
            assert!((e.metal_removed - 0.05).abs() < 1e-9);
            let cv = choose_cap(&vk, Some(&wk), CapRule::Cv(0.3)).unwrap();
            let e = &capping(&vk, Some(&wk), Some(&[cv])).unwrap()[0];
            assert!((e.cv - 0.3).abs() < 1e-9);
        }
        assert!(q[0] != q[1] && q[1] != q[2]);
        let capped = cap_values(&v, &c, &q);
        for i in 0..v.len() {
            let cap = q[c[i] as usize];
            assert!(capped[i] <= cap || v[i].is_nan());
            assert!(v[i] > cap || capped[i].to_bits() == v[i].to_bits());
        }
        assert!(choose_cap(&v, None, CapRule::MetalRemoved(1.0)).is_err());
    }

    #[test]
    fn h_scatter_correlation_decays_with_lag() {
        let coords: Vec<[f64; 3]> = (0..300).map(|i| [i as f64, 0.0, 0.0]).collect();
        let values: Vec<f64> = coords.iter().map(|c| (c[0] / 15.0).sin()).collect();
        let r = |lag| h_scatter(&coords, &values, None, lag, 0.5, None).unwrap().2;
        assert!(r(1.0) > 0.99);
        assert!(r(1.0) > r(10.0) && r(10.0) > r(20.0));
    }

    #[test]
    fn h_scatter_follows_azimuth() {
        let coords: Vec<[f64; 3]> = (0..100)
            .map(|i| [(i % 10) as f64, (i / 10) as f64, 0.0])
            .collect();
        let values: Vec<f64> = coords.iter().map(|c| c[0]).collect();
        let north = Direction {
            azimuth: 0.0,
            dip: 0.0,
            tolerance: 10.0,
            bandwidth: None,
        };
        let (head, tail, _) = h_scatter(&coords, &values, None, 1.0, 0.1, Some(&north)).unwrap();
        assert_eq!(head.len(), 90);
        assert_eq!(head, tail);
        let east = Direction {
            azimuth: 90.0,
            ..north
        };
        let (head, tail, _) = h_scatter(&coords, &values, None, 1.0, 0.1, Some(&east)).unwrap();
        assert!(head.iter().zip(&tail).all(|(h, t)| h - t == 1.0));
    }

    #[test]
    fn correlation_pairwise_and_rank() {
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let mut y: Vec<f64> = x.iter().map(|x| x.powi(3)).collect();
        y[3] = f64::NAN;
        let cols = [x, y];
        let p = correlation(&cols, None, Method::Pearson).unwrap();
        let s = correlation(&cols, None, Method::Spearman).unwrap();
        assert!(p[0][1] < 0.95 && close(s[0][1], 1.0) && close(p[1][1], 1.0));
        assert!(correlation(&cols, Some(&[1.0]), Method::Pearson).is_err());
    }

    #[test]
    fn covariance_diagonal_is_the_variance() {
        let (v, _, w) = skewed();
        let other: Vec<f64> = v.iter().map(|x| 3.0 - 2.0 * x).collect();
        let c = correlation(&[v.clone(), other], Some(&w), Method::Covariance).unwrap();
        let s = describe(&v, Some(&w), &[]).unwrap();
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9 * b.abs();
        assert!(close(c[0][0], s.variance) && close(c[1][1], 4.0 * s.variance));
        assert!(close(c[0][1], -2.0 * s.variance) && c[0][1] == c[1][0]);
    }

    #[test]
    fn zero_tolerance_groups_exact_duplicates_only() {
        let coords = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [5.0, 5.0, -0.0],
            [5.0, 5.0, 0.0],
            [5.0, 5.0, 1e-12],
        ];
        assert_eq!(
            duplicates(&coords, 0.0).unwrap(),
            vec![vec![0, 2], vec![3, 4]]
        );
        assert_eq!(
            duplicates(&coords, 1e-9).unwrap(),
            vec![vec![0, 2], vec![3, 4, 5]]
        );
        assert!(duplicates(&coords, -1.0).is_err());
    }

    #[test]
    fn groups_are_transitive_and_independent_of_row_order() {
        let chain = [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        assert_eq!(duplicates(&chain, 1.0).unwrap(), vec![vec![0, 1, 2]]);

        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 20.0
        };
        let coords: Vec<[f64; 3]> = (0..400).map(|_| [next(), next(), next()]).collect();
        let tol = 1.0;
        let groups = duplicates(&coords, tol).unwrap();
        let mut label: Vec<usize> = (0..coords.len()).collect();
        loop {
            let mut changed = false;
            for i in 0..coords.len() {
                for j in 0..coords.len() {
                    let d2: f64 = (0..3).map(|k| (coords[i][k] - coords[j][k]).powi(2)).sum();
                    if d2 <= tol * tol && label[j] < label[i] {
                        label[i] = label[j];
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut brute: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        (0..coords.len()).for_each(|i| brute.entry(label[i]).or_default().push(i));
        let brute: Vec<Vec<usize>> = brute.into_values().filter(|g| g.len() > 1).collect();
        assert!(!brute.is_empty());
        assert_eq!(groups, brute);

        let reversed: Vec<[f64; 3]> = coords.iter().rev().copied().collect();
        let n = coords.len();
        let mut back: Vec<Vec<usize>> = duplicates(&reversed, tol)
            .unwrap()
            .into_iter()
            .map(|g| {
                let mut g: Vec<usize> = g.into_iter().map(|i| n - 1 - i).collect();
                g.sort();
                g
            })
            .collect();
        back.sort();
        assert_eq!(back, groups);
    }

    #[test]
    fn merged_mean_weighted_by_count_keeps_the_mean() {
        let values: Vec<f64> = (0..12).map(|i| f64::from(i * i % 7) + 0.5).collect();
        let groups = vec![vec![1, 4, 9], vec![2, 3], vec![6, 7, 8, 11]];
        let merged = merge_duplicates(&values, &groups, Merge::Mean, None).unwrap();
        assert_eq!(merged.len(), 12 - 2 - 1 - 3);
        let kept = [0, 1, 2, 5, 6, 10];
        let count = |i: &usize| {
            groups
                .iter()
                .find(|g| g[0] == *i)
                .map_or(1.0, |g| g.len() as f64)
        };
        let pooled: f64 = kept
            .iter()
            .zip(&merged)
            .map(|(i, v)| count(i) * v)
            .sum::<f64>()
            / 12.0;
        assert!(close(pooled, values.iter().sum::<f64>() / 12.0));

        let w = [1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let weighted = merge_duplicates(&values, &groups, Merge::Mean, Some(&w)).unwrap();
        assert!(close(
            weighted[1],
            (3.0 * values[1] + values[4] + values[9]) / 5.0
        ));

        let mut with_nan = values.clone();
        with_nan[2] = f64::NAN;
        let first = merge_duplicates(&with_nan, &groups, Merge::First, None).unwrap();
        let max = merge_duplicates(&with_nan, &groups, Merge::Max, None).unwrap();
        let mean = merge_duplicates(&with_nan, &groups, Merge::Mean, None).unwrap();
        assert!(first[2].is_nan() && max[2] == values[3] && mean[2] == values[3]);
        assert_eq!(max[3], values[5]);
        assert_eq!(
            max[4],
            [6, 7, 8, 11]
                .map(|i| values[i])
                .into_iter()
                .fold(0.0, f64::max)
        );
        assert!(merge_duplicates(&values, &[vec![1, 2], vec![2, 3]], Merge::Mean, None).is_err());
    }

    #[test]
    fn twins_are_recovered_with_their_bias() {
        use rand::{Rng, SeedableRng, seq::SliceRandom};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let a: Vec<[f64; 3]> = (0..400)
            .map(|k| [20.0 * (k % 20) as f64, 20.0 * (k / 20) as f64, 0.0])
            .map(|p| p.map(|x| x + rng.gen_range(-2.0..2.0)))
            .collect();
        let grade: Vec<f64> = (0..a.len()).map(|_| rng.gen_range(0.5..10.0)).collect();
        let mut twin: Vec<usize> = (0..a.len()).collect();
        twin.shuffle(&mut rng);
        let mut b = vec![[0.0; 3]; a.len()];
        let mut vb = vec![0.0; a.len()];
        for (i, &j) in twin.iter().enumerate() {
            b[j] = a[i].map(|x| x + rng.gen_range(-1.0..1.0));
            vb[j] = 1.2 * grade[i];
        }
        let p = pairs(&a, &b, 5.0, Some((&grade, &vb)), None, true).unwrap();
        assert_eq!(p.len(), a.len());
        assert!(p.iter().all(|&(i, j, d)| twin[i] == j && d <= 3f64.sqrt()));
        let (d, (va, vbs)): (Vec<f64>, (Vec<f64>, Vec<f64>)) =
            p.iter().map(|&(i, j, d)| (d, (grade[i], vb[j]))).unzip();
        let bias = paired_bias(&d, &va, &vbs, &[0.0, 1.0, 2.0]).unwrap();
        assert_eq!(bias.iter().map(|b| b.n).sum::<usize>(), a.len());
        assert!(bias.iter().all(|b| close(b.bias, 0.2)));
    }

    #[test]
    fn equivalent_spacing_of_a_square_grid_is_its_spacing() {
        use rand::{Rng, SeedableRng};
        let (s, c) = (25.0, 2.0);
        let hole: Vec<[f64; 3]> = (0..21 * 21)
            .map(|k| [s * (k % 21) as f64, s * (k / 21) as f64, 0.0])
            .collect();
        let (mut data, mut codes) = (vec![], vec![]);
        for (h, p) in hole.iter().enumerate() {
            for i in 0..100 {
                data.push([p[0], p[1], c * (i as f64 + 0.5)]);
                codes.push(h as u32);
            }
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let targets: Vec<[f64; 3]> = (0..400)
            .map(|_| {
                [
                    rng.gen_range(7.0 * s..13.0 * s),
                    rng.gen_range(7.0 * s..13.0 * s),
                    rng.gen_range(80.0..120.0),
                ]
            })
            .collect();
        let mean = |d: Vec<f64>| d.iter().sum::<f64>() / d.len() as f64;
        let volume = Spacing::Volume {
            anisotropy: None,
            radius: 3.0 * s,
            composite_length: c,
        };
        let ds = mean(spacing(&data, &targets, volume, None).unwrap());
        assert!((ds / s - 1.0).abs() < 0.02, "volume {ds}");
        let ns: Vec<usize> = (4..11).collect();
        let plan = spacing(&data, &targets, Spacing::Plan(&ns), Some(&codes)).unwrap();
        assert!((mean(plan.clone()) / s - 1.0).abs() < 0.05, "plan");
        assert_eq!(
            plan,
            spacing(&hole, &targets, Spacing::Plan(&ns), None).unwrap()
        );
        let each = spacing(&data, &targets, Spacing::Plan(&ns), None).unwrap();
        assert!(mean(each) < s / 2.0);
        let few = spacing(&hole[..5], &targets[..1], Spacing::Plan(&ns), None).unwrap();
        assert!(few[0].is_nan());
        assert!(spacing(&hole, &targets, Spacing::Plan(&[0]), None).is_err());
        assert!(spacing(&data, &targets, volume, Some(&codes)).is_err());
        let threads = |n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
                .install(|| spacing(&data, &targets, volume, None).unwrap())
        };
        assert_eq!(threads(1), threads(4));
    }

    #[test]
    fn unique_pairs_are_greedy_and_skip_holes() {
        let a = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let b = [[1.0, 0.0, 0.0]];
        assert_eq!(
            pairs(&a, &b, 2.0, None, None, true).unwrap(),
            vec![(1, 0, 0.5)]
        );
        assert_eq!(pairs(&a, &b, 2.0, None, None, false).unwrap().len(), 2);
        let holes = (&[3, 4][..], &[4][..]);
        assert_eq!(
            pairs(&a, &b, 2.0, None, Some(holes), true).unwrap(),
            vec![(0, 0, 1.0)]
        );
        let nan = (&[1.0, f64::NAN][..], &[1.0][..]);
        assert_eq!(
            pairs(&a, &b, 2.0, Some(nan), None, true).unwrap(),
            vec![(0, 0, 1.0)]
        );
    }

    #[test]
    fn required_spacing_interpolates_a_linear_curve() {
        let s: Vec<f64> = (0..10).map(|i| 5.0 + 2.0 * f64::from(i)).collect();
        let u: Vec<f64> = s.iter().map(|s| 0.01 * s).collect();
        let (r, beyond) = required_spacing(&s, &u, 0.15);
        assert!((r - 15.0).abs() < 1e-12 && !beyond);
        assert!((required_spacing(&s, &u, 0.1).0 - 10.0).abs() < 1e-12);
        assert_eq!(required_spacing(&s, &u, 1.0), (23.0, true));
        let (r, beyond) = required_spacing(&s, &u, 0.0);
        assert!(r.is_nan() && !beyond);
        assert!(required_spacing(&[], &[], 0.1).0.is_nan());
    }

    #[test]
    fn uncertainty_curve_bins_and_forces_monotone_quantiles() {
        let mut spacing = Vec::new();
        let mut uncertainty = Vec::new();
        for (k, &level) in [0.1, 0.3, 0.2, 0.4].iter().enumerate() {
            for i in 0..10 {
                spacing.push(10.0 * k as f64 + 1.0 + 0.1 * f64::from(i));
                uncertainty.push(level);
            }
        }
        spacing.extend([35.0, f64::NAN, 1.0]);
        uncertainty.extend([0.9, 0.5, f64::NAN]);
        let edges = Bins::Edges(vec![0.0, 10.0, 20.0, 30.0, 40.0]);
        let c = uncertainty_curve(&spacing, &uncertainty, &edges, &[0.5], 0.25, 8).unwrap();
        let q: Vec<f64> = c.iter().map(|b| b.quantiles[0]).collect();
        assert_eq!(q, [0.1, 0.3, 0.3, 0.4]);
        assert_eq!(c.iter().map(|b| b.n).collect::<Vec<_>>(), [10, 10, 10, 11]);
        assert_eq!(c[0].share, 1.0);
        assert_eq!(c[1].share, 0.0);
        assert!((c[0].spacing - 1.45).abs() < 1e-12);
        let sparse = uncertainty_curve(&spacing, &uncertainty, &edges, &[0.5], 0.25, 11).unwrap();
        assert_eq!(sparse.len(), 1);
        let auto = uncertainty_curve(&spacing, &uncertainty, &Bins::Auto, &[0.5], 0.25, 1).unwrap();
        assert_eq!(auto.iter().map(|b| b.n).sum::<usize>(), 41);
    }
}
