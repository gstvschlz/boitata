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

/// Grade–tonnage curve of the data: each value stands for weight (e.g. volume)
/// × density tonnes, 1 by default.
pub fn grade_tonnage(
    values: &[f64],
    weights: Option<&[f64]>,
    density: Option<&[f64]>,
    cutoffs: &[f64],
) -> Result<Vec<Tonnage>> {
    check(values.len(), values, density)?;
    if cutoffs.iter().any(|c| c.is_nan()) {
        return invalid("cutoffs must not be NaN");
    }
    let tonnes: Option<Vec<f64>> = match (weights, density) {
        (Some(w), Some(d)) => {
            check(values.len(), values, Some(w))?;
            Some(w.iter().zip(d).map(|(w, d)| w * d).collect())
        }
        (w, d) => w.or(d).map(<[f64]>::to_vec),
    };
    let (v, t) = valid(values, tonnes.as_deref())?;
    Ok(cutoffs
        .iter()
        .map(|&cutoff| {
            let (tonnage, metal) = v
                .iter()
                .zip(&t)
                .filter(|(v, _)| **v >= cutoff)
                .fold((0.0, 0.0), |(s, m), (v, t)| (s + t, m + t * v));
            let mean_grade = if tonnage > 0.0 {
                metal / tonnage
            } else {
                f64::NAN
            };
            Tonnage {
                cutoff,
                tonnage,
                mean_grade,
                metal,
            }
        })
        .collect())
}

/// Mean and count per bin; `centres` ascending, empty bins omitted.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub centres: Vec<f64>,
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
        p.centres.push((k as f64 + 0.5) * width);
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
/// domain in the same hole, negative inside.
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
            let k = (d / bin).floor() as i64;
            let k = if domains[i] == inside { -k - 1 } else { k };
            let b: &mut (f64, f64, usize) = bins.entry(k).or_default();
            b.0 += 1.0;
            b.1 += values[i];
            b.2 += 1;
        }
    }
    Ok(profile(bins, bin))
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
    let capped: Vec<f64> = values
        .iter()
        .zip(categories)
        .map(|(&v, &c)| {
            if v > caps[c as usize] {
                caps[c as usize]
            } else {
                v
            }
        })
        .collect();
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

fn pearson(x: &[f64], y: &[f64], w: &[f64]) -> f64 {
    let sw: f64 = w.iter().sum();
    if x.len() < 2 || sw <= 0.0 {
        return f64::NAN;
    }
    let mx = x.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / sw;
    let my = y.iter().zip(w).map(|(y, w)| y * w).sum::<f64>() / sw;
    let (mut cxy, mut cxx, mut cyy) = (0.0, 0.0, 0.0);
    for ((x, y), w) in x.iter().zip(y).zip(w) {
        cxy += w * (x - mx) * (y - my);
        cxx += w * (x - mx) * (x - mx);
        cyy += w * (y - my) * (y - my);
    }
    cxy / (cxx * cyy).sqrt()
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

/// Weighted correlation matrix of `columns`, each pair over rows where both are
/// not NaN.
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
            out[a][b] = pearson(&x, &y, &w);
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

#[cfg(test)]
mod tests {
    use super::*;

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
        for (c, m) in s.centres.iter().zip(&s.mean) {
            assert!(close(*c, *m));
        }
        let y = swath(&coords, &values, None, None, 10.0, Along::Axis(1)).unwrap();
        assert_eq!((y.centres, y.count), (vec![5.0], vec![100]));
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
        for (d, m) in c.centres.iter().zip(&c.mean) {
            assert_eq!(*m, if *d < 0.0 { 1.0 } else { 5.0 });
        }
        assert_eq!(c.count.iter().sum::<usize>(), 20);
        assert_eq!(c.centres.first(), Some(&-11.0));
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
}
