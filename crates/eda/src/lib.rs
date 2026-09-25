//! Exploratory data analysis on raw columns: NaN values are skipped, weights are
//! declustering weights.

use std::collections::BTreeMap;

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
    })
}

/// Mean and count per bin; `centres` ascending, empty bins omitted.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub centres: Vec<f64>,
    pub mean: Vec<f64>,
    pub count: Vec<usize>,
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

/// Weighted mean per slice of `width` along a direction; slices start at
/// coordinate 0 so profiles of different data line up.
pub fn swath(
    coords: &[[f64; 3]],
    values: &[f64],
    weights: Option<&[f64]>,
    width: f64,
    along: Along,
) -> Result<Profile> {
    check(coords.len(), values, weights)?;
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
    let mut bins = BTreeMap::new();
    for (i, (p, &z)) in coords.iter().zip(values).enumerate() {
        if z.is_nan() {
            continue;
        }
        let w = weights.map_or(1.0, |w| w[i]);
        let t = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
        let b: &mut (f64, f64, usize) = bins.entry((t / width).floor() as i64).or_default();
        b.0 += w;
        b.1 += w * z;
        b.2 += 1;
    }
    Ok(profile(bins, width))
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
        let s = swath(&coords, &values, None, 10.0, Along::Azimuth(90.0)).unwrap();
        assert_eq!(s.count, vec![10; 10]);
        for (c, m) in s.centres.iter().zip(&s.mean) {
            assert!(close(*c, *m));
        }
        let y = swath(&coords, &values, None, 10.0, Along::Axis(1)).unwrap();
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
}
