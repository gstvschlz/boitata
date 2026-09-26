//! Automatic variogram-model fitting by weighted least squares.

use crate::aniso::{Angles, Anisotropy, rotation_matrix};
use crate::composite::Variogram;
use crate::coreg::{CoregStructure, Coregionalization};
use crate::empirical::Experimental;
use crate::error::{Result, VarioError};
use crate::model::{Model, Structure, shape};
use crate::surface::unit_vector;
use ceres_core::angles_from_axes;
use nalgebra::{DMatrix, Matrix3, SymmetricEigen, Vector3};
use serde::{Deserialize, Serialize};

/// Weighting scheme for the least-squares objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Weighting {
    /// Uniform weights.
    Uniform,
    /// Weight by pair count `N(h)` (more pairs ⇒ more reliable lag).
    ByCount,
    /// `N(h) / γ_model(h)²` (emphasizes short lags).
    ByCountOverGamma,
    /// `N(h) / h²` (emphasizes short lags, independently of the model).
    ByCountOverDistance,
}

/// Lower and upper bound of a fitted parameter; equal bounds fix it.
pub type Bounds = (f64, f64);

/// One structure to fit: its shape and optional bounds on its partial sill and range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructureSpec {
    pub model: Model,
    pub sill: Option<Bounds>,
    pub range: Option<Bounds>,
}

impl StructureSpec {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            sill: None,
            range: None,
        }
    }
}

/// Optional nugget bounds and one to three nested structures to fit.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedSpec {
    pub nugget: Option<Bounds>,
    pub structures: Vec<StructureSpec>,
}

/// Result of a fit: the model and the achieved weighted SSE.
#[derive(Debug, Clone)]
pub struct FitResult {
    pub variogram: Variogram,
    pub wsse: f64,
}

/// Fit a single-structure model (with nugget) to an experimental variogram by WLS.
///
/// Uses a coarse grid over `(nugget, sill, range)` followed by coordinate-descent
/// refinement. The `Power` and `SineHole` models are supported but fit only their
/// scaling/range (no bounded sill for `Power`).
pub fn fit(exp: &Experimental, model: Model, weighting: Weighting) -> Result<FitResult> {
    if exp.lags.is_empty() {
        return Err(VarioError::FittingFailed(
            "empty experimental variogram".into(),
        ));
    }

    let max_gamma = exp
        .gammas
        .iter()
        .cloned()
        .fold(f64::MIN, f64::max)
        .max(f64::MIN_POSITIVE);
    let max_lag = exp.lags.iter().cloned().fold(f64::MIN, f64::max);

    let sill_hi = max_gamma * 1.5;
    let range_hi = max_lag * 1.5;

    let mut best = FitResult {
        variogram: Variogram {
            nugget: 0.0,
            structures: vec![Structure::new(model, max_gamma, max_lag)],
            anisotropy: None,
        },
        wsse: f64::INFINITY,
    };

    let steps = 12;
    for ni in 0..=6 {
        let nugget = max_gamma * (ni as f64) / 12.0; // nugget up to half the sill
        for si in 1..=steps {
            let sill = sill_hi * (si as f64) / steps as f64;
            for ri in 1..=steps {
                let range = range_hi * (ri as f64) / steps as f64;
                let cand = Variogram {
                    nugget,
                    structures: vec![Structure::new(model, sill, range)],
                    anisotropy: None,
                };
                let e = wsse(&cand, exp, weighting);
                if e < best.wsse {
                    best = FitResult {
                        variogram: cand,
                        wsse: e,
                    };
                }
            }
        }
    }

    refine(&mut best, exp, model, weighting);

    Ok(best)
}

fn refine(best: &mut FitResult, exp: &Experimental, model: Model, weighting: Weighting) {
    let mut nugget = best.variogram.nugget;
    let mut sill = best.variogram.structures[0].sill;
    let mut range = best.variogram.structures[0].range;

    let mut scale = 0.5;
    for _ in 0..40 {
        let mut improved = false;
        for &(dn, ds, dr) in &[
            (scale, 0.0, 0.0),
            (-scale, 0.0, 0.0),
            (0.0, scale, 0.0),
            (0.0, -scale, 0.0),
            (0.0, 0.0, scale),
            (0.0, 0.0, -scale),
        ] {
            let cn = (nugget * (1.0 + dn)).max(0.0);
            let cs = (sill * (1.0 + ds)).max(1e-9);
            let cr = (range * (1.0 + dr)).max(1e-6);
            let cand = Variogram {
                nugget: cn,
                structures: vec![Structure::new(model, cs, cr)],
                anisotropy: None,
            };
            let e = wsse(&cand, exp, weighting);
            if e < best.wsse {
                best.wsse = e;
                best.variogram = cand;
                nugget = cn;
                sill = cs;
                range = cr;
                improved = true;
            }
        }
        if !improved {
            scale *= 0.5;
            if scale < 1e-4 {
                break;
            }
        }
    }
}

fn wsse(v: &Variogram, exp: &Experimental, weighting: Weighting) -> f64 {
    wsse_at(v, exp, &exp.lags, weighting)
}

/// Weighted SSE with γ at `exp.lags` and the N(h)/h² weights at `dist`.
fn wsse_at(v: &Variogram, exp: &Experimental, dist: &[f64], weighting: Weighting) -> f64 {
    let mut acc = 0.0;
    for i in 0..exp.lags.len() {
        let model_g = v.gamma(exp.lags[i]);
        let resid = exp.gammas[i] - model_g;
        acc += weight(weighting, exp.counts[i], dist[i], model_g) * resid * resid;
    }
    acc
}

fn weight(weighting: Weighting, count: usize, h: f64, model_g: f64) -> f64 {
    match weighting {
        Weighting::Uniform => 1.0,
        Weighting::ByCount => count as f64,
        Weighting::ByCountOverGamma => {
            let denom = model_g.max(1e-6);
            count as f64 / (denom * denom)
        }
        Weighting::ByCountOverDistance => {
            let h = h.max(1e-6);
            count as f64 / (h * h)
        }
    }
}

/// Nugget extrapolated from the first `lags` lags: the intercept of a
/// pair-count-weighted straight line through them, with intercept and slope
/// both non-negative. Meant for a [`downhole`](crate::empirical::downhole)
/// variogram, whose shortest lags are the closest pairs available.
pub fn extrapolated_nugget(exp: &Experimental, lags: usize) -> Result<f64> {
    let n = lags.min(exp.lags.len());
    if lags < 2 || n < 2 {
        return Err(VarioError::FittingFailed(
            "the nugget needs at least two lags".into(),
        ));
    }
    let rows: Vec<Vec<f64>> = exp.lags[..n].iter().map(|&h| vec![1.0, h]).collect();
    let w: Vec<f64> = exp.counts[..n].iter().map(|&c| c as f64).collect();
    let bounds = [(0.0, f64::INFINITY); 2];
    Ok(bounded_lsq(&rows, &exp.gammas[..n], &w, &bounds)[0])
}

/// Fit a nugget plus one to three nested structures by WLS, each parameter free,
/// bounded or fixed.
///
/// Given the ranges, γ is linear in the nugget and sills, so these are solved
/// exactly by bounded least squares; the ranges are searched on a geometric grid
/// and the best starts refined by coordinate descent. Ranges increase in the
/// listed order. With one unbounded structure and a free nugget this is [`fit`].
pub fn fit_nested(
    exp: &Experimental,
    spec: &NestedSpec,
    weighting: Weighting,
) -> Result<FitResult> {
    if let [s] = spec.structures[..]
        && spec.nugget.is_none()
        && s.sill.is_none()
        && s.range.is_none()
    {
        return fit(exp, s.model, weighting);
    }
    let n = spec.structures.len();
    let (linear, ranges) = parameters(spec)?;
    let h_max = exp.lags.iter().cloned().fold(0.0, f64::max);
    if h_max <= 0.0 {
        return Err(VarioError::FittingFailed(
            "empty experimental variogram".into(),
        ));
    }
    let h_min = exp
        .lags
        .iter()
        .cloned()
        .filter(|&h| h > 0.0)
        .fold(h_max, f64::min);
    let ordered = |r: &[f64]| r.windows(2).all(|w| w[0] < w[1]);
    let eval = |r: &[f64]| {
        let v = solve(exp, &exp.lags, spec, &linear, r, weighting);
        (wsse(&v, exp, weighting), v)
    };

    let grids: Vec<Vec<f64>> = ranges
        .iter()
        .map(|&(lo, hi)| grid(lo, hi, h_min, h_max))
        .collect();
    let mut starts = Vec::new();
    let mut idx = vec![0; n];
    'grid: loop {
        let r: Vec<f64> = idx.iter().zip(&grids).map(|(&i, g)| g[i]).collect();
        if ordered(&r) {
            starts.push((eval(&r).0, r));
        }
        for k in 0..n {
            idx[k] += 1;
            if idx[k] < grids[k].len() {
                continue 'grid;
            }
            idx[k] = 0;
        }
        break;
    }
    starts.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut best: Option<(f64, Variogram)> = None;
    for (_, start) in starts.into_iter().take(16) {
        let mut r = start;
        let mut cur = eval(&r);
        let mut scale = 0.5;
        for _ in 0..500 {
            let mut improved = false;
            for k in 0..n {
                for d in [scale, -scale] {
                    let mut cand = r.clone();
                    cand[k] = (r[k] * (1.0 + d)).clamp(ranges[k].0, ranges[k].1);
                    if cand[k] == r[k] || !ordered(&cand) {
                        continue;
                    }
                    let e = eval(&cand);
                    if e.0 < cur.0 {
                        cur = e;
                        r = cand;
                        improved = true;
                    }
                }
            }
            if !improved {
                scale *= 0.5;
                if scale < 1e-7 {
                    break;
                }
            }
        }
        if best.as_ref().is_none_or(|b| cur.0 < b.0) {
            best = Some(cur);
        }
    }
    let (wsse, variogram) = best.ok_or_else(|| {
        VarioError::InvalidParameters("range bounds leave no increasing order".into())
    })?;
    Ok(FitResult { variogram, wsse })
}

/// Bounds on the anisotropy of [`fit_directional`]: angles in degrees, `semi`
/// and `minor` as ratios to the major range. `None` leaves an angle free and a
/// ratio in `(0, 1]`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AnisotropySpec {
    pub azimuth: Option<Bounds>,
    pub dip: Option<Bounds>,
    pub rake: Option<Bounds>,
    pub semi: Option<Bounds>,
    pub minor: Option<Bounds>,
}

/// Fit one anisotropic model, its nested structures sharing the anisotropy,
/// jointly to experimental variograms along `directions` (azimuth, dip in
/// degrees). Structure ranges are major-axis ranges.
///
/// Lags are mapped to anisotropic distances, so for given angles, ratios and
/// ranges the nugget and sills are solved exactly as in [`fit_nested`]. The
/// start rescales the pooled isotropic fit per direction: the inverse squared
/// scale is a quadratic form in the direction, fitted by least squares and
/// diagonalised for the axes. Angles, ratios and ranges are then refined by
/// coordinate descent, deterministically. Ranges without bounds stay within
/// the largest lag. When every direction is horizontal, dip, rake and
/// the minor ratio default to 0, 0 and 1. Free angles come back
/// with azimuth and rake in `[0, 180)` and, with both ratios free,
/// `semi >= minor`.
pub fn fit_directional(
    exps: &[Experimental],
    directions: &[(f64, f64)],
    spec: &NestedSpec,
    aniso: &AnisotropySpec,
    weighting: Weighting,
) -> Result<FitResult> {
    if exps.len() != directions.len() {
        return Err(VarioError::InvalidParameters(format!(
            "{} experimental variograms for {} directions",
            exps.len(),
            directions.len()
        )));
    }
    let (linear, _) = parameters(spec)?;
    let units = units(directions);
    let mut pooled = Experimental {
        lags: Vec::new(),
        gammas: Vec::new(),
        counts: Vec::new(),
        covariances: None,
    };
    let mut owner = Vec::new();
    for (d, e) in exps.iter().enumerate() {
        pooled.lags.extend(&e.lags);
        pooled.gammas.extend(&e.gammas);
        pooled.counts.extend(&e.counts);
        owner.extend(std::iter::repeat_n(d, e.lags.len()));
    }
    let reach = pooled.lags.iter().cloned().fold(0.0, f64::max);
    let limits = limits(directions, aniso, spec, reach)?;
    let eval = |x: &[f64]| {
        let k = stretch(x, &units);
        let mut e = pooled.clone();
        for (h, &d) in e.lags.iter_mut().zip(&owner) {
            *h *= k[d];
        }
        let v = solve(&e, &pooled.lags, spec, &linear, &x[5..], weighting);
        (wsse_at(&v, &e, &pooled.lags, weighting), v)
    };

    let iso = fit_nested(&pooled, spec, weighting)?.variogram;
    let mut normal = vec![vec![0.0; 7]; 6];
    let mut mean = 0.0;
    for (e, u) in exps.iter().zip(&units) {
        if e.lags.is_empty() {
            continue;
        }
        let cost = |s: f64| {
            let mut r = e.clone();
            r.lags.iter_mut().for_each(|h| *h /= s);
            wsse_at(&iso, &r, &e.lags, weighting)
        };
        let t = scale_search(cost).powi(-2);
        let q = [
            u.x * u.x,
            u.y * u.y,
            u.z * u.z,
            2.0 * u.x * u.y,
            2.0 * u.x * u.z,
            2.0 * u.y * u.z,
        ];
        for j in 0..6 {
            for k in 0..6 {
                normal[j][k] += q[j] * q[k];
            }
            normal[j][6] += q[j] * t;
        }
        mean += t / directions.len() as f64;
    }
    let ridge = 1e-6 * (0..6).map(|j| normal[j][j]).sum::<f64>();
    for (j, row) in normal.iter_mut().enumerate() {
        row[j] += ridge;
        if j < 3 {
            row[6] += ridge * mean;
        }
    }
    let m = gauss(&mut normal).unwrap_or(vec![mean, mean, mean, 0.0, 0.0, 0.0]);
    let form = Matrix3::new(m[0], m[3], m[4], m[3], m[1], m[5], m[4], m[5], m[2]);
    let eig = SymmetricEigen::new(form);
    let mut axes = [0, 1, 2];
    axes.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
    let col = |i: usize| {
        let c = eig.eigenvectors.column(axes[i]);
        [c[0], c[1], c[2]]
    };
    let mut round = angles_from_axes(col(0), col(1)).to_vec();
    round.extend([1.0, 1.0]);
    round.extend(iso.structures.iter().map(|s| s.range));
    clamp(&mut round, &limits);
    let r = rotation_matrix(round[0], round[1], round[2]);
    let floor = 1e-2 * eig.eigenvalues.amax();
    let d: Vec<f64> = (0..3)
        .map(|i| (r.row(i) * form * r.row(i).transpose())[0].max(floor))
        .collect();
    let mut stretched = round.clone();
    stretched[3] = (d[0] / d[1]).sqrt();
    stretched[4] = (d[0] / d[2]).sqrt();
    for (r, s) in stretched[5..].iter_mut().zip(&iso.structures) {
        *r = s.range / d[0].sqrt();
    }
    clamp(&mut stretched, &limits);

    let (x, (wsse, variogram)) = [stretched, round]
        .into_iter()
        .filter(|x| ordered(&x[5..]))
        .map(|x| descend(x, &limits, eval))
        .reduce(|a, b| if b.1.0 < a.1.0 { b } else { a })
        .ok_or_else(|| {
            VarioError::InvalidParameters("range bounds leave no increasing order".into())
        })?;
    let variogram = variogram.with_anisotropy(canonical(&x, &limits, aniso)?);
    Ok(FitResult { variogram, wsse })
}

fn ordered(r: &[f64]) -> bool {
    r.windows(2).all(|w| w[0] < w[1])
}

fn units(directions: &[(f64, f64)]) -> Vec<Vector3<f64>> {
    directions
        .iter()
        .map(|&(az, dip)| {
            let (x, y, z) = unit_vector(az, dip);
            Vector3::new(x, y, z)
        })
        .collect()
}

/// Per direction, the factor mapping a lag to its distance along the major
/// axis for `x = [azimuth, dip, rake, semi, minor, ..]`.
fn stretch(x: &[f64], units: &[Vector3<f64>]) -> Vec<f64> {
    let a = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0 / x[3], 1.0 / x[4]))
        * rotation_matrix(x[0], x[1], x[2]);
    units.iter().map(|u| (a * u).norm()).collect()
}

/// Bounds on `[azimuth, dip, rake, semi, minor, ranges..]`, `None` for a free
/// angle; free ranges stop at `reach`.
fn limits(
    directions: &[(f64, f64)],
    aniso: &AnisotropySpec,
    spec: &NestedSpec,
    reach: f64,
) -> Result<Vec<Option<Bounds>>> {
    if directions
        .iter()
        .any(|d| !(d.0.is_finite() && d.1.is_finite()))
    {
        return Err(VarioError::InvalidParameters(
            "directions must be finite".into(),
        ));
    }
    let (_, ranges) = parameters(spec)?;
    let flat = directions.iter().all(|d| d.1 == 0.0);
    let angle = |b: Option<Bounds>, flat_value: Option<f64>| match b {
        Some((lo, hi)) if lo.is_finite() && hi.is_finite() && lo <= hi => Ok(Some((lo, hi))),
        Some((lo, hi)) => Err(VarioError::InvalidParameters(format!(
            "angle bounds need finite low <= high, got ({lo}, {hi})"
        ))),
        None => Ok(flat_value.filter(|_| flat).map(|v| (v, v))),
    };
    let ratio = |b: Option<Bounds>| {
        let (lo, hi) = bounds(b.or(Some((0.0, 1.0))), "ratio")?;
        if hi > 0.0 {
            Ok(Some((lo.max(1e-6).min(hi), hi)))
        } else {
            Err(VarioError::InvalidParameters("ratios must be > 0".into()))
        }
    };
    let mut limits = vec![
        angle(aniso.azimuth, None)?,
        angle(aniso.dip, Some(0.0))?,
        angle(aniso.rake, Some(0.0))?,
        ratio(aniso.semi)?,
        ratio(aniso.minor.or(flat.then_some((1.0, 1.0))))?,
    ];
    limits.extend(
        spec.structures
            .iter()
            .zip(&ranges)
            .map(|(s, &(lo, hi))| Some((lo, if s.range.is_none() { reach.max(lo) } else { hi }))),
    );
    Ok(limits)
}

fn clamp(x: &mut [f64], limits: &[Option<Bounds>]) {
    for (v, l) in x.iter_mut().zip(limits) {
        if let Some((lo, hi)) = *l {
            *v = if lo == hi { lo } else { v.clamp(lo, hi) };
        }
    }
}

/// Coordinate descent on `[azimuth, dip, rake, semi, minor, ranges..]` within
/// `limits`: additive steps on the angles, relative steps on the rest.
fn descend<T>(
    mut x: Vec<f64>,
    limits: &[Option<Bounds>],
    eval: impl Fn(&[f64]) -> (f64, T),
) -> (Vec<f64>, (f64, T)) {
    let mut cur = eval(&x);
    let mut scale = 0.5;
    for _ in 0..5000 {
        let mut improved = false;
        for k in 0..x.len() {
            for step in [scale, -scale] {
                let mut cand = x.clone();
                cand[k] = if k < 3 {
                    x[k] + 90.0 * step
                } else {
                    x[k] * (1.0 + step)
                };
                clamp(&mut cand, limits);
                if cand[k] == x[k] || !ordered(&cand[5..]) {
                    continue;
                }
                let e = eval(&cand);
                if e.0 < cur.0 {
                    cur = e;
                    x = cand;
                    improved = true;
                }
            }
        }
        if !improved {
            scale *= 0.5;
            if scale < 1e-9 {
                break;
            }
        }
    }
    (x, cur)
}

/// The anisotropy of `x`, free angles returned with azimuth and rake in
/// `[0, 180)` and, with both ratios free, `semi >= minor`.
fn canonical(x: &[f64], limits: &[Option<Bounds>], aniso: &AnisotropySpec) -> Result<Anisotropy> {
    let (mut semi, mut minor) = (x[3], x[4]);
    let mut angles = [x[0], x[1], x[2]];
    if limits[..3].iter().all(Option::is_none) {
        let r = rotation_matrix(x[0], x[1], x[2]);
        let (mut m, mut s) = (r.row(0).transpose(), r.row(1).transpose());
        if aniso.semi.is_none() && aniso.minor.is_none() && semi < minor {
            s = m.cross(&s);
            (semi, minor) = (minor, semi);
        }
        let arr = |v: &Vector3<f64>| [v.x, v.y, v.z];
        angles = angles_from_axes(arr(&m), arr(&s));
        if angles[0] >= 180.0 {
            (m, s) = (-m, -s);
            angles = angles_from_axes(arr(&m), arr(&s));
        }
        if angles[2] >= 180.0 {
            angles = angles_from_axes(arr(&m), arr(&-s));
        }
    } else if limits[0].is_none() {
        let period = if angles[1] == 0.0 && angles[2].rem_euclid(180.0) == 0.0 {
            180.0
        } else {
            360.0
        };
        angles[0] = angles[0].rem_euclid(period);
    }
    Anisotropy::new(Angles {
        azimuth: angles[0],
        dip: angles[1],
        rake: angles[2],
        major: 1.0,
        semi,
        minor,
    })
}

/// The scale `s` in `[1/16, 16]` minimizing `cost(s)`: a geometric grid, then
/// shrinking steps.
fn scale_search(cost: impl Fn(f64) -> f64) -> f64 {
    let (mut s, mut best) = (-32..=32)
        .map(|i| {
            let s = 2f64.powf(i as f64 / 8.0);
            (s, cost(s))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("non-empty grid");
    let mut f = 2f64.powf(1.0 / 16.0);
    while f > 1.0 + 1e-10 {
        match [s * f, s / f]
            .map(|t| (t, cost(t)))
            .into_iter()
            .find(|c| c.1 < best)
        {
            Some(c) => (s, best) = c,
            None => f = f.sqrt(),
        }
    }
    s
}

/// Nugget-and-sill bounds and range bounds of a spec.
fn parameters(spec: &NestedSpec) -> Result<(Vec<Bounds>, Vec<Bounds>)> {
    let n = spec.structures.len();
    if !(1..=3).contains(&n) {
        return Err(VarioError::InvalidParameters(format!(
            "fit one to three structures, got {n}"
        )));
    }
    let linear = std::iter::once(bounds(spec.nugget, "nugget"))
        .chain(spec.structures.iter().map(|s| bounds(s.sill, "sill")))
        .collect::<Result<Vec<_>>>()?;
    let ranges = spec
        .structures
        .iter()
        .map(|s| {
            let (lo, hi) = bounds(s.range, "range")?;
            if hi > 0.0 {
                Ok((lo.max(1e-6).min(hi), hi))
            } else {
                Err(VarioError::InvalidParameters("ranges must be > 0".into()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((linear, ranges))
}

fn bounds(b: Option<Bounds>, what: &str) -> Result<Bounds> {
    let (lo, hi) = b.unwrap_or((0.0, f64::INFINITY));
    if lo.is_finite() && lo >= 0.0 && lo <= hi {
        Ok((lo, hi))
    } else {
        Err(VarioError::InvalidParameters(format!(
            "{what} bounds need 0 <= low <= high, got ({lo}, {hi})"
        )))
    }
}

fn grid(lo: f64, hi: f64, h_min: f64, h_max: f64) -> Vec<f64> {
    let a = lo.max(h_min / 2.0).min(hi);
    let b = hi.min(1.5 * h_max).max(a);
    if b == a {
        return vec![a];
    }
    let steps = 16;
    (0..steps)
        .map(|i| a * (b / a).powf(i as f64 / (steps - 1) as f64))
        .collect()
}

/// The nugget and sills minimizing the weighted SSE for fixed ranges; the
/// model-dependent weighting is iterated from the experimental γ.
fn solve(
    exp: &Experimental,
    dist: &[f64],
    spec: &NestedSpec,
    linear: &[Bounds],
    ranges: &[f64],
    weighting: Weighting,
) -> Variogram {
    let rows: Vec<Vec<f64>> = exp
        .lags
        .iter()
        .map(|&h| {
            std::iter::once(if h > 0.0 { 1.0 } else { 0.0 })
                .chain(
                    spec.structures
                        .iter()
                        .zip(ranges)
                        .map(|(s, &r)| shape(s.model, h, r)),
                )
                .collect()
        })
        .collect();
    let passes = if weighting == Weighting::ByCountOverGamma {
        4
    } else {
        1
    };
    let mut model_g = exp.gammas.clone();
    let mut coef = Vec::new();
    for _ in 0..passes {
        let w: Vec<f64> = (0..rows.len())
            .map(|i| weight(weighting, exp.counts[i], dist[i], model_g[i]))
            .collect();
        coef = bounded_lsq(&rows, &exp.gammas, &w, linear);
        model_g = rows
            .iter()
            .map(|a| a.iter().zip(&coef).map(|(x, c)| x * c).sum())
            .collect();
    }
    Variogram {
        nugget: coef[0],
        structures: spec
            .structures
            .iter()
            .zip(ranges)
            .zip(&coef[1..])
            .map(|((s, &r), &c)| Structure::new(s.model, c, r))
            .collect(),
        anisotropy: None,
    }
}

/// Box-constrained weighted least squares, exact for the few unknowns here:
/// every variable is tried free, at its lower or at its upper bound, and the
/// best feasible solution kept.
fn bounded_lsq(rows: &[Vec<f64>], y: &[f64], w: &[f64], bounds: &[Bounds]) -> Vec<f64> {
    let m = bounds.len();
    let mut a = vec![vec![0.0; m]; m];
    let mut b = vec![0.0; m];
    for ((row, &yi), &wi) in rows.iter().zip(y).zip(w) {
        for j in 0..m {
            b[j] += wi * row[j] * yi;
            for k in 0..m {
                a[j][k] += wi * row[j] * row[k];
            }
        }
    }
    let objective = |c: &[f64]| {
        (0..m)
            .map(|j| c[j] * ((0..m).map(|k| a[j][k] * c[k]).sum::<f64>() - 2.0 * b[j]))
            .sum::<f64>()
    };
    let mut best: Vec<f64> = bounds.iter().map(|b| 0.0_f64.clamp(b.0, b.1)).collect();
    let mut best_obj = objective(&best);
    'sets: for code in 0..3usize.pow(m as u32) {
        let mut c = vec![0.0; m];
        let mut free = Vec::new();
        let mut rest = code;
        for (j, &(lo, hi)) in bounds.iter().enumerate() {
            match rest % 3 {
                0 if lo.is_finite() => c[j] = lo,
                1 if hi.is_finite() && hi > lo => c[j] = hi,
                2 if hi > lo => free.push(j),
                _ => continue 'sets,
            }
            rest /= 3;
        }
        let mut sys: Vec<Vec<f64>> = free
            .iter()
            .map(|&j| {
                let mut row: Vec<f64> = free.iter().map(|&k| a[j][k]).collect();
                row.push(b[j] - (0..m).map(|k| a[j][k] * c[k]).sum::<f64>());
                row
            })
            .collect();
        let Some(x) = gauss(&mut sys) else {
            continue;
        };
        for (&j, &xj) in free.iter().zip(&x) {
            if !(xj >= bounds[j].0 && xj <= bounds[j].1) {
                continue 'sets;
            }
            c[j] = xj;
        }
        let obj = objective(&c);
        if obj < best_obj {
            best_obj = obj;
            best = c;
        }
    }
    best
}

/// Solve an augmented `n × (n+1)` system by elimination with partial pivoting;
/// `None` when singular.
fn gauss(sys: &mut [Vec<f64>]) -> Option<Vec<f64>> {
    let n = sys.len();
    let tiny = 1e-12
        * sys
            .iter()
            .flat_map(|r| r[..n].iter())
            .fold(0.0_f64, |m, x| m.max(x.abs()));
    for col in 0..n {
        let p = (col..n).max_by(|&i, &j| sys[i][col].abs().total_cmp(&sys[j][col].abs()))?;
        if sys[p][col].abs() <= tiny {
            return None;
        }
        sys.swap(col, p);
        for i in col + 1..n {
            let f = sys[i][col] / sys[col][col];
            for k in col..=n {
                sys[i][k] -= f * sys[col][k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (sys[i][n] - (i + 1..n).map(|k| sys[i][k] * x[k]).sum::<f64>()) / sys[i][i];
    }
    Some(x)
}

/// Result of [`fit_coregionalization`]: the model and the achieved weighted
/// SSE of the sill-scaled residuals.
#[derive(Debug, Clone)]
pub struct CoregFit {
    pub coregionalization: Coregionalization,
    pub wsse: f64,
}

/// One pair's samples: lags along the major axis, original distances, their
/// direction indices, sill-scaled γ and counts.
#[derive(Clone)]
struct Pair {
    i: usize,
    j: usize,
    lags: Vec<f64>,
    dist: Vec<f64>,
    owner: Vec<usize>,
    gammas: Vec<f64>,
    counts: Vec<usize>,
}

/// Fit a linear model of coregionalization to the direct and cross
/// experimental variograms of `nvar` variables together.
///
/// `exps[i][j]` for `i <= j` holds the experimental variograms of the pair,
/// one per direction; the lower triangle is ignored, the diagonal is
/// required, and a missing cross pair is left to the positive semi-definite
/// constraint. Without `geometry` each cell holds one omnidirectional
/// variogram; with `(directions, anisotropy)` it holds one per direction
/// (azimuth, dip in degrees), and the shared anisotropy, bounded as in
/// [`fit_directional`], maps lags to distances along the major axis.
///
/// Each pair's residuals are divided by `√(sᵢ sⱼ)`, `sᵢ` the largest γ of
/// variable `i`, so variables count alike whatever their units; cross pairs
/// count twice, once per side of the symmetric matrix, and `ByCountOverGamma`
/// uses `√(γᵢᵢ γⱼⱼ)` of the model on them. For given ranges, each pair's
/// nugget and sills are solved exactly with non-negative direct sills; when
/// every matrix comes out positive semi-definite this is the optimum, and
/// otherwise the matrices are updated in turn, each by a weighted
/// least-squares step projected on the positive semi-definite cone by clipping
/// negative eigenvalues. With the same lags and counts for every pair each step
/// is the exact minimiser, and one variable gives [`fit_nested`]. The ranges,
/// and with `geometry` the angles and ratios, start from [`fit_nested`] or
/// [`fit_directional`] on the sill-scaled direct variograms and are refined
/// by coordinate descent, deterministically. `spec` takes one to three structures with optional range
/// bounds; the nugget is free or, with `Some((0, 0))`, absent, and sill bounds
/// are rejected.
pub fn fit_coregionalization(
    exps: &[Vec<Option<Vec<Experimental>>>],
    geometry: Option<(&[(f64, f64)], &AnisotropySpec)>,
    spec: &NestedSpec,
    weighting: Weighting,
) -> Result<CoregFit> {
    let nvar = exps.len();
    if nvar == 0 || exps.iter().any(|row| row.len() != nvar) {
        return Err(VarioError::InvalidParameters(
            "experimental variograms must form an nvar x nvar matrix".into(),
        ));
    }
    let nugget = match spec.nugget {
        None => true,
        Some((0.0, 0.0)) => false,
        Some(_) => {
            return Err(VarioError::InvalidParameters(
                "the nugget matrix is free or fixed at zero".into(),
            ));
        }
    };
    if spec.structures.iter().any(|s| s.sill.is_some()) {
        return Err(VarioError::InvalidParameters(
            "sill bounds do not apply to coregionalization matrices".into(),
        ));
    }
    let (_, ranges) = parameters(spec)?;
    let dirs = geometry.map_or(&[(0.0, 0.0)][..], |g| g.0);
    let reduce = |cell: &[Experimental]| -> Result<(Vec<usize>, Experimental)> {
        if cell.len() != dirs.len() {
            return Err(VarioError::InvalidParameters(format!(
                "{} experimental variograms for {} directions",
                cell.len(),
                dirs.len()
            )));
        }
        let mut e = Experimental {
            lags: Vec::new(),
            gammas: Vec::new(),
            counts: Vec::new(),
            covariances: None,
        };
        let mut owner = Vec::new();
        for (d, x) in cell.iter().enumerate() {
            owner.extend(std::iter::repeat_n(d, x.lags.len()));
            e.lags.extend(&x.lags);
            e.gammas.extend(&x.gammas);
            e.counts.extend(&x.counts);
        }
        if e.lags.is_empty() {
            return Err(VarioError::FittingFailed(
                "empty experimental variogram".into(),
            ));
        }
        Ok((owner, e))
    };

    let mut direct = Vec::with_capacity(nvar);
    for (i, row) in exps.iter().enumerate() {
        let cell = row[i].as_deref().ok_or_else(|| {
            VarioError::InvalidParameters(format!("missing direct variogram {i}"))
        })?;
        direct.push(reduce(cell)?);
    }
    let sills: Vec<f64> = direct
        .iter()
        .map(|(_, e)| e.gammas.iter().cloned().fold(0.0, f64::max))
        .collect();
    if sills.iter().any(|&s| !(s > 0.0 && s.is_finite())) {
        return Err(VarioError::FittingFailed(
            "direct variograms need a positive finite γ".into(),
        ));
    }
    let mut pairs = Vec::new();
    for (i, row) in exps.iter().enumerate() {
        for (j, cell) in row.iter().enumerate().skip(i) {
            let Some(cell) = cell else { continue };
            let (owner, e) = if i == j {
                direct[i].clone()
            } else {
                reduce(cell)?
            };
            let s = (sills[i] * sills[j]).sqrt();
            pairs.push(Pair {
                i,
                j,
                gammas: e.gammas.iter().map(|g| g / s).collect(),
                dist: e.lags.clone(),
                lags: e.lags,
                owner,
                counts: e.counts,
            });
        }
    }

    let (start, x, limits) = match geometry {
        None => {
            let mut pooled = Experimental {
                lags: Vec::new(),
                gammas: Vec::new(),
                counts: Vec::new(),
                covariances: None,
            };
            for p in pairs.iter().filter(|p| p.i == p.j) {
                pooled.lags.extend(&p.lags);
                pooled.gammas.extend(&p.gammas);
                pooled.counts.extend(&p.counts);
            }
            let start = fit_nested(&pooled, spec, weighting)?.variogram;
            let mut x = vec![0.0, 0.0, 0.0, 1.0, 1.0];
            x.extend(start.structures.iter().map(|s| s.range));
            let limits = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0)]
                .into_iter()
                .chain(ranges)
                .map(Some)
                .collect();
            (start, x, limits)
        }
        Some((_, aniso)) => {
            let mut scaled = Vec::new();
            for (i, row) in exps.iter().enumerate() {
                for e in row[i].iter().flatten() {
                    let mut e = e.clone();
                    e.gammas.iter_mut().for_each(|g| *g /= sills[i]);
                    scaled.push(e);
                }
            }
            let mut start =
                fit_directional(&scaled, &dirs.repeat(nvar), spec, aniso, weighting)?.variogram;
            let a = start
                .anisotropy
                .take()
                .expect("fit_directional sets it")
                .angles;
            let mut x = vec![a.azimuth, a.dip, a.rake, a.semi, a.minor];
            x.extend(start.structures.iter().map(|s| s.range));
            let reach = scaled
                .iter()
                .flat_map(|e| &e.lags)
                .fold(0.0, |m: f64, &h| m.max(h));
            (start, x, limits(dirs, aniso, spec, reach)?)
        }
    };
    let units = units(dirs);
    let eval = |x: &[f64]| {
        let k = stretch(x, &units);
        let stretched: Vec<Pair> = pairs
            .iter()
            .map(|p| Pair {
                lags: p
                    .dist
                    .iter()
                    .zip(&p.owner)
                    .map(|(h, &d)| h * k[d])
                    .collect(),
                ..p.clone()
            })
            .collect();
        solve_coreg(&stretched, nvar, nugget, spec, &x[5..], &start, weighting)
    };
    let (x, (wsse, blocks)) = descend(x, &limits, eval);

    let unscale = |b: &DMatrix<f64>| -> Vec<Vec<f64>> {
        (0..nvar)
            .map(|i| {
                (0..nvar)
                    .map(|j| b[(i, j)] * (sills[i] * sills[j]).sqrt())
                    .collect()
            })
            .collect()
    };
    let structures = spec
        .structures
        .iter()
        .zip(&x[5..])
        .zip(&blocks[1..])
        .map(|((s, &range), b)| CoregStructure {
            model: s.model,
            range,
            sills: unscale(b),
        })
        .collect();
    let mut coregionalization = Coregionalization::new(unscale(&blocks[0]), structures)?;
    coregionalization.anisotropy = match geometry {
        Some((_, aniso)) => Some(canonical(&x, &limits, aniso)?),
        None => None,
    };
    Ok(CoregFit {
        coregionalization,
        wsse,
    })
}

/// The nugget and structure matrices (sill-scaled) minimising the weighted SSE
/// for fixed ranges, exactly per pair or by block-wise projected steps; weights
/// depending on the
/// model are iterated as in [`fit_nested`].
fn solve_coreg(
    pairs: &[Pair],
    nvar: usize,
    nugget: bool,
    spec: &NestedSpec,
    ranges: &[f64],
    start: &Variogram,
    weighting: Weighting,
) -> (f64, Vec<DMatrix<f64>>) {
    let nb = ranges.len() + 1;
    let shapes: Vec<Vec<Vec<f64>>> = pairs
        .iter()
        .map(|p| {
            p.lags
                .iter()
                .map(|&h| {
                    std::iter::once(if h > 0.0 { 1.0 } else { 0.0 })
                        .chain(
                            spec.structures
                                .iter()
                                .zip(ranges)
                                .map(|(s, &r)| shape(s.model, h, r)),
                        )
                        .collect()
                })
                .collect()
        })
        .collect();
    let model = |b: &[DMatrix<f64>], p: usize, i: usize, j: usize, k: usize| {
        (0..nb).map(|s| b[s][(i, j)] * shapes[p][k][s]).sum::<f64>()
    };
    let doubled = |p: &Pair| if p.i == p.j { 1.0 } else { 2.0 };
    let weights = |b: Option<&[DMatrix<f64>]>| -> Vec<Vec<f64>> {
        pairs
            .iter()
            .enumerate()
            .map(|(pi, p)| {
                (0..p.lags.len())
                    .map(|k| {
                        let g = match b {
                            Some(b) => (model(b, pi, p.i, p.i, k) * model(b, pi, p.j, p.j, k))
                                .max(0.0)
                                .sqrt(),
                            None if p.i == p.j => p.gammas[k],
                            None => start.gamma(p.lags[k]),
                        };
                        doubled(p) * weight(weighting, p.counts[k], p.dist[k], g)
                    })
                    .collect()
            })
            .collect()
    };
    let objective = |b: &[DMatrix<f64>], w: &[Vec<f64>]| {
        pairs
            .iter()
            .enumerate()
            .map(|(pi, p)| {
                (0..p.lags.len())
                    .map(|k| {
                        let r = p.gammas[k] - model(b, pi, p.i, p.j, k);
                        w[pi][k] * r * r
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    };

    let mut b = Vec::new();
    let passes = if weighting == Weighting::ByCountOverGamma {
        4
    } else {
        1
    };
    let mut w = weights(None);
    for pass in 0..passes {
        if pass > 0 {
            w = weights(Some(&b));
        }
        let limits: Vec<Vec<Bounds>> = pairs
            .iter()
            .map(|p| {
                (0..nb)
                    .map(|s| match (s, p.i == p.j) {
                        (0, _) if !nugget => (0.0, 0.0),
                        (_, true) => (0.0, f64::INFINITY),
                        _ => (f64::NEG_INFINITY, f64::INFINITY),
                    })
                    .collect()
            })
            .collect();
        b = vec![DMatrix::<f64>::zeros(nvar, nvar); nb];
        for (pi, p) in pairs.iter().enumerate() {
            let coef = bounded_lsq(&shapes[pi], &p.gammas, &w[pi], &limits[pi]);
            for (s, c) in coef.into_iter().enumerate() {
                b[s][(p.i, p.j)] = c;
                b[s][(p.j, p.i)] = c;
            }
        }
        if b.iter().all(|m| m.symmetric_eigenvalues().min() >= 0.0) {
            continue;
        }
        b = b.into_iter().map(psd).collect();
        let (mut gram, mut cross, mut yy) = (
            vec![vec![vec![0.0; nb]; nb]; pairs.len()],
            vec![vec![0.0; nb]; pairs.len()],
            0.0,
        );
        for (pi, p) in pairs.iter().enumerate() {
            for (k, g) in shapes[pi].iter().enumerate() {
                yy += w[pi][k] * p.gammas[k] * p.gammas[k];
                for s in 0..nb {
                    cross[pi][s] += w[pi][k] * g[s] * p.gammas[k];
                    for t in 0..nb {
                        gram[pi][s][t] += w[pi][k] * g[s] * g[t];
                    }
                }
            }
        }
        let objective = |b: &[DMatrix<f64>]| {
            yy + pairs
                .iter()
                .enumerate()
                .map(|(pi, p)| {
                    (0..nb)
                        .map(|s| {
                            let c = b[s][(p.i, p.j)];
                            let quad: f64 =
                                (0..nb).map(|t| gram[pi][s][t] * b[t][(p.i, p.j)]).sum();
                            c * (quad - 2.0 * cross[pi][s])
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        };
        let mut prev = objective(&b);
        for _ in 0..5000 {
            for s in (0..nb).filter(|&s| s > 0 || nugget) {
                let mut a = DMatrix::<f64>::zeros(nvar, nvar);
                let mut target = b[s].clone();
                let mut rhs = DMatrix::<f64>::zeros(nvar, nvar);
                for (pi, p) in pairs.iter().enumerate() {
                    let rest: f64 = (0..nb)
                        .filter(|&t| t != s)
                        .map(|t| gram[pi][s][t] * b[t][(p.i, p.j)])
                        .sum();
                    a[(p.i, p.j)] += gram[pi][s][s];
                    rhs[(p.i, p.j)] += cross[pi][s] - rest;
                }
                let c_max = a.max();
                if c_max <= 0.0 {
                    continue;
                }
                for i in 0..nvar {
                    for j in i..nvar {
                        let c = a[(i, j)];
                        if c > 0.0 {
                            let step = b[s][(i, j)] + (rhs[(i, j)] / c - b[s][(i, j)]) * c / c_max;
                            target[(i, j)] = step;
                            target[(j, i)] = step;
                        }
                    }
                }
                b[s] = psd(target);
            }
            let cur = objective(&b);
            if cur <= 1e-30 || prev - cur <= 1e-13 * prev {
                break;
            }
            prev = cur;
        }
    }
    if passes > 1 {
        w = weights(Some(&b));
    }
    (objective(&b, &w), b)
}

/// The nearest positive semi-definite matrix in the Frobenius norm.
fn psd(m: DMatrix<f64>) -> DMatrix<f64> {
    let eig = SymmetricEigen::new(m);
    let clipped = eig.eigenvalues.map(|l| l.max(0.0));
    let v = &eig.eigenvectors;
    let p = v * DMatrix::from_diagonal(&clipped) * v.transpose();
    (&p + p.transpose()) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::empirical::{Estimator, LagBins, downhole, experimental};

    #[test]
    fn downhole_pairs_recover_the_nugget() {
        // Vertical holes 0.3 m apart, each an independent exponential process
        // (sill 1, range 20 m) plus a 0.3 nugget, sampled every metre: pairs
        // across holes would read the full sill at the shortest lags.
        use rand::{SeedableRng, rngs::StdRng};
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = StdRng::seed_from_u64(7);
        let (nugget, phi) = (0.3_f64, (-1.0_f64 / 20.0).exp());
        let (mut locs, mut values, mut holes) = (Vec::new(), Vec::new(), Vec::new());
        for hole in 0..400u32 {
            let mut z: f64 = StandardNormal.sample(&mut rng);
            for k in 0..60 {
                let e: f64 = StandardNormal.sample(&mut rng);
                z = phi * z + (1.0 - phi * phi).sqrt() * e;
                let noise: f64 = StandardNormal.sample(&mut rng);
                locs.push((0.3 * hole as f64, 0.0, -(k as f64)));
                values.push(z + nugget.sqrt() * noise);
                holes.push(hole * 7 % 400);
            }
        }
        let bins = LagBins {
            max_lag: 10.0,
            lag_width: 1.0,
        };
        let exp = downhole(&locs, &values, &holes, &bins, Estimator::Matheron, false).unwrap();
        assert!((exp.lags[0] - 1.0).abs() < 1e-9);
        let c0 = extrapolated_nugget(&exp, 3).unwrap();
        assert!((c0 - nugget).abs() < 0.03, "{c0}");
    }

    #[test]
    fn recovers_spherical_parameters() {
        // Generate synthetic samples from a known spherical model via a moving average,
        // then check the fit lands in a sensible neighborhood.
        // Here we build an experimental variogram directly from a known model.
        let truth = Variogram::single(Model::Spherical, 2.0, 120.0);
        let lags: Vec<f64> = (1..20).map(|i| i as f64 * 10.0).collect();
        let gammas: Vec<f64> = lags.iter().map(|&h| truth.gamma(h)).collect();
        let counts: Vec<usize> = lags.iter().map(|_| 100).collect();
        let exp = Experimental {
            lags,
            gammas,
            counts,
            covariances: None,
        };

        let fit = fit(&exp, Model::Spherical, Weighting::ByCount).unwrap();
        let s = &fit.variogram.structures[0];
        // Sill+nugget should approach 2.0; range near 120.
        assert!(
            (fit.variogram.total_sill() - 2.0).abs() < 0.3,
            "sill {}",
            fit.variogram.total_sill()
        );
        assert!((s.range - 120.0).abs() < 40.0, "range {}", s.range);
    }

    #[test]
    fn fits_experimental_from_data() {
        let locs: Vec<(f64, f64, f64)> = (0..30).map(|i| (i as f64 * 5.0, 0.0, 0.0)).collect();
        let vals: Vec<f64> = (0..30).map(|i| ((i as f64) * 0.3).sin()).collect();
        let bins = LagBins {
            max_lag: 100.0,
            lag_width: 10.0,
        };
        let exp = experimental(&locs, &vals, &bins, Estimator::Matheron, None, false).unwrap();
        let fit = fit(&exp, Model::Exponential, Weighting::ByCountOverGamma).unwrap();
        assert!(fit.wsse.is_finite());
        assert!(fit.variogram.total_sill() > 0.0);
    }

    fn exact(truth: &Variogram) -> Experimental {
        let lags: Vec<f64> = (1..=40).map(|i| i as f64 * 5.0).collect();
        Experimental {
            gammas: lags.iter().map(|&h| truth.gamma(h)).collect(),
            counts: lags.iter().map(|&h| 50 + (400.0 / h) as usize).collect(),
            lags,
            covariances: None,
        }
    }

    fn two_structures() -> Variogram {
        Variogram {
            nugget: 0.1,
            structures: vec![
                Structure::new(Model::Spherical, 0.5, 30.0),
                Structure::new(Model::Spherical, 0.4, 120.0),
            ],
            anisotropy: None,
        }
    }

    fn free(models: &[Model]) -> NestedSpec {
        NestedSpec {
            nugget: None,
            structures: models.iter().copied().map(StructureSpec::new).collect(),
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-4 * b.abs().max(1.0)
    }

    #[test]
    fn nested_recovers_exact_two_structure_model() {
        let truth = two_structures();
        let exp = exact(&truth);
        for weighting in [
            Weighting::Uniform,
            Weighting::ByCount,
            Weighting::ByCountOverGamma,
            Weighting::ByCountOverDistance,
        ] {
            let v = fit_nested(&exp, &free(&[Model::Spherical; 2]), weighting)
                .unwrap()
                .variogram;
            assert!(close(v.nugget, 0.1), "{weighting:?} nugget {}", v.nugget);
            for (got, want) in v.structures.iter().zip(&truth.structures) {
                assert!(close(got.sill, want.sill), "{weighting:?} {got:?}");
                assert!(close(got.range, want.range), "{weighting:?} {got:?}");
            }
        }
    }

    #[test]
    fn nested_recovers_mixed_shapes() {
        let truth = Variogram {
            nugget: 0.05,
            structures: vec![
                Structure::new(Model::Spherical, 0.6, 40.0),
                Structure::new(Model::Exponential, 0.35, 150.0),
            ],
            anisotropy: None,
        };
        let spec = free(&[Model::Spherical, Model::Exponential]);
        let v = fit_nested(&exact(&truth), &spec, Weighting::ByCount)
            .unwrap()
            .variogram;
        assert!(close(v.nugget, 0.05), "nugget {}", v.nugget);
        for (got, want) in v.structures.iter().zip(&truth.structures) {
            assert!(
                close(got.sill, want.sill) && close(got.range, want.range),
                "{got:?}"
            );
        }
    }

    #[test]
    fn nested_fixed_parameters_stay_fixed() {
        let exp = exact(&two_structures());
        let mut spec = free(&[Model::Spherical; 2]);
        spec.nugget = Some((0.2, 0.2));
        spec.structures[0].range = Some((25.0, 25.0));
        spec.structures[1].sill = Some((0.3, 0.3));
        let v = fit_nested(&exp, &spec, Weighting::ByCount)
            .unwrap()
            .variogram;
        assert_eq!(v.nugget, 0.2);
        assert_eq!(v.structures[0].range, 25.0);
        assert_eq!(v.structures[1].sill, 0.3);
    }

    #[test]
    fn nested_bounds_are_honoured() {
        let exp = exact(&two_structures());
        let mut spec = free(&[Model::Spherical; 2]);
        spec.nugget = Some((0.0, 0.05));
        spec.structures[0].sill = Some((0.55, 1.0));
        spec.structures[1].range = Some((80.0, 100.0));
        let v = fit_nested(&exp, &spec, Weighting::ByCount)
            .unwrap()
            .variogram;
        assert!(v.nugget <= 0.05);
        assert!(v.structures[0].sill >= 0.55 && v.structures[0].sill <= 1.0);
        assert!((80.0..=100.0).contains(&v.structures[1].range));
    }

    #[test]
    fn nested_one_structure_is_fit() {
        let exp = exact(&two_structures());
        for weighting in [Weighting::ByCount, Weighting::ByCountOverGamma] {
            let old = fit(&exp, Model::Spherical, weighting).unwrap();
            let new = fit_nested(&exp, &free(&[Model::Spherical]), weighting).unwrap();
            assert_eq!(old.wsse.to_bits(), new.wsse.to_bits());
            assert_eq!(
                old.variogram.nugget.to_bits(),
                new.variogram.nugget.to_bits()
            );
            assert_eq!(old.variogram.structures, new.variogram.structures);
        }
    }

    #[test]
    fn nested_is_deterministic() {
        let exp = exact(&two_structures());
        let spec = free(&[Model::Spherical, Model::Spherical, Model::Exponential]);
        let a = fit_nested(&exp, &spec, Weighting::ByCountOverGamma).unwrap();
        let b = fit_nested(&exp, &spec, Weighting::ByCountOverGamma).unwrap();
        assert_eq!(a.wsse.to_bits(), b.wsse.to_bits());
        assert_eq!(a.variogram.structures, b.variogram.structures);
    }

    #[test]
    fn nested_rejects_bad_specs() {
        let exp = exact(&two_structures());
        assert!(fit_nested(&exp, &free(&[]), Weighting::ByCount).is_err());
        assert!(fit_nested(&exp, &free(&[Model::Spherical; 4]), Weighting::ByCount).is_err());
        let mut spec = free(&[Model::Spherical; 2]);
        spec.nugget = Some((0.3, 0.1));
        assert!(fit_nested(&exp, &spec, Weighting::ByCount).is_err());
        spec.nugget = Some((f64::NAN, 1.0));
        assert!(fit_nested(&exp, &spec, Weighting::ByCount).is_err());
        spec.nugget = None;
        spec.structures[0].range = Some((0.0, 0.0));
        assert!(fit_nested(&exp, &spec, Weighting::ByCount).is_err());
        spec.structures[0].range = Some((100.0, 200.0));
        spec.structures[1].range = Some((10.0, 50.0));
        assert!(fit_nested(&exp, &spec, Weighting::ByCount).is_err());
    }

    fn sphere() -> Vec<(f64, f64)> {
        let mut dirs: Vec<(f64, f64)> = (0..6)
            .flat_map(|i| [-60.0, -30.0, 0.0, 30.0, 60.0].map(|dip| (i as f64 * 30.0, dip)))
            .collect();
        dirs.push((0.0, 90.0));
        dirs
    }

    fn along(truth: &Variogram, dirs: &[(f64, f64)]) -> Vec<Experimental> {
        dirs.iter()
            .map(|&(az, dip)| {
                let (x, y, z) = unit_vector(az, dip);
                let lags: Vec<f64> = (1..=30).map(|i| i as f64 * 5.0).collect();
                Experimental {
                    gammas: lags
                        .iter()
                        .map(|&h| truth.gamma_points(&(0.0, 0.0, 0.0), &(h * x, h * y, h * z)))
                        .collect(),
                    counts: lags.iter().map(|&h| 50 + (400.0 / h) as usize).collect(),
                    lags,
                    covariances: None,
                }
            })
            .collect()
    }

    fn rotated(azimuth: f64, dip: f64, rake: f64, semi: f64, minor: f64) -> Variogram {
        let angles = Angles {
            azimuth,
            dip,
            rake,
            major: 1.0,
            semi,
            minor,
        };
        two_structures().with_anisotropy(Anisotropy::new(angles).unwrap())
    }

    fn joint(truth: &Variogram, dirs: &[(f64, f64)], weighting: Weighting) -> Variogram {
        let spec = free(&[Model::Spherical; 2]);
        fit_directional(
            &along(truth, dirs),
            dirs,
            &spec,
            &AnisotropySpec::default(),
            weighting,
        )
        .unwrap()
        .variogram
    }

    fn assert_recovers(got: &Variogram, truth: &Variogram) {
        let (g, t) = (
            &got.anisotropy.as_ref().unwrap().angles,
            &truth.anisotropy.as_ref().unwrap().angles,
        );
        for (a, b) in [
            (g.azimuth, t.azimuth),
            (g.dip, t.dip),
            (g.rake, t.rake),
            (g.semi, t.semi),
            (g.minor, t.minor),
        ] {
            assert!((a - b).abs() < 1e-3, "{g:?}");
        }
        assert!(close(got.nugget, truth.nugget), "nugget {}", got.nugget);
        for (a, b) in got.structures.iter().zip(&truth.structures) {
            assert!(close(a.sill, b.sill) && close(a.range, b.range), "{a:?}");
        }
    }

    #[test]
    fn joint_fit_recovers_rotated_anisotropy() {
        let truth = rotated(35.0, 20.0, 50.0, 0.6, 0.25);
        for weighting in [Weighting::ByCount, Weighting::ByCountOverGamma] {
            assert_recovers(&joint(&truth, &sphere(), weighting), &truth);
        }
    }

    #[test]
    fn joint_fit_returns_the_canonical_symmetric_angles() {
        let truth = rotated(35.0, 20.0, 50.0, 0.6, 0.25);
        for same in [
            rotated(215.0, -20.0, 310.0, 0.6, 0.25),
            rotated(35.0, 20.0, 140.0, 0.25, 0.6),
        ] {
            assert_recovers(&joint(&same, &sphere(), Weighting::ByCount), &truth);
        }
    }

    #[test]
    fn joint_fit_caps_free_ranges_at_the_largest_lag() {
        let truth = rotated(35.0, 20.0, 50.0, 0.6, 0.25);
        let dirs = sphere();
        let mut exps = along(&truth, &dirs);
        for e in &mut exps {
            e.lags.truncate(20);
            e.gammas.truncate(20);
            e.counts.truncate(20);
        }
        let fit = |spec: &NestedSpec| {
            fit_directional(
                &exps,
                &dirs,
                spec,
                &AnisotropySpec::default(),
                Weighting::ByCount,
            )
            .unwrap()
            .variogram
        };
        let mut spec = free(&[Model::Spherical; 2]);
        assert!(fit(&spec).structures[1].range <= 100.0);
        spec.structures[1].range = Some((50.0, 300.0));
        assert_recovers(&fit(&spec), &truth);
    }

    #[test]
    fn joint_fit_of_isotropic_data_has_unit_ratios() {
        let v = joint(&two_structures(), &sphere(), Weighting::ByCount);
        let a = &v.anisotropy.unwrap().angles;
        assert!(
            (a.semi - 1.0).abs() < 1e-3 && (a.minor - 1.0).abs() < 1e-3,
            "{a:?}"
        );
    }

    #[test]
    fn joint_fit_of_horizontal_directions_is_two_dimensional() {
        let truth = rotated(170.0, 0.0, 0.0, 0.33, 1.0);
        let dirs: Vec<(f64, f64)> = (0..8).map(|i| (i as f64 * 22.5, 0.0)).collect();
        assert_recovers(&joint(&truth, &dirs, Weighting::ByCount), &truth);
    }

    #[test]
    fn joint_fit_is_deterministic() {
        let truth = rotated(35.0, 20.0, 50.0, 0.6, 0.25);
        let (a, b) = (
            joint(&truth, &sphere(), Weighting::ByCountOverGamma),
            joint(&truth, &sphere(), Weighting::ByCountOverGamma),
        );
        assert_eq!(a.structures, b.structures);
        assert_eq!(a.anisotropy.unwrap().angles, b.anisotropy.unwrap().angles);
    }

    #[test]
    fn joint_fit_honours_fixed_angles_and_rejects_bad_input() {
        let truth = rotated(35.0, 20.0, 50.0, 0.6, 0.25);
        let (dirs, spec) = (sphere(), free(&[Model::Spherical; 2]));
        let exps = along(&truth, &dirs);
        let aniso = AnisotropySpec {
            azimuth: Some((30.0, 30.0)),
            minor: Some((0.2, 0.3)),
            ..Default::default()
        };
        let v = fit_directional(&exps, &dirs, &spec, &aniso, Weighting::ByCount)
            .unwrap()
            .variogram;
        let a = v.anisotropy.unwrap().angles;
        assert!(a.azimuth == 30.0 && (0.2..=0.3).contains(&a.minor));
        let bad = |aniso: AnisotropySpec, dirs: &[(f64, f64)]| {
            fit_directional(&exps, dirs, &spec, &aniso, Weighting::ByCount).is_err()
        };
        assert!(bad(AnisotropySpec::default(), &dirs[1..]));
        let mut nan = dirs.clone();
        nan[0].1 = f64::NAN;
        assert!(bad(AnisotropySpec::default(), &nan));
        let flipped = AnisotropySpec {
            dip: Some((10.0, -10.0)),
            ..Default::default()
        };
        assert!(bad(flipped, &dirs));
        let zero = AnisotropySpec {
            semi: Some((0.0, 0.0)),
            ..Default::default()
        };
        assert!(bad(zero, &dirs));
    }

    fn lmc() -> Coregionalization {
        Coregionalization::new(
            vec![
                vec![0.1, 0.02, 0.0],
                vec![0.02, 0.2, 0.0],
                vec![0.0, 0.0, 0.05],
            ],
            vec![
                CoregStructure {
                    model: Model::Spherical,
                    range: 30.0,
                    sills: vec![
                        vec![0.5, 0.3, 0.1],
                        vec![0.3, 0.4, 0.05],
                        vec![0.1, 0.05, 0.3],
                    ],
                },
                CoregStructure {
                    model: Model::Spherical,
                    range: 120.0,
                    sills: vec![
                        vec![0.4, -0.2, 0.2],
                        vec![-0.2, 0.1, -0.1],
                        vec![0.2, -0.1, 0.1],
                    ],
                },
            ],
        )
        .unwrap()
    }

    /// Noise-free experimental variograms of `truth` along `dirs`, scaled by
    /// `units` per variable.
    fn cross_along(
        truth: &Coregionalization,
        dirs: &[(f64, f64)],
        units: &[f64],
    ) -> Vec<Vec<Option<Vec<Experimental>>>> {
        let n = truth.nvar;
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let cell = dirs
                            .iter()
                            .map(|&(az, dip)| {
                                let (x, y, z) = unit_vector(az, dip);
                                let lags: Vec<f64> = (1..=40).map(|k| k as f64 * 5.0).collect();
                                let o = (0.0, 0.0, 0.0);
                                Experimental {
                                    gammas: lags
                                        .iter()
                                        .map(|&h| {
                                            units[i]
                                                * units[j]
                                                * (truth.sill(i, j)
                                                    - truth.cross_cov(
                                                        i,
                                                        j,
                                                        &o,
                                                        &(h * x, h * y, h * z),
                                                    ))
                                        })
                                        .collect(),
                                    counts: lags
                                        .iter()
                                        .map(|&h| 50 + (400.0 / h) as usize)
                                        .collect(),
                                    lags,
                                    covariances: None,
                                }
                            })
                            .collect();
                        (i <= j).then_some(cell)
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_matrices(got: &Coregionalization, truth: &Coregionalization, units: &[f64]) {
        let pairs = std::iter::once((&got.nugget, &truth.nugget)).chain(
            got.structures
                .iter()
                .zip(&truth.structures)
                .map(|(g, t)| (&g.sills, &t.sills)),
        );
        for (g, t) in pairs {
            for i in 0..truth.nvar {
                for j in 0..truth.nvar {
                    let want = t[i][j] * units[i] * units[j];
                    let tol = 1e-4 * units[i] * units[j];
                    assert!((g[i][j] - want).abs() < tol, "{g:?} vs {t:?}");
                }
            }
        }
        for (g, t) in got.structures.iter().zip(&truth.structures) {
            assert!(close(g.range, t.range), "range {}", g.range);
        }
    }

    fn two_spherical() -> NestedSpec {
        free(&[Model::Spherical; 2])
    }

    fn min_eigenvalue(m: &[Vec<f64>]) -> f64 {
        let n = m.len();
        let d = DMatrix::from_fn(n, n, |i, j| m[i][j]);
        SymmetricEigen::new(d).eigenvalues.min()
    }

    #[test]
    fn coregionalization_recovers_a_known_model() {
        let truth = lmc();
        let units = [1.0, 1000.0, 0.01];
        let exps = cross_along(&truth, &[(0.0, 0.0)], &units);
        for weighting in [
            Weighting::Uniform,
            Weighting::ByCount,
            Weighting::ByCountOverGamma,
            Weighting::ByCountOverDistance,
        ] {
            let fit = fit_coregionalization(&exps, None, &two_spherical(), weighting).unwrap();
            assert_matrices(&fit.coregionalization, &truth, &units);
        }
    }

    #[test]
    fn coregionalization_recovers_a_fixed_anisotropy() {
        let truth = lmc().with_anisotropy(rotated(35.0, 20.0, 50.0, 0.6, 0.25).anisotropy.unwrap());
        let dirs = sphere();
        let units = [1.0, 1.0, 1.0];
        let exps = cross_along(&truth, &dirs, &units);
        let fixed = AnisotropySpec {
            azimuth: Some((35.0, 35.0)),
            dip: Some((20.0, 20.0)),
            rake: Some((50.0, 50.0)),
            semi: Some((0.6, 0.6)),
            minor: Some((0.25, 0.25)),
        };
        let fit = fit_coregionalization(
            &exps,
            Some((&dirs, &fixed)),
            &two_spherical(),
            Weighting::ByCount,
        )
        .unwrap();
        assert_matrices(&fit.coregionalization, &truth, &units);
        assert_eq!(
            fit.coregionalization.anisotropy.unwrap().angles,
            truth.anisotropy.unwrap().angles
        );
    }

    #[test]
    fn coregionalization_recovers_a_free_anisotropy() {
        let truth = rotated(35.0, 20.0, 50.0, 0.6, 0.25);
        let lmc = lmc().with_anisotropy(truth.anisotropy.clone().unwrap());
        let dirs = sphere();
        let units = [1.0, 1000.0, 0.01];
        let exps = cross_along(&lmc, &dirs, &units);
        let fit = fit_coregionalization(
            &exps,
            Some((&dirs, &AnisotropySpec::default())),
            &two_spherical(),
            Weighting::ByCount,
        )
        .unwrap()
        .coregionalization;
        assert_matrices(&fit, &lmc, &units);
        let (g, t) = (
            &fit.anisotropy.as_ref().unwrap().angles,
            &truth.anisotropy.as_ref().unwrap().angles,
        );
        for (a, b) in [
            (g.azimuth, t.azimuth),
            (g.dip, t.dip),
            (g.rake, t.rake),
            (g.semi, t.semi),
            (g.minor, t.minor),
        ] {
            assert!((a - b).abs() < 1e-3, "{g:?}");
        }
        for m in std::iter::once(&fit.nugget).chain(fit.structures.iter().map(|s| &s.sills)) {
            assert!(min_eigenvalue(m) >= -1e-12, "{m:?}");
        }
    }

    /// Noisy variograms whose cross pairs alone would give non-PSD matrices.
    fn noisy() -> Vec<Vec<Option<Vec<Experimental>>>> {
        let mut exps = cross_along(&lmc(), &[(0.0, 0.0)], &[1.0, 1.0, 1.0]);
        for (i, row) in exps.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                if let Some(cell) = cell {
                    for (k, g) in cell[0].gammas.iter_mut().enumerate() {
                        *g *= 1.0 + 0.2 * ((k * 7 + i * 3 + j) as f64).sin();
                        if i != j {
                            *g *= 1.8;
                        }
                    }
                }
            }
        }
        exps
    }

    #[test]
    fn coregionalization_matrices_are_positive_semidefinite() {
        let exps = noisy();
        for (weighting, nugget) in [
            (Weighting::ByCount, None),
            (Weighting::ByCountOverGamma, Some((0.0, 0.0))),
        ] {
            let mut spec = two_spherical();
            spec.nugget = nugget;
            let c = fit_coregionalization(&exps, None, &spec, weighting)
                .unwrap()
                .coregionalization;
            assert!(
                c.nugget
                    .iter()
                    .flatten()
                    .all(|&x| nugget.is_none() || x == 0.0)
            );
            for m in std::iter::once(&c.nugget).chain(c.structures.iter().map(|s| &s.sills)) {
                assert!(min_eigenvalue(m) >= -1e-12, "{m:?}");
            }
            assert!(Coregionalization::new(c.nugget.clone(), c.structures.clone()).is_ok());
        }
        let mut gap = exps.clone();
        gap[0][2] = None;
        let c = fit_coregionalization(&gap, None, &two_spherical(), Weighting::ByCount)
            .unwrap()
            .coregionalization;
        for s in &c.structures {
            assert!(min_eigenvalue(&s.sills) >= -1e-12);
        }
    }

    #[test]
    fn coregionalization_of_one_variable_is_fit_nested() {
        let exps = noisy();
        let one = vec![vec![exps[1][1].clone()]];
        let exp = &exps[1][1].as_ref().unwrap()[0];
        for weighting in [Weighting::ByCount, Weighting::ByCountOverGamma] {
            let v = fit_nested(exp, &two_spherical(), weighting)
                .unwrap()
                .variogram;
            let c = fit_coregionalization(&one, None, &two_spherical(), weighting)
                .unwrap()
                .coregionalization;
            let o = (0.0, 0.0, 0.0);
            for &h in &exp.lags {
                let gamma = c.sill(0, 0) - c.cross_cov(0, 0, &o, &(h, 0.0, 0.0));
                assert!(
                    (gamma - v.gamma(h)).abs() < 1e-6 * v.total_sill(),
                    "{c:?} {v:?}"
                );
            }
        }
    }

    #[test]
    fn coregionalization_is_deterministic() {
        let exps = noisy();
        let fit = || {
            fit_coregionalization(&exps, None, &two_spherical(), Weighting::ByCountOverGamma)
                .unwrap()
        };
        let (a, b) = (fit(), fit());
        assert_eq!(a.wsse.to_bits(), b.wsse.to_bits());
        assert_eq!(
            serde_json::to_string(&a.coregionalization).unwrap(),
            serde_json::to_string(&b.coregionalization).unwrap()
        );
    }

    #[test]
    fn coregionalization_rejects_bad_input() {
        let exps = noisy();
        let spec = two_spherical();
        let bad = |e: &[Vec<Option<Vec<Experimental>>>], spec: &NestedSpec| {
            fit_coregionalization(e, None, spec, Weighting::ByCount).is_err()
        };
        let mut missing = exps.clone();
        missing[1][1] = None;
        assert!(bad(&missing, &spec));
        assert!(bad(&exps[..2], &spec));
        let mut sill = spec.clone();
        sill.structures[0].sill = Some((0.1, 1.0));
        assert!(bad(&exps, &sill));
        let mut nugget = spec.clone();
        nugget.nugget = Some((0.0, 0.1));
        assert!(bad(&exps, &nugget));
        let dirs = [(0.0, 0.0), (90.0, 0.0)];
        let a = AnisotropySpec::default();
        assert!(
            fit_coregionalization(&exps, Some((&dirs, &a)), &spec, Weighting::ByCount).is_err()
        );
    }
}
