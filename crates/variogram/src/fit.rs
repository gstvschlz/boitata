//! Automatic variogram-model fitting by weighted least squares.

use crate::composite::Variogram;
use crate::empirical::Experimental;
use crate::error::{Result, VarioError};
use crate::model::{Model, Structure, shape};
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
    let mut acc = 0.0;
    for i in 0..exp.lags.len() {
        let h = exp.lags[i];
        let model_g = v.gamma(h);
        let resid = exp.gammas[i] - model_g;
        acc += weight(weighting, exp.counts[i], h, model_g) * resid * resid;
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
    if !(1..=3).contains(&n) {
        return Err(VarioError::InvalidParameters(format!(
            "fit one to three structures, got {n}"
        )));
    }
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

    let linear = std::iter::once(bounds(spec.nugget, "nugget")?)
        .chain(
            spec.structures
                .iter()
                .map(|s| bounds(s.sill, "sill"))
                .collect::<Result<Vec<_>>>()?,
        )
        .collect::<Vec<_>>();
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
    let ordered = |r: &[f64]| r.windows(2).all(|w| w[0] < w[1]);
    let eval = |r: &[f64]| {
        let v = solve(exp, spec, &linear, r, weighting);
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
            .map(|i| weight(weighting, exp.counts[i], exp.lags[i], model_g[i]))
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
    let mut best: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let mut best_obj = objective(&best);
    'sets: for code in 0..3usize.pow(m as u32) {
        let mut c = vec![0.0; m];
        let mut free = Vec::new();
        let mut rest = code;
        for (j, &(lo, hi)) in bounds.iter().enumerate() {
            match rest % 3 {
                0 => c[j] = lo,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::empirical::{Estimator, LagBins, experimental};

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
}
