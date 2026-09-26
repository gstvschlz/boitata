//! Drillhole data processing: desurvey and compositing.
//!
//! Handles wellbore survey-to-coordinates conversion (minimum curvature,
//! tangential, balanced tangential) and domain-aware sample compositing with
//! length-weighted averaging.

mod error;
pub use error::{DrillholeError, Result};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wellbore collar (surface location).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Collar {
    pub hole_id: String,
    pub east: f64,
    pub north: f64,
    pub collar_elev: f64,
}

/// Survey station at measured depth with azimuth and inclination.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurveyStation {
    pub hole_id: String,
    pub depth: f64,
    pub azimuth: f64,     // degrees, 0-360
    pub inclination: f64, // degrees, 0 = vertical
}

/// Desurvey method: how the hole runs between two survey stations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DesurveyMethod {
    /// Circular arc tangent to both stations.
    #[default]
    MinimumCurvature,
    /// Straight line along the upper station's direction.
    Tangential,
    /// Half the segment along each station's direction.
    BalancedTangential,
}

/// Desurvey wellbore by `method`.
///
/// # Algorithm
/// Each segment between stations i and i+1 is displaced by
/// `w₁·t₁ + w₂·t₂`, `t` the unit direction at each station and `Δmd` the
/// segment length: tangential takes `(Δmd, 0)`, balanced tangential
/// `(Δmd/2, Δmd/2)`, and minimum curvature scales the balanced weights by
/// `2/β · tan(β/2)` of the dogleg `β`, which puts the segment on a circular
/// arc.
///
/// # References
/// - Marschall, F. (1988). "Calculation of Hole Trajectories"
/// - Convention: azimuth from North (0-360°), inclination from vertical (0-90°)
pub fn desurvey_wellbore(
    collar: &Collar,
    stations: &[SurveyStation],
    method: DesurveyMethod,
) -> Result<Vec<WellborePoint>> {
    if stations.is_empty() {
        return Err(DrillholeError::InvalidSurveyStation(
            "At least one survey station required".to_string(),
        ));
    }

    for station in stations {
        if station.hole_id != collar.hole_id {
            return Err(DrillholeError::InvalidSurveyStation(format!(
                "Station {} belongs to hole {}, expected {}",
                station.depth, station.hole_id, collar.hole_id
            )));
        }
    }

    let mut sorted = stations.to_vec();
    sorted.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());

    let mut points = vec![WellborePoint {
        measured_depth: 0.0,
        east: collar.east,
        north: collar.north,
        elev: collar.collar_elev,
    }];

    if !sorted.is_empty() {
        let s1 = &sorted[0];
        let from_survey = SurveyStation {
            hole_id: collar.hole_id.clone(),
            depth: 0.0,
            azimuth: s1.azimuth,
            inclination: s1.inclination,
        };

        let segment = desurvey_segment(&points[points.len() - 1], &from_survey, s1, method)?;
        points.extend(segment);

        for i in 1..sorted.len() {
            let segment = desurvey_segment(
                &points[points.len() - 1],
                &sorted[i - 1],
                &sorted[i],
                method,
            )?;
            points.extend(segment);
        }
    }

    Ok(points)
}

fn desurvey_segment(
    prev_point: &WellborePoint,
    from: &SurveyStation,
    to: &SurveyStation,
    method: DesurveyMethod,
) -> Result<Vec<WellborePoint>> {
    let md_from = from.depth;
    let md_to = to.depth;
    let delta_md = md_to - md_from;

    if delta_md <= 0.0 {
        return Ok(vec![]);
    }

    let (a1, i1) = (from.azimuth.to_radians(), from.inclination.to_radians());
    let (a2, i2) = (to.azimuth.to_radians(), to.inclination.to_radians());
    let (w1, w2) = match method {
        DesurveyMethod::Tangential => (delta_md, 0.0),
        DesurveyMethod::BalancedTangential => (delta_md / 2.0, delta_md / 2.0),
        DesurveyMethod::MinimumCurvature => {
            let cos_dogleg = (i2 - i1).cos() - i1.sin() * i2.sin() * (1.0 - (a2 - a1).cos());
            let dogleg = cos_dogleg.clamp(-1.0, 1.0).acos();
            let ratio = if dogleg < 1e-9 {
                1.0
            } else {
                2.0 / dogleg * (dogleg / 2.0).tan()
            };
            (delta_md / 2.0 * ratio, delta_md / 2.0 * ratio)
        }
    };
    let delta_e = w1 * i1.sin() * a1.sin() + w2 * i2.sin() * a2.sin();
    let delta_n = w1 * i1.sin() * a1.cos() + w2 * i2.sin() * a2.cos();
    let delta_v = -(w1 * i1.cos() + w2 * i2.cos());

    Ok(vec![WellborePoint {
        measured_depth: md_to,
        east: prev_point.east + delta_e,
        north: prev_point.north + delta_n,
        elev: prev_point.elev + delta_v,
    }])
}

/// Splits two interval tables of one hole at the union of their boundaries.
///
/// Each output piece is `(from, to, a, b)` with the index of the `a` and `b`
/// interval covering it, `None` where that table has no interval. Pieces
/// covered by neither are dropped. Intervals within a table must not overlap.
pub fn merge_intervals(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
) -> Vec<(f64, f64, Option<usize>, Option<usize>)> {
    let sorted = |t: &[(f64, f64)]| {
        let mut idx: Vec<usize> = (0..t.len()).collect();
        idx.sort_by(|&i, &j| t[i].0.total_cmp(&t[j].0));
        idx
    };
    let (ia, ib) = (sorted(a), sorted(b));
    let covering = |t: &[(f64, f64)], idx: &[usize], x: f64| {
        let k = idx.partition_point(|&i| t[i].0 <= x);
        (k > 0 && t[idx[k - 1]].1 > x).then(|| idx[k - 1])
    };
    let mut cuts: Vec<f64> = a.iter().chain(b).flat_map(|&(f, t)| [f, t]).collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    cuts.windows(2)
        .filter_map(|w| {
            let mid = (w[0] + w[1]) / 2.0;
            let (x, y) = (covering(a, &ia, mid), covering(b, &ib, mid));
            (x.is_some() || y.is_some()).then_some((w[0], w[1], x, y))
        })
        .collect()
}

/// Location at measured `depth` along a desurveyed `path`, interpolated
/// linearly between stations and extended along the last segment.
pub fn position_at(path: &[WellborePoint], depth: f64) -> (f64, f64, f64) {
    let at = |p: &WellborePoint| (p.east, p.north, p.elev);
    match path {
        [] => (f64::NAN, f64::NAN, f64::NAN),
        [only] => at(only),
        _ => {
            let i = path
                .partition_point(|p| p.measured_depth <= depth)
                .clamp(1, path.len() - 1);
            let (a, b) = (&path[i - 1], &path[i]);
            let t = (depth - a.measured_depth) / (b.measured_depth - a.measured_depth);
            (
                a.east + t * (b.east - a.east),
                a.north + t * (b.north - a.north),
                a.elev + t * (b.elev - a.elev),
            )
        }
    }
}

/// Wellbore point (3D location at measured depth).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WellborePoint {
    pub measured_depth: f64,
    pub east: f64,
    pub north: f64,
    pub elev: f64,
}

/// Composite (aggregated interval respecting domain boundaries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Composite {
    pub hole_id: String,
    pub from_depth: f64,
    pub to_depth: f64,
    /// Sampled length actually accumulated, which is `to_depth - from_depth`
    /// minus any unsampled gaps the composite spans.
    pub length: f64,
    pub domain: String,
    /// Length-weighted mean per grade column. A column is **absent** when no
    /// sampled length in this composite carried a value for it — an unsampled
    /// interval must not be read as a zero grade. Categorical columns hold the
    /// code covering the most length.
    pub attributes: HashMap<String, f64>,
}

/// What happens to the short tail of a run in length compositing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Residual {
    #[default]
    Keep,
    Drop,
    /// Absorbed into the previous composite of the run, if there is one.
    Merge,
}

/// Compositing parameters.
#[derive(Debug, Clone)]
pub struct CompositeParams {
    /// Target composite length (meters); `f64::INFINITY` gives one composite
    /// per run of a domain.
    pub composite_length: f64,
    /// Domain column name in assay data
    pub domain_column: String,
    /// Grade/numeric columns to composite (length-weighted average)
    pub grade_columns: Vec<String>,
    /// Columns of category codes, composited to their length-weighted majority.
    pub categorical_columns: Vec<String>,
    /// Non-overlapping `(from, to)` intervals to composite to instead of
    /// `composite_length`, e.g. benches; ground outside them is left out.
    pub intervals: Option<Vec<(f64, f64)>>,
    /// Rule for a run's tail shorter than `min_fraction × composite_length`.
    pub residual: Residual,
    pub min_fraction: f64,
}

impl Default for CompositeParams {
    fn default() -> Self {
        Self {
            composite_length: 2.0,
            domain_column: "domain".to_string(),
            grade_columns: vec!["grade".to_string()],
            categorical_columns: vec![],
            intervals: None,
            residual: Residual::Keep,
            min_fraction: 0.5,
        }
    }
}

/// Length tolerance for closing a composite, in metres. Splitting produces
/// exact arithmetic in principle, but repeated subtraction leaves crumbs; this
/// keeps a 2 m target from emitting a 2 m composite plus a 1e-16 m sliver.
const LENGTH_EPS: f64 = 1e-9;

/// Composite assay intervals into regular downhole lengths, never averaging
/// across a domain boundary.
///
/// # Algorithm
/// 1. Reject non-finite or inverted intervals, then sort by depth.
/// 2. Split the hole into runs of constant domain — a composite never spans a
///    contact, because averaging grade across one destroys the very boundary
///    domained estimation exists to honour.
/// 3. Within a run, walk the sampled length and **cut intervals** at every
///    `composite_length` boundary, so composites come out at the target length
///    rather than at whatever length the assay intervals happen to sum to.
/// 4. Length-weight each grade column over the length that actually carried a
///    value for it, so unsampled core dilutes nothing.
/// 5. Emit the tail of each run at its true (short) length, or drop or merge
///    it by `residual` when shorter than `min_fraction × composite_length`.
///
/// With `intervals`, step 3 cuts at their boundaries instead and each piece of
/// an interval within one run becomes a composite.
///
/// Unsampled gaps inside a run are skipped, not bridged: `length` counts only
/// sampled ground, so a composite spanning a gap reports a `length` shorter
/// than `to_depth - from_depth`.
pub fn composite_intervals(
    hole_id: &str,
    assays: &[(f64, f64, String, HashMap<String, f64>)],
    params: &CompositeParams,
) -> Result<Vec<Composite>> {
    if assays.is_empty() {
        return Ok(vec![]);
    }
    if params.composite_length.is_nan() || params.composite_length <= 0.0 {
        return Err(DrillholeError::CompositeError(format!(
            "composite length must be a positive number, got {}",
            params.composite_length
        )));
    }
    if !(params.min_fraction.is_finite() && params.min_fraction >= 0.0) {
        return Err(DrillholeError::CompositeError(format!(
            "minimum fraction must be a non-negative number, got {}",
            params.min_fraction
        )));
    }
    let mut targets = params.intervals.clone().unwrap_or_default();
    targets.sort_by(|a, b| a.0.total_cmp(&b.0));
    if targets
        .iter()
        .any(|(f, t)| !(f.is_finite() && t > f && t.is_finite()))
        || targets.windows(2).any(|w| w[1].0 < w[0].1 - LENGTH_EPS)
    {
        return Err(DrillholeError::CompositeError(
            "target intervals must be finite, non-empty and non-overlapping".to_string(),
        ));
    }
    for (from, to, _, _) in assays {
        if !from.is_finite() || !to.is_finite() {
            return Err(DrillholeError::CompositeError(
                "interval depths must be finite numbers".to_string(),
            ));
        }
        if to < from {
            return Err(DrillholeError::CompositeError(format!(
                "interval ends above its start ({from} to {to})"
            )));
        }
    }

    let mut sorted = assays.to_vec();
    // total_cmp, not partial_cmp().unwrap() — the latter panics on a NaN depth,
    // which is a data problem the caller should see as an error. (Depths are
    // already checked finite above; this keeps the sort total regardless.)
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut composites = Vec::new();
    let mut builder = CompositeBuilder::new(hole_id, params);

    for (index, (from, to, domain, attrs)) in sorted.iter().enumerate() {
        let starts_new_domain = index > 0 && domain != &sorted[index - 1].2;
        if starts_new_domain {
            builder.end_run(&mut composites);
        }
        builder.domain.clone_from(domain);

        // Walk the interval, cutting it wherever a composite boundary falls.
        let mut cursor = *from;
        while cursor < *to - LENGTH_EPS {
            if params.intervals.is_none() {
                let take = (params.composite_length - builder.length()).min(*to - cursor);
                builder.accumulate(cursor, take, attrs);
                cursor += take;
                if builder.length() >= params.composite_length - LENGTH_EPS {
                    builder.flush();
                }
                continue;
            }
            let k = targets.partition_point(|t| t.0 <= cursor + LENGTH_EPS);
            if k > 0 && targets[k - 1].1 > cursor + LENGTH_EPS {
                if builder.target != Some(k - 1) {
                    builder.flush();
                    builder.target = Some(k - 1);
                }
                let end = targets[k - 1].1.min(*to);
                builder.accumulate(cursor, end - cursor, attrs);
                cursor = end;
            } else {
                cursor = targets.get(k).map_or(*to, |t| t.0.min(*to));
            }
        }
    }
    builder.end_run(&mut composites);

    Ok(composites)
}

/// One composite's accumulated ground.
#[derive(Default)]
struct Part {
    from: f64,
    to: f64,
    /// Sampled length.
    length: f64,
    /// Per grade column: (Σ value × length, length that carried a value).
    sums: HashMap<String, (f64, f64)>,
    /// Per categorical column: length carried by each code (as bits).
    votes: HashMap<String, HashMap<u64, f64>>,
}

impl Part {
    fn absorb(&mut self, other: Part) {
        self.to = other.to;
        self.length += other.length;
        for (col, (sum, len)) in other.sums {
            let entry = self.sums.entry(col).or_default();
            entry.0 += sum;
            entry.1 += len;
        }
        for (col, votes) in other.votes {
            let entry = self.votes.entry(col).or_default();
            for (code, len) in votes {
                *entry.entry(code).or_default() += len;
            }
        }
    }
}

/// Accumulates the composites of one run of a domain.
struct CompositeBuilder<'a> {
    hole_id: &'a str,
    params: &'a CompositeParams,
    domain: String,
    /// Target interval the current part belongs to.
    target: Option<usize>,
    part: Option<Part>,
    run: Vec<Part>,
}

impl<'a> CompositeBuilder<'a> {
    fn new(hole_id: &'a str, params: &'a CompositeParams) -> Self {
        Self {
            hole_id,
            params,
            domain: String::new(),
            target: None,
            part: None,
            run: Vec::new(),
        }
    }

    fn length(&self) -> f64 {
        self.part.as_ref().map_or(0.0, |p| p.length)
    }

    /// Adds `take` metres of an interval starting at `at`.
    fn accumulate(&mut self, at: f64, take: f64, attrs: &HashMap<String, f64>) {
        if take <= 0.0 {
            return;
        }
        // Started lazily so a composite starts where its first sampled metre
        // is, not at the boundary of a gap that preceded it.
        let part = self.part.get_or_insert_with(|| Part {
            from: at,
            ..Part::default()
        });
        part.to = at + take;
        part.length += take;
        for col in &self.params.grade_columns {
            if let Some(&value) = attrs.get(col)
                && value.is_finite()
            {
                let entry = part.sums.entry(col.clone()).or_default();
                entry.0 += value * take;
                entry.1 += take;
            }
        }
        for col in &self.params.categorical_columns {
            if let Some(&code) = attrs.get(col)
                && code.is_finite()
            {
                *part
                    .votes
                    .entry(col.clone())
                    .or_default()
                    .entry(code.to_bits())
                    .or_default() += take;
            }
        }
    }

    /// Closes the in-progress composite.
    fn flush(&mut self) {
        if let Some(part) = self.part.take() {
            self.run.push(part);
        }
    }

    /// Closes the run, applies the residual rule and emits its composites.
    fn end_run(&mut self, out: &mut Vec<Composite>) {
        self.flush();
        self.target = None;
        let p = self.params;
        let short = |part: &Part| part.length < p.min_fraction * p.composite_length - LENGTH_EPS;
        if p.intervals.is_none()
            && p.composite_length.is_finite()
            && self.run.last().is_some_and(short)
        {
            match p.residual {
                Residual::Keep => {}
                Residual::Drop => {
                    self.run.pop();
                }
                Residual::Merge => {
                    if self.run.len() > 1
                        && let Some(last) = self.run.pop()
                        && let Some(prev) = self.run.last_mut()
                    {
                        prev.absorb(last);
                    }
                }
            }
        }
        out.extend(self.run.drain(..).map(|part| {
            let mut attributes: HashMap<String, f64> = part
                .sums
                .into_iter()
                .filter(|(_, (_, len))| *len > 0.0)
                .map(|(col, (sum, len))| (col, sum / len))
                .collect();
            for (col, votes) in part.votes {
                let majority = votes
                    .into_iter()
                    .map(|(code, len)| (f64::from_bits(code), len))
                    .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.total_cmp(&a.0)));
                if let Some((code, _)) = majority {
                    attributes.insert(col, code);
                }
            }
            Composite {
                hole_id: self.hole_id.to_string(),
                from_depth: part.from,
                to_depth: part.to,
                length: part.length,
                domain: self.domain.clone(),
                attributes,
            }
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hole(stations: &[(f64, f64, f64)]) -> Vec<WellborePoint> {
        hole_by(stations, DesurveyMethod::MinimumCurvature)
    }

    fn hole_by(stations: &[(f64, f64, f64)], method: DesurveyMethod) -> Vec<WellborePoint> {
        let collar = Collar {
            hole_id: "H".into(),
            east: 0.0,
            north: 0.0,
            collar_elev: 0.0,
        };
        let stations: Vec<SurveyStation> = stations
            .iter()
            .map(|&(depth, azimuth, inclination)| SurveyStation {
                hole_id: "H".into(),
                depth,
                azimuth,
                inclination,
            })
            .collect();
        desurvey_wellbore(&collar, &stations, method).unwrap()
    }

    const METHODS: [DesurveyMethod; 3] = [
        DesurveyMethod::MinimumCurvature,
        DesurveyMethod::Tangential,
        DesurveyMethod::BalancedTangential,
    ];

    #[test]
    fn straight_holes_agree_for_every_method() {
        for stations in [
            [(0.0, 0.0, 0.0), (40.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
            [
                (0.0, 135.0, 60.0),
                (40.0, 135.0, 60.0),
                (100.0, 135.0, 60.0),
            ],
        ] {
            let reference = hole(&stations);
            for method in METHODS {
                for (a, b) in reference.iter().zip(hole_by(&stations, method)) {
                    assert!(
                        (a.east - b.east).abs() < 1e-9
                            && (a.north - b.north).abs() < 1e-9
                            && (a.elev - b.elev).abs() < 1e-9,
                        "{method:?}"
                    );
                }
            }
        }
    }

    /// On a true circular arc minimum curvature is exact; balanced tangential
    /// sits between it and tangential in departure and in error.
    #[test]
    fn balanced_tangential_lies_between_on_a_curve() {
        let r = 100.0;
        let stations: Vec<(f64, f64, f64)> = (0..=9)
            .map(|k| (r * (k as f64 * 10.0).to_radians(), 90.0, k as f64 * 10.0))
            .collect();
        let end = |m| hole_by(&stations, m).last().cloned().unwrap();
        let (mc, t, bt) = (
            end(DesurveyMethod::MinimumCurvature),
            end(DesurveyMethod::Tangential),
            end(DesurveyMethod::BalancedTangential),
        );
        assert!((mc.east - r).abs() < 1e-9 && (mc.elev + r).abs() < 1e-9);
        assert!(t.east < bt.east && bt.east < mc.east);
        let error = |p: &WellborePoint| ((p.east - r).powi(2) + (p.elev + r).powi(2)).sqrt();
        assert!(error(&t) > error(&bt) && error(&bt) > 1e-3);
    }

    #[test]
    fn merge_splits_at_every_boundary() {
        let assay = [(0.0, 2.0), (2.0, 5.0), (7.0, 8.0)];
        let geology = [(0.0, 3.0), (3.0, 8.0)];
        let merged = merge_intervals(&assay, &geology);
        let expect = vec![
            (0.0, 2.0, Some(0), Some(0)),
            (2.0, 3.0, Some(1), Some(0)),
            (3.0, 5.0, Some(1), Some(1)),
            (5.0, 7.0, None, Some(1)),
            (7.0, 8.0, Some(2), Some(1)),
        ];
        assert_eq!(merged, expect);
    }

    #[test]
    fn straight_inclined_hole() {
        let path = hole(&[(0.0, 90.0, 45.0), (100.0, 90.0, 45.0)]);
        let end = path.last().unwrap();
        let d = 100.0 / 2f64.sqrt();
        assert!(
            (end.east - d).abs() < 1e-9 && end.north.abs() < 1e-9 && (end.elev + d).abs() < 1e-9
        );
    }

    #[test]
    fn quarter_circle_arc() {
        let r = 100.0;
        let path = hole(&[
            (0.0, 90.0, 0.0),
            (r * std::f64::consts::FRAC_PI_2, 90.0, 90.0),
        ]);
        let end = path.last().unwrap();
        assert!((end.east - r).abs() < 1e-9 && (end.elev + r).abs() < 1e-9);
    }

    #[test]
    fn collar_segment_follows_the_first_survey() {
        let path = hole(&[(50.0, 0.0, 30.0)]);
        let end = path.last().unwrap();
        assert!(
            (end.north - 25.0).abs() < 1e-9
                && (end.elev + 50.0 * 30f64.to_radians().cos()).abs() < 1e-9
        );
    }

    #[test]
    fn position_between_and_beyond_stations() {
        let path = hole(&[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]);
        assert_eq!(position_at(&path, 25.0), (0.0, 0.0, -25.0));
        assert_eq!(position_at(&path, 120.0), (0.0, 0.0, -120.0));
    }

    #[test]
    fn test_vertical_well_desurvey() {
        let collar = Collar {
            hole_id: "BH001".to_string(),
            east: 1000.0,
            north: 2000.0,
            collar_elev: 500.0,
        };

        let station = SurveyStation {
            hole_id: "BH001".to_string(),
            depth: 50.0,
            azimuth: 0.0,
            inclination: 0.0,
        };

        let result = desurvey_wellbore(&collar, &[station], DesurveyMethod::MinimumCurvature);
        assert!(result.is_ok());

        let traj = result.unwrap();
        assert_eq!(traj.len(), 2);
        assert_eq!(traj[0].measured_depth, 0.0);
        assert_eq!(traj[1].measured_depth, 50.0);
    }

    #[test]
    fn test_empty_compositing() {
        let params = CompositeParams::default();
        let result = composite_intervals("BH001", &[], &params);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    /// One assay interval: `from`, `to`, domain, optional grade.
    fn assay(
        from: f64,
        to: f64,
        domain: &str,
        grade: Option<f64>,
    ) -> (f64, f64, String, HashMap<String, f64>) {
        let mut attrs = HashMap::new();
        if let Some(g) = grade {
            attrs.insert("grade".to_string(), g);
        }
        (from, to, domain.to_string(), attrs)
    }

    fn params(composite_length: f64) -> CompositeParams {
        CompositeParams {
            composite_length,
            ..CompositeParams::default()
        }
    }

    fn metal(composites: &[Composite]) -> f64 {
        composites
            .iter()
            .map(|c| c.attributes["grade"] * c.length)
            .sum()
    }

    /// Two domains, 7.3 m of "a" then 2.7 m of "b", irregular sampling.
    fn two_domains() -> (Vec<(f64, f64, String, HashMap<String, f64>)>, f64) {
        let rows = [
            (0.0, 1.3, "a", 1.0),
            (1.3, 2.9, "a", 4.0),
            (2.9, 5.0, "a", 2.0),
            (5.0, 7.3, "a", 0.5),
            (7.3, 8.0, "b", 3.0),
            (8.0, 10.0, "b", 6.0),
        ];
        let metal = rows.iter().map(|r| (r.1 - r.0) * r.3).sum();
        (
            rows.iter()
                .map(|r| assay(r.0, r.1, r.2, Some(r.3)))
                .collect(),
            metal,
        )
    }

    #[test]
    fn whole_runs_give_one_composite_per_domain_and_conserve_metal() {
        let (assays, expected) = two_domains();
        let c = composite_intervals("H", &assays, &params(f64::INFINITY)).unwrap();
        let spans: Vec<_> = c.iter().map(|c| (c.from_depth, c.to_depth)).collect();
        assert_eq!(spans, vec![(0.0, 7.3), (7.3, 10.0)]);
        assert!((metal(&c) - expected).abs() < 1e-9);
    }

    #[test]
    fn composites_to_given_intervals_split_at_contacts_and_conserve_metal() {
        let (assays, expected) = two_domains();
        let p = CompositeParams {
            intervals: Some(vec![(5.0, 10.0), (0.0, 5.0)]),
            ..params(2.0)
        };
        let c = composite_intervals("H", &assays, &p).unwrap();
        let spans: Vec<_> = c.iter().map(|c| (c.from_depth, c.to_depth)).collect();
        assert_eq!(spans, vec![(0.0, 5.0), (5.0, 7.3), (7.3, 10.0)]);
        assert!((metal(&c) - expected).abs() < 1e-9);

        let p = CompositeParams {
            intervals: Some(vec![(1.0, 3.0), (6.0, 9.0)]),
            ..params(2.0)
        };
        let c = composite_intervals("H", &assays, &p).unwrap();
        let inside = 0.3 * 1.0 + 1.6 * 4.0 + 0.1 * 2.0 + 1.3 * 0.5 + 0.7 * 3.0 + 1.0 * 6.0;
        assert!((metal(&c) - inside).abs() < 1e-9);

        let p = CompositeParams {
            intervals: Some(vec![(0.0, 5.0), (4.0, 6.0)]),
            ..params(2.0)
        };
        assert!(composite_intervals("H", &assays, &p).is_err());
    }

    #[test]
    fn residual_rules_conserve_metal() {
        let (assays, expected) = two_domains();
        let rule = |residual| CompositeParams {
            residual,
            min_fraction: 0.5,
            ..params(3.0)
        };
        // Runs of 7.3 m and 2.7 m at 3 m: tails of 1.3 m (short) and 2.7 m.
        let keep = composite_intervals("H", &assays, &rule(Residual::Keep)).unwrap();
        assert_eq!(keep.len(), 4);
        assert!((metal(&keep) - expected).abs() < 1e-9);

        let merge = composite_intervals("H", &assays, &rule(Residual::Merge)).unwrap();
        assert_eq!(merge.len(), 3);
        assert!((merge[1].length - 4.3).abs() < 1e-9);
        assert!((metal(&merge) - expected).abs() < 1e-9);

        let drop = composite_intervals("H", &assays, &rule(Residual::Drop)).unwrap();
        assert_eq!(drop.len(), 3);
        let tail = 0.5 * 1.3;
        assert!((metal(&drop) + tail - expected).abs() < 1e-9);
    }

    #[test]
    fn categories_take_the_length_weighted_majority() {
        let with = |from, to, code: f64| {
            let mut row = assay(from, to, "g", Some(1.0));
            row.3.insert("lith".into(), code);
            row
        };
        let assays = vec![
            with(0.0, 0.8, 1.0),
            with(0.8, 1.5, 2.0),
            with(1.5, 2.0, 1.0),
        ];
        let p = CompositeParams {
            categorical_columns: vec!["lith".into()],
            ..params(2.0)
        };
        let c = composite_intervals("H", &assays, &p).unwrap();
        assert_eq!(c[0].attributes["lith"], 1.0);
        assert!((metal(&c) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_domain_composite() {
        let assays = vec![assay(0.0, 2.0, "granite", Some(2.5))];
        let composites = composite_intervals("BH001", &assays, &params(2.0)).expect("composites");

        assert_eq!(composites.len(), 1);
        assert_eq!(composites[0].domain, "granite");
        assert!((composites[0].attributes["grade"] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_domain_boundary_respect() {
        let assays = vec![
            assay(0.0, 1.5, "granite", Some(2.0)),
            assay(1.5, 3.0, "basalt", Some(3.0)),
        ];
        // Target far longer than the hole: the contact must still split it.
        let composites = composite_intervals("BH001", &assays, &params(10.0)).expect("composites");

        assert_eq!(composites.len(), 2);
        assert_eq!(composites[0].domain, "granite");
        assert_eq!(composites[1].domain, "basalt");
        assert!((composites[0].attributes["grade"] - 2.0).abs() < 1e-10);
        assert!((composites[1].attributes["grade"] - 3.0).abs() < 1e-10);
    }

    /// Composites must come out at the target length, which means cutting assay
    /// intervals at the boundary. Accumulating whole intervals instead gives
    /// 3 m composites for a 2 m target — the support the run is meant to
    /// equalize ends up varying with the sampling pattern.
    #[test]
    fn test_intervals_are_split_at_composite_boundaries() {
        let assays = vec![
            assay(0.0, 1.5, "g", Some(1.0)),
            assay(1.5, 3.0, "g", Some(1.0)),
            assay(3.0, 4.5, "g", Some(1.0)),
            assay(4.5, 6.0, "g", Some(1.0)),
        ];
        let composites = composite_intervals("BH001", &assays, &params(2.0)).expect("composites");

        assert_eq!(composites.len(), 3);
        for c in &composites {
            assert!(
                (c.length - 2.0).abs() < 1e-9,
                "expected 2 m composites, got {}",
                c.length
            );
        }
        assert!((composites[0].from_depth - 0.0).abs() < 1e-9);
        assert!((composites[2].to_depth - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_one_long_interval_splits_into_several_composites() {
        let assays = vec![assay(0.0, 5.0, "g", Some(4.0))];
        let composites = composite_intervals("BH001", &assays, &params(2.0)).expect("composites");

        let lengths: Vec<f64> = composites.iter().map(|c| c.length).collect();
        assert_eq!(lengths.len(), 3);
        assert!((lengths[0] - 2.0).abs() < 1e-9);
        assert!((lengths[1] - 2.0).abs() < 1e-9);
        assert!(
            (lengths[2] - 1.0).abs() < 1e-9,
            "tail keeps its true length"
        );
        // Splitting a uniform interval can't change its grade.
        for c in &composites {
            assert!((c.attributes["grade"] - 4.0).abs() < 1e-10);
        }
    }

    /// Unsampled core is absent, not zero. Weighting it as zero grade halves
    /// this composite and flows straight into estimation.
    #[test]
    fn test_unsampled_length_does_not_dilute_grade() {
        let assays = vec![assay(0.0, 1.0, "g", Some(4.0)), assay(1.0, 2.0, "g", None)];
        let composites = composite_intervals("BH001", &assays, &params(2.0)).expect("composites");

        assert_eq!(composites.len(), 1);
        assert!(
            (composites[0].attributes["grade"] - 4.0).abs() < 1e-10,
            "expected the mean of measured core (4.0), got {}",
            composites[0].attributes["grade"]
        );
        // The composite still spans 2 m of ground.
        assert!((composites[0].length - 2.0).abs() < 1e-9);
    }

    /// A composite with no measured value anywhere reports no grade at all
    /// rather than a fabricated zero.
    #[test]
    fn test_composite_with_no_values_omits_the_column() {
        let assays = vec![assay(0.0, 2.0, "g", None)];
        let composites = composite_intervals("BH001", &assays, &params(2.0)).expect("composites");

        assert_eq!(composites.len(), 1);
        assert!(!composites[0].attributes.contains_key("grade"));
    }

    #[test]
    fn test_length_weighting_favours_the_longer_interval() {
        let assays = vec![
            assay(0.0, 3.0, "g", Some(1.0)),
            assay(3.0, 4.0, "g", Some(5.0)),
        ];
        let composites = composite_intervals("BH001", &assays, &params(4.0)).expect("composites");

        assert_eq!(composites.len(), 1);
        // (1.0×3 + 5.0×1) / 4 = 2.0, not the naive mean of 3.0.
        assert!((composites[0].attributes["grade"] - 2.0).abs() < 1e-10);
    }

    /// Gaps are skipped, not bridged: `length` counts sampled ground only.
    #[test]
    fn test_unsampled_gap_is_not_counted_as_length() {
        let assays = vec![
            assay(0.0, 1.0, "g", Some(2.0)),
            assay(5.0, 6.0, "g", Some(2.0)),
        ];
        let composites = composite_intervals("BH001", &assays, &params(10.0)).expect("composites");

        assert_eq!(composites.len(), 1);
        assert!((composites[0].length - 2.0).abs() < 1e-9, "2 m sampled");
        assert!((composites[0].from_depth - 0.0).abs() < 1e-9);
        assert!((composites[0].to_depth - 6.0).abs() < 1e-9, "spans the gap");
    }

    #[test]
    fn test_out_of_order_intervals_are_sorted() {
        let assays = vec![
            assay(2.0, 4.0, "g", Some(1.0)),
            assay(0.0, 2.0, "g", Some(3.0)),
        ];
        let composites = composite_intervals("BH001", &assays, &params(2.0)).expect("composites");

        assert_eq!(composites.len(), 2);
        assert!((composites[0].attributes["grade"] - 3.0).abs() < 1e-10);
        assert!((composites[1].attributes["grade"] - 1.0).abs() < 1e-10);
    }

    /// A NaN depth used to panic through `partial_cmp().unwrap()`; bad data is
    /// the caller's to see, not a crash.
    #[test]
    fn test_non_finite_depth_is_an_error() {
        let assays = vec![assay(f64::NAN, 1.0, "g", Some(1.0))];
        assert!(composite_intervals("BH001", &assays, &params(2.0)).is_err());
    }

    #[test]
    fn test_non_positive_composite_length_is_an_error() {
        let assays = vec![assay(0.0, 1.0, "g", Some(1.0))];
        assert!(composite_intervals("BH001", &assays, &params(0.0)).is_err());
        assert!(composite_intervals("BH001", &assays, &params(-1.0)).is_err());
    }

    #[test]
    fn test_inverted_interval_is_an_error() {
        let assays = vec![assay(5.0, 1.0, "g", Some(1.0))];
        assert!(composite_intervals("BH001", &assays, &params(2.0)).is_err());
    }

    /// Mass balance: total composited length equals total sampled length, and
    /// metal (grade × length) is conserved across the split.
    #[test]
    fn test_conserves_length_and_metal() {
        let assays = vec![
            assay(0.0, 1.3, "g", Some(1.0)),
            assay(1.3, 2.9, "g", Some(4.0)),
            assay(2.9, 5.0, "g", Some(2.0)),
        ];
        let composites = composite_intervals("BH001", &assays, &params(1.5)).expect("composites");

        let total_length: f64 = composites.iter().map(|c| c.length).sum();
        assert!((total_length - 5.0).abs() < 1e-9);

        let metal: f64 = composites
            .iter()
            .map(|c| c.attributes["grade"] * c.length)
            .sum();
        let expected = 1.0 * 1.3 + 4.0 * 1.6 + 2.0 * 2.1;
        assert!(
            (metal - expected).abs() < 1e-9,
            "expected {expected} metal, got {metal}"
        );
    }
}
