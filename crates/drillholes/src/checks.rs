//! Per-record checks of collar, survey and interval tables.
//!
//! Each check returns named boolean columns aligned with the input rows. A
//! defect is flagged once: rows with a missing or sentinel required value are
//! left out of the other checks, a duplicate is the later occurrence, and a
//! gap, overlap or deviation is flagged on the deeper record.

use std::collections::{HashMap, HashSet};

/// Named flag columns, in row order.
pub type Flags = Vec<(&'static str, Vec<bool>)>;

/// Whether `v` equals one of `sentinels` (to f32 precision).
pub fn is_sentinel(v: f64, sentinels: &[f64]) -> bool {
    sentinels
        .iter()
        .any(|s| (v - s).abs() <= 1e-6 * s.abs().max(1.0))
}

/// Rows where any of `columns` holds a sentinel.
pub fn sentinel_rows(columns: &[Vec<Option<f64>>], rows: usize, sentinels: &[f64]) -> Vec<bool> {
    (0..rows)
        .map(|i| {
            columns
                .iter()
                .any(|c| c[i].is_some_and(|v| is_sentinel(v, sentinels)))
        })
        .collect()
}

/// Rows whose id is not among `others`; rows without an id are not flagged.
pub fn absent(ids: &[Option<String>], others: &[Option<String>]) -> Vec<bool> {
    let others: HashSet<&str> = others.iter().flatten().map(String::as_str).collect();
    ids.iter()
        .map(|id| id.as_deref().is_some_and(|id| !others.contains(id)))
        .collect()
}

fn missing(v: Option<f64>) -> bool {
    v.is_none_or(|v| !v.is_finite())
}

fn usable(v: Option<f64>, sentinels: &[f64]) -> Option<f64> {
    v.filter(|v| v.is_finite() && !is_sentinel(*v, sentinels))
}

/// Collar ids and coordinates, plus the hole length when given.
///
/// `duplicate`: the id was seen on an earlier row. `missing`: null or
/// non-finite id, coordinate or length. `out_of_range`: negative length.
pub fn check_collars(
    ids: &[Option<String>],
    coords: [&[Option<f64>]; 3],
    depth: Option<&[Option<f64>]>,
    sentinels: &[f64],
) -> Flags {
    let n = ids.len();
    let mut seen = HashSet::new();
    let duplicate = ids
        .iter()
        .map(|id| id.as_deref().is_some_and(|id| !seen.insert(id)))
        .collect();
    let missing = (0..n)
        .map(|i| {
            ids[i].is_none()
                || coords.iter().any(|c| missing(c[i]))
                || depth.is_some_and(|d| missing(d[i]))
        })
        .collect();
    let out_of_range = (0..n)
        .map(|i| depth.is_some_and(|d| usable(d[i], sentinels).is_some_and(|d| d < 0.0)))
        .collect();
    vec![
        ("duplicate", duplicate),
        ("missing", missing),
        ("out_of_range", out_of_range),
    ]
}

fn direction(azimuth: f64, dip: f64) -> [f64; 3] {
    let (a, d) = (azimuth.to_radians(), dip.to_radians());
    [d.cos() * a.sin(), d.cos() * a.cos(), -d.sin()]
}

/// Survey stations: depth, azimuth (clockwise from north) and dip (positive
/// down), against the hole lengths in `lengths` when given.
///
/// `missing`: null or non-finite value. `out_of_range`: negative depth,
/// azimuth outside [0, 360] or dip outside [-90, 90]. `duplicate`: a hole's
/// depth seen on an earlier row. `deviation`: direction more than
/// `max_deviation` degrees from the station above; one wrong station is
/// flagged twice, on itself and on the station below. `past_depth`: deeper
/// than the hole length plus `tolerance`.
#[allow(clippy::too_many_arguments)]
pub fn check_survey(
    ids: &[Option<String>],
    depth: &[Option<f64>],
    azimuth: &[Option<f64>],
    dip: &[Option<f64>],
    lengths: Option<&HashMap<String, f64>>,
    max_deviation: f64,
    tolerance: f64,
    sentinels: &[f64],
) -> Flags {
    let n = ids.len();
    let rows: Vec<Option<(&str, f64, f64, f64)>> = (0..n)
        .map(|i| {
            Some((
                ids[i].as_deref()?,
                usable(depth[i], sentinels)?,
                usable(azimuth[i], sentinels)?,
                usable(dip[i], sentinels)?,
            ))
        })
        .collect();
    let missing = (0..n)
        .map(|i| ids[i].is_none() || missing(depth[i]) || missing(azimuth[i]) || missing(dip[i]))
        .collect();
    let out_of_range: Vec<bool> = rows
        .iter()
        .map(|r| {
            r.is_some_and(|(_, d, a, p)| {
                d < 0.0 || !(0.0..=360.0).contains(&a) || !(-90.0..=90.0).contains(&p)
            })
        })
        .collect();
    let mut seen = HashSet::new();
    let duplicate: Vec<bool> = rows
        .iter()
        .map(|r| r.is_some_and(|(id, d, _, _)| !seen.insert((id, d.to_bits()))))
        .collect();

    let mut holes: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        if let Some((id, ..)) = r
            && !out_of_range[i]
            && !duplicate[i]
        {
            holes.entry(id).or_default().push(i);
        }
    }
    let mut deviation = vec![false; n];
    for stations in holes.values_mut() {
        let at = |i: usize| rows[i].expect("usable station");
        stations.sort_by(|&i, &j| at(i).1.total_cmp(&at(j).1).then(i.cmp(&j)));
        for w in stations.windows(2) {
            let ((.., a1, p1), (.., a2, p2)) = (at(w[0]), at(w[1]));
            let (u, v) = (direction(a1, p1), direction(a2, p2));
            let cos = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            deviation[w[1]] = cos.clamp(-1.0, 1.0).acos().to_degrees() > max_deviation;
        }
    }

    let mut flags = vec![
        ("duplicate", duplicate),
        ("missing", missing),
        ("out_of_range", out_of_range),
        ("deviation", deviation),
    ];
    if let Some(lengths) = lengths {
        let past = rows
            .iter()
            .map(|r| {
                r.is_some_and(|(id, d, ..)| lengths.get(id).is_some_and(|&l| d > l + tolerance))
            })
            .collect();
        flags.push(("past_depth", past));
    }
    flags
}

/// Interval tables (assays, geology), against the hole lengths in `lengths`
/// when given.
///
/// `missing`: null or non-finite id, from or to. `inverted`: from ≥ to.
/// `out_of_range`: negative from. Each hole's remaining intervals are sorted
/// by from: `overlap` flags an interval starting more than `tolerance` above
/// the deepest end of the intervals kept so far, `gap` one starting more
/// than `tolerance` below it. Overlapping intervals are not kept, so the
/// unflagged intervals never overlap. `past_depth`: ends deeper than the hole
/// length plus `tolerance`.
pub fn check_intervals(
    ids: &[Option<String>],
    from: &[Option<f64>],
    to: &[Option<f64>],
    lengths: Option<&HashMap<String, f64>>,
    tolerance: f64,
    sentinels: &[f64],
) -> Flags {
    let n = ids.len();
    let rows: Vec<Option<(&str, f64, f64)>> = (0..n)
        .map(|i| {
            Some((
                ids[i].as_deref()?,
                usable(from[i], sentinels)?,
                usable(to[i], sentinels)?,
            ))
        })
        .collect();
    let missing = (0..n)
        .map(|i| ids[i].is_none() || missing(from[i]) || missing(to[i]))
        .collect();
    let inverted: Vec<bool> = rows
        .iter()
        .map(|r| r.is_some_and(|(_, f, t)| f >= t))
        .collect();
    let out_of_range = rows
        .iter()
        .map(|r| r.is_some_and(|(_, f, _)| f < 0.0))
        .collect();

    let mut holes: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        if let Some((id, ..)) = r
            && !inverted[i]
        {
            holes.entry(id).or_default().push(i);
        }
    }
    let (mut gap, mut overlap) = (vec![false; n], vec![false; n]);
    for intervals in holes.values_mut() {
        let at = |i: usize| rows[i].expect("usable interval");
        intervals.sort_by(|&i, &j| {
            let (a, b) = (at(i), at(j));
            a.1.total_cmp(&b.1)
                .then(a.2.total_cmp(&b.2))
                .then(i.cmp(&j))
        });
        let mut reach = at(intervals[0]).2;
        for &i in &intervals[1..] {
            let (_, f, t) = at(i);
            if f < reach - tolerance {
                overlap[i] = true;
            } else {
                gap[i] = f > reach + tolerance;
                reach = t;
            }
        }
    }

    let mut flags = vec![
        ("missing", missing),
        ("inverted", inverted),
        ("out_of_range", out_of_range),
        ("gap", gap),
        ("overlap", overlap),
    ];
    if let Some(lengths) = lengths {
        let past = rows
            .iter()
            .map(|r| {
                r.is_some_and(|(id, _, t)| lengths.get(id).is_some_and(|&l| t > l + tolerance))
            })
            .collect();
        flags.push(("past_depth", past));
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    const SENTINELS: [f64; 4] = [-99.0, -999.0, -9999.0, 1e21];

    fn ids(v: &[&str]) -> Vec<Option<String>> {
        v.iter().map(|s| Some(s.to_string())).collect()
    }

    fn some(v: &[f64]) -> Vec<Option<f64>> {
        v.iter().map(|&x| Some(x)).collect()
    }

    fn flagged(flags: &Flags) -> Vec<(&'static str, usize)> {
        flags
            .iter()
            .flat_map(|(name, f)| {
                f.iter()
                    .enumerate()
                    .filter(|x| *x.1)
                    .map(move |(i, _)| (*name, i))
            })
            .collect()
    }

    #[test]
    fn clean_tables_give_no_flags() {
        let c = check_collars(
            &ids(&["A", "B"]),
            [&some(&[0.0, 1.0]), &some(&[0.0, 1.0]), &some(&[9.0, 9.0])],
            Some(&some(&[30.0, 20.0])),
            &SENTINELS,
        );
        assert!(flagged(&c).is_empty());
        let lengths = HashMap::from([("A".to_string(), 30.0), ("B".to_string(), 20.0)]);
        let s = check_survey(
            &ids(&["A", "A", "B"]),
            &some(&[0.0, 30.0, 0.0]),
            &some(&[10.0, 15.0, 0.0]),
            &some(&[60.0, 55.0, 90.0]),
            Some(&lengths),
            20.0,
            1e-6,
            &SENTINELS,
        );
        assert!(flagged(&s).is_empty());
        let i = check_intervals(
            &ids(&["A", "A", "A", "B"]),
            &some(&[2.0, 0.0, 1.0, 5.0]),
            &some(&[3.0, 1.0, 2.0, 20.0]),
            Some(&lengths),
            1e-6,
            &SENTINELS,
        );
        assert!(flagged(&i).is_empty());
    }

    #[test]
    fn each_collar_defect_is_flagged_once() {
        let c = check_collars(
            &[
                Some("A".into()),
                Some("A".into()),
                None,
                Some("C".into()),
                Some("D".into()),
            ],
            [
                &some(&[0.0; 5]),
                &[Some(0.0), Some(0.0), Some(0.0), Some(f64::NAN), Some(0.0)],
                &some(&[0.0; 5]),
            ],
            Some(&some(&[10.0, 10.0, 10.0, 10.0, -5.0])),
            &SENTINELS,
        );
        assert_eq!(
            flagged(&c),
            [
                ("duplicate", 1),
                ("missing", 2),
                ("missing", 3),
                ("out_of_range", 4)
            ]
        );
    }

    #[test]
    fn each_survey_defect_is_flagged_once() {
        let lengths = HashMap::from([("A".to_string(), 100.0)]);
        let s = check_survey(
            &ids(&["A", "A", "A", "A", "A", "A", "A"]),
            &[
                Some(0.0),
                Some(10.0),
                Some(10.0),
                Some(20.0),
                None,
                Some(-999.0),
                Some(150.0),
            ],
            &some(&[0.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0]),
            &some(&[60.0, 60.0, 30.0, 60.0, 60.0, 60.0, 60.0]),
            Some(&lengths),
            20.0,
            1e-6,
            &SENTINELS,
        );
        assert_eq!(
            flagged(&s),
            [
                ("duplicate", 2),
                ("missing", 4),
                ("out_of_range", 3),
                ("past_depth", 6)
            ]
        );
    }

    #[test]
    fn a_wrong_station_flags_itself_and_the_one_below() {
        let s = check_survey(
            &ids(&["A", "A", "A", "A"]),
            &some(&[0.0, 10.0, 20.0, 30.0]),
            &some(&[90.0, 90.0, 90.0, 90.0]),
            &some(&[60.0, -60.0, 60.0, 65.0]),
            None,
            20.0,
            1e-6,
            &SENTINELS,
        );
        assert_eq!(flagged(&s), [("deviation", 1), ("deviation", 2)]);
    }

    #[test]
    fn each_interval_defect_is_flagged_once() {
        let lengths = HashMap::from([("A".to_string(), 10.0)]);
        let i = check_intervals(
            &ids(&["A", "A", "A", "A", "A", "A", "B"]),
            &[
                Some(0.0),
                Some(1.0),
                Some(0.5),
                Some(4.0),
                Some(6.0),
                None,
                Some(-1.0),
            ],
            &[
                Some(1.0),
                Some(3.0),
                Some(2.0),
                Some(4.0),
                Some(12.0),
                Some(3.0),
                Some(-0.5),
            ],
            Some(&lengths),
            1e-6,
            &SENTINELS,
        );
        assert_eq!(
            flagged(&i),
            [
                ("missing", 5),
                ("inverted", 3),
                ("out_of_range", 6),
                ("gap", 4),
                ("overlap", 2),
                ("past_depth", 4)
            ]
        );
    }

    #[test]
    fn overlaps_measure_against_kept_intervals() {
        let i = check_intervals(
            &ids(&["A", "A", "A"]),
            &some(&[0.0, 1.0, 2.0]),
            &some(&[2.0, 5.0, 3.0]),
            None,
            1e-6,
            &SENTINELS,
        );
        assert_eq!(flagged(&i), [("overlap", 1)]);
    }

    #[test]
    fn sentinels_and_absent_holes() {
        let cols = vec![some(&[1.0, -999.0, 2.0]), vec![None, Some(3.0), Some(1e21)]];
        assert_eq!(sentinel_rows(&cols, 3, &SENTINELS), [false, true, true]);
        assert!(is_sentinel(1e21_f32 as f64, &SENTINELS));
        assert_eq!(absent(&ids(&["A", "B"]), &ids(&["B"])), [true, false]);
    }
}
