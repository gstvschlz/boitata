//! Ore and waste runs along a hole.

use crate::{DrillholeError, Result};

const EPS: f64 = 1e-9;

/// Rules turning sample flags into mineable runs.
#[derive(Debug, Clone, Default)]
pub struct RunRules {
    /// Grade a diluted run must keep; without it internal dilution is always
    /// taken in.
    pub cutoff: Option<f64>,
    /// Runs spanning less are merged into their neighbors.
    pub min_length: f64,
    /// Waste runs between ore spanning at most this are taken into the ore
    /// when the combined grade stays at or above `cutoff`.
    pub max_dilution: f64,
    /// Waste taken into each ore run on each side where it meets waste.
    pub edge: f64,
}

/// One run: depths, sampled length, length-weighted grade and flag.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub from: f64,
    pub to: f64,
    pub length: f64,
    pub grade: f64,
    pub ore: bool,
}

#[derive(Clone, Copy)]
struct Span {
    from: f64,
    to: f64,
    ore: bool,
}

/// Splits a hole into alternating ore and waste runs.
///
/// # Algorithm
/// 1. Group consecutive samples of the same flag; unsampled gaps inside a run
///    are skipped, not bridged, as in compositing to runs.
/// 2. Internal dilution, down the hole: a waste run spanning at most
///    `max_dilution` between two ore runs joins them when the grade of the
///    three together is at or above `cutoff`.
/// 3. Edge dilution: every ore run extends `edge` into each waste neighbor;
///    a waste run left with no span is absorbed.
/// 4. Minimum length: while a run spans less than `min_length`, the shortest
///    (the shallowest on ties) takes its neighbors' flag and merges with them,
///    so every run of a hole with more than one spans at least `min_length`.
/// 5. Grades are integrated over the samples each run covers, so the runs
///    partition the samples and Σ grade × length equals the sample metal.
///
/// `samples` are `(from, to, grade, ore)`; they must not overlap.
pub fn ore_runs(samples: &[(f64, f64, f64, bool)], rules: &RunRules) -> Result<Vec<Run>> {
    for (name, v) in [
        ("minimum length", rules.min_length),
        ("maximum dilution", rules.max_dilution),
        ("edge dilution", rules.edge),
    ] {
        if !(v.is_finite() && v >= 0.0) {
            return Err(DrillholeError::CompositeError(format!(
                "{name} must be a non-negative number, got {v}"
            )));
        }
    }
    let mut s = samples.to_vec();
    if s.iter()
        .any(|&(f, t, g, _)| !(f.is_finite() && t.is_finite() && g.is_finite() && t >= f))
    {
        return Err(DrillholeError::CompositeError(
            "sample depths and grades must be finite, each interval downward".to_string(),
        ));
    }
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    if let Some(i) = s.windows(2).position(|w| w[1].0 < w[0].1 - EPS) {
        return Err(DrillholeError::CompositeError(format!(
            "samples overlap at {} to {}",
            s[i + 1].0,
            s[i].1
        )));
    }
    let measure = |from: f64, to: f64| {
        s.iter().fold((0.0, 0.0), |(l, m), &(f, t, g, _)| {
            let o = (to.min(t) - from.max(f)).max(0.0);
            (l + o, m + g * o)
        })
    };

    let mut spans: Vec<Span> = Vec::new();
    for &(f, t, _, ore) in &s {
        match spans.last_mut() {
            Some(last) if last.ore == ore => last.to = t,
            _ => spans.push(Span {
                from: f,
                to: t,
                ore,
            }),
        }
    }

    let mut diluted: Vec<Span> = Vec::new();
    for span in spans {
        diluted.push(span);
        let n = diluted.len();
        if n >= 3 && diluted[n - 1].ore && !diluted[n - 2].ore {
            let (a, w, b) = (diluted[n - 3], diluted[n - 2], diluted[n - 1]);
            let (l, m) = measure(a.from, b.to);
            let keeps = rules.cutoff.is_none_or(|c| l > 0.0 && m / l >= c);
            if w.to - w.from <= rules.max_dilution + EPS && keeps {
                diluted.truncate(n - 3);
                diluted.push(Span { to: b.to, ..a });
            }
        }
    }
    let mut spans = diluted;

    if rules.edge > 0.0 {
        let n = spans.len();
        let mut out: Vec<Span> = Vec::new();
        let mut from_next: Option<f64> = None;
        let mut join_next = false;
        for (i, &r) in spans.iter().enumerate() {
            if r.ore {
                if join_next && let Some(last) = out.last_mut() {
                    last.to = r.to;
                } else {
                    out.push(Span {
                        from: from_next.unwrap_or(r.from),
                        ..r
                    });
                }
                (from_next, join_next) = (None, false);
                continue;
            }
            let lo = if i > 0 { r.from + rules.edge } else { r.from };
            let hi = if i + 1 < n { r.to - rules.edge } else { r.to };
            if hi - lo > EPS {
                if i > 0
                    && let Some(last) = out.last_mut()
                {
                    last.to = lo;
                }
                out.push(Span {
                    from: lo,
                    to: hi,
                    ore: false,
                });
                from_next = Some(hi);
            } else if i > 0
                && let Some(last) = out.last_mut()
            {
                last.to = r.to;
                join_next = true;
            } else {
                from_next = Some(r.from);
            }
        }
        spans = out;
    }

    while spans.len() > 1 {
        let short = spans
            .iter()
            .enumerate()
            .filter(|(_, r)| r.to - r.from < rules.min_length - EPS)
            .min_by(|a, b| (a.1.to - a.1.from).total_cmp(&(b.1.to - b.1.from)))
            .map(|(i, _)| i);
        let Some(i) = short else { break };
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(spans.len() - 1);
        let merged = Span {
            from: spans[lo].from,
            to: spans[hi].to,
            ore: !spans[i].ore,
        };
        spans.splice(lo..=hi, [merged]);
    }

    Ok(spans
        .into_iter()
        .filter_map(|r| {
            let (length, metal) = measure(r.from, r.to);
            (length > EPS).then(|| Run {
                from: r.from,
                to: r.to,
                length,
                grade: metal / length,
                ore: r.ore,
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meters(grades: &[f64], cutoff: f64) -> Vec<(f64, f64, f64, bool)> {
        grades
            .iter()
            .enumerate()
            .map(|(i, &g)| (i as f64, i as f64 + 1.0, g, g >= cutoff))
            .collect()
    }

    fn spans(runs: &[Run]) -> Vec<(f64, f64, bool)> {
        runs.iter().map(|r| (r.from, r.to, r.ore)).collect()
    }

    fn rules(min_length: f64, max_dilution: f64, edge: f64) -> RunRules {
        RunRules {
            cutoff: Some(1.0),
            min_length,
            max_dilution,
            edge,
        }
    }

    #[test]
    fn internal_dilution_by_hand() {
        let s = meters(&[2.0, 2.0, 0.1, 3.0, 0.1, 0.1, 0.1], 1.0);
        let runs = ore_runs(&s, &rules(0.0, 1.0, 0.0)).unwrap();
        assert_eq!(spans(&runs), [(0.0, 4.0, true), (4.0, 7.0, false)]);
        assert!((runs[0].grade - 7.1 / 4.0).abs() < 1e-12);

        let s = meters(&[1.2, 0.0, 1.2], 1.0);
        let runs = ore_runs(&s, &rules(0.0, 1.0, 0.0)).unwrap();
        assert_eq!(runs.len(), 3, "2.4 / 3 = 0.8 is below the cutoff");
    }

    #[test]
    fn short_runs_merge_into_neighbors() {
        let s = meters(&[0.1, 0.1, 2.0, 0.1, 0.1, 3.0, 3.0, 3.0], 1.0);
        let runs = ore_runs(&s, &rules(2.0, 0.0, 0.0)).unwrap();
        assert_eq!(spans(&runs), [(0.0, 5.0, false), (5.0, 8.0, true)]);
        assert!((runs[0].grade - 2.4 / 5.0).abs() < 1e-12);
        assert!(runs.iter().all(|r| r.to - r.from >= 2.0));
    }

    #[test]
    fn edges_take_waste_into_ore() {
        let s = meters(&[0.0, 0.0, 4.0, 4.0, 0.0, 0.0], 1.0);
        let runs = ore_runs(&s, &rules(0.0, 0.0, 0.5)).unwrap();
        assert_eq!(
            spans(&runs),
            [(0.0, 1.5, false), (1.5, 4.5, true), (4.5, 6.0, false)]
        );
        assert!((runs[1].grade - 8.0 / 3.0).abs() < 1e-12);

        let s = meters(&[4.0, 0.0, 4.0], 1.0);
        let runs = ore_runs(&s, &rules(0.0, 0.0, 0.5)).unwrap();
        assert_eq!(spans(&runs), [(0.0, 3.0, true)]);
    }

    #[test]
    fn runs_conserve_metal() {
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut s = Vec::new();
        let mut at = 0.0;
        for _ in 0..300 {
            let len = 0.5 + next() * 1.5;
            let g = (-(next() + 1e-9).ln()) * 0.8;
            at += if next() < 0.1 { next() * 2.0 } else { 0.0 };
            s.push((at, at + len, g, g >= 1.0));
            at += len;
        }
        let metal: f64 = s.iter().map(|&(f, t, g, _)| g * (t - f)).sum();
        for r in [
            rules(0.0, 0.0, 0.0),
            rules(3.0, 2.0, 0.0),
            rules(3.0, 2.0, 0.7),
        ] {
            let runs = ore_runs(&s, &r).unwrap();
            let total: f64 = runs.iter().map(|r| r.grade * r.length).sum();
            assert!((total - metal).abs() < 1e-9 * metal);
            assert!(runs.windows(2).all(|w| w[0].to <= w[1].from + 1e-9));
            assert!(runs.iter().all(|x| x.to - x.from >= r.min_length - 1e-9));
            assert_eq!(runs, ore_runs(&s, &r).unwrap());
        }
    }

    #[test]
    fn rejects_bad_input() {
        let s = [(0.0, 2.0, 1.0, true), (1.0, 3.0, 1.0, true)];
        assert!(ore_runs(&s, &RunRules::default()).is_err());
        assert!(ore_runs(&[], &rules(-1.0, 0.0, 0.0)).is_err());
    }
}
