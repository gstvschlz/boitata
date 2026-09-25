//! Search neighborhoods for local estimation.
//!
//! Selects the samples used to estimate a target location, honoring an (optionally
//! anisotropic) search ellipsoid, min/max sample counts, a per-hole cap, and optional
//! octant balancing. Distances use the variogram's anisotropy when supplied.

use crate::Sample;
use crate::error::{EstimError, Result};
use serde::{Deserialize, Serialize};
use variogram::Variogram;
use variogram::aniso::euclidean;

/// Search-neighborhood parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Search {
    pub min_samples: usize,
    pub max_samples: usize,
    /// Search radius in the same (reduced) units as the variogram lag.
    /// With anisotropy this is in reduced units (≈ number of ranges); without it, meters.
    pub radius: f64,
    /// Maximum samples taken from any single drill hole (requires `Sample::hole`).
    pub max_per_hole: Option<usize>,
    /// Balance samples across 8 octants around the target.
    pub octant: bool,
}

impl Default for Search {
    fn default() -> Self {
        Self {
            min_samples: 4,
            max_samples: 16,
            radius: f64::INFINITY,
            max_per_hole: None,
            octant: false,
        }
    }
}

/// Compute the reduced lag from `target` to `sample`, honoring anisotropy if present.
fn lag(target: &(f64, f64, f64), s: &(f64, f64, f64), vg: Option<&Variogram>) -> f64 {
    match vg {
        Some(v) => v.lag(target, s),
        None => euclidean(target, s),
    }
}

/// Octant index (0..8) of `sample` relative to `target`.
fn octant_of(target: &(f64, f64, f64), s: &(f64, f64, f64)) -> usize {
    let bx = (s.0 >= target.0) as usize;
    let by = (s.1 >= target.1) as usize;
    let bz = (s.2 >= target.2) as usize;
    (bx << 2) | (by << 1) | bz
}

/// Select neighbor sample indices for a target location.
pub fn neighbors(
    target: &(f64, f64, f64),
    samples: &[Sample],
    params: &Search,
    vg: Option<&Variogram>,
) -> Result<Vec<usize>> {
    // Candidate (index, distance) within radius.
    let mut cand: Vec<(usize, f64)> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| (i, lag(target, &s.loc, vg)))
        .filter(|(_, d)| *d <= params.radius)
        .collect();

    cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Apply per-hole and octant caps greedily by increasing distance.
    let mut chosen: Vec<usize> = Vec::with_capacity(params.max_samples);
    let mut per_hole: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut per_octant = [0usize; 8];
    let octant_cap = if params.octant {
        // Distribute max_samples across 8 octants (ceil).
        params.max_samples.div_ceil(8)
    } else {
        usize::MAX
    };

    for (idx, _) in cand.into_iter() {
        if chosen.len() >= params.max_samples {
            break;
        }
        if let (Some(cap), Some(h)) = (params.max_per_hole, samples[idx].hole) {
            let c = per_hole.entry(h).or_insert(0);
            if *c >= cap {
                continue;
            }
        }
        if params.octant {
            let o = octant_of(target, &samples[idx].loc);
            if per_octant[o] >= octant_cap {
                continue;
            }
            per_octant[o] += 1;
        }
        if let Some(h) = samples[idx].hole {
            *per_hole.entry(h).or_insert(0) += 1;
        }
        chosen.push(idx);
    }

    if chosen.len() < params.min_samples {
        return Err(EstimError::SearchFailed(format!(
            "found {} samples, need at least {}",
            chosen.len(),
            params.min_samples
        )));
    }

    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, y: f64, z: f64, v: f64, hole: Option<u32>) -> Sample {
        Sample {
            loc: (x, y, z),
            value: v,
            hole,
        }
    }

    #[test]
    fn selects_nearest_within_radius() {
        let samples = vec![
            s(1.0, 0.0, 0.0, 1.0, None),
            s(2.0, 0.0, 0.0, 2.0, None),
            s(100.0, 0.0, 0.0, 3.0, None),
        ];
        let p = Search {
            min_samples: 1,
            max_samples: 2,
            radius: 10.0,
            max_per_hole: None,
            octant: false,
        };
        let got = neighbors(&(0.0, 0.0, 0.0), &samples, &p, None).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], 0); // nearest first
    }

    #[test]
    fn respects_max_per_hole() {
        let samples = vec![
            s(1.0, 0.0, 0.0, 1.0, Some(1)),
            s(2.0, 0.0, 0.0, 2.0, Some(1)),
            s(3.0, 0.0, 0.0, 3.0, Some(1)),
            s(4.0, 0.0, 0.0, 4.0, Some(2)),
        ];
        let p = Search {
            min_samples: 1,
            max_samples: 10,
            radius: 100.0,
            max_per_hole: Some(2),
            octant: false,
        };
        let got = neighbors(&(0.0, 0.0, 0.0), &samples, &p, None).unwrap();
        // At most 2 from hole 1, plus the one from hole 2.
        let from_hole1 = got.iter().filter(|&&i| samples[i].hole == Some(1)).count();
        assert!(from_hole1 <= 2);
        assert!(got.contains(&3));
    }

    #[test]
    fn errors_when_too_few() {
        let samples = vec![s(1.0, 0.0, 0.0, 1.0, None)];
        let p = Search {
            min_samples: 5,
            max_samples: 10,
            radius: 100.0,
            max_per_hole: None,
            octant: false,
        };
        assert!(neighbors(&(0.0, 0.0, 0.0), &samples, &p, None).is_err());
    }
}
