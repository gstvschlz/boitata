//! Plurigaussian simulation (PGS) of categorical facies.
//!
//! Simulates facies by thresholding one or two latent Gaussian random fields
//! through a **truncation rule**: a partition of the (`Y₁`, `Y₂`) plane into
//! rectangular regions, each mapped to a facies. One field reproduces ordered
//! (sequential) facies; two independent fields reproduce more complex contact
//! relationships.
//!
//! Conditioning is the standard PGS pipeline:
//! 1. each datum's observed facies constrains every field to an interval
//!    (the facies' box on that axis);
//! 2. a Gibbs sweep ([`crate::gibbs`]) draws consistent Gaussian values at the
//!    data honoring those intervals + the spatial correlation;
//! 3. each field is conditioned to those values by turning bands
//!    ([`crate::turning_bands::conditional_gaussian_field`]);
//! 4. the truncation rule classifies each grid node.
//!
//! Fields are assumed independent (rectangle "flag" rule), the common PGS setup.

use crate::error::{Result, SimError};
use crate::gibbs::{GibbsParams, gibbs};
use crate::turning_bands::{TurningBandsParams, conditional_gaussian_field};
use rand::SeedableRng;
use rand::rngs::StdRng;
use variogram::Variogram;

/// A rectangular region of the (`Y₁`, `Y₂`) plane mapped to a facies.
#[derive(Debug, Clone, Copy)]
pub struct Region {
    pub y1: (f64, f64),
    pub y2: (f64, f64),
    pub facies: usize,
}

/// A truncation rule: regions that partition the Gaussian plane.
#[derive(Debug, Clone)]
pub struct TruncationRule {
    pub regions: Vec<Region>,
}

impl TruncationRule {
    /// Ordered single-field rule from facies **proportions** (summing to 1): the
    /// `Y₁` axis is cut at the cumulative Gaussian quantiles; `Y₂` is unused.
    pub fn from_proportions(proportions: &[f64]) -> Self {
        let total: f64 = proportions.iter().sum();
        let mut regions = Vec::with_capacity(proportions.len());
        let mut cum = 0.0;
        let mut lo = f64::NEG_INFINITY;
        for (facies, &p) in proportions.iter().enumerate() {
            cum += p / total;
            let hi = if facies == proportions.len() - 1 {
                f64::INFINITY
            } else {
                transforms::normal::probit(cum)
            };
            regions.push(Region {
                y1: (lo, hi),
                y2: (f64::NEG_INFINITY, f64::INFINITY),
                facies,
            });
            lo = hi;
        }
        Self { regions }
    }

    /// Facies at a Gaussian pair; falls back to the first region if uncovered.
    pub fn classify(&self, y1: f64, y2: f64) -> usize {
        for r in &self.regions {
            if y1 > r.y1.0 && y1 <= r.y1.1 && y2 > r.y2.0 && y2 <= r.y2.1 {
                return r.facies;
            }
        }
        self.regions.first().map(|r| r.facies).unwrap_or(0)
    }

    /// Per-field intervals a facies imposes: the union box over all regions
    /// mapping to that facies (its bounding interval on each axis).
    fn facies_intervals(&self, facies: usize) -> ((f64, f64), (f64, f64)) {
        let mut y1 = (f64::INFINITY, f64::NEG_INFINITY);
        let mut y2 = (f64::INFINITY, f64::NEG_INFINITY);
        for r in self.regions.iter().filter(|r| r.facies == facies) {
            y1.0 = y1.0.min(r.y1.0);
            y1.1 = y1.1.max(r.y1.1);
            y2.0 = y2.0.min(r.y2.0);
            y2.1 = y2.1.max(r.y2.1);
        }
        if y1.0 > y1.1 {
            // Unknown facies → unconstrained.
            y1 = (f64::NEG_INFINITY, f64::INFINITY);
            y2 = (f64::NEG_INFINITY, f64::INFINITY);
        }
        (y1, y2)
    }
}

/// PGS parameters.
#[derive(Debug, Clone)]
pub struct PgsParams {
    pub bands: TurningBandsParams,
    pub gibbs: GibbsParams,
    pub seed: u64,
    /// Whether the truncation rule uses the second Gaussian field.
    pub two_fields: bool,
}

impl Default for PgsParams {
    fn default() -> Self {
        Self {
            bands: TurningBandsParams::default(),
            gibbs: GibbsParams::default(),
            seed: 1,
            two_fields: false,
        }
    }
}

/// Conditional plurigaussian simulation: facies index per grid node.
///
/// `data_facies` are the observed facies at `data_locs`. `vg1`/`vg2` are the
/// (unit-sill) Gaussian variograms of the two latent fields (`vg2` ignored when
/// `two_fields` is false).
pub fn plurigaussian(
    data_locs: &[(f64, f64, f64)],
    data_facies: &[usize],
    grid: &[(f64, f64, f64)],
    vg1: &Variogram,
    vg2: &Variogram,
    rule: &TruncationRule,
    params: &PgsParams,
) -> Result<Vec<usize>> {
    if data_locs.len() != data_facies.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if grid.is_empty() {
        return Ok(vec![]);
    }

    // Per-datum facies intervals for each field.
    let intervals: Vec<((f64, f64), (f64, f64))> = data_facies
        .iter()
        .map(|&f| rule.facies_intervals(f))
        .collect();
    let b1: Vec<(f64, f64)> = intervals.iter().map(|(a, _)| *a).collect();

    // Field 1: Gibbs at data → condition over grid.
    let mut rng = StdRng::seed_from_u64(params.seed);
    let g1 = GibbsParams {
        seed: params.seed,
        ..params.gibbs.clone()
    };
    let y1_data = gibbs(data_locs, &b1, vg1, &g1)?;
    let b1p = TurningBandsParams {
        seed: params.seed.wrapping_add(11),
        ..params.bands.clone()
    };
    let y1_grid = conditional_gaussian_field(data_locs, &y1_data, grid, vg1, &b1p, &mut rng)?;

    // Field 2 (optional).
    let y2_grid = if params.two_fields {
        let b2: Vec<(f64, f64)> = intervals.iter().map(|(_, b)| *b).collect();
        let g2 = GibbsParams {
            seed: params.seed.wrapping_add(101),
            ..params.gibbs.clone()
        };
        let y2_data = gibbs(data_locs, &b2, vg2, &g2)?;
        let b2p = TurningBandsParams {
            seed: params.seed.wrapping_add(211),
            ..params.bands.clone()
        };
        conditional_gaussian_field(data_locs, &y2_data, grid, vg2, &b2p, &mut rng)?
    } else {
        vec![0.0; grid.len()]
    };

    Ok(grid
        .iter()
        .enumerate()
        .map(|(i, _)| rule.classify(y1_grid[i], y2_grid[i]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    #[test]
    fn proportions_rule_partitions_and_thresholds() {
        let rule = TruncationRule::from_proportions(&[0.5, 0.5]);
        // Two equal facies split at Y₁ = 0.
        assert_eq!(rule.classify(-1.0, 0.0), 0);
        assert_eq!(rule.classify(1.0, 0.0), 1);
        let (i0, _) = rule.facies_intervals(0);
        assert!(
            i0.0.is_infinite() && (i0.1 - 0.0).abs() < 1e-9,
            "facies 0 interval {:?}",
            i0
        );
    }

    #[test]
    fn conditional_facies_honor_data() {
        // Single-field PGS: simulated facies at data locations match the data.
        let data_locs = vec![
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (0.0, 100.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let data_facies = vec![0usize, 1, 1, 0];
        // Grid includes the data locations first.
        let mut grid = data_locs.clone();
        grid.push((50.0, 50.0, 0.0));
        let vg = Variogram::single(Model::Spherical, 1.0, 250.0);
        let rule = TruncationRule::from_proportions(&[0.5, 0.5]);
        let params = PgsParams {
            bands: TurningBandsParams {
                n_bands: 200,
                step: 5.0,
                seed: 3,
            },
            gibbs: GibbsParams {
                iterations: 200,
                burn_in: 50,
                seed: 3,
            },
            seed: 3,
            two_fields: false,
        };
        let facies =
            plurigaussian(&data_locs, &data_facies, &grid, &vg, &vg, &rule, &params).unwrap();
        // Data nodes should reproduce their facies (strong conditioning).
        for i in 0..data_facies.len() {
            assert_eq!(
                facies[i], data_facies[i],
                "node {i} facies {} != data {}",
                facies[i], data_facies[i]
            );
        }
        assert_eq!(facies.len(), grid.len());
    }

    #[test]
    fn two_field_rule_classifies() {
        // A 2×1 rectangle rule over both fields.
        let rule = TruncationRule {
            regions: vec![
                Region {
                    y1: (f64::NEG_INFINITY, 0.0),
                    y2: (f64::NEG_INFINITY, f64::INFINITY),
                    facies: 0,
                },
                Region {
                    y1: (0.0, f64::INFINITY),
                    y2: (f64::NEG_INFINITY, 0.0),
                    facies: 1,
                },
                Region {
                    y1: (0.0, f64::INFINITY),
                    y2: (0.0, f64::INFINITY),
                    facies: 2,
                },
            ],
        };
        assert_eq!(rule.classify(-1.0, 5.0), 0);
        assert_eq!(rule.classify(1.0, -1.0), 1);
        assert_eq!(rule.classify(1.0, 1.0), 2);
    }
}
