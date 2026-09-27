//! Locally varying category proportions: a vertical proportion curve, and its
//! combination with areal proportion maps into proportions in 3D.

use std::collections::BTreeMap;

use crate::error::{Result, TransformError};

fn invalid(msg: impl Into<String>) -> TransformError {
    TransformError::InvalidParameters(msg.into())
}

/// Weighted proportion of each category per slice of height (elevation or
/// stratigraphic height).
#[derive(Debug, Clone, PartialEq)]
pub struct VerticalCurve {
    /// Slice centers, increasing.
    pub heights: Vec<f64>,
    /// Weight of the data in each slice.
    pub weights: Vec<f64>,
    /// Proportions per slice, summing to 1.
    pub proportions: Vec<Vec<f64>>,
}

impl VerticalCurve {
    /// A curve from its slices; rows are rescaled to sum 1.
    pub fn new(heights: Vec<f64>, weights: Vec<f64>, proportions: Vec<Vec<f64>>) -> Result<Self> {
        if heights.is_empty()
            || weights.len() != heights.len()
            || proportions.len() != heights.len()
        {
            return Err(invalid("one weight and one row of proportions per slice"));
        }
        if heights.iter().any(|h| !h.is_finite()) || heights.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("slice heights must be finite and increasing"));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) || weights.iter().sum::<f64>() <= 0.0
        {
            return Err(invalid("slice weights must be non-negative, not all zero"));
        }
        let k = proportions[0].len();
        let proportions = proportions
            .into_iter()
            .map(|row| {
                if row.len() != k {
                    return Err(invalid("every slice needs the same categories"));
                }
                closed(row).ok_or_else(|| {
                    invalid("proportions must be finite and non-negative, not all zero")
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            heights,
            weights,
            proportions,
        })
    }

    /// Proportions of categories `0..k` in slices of `size` starting at
    /// multiples of `size`; slices without weight are left out.
    pub fn fit(
        heights: &[f64],
        categories: &[usize],
        k: usize,
        weights: Option<&[f64]>,
        size: f64,
    ) -> Result<Self> {
        if categories.len() != heights.len() || weights.is_some_and(|w| w.len() != heights.len()) {
            return Err(invalid("one height, category and weight per sample"));
        }
        if !(size.is_finite() && size > 0.0) {
            return Err(invalid("size must be positive"));
        }
        if heights.iter().any(|h| !h.is_finite()) {
            return Err(invalid("heights must be finite"));
        }
        if let Some(c) = categories.iter().find(|&&c| c >= k) {
            return Err(invalid(format!("category {c} is not below {k}")));
        }
        let mut slices: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        for (i, (&h, &c)) in heights.iter().zip(categories).enumerate() {
            let w = weights.map_or(1.0, |w| w[i]);
            if !(w.is_finite() && w >= 0.0) {
                return Err(invalid("weights must be finite and non-negative"));
            }
            slices
                .entry((h / size).floor() as i64)
                .or_insert(vec![0.0; k])[c] += w;
        }
        let slices: Vec<_> = slices
            .into_iter()
            .filter(|(_, row)| row.iter().sum::<f64>() > 0.0)
            .collect();
        if slices.is_empty() {
            return Err(TransformError::InsufficientData(
                "no sample with a positive weight".into(),
            ));
        }
        Self::new(
            slices
                .iter()
                .map(|(i, _)| (*i as f64 + 0.5) * size)
                .collect(),
            slices.iter().map(|(_, row)| row.iter().sum()).collect(),
            slices.into_iter().map(|(_, row)| row).collect(),
        )
    }

    /// Weighted mean of the slices: the global proportions.
    pub fn global(&self) -> Vec<f64> {
        let total: f64 = self.weights.iter().sum();
        let mut g = vec![0.0; self.proportions[0].len()];
        for (row, w) in self.proportions.iter().zip(&self.weights) {
            for (g, p) in g.iter_mut().zip(row) {
                *g += w * p / total;
            }
        }
        g
    }

    /// Proportions at height `h`, linear between slice centers and constant
    /// beyond the first and last.
    pub fn at(&self, h: f64) -> Vec<f64> {
        let n = self.heights.len();
        let i = self.heights.partition_point(|&c| c <= h);
        if i == 0 || i == n {
            return self.proportions[i.min(n - 1)].clone();
        }
        let (a, b) = (self.heights[i - 1], self.heights[i]);
        let t = (h - a) / (b - a);
        self.proportions[i - 1]
            .iter()
            .zip(&self.proportions[i])
            .map(|(p, q)| p + t * (q - p))
            .collect()
    }

    /// Proportions in 3D at heights `heights` under the areal proportions
    /// `areal`: `vertical × areal / global`, rescaled to sum 1. A missing
    /// areal row keeps the curve alone, as does one sharing no category with
    /// it.
    pub fn combine(&self, heights: &[f64], areal: &[Option<Vec<f64>>]) -> Result<Vec<Vec<f64>>> {
        if heights.len() != areal.len() {
            return Err(invalid("one height and one areal row per target"));
        }
        let global = self.global();
        let k = global.len();
        heights
            .iter()
            .zip(areal)
            .map(|(&h, a)| {
                if !h.is_finite() {
                    return Err(invalid("heights must be finite"));
                }
                let v = self.at(h);
                let Some(a) = a else { return Ok(v) };
                if a.len() != k {
                    return Err(invalid(format!("areal proportions need {k} categories")));
                }
                if a.iter().any(|p| !p.is_finite() || *p < 0.0) {
                    return Err(invalid("areal proportions must be finite and non-negative"));
                }
                let row = v
                    .iter()
                    .zip(a)
                    .zip(&global)
                    .map(|((v, a), g)| if *g > 0.0 { v * a / g } else { 0.0 })
                    .collect();
                Ok(closed(row).unwrap_or(v))
            })
            .collect()
    }
}

/// `row` rescaled to sum 1; None unless finite, non-negative and not all zero.
fn closed(mut row: Vec<f64>) -> Option<Vec<f64>> {
    let s: f64 = row.iter().sum();
    if row.iter().any(|p| !p.is_finite() || *p < 0.0) || s <= 0.0 {
        return None;
    }
    row.iter_mut().for_each(|p| *p /= s);
    Some(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Clay below 2 m, sand over slimes in the north, slimes in the south.
    fn layered() -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(7);
        let (mut ys, mut hs, mut cs) = (vec![], vec![], vec![]);
        for _ in 0..2000 {
            let (y, h): (f64, f64) = (rng.r#gen::<f64>() * 100.0, rng.r#gen::<f64>() * 10.0);
            let c = if h < 2.0 {
                0
            } else if rng.r#gen::<f64>() < y / 100.0 {
                1
            } else {
                2
            };
            ys.push(y);
            hs.push(h);
            cs.push(c);
        }
        (ys, hs, cs)
    }

    #[test]
    fn combined_proportions_are_closed_and_follow_both() {
        let (ys, hs, cs) = layered();
        let curve = VerticalCurve::fit(&hs, &cs, 3, None, 1.0).unwrap();
        assert_eq!(curve.heights.len(), 10);
        assert!(curve.proportions[0][0] > 0.99 && curve.proportions[9][0] < 0.01);
        let areal: Vec<Option<Vec<f64>>> = ys
            .iter()
            .map(|y| {
                let clay = 0.2;
                Some(vec![clay, 0.8 * y / 100.0, 0.8 * (1.0 - y / 100.0)])
            })
            .collect();
        let p = curve.combine(&hs, &areal).unwrap();
        for (row, (&y, &h)) in p.iter().zip(ys.iter().zip(&hs)) {
            assert!(row.iter().all(|q| (0.0..=1.0).contains(q)));
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            if (3.0..9.0).contains(&h) {
                assert!(row[0] < 0.01);
                assert!((row[1] - y / 100.0).abs() < 0.1, "{row:?} at y {y}");
            }
        }
        assert_eq!(p, curve.combine(&hs, &areal).unwrap());
    }

    #[test]
    fn a_constant_field_gives_constant_proportions() {
        let (ys, hs, _) = layered();
        let sand = vec![1; hs.len()];
        let curve = VerticalCurve::fit(&hs, &sand, 3, None, 1.0).unwrap();
        let areal: Vec<_> = ys.iter().map(|_| Some(vec![0.0, 1.0, 0.0])).collect();
        for row in curve.combine(&hs, &areal).unwrap() {
            assert_eq!(row, vec![0.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn global_areal_proportions_leave_the_curve() {
        let (_, hs, cs) = layered();
        let w: Vec<f64> = hs.iter().map(|h| 1.0 + h).collect();
        let curve = VerticalCurve::fit(&hs, &cs, 3, Some(&w), 2.5).unwrap();
        let g = curve.global();
        let targets = [-3.0, 1.25, 2.0, 6.0, 20.0];
        let areal = vec![Some(g.clone()); targets.len()];
        for (row, &h) in curve
            .combine(&targets, &areal)
            .unwrap()
            .iter()
            .zip(&targets)
        {
            for (a, b) in row.iter().zip(curve.at(h)) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        assert_eq!(curve.at(-3.0), curve.proportions[0]);
        assert_eq!(curve.at(1.25), curve.proportions[0]);
        assert_eq!(curve.at(20.0), curve.proportions[3]);
        let mid = curve.at(2.5);
        for c in 0..3 {
            let want = (curve.proportions[0][c] + curve.proportions[1][c]) / 2.0;
            assert!((mid[c] - want).abs() < 1e-12);
        }
        let none = curve.combine(&[6.0], &[None]).unwrap();
        assert_eq!(none[0], curve.at(6.0));
    }

    #[test]
    fn rejects_bad_input() {
        assert!(VerticalCurve::fit(&[0.0], &[3], 3, None, 1.0).is_err());
        assert!(VerticalCurve::fit(&[0.0], &[0], 3, None, 0.0).is_err());
        assert!(VerticalCurve::fit(&[f64::NAN], &[0], 3, None, 1.0).is_err());
        assert!(VerticalCurve::fit(&[0.0], &[0], 3, Some(&[0.0]), 1.0).is_err());
        assert!(VerticalCurve::new(vec![1.0, 0.0], vec![1.0; 2], vec![vec![1.0]; 2]).is_err());
        let curve = VerticalCurve::fit(&[0.0], &[0], 2, None, 1.0).unwrap();
        assert!(curve.combine(&[0.0], &[Some(vec![1.0])]).is_err());
        assert!(curve.combine(&[0.0, 1.0], &[None]).is_err());
    }
}
