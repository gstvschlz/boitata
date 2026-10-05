use std::collections::BTreeSet;

use crate::{DrillholeError, Result, WellborePoint};

/// Collar grid and hole direction of a drilling plan, angles in degrees.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanGrid {
    /// (along, across) collar spacing; along follows `rotation`.
    pub spacing: [f64; 2],
    /// Azimuth of the along axis, clockwise from north.
    pub rotation: f64,
    pub azimuth: f64,
    /// Positive down; 90 is vertical.
    pub dip: f64,
    /// (along, across) shift of the grid from the footprint's corner.
    pub offset: [f64; 2],
}

/// Straight holes on a rotated collar grid over `targets`, as `(id, path)`
/// with ids `P0001`, `P0002`, ... row by row across, then along.
///
/// # Algorithm
/// 1. Each target moves up the hole direction to the elevation `top`, where
///    a hole collared at `top` would pierce it.
/// 2. Grid nodes sit at the footprint's lowest (along, across) corner plus
///    `offset`, every `spacing`; each pierce point claims its nearest node,
///    so a node becomes a collar when a target lies within half a cell of
///    its hole.
/// 3. The collar takes the elevation `ground(x, y)`, or `top` where that is
///    None, and the hole runs down to the elevation `bottom`. Holes collared
///    at or below `bottom` are dropped.
pub fn planned_holes(
    targets: &[[f64; 3]],
    top: f64,
    bottom: f64,
    grid: &PlanGrid,
    mut ground: impl FnMut(f64, f64) -> Option<f64>,
) -> Result<Vec<(String, Vec<WellborePoint>)>> {
    let bad = |m: &str| Err(DrillholeError::InvalidCollar(m.to_string()));
    if grid.spacing.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        return bad("spacing must be positive");
    }
    if !(grid.dip > 0.0 && grid.dip <= 90.0) {
        return bad("dip must be in (0, 90]");
    }
    let finite = [
        grid.rotation,
        grid.azimuth,
        top,
        bottom,
        grid.offset[0],
        grid.offset[1],
    ];
    if finite.iter().any(|v| !v.is_finite()) || targets.iter().flatten().any(|v| !v.is_finite()) {
        return bad("targets, angles and offset must be finite");
    }
    let (r, a, d) = (
        grid.rotation.to_radians(),
        grid.azimuth.to_radians(),
        grid.dip.to_radians(),
    );
    let (along, across) = ([r.sin(), r.cos()], [r.cos(), -r.sin()]);
    let direction = [d.cos() * a.sin(), d.cos() * a.cos(), -d.sin()];
    let reach = d.cos() / d.sin();
    let frame: Vec<[f64; 2]> = targets
        .iter()
        .map(|p| {
            let up = (top - p[2]) * reach;
            let q = [p[0] - up * a.sin(), p[1] - up * a.cos()];
            [
                q[0] * along[0] + q[1] * along[1],
                q[0] * across[0] + q[1] * across[1],
            ]
        })
        .collect();
    let corner =
        [0, 1].map(|k| frame.iter().map(|f| f[k]).fold(f64::INFINITY, f64::min) + grid.offset[k]);
    let nodes: BTreeSet<(i64, i64)> = frame
        .iter()
        .map(|f| {
            let [i, j] = [0, 1].map(|k| ((f[k] - corner[k]) / grid.spacing[k]).round() as i64);
            (j, i)
        })
        .collect();
    let mut holes = Vec::new();
    for (j, i) in nodes {
        let (u, v) = (
            corner[0] + i as f64 * grid.spacing[0],
            corner[1] + j as f64 * grid.spacing[1],
        );
        let (x, y) = (u * along[0] + v * across[0], u * along[1] + v * across[1]);
        let z = ground(x, y).unwrap_or(top);
        let length = (z - bottom) / d.sin();
        if length <= 0.0 {
            continue;
        }
        let at = |t: f64| WellborePoint {
            measured_depth: t,
            east: x + t * direction[0],
            north: y + t * direction[1],
            elev: z + t * direction[2],
        };
        holes.push((
            format!("P{:04}", holes.len() + 1),
            vec![at(0.0), at(length)],
        ));
    }
    Ok(holes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rotation: f64, dip: f64) -> PlanGrid {
        PlanGrid {
            spacing: [20.0, 10.0],
            rotation,
            azimuth: 120.0,
            dip,
            offset: [0.0, 0.0],
        }
    }

    /// Targets down the holes collared at -10 on a rotated 5 x 3 grid.
    fn targets(g: &PlanGrid) -> Vec<[f64; 3]> {
        let (r, a, d) = (
            g.rotation.to_radians(),
            g.azimuth.to_radians(),
            g.dip.to_radians(),
        );
        let mut out = vec![];
        for z in [-40.0, -25.0, -10.0] {
            let run = (-10.0 - z) * d.cos() / d.sin();
            for i in 0..5 {
                for j in 0..3 {
                    let (u, v) = (100.0 + i as f64 * g.spacing[0], j as f64 * g.spacing[1]);
                    let (x, y) = (u * r.sin() + v * r.cos(), u * r.cos() - v * r.sin());
                    out.push([x + run * a.sin(), y + run * a.cos(), z]);
                }
            }
        }
        out
    }

    #[test]
    fn rotated_grid_puts_one_collar_on_each_node_on_the_ground() {
        let g = grid(30.0, 90.0);
        let plane = |x: f64, y: f64| 20.0 + 0.1 * x - 0.05 * y;
        let holes =
            planned_holes(&targets(&g), -10.0, -40.0, &g, |x, y| Some(plane(x, y))).unwrap();
        assert_eq!(holes.len(), 15);
        assert_eq!(
            (holes[0].0.as_str(), holes[14].0.as_str()),
            ("P0001", "P0015")
        );
        let gap = |a: &WellborePoint, b: &WellborePoint| (a.east - b.east).hypot(a.north - b.north);
        assert!((gap(&holes[0].1[0], &holes[1].1[0]) - 20.0).abs() < 1e-9);
        assert!((gap(&holes[0].1[0], &holes[5].1[0]) - 10.0).abs() < 1e-9);
        for (_, path) in &holes {
            let (collar, end) = (&path[0], &path[1]);
            assert!((collar.elev - plane(collar.east, collar.north)).abs() < 1e-9);
            assert!((end.elev + 40.0).abs() < 1e-9);
            assert!((end.measured_depth - (collar.elev + 40.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn inclined_holes_reach_the_bottom_through_the_targets() {
        let g = grid(30.0, 60.0);
        let holes = planned_holes(&targets(&g), -10.0, -40.0, &g, |_, _| None).unwrap();
        assert_eq!(holes.len(), 15);
        for (_, path) in &holes {
            assert!((path[0].elev + 10.0).abs() < 1e-9 && (path[1].elev + 40.0).abs() < 1e-9);
            let length = 30.0 / 60f64.to_radians().sin();
            assert!((path[1].measured_depth - length).abs() < 1e-9);
        }
        assert!(planned_holes(&targets(&g), 0.0, -1.0, &grid(0.0, 0.0), |_, _| None).is_err());
    }
}
