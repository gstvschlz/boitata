//! Unfolding: coordinates in the frame of a layer between two surfaces.

use boitata_core::Mesh;
use rayon::prelude::*;

use crate::{Result, Surface};

/// What the third unfolded coordinate measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnfoldMode {
    /// 0 on the footwall, 1 on the hanging wall.
    Proportional,
    /// Height above the footwall.
    Footwall,
    /// Height relative to the hanging wall, negative below it.
    Hangingwall,
}

/// Surface along which the first two unfolded coordinates are arc lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reference {
    Footwall,
    Hangingwall,
}

/// Maps points between real space and the frame of a layer bounded by a
/// footwall and a hanging wall surface. The third coordinate `w` places a
/// point between the surfaces, vertically; `u` and `v` are its easting and
/// northing, or, with a reference surface, arc lengths along that surface in
/// the x and y directions, measured from its south-west corner and offset so
/// that a flat surface keeps x and y.
#[derive(Clone)]
pub struct Unfold {
    footwall: Surface,
    hangingwall: Surface,
    mode: UnfoldMode,
    along: Option<ArcLength>,
    extrapolate: bool,
}

impl Unfold {
    pub fn new(
        footwall: &Mesh,
        hangingwall: &Mesh,
        mode: UnfoldMode,
        reference: Option<Reference>,
        extrapolate: bool,
    ) -> Result<Self> {
        let footwall = Surface::new(footwall)?;
        let hangingwall = Surface::new(hangingwall)?;
        let along = reference.map(|r| {
            ArcLength::new(match r {
                Reference::Footwall => &footwall,
                Reference::Hangingwall => &hangingwall,
            })
        });
        Ok(Self {
            footwall,
            hangingwall,
            mode,
            along,
            extrapolate,
        })
    }

    /// Footwall and hanging wall elevations at `(x, y)`, when both cover it
    /// and the hanging wall is not below the footwall.
    fn walls(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let f = self.footwall.elevation(x, y)?;
        let h = self.hangingwall.elevation(x, y)?;
        let open = h > f || (h == f && self.mode != UnfoldMode::Proportional);
        open.then_some((f, h))
    }

    /// `z` within the layer, or on its bounds up to rounding; any `z` when
    /// extrapolating.
    fn inside(&self, z: f64, f: f64, h: f64) -> Option<f64> {
        if self.extrapolate {
            return Some(z);
        }
        let tolerance = 1e-9 * f.abs().max(h.abs()).max(1.0);
        (z >= f - tolerance && z <= h + tolerance).then(|| z.clamp(f, h))
    }

    /// `(u, v, w)` of each point; NaN where the surfaces do not both cover
    /// it, the layer is pinched out or inverted, or, unless extrapolating, the
    /// point is outside the layer.
    pub fn transform(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points
            .par_iter()
            .map(|&[x, y, z]| {
                let Some((f, h)) = self.walls(x, y) else {
                    return [f64::NAN; 3];
                };
                let Some(z) = self.inside(z, f, h) else {
                    return [f64::NAN; 3];
                };
                let w = match self.mode {
                    UnfoldMode::Proportional => (z - f) / (h - f),
                    UnfoldMode::Footwall => z - f,
                    UnfoldMode::Hangingwall => z - h,
                };
                let [u, v] = self.along.as_ref().map_or([x, y], |a| a.at(x, y));
                [u, v, w]
            })
            .collect()
    }

    /// Real `(x, y, z)` of unfolded `(u, v, w)` points; NaN where
    /// [`transform`](Self::transform) would give NaN.
    pub fn inverse(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points
            .par_iter()
            .map(|&[u, v, w]| {
                let [x, y] = match &self.along {
                    Some(a) => a.solve(u, v),
                    None => [u, v],
                };
                let Some((f, h)) = self.walls(x, y) else {
                    return [f64::NAN; 3];
                };
                let z = match self.mode {
                    UnfoldMode::Proportional => f + w * (h - f),
                    UnfoldMode::Footwall => f + w,
                    UnfoldMode::Hangingwall => h + w,
                };
                match self.inside(z, f, h) {
                    Some(z) => [x, y, z],
                    None => [f64::NAN; 3],
                }
            })
            .collect()
    }
}

/// Arc lengths along a surface tabulated on a plan grid: `u` accumulates
/// along each row in x, `v` along each column in y. A step with a corner off
/// the surface counts its plan length.
#[derive(Clone)]
struct ArcLength {
    lo: [f64; 2],
    step: [f64; 2],
    n: [usize; 2],
    u: Vec<f64>,
    v: Vec<f64>,
}

impl ArcLength {
    fn new(surface: &Surface) -> Self {
        let (lo, hi) = surface.bounds();
        let n = surface.resolution().map(|s| (2 * s + 1).clamp(2, 4097));
        let step = [0, 1].map(|a| (hi[a] - lo[a]) / (n[a] - 1) as f64);
        let at = |i: usize, j: usize| {
            let (x, y) = (lo[0] + i as f64 * step[0], lo[1] + j as f64 * step[1]);
            surface.elevation(x, y)
        };
        let z: Vec<Option<f64>> = (0..n[1])
            .into_par_iter()
            .flat_map_iter(|j| (0..n[0]).map(move |i| at(i, j)))
            .collect();
        let length = |a: Option<f64>, b: Option<f64>, d: f64| match (a, b) {
            (Some(a), Some(b)) => d.hypot(b - a),
            _ => d,
        };
        let u: Vec<f64> = (0..n[1])
            .into_par_iter()
            .flat_map_iter(|j| {
                let row = &z[j * n[0]..(j + 1) * n[0]];
                std::iter::once(lo[0]).chain(row.windows(2).scan(lo[0], |s, p| {
                    *s += length(p[0], p[1], step[0]);
                    Some(*s)
                }))
            })
            .collect();
        let columns: Vec<Vec<f64>> = (0..n[0])
            .into_par_iter()
            .map(|i| {
                let mut s = lo[1];
                let mut column = vec![s];
                for j in 1..n[1] {
                    s += length(z[(j - 1) * n[0] + i], z[j * n[0] + i], step[1]);
                    column.push(s);
                }
                column
            })
            .collect();
        let v = (0..n[1] * n[0])
            .map(|k| columns[k % n[0]][k / n[0]])
            .collect();
        Self {
            lo: [lo[0], lo[1]],
            step,
            n,
            u,
            v,
        }
    }

    /// Bilinear `(u, v)` at `(x, y)`; beyond the table the arc lengths grow
    /// with plan distance.
    fn at(&self, x: f64, y: f64) -> [f64; 2] {
        if x.is_nan() || y.is_nan() {
            return [f64::NAN; 2];
        }
        let index = |p: f64, a: usize| {
            let t = if self.step[a] > 0.0 {
                (p - self.lo[a]) / self.step[a]
            } else {
                0.0
            };
            let clamped = t.clamp(0.0, (self.n[a] - 1) as f64);
            let i = (clamped.floor() as usize).min(self.n[a] - 2);
            (i, clamped - i as f64, (t - clamped) * self.step[a])
        };
        let ((i, fx, ex), (j, fy, ey)) = (index(x, 0), index(y, 1));
        let bilinear = |t: &[f64]| {
            let k = j * self.n[0] + i;
            let (a, b) = (t[k], t[k + 1]);
            let (c, d) = (t[k + self.n[0]], t[k + self.n[0] + 1]);
            (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy
        };
        [bilinear(&self.u) + ex, bilinear(&self.v) + ey]
    }

    /// `(x, y)` where the arc lengths are `(u, v)`, by Newton steps on each
    /// axis; NaN without convergence.
    fn solve(&self, u: f64, v: f64) -> [f64; 2] {
        let scale = u.abs().max(v.abs()).max(1.0);
        let h = [self.step[0].max(1e-9), self.step[1].max(1e-9)];
        let [mut x, mut y] = [u, v];
        for _ in 0..100 {
            let [a, b] = self.at(x, y);
            let [ax, _] = self.at(x + h[0], y);
            let [_, by] = self.at(x, y + h[1]);
            let dx = (u - a) / ((ax - a) / h[0]).max(1.0);
            let dy = (v - b) / ((by - b) / h[1]).max(1.0);
            if !(dx.is_finite() && dy.is_finite()) {
                break;
            }
            x += dx;
            y += dy;
            if dx.abs().max(dy.abs()) <= 1e-10 * scale {
                return [x, y];
            }
        }
        [f64::NAN; 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Surface over `n x n` nodes at `spacing` with elevation `z(x, y)`.
    fn surface(n: usize, spacing: f64, z: impl Fn(f64, f64) -> f64) -> Mesh {
        let mut vertices = vec![];
        for j in 0..n {
            for i in 0..n {
                let (x, y) = (i as f64 * spacing, j as f64 * spacing);
                vertices.push([x, y, z(x, y)]);
            }
        }
        let node = |i: usize, j: usize| (j * n + i) as u32;
        let mut triangles = vec![];
        for j in 1..n {
            for i in 1..n {
                let [a, b, c, d] = [
                    node(i - 1, j - 1),
                    node(i, j - 1),
                    node(i, j),
                    node(i - 1, j),
                ];
                triangles.extend([[a, b, c], [a, c, d]]);
            }
        }
        Mesh::new(vertices, triangles).unwrap()
    }

    fn fold(x: f64, _: f64) -> f64 {
        20.0 * (x / 30.0).sin()
    }

    #[test]
    fn flat_layer_is_z_over_thickness() {
        let unfold = Unfold::new(
            &surface(11, 10.0, |_, _| 0.0),
            &surface(11, 10.0, |_, _| 10.0),
            UnfoldMode::Proportional,
            None,
            false,
        )
        .unwrap();
        let points: Vec<[f64; 3]> = (0..=40)
            .map(|k| [k as f64 * 2.5, 100.0 - k as f64 * 2.5, k as f64 / 4.0])
            .collect();
        let close = |a: [f64; 3], b: [f64; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-12);
        for (p, q) in points.iter().zip(unfold.transform(&points)) {
            assert!(close(q, [p[0], p[1], p[2] / 10.0]), "{p:?} {q:?}");
        }
        let back = unfold.inverse(&unfold.transform(&points));
        assert!(points.iter().zip(back).all(|(p, q)| close(*p, q)));
        let outside =
            unfold.transform(&[[50.0, 50.0, 10.5], [50.0, 50.0, -0.5], [101.0, 5.0, 5.0]]);
        assert!(outside.iter().flatten().all(|w| w.is_nan()));
    }

    #[test]
    fn folded_walls_are_zero_and_one() {
        let footwall = surface(61, 5.0, fold);
        let hangingwall = surface(61, 5.0, |x, y| fold(x, y) + 8.0 + x / 50.0);
        let unfold = Unfold::new(
            &footwall,
            &hangingwall,
            UnfoldMode::Proportional,
            Some(Reference::Footwall),
            false,
        )
        .unwrap();
        let on = |wall: &Mesh| unfold.transform(wall.vertices());
        assert!(on(&footwall).iter().all(|p| p[2] == 0.0));
        assert!(on(&hangingwall).iter().all(|p| p[2] == 1.0));
        let heights = Unfold::new(&footwall, &hangingwall, UnfoldMode::Footwall, None, false)
            .unwrap()
            .transform(hangingwall.vertices());
        for (p, q) in hangingwall.vertices().iter().zip(heights) {
            assert!((q[2] - 8.0 - p[0] / 50.0).abs() < 1e-9);
        }
        let outside = Unfold::new(&footwall, &hangingwall, UnfoldMode::Footwall, None, true)
            .unwrap()
            .transform(&[[100.0, 100.0, fold(100.0, 0.0) - 3.0]]);
        assert!((outside[0][2] + 3.0).abs() < 1e-9);
    }

    #[test]
    fn arc_length_along_a_cylindrical_fold() {
        let footwall = surface(61, 5.0, fold);
        let hangingwall = surface(61, 5.0, |x, y| fold(x, y) + 10.0);
        let unfold = Unfold::new(
            &footwall,
            &hangingwall,
            UnfoldMode::Footwall,
            Some(Reference::Footwall),
            false,
        )
        .unwrap();
        // Polyline length of the footwall between x = 0 and each node.
        let mut length = 0.0;
        for i in 1..=60 {
            let (a, b) = ((i - 1) as f64 * 5.0, i as f64 * 5.0);
            length += 5.0f64.hypot(fold(b, 0.0) - fold(a, 0.0));
            let p = unfold.transform(&[[b, 150.0, fold(b, 0.0) + 4.0]])[0];
            assert!((p[0] - length).abs() < 1e-3 * length, "{} {length}", p[0]);
            assert!((p[1] - 150.0).abs() < 1e-9 && (p[2] - 4.0).abs() < 1e-9);
        }
        let points: Vec<[f64; 3]> = (0..200)
            .map(|k| {
                let (x, y) = (k as f64 * 1.49, (k * 7 % 300) as f64);
                [x, y, fold(x, y) + 0.5 + (k % 9) as f64]
            })
            .collect();
        let back = unfold.inverse(&unfold.transform(&points));
        for (p, q) in points.iter().zip(back) {
            assert!((0..3).all(|a| (p[a] - q[a]).abs() < 1e-6), "{p:?} {q:?}");
        }
    }

    #[test]
    fn same_on_one_and_eight_threads() {
        let footwall = surface(41, 5.0, |x, y| fold(x, y) + y / 20.0);
        let hangingwall = surface(41, 5.0, |x, y| fold(x, y) + 12.0 - y / 40.0);
        let points: Vec<[f64; 3]> = (0..5000)
            .map(|k| {
                let (x, y) = ((k % 71) as f64 * 2.8, (k / 71) as f64 * 2.8);
                [x, y, fold(x, y) + (k % 13) as f64]
            })
            .collect();
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let unfold = Unfold::new(
                        &footwall,
                        &hangingwall,
                        UnfoldMode::Proportional,
                        Some(Reference::Hangingwall),
                        false,
                    )
                    .unwrap();
                    let t = unfold.transform(&points);
                    let back = unfold.inverse(&t);
                    [t, back]
                        .concat()
                        .into_iter()
                        .flatten()
                        .map(f64::to_bits)
                        .collect::<Vec<_>>()
                })
        };
        assert_eq!(run(1), run(8));
    }
}
