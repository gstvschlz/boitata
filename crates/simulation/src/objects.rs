//! Object-based training images: sinuous channels and ellipsoids drawn into
//! a grid, for a geological concept without an analog image.
//!
//! The image starts as the background code. Each [`ObjectSet`], in order,
//! adds objects of its shape and code at random places until its code covers
//! `proportion` of the cells. A later set overwrites the cells of earlier ones,
//! so their final shares can end below their targets; the last object of a
//! set can overshoot its target. Objects cut by the edges of the grid are as
//! frequent as whole ones.
//!
//! Lengths are in the units of the cell sizes, angles in degrees: azimuth
//! clockwise from north, dip positive down, both in world coordinates, so a
//! rotated grid holds the same objects as an unrotated one.

use std::f64::consts::TAU;
use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch};
use boitata_core::rng::realization_seed;
use boitata_core::{BlockModel, Geometry, block_frame, rotation_matrix};
use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::error::{Result, SimError};
use crate::training_image::NO_CODE;

/// Objects one set may add before it gives up on its proportion.
const MAX_OBJECTS: u64 = 100_000;

/// A size or angle of an object: fixed, or drawn uniformly per object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Param {
    Fixed(f64),
    /// Uniform from the first value to the second.
    Uniform(f64, f64),
}

impl Param {
    fn draw(self, rng: &mut StdRng) -> f64 {
        match self {
            Param::Fixed(v) => v,
            Param::Uniform(lo, hi) => lo + (hi - lo) * rng.r#gen::<f64>(),
        }
    }

    fn check(self, set: usize, name: &str, valid: fn(f64) -> bool, what: &str) -> Result<()> {
        let (lo, hi) = match self {
            Param::Fixed(v) => (v, v),
            Param::Uniform(lo, hi) => (lo, hi),
        };
        if !(lo.is_finite() && hi.is_finite() && valid(lo) && valid(hi)) {
            return Err(SimError::InvalidParameters(format!(
                "{name} of object set {set} must be {what}, got {self:?}"
            )));
        }
        if lo > hi {
            return Err(SimError::InvalidParameters(format!(
                "{name} of object set {set} is the range ({lo}, {hi}); give it as (min, max)"
            )));
        }
        Ok(())
    }
}

/// The shape of the objects of a set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// A channel across the whole grid, with a flat top and a lens-shaped
    /// cross-section: depth `thickness · (1 − (2d / width)²)` at distance `d`
    /// from the centreline. The centreline runs along `azimuth` and meanders
    /// as `amplitude · sin(2π s / wavelength + φ)`, with a random phase φ. On a
    /// 2D grid, a sinuous band `width` wide.
    Channel {
        width: Param,
        thickness: Param,
        azimuth: Param,
        wavelength: Param,
        amplitude: Param,
    },
    /// An ellipsoid with semi-axes `radii` (major, semi-major, minor), the
    /// major axis along `azimuth` and `dip`, turned by `rake` about it. On a
    /// 2D grid, an ellipse of the first two radii along `azimuth`.
    Ellipsoid {
        radii: [Param; 3],
        azimuth: Param,
        dip: Param,
        rake: Param,
    },
}

/// Objects of one shape and code, added until the code covers `proportion`
/// of the cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectSet {
    /// Category code, 0 to 254.
    pub code: u8,
    /// Share of the cells the code covers when the set is done, in (0, 1).
    pub proportion: f64,
    pub shape: Shape,
}

/// A training image on `geometry` built from `sets` on a `background` code:
/// a regular block model with the codes, as floats, in `column`. The same
/// `seed` gives the same image on any number of threads.
pub fn object_training_image(
    geometry: Geometry,
    sets: &[ObjectSet],
    background: u8,
    seed: u64,
    column: &str,
) -> Result<BlockModel> {
    let invalid = |e: boitata_core::Error| SimError::InvalidParameters(e.to_string());
    geometry.validate().map_err(invalid)?;
    if background == NO_CODE {
        return Err(SimError::InvalidParameters(format!(
            "the background code must be 0 to {}, got {background}",
            NO_CODE - 1
        )));
    }
    for (i, set) in sets.iter().enumerate() {
        check(i, set)?;
    }
    let grid = Grid::new(geometry);
    let n = geometry.cells();
    let mut codes = vec![background; n as usize];
    for (i, set) in sets.iter().enumerate() {
        let target = (set.proportion * n as f64).ceil() as u64;
        let mut count = codes.iter().filter(|&&c| c == set.code).count() as u64;
        let mut object = 0;
        while count < target {
            if object == MAX_OBJECTS {
                return Err(SimError::InvalidParameters(format!(
                    "object set {i} (code {}) covers {:.3} of the grid after {MAX_OBJECTS} objects, \
                     short of its proportion {}; lower the proportion or enlarge the objects",
                    set.code,
                    count as f64 / n as f64,
                    set.proportion
                )));
            }
            let key = realization_seed(realization_seed(seed, i as u64), object);
            let mut rng = StdRng::seed_from_u64(key);
            count += match set.shape {
                Shape::Channel { .. } => grid.channel(&mut codes, set, &mut rng),
                Shape::Ellipsoid { .. } => grid.ellipsoid(&mut codes, set, &mut rng),
            };
            object += 1;
        }
    }
    let values = Arc::new(Float64Array::from_iter_values(
        codes.into_iter().map(f64::from),
    ));
    let batch = RecordBatch::try_from_iter([(column, values as _)])
        .map_err(|e| SimError::InvalidParameters(e.to_string()))?;
    BlockModel::regular(geometry, batch).map_err(invalid)
}

fn check(i: usize, set: &ObjectSet) -> Result<()> {
    if set.code == NO_CODE {
        return Err(SimError::InvalidParameters(format!(
            "object set {i} has code {}; codes are 0 to {}",
            set.code,
            NO_CODE - 1
        )));
    }
    if !(set.proportion > 0.0 && set.proportion < 1.0) {
        return Err(SimError::InvalidParameters(format!(
            "the proportion of object set {i} must be above 0 and below 1, got {}",
            set.proportion
        )));
    }
    let positive = |v: f64| v > 0.0;
    let any = |_: f64| true;
    match set.shape {
        Shape::Channel {
            width,
            thickness,
            azimuth,
            wavelength,
            amplitude,
        } => {
            width.check(i, "width", positive, "above 0")?;
            thickness.check(i, "thickness", positive, "above 0")?;
            azimuth.check(i, "azimuth", any, "finite")?;
            wavelength.check(i, "wavelength", positive, "above 0")?;
            amplitude.check(i, "amplitude", |v| v >= 0.0, "0 or more")
        }
        Shape::Ellipsoid {
            radii,
            azimuth,
            dip,
            rake,
        } => {
            for r in radii {
                r.check(i, "radius", positive, "above 0")?;
            }
            azimuth.check(i, "azimuth", any, "finite")?;
            dip.check(i, "dip", any, "finite")?;
            rake.check(i, "rake", any, "finite")
        }
    }
}

/// Sets `cell` to `code` where `inside`; 1 if that changed it.
fn paint(cell: &mut u8, code: u8, inside: bool) -> u64 {
    let new = inside && *cell != code;
    if inside {
        *cell = code;
    }
    u64::from(new)
}

/// The grid in its own frame: offsets from its centre along its axes.
struct Grid {
    geometry: Geometry,
    /// Grid axes to world.
    to_world: Matrix3<f64>,
    half: Vector3<f64>,
    flat: bool,
}

impl Grid {
    fn new(geometry: Geometry) -> Self {
        Self {
            geometry,
            to_world: block_frame(geometry.rotation).transpose(),
            half: Vector3::from_fn(|a, _| 0.5 * geometry.count[a] as f64 * geometry.size[a]),
            flat: geometry.count[2] == 1,
        }
    }

    /// Offset of the centre of cell `ijk` from the centre of the grid.
    fn offset(&self, ijk: [usize; 3]) -> Vector3<f64> {
        Vector3::from_fn(|a, _| (ijk[a] as f64 + 0.5) * self.geometry.size[a] - self.half[a])
    }

    fn channel(&self, codes: &mut [u8], set: &ObjectSet, rng: &mut StdRng) -> u64 {
        let Shape::Channel {
            width,
            thickness,
            azimuth,
            wavelength,
            amplitude,
        } = set.shape
        else {
            unreachable!("called for channels")
        };
        let (width, thickness) = (width.draw(rng), thickness.draw(rng));
        let m = rotation_matrix(azimuth.draw(rng), 0.0, 0.0) * self.to_world;
        let (wavelength, amplitude) = (wavelength.draw(rng), amplitude.draw(rng));
        let phase = TAU * rng.r#gen::<f64>();
        // The centreline passes at a random offset across the azimuth from the
        // grid centre, far enough out that channels cut by the edges are as
        // likely as others.
        let reach = self.half.norm() + 0.5 * width + amplitude;
        let across = reach * (2.0 * rng.r#gen::<f64>() - 1.0);
        // The flat top: from the lowest cell centre to `thickness` above the
        // highest one.
        let height: f64 = (0..3)
            .map(|a| m[(2, a)].abs() * (self.half[a] - 0.5 * self.geometry.size[a]))
            .sum();
        let top = -height + (2.0 * height + thickness) * rng.r#gen::<f64>();
        let code = set.code;
        codes
            .par_iter_mut()
            .enumerate()
            .map(|(cell, c)| {
                let p = m * self.offset(self.geometry.ijk(cell as u64));
                let angle = TAU * p.x / wavelength + phase;
                let slope = amplitude * TAU / wavelength * angle.cos();
                // Distance from the centreline as a share of the half-width
                // across the azimuth, which grows where the centreline turns.
                let d = (p.y - across - amplitude * angle.sin()) / (0.5 * width * slope.hypot(1.0));
                let inside = d.abs() <= 1.0
                    && (self.flat || (top - thickness * (1.0 - d * d) <= p.z && p.z <= top));
                paint(c, code, inside)
            })
            .sum()
    }

    fn ellipsoid(&self, codes: &mut [u8], set: &ObjectSet, rng: &mut StdRng) -> u64 {
        let Shape::Ellipsoid {
            radii,
            azimuth,
            dip,
            rake,
        } = set.shape
        else {
            unreachable!("called for ellipsoids")
        };
        let r = radii.map(|p| p.draw(rng));
        let (azimuth, dip, rake) = (azimuth.draw(rng), dip.draw(rng), rake.draw(rng));
        let frame = match self.flat {
            true => rotation_matrix(azimuth, 0.0, 0.0),
            false => rotation_matrix(azimuth, dip, rake),
        };
        let m = frame * self.to_world;
        let reach = match self.flat {
            true => r[0].max(r[1]),
            false => r[0].max(r[1]).max(r[2]),
        };
        let centre = Vector3::from_fn(|a, _| match self.flat && a == 2 {
            true => 0.0,
            false => (self.half[a] + reach) * (2.0 * rng.r#gen::<f64>() - 1.0),
        });
        // The cells whose centres lie within `reach` of the centre along every axis.
        let span = |a: usize| {
            let (size, n) = (self.geometry.size[a], self.geometry.count[a]);
            let at = |x: f64| (centre[a] + self.half[a] + x) / size - 0.5;
            let lo = at(-reach).ceil().max(0.0) as usize;
            let hi = at(reach).floor().min(n as f64 - 1.0);
            lo..if hi < 0.0 { 0 } else { hi as usize + 1 }
        };
        let mut new = 0;
        for k in span(2) {
            for j in span(1) {
                for i in span(0) {
                    let p = m * (self.offset([i, j, k]) - centre);
                    let w = if self.flat { 0.0 } else { p.z / r[2] };
                    let inside = (p.x / r[0]).powi(2) + (p.y / r[1]).powi(2) + w * w <= 1.0;
                    let cell = self.geometry.index([i, j, k]) as usize;
                    new += paint(&mut codes[cell], set.code, inside);
                }
            }
        }
        new
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float64Type;

    use super::*;

    fn geometry(count: [usize; 3], rotation: [f64; 3]) -> Geometry {
        Geometry {
            origin: [1000.0, 2000.0, 0.0],
            size: [2.0, 2.0, 1.0],
            count,
            rotation,
        }
    }

    fn codes(model: &BlockModel) -> Vec<u8> {
        let column = model.attributes().column(0).as_primitive::<Float64Type>();
        column.values().iter().map(|&v| v as u8).collect()
    }

    fn channels(azimuth: Param, amplitude: f64) -> Shape {
        Shape::Channel {
            width: Param::Fixed(8.0),
            thickness: Param::Fixed(3.0),
            azimuth,
            wavelength: Param::Fixed(80.0),
            amplitude: Param::Fixed(amplitude),
        }
    }

    fn lobes(azimuth: f64) -> Shape {
        Shape::Ellipsoid {
            radii: [Param::Fixed(12.0), Param::Fixed(4.0), Param::Fixed(2.0)],
            azimuth: Param::Fixed(azimuth),
            dip: Param::Fixed(0.0),
            rake: Param::Fixed(0.0),
        }
    }

    /// Azimuth, modulo 180, of the major axis of the cells holding `code`.
    fn principal_azimuth(model: &BlockModel, code: u8) -> f64 {
        let points: Vec<[f64; 3]> = model
            .centroids()
            .into_iter()
            .zip(codes(model))
            .filter(|&(_, c)| c == code)
            .map(|(p, _)| p)
            .collect();
        let n = points.len() as f64;
        let mean = |a: usize| points.iter().map(|p| p[a]).sum::<f64>() / n;
        let (mx, my) = (mean(0), mean(1));
        let moment = |f: &dyn Fn(&[f64; 3]) -> f64| points.iter().map(f).sum::<f64>() / n;
        let sxx = moment(&|p| (p[0] - mx).powi(2));
        let syy = moment(&|p| (p[1] - my).powi(2));
        let sxy = moment(&|p| (p[0] - mx) * (p[1] - my));
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        (90.0 - theta.to_degrees()).rem_euclid(180.0)
    }

    #[test]
    fn same_seed_same_image_on_any_thread_count() {
        let sets = [
            ObjectSet {
                code: 1,
                proportion: 0.3,
                shape: channels(Param::Uniform(-20.0, 20.0), 6.0),
            },
            ObjectSet {
                code: 2,
                proportion: 0.1,
                shape: lobes(45.0),
            },
        ];
        let g = geometry([60, 50, 12], [15.0, 0.0, 0.0]);
        let make = |seed| object_training_image(g, &sets, 0, seed, "facies").unwrap();
        let a = codes(&make(3));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        assert_eq!(a, codes(&pool.install(|| make(3))));
        assert_ne!(a, codes(&make(4)));
    }

    #[test]
    fn a_set_reaches_its_proportion_by_at_most_one_object() {
        let g = geometry([100, 100, 1], [0.0; 3]);
        let sets = [ObjectSet {
            code: 3,
            proportion: 0.2,
            shape: lobes(0.0),
        }];
        let model = object_training_image(g, &sets, 1, 0, "facies").unwrap();
        let ti = crate::TrainingImage::categorical(&model, "facies").unwrap();
        let share = ti.proportions()[3];
        let one = std::f64::consts::PI * 12.0 * 4.0 / (4.0 * 1e4);
        assert!((0.2..0.2 + 1.2 * one).contains(&share), "{share}");
        assert!(codes(&model).iter().all(|&c| c == 1 || c == 3));
    }

    #[test]
    fn a_straight_channel_runs_along_its_azimuth_on_a_rotated_grid() {
        let g = geometry([150, 150, 1], [20.0, 0.0, 0.0]);
        let sets = [ObjectSet {
            code: 1,
            proportion: 1e-6,
            shape: channels(Param::Fixed(30.0), 0.0),
        }];
        let model = object_training_image(g, &sets, 0, 7, "facies").unwrap();
        let azimuth = principal_azimuth(&model, 1);
        assert!((azimuth - 30.0).abs() < 1.0, "{azimuth}");
    }

    #[test]
    fn an_ellipse_points_along_its_azimuth_and_fills_its_area() {
        let g = geometry([200, 200, 1], [0.0; 3]);
        for seed in 0.. {
            let sets = [ObjectSet {
                code: 1,
                proportion: 1e-6,
                shape: lobes(60.0),
            }];
            let model = object_training_image(g, &sets, 0, seed, "facies").unwrap();
            let c = codes(&model);
            let edge = |cell: usize| {
                let [i, j, _] = g.ijk(cell as u64);
                i == 0 || j == 0 || i == 199 || j == 199
            };
            if (0..c.len()).any(|cell| c[cell] == 1 && edge(cell)) {
                continue;
            }
            let azimuth = principal_azimuth(&model, 1);
            assert!((azimuth - 60.0).abs() < 2.0, "{azimuth}");
            let area = c.iter().filter(|&&v| v == 1).count() as f64 * 4.0;
            let exact = std::f64::consts::PI * 12.0 * 4.0;
            assert!((area / exact - 1.0).abs() < 0.1, "{area} vs {exact}");
            break;
        }
    }

    #[test]
    fn channels_have_a_flat_top_in_3d() {
        let g = geometry([40, 40, 20], [0.0; 3]);
        let sets = [ObjectSet {
            code: 1,
            proportion: 1e-6,
            shape: channels(Param::Fixed(90.0), 0.0),
        }];
        let c = codes(&object_training_image(g, &sets, 0, 11, "facies").unwrap());
        let top = (0..c.len())
            .filter(|&cell| c[cell] == 1)
            .map(|cell| g.ijk(cell as u64)[2])
            .max()
            .unwrap();
        // The deepest layer of the lens is at the centreline, 3 layers at most.
        for cell in (0..c.len()).filter(|&cell| c[cell] == 1) {
            let [i, j, k] = g.ijk(cell as u64);
            assert!(top - k <= 3);
            if k < top {
                assert_eq!(c[g.index([i, j, k + 1]) as usize], 1);
            }
        }
    }

    #[test]
    fn checks_name_the_set_and_the_parameter() {
        let g = geometry([10, 10, 1], [0.0; 3]);
        let bad = ObjectSet {
            code: 1,
            proportion: 0.2,
            shape: Shape::Channel {
                width: Param::Fixed(-1.0),
                thickness: Param::Fixed(1.0),
                azimuth: Param::Fixed(0.0),
                wavelength: Param::Fixed(1.0),
                amplitude: Param::Fixed(0.0),
            },
        };
        let e = object_training_image(g, &[bad], 0, 0, "f").unwrap_err();
        assert!(e.to_string().contains("width of object set 0"), "{e}");
        let reversed = Param::Uniform(5.0, 2.0).check(0, "radius", |v| v > 0.0, "above 0");
        assert!(reversed.unwrap_err().to_string().contains("(min, max)"));
        let code = ObjectSet { code: 255, ..bad };
        assert!(object_training_image(g, &[code], 0, 0, "f").is_err());
        assert!(object_training_image(g, &[], 255, 0, "f").is_err());
    }
}
