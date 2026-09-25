//! Implicit modeling: fit a scalar field through located samples, sample it on
//! a lattice, and extract an isosurface.
//!
//! Three engines produce the field, and a surface picks one:
//!
//! - [`Engine::Rbf`] — a radial basis function solved from scratch
//!   ([`crate::rbf`]). No variogram needed; the kernel and the drift degree are
//!   the whole parameter surface.
//! - [`Engine::DualKriging`] — the dual form of kriging ([`crate::hermite`]),
//!   driven by a fitted variogram. Same global-solve shape as the RBF (one
//!   factorization, then a cheap evaluation per node), which is what makes it
//!   usable over a dense grid where the neighbourhood-search estimators are
//!   not. With no derivative data it is exactly `estimation::DualKriging`.
//! - [`Engine::SparseGp`] — a sparse variational Gaussian process
//!   ([`crate::svgp`]). Not a global solve: `m` inducing values stand in for
//!   the data, so the cost is `O(nm²)` and no decimation is needed. It is the
//!   only engine that returns a posterior *variance* alongside the field.
//!
//! The first two are *global* solves: cost is cubic in the constraint count, so
//! [`decimate`] caps what reaches the solver and the caller reports the cap.
//! The third exists because that cap is a ceiling on the answer's quality.
//!
//! All three take more than values: a boundary pick lands on the contact and a
//! structural reading constrains the gradient. [`build_surface_constrained`] is
//! the entry point for those; [`build_surface`] is the value-only shorthand.

use estimation::Sample;
use variogram::Variogram;

use crate::constraint::ConstraintSet;
use crate::error::{ModelError, Result};
use crate::grid::{ScalarGrid, TriMesh};
use crate::hermite::{HermiteKriging, HermiteSpec};
use crate::rbf::{Rbf, RbfSpec};
use crate::svgp::{FitReport, Svgp, SvgpSpec};

/// What the modelled field means, and so what its isosurface is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Driver {
    /// The field *is* the sample value — the isosurface is a grade shell at
    /// the cutoff.
    Value,
    /// The field is a signed indicator of `value >= threshold` — the
    /// isosurface at 0 is the domain boundary. Modelling ±1 rather than 1/0
    /// puts the boundary in the middle of the interpolated range instead of at
    /// its edge, so a smooth interpolant crosses it cleanly.
    Indicator { threshold: f64 },
}

impl Driver {
    /// The field value a sample contributes.
    pub fn encode(self, value: f64) -> f64 {
        match self {
            Driver::Value => value,
            Driver::Indicator { threshold } => {
                if value >= threshold {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    /// The isosurface level that separates inside from outside. Grade shells
    /// take the user's cutoff; an indicator field always cuts at 0.
    pub fn isovalue(self, cutoff: f64) -> f64 {
        match self {
            Driver::Value => cutoff,
            Driver::Indicator { .. } => 0.0,
        }
    }
}

/// Which solver builds the field.
#[derive(Debug, Clone)]
pub enum Engine {
    Rbf(RbfSpec),
    DualKriging {
        variogram: Variogram,
        /// Drift order and the tolerances on boundary and structural rows.
        spec: HermiteSpec,
    },
    /// Sparse variational GP. `emit_variance` asks for the posterior variance
    /// on the same lattice — an extra `O(m²)` per node, so it is opt-in rather
    /// than always paid for.
    SparseGp {
        spec: SvgpSpec,
        emit_variance: bool,
    },
}

/// The lattice the field is sampled on.
#[derive(Debug, Clone)]
pub struct GridSpec {
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub counts: [usize; 3],
}

impl GridSpec {
    pub fn node_count(&self) -> usize {
        self.counts[0] * self.counts[1] * self.counts[2]
    }

    /// A lattice covering `bounds` padded by `padding` on every side, with
    /// nodes about `resolution` apart.
    ///
    /// The padding matters: an isosurface only closes where the field is
    /// sampled, so a solid that runs to the edge of the lattice comes out with
    /// an open face. `max_nodes` then coarsens the spacing uniformly rather
    /// than cropping the extent — a smaller surface is a worse answer than a
    /// blockier one. The returned flag says whether that happened.
    pub fn covering(
        bounds: ([f64; 3], [f64; 3]),
        padding: f64,
        resolution: f64,
        max_nodes: usize,
    ) -> Result<(Self, bool)> {
        if !resolution.is_finite() || resolution <= 0.0 {
            return Err(ModelError::InvalidParameter(
                "resolution must be positive".into(),
            ));
        }
        if padding < 0.0 {
            return Err(ModelError::InvalidParameter(
                "padding must not be negative".into(),
            ));
        }
        if max_nodes < 8 {
            return Err(ModelError::InvalidParameter(
                "node budget must allow at least one cell".into(),
            ));
        }
        let (min, max) = bounds;
        let mut origin = [0.0; 3];
        let mut extent = [0.0; 3];
        for k in 0..3 {
            origin[k] = min[k] - padding;
            extent[k] = (max[k] - min[k]) + 2.0 * padding;
            if !extent[k].is_finite() || extent[k] < 0.0 {
                return Err(ModelError::InvalidParameter("degenerate bounds".into()));
            }
        }

        let mut spacing = resolution;
        let mut counts = node_counts(&extent, spacing);
        let mut coarsened = false;
        // Each pass scales the spacing by the cube root of the overshoot;
        // integer rounding means it can take a couple of rounds to land under
        // the budget.
        while counts[0] * counts[1] * counts[2] > max_nodes {
            let over = (counts[0] * counts[1] * counts[2]) as f64 / max_nodes as f64;
            spacing *= over.cbrt().max(1.01);
            counts = node_counts(&extent, spacing);
            coarsened = true;
        }
        Ok((
            Self {
                origin,
                spacing: [spacing; 3],
                counts,
            },
            coarsened,
        ))
    }
}

fn node_counts(extent: &[f64; 3], spacing: f64) -> [usize; 3] {
    let mut counts = [2usize; 3];
    for k in 0..3 {
        counts[k] = ((extent[k] / spacing).ceil() as usize + 1).max(2);
    }
    counts
}

/// A modelled surface plus what it took to get there.
#[derive(Debug, Clone)]
pub struct ImplicitSurface {
    pub mesh: TriMesh,
    pub grid: GridSpec,
    /// Range of the field over the lattice — the isovalue must fall inside it
    /// for the surface to be non-empty.
    pub field_min: f64,
    pub field_max: f64,
    /// The field itself, kept when an engine can say more about a node than
    /// where the isosurface crosses it.
    pub field: ScalarGrid,
    /// Posterior variance on the same lattice. Only [`Engine::SparseGp`]
    /// produces one, and only when asked.
    pub variance: Option<ScalarGrid>,
    /// How the fit went, for engines whose fit can go badly. `None` for the two
    /// direct solves, which either factorize or fail.
    pub fit: Option<FitReport>,
}

/// Fit the field through `samples` and extract the `isovalue` surface.
///
/// `samples` carry the already-encoded field values (see [`Driver::encode`]),
/// which is what lets a grade shell and a domain boundary share one path. For
/// boundary picks and structural readings, use [`build_surface_constrained`].
pub fn build_surface(
    samples: &[Sample],
    engine: &Engine,
    grid: &GridSpec,
    isovalue: f64,
) -> Result<ImplicitSurface> {
    if samples.is_empty() {
        return Err(ModelError::InsufficientData("no samples".into()));
    }
    build_surface_constrained(
        &ConstraintSet::from_samples(samples),
        engine,
        grid,
        isovalue,
    )
}

/// Fit the field honouring every constraint in `set` and extract the `isovalue`
/// surface.
pub fn build_surface_constrained(
    set: &ConstraintSet,
    engine: &Engine,
    grid: &GridSpec,
    isovalue: f64,
) -> Result<ImplicitSurface> {
    set.validate()?;
    let (field, variance, fit) = match engine {
        Engine::Rbf(spec) => {
            let rbf = Rbf::fit_constraints(set, spec)?;
            (evaluate(grid, |p| rbf.value(p))?, None, None)
        }
        Engine::DualKriging { variogram, spec } => {
            let dual = HermiteKriging::new(set, variogram, spec)?;
            (evaluate(grid, |p| dual.value(p))?, None, None)
        }
        Engine::SparseGp {
            spec,
            emit_variance,
        } => {
            let gp = Svgp::fit(set, spec)?;
            let mean = evaluate(grid, |p| gp.value(p))?;
            let variance = if *emit_variance {
                Some(evaluate(grid, |p| gp.variance(p))?)
            } else {
                None
            };
            (mean, variance, Some(gp.report().clone()))
        }
    };
    let (field_min, field_max) = field.value_range().unwrap_or((f64::NAN, f64::NAN));
    Ok(ImplicitSurface {
        mesh: crate::isosurface::marching_tetrahedra(&field, isovalue),
        grid: grid.clone(),
        field_min,
        field_max,
        field,
        variance,
        fit,
    })
}

fn evaluate(grid: &GridSpec, f: impl Fn(&[f64; 3]) -> f64) -> Result<ScalarGrid> {
    ScalarGrid::evaluate(grid.origin, grid.spacing, grid.counts, f)
}

/// Keep at most `max` items, at an even stride across the whole set.
///
/// Deterministic — no RNG — so the same inputs always give the same surface,
/// and spatially unbiased only to the extent the source ordering is; callers
/// report the reduction rather than hiding it.
pub fn decimate<T: Clone>(items: &[T], max: usize) -> Vec<T> {
    if max == 0 || items.len() <= max {
        return items.to_vec();
    }
    let stride = items.len() as f64 / max as f64;
    (0..max)
        .map(|i| items[((i as f64 * stride) as usize).min(items.len() - 1)].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rbf::Kernel;
    use variogram::{Model, Variogram};

    /// Indicator samples of a spherical domain: +1 inside the radius, −1
    /// outside, on a 4 m lattice over ±16 m. Samples within 1.5 m of the
    /// contact are dropped so the interpolant, not the sampling, decides where
    /// the boundary lands — which is the thing under test.
    fn shell_samples(radius: f64) -> Vec<Sample> {
        let mut out = Vec::new();
        for i in -4..=4 {
            for j in -4..=4 {
                for k in -4..=4 {
                    let p = (i as f64 * 4.0, j as f64 * 4.0, k as f64 * 4.0);
                    let d = (p.0 * p.0 + p.1 * p.1 + p.2 * p.2).sqrt();
                    if (d - radius).abs() > 1.5 {
                        out.push(Sample::new(p, if d < radius { 1.0 } else { -1.0 }));
                    }
                }
            }
        }
        out
    }

    /// A lattice covering the sampled extent with a little padding, so the
    /// modelled shell closes inside it.
    fn grid() -> GridSpec {
        GridSpec {
            origin: [-20.0; 3],
            spacing: [2.0; 3],
            counts: [21; 3],
        }
    }

    /// Radius of the ball with the same volume as the modelled shell.
    fn effective_radius(surface: &ImplicitSurface) -> f64 {
        (surface.mesh.enclosed_volume() * 3.0 / (4.0 * std::f64::consts::PI)).cbrt()
    }

    /// `shell_samples(10.0)` leaves the contact unsampled between r = 8 (the
    /// outermost +1) and r ≈ 11.3 (the innermost −1). Any boundary inside that
    /// gap honours the data; the true radius is 10 and a 4 m sample spacing
    /// makes half a spacing the accuracy worth asking for.
    fn assert_recovers_the_shell(surface: &ImplicitSurface, engine: &str) {
        let r = effective_radius(surface);
        assert!(
            (8.0..11.32).contains(&r),
            "{engine}: contact at r = {r} contradicts the samples"
        );
        assert!(
            (r - 10.0).abs() < 2.0,
            "{engine}: contact at r = {r}, want 10"
        );
    }

    #[test]
    fn rbf_engine_recovers_a_spherical_domain() {
        let surface = build_surface(
            &shell_samples(10.0),
            &Engine::Rbf(RbfSpec {
                kernel: Kernel::Biharmonic,
                ..Default::default()
            }),
            &grid(),
            0.0,
        )
        .unwrap();
        assert_recovers_the_shell(&surface, "rbf");
    }

    #[test]
    fn kriging_engine_recovers_the_same_domain() {
        let surface = build_surface(
            &shell_samples(10.0),
            &Engine::DualKriging {
                variogram: Variogram::single(Model::Spherical, 1.0, 30.0),
                spec: HermiteSpec::default(),
            },
            &grid(),
            0.0,
        )
        .unwrap();
        assert_recovers_the_shell(&surface, "kriging");
    }

    #[test]
    fn the_two_engines_agree_on_the_same_data() {
        let samples = shell_samples(10.0);
        let rbf = build_surface(&samples, &Engine::Rbf(RbfSpec::default()), &grid(), 0.0).unwrap();
        let kriged = build_surface(
            &samples,
            &Engine::DualKriging {
                variogram: Variogram::single(Model::Spherical, 1.0, 30.0),
                spec: HermiteSpec::default(),
            },
            &grid(),
            0.0,
        )
        .unwrap();
        let gap = (effective_radius(&rbf) - effective_radius(&kriged)).abs();
        assert!(gap < 1.0, "engines disagree by {gap} m of radius");
    }

    #[test]
    fn a_higher_cutoff_encloses_less_of_a_grade_shell() {
        // Field decreasing away from the origin: raising the cutoff must
        // shrink the shell.
        let mut samples = Vec::new();
        for i in -3..=3 {
            for j in -3..=3 {
                for k in -3..=3 {
                    let p = (i as f64 * 6.0, j as f64 * 6.0, k as f64 * 6.0);
                    let d = (p.0 * p.0 + p.1 * p.1 + p.2 * p.2).sqrt();
                    samples.push(Sample::new(p, 10.0 - d * 0.3));
                }
            }
        }
        let engine = Engine::Rbf(RbfSpec::default());
        let low = build_surface(&samples, &engine, &grid(), 5.0).unwrap();
        let high = build_surface(&samples, &engine, &grid(), 8.0).unwrap();
        assert!(low.mesh.enclosed_volume() > high.mesh.enclosed_volume());
    }

    #[test]
    fn an_out_of_range_isovalue_gives_an_empty_surface() {
        let surface = build_surface(
            &shell_samples(10.0),
            &Engine::Rbf(RbfSpec::default()),
            &grid(),
            50.0,
        )
        .unwrap();
        assert!(surface.mesh.is_empty());
        assert!(surface.field_max < 50.0);
    }

    #[test]
    fn build_surface_rejects_an_empty_sample_set() {
        assert!(build_surface(&[], &Engine::Rbf(RbfSpec::default()), &grid(), 0.0).is_err());
    }

    #[test]
    fn driver_encodes_grade_and_indicator_fields() {
        assert_eq!(Driver::Value.encode(3.5), 3.5);
        assert_eq!(Driver::Value.isovalue(1.2), 1.2);
        let ind = Driver::Indicator { threshold: 1.0 };
        assert_eq!(ind.encode(1.0), 1.0);
        assert_eq!(ind.encode(0.99), -1.0);
        // The cutoff belongs to the encoding, so the surface always cuts at 0.
        assert_eq!(ind.isovalue(1.0), 0.0);
    }

    #[test]
    fn covering_pads_the_bounds_and_honours_the_node_budget() {
        let bounds = ([0.0, 0.0, 0.0], [100.0, 50.0, 20.0]);
        let (spec, coarsened) = GridSpec::covering(bounds, 10.0, 5.0, 1_000_000).unwrap();
        assert!(!coarsened);
        assert_eq!(spec.origin, [-10.0, -10.0, -10.0]);
        // 120 m of padded extent at 5 m spacing → 25 nodes.
        assert_eq!(spec.counts[0], 25);
        let last = spec.origin[0] + (spec.counts[0] - 1) as f64 * spec.spacing[0];
        assert!(last >= 110.0, "lattice stops short of the padded extent");

        let (capped, coarsened) = GridSpec::covering(bounds, 10.0, 0.5, 50_000).unwrap();
        assert!(coarsened);
        assert!(capped.node_count() <= 50_000);
        assert!(capped.spacing[0] > 0.5);
    }

    #[test]
    fn covering_rejects_nonsense_parameters() {
        let bounds = ([0.0; 3], [10.0; 3]);
        assert!(GridSpec::covering(bounds, 1.0, 0.0, 1000).is_err());
        assert!(GridSpec::covering(bounds, -1.0, 1.0, 1000).is_err());
        assert!(GridSpec::covering(bounds, 1.0, 1.0, 4).is_err());
    }

    /// End-to-end: the same samples, once with dips and once without, must come
    /// out as different solids. Proves the constraint plumbing reaches the
    /// extracted mesh and not just the fitted field.
    #[test]
    fn structural_constraints_change_the_extracted_surface() {
        use crate::constraint::PlaneEncoding;
        use crate::orientation::Plane;
        use crate::rbf::Kernel;

        let mut base = ConstraintSet::new();
        for i in -2..=2 {
            for j in -2..=2 {
                let x = i as f64 * 8.0;
                let y = j as f64 * 8.0;
                base.push_sample([x, y, -8.0], 1.0);
                base.push_sample([x, y, 8.0], -1.0);
            }
        }
        let engine = Engine::Rbf(RbfSpec {
            kernel: Kernel::Triharmonic,
            ..Default::default()
        });
        let flat = build_surface_constrained(&base, &engine, &grid(), 0.0).unwrap();

        let mut dipped = base.clone();
        for i in -2..=2 {
            dipped
                .push_plane(
                    [i as f64 * 8.0, 0.0, 0.0],
                    Plane {
                        dip: 45.0,
                        dip_direction: 90.0,
                    },
                    PlaneEncoding::Tangents,
                    1.0,
                )
                .unwrap();
        }
        let tilted = build_surface_constrained(&dipped, &engine, &grid(), 0.0).unwrap();

        assert!(!flat.mesh.is_empty() && !tilted.mesh.is_empty());
        // Not just "the surface moved" — it has to have acquired a *dip*, so
        // measure how strongly its elevation tracks easting. A contact that is
        // flat by symmetry scores ~0; one leaning east scores towards ±1.
        let dip_correlation = |s: &ImplicitSurface| {
            let v = &s.mesh.vertices;
            let n = v.len() as f64;
            let (mx, mz) = (
                v.iter().map(|p| p[0]).sum::<f64>() / n,
                v.iter().map(|p| p[2]).sum::<f64>() / n,
            );
            let cov: f64 = v.iter().map(|p| (p[0] - mx) * (p[2] - mz)).sum::<f64>() / n;
            let sx = (v.iter().map(|p| (p[0] - mx).powi(2)).sum::<f64>() / n).sqrt();
            let sz = (v.iter().map(|p| (p[2] - mz).powi(2)).sum::<f64>() / n).sqrt();
            if sx * sz > 0.0 { cov / (sx * sz) } else { 0.0 }
        };
        assert!(
            dip_correlation(&flat).abs() < 1e-9,
            "the value-only contact already dips: {}",
            dip_correlation(&flat)
        );
        // Negative: elevation falls as easting rises, which is what a reading
        // of 45° towards 090° means. A test that only checked the magnitude
        // would pass on a surface dipping the wrong way.
        assert!(
            dip_correlation(&tilted) < -0.7,
            "dips did not fold the surface the way they were read: {}",
            dip_correlation(&tilted)
        );
    }

    /// A boundary pick is a point the contact is known to pass through, so the
    /// extracted mesh has to pass near it.
    #[test]
    fn boundary_picks_pull_the_surface_through_them() {
        use crate::rbf::Kernel;

        let mut set = ConstraintSet::new();
        for i in -2..=2 {
            for j in -2..=2 {
                let (x, y) = (i as f64 * 8.0, j as f64 * 8.0);
                set.push_sample([x, y, -10.0], 1.0);
                set.push_sample([x, y, 10.0], -1.0);
            }
        }
        // Without the pick the contact sits at z = 0; this one says it is at 6.
        let pick = [0.0, 0.0, 6.0];
        set.push_boundary(pick, 0.0);

        let surface = build_surface_constrained(
            &set,
            &Engine::Rbf(RbfSpec {
                kernel: Kernel::Triharmonic,
                ..Default::default()
            }),
            &grid(),
            0.0,
        )
        .unwrap();
        let nearest = surface
            .mesh
            .vertices
            .iter()
            .map(|v| {
                ((v[0] - pick[0]).powi(2) + (v[1] - pick[1]).powi(2) + (v[2] - pick[2]).powi(2))
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        // The lattice is 2 m, so the surface can only be resolved to about that.
        assert!(nearest < 2.0, "surface misses the pick by {nearest} m");
    }

    #[test]
    fn decimate_spans_the_whole_set() {
        let items: Vec<usize> = (0..100).collect();
        assert_eq!(decimate(&items, 200).len(), 100);
        let kept = decimate(&items, 10);
        assert_eq!(kept.len(), 10);
        assert_eq!(kept[0], 0);
        assert!(*kept.last().unwrap() >= 90);
        // Deterministic.
        assert_eq!(kept, decimate(&items, 10));
    }
}
