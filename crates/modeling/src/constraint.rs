//! What the implicit field is asked to honour.
//!
//! [`crate::rbf`] and [`crate::hermite`] both fit one field, and both take the
//! same three kinds of observation:
//!
//! - a **sample** — an inside/outside coding or a grade, pinning `f(x)` to a
//!   value the driver chose;
//! - a **boundary point** — a pick that lies *on* the contact, pinning `f(x)`
//!   to the isovalue itself. Coding such a pick as inside or outside is a
//!   fudge: it is neither, it is a known zero of the field;
//! - a **derivative** — a structural reading, which says nothing about the
//!   field's value and everything about how it changes: `∇f(x)·d = v`.
//!
//! The first two are the same row shape and differ only in where the value came
//! from, so they share [`ValueConstraint`] and are told apart by [`ValueKind`]
//! for reporting and for the tolerance each is given.

use estimation::Sample;

use crate::error::{ModelError, Result};
use crate::orientation::{Lineation, Plane, dot, unit};

/// Where a value row's number came from. Only affects reporting and which
/// tolerance the row is given — the algebra is identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// An ordinary logged sample, coded by the driver.
    Sample,
    /// A pick known to lie on the contact.
    Boundary,
}

/// `f(at) = value`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueConstraint {
    pub at: [f64; 3],
    pub value: f64,
    pub kind: ValueKind,
}

/// `∇f(at) · direction = value`.
///
/// `direction` is a unit vector in world coordinates. A structural *normal*
/// carries a non-zero `value` (the field changes across the structure); an
/// in-plane *tangent* carries zero (it does not change along it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivativeConstraint {
    pub at: [f64; 3],
    pub direction: [f64; 3],
    pub value: f64,
}

/// How a planar reading is turned into derivative rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneEncoding {
    /// One row: `∇f·n = magnitude`. Fixes both the direction and the rate of
    /// change, so it is the stronger statement — and the one that needs the
    /// magnitude to be meaningful in the field's units.
    Normal,
    /// Two rows: `∇f·t = 0` for the two in-plane directions. Says only that the
    /// field is flat *along* the structure, which is the scale-free half of the
    /// reading and the safer default when the field is an indicator coding
    /// whose gradient has no natural size.
    Tangents,
    /// All three rows.
    Both,
}

/// Everything one fit is asked to honour.
#[derive(Debug, Clone, Default)]
pub struct ConstraintSet {
    pub values: Vec<ValueConstraint>,
    pub derivatives: Vec<DerivativeConstraint>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ordinary samples, already carrying their driver-encoded value.
    pub fn from_samples(samples: &[Sample]) -> Self {
        Self {
            values: samples
                .iter()
                .map(|s| ValueConstraint {
                    at: [s.loc.0, s.loc.1, s.loc.2],
                    value: s.value,
                    kind: ValueKind::Sample,
                })
                .collect(),
            derivatives: Vec::new(),
        }
    }

    pub fn push_sample(&mut self, at: [f64; 3], value: f64) {
        self.values.push(ValueConstraint {
            at,
            value,
            kind: ValueKind::Sample,
        });
    }

    /// A pick on the contact. `isovalue` is the level the surface is extracted
    /// at, so the pick is pinned to exactly that.
    pub fn push_boundary(&mut self, at: [f64; 3], isovalue: f64) {
        self.values.push(ValueConstraint {
            at,
            value: isovalue,
            kind: ValueKind::Boundary,
        });
    }

    /// A raw derivative row. `direction` is normalized here; a zero-length one
    /// is rejected rather than silently dropped.
    pub fn push_derivative(&mut self, at: [f64; 3], direction: [f64; 3], value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(ModelError::InvalidParameter(
                "derivative value must be finite".into(),
            ));
        }
        self.derivatives.push(DerivativeConstraint {
            at,
            direction: unit(&direction)?,
            value,
        });
        Ok(())
    }

    /// A planar structural reading, encoded as [`PlaneEncoding`] asks.
    ///
    /// `magnitude` is the rate the field changes across the structure, in field
    /// units per metre. It is ignored by [`PlaneEncoding::Tangents`].
    pub fn push_plane(
        &mut self,
        at: [f64; 3],
        plane: Plane,
        encoding: PlaneEncoding,
        magnitude: f64,
    ) -> Result<()> {
        if matches!(encoding, PlaneEncoding::Normal | PlaneEncoding::Both) {
            if !magnitude.is_finite() || magnitude == 0.0 {
                return Err(ModelError::InvalidParameter(
                    "a normal constraint needs a non-zero, finite gradient magnitude".into(),
                ));
            }
            self.push_derivative(at, plane.normal()?, magnitude)?;
        }
        if matches!(encoding, PlaneEncoding::Tangents | PlaneEncoding::Both) {
            let (down_dip, strike) = plane.tangents()?;
            self.push_derivative(at, down_dip, 0.0)?;
            self.push_derivative(at, strike, 0.0)?;
        }
        Ok(())
    }

    /// A lineation: the field does not change along it. Always one row, always
    /// zero — a lineation carries no rate.
    pub fn push_lineation(&mut self, at: [f64; 3], lineation: Lineation) -> Result<()> {
        self.push_derivative(at, lineation.direction()?, 0.0)
    }

    pub fn sample_count(&self) -> usize {
        self.values
            .iter()
            .filter(|v| v.kind == ValueKind::Sample)
            .count()
    }

    pub fn boundary_count(&self) -> usize {
        self.values
            .iter()
            .filter(|v| v.kind == ValueKind::Boundary)
            .count()
    }

    pub fn derivative_count(&self) -> usize {
        self.derivatives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty() && self.derivatives.is_empty()
    }

    /// Reject a system that no isovalue can be read off.
    ///
    /// A derivative row is invariant to the field's overall level *and* to its
    /// scale: if `f` satisfies every gradient constraint with all-zero
    /// right-hand sides, so does `αf + β`. A system built from orientations
    /// alone is therefore determined only up to that freedom — fine when all
    /// that is wanted is "a surface parallel to the foliation", since any level
    /// gives one, but meaningless the moment a *particular* isovalue is asked
    /// for. That is the difference between a potential field and a modelled
    /// contact, and it is worth failing loudly over rather than returning a
    /// surface at an arbitrary level.
    pub fn validate(&self) -> Result<()> {
        if self.is_empty() {
            return Err(ModelError::InsufficientData("no constraints".into()));
        }
        if self.values.is_empty() {
            return Err(ModelError::InsufficientData(format!(
                "{} structural constraints but no sample or boundary point: orientations fix \
                 the shape of the field, not its level, so no particular isovalue means \
                 anything. Add at least one logged sample or one boundary pick.",
                self.derivatives.len()
            )));
        }
        for v in &self.values {
            if !v.value.is_finite() || v.at.iter().any(|c| !c.is_finite()) {
                return Err(ModelError::InvalidParameter(
                    "a value constraint has a non-finite coordinate or value".into(),
                ));
            }
        }
        for d in &self.derivatives {
            if d.at.iter().any(|c| !c.is_finite()) {
                return Err(ModelError::InvalidParameter(
                    "a derivative constraint has a non-finite coordinate".into(),
                ));
            }
        }
        Ok(())
    }

    /// Bounding box of every constrained location, structural readings included
    /// — the lattice has to cover the orientations too, not just the samples.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut iter = self
            .values
            .iter()
            .map(|v| v.at)
            .chain(self.derivatives.iter().map(|d| d.at));
        let first = iter.next()?;
        Some(iter.fold((first, first), |(lo, hi), p| {
            (
                [lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])],
                [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])],
            )
        }))
    }

    /// Structural readings closer together than `tolerance`, which is where a
    /// derivative system goes singular: two orientations at the same place are
    /// the same row twice, and the biharmonic kernel's derivative covariance is
    /// genuinely infinite at zero separation.
    pub fn coincident_derivatives(&self, tolerance: f64) -> usize {
        let mut count = 0;
        for (i, a) in self.derivatives.iter().enumerate() {
            for b in &self.derivatives[i + 1..] {
                let d = [a.at[0] - b.at[0], a.at[1] - b.at[1], a.at[2] - b.at[2]];
                // Two rows at one location are only a duplicate when they also
                // point the same way; a plane encoded as a normal plus its
                // tangents is three rows at one point by construction.
                if crate::orientation::norm(&d) < tolerance
                    && dot(&a.direction, &b.direction).abs() > 1.0 - 1e-9
                {
                    count += 1;
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_separate_samples_from_boundary_picks() {
        let mut set = ConstraintSet::new();
        set.push_sample([0.0, 0.0, 0.0], 1.0);
        set.push_sample([10.0, 0.0, 0.0], -1.0);
        set.push_boundary([5.0, 0.0, 0.0], 0.0);
        assert_eq!(set.sample_count(), 2);
        assert_eq!(set.boundary_count(), 1);
        assert_eq!(set.derivative_count(), 0);
        assert_eq!(set.values.len(), 3);
    }

    #[test]
    fn a_boundary_pick_is_pinned_to_the_isovalue() {
        let mut set = ConstraintSet::new();
        set.push_boundary([1.0, 2.0, 3.0], 0.75);
        assert_eq!(set.values[0].value, 0.75);
        assert_eq!(set.values[0].kind, ValueKind::Boundary);
    }

    #[test]
    fn a_plane_encodes_to_the_rows_its_encoding_asks_for() {
        let plane = Plane {
            dip: 30.0,
            dip_direction: 90.0,
        };
        let at = [0.0; 3];

        let mut normal_only = ConstraintSet::new();
        normal_only
            .push_plane(at, plane, PlaneEncoding::Normal, 1.0)
            .unwrap();
        assert_eq!(normal_only.derivative_count(), 1);
        assert_eq!(normal_only.derivatives[0].value, 1.0);

        let mut tangents = ConstraintSet::new();
        tangents
            .push_plane(at, plane, PlaneEncoding::Tangents, 1.0)
            .unwrap();
        assert_eq!(tangents.derivative_count(), 2);
        assert!(tangents.derivatives.iter().all(|d| d.value == 0.0));

        let mut both = ConstraintSet::new();
        both.push_plane(at, plane, PlaneEncoding::Both, 1.0)
            .unwrap();
        assert_eq!(both.derivative_count(), 3);
    }

    #[test]
    fn a_lineation_is_always_a_zero_row() {
        let mut set = ConstraintSet::new();
        set.push_lineation(
            [4.0, 5.0, 6.0],
            Lineation {
                plunge: 20.0,
                trend: 300.0,
            },
        )
        .unwrap();
        assert_eq!(set.derivative_count(), 1);
        assert_eq!(set.derivatives[0].value, 0.0);
    }

    #[test]
    fn directions_are_normalized_on_the_way_in() {
        let mut set = ConstraintSet::new();
        set.push_derivative([0.0; 3], [0.0, 0.0, 7.0], 2.0).unwrap();
        assert_eq!(set.derivatives[0].direction, [0.0, 0.0, 1.0]);
        assert!(set.push_derivative([0.0; 3], [0.0; 3], 1.0).is_err());
    }

    /// The acceptance criterion this file exists for: orientations alone fix
    /// the field's shape but not its level, so asking for a specific isovalue
    /// on one is meaningless and must say so.
    #[test]
    fn a_structural_only_system_is_rejected_by_name() {
        let mut set = ConstraintSet::new();
        for i in 0..5 {
            set.push_plane(
                [i as f64 * 10.0, 0.0, 0.0],
                Plane {
                    dip: 25.0,
                    dip_direction: 45.0,
                },
                PlaneEncoding::Tangents,
                1.0,
            )
            .unwrap();
        }
        let err = set.validate().unwrap_err().to_string();
        assert!(err.contains("boundary pick"), "unhelpful message: {err}");
        assert!(err.contains("level"), "unhelpful message: {err}");

        // One boundary pick is enough to make the level meaningful.
        set.push_boundary([0.0, 0.0, 0.0], 0.0);
        assert!(set.validate().is_ok());
    }

    #[test]
    fn an_empty_system_is_rejected() {
        assert!(ConstraintSet::new().validate().is_err());
    }

    #[test]
    fn bounds_cover_structural_readings_as_well_as_samples() {
        let mut set = ConstraintSet::new();
        set.push_sample([0.0, 0.0, 0.0], 1.0);
        set.push_derivative([50.0, -20.0, 5.0], [0.0, 0.0, 1.0], 1.0)
            .unwrap();
        let (lo, hi) = set.bounds().unwrap();
        assert_eq!(lo, [0.0, -20.0, 0.0]);
        assert_eq!(hi, [50.0, 0.0, 5.0]);
    }

    /// A plane encoded as normal + tangents is three rows at one location, and
    /// that is not a duplicate — only two readings pointing the same way are.
    #[test]
    fn coincidence_counts_repeated_directions_not_repeated_locations() {
        let plane = Plane {
            dip: 40.0,
            dip_direction: 10.0,
        };
        let mut set = ConstraintSet::new();
        set.push_plane([0.0; 3], plane, PlaneEncoding::Both, 1.0)
            .unwrap();
        assert_eq!(set.coincident_derivatives(0.1), 0);

        // The same reading logged twice at the same place is.
        set.push_plane([0.0; 3], plane, PlaneEncoding::Normal, 1.0)
            .unwrap();
        assert_eq!(set.coincident_derivatives(0.1), 1);
    }

    #[test]
    fn from_samples_marks_every_row_as_a_sample() {
        let samples = vec![
            Sample::new((0.0, 0.0, 0.0), 1.0),
            Sample::new((1.0, 0.0, 0.0), -1.0),
        ];
        let set = ConstraintSet::from_samples(&samples);
        assert_eq!(set.sample_count(), 2);
        assert_eq!(set.boundary_count(), 0);
        assert!(set.validate().is_ok());
    }
}
