//! Column units, kept in Arrow field metadata under `unit`.
//!
//! A unit is a scale and a dimension: integer powers of length, mass, grade,
//! angle and one axis per currency. Grade (a mass fraction such as `g/t`) is
//! its own axis, so metal (`t` × `g/t`, written `"g metal"`) is never ore mass
//! and a grade never converts to a plain ratio (`ratio`, `ratio%`). Currencies
//! never convert into each other: an exchange rate is data, not a unit.
//!
//! Units combine with `*`, `/`, `^n` and parentheses; a trailing digit is a
//! power (`m3`). `oz` is the troy ounce and `oz/t` is troy ounces per short
//! ton, as written in North American reports.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;
use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};

use crate::{Error, Result};

const KEY: &str = "unit";
const CURRENCIES: [&str; 5] = ["USD", "BRL", "EUR", "CAD", "AUD"];
const TROY_OUNCE: f64 = 31.103_476_8e-6;
const POUND: f64 = 0.453_592_37e-3;
const SHORT_TON: f64 = 0.907_184_74;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Axis {
    Length,
    Mass,
    Grade,
    Angle,
    Currency(&'static str),
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Axis::Length => "length",
            Axis::Mass => "mass",
            Axis::Grade => "grade",
            Axis::Angle => "angle",
            Axis::Currency(code) => code,
        })
    }
}

/// Powers of the base axes; empty for a ratio.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dimension(Vec<(Axis, i32)>);

impl Dimension {
    fn of(axis: Axis) -> Self {
        Self(vec![(axis, 1)])
    }

    fn combine(&self, other: &Self, sign: i32) -> Self {
        let mut powers = self.0.clone();
        for (axis, e) in &other.0 {
            match powers.iter_mut().find(|(a, _)| a == axis) {
                Some((_, p)) => *p += sign * e,
                None => powers.push((axis.clone(), sign * e)),
            }
        }
        powers.retain(|(_, e)| *e != 0);
        powers.sort_by(|a, b| a.0.cmp(&b.0));
        Self(powers)
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Axis::*;
        let name = match self.0.as_slice() {
            [] => "ratio",
            [(Length, 1)] => "length",
            [(Length, 2)] => "area",
            [(Length, 3)] => "volume",
            [(Mass, 1)] => "mass",
            [(Length, -3), (Mass, 1)] => "density",
            [(Mass, 1), (Grade, 1)] => "metal",
            [(Grade, 1)] => "grade",
            [(Angle, 1)] => "angle",
            [(Currency(code), 1)] => code,
            powers => {
                let parts: Vec<String> = powers
                    .iter()
                    .map(|(a, e)| match e {
                        1 => a.to_string(),
                        _ => format!("{a}^{e}"),
                    })
                    .collect();
                return f.write_str(&parts.join("·"));
            }
        };
        f.write_str(name)
    }
}

/// A parsed unit: `factor` base units (m, t, grade fraction, rad, one unit of
/// currency) of `dimension`.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    factor: f64,
    dimension: Dimension,
}

impl Unit {
    fn new(factor: f64, dimension: Dimension) -> Self {
        Self { factor, dimension }
    }

    pub fn factor(&self) -> f64 {
        self.factor
    }

    pub fn dimension(&self) -> &Dimension {
        &self.dimension
    }

    pub fn powi(&self, n: i32) -> Self {
        let powers = self
            .dimension
            .0
            .iter()
            .map(|(a, e)| (a.clone(), e * n))
            .filter(|(_, e)| *e != 0)
            .collect();
        Self::new(self.factor.powi(n), Dimension(powers))
    }
}

/// Mass over mass is a grade (`mg/kg`, `oz/st`), not a plain ratio.
fn product(a: &Dimension, b: &Dimension, sign: i32) -> Dimension {
    let dimension = a.combine(b, sign);
    let mass = |d: &Dimension| !d.0.is_empty() && d.0.iter().all(|(axis, _)| *axis == Axis::Mass);
    if dimension.0.is_empty() && mass(a) && mass(b) {
        Dimension::of(Axis::Grade)
    } else {
        dimension
    }
}

impl std::ops::Mul for &Unit {
    type Output = Unit;
    fn mul(self, rhs: &Unit) -> Unit {
        Unit::new(
            self.factor * rhs.factor,
            product(&self.dimension, &rhs.dimension, 1),
        )
    }
}

impl std::ops::Div for &Unit {
    type Output = Unit;
    fn div(self, rhs: &Unit) -> Unit {
        Unit::new(
            self.factor / rhs.factor,
            product(&self.dimension, &rhs.dimension, -1),
        )
    }
}

fn catalog() -> &'static [(String, Unit)] {
    static CATALOG: OnceLock<Vec<(String, Unit)>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        use Axis::*;
        let mut units: Vec<(String, Unit)> = [
            ("m", 1.0, Length),
            ("mm", 1e-3, Length),
            ("cm", 1e-2, Length),
            ("km", 1e3, Length),
            ("ft", 0.3048, Length),
            ("in", 0.0254, Length),
            ("yd", 0.9144, Length),
            ("mi", 1609.344, Length),
            ("t", 1.0, Mass),
            ("kt", 1e3, Mass),
            ("Mt", 1e6, Mass),
            ("g", 1e-6, Mass),
            ("mg", 1e-9, Mass),
            ("kg", 1e-3, Mass),
            ("oz", TROY_OUNCE, Mass),
            ("ozt", TROY_OUNCE, Mass),
            ("koz", TROY_OUNCE * 1e3, Mass),
            ("Moz", TROY_OUNCE * 1e6, Mass),
            ("lb", POUND, Mass),
            ("klb", POUND * 1e3, Mass),
            ("Mlb", POUND * 1e6, Mass),
            ("st", SHORT_TON, Mass),
            ("g/t", 1e-6, Grade),
            ("ppm", 1e-6, Grade),
            ("ppb", 1e-9, Grade),
            ("%", 1e-2, Grade),
            ("kg/t", 1e-3, Grade),
            ("oz/t", TROY_OUNCE / SHORT_TON, Grade),
            ("lb/st", POUND / SHORT_TON, Grade),
            ("rad", 1.0, Angle),
            ("deg", PI / 180.0, Angle),
            ("°", PI / 180.0, Angle),
        ]
        .into_iter()
        .map(|(name, factor, axis)| (name.to_string(), Unit::new(factor, Dimension::of(axis))))
        .collect();
        units.push(("ratio".into(), Unit::new(1.0, Dimension::default())));
        units.push(("ratio%".into(), Unit::new(1e-2, Dimension::default())));
        for code in CURRENCIES {
            for (prefix, factor) in [("", 1.0), ("k", 1e3), ("M", 1e6)] {
                let unit = Unit::new(factor, Dimension::of(Currency(code)));
                units.push((format!("{prefix}{code}"), unit));
            }
        }
        units.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));
        units
    })
}

struct Parser<'a> {
    rest: &'a str,
}

impl Parser<'_> {
    fn expr(&mut self) -> Option<Unit> {
        let mut unit = self.term()?;
        loop {
            if let Some(r) = self.rest.strip_prefix(['*', '·']) {
                self.rest = r;
                unit = &unit * &self.term()?;
            } else if let Some(r) = self.rest.strip_prefix('/') {
                self.rest = r;
                unit = &unit / &self.term()?;
            } else {
                return Some(unit);
            }
        }
    }

    fn term(&mut self) -> Option<Unit> {
        let base = match self.rest.strip_prefix('(') {
            Some(r) => {
                self.rest = r;
                let unit = self.expr()?;
                self.rest = self.rest.strip_prefix(')')?;
                unit
            }
            None => self.atom()?,
        };
        Some(base.powi(self.exponent()?))
    }

    fn atom(&mut self) -> Option<Unit> {
        let (name, unit) = catalog().iter().find(|(name, _)| {
            self.rest
                .strip_prefix(name.as_str())
                .is_some_and(|r| !r.starts_with(|c: char| c.is_alphabetic() || c == '%'))
        })?;
        self.rest = &self.rest[name.len()..];
        Some(unit.clone())
    }

    fn exponent(&mut self) -> Option<i32> {
        for (sign, power) in [('²', 2), ('³', 3)] {
            if let Some(r) = self.rest.strip_prefix(sign) {
                self.rest = r;
                return Some(power);
            }
        }
        let (negative, digits) = match self.rest.strip_prefix('^') {
            Some(r) => match r.strip_prefix('-') {
                Some(r) => (true, r),
                None => (false, r),
            },
            None if self.rest.starts_with(|c: char| c.is_ascii_digit()) => (false, self.rest),
            None => return Some(1),
        };
        let n = digits
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(digits.len());
        let power: i32 = digits[..n].parse().ok()?;
        self.rest = &digits[n..];
        Some(if negative { -power } else { power })
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let next = (row[j + 1] + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

fn unknown(text: &str) -> Error {
    let closest = catalog()
        .iter()
        .map(|(name, _)| (edit_distance(text, name), name))
        .filter(|(d, name)| *d <= 2 && *d < name.chars().count())
        .min_by_key(|(d, _)| *d);
    Error::Units(match closest {
        Some((_, name)) => format!("unknown unit `{text}`; did you mean `{name}`?"),
        None => format!("unknown unit `{text}`"),
    })
}

/// Parses a unit such as `g/t`, `t/m3`, `(g/t)^2`, `120 USD/m`'s `USD/m` or
/// `Moz metal` (a mass of metal).
pub fn parse(text: &str) -> Result<Unit> {
    let trimmed = text.trim();
    if let Some(mass) = trimmed.strip_suffix("metal").map(str::trim_end) {
        let unit = parse(mass).map_err(|_| unknown(text))?;
        if unit.dimension != Dimension::of(Axis::Mass) {
            return Err(Error::Units(format!(
                "`metal` follows a mass unit, as in `kg metal`; got `{text}`"
            )));
        }
        let metal = unit.dimension.combine(&Dimension::of(Axis::Grade), 1);
        return Ok(Unit::new(unit.factor, metal));
    }
    let compact: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    let mut parser = Parser { rest: &compact };
    match parser.expr() {
        Some(unit) if parser.rest.is_empty() => Ok(unit),
        _ => Err(unknown(text)),
    }
}

fn scale(from: (&str, &Unit), to: (&str, &Unit)) -> Result<f64> {
    if from.1.dimension != to.1.dimension {
        return Err(Error::Units(format!(
            "cannot convert {} ({}) to {} ({})",
            from.1.dimension, from.0, to.1.dimension, to.0
        )));
    }
    let factor = from.1.factor / to.1.factor;
    // Drops the rounding of prefixes and powers, so g/cm3 to t/m3 is exactly 1.
    Ok(format!("{factor:.14e}").parse().expect("formatted float"))
}

/// Factor taking values in `from` to `to`; errors across dimensions.
pub fn conversion(from: &str, to: &str) -> Result<f64> {
    scale((from, &parse(from)?), (to, &parse(to)?))
}

/// The units a mass of ore reads in, smallest first.
pub const TONNAGE: [&str; 3] = ["t", "kt", "Mt"];

/// The units the metal of ore with grades in `grade` reads in, smallest
/// first: troy ounces for `oz/t`, pounds for `lb/st`, tonnes for `%`.
pub fn metal_units(grade: &str) -> [&'static str; 3] {
    match grade.trim() {
        "oz/t" => ["oz metal", "koz metal", "Moz metal"],
        "lb/st" => ["lb metal", "klb metal", "Mlb metal"],
        "%" => ["kg metal", "t metal", "kt metal"],
        _ => ["g metal", "kg metal", "t metal"],
    }
}

/// The unit of `family` (smallest first) that `values`, in `unit`, read
/// best in: the largest where the median non-zero magnitude is at least 1.
/// Returns it with the factor taking the values there.
pub fn autoscale(values: &[f64], unit: &str, family: &[&str]) -> Result<(String, f64)> {
    let mut magnitudes: Vec<f64> = values
        .iter()
        .map(|v| v.abs())
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    magnitudes.sort_by(f64::total_cmp);
    let median = magnitudes.get(magnitudes.len() / 2).copied().unwrap_or(0.0);
    let mut best = (family[0].to_string(), conversion(unit, family[0])?);
    for to in &family[1..] {
        let factor = conversion(unit, to)?;
        if median * factor >= 1.0 {
            best = (to.to_string(), factor);
        }
    }
    Ok(best)
}

/// Errors unless `unit` is a length, as coordinates need.
pub fn check_length(unit: &str) -> Result<()> {
    let parsed = parse(unit)?;
    if parsed.dimension != Dimension::of(Axis::Length) {
        return Err(Error::Units(format!(
            "coordinates need a length unit such as m or ft, got {} ({unit})",
            parsed.dimension
        )));
    }
    Ok(())
}

/// The factor taking coordinates in `from` to `unit`, and the CRS they then
/// have: `crs` must name it when the coordinates had one.
pub fn rescale(
    from: Option<&str>,
    old_crs: Option<&str>,
    unit: &str,
    crs: Option<String>,
) -> Result<(f64, Option<String>)> {
    check_length(unit)?;
    let from = from.ok_or_else(|| {
        Error::Units("declare the length unit of the coordinates before converting them".into())
    })?;
    if let (Some(old), None) = (old_crs, &crs) {
        return Err(Error::Units(format!(
            "the coordinates are in CRS {old}; give their CRS in {unit} as crs="
        )));
    }
    Ok((conversion(from, unit)?, crs))
}

/// Errors when `a` and `b` (what each holds, such as "samples" and
/// "targets") both declare a length unit and they differ; an undeclared one
/// matches any.
pub fn same_length_unit(a: (&str, Option<&str>), b: (&str, Option<&str>)) -> Result<()> {
    match (a.1, b.1) {
        (Some(x), Some(y)) if x != y => Err(Error::Units(format!(
            "{} are in {x} and {} in {y}; convert one with to_length_unit('{x}')",
            a.0, b.0
        ))),
        _ => Ok(()),
    }
}

/// The unit of column `name`, if it has one.
pub fn unit(table: &RecordBatch, name: &str) -> Result<Option<String>> {
    let schema = table.schema();
    let field = schema
        .field_with_name(name)
        .map_err(|_| Error::MissingColumn(name.into()))?;
    Ok(field.metadata().get(KEY).cloned())
}

/// The columns that have a unit, in column order.
pub fn units(table: &RecordBatch) -> Vec<(String, String)> {
    table
        .schema()
        .fields()
        .iter()
        .filter_map(|f| Some((f.name().clone(), f.metadata().get(KEY)?.clone())))
        .collect()
}

/// `table` with the unit of column `name` set, or removed for `None`. The
/// unit must parse; see [`with_unit_unchecked`] for readers.
pub fn with_unit(table: &RecordBatch, name: &str, unit: Option<&str>) -> Result<RecordBatch> {
    if let Some(u) = unit {
        parse(u)?;
    }
    with_unit_unchecked(table, name, unit)
}

/// [`with_unit`] keeping a unit as written even if it does not parse, as
/// readers do: such a unit is shown but never converted.
pub fn with_unit_unchecked(
    table: &RecordBatch,
    name: &str,
    unit: Option<&str>,
) -> Result<RecordBatch> {
    let schema = table.schema();
    let i = schema
        .index_of(name)
        .map_err(|_| Error::MissingColumn(name.into()))?;
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut metadata: HashMap<String, String> = fields[i].metadata().clone();
    match unit {
        Some(u) => metadata.insert(KEY.into(), u.into()),
        None => metadata.remove(KEY),
    };
    fields[i] = fields[i].clone().with_metadata(metadata);
    let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    Ok(RecordBatch::try_new_with_options(
        schema,
        table.columns().to_vec(),
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
    )?)
}

/// `table` with column `name` converted from its unit to `to`, which must
/// have the same dimension.
pub fn convert_units(table: &RecordBatch, name: &str, to: &str) -> Result<RecordBatch> {
    let from =
        unit(table, name)?.ok_or_else(|| Error::Units(format!("column `{name}` has no unit")))?;
    let parsed = parse(&from).map_err(|_| {
        Error::Units(format!(
            "the unit `{from}` of column `{name}` is not understood"
        ))
    })?;
    let factor = scale((&from, &parsed), (to, &parse(to)?))?;
    let column = table.column(table.schema().index_of(name)?);
    let values = cast(column, &DataType::Float64)
        .map_err(|_| Error::Units(format!("column `{name}` is not numeric")))?;
    let scaled: ArrayRef = Arc::new(
        values
            .as_primitive::<Float64Type>()
            .unary::<_, Float64Type>(|v| v * factor),
    );
    let replaced = crate::set_column(table, name, scaled)?;
    with_unit(&replaced, name, Some(to))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float64Array};
    use proptest::prelude::*;

    fn same(a: &Unit, b: &Unit) -> bool {
        a.dimension == b.dimension && (a.factor / b.factor - 1.0).abs() < 1e-12
    }

    fn u(text: &str) -> Unit {
        parse(text).unwrap()
    }

    fn grades() -> RecordBatch {
        let au: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(34.285_714)]));
        let batch = RecordBatch::try_from_iter([("au", au)]).unwrap();
        with_unit(&batch, "au", Some("g/t")).unwrap()
    }

    /// Theory check: volume × density is ore mass, ore mass × grade is metal,
    /// and metal over ore mass is the grade back.
    #[test]
    fn tonnage_and_metal_follow_from_the_algebra() {
        assert!(same(&(&u("m3") * &u("t/m3")), &u("t")));
        let ore = &u("km^3") * &u("g/cm3");
        assert!(ore.dimension() == u("t").dimension() && (ore.factor() - 1e9).abs() < 1e-3);
        assert!(same(&(&u("Mt") * &u("g/t")), &u("t metal")));
        assert!(same(&(&u("st") * &u("oz/t")), &u("oz metal")));
        assert!(same(&(&u("kg metal") / &u("t")), &u("kg/t")));
        assert_eq!((&u("g/t") * &u("ratio%")).dimension(), u("%").dimension());
        assert!(same(&(&u("mg") / &u("kg")), &u("ppm")));
        assert!(same(&(&u("t") * &u("t^-1")), &u("t/t")));
        assert_eq!(u("oz/st").dimension(), u("g/t").dimension());
        assert!(same(&(&u("USD/t") * &u("t")), &u("USD")));
        assert!(same(&(&u("t/m3") / &u("g/cm3")), &u("ratio")));
        assert!(same(&u("(g/t)^2"), &(&u("g/t") * &u("g/t"))));
        assert!(same(&u("m²"), &u("m^2")) && same(&u("m^-1"), &(&u("m") / &u("m2"))));
    }

    #[test]
    fn autoscale_picks_the_largest_unit_the_values_reach() {
        let (unit, k) = autoscale(&[2.0e6, 3.0e6, 0.0], "t", &TONNAGE).unwrap();
        assert_eq!((unit.as_str(), k), ("Mt", 1e-6));
        let (unit, _) = autoscale(&[500.0], "t", &TONNAGE).unwrap();
        assert_eq!(unit, "t");
        let tonnes_times_grade = "t*(oz/t)";
        let (unit, k) = autoscale(&[3.2e6], tonnes_times_grade, &metal_units("oz/t")).unwrap();
        assert_eq!(unit, "Moz metal");
        assert!((3.2e6 * k - 3.2e6 * TROY_OUNCE / SHORT_TON / (TROY_OUNCE * 1e6)).abs() < 1e-12);
        assert_eq!(autoscale(&[], "t", &TONNAGE).unwrap().0, "t");
        assert!(autoscale(&[1.0], "m", &TONNAGE).is_err());
    }

    #[test]
    fn grade_conversions_match_known_factors() {
        assert!((conversion("oz/t", "g/t").unwrap() - 34.285_714_285_714).abs() < 1e-9);
        assert!((conversion("%", "ppm").unwrap() - 1e4).abs() < 1e-9);
        assert!((conversion("lb/st", "%").unwrap() - 0.05).abs() < 1e-12);
        assert!((conversion("kUSD", "USD").unwrap() - 1e3).abs() < 1e-12);
        assert!((conversion("g/cm3", "t/m3").unwrap() - 1.0).abs() < 1e-12);
        assert!((conversion("ft", "m").unwrap() - 0.3048).abs() < 1e-12);
    }

    #[test]
    fn absurd_conversions_are_refused() {
        for (from, to) in [
            ("%", "ratio%"),
            ("deg", "g/t"),
            ("t metal", "t"),
            ("USD", "BRL"),
            ("g/t", "m"),
            ("ratio", "rad"),
        ] {
            let e = conversion(from, to).unwrap_err().to_string();
            assert!(e.starts_with("cannot convert"), "{from} -> {to}: {e}");
        }
        assert_eq!(
            conversion("%", "ratio%").unwrap_err().to_string(),
            "cannot convert grade (%) to ratio (ratio%)"
        );
    }

    #[test]
    fn unknown_units_are_named_with_a_suggestion() {
        assert_eq!(
            parse("ppn").unwrap_err().to_string(),
            "unknown unit `ppn`; did you mean `ppm`?"
        );
        assert_eq!(
            parse("furlong").unwrap_err().to_string(),
            "unknown unit `furlong`"
        );
        for bad in ["", "g/tonne", "m^", "(m", "kgx", "m metal", "metal"] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn conversions_round_trip_and_keep_nulls() {
        let t = grades();
        assert_eq!(unit(&t, "au").unwrap().as_deref(), Some("g/t"));
        let oz = convert_units(&t, "au", "oz/t").unwrap();
        let v = oz.column(0).as_primitive::<Float64Type>();
        assert!((v.value(2) - 1.0).abs() < 1e-6 && v.is_null(1));
        let back = convert_units(&convert_units(&t, "au", "ppb").unwrap(), "au", "g/t").unwrap();
        assert!((back.column(0).as_primitive::<Float64Type>().value(0) - 1.0).abs() < 1e-12);
        assert_eq!(units(&back), vec![("au".into(), "g/t".into())]);
    }

    #[test]
    fn refuses_other_dimensions_and_unknown_or_missing_units() {
        let t = grades();
        assert!(matches!(convert_units(&t, "au", "m"), Err(Error::Units(_))));
        assert!(matches!(
            convert_units(&t, "au", "furlong"),
            Err(Error::Units(_))
        ));
        assert!(matches!(
            with_unit(&t, "au", Some("dwt")),
            Err(Error::Units(_))
        ));
        let opaque = with_unit_unchecked(&t, "au", Some("dwt")).unwrap();
        assert_eq!(unit(&opaque, "au").unwrap().as_deref(), Some("dwt"));
        assert_eq!(
            convert_units(&opaque, "au", "g/t").unwrap_err().to_string(),
            "the unit `dwt` of column `au` is not understood"
        );
        let none = with_unit(&t, "au", None).unwrap();
        assert!(units(&none).is_empty());
        assert!(matches!(
            convert_units(&none, "au", "ppm"),
            Err(Error::Units(_))
        ));
    }

    fn atom() -> impl Strategy<Value = &'static str> {
        prop::sample::select(
            catalog()
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
        )
    }

    proptest! {
        #[test]
        fn compound_units_parse_and_convert_both_ways(a in atom(), b in atom(), n in 1i32..4) {
            let text = format!("({a})^{n}/({b})");
            let unit = parse(&text).unwrap();
            prop_assert!(same(&unit, &(&u(a).powi(n) / &u(b))));
            let there = conversion(&text, &format!("({a})^{n}*({b})^-1"));
            prop_assert!((there.unwrap() - 1.0).abs() < 1e-12);
            prop_assert!((conversion(a, a).unwrap() - 1.0).abs() < 1e-15);
        }
    }
}
