//! `#[serde(with = "boitata_core::nonfinite")]` for floats that may be NaN or
//! infinite, which JSON has no numbers for: they are written as the strings
//! `"NaN"`, `"inf"` and `"-inf"`. Works on `f64` and on options, pairs and
//! vectors of it.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum Float {
    Number(f64),
    Text(String),
}

pub trait Floats: Sized {
    type Repr: Serialize + for<'de> Deserialize<'de>;
    fn repr(&self) -> Self::Repr;
    fn parse(repr: Self::Repr) -> Result<Self, String>;
}

impl Floats for f64 {
    type Repr = Float;

    fn repr(&self) -> Float {
        match *self {
            v if v.is_finite() => Float::Number(v),
            v if v.is_nan() => Float::Text("NaN".into()),
            v => Float::Text(if v > 0.0 { "inf" } else { "-inf" }.into()),
        }
    }

    fn parse(repr: Float) -> Result<Self, String> {
        match repr {
            Float::Number(v) => Ok(v),
            Float::Text(t) => match t.as_str() {
                "NaN" => Ok(f64::NAN),
                "inf" => Ok(f64::INFINITY),
                "-inf" => Ok(f64::NEG_INFINITY),
                _ => Err(format!("{t:?} is not a number")),
            },
        }
    }
}

impl<T: Floats> Floats for Option<T> {
    type Repr = Option<T::Repr>;

    fn repr(&self) -> Self::Repr {
        self.as_ref().map(T::repr)
    }

    fn parse(repr: Self::Repr) -> Result<Self, String> {
        repr.map(T::parse).transpose()
    }
}

impl<A: Floats, B: Floats> Floats for (A, B) {
    type Repr = (A::Repr, B::Repr);

    fn repr(&self) -> Self::Repr {
        (self.0.repr(), self.1.repr())
    }

    fn parse(repr: Self::Repr) -> Result<Self, String> {
        Ok((A::parse(repr.0)?, B::parse(repr.1)?))
    }
}

impl<T: Floats> Floats for Vec<T> {
    type Repr = Vec<T::Repr>;

    fn repr(&self) -> Self::Repr {
        self.iter().map(T::repr).collect()
    }

    fn parse(repr: Self::Repr) -> Result<Self, String> {
        repr.into_iter().map(T::parse).collect()
    }
}

pub fn serialize<T: Floats, S: Serializer>(value: &T, s: S) -> Result<S::Ok, S::Error> {
    value.repr().serialize(s)
}

pub fn deserialize<'de, T: Floats, D: Deserializer<'de>>(d: D) -> Result<T, D::Error> {
    T::parse(T::Repr::deserialize(d)?).map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct S {
        #[serde(with = "super")]
        x: Vec<f64>,
        #[serde(with = "super")]
        pair: Option<(f64, f64)>,
    }

    #[test]
    fn nan_and_infinities_round_trip() {
        let s = S {
            x: vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.1, -0.0, 1e300],
            pair: Some((f64::NEG_INFINITY, 2.5)),
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: S = serde_json::from_str(&json).unwrap();
        assert!(back.x[0].is_nan());
        let bits = |v: &[f64]| v[1..].iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&back.x), bits(&s.x));
        assert_eq!(back.pair, s.pair);
        assert!(serde_json::from_str::<S>(r#"{"x":["one"],"pair":null}"#).is_err());
    }
}
