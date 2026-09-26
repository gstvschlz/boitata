use pyo3::PyClass;
use pyo3::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::invalid;

const FORMAT: u64 = 1;

/// `{"type": <class name>, "format": 1, <fields of value>...}`.
pub fn to_json<T: PyClass + Serialize>(value: &T) -> PyResult<String> {
    let fields = serde_json::to_string(value).map_err(invalid)?;
    let name = <T as PyClass>::NAME;
    let fields = match fields.strip_prefix('{') {
        Some("}") => "}".into(),
        Some(rest) => format!(",{rest}"),
        None => return Err(invalid(format!("{name} does not serialize to an object"))),
    };
    Ok(format!(r#"{{"type":"{name}","format":{FORMAT}{fields}"#))
}

pub fn from_json<T: PyClass + DeserializeOwned>(text: &str) -> PyResult<T> {
    let Value::Object(mut map) = serde_json::from_str(text).map_err(invalid)? else {
        return Err(invalid("expected a JSON object"));
    };
    let name = <T as PyClass>::NAME;
    let found = map.remove("type").unwrap_or_default();
    if found.as_str() != Some(name) {
        return Err(invalid(format!("expected a {name}, found {found}")));
    }
    match map.remove("format").and_then(|f| f.as_u64()) {
        Some(f) if f <= FORMAT => {}
        f => {
            return Err(invalid(format!(
                "format {f:?} is not readable by this ceres (up to {FORMAT})"
            )));
        }
    }
    serde_json::from_value(Value::Object(map)).map_err(invalid)
}
