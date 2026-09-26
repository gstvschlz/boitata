use std::collections::BTreeMap;

use ceres_core::Categories as Core;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

use crate::args::{array1, floats, optional_finite, text, texts};
use crate::invalid;

/// Named categories with integer codes: a code is the index of its name.
///
/// Parameters
/// ----------
/// names : sequence of str
///     Category names in code order.
/// colors : sequence of str, optional
///     One matplotlib color per name; None picks them at plot time.
/// mapping : dict, optional
///     Raw label to name, many to one.
/// other : str, optional
///     Name that unlisted labels fall into; appended when not listed, and
///     always the last code.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
#[pyclass(module = "ceres", name = "Categories", frozen, eq, from_py_object)]
pub struct Categories(Core);

fn name(obj: Bound<PyAny>) -> PyResult<String> {
    text(&obj)?.ok_or_else(|| invalid("names and mapping entries must not be null"))
}

fn mapping(obj: Option<&Bound<PyDict>>) -> PyResult<BTreeMap<String, String>> {
    obj.map_or(Ok(BTreeMap::new()), |d| {
        d.iter().map(|(k, v)| Ok((name(k)?, name(v)?))).collect()
    })
}

fn labels(obj: &Bound<PyAny>) -> PyResult<Vec<Option<String>>> {
    texts(obj, "values")
}

fn refs(labels: &[Option<String>]) -> Vec<Option<&str>> {
    labels.iter().map(Option::as_deref).collect()
}

fn codes(obj: &Bound<PyAny>) -> PyResult<Vec<Option<u32>>> {
    floats(obj, "codes")?
        .into_iter()
        .map(|c| match c {
            c if c.is_nan() => Ok(None),
            c if c >= 0.0 && c.fract() == 0.0 && c < u32::MAX as f64 => Ok(Some(c as u32)),
            c => Err(invalid(format!("{c} is not a code"))),
        })
        .collect()
}

#[pymethods]
impl Categories {
    #[new]
    #[pyo3(signature = (names, *, colors=None, mapping=None, other=None))]
    fn new(
        names: &Bound<PyAny>,
        colors: Option<Vec<String>>,
        mapping: Option<&Bound<PyDict>>,
        other: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        let names = names
            .try_iter()?
            .map(|n| name(n?))
            .collect::<PyResult<_>>()?;
        let other = other.map(|o| name(o.clone())).transpose()?;
        Core::new(names, colors, self::mapping(mapping)?, other)
            .map(Self)
            .map_err(invalid)
    }

    /// Categories of observed values.
    ///
    /// `mapping` is applied first; the distinct names are then sorted,
    /// numerically when all are integers. Names whose weighted share is
    /// below `min_share` are lumped into `other`, the last code, and
    /// recorded in `mapping`; `other` exists only when something was
    /// lumped. Integral floats are integer labels (1.0 is "1"); None and
    /// NaN are skipped. The result does not depend on row order.
    #[staticmethod]
    #[pyo3(signature = (values, *, weights=None, min_share=0.0, mapping=None, other="other", colors=None))]
    fn from_values(
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        min_share: f64,
        mapping: Option<&Bound<PyDict>>,
        other: &str,
        colors: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let labels = labels(values)?;
        let weights = optional_finite(weights, "weights")?;
        let c = Core::from_labels(
            &refs(&labels),
            weights.as_deref(),
            min_share,
            self::mapping(mapping)?,
            other,
        )
        .map_err(invalid)?;
        let other = c.other().map(str::to_string);
        Core::new(c.names().to_vec(), colors, c.mapping().clone(), other)
            .map(Self)
            .map_err(invalid)
    }

    /// Code of each value as float64, NaN for null. An unlisted label goes
    /// to `other`, else raises InvalidInput.
    fn encode<'py>(&self, py: Python<'py>, values: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let codes = self.0.encode(&refs(&labels(values)?)).map_err(invalid)?;
        let codes = codes.into_iter().map(|c| c.map_or(f64::NAN, f64::from));
        Ok(array1(py, codes.collect()).into_any())
    }

    /// Name of each code; None for NaN.
    fn decode(&self, codes: &Bound<PyAny>) -> PyResult<Vec<Option<String>>> {
        let names = self.0.decode(&self::codes(codes)?).map_err(invalid)?;
        Ok(names.into_iter().map(|n| n.map(str::to_string)).collect())
    }

    /// New categories with `names` merged into `into`: a listed name, else
    /// `other` when there is none, else a new name before `other`. The
    /// merged names become mapping entries.
    #[pyo3(signature = (names, *, into="other"))]
    fn lump(&self, names: Vec<String>, into: &str) -> PyResult<Self> {
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        self.0.lump(&names, into).map(Self).map_err(invalid)
    }

    /// Weighted proportion of each code, nulls skipped.
    #[pyo3(signature = (codes, *, weights=None))]
    fn shares<'py>(
        &self,
        py: Python<'py>,
        codes: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let weights = optional_finite(weights, "weights")?;
        let shares = self
            .0
            .shares(&self::codes(codes)?, weights.as_deref())
            .map_err(invalid)?;
        Ok(array1(py, shares).into_any())
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.0.names().to_vec()
    }

    #[getter]
    fn colors(&self) -> Option<Vec<String>> {
        self.0.colors().map(<[String]>::to_vec)
    }

    #[getter]
    fn other(&self) -> Option<&str> {
        self.0.other()
    }

    #[getter]
    fn mapping(&self) -> BTreeMap<String, String> {
        self.0.mapping().clone()
    }

    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        let other = self
            .0
            .other()
            .map_or(String::new(), |o| format!(", other={o:?}"));
        format!("Categories({:?}{other})", self.0.names())
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Categories>()
}
