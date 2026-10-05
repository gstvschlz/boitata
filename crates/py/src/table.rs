use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{
    ArrayRef, Float32Array, Float64Array, RecordBatch, RecordBatchOptions, StructArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::ffi::{
    ArrayIterator, to_array_pycapsules, to_schema_pycapsule, to_stream_pycapsule,
};
use pyo3_arrow::input::AnyRecordBatch;

use crate::invalid;

/// The units of `batch`'s columns that have one.
pub fn units<'py>(py: Python<'py>, batch: &RecordBatch) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, unit) in boitata_core::units::units(batch) {
        dict.set_item(name, unit)?;
    }
    Ok(dict)
}

/// `batch` with the units of `units` set; None removes one.
pub fn with_units(
    batch: &RecordBatch,
    units: std::collections::HashMap<String, Option<String>>,
) -> PyResult<RecordBatch> {
    let mut batch = batch.clone();
    for (name, unit) in units {
        batch = boitata_core::units::with_unit(&batch, &name, unit.as_deref()).map_err(invalid)?;
    }
    Ok(batch)
}

pub fn convert_units(batch: &RecordBatch, column: &str, to: &str) -> PyResult<RecordBatch> {
    boitata_core::units::convert_units(batch, column, to).map_err(invalid)
}

/// Columnar attribute table; exchanges data with pyarrow, polars and pandas
/// through the Arrow PyCapsule interface.
#[pyclass(module = "boitata", name = "Table", frozen)]
pub struct Table(pub RecordBatch);

pub fn empty(rows: usize) -> RecordBatch {
    RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .expect("empty batch")
}

fn require<'py>(
    py: Python<'py>,
    method: &str,
    module: &str,
    packages: &str,
) -> PyResult<Bound<'py, PyModule>> {
    py.import(module).map_err(|e| {
        let err = PyImportError::new_err(format!(
            "Table.{method} needs {packages}: pip install {packages} or conda install -c conda-forge {packages}"
        ));
        err.set_cause(py, Some(e));
        err
    })
}

/// Any Arrow-compatible object or a dict of 1-D numeric (NaN is null), boolean or text arrays.
pub fn to_batch(data: &Bound<PyAny>) -> PyResult<RecordBatch> {
    if let Ok(dict) = data.cast::<PyDict>() {
        let np = data.py().import("numpy")?;
        let mut columns = Vec::with_capacity(dict.len());
        for (name, values) in dict.iter() {
            let array = np.call_method1("asarray", (&values,))?;
            let kind: String = array.getattr("dtype")?.getattr("kind")?.extract()?;
            let column: ArrayRef = if matches!(kind.as_str(), "U" | "S" | "O") {
                let text: Vec<Option<String>> = array
                    .call_method0("tolist")?
                    .extract()
                    .map_err(|_| invalid(format!("column {name} is not a 1-D array of text")))?;
                Arc::new(arrow_array::StringArray::from(text))
            } else if kind == "b" {
                let flags: Vec<bool> = array.call_method0("tolist")?.extract().map_err(|_| {
                    invalid(format!("column {name} is not a 1-D array of booleans"))
                })?;
                Arc::new(arrow_array::BooleanArray::from(flags))
            } else {
                let not_numeric = || invalid(format!("column {name} is not a 1-D numeric array"));
                let dtype: String = array.getattr("dtype")?.getattr("name")?.extract()?;
                let wide = if dtype == "float32" {
                    "float32"
                } else {
                    "float64"
                };
                let array = np
                    .call_method1("asarray", (values, wide))
                    .map_err(|_| not_numeric())?;
                let has_nan: bool = np
                    .call_method1("isnan", (&array,))
                    .map_err(|_| not_numeric())?
                    .call_method0("any")?
                    .extract()?;
                if wide == "float32" {
                    let values: PyReadonlyArray1<f32> =
                        array.extract().map_err(|_| not_numeric())?;
                    let v = values.as_array();
                    match has_nan {
                        true => Arc::new(
                            v.iter()
                                .map(|v| (!v.is_nan()).then_some(*v))
                                .collect::<Float32Array>(),
                        ),
                        false => Arc::new(Float32Array::from_iter_values(v.iter().copied())),
                    }
                } else {
                    let values: PyReadonlyArray1<f64> =
                        array.extract().map_err(|_| not_numeric())?;
                    let v = values.as_array();
                    match has_nan {
                        true => Arc::new(
                            v.iter()
                                .map(|v| (!v.is_nan()).then_some(*v))
                                .collect::<Float64Array>(),
                        ),
                        false => Arc::new(Float64Array::from_iter_values(v.iter().copied())),
                    }
                }
            };
            columns.push((name.extract::<String>()?, column));
        }
        return RecordBatch::try_from_iter(columns).map_err(invalid);
    }
    let table = data
        .extract::<AnyRecordBatch>()
        .map_err(|_| invalid("expected an Arrow-compatible table or a dict of arrays"))?
        .into_table()?;
    let (batches, schema) = table.into_inner();
    arrow_select::concat::concat_batches(&schema, &batches).map_err(invalid)
}

/// As [`to_batch`], but an Arrow table stays in its batches.
pub fn to_batches(data: &Bound<PyAny>) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    if data.cast::<PyDict>().is_ok() {
        let batch = to_batch(data)?;
        return Ok((batch.schema(), vec![batch]));
    }
    let table = data
        .extract::<AnyRecordBatch>()
        .map_err(|_| invalid("expected an Arrow-compatible table or a dict of arrays"))?
        .into_table()?;
    let (batches, schema) = table.into_inner();
    Ok((schema, batches))
}

/// `MissingColumn` for `name`, listing the `columns` there are.
pub fn missing(name: &str, columns: Vec<String>) -> PyErr {
    crate::error(
        "MissingColumn",
        format!("no column {name:?}; columns: {}", columns.join(", ")),
    )
}

pub fn names(batch: &RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// Numeric columns as float64 arrays (null as NaN), boolean ones as bool
/// arrays (null as false), others as object arrays of str (null as None).
pub fn column<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| missing(name, names(batch)))?;
    if let Some(flags) = array.as_boolean_opt() {
        let values: Vec<bool> = flags.iter().map(|v| v.unwrap_or(false)).collect();
        return Ok(PyArray1::from_vec(py, values).into_any());
    }
    if array.data_type().is_numeric() {
        let values = arrow_cast::cast(array, &DataType::Float64).map_err(invalid)?;
        let values: Vec<f64> = values
            .as_primitive::<Float64Type>()
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        return Ok(PyArray1::from_vec(py, values).into_any());
    }
    let text = arrow_cast::cast(array, &DataType::Utf8).map_err(invalid)?;
    let items: Vec<Option<&str>> = text.as_string::<i32>().iter().collect();
    py.import("numpy")?
        .call_method1("array", (PyList::new(py, items)?, "object"))
}

/// The numeric column `name` as floats, null as NaN.
pub fn floats(batch: &RecordBatch, name: &str) -> PyResult<Vec<f64>> {
    let array = batch
        .column_by_name(name)
        .ok_or_else(|| missing(name, names(batch)))?;
    if !array.data_type().is_numeric() {
        return Err(invalid(format!("column {name} is not numeric")));
    }
    let values = arrow_cast::cast(array, &DataType::Float64).map_err(invalid)?;
    Ok(values
        .as_primitive::<Float64Type>()
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect())
}

fn struct_field(batch: &RecordBatch) -> Arc<Field> {
    let schema = batch.schema();
    Arc::new(
        Field::new_struct("", schema.fields().clone(), false)
            .with_metadata(schema.metadata().clone()),
    )
}

pub fn arrow_c_stream<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
    requested_schema: Option<Bound<'py, PyCapsule>>,
) -> PyArrowResult<Bound<'py, PyCapsule>> {
    let array: ArrayRef = Arc::new(StructArray::from(batch.clone()));
    let reader = ArrayIterator::new(vec![Ok(array)], struct_field(batch));
    to_stream_pycapsule(py, Box::new(reader), requested_schema)
}

pub fn describe(batch: &RecordBatch) -> String {
    batch
        .schema()
        .fields()
        .iter()
        .map(|f| format!("\n  {}: {}", f.name(), f.data_type()))
        .collect()
}

#[pymethods]
impl Table {
    #[new]
    fn new(data: &Bound<PyAny>) -> PyResult<Self> {
        Ok(Self(to_batch(data)?))
    }

    #[getter]
    fn num_rows(&self) -> usize {
        self.0.num_rows()
    }

    #[getter]
    fn column_names(&self) -> Vec<String> {
        names(&self.0)
    }

    fn column<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        column(py, &self.0, name)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        column(py, &self.0, name)
    }

    /// Units of the columns that have one, kept in the Arrow field metadata
    /// and through Parquet.
    #[getter(units)]
    fn units_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        units(py, &self.0)
    }

    /// New table with the units of `units`, column name to unit; None
    /// removes one. Replacing a column's values drops its unit.
    #[pyo3(name = "with_units")]
    fn with_units_(
        &self,
        units: std::collections::HashMap<String, Option<String>>,
    ) -> PyResult<Self> {
        Ok(Self(with_units(&self.0, units)?))
    }

    /// New table with `column` converted to unit `to`: grades (ppb, ppm, g/t,
    /// %, oz/t as troy ounces per short ton), lengths (m, ft) or densities
    /// (t/m3, g/cm3). The column must have a unit.
    #[pyo3(name = "convert_units", signature = (column, *, to))]
    fn convert_units_(&self, column: &str, to: &str) -> PyResult<Self> {
        Ok(Self(convert_units(&self.0, column, to)?))
    }

    /// Rows where `mask` is true.
    fn filter(&self, mask: numpy::PyReadonlyArray1<bool>) -> PyResult<Self> {
        let mask = mask.as_array();
        if mask.len() != self.0.num_rows() {
            return Err(invalid(format!(
                "mask has {} values, the table {} rows",
                mask.len(),
                self.0.num_rows()
            )));
        }
        let mask = arrow_array::BooleanArray::from(mask.to_vec());
        Ok(Self(
            arrow_select::filter::filter_record_batch(&self.0, &mask).map_err(invalid)?,
        ))
    }

    fn __len__(&self) -> usize {
        self.0.num_rows()
    }

    fn __repr__(&self) -> String {
        format!("Table({} rows){}", self.0.num_rows(), describe(&self.0))
    }

    fn to_polars<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        require(slf.py(), "to_polars", "polars", "polars")?.call_method1("DataFrame", (slf,))
    }

    fn to_pyarrow<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        require(slf.py(), "to_pyarrow", "pyarrow", "pyarrow")?.call_method1("table", (slf,))
    }

    fn to_pandas<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        require(py, "to_pandas", "pandas", "pandas pyarrow")?;
        require(py, "to_pandas", "pyarrow", "pandas pyarrow")?
            .call_method1("table", (slf,))?
            .call_method0("to_pandas")
    }

    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.0.schema().as_ref())
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyTuple>> {
        let array = StructArray::from(self.0.clone());
        to_array_pycapsules(py, struct_field(&self.0), &array, requested_schema)
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        arrow_c_stream(py, &self.0, requested_schema)
    }
}
