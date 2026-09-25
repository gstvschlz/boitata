use pyo3::prelude::*;
use pyo3::types::PyType;

mod args;
mod coda;
mod cokriging;
mod containers;
mod drillholes;
mod estimation;
mod io;
mod simulation;
mod table;
mod transforms;
mod variogram;

/// Raises `ceres.errors.<kind>`, a subclass of both `CeresError` and a builtin.
pub(crate) fn error(kind: &str, message: impl ToString) -> PyErr {
    Python::attach(|py| {
        match py
            .import("ceres.errors")
            .and_then(|m| m.getattr(kind))
            .and_then(|c| Ok(c.cast_into::<PyType>()?))
        {
            Ok(class) => PyErr::from_type(class, message.to_string()),
            Err(e) => e,
        }
    })
}

pub(crate) fn invalid(message: impl ToString) -> PyErr {
    error("InvalidInput", message)
}

#[pymodule]
fn _ceres(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<table::Table>()?;
    m.add_class::<containers::PyPointSet>()?;
    m.add_class::<containers::PyBlockModel>()?;
    io::register(m)?;
    transforms::register(m)?;
    variogram::register(m)?;
    estimation::register(m)?;
    cokriging::register(m)?;
    drillholes::register(m)?;
    coda::register(m)?;
    simulation::register(m)?;
    Ok(())
}
