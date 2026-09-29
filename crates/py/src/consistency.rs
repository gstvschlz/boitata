use pyo3::prelude::*;
use pyo3::types::PyDict;
use simulation::{ConsistencyParams, TrainingImage};

use crate::args::{array1, floats};
use crate::containers::{PyBlockModel, coords_arg};
use crate::invalid;

/// How often the patterns of the hard data occur in a training image, after
/// Boisvert, Pyrcz and Deutsch (2007); run it to reject an image before
/// simulating with it.
///
/// The data are placed in the cells of `grid`, one per cell. Along `axis`,
/// runs of consecutive informed cells (the holes) are cut into data events
/// of `pattern_length` cells, and the frequencies of their patterns are
/// compared with those of every run of `pattern_length` cells along the same
/// axis of the image, by their Jensen-Shannon divergence: 0 for the same
/// frequencies, 1 when no pattern is shared. Holes of the data's lengths are
/// then drilled `n_samples` times through the image at random columns and
/// depths and compared with it the same way; the p-value is the share of
/// these reference divergences at least as large as the data's.
///
/// Parameters
/// ----------
/// ti : BlockModel
///     The training image, regular or masked; read one image cell per grid
///     cell, as in a simulation.
/// column : str
///     Column of `ti` holding the image: integer codes 0 to 254 when
///     `categorical`, else values.
/// coords : PointSet or array_like
///     The hard data, or their ``(n, 2)`` or ``(n, 3)`` coordinates, sampled
///     at about the grid's cell size along `axis`.
/// values : array_like or str
///     Codes of the image (categorical) or values of the hard data, or their
///     column in `coords`. Data without a value or outside `grid` are left
///     out.
/// grid : BlockModel, optional
///     The simulation grid; by default the geometry of `ti`.
/// categorical : bool, default True
///     Compare codes; otherwise cut the values into `n_classes` classes.
/// axis : {"x", "y", "z"}, optional
///     Axis the holes follow; by default "z" on a 3D grid, "y" on a 2D one.
/// pattern_length : int, default 4
///     Cells in a data event, at least 2. Longer patterns tell images apart
///     better but need longer runs of data.
/// n_classes : int, default 4
///     Continuous images only: classes cut at the quantiles of the hard data,
///     2 to 254.
/// n_samples : int, default 200
///     Reference draws; the smallest p-value is ``1 / (n_samples + 1)``.
/// seed : int, default 0
///     The same seed gives the same result on any number of threads.
///
/// Returns
/// -------
/// dict
///     ``distance`` (the data's divergence from the image, 0 to 1),
///     ``p_value`` (below 0.05, holes like the data seldom come from the
///     image), ``reference`` (the reference divergences, ascending),
///     ``unseen`` (share of the data events whose pattern the image lacks),
///     ``n_events``, ``n_cells`` (grid cells with data), and for categorical
///     images ``proportions`` and ``data_proportions`` (share of each code in
///     the image and in the data cells; None for continuous images).
///
/// Raises
/// ------
/// InvalidInput
///     If a data code is not a code of the image, an option is out of range,
///     the grid or image is shorter than `pattern_length` along `axis`, or no
///     hole has `pattern_length` consecutive informed cells.
///
/// Notes
/// -----
/// Only one axis is compared: vertical holes test the thickness, order and
/// proportions of the units, not the width or orientation of bodies.
///
/// Examples
/// --------
/// >>> check = cs.training_image_consistency(ti, "facies", holes, "facies", seed=0)
/// >>> check["p_value"]
#[pyfunction]
#[pyo3(signature = (ti, column, coords, values, *, grid=None, categorical=true, axis=None, pattern_length=4, n_classes=4, n_samples=200, seed=0))]
#[allow(clippy::too_many_arguments)]
fn training_image_consistency<'py>(
    py: Python<'py>,
    ti: PyRef<PyBlockModel>,
    column: &str,
    coords: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    grid: Option<PyRef<PyBlockModel>>,
    categorical: bool,
    axis: Option<&str>,
    pattern_length: usize,
    n_classes: usize,
    n_samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let values = floats(
        &crate::args::column(Some(coords), values, "values")?,
        "values",
    )?;
    let coords = coords_arg(coords)?;
    let geometry = *grid.as_ref().unwrap_or(&ti).0.geometry();
    let axis = match axis {
        None if geometry.count[2] == 1 => 1,
        None => 2,
        Some("x") => 0,
        Some("y") => 1,
        Some("z") => 2,
        Some(a) => {
            return Err(invalid(format!(
                "axis must be \"x\", \"y\" or \"z\", got {a:?}"
            )));
        }
    };
    let params = ConsistencyParams {
        axis,
        pattern_length,
        n_classes,
        n_samples,
        seed,
    };
    let model = &ti.0;
    let result = py
        .detach(|| {
            let image = match categorical {
                true => TrainingImage::categorical(model, column),
                false => TrainingImage::continuous(model, column),
            }?;
            simulation::consistency(&image, &geometry, &coords, &values, &params)
        })
        .map_err(invalid)?;
    let d = PyDict::new(py);
    d.set_item("distance", result.distance)?;
    d.set_item("p_value", result.p_value)?;
    d.set_item("reference", array1(py, result.reference))?;
    d.set_item("unseen", result.unseen)?;
    d.set_item("n_events", result.n_events)?;
    d.set_item("n_cells", result.n_cells)?;
    for (key, shares) in [
        ("proportions", result.proportions),
        ("data_proportions", result.data_proportions),
    ] {
        match categorical {
            true => d.set_item(key, array1(py, shares))?,
            false => d.set_item(key, py.None())?,
        }
    }
    Ok(d)
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(training_image_consistency, m)?)
}
