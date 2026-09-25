//! Spatial containers, rotations and geometry shared across ceres.

mod block_model;
mod error;
mod points;
mod rotation;

pub use arrow_array::RecordBatch;
pub use block_model::{BlockModel, Geometry, Layout};
pub use error::{Error, Result};
pub use points::PointSet;
pub use rotation::{block_frame, rotation_matrix};

fn check_rows(expected: usize, table: &RecordBatch) -> Result<()> {
    if table.num_rows() == expected {
        Ok(())
    } else {
        Err(Error::Length {
            expected,
            found: table.num_rows(),
        })
    }
}
