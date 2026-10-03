//! Spatial containers, rotations and geometry shared across Boitatá.

mod block_model;
mod categories;
mod error;
mod mesh;
pub mod nonfinite;
mod points;
mod polylines;
mod progress;
pub mod rng;
mod rotation;
pub mod units;

pub use arrow_array::RecordBatch;
pub use block_model::{BlockModel, Geometry, Layout};
pub use categories::Categories;
pub use error::{Error, Result};
pub use mesh::{Mesh, MeshProblem, MeshProblemKind, MeshReport, MeshSummary, signed_solid_angle};
pub use points::PointSet;
pub use polylines::Polylines;
pub use progress::Progress;
pub use rotation::{angles_from_axes, block_frame, rotation_matrix};

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

fn set_column(
    table: &RecordBatch,
    name: &str,
    column: arrow_array::ArrayRef,
) -> Result<RecordBatch> {
    check_rows(column.len(), table)?;
    let schema = table.schema();
    let mut fields: Vec<_> = schema.fields().iter().cloned().collect();
    let mut columns = table.columns().to_vec();
    let field = std::sync::Arc::new(arrow_schema::Field::new(
        name,
        column.data_type().clone(),
        true,
    ));
    match schema.index_of(name) {
        Ok(i) => {
            fields[i] = field;
            columns[i] = column;
        }
        Err(_) => {
            fields.push(field);
            columns.push(column);
        }
    }
    let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(table.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        std::sync::Arc::new(arrow_schema::Schema::new_with_metadata(
            fields,
            schema.metadata().clone(),
        )),
        columns,
        &options,
    )?)
}
