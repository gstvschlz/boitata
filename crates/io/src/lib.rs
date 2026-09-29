//! Tabular and mesh readers and writers. Every reader returns an Arrow record batch
//! with nullable columns; numbers are `Float64`, missing values are null.

mod csv;
mod error;
mod geotiff;
mod gslib;
mod mesh;
mod parquet;
mod shapefile;

pub use csv::{CsvOptions, read_csv, write_csv};
pub use error::{Error, Result};
pub use geotiff::{read_geotiff, write_geotiff};
pub use gslib::{read_gslib, write_gslib};
pub use mesh::{read_mesh, write_mesh};
pub use parquet::{
    BlockChunks, BlockModelReader, BlockModelWriter, FileLayout, Stored, read_model, read_parquet,
    stream_map, write_block_model, write_model, write_parquet, write_parquet_batches, write_points,
    write_polylines,
};
pub use shapefile::{Shapes, read_shapefile, write_polylines_shapefile, write_shapefile};

/// A value read as null. A number matches every token that parses to it, so
/// `-999` matches `-999.0`; text matches the token case-insensitively.
#[derive(Debug, Clone, PartialEq)]
pub enum Nodata {
    Number(f64),
    Text(String),
}

/// Values read as null unless the caller overrides them.
pub fn default_nodata() -> Vec<Nodata> {
    let text = [
        "NA", "N/A", "N.A.", "ND", "N/D", "NULL", "NONE", "NAN", "#N/A", "-", "--",
    ];
    [-99.0, -999.0, 1e21]
        .map(Nodata::Number)
        .into_iter()
        .chain(text.map(|s| Nodata::Text(s.into())))
        .collect()
}

fn is_nodata(token: &str, nodata: &[Nodata]) -> bool {
    let number = token.parse::<f64>().ok();
    token.is_empty()
        || nodata.iter().any(|n| match n {
            Nodata::Number(v) => number == Some(*v),
            Nodata::Text(s) => s.eq_ignore_ascii_case(token),
        })
}
