//! Tabular and mesh readers and writers. Every reader returns an Arrow record batch
//! with nullable columns; numbers are `Float64`, missing values are null.

mod csv;
mod error;
mod gslib;
mod mesh;
mod parquet;
mod shapefile;

pub use csv::{CsvOptions, read_csv, write_csv};
pub use error::{Error, Result};
pub use gslib::{read_gslib, write_gslib};
pub use mesh::{read_mesh, write_mesh};
pub use parquet::{
    BlockChunks, BlockModelReader, BlockModelWriter, FileLayout, Stored, read_model, read_parquet,
    stream_map, write_block_model, write_model, write_parquet, write_points,
};
pub use shapefile::{read_shapefile, write_shapefile};

/// Values read as null unless the caller overrides them (case-insensitive).
pub const NODATA: &[&str] = &[
    "-99", "-999", "1e21", "1e+21", "NA", "N/A", "N.A.", "ND", "N/D", "NULL", "NONE", "NAN",
    "#N/A", "-", "--",
];

fn is_nodata(token: &str, nodata: &[String]) -> bool {
    token.is_empty() || nodata.iter().any(|s| s.eq_ignore_ascii_case(token))
}

fn default_nodata() -> Vec<String> {
    NODATA.iter().map(|s| s.to_string()).collect()
}
