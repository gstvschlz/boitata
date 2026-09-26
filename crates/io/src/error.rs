#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error(transparent)]
    Regex(#[from] regex::Error),
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error(transparent)]
    Container(#[from] ceres_core::Error),
    #[error("invalid ceres metadata: {0}")]
    Metadata(String),
    #[error("column `{0}` is not numeric")]
    NotNumeric(String),
    #[error("invalid mesh file: {0}")]
    Mesh(String),
    #[error("shapefile: {0}")]
    Shapefile(String),
    #[error("line {line}: {message}")]
    Format { line: usize, message: String },
}

pub type Result<T> = std::result::Result<T, Error>;
