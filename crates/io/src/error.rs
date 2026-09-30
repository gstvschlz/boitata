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
    Container(#[from] boitata_core::Error),
    #[error("invalid boitata metadata: {0}")]
    Metadata(String),
    #[error("column `{0}` is not numeric")]
    NotNumeric(String),
    #[error("invalid mesh file: {0}")]
    Mesh(String),
    #[error("GeoTIFF: {0}")]
    GeoTiff(String),
    #[error("SEG-Y: {0}")]
    Segy(String),
    #[error("shapefile: {0}")]
    Shapefile(String),
    #[error("line {line}: {message}")]
    Format { line: usize, message: String },
}

pub type Result<T> = std::result::Result<T, Error>;
