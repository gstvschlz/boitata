#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("expected {expected} rows, found {found}")]
    Length { expected: usize, found: usize },
    #[error("column `{0}` not found")]
    MissingColumn(String),
    #[error("coordinate column `{0}` must be numeric without nulls")]
    InvalidCoordinates(String),
    #[error("invalid geometry: {0}")]
    Geometry(String),
    #[error("invalid categories: {0}")]
    Categories(String),
    #[error("{0}")]
    UnknownLabels(String),
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),
}

pub type Result<T> = std::result::Result<T, Error>;
