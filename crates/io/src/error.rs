#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error(transparent)]
    Regex(#[from] regex::Error),
    #[error("column `{0}` is not numeric")]
    NotNumeric(String),
    #[error("line {line}: {message}")]
    Format { line: usize, message: String },
}

pub type Result<T> = std::result::Result<T, Error>;
