use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodaError {
    #[error("invalid composition: {0}")]
    InvalidComposition(String),

    #[error("invalid parameters: {0}")]
    InvalidParameters(String),
}

pub type Result<T> = std::result::Result<T, CodaError>;
