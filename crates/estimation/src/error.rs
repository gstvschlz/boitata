use thiserror::Error;

#[derive(Error, Debug)]
pub enum EstimError {
    #[error("singular kriging system: {0}")]
    Singular(String),

    #[error("insufficient data: {0}")]
    InsufficientData(String),

    #[error("invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("search failed: {0}")]
    SearchFailed(String),
}

pub type Result<T> = std::result::Result<T, EstimError>;
