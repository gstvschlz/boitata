use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransformError {
    #[error("insufficient data: {0}")]
    InsufficientData(String),

    #[error("invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("fitting failed: {0}")]
    FittingFailed(String),
}

pub type Result<T> = std::result::Result<T, TransformError>;
