use thiserror::Error;

#[derive(Error, Debug)]
pub enum VarioError {
    #[error("insufficient data: {0}")]
    InsufficientData(String),

    #[error("invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("model fitting failed: {0}")]
    FittingFailed(String),

    #[error("invalid anisotropy: {0}")]
    InvalidAnisotropy(String),
}

pub type Result<T> = std::result::Result<T, VarioError>;
