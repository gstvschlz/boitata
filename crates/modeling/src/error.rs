use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("insufficient data: {0}")]
    InsufficientData(String),
    #[error("singular system: {0}")]
    Singular(String),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error(transparent)]
    Variogram(#[from] variogram::VarioError),
    #[error(transparent)]
    Estimation(#[from] estimation::EstimError),
}

pub type Result<T> = std::result::Result<T, ModelError>;
