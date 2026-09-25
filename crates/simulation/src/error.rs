use thiserror::Error;

#[derive(Error, Debug)]
pub enum SimError {
    #[error("insufficient data: {0}")]
    InsufficientData(String),

    #[error("invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("estimation error during simulation: {0}")]
    Estimation(String),

    #[error("transform error: {0}")]
    Transform(String),
}

pub type Result<T> = std::result::Result<T, SimError>;
