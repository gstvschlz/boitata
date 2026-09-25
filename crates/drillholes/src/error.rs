use thiserror::Error;

#[derive(Error, Debug)]
pub enum DrillholeError {
    #[error("Invalid collar data: {0}")]
    InvalidCollar(String),

    #[error("Invalid survey station: {0}")]
    InvalidSurveyStation(String),

    #[error("Desurvey failed: {0}")]
    DesurveylFailed(String),

    #[error("Compositing error: {0}")]
    CompositeError(String),

    #[error("Missing data: {0}")]
    MissingData(String),
}

pub type Result<T> = std::result::Result<T, DrillholeError>;
