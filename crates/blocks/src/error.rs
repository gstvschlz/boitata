use thiserror::Error;

#[derive(Error, Debug)]
pub enum BlockModelError {
    #[error("Invalid grid parameters: {0}")]
    InvalidGridParams(String),

    #[error("Invalid mesh: {0}")]
    InvalidMesh(String),

    #[error("Point-in-solid test failed: {0}")]
    PointInSolidFailed(String),

    #[error("Domain assignment failed: {0}")]
    DomainAssignmentFailed(String),
}

pub type Result<T> = std::result::Result<T, BlockModelError>;
