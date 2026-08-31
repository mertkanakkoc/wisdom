mod artifact;
mod linear_reg;

pub use artifact::{ArtifactError, ModelArchitecture, ModelArtifact, TrainingOutput};
pub use linear_reg::train_linear_regression;
