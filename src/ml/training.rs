mod artifact;
mod linear_reg;

pub use artifact::{ArtifactError, MAX_LABEL_CHARACTER, ModelArchitecture, ModelArtifact, TrainingOutput};
pub use linear_reg::train_linear_regression;
