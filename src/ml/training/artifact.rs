use std::{io::BufReader, path::Path, time::SystemTime};

use crate::feature_store::SnapshotId;
use candle_core::Device;
use candle_nn::{Linear, VarBuilder, VarMap, linear};
use serde::{Deserialize, Serialize};

/// The maximum number of characters allowed in a [`ModelArtifact`]'s `label` (see
/// [`ModelArtifact::new`]) — kept well under common filesystem filename limits even after
/// accounting for the fixed-length [`SnapshotId`] and file extension that share the same
/// filename.
pub const MAX_LABEL_CHARACTER: usize = 64;

/// Errors that can occur while creating or saving a [`ModelArtifact`].
#[derive(Debug)]
pub enum ArtifactError {
    /// [`ModelArtifact::new`] was given an empty `label`.
    EmptyLabel,
    /// [`ModelArtifact::new`] was given a `label` longer than [`MAX_LABEL_CHARACTER`].
    LabelTooLong,
    /// [`ModelArtifact::new`] was given a `label` containing a character other than ASCII
    /// letters, digits, `-`, or `_` — `label` becomes part of a filename (see
    /// [`ModelArtifact::save`]), so only filesystem-safe characters are allowed.
    InvalidCharacter,
    /// [`ModelArtifact::save`] failed while writing the model's weights (via
    /// [`candle_nn::VarMap::save`]).
    WeightsSaveFailed(candle_core::Error),
    /// [`ModelArtifact::save`] failed while creating the metadata file on disk.
    FileCreationFailed(std::io::Error),
    /// [`ModelArtifact::save`] failed while writing the metadata file's JSON contents.
    SerializationFailed(serde_json::Error),
    /// [`ModelArtifact::load`] failed while parsing the metadata file's JSON contents.
    DeserializationFailed(serde_json::Error),
    /// [`ModelArtifact::load`] failed to open the metadata file (e.g. no artifact exists for
    /// the given `snapshot_id`/`label` pair in the given directory).
    ArtifactFileOpenFailed(std::io::Error),
    /// [`ModelArtifact::load`] failed while reading the weights file (via
    /// [`candle_nn::VarMap::load`]).
    WeightsLoadedFailed(candle_core::Error),
    /// [`ModelArtifact::load`] failed while rebuilding the untrained, same-shaped model (e.g.
    /// via [`candle_nn::linear()`]) that the saved weights are loaded into.
    ModelCreationFailed(candle_core::Error),
}

/// Describes a model's shape — what's needed to rebuild a same-shaped, untrained model before
/// loading saved weights into it (`candle_nn::VarMap::load` fills in *already-declared*
/// variables; it doesn't recreate the architecture that produced them).
///
/// One variant per known model type, in the same spirit as [`Transformation`] — new variants
/// are added only once a real second model type exists, not speculatively.
///
/// [`Transformation`]: crate::feature_store::Transformation
#[derive(Debug, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Linear {
        in_features: usize,
        out_features: usize,
    },
}

/// A model reconstructed by [`ModelArtifact::load`], tagged by which concrete type it turned
/// out to be.
///
/// One variant per known model type, growing in step with [`ModelArchitecture`] — kept separate
/// from `ModelArchitecture` itself (which only describes *shape*, not real weight values) so
/// that saving a [`ModelArtifact`]'s metadata to JSON never accidentally tries to serialize an
/// actual `Tensor` (which `candle` doesn't support anyway) into the wrong file.
pub enum LoadedModel {
    Linear(Linear),
}
/// Everything [`crate::ml::training::train_linear_regression`] produces, bundled together:
/// the trained, ready-to-use model itself (`model`), its backing [`VarMap`] (needed to persist
/// the weights via [`ModelArtifact::save`]), and its [`ModelArchitecture`] (needed to rebuild
/// the same shape when loading it back later).
///
/// Generic over the model type `M` so future training functions (e.g. for a second model type)
/// can reuse this same shape.
pub struct TrainingOutput<M> {
    pub model: M,
    pub varmap: VarMap,
    pub architecture: ModelArchitecture,
}

/// A persisted, reloadable record of a trained model — the metadata half of what
/// [`ModelArtifact::save`] writes to disk (the weights themselves live in a separate
/// `.safetensors` file, written via [`VarMap::save`]).
///
/// Ties a model back to the exact [`Snapshot`](crate::feature_store::Snapshot) (via
/// [`SnapshotId`]) it was trained against, so it can later be reconstructed and matched with
/// the right feature data. `label` is a caller-chosen, filesystem-safe name (see
/// [`ModelArtifact::new`]) used together with the `snapshot_id` to name the saved files.
#[derive(Serialize, Deserialize)]
pub struct ModelArtifact {
    architecture: ModelArchitecture,
    snapshot_id: SnapshotId,
    label: String,
    #[serde(with = "unix_seconds")]
    timestamp: SystemTime,
}

impl ModelArtifact {
    /// Creates a new `ModelArtifact` record, validating `label` in the process. `timestamp` is
    /// set to the current time automatically.
    ///
    /// # Errors
    /// Returns [`ArtifactError::EmptyLabel`] if `label` is empty,
    /// [`ArtifactError::LabelTooLong`] if it's longer than [`MAX_LABEL_CHARACTER`], or
    /// [`ArtifactError::InvalidCharacter`] if it contains anything other than ASCII
    /// letters/digits, `-`, or `_`.
    pub fn new(
        architecture: ModelArchitecture,
        snapshot_id: SnapshotId,
        label: String,
    ) -> Result<Self, ArtifactError> {
        if label.is_empty() {
            return Err(ArtifactError::EmptyLabel);
        }
        if label.chars().count() > MAX_LABEL_CHARACTER {
            return Err(ArtifactError::LabelTooLong);
        }
        if !label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(ArtifactError::InvalidCharacter);
        }
        Ok(Self {
            architecture,
            snapshot_id,
            label,
            timestamp: SystemTime::now(),
        })
    }

    /// This artifact's model shape.
    pub fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }

    /// The [`Snapshot`](crate::feature_store::Snapshot) this model was trained against.
    pub fn snapshot_id(&self) -> &SnapshotId {
        &self.snapshot_id
    }

    /// This artifact's caller-chosen, filesystem-safe label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// When this artifact was created.
    pub fn timestamp(&self) -> SystemTime {
        self.timestamp
    }

    /// Writes this artifact to `dir`, as two files named `{snapshot_id}_{label}.safetensors`
    /// (the model's weights, via [`VarMap::save`]) and `{snapshot_id}_{label}.json` (this
    /// artifact's own metadata, via `serde_json`).
    ///
    /// The weights file is written *before* the metadata file — a reader can treat the presence
    /// of the metadata file as confirmation that the pair is complete, since a crash between
    /// the two writes leaves only the (harmless, ignorable) weights file behind.
    ///
    /// # Errors
    /// Returns [`ArtifactError::WeightsSaveFailed`] if writing the weights fails,
    /// [`ArtifactError::FileCreationFailed`] if the metadata file can't be created, or
    /// [`ArtifactError::SerializationFailed`] if writing the metadata file's JSON fails.
    pub fn save(&self, varmap: &VarMap, dir: &Path) -> Result<String, ArtifactError> {
        let file_name = format!("{}_{}", self.snapshot_id, self.label);
        let varmap_file_name = dir.join(format!("{}.safetensors", file_name));
        let artifact_file_name = dir.join(format!("{}.json", file_name));

        varmap
            .save(varmap_file_name)
            .map_err(ArtifactError::WeightsSaveFailed)?;

        let artifact_file = std::fs::File::create(&artifact_file_name)
            .map_err(ArtifactError::FileCreationFailed)?;

        serde_json::to_writer(artifact_file, self).map_err(ArtifactError::SerializationFailed)?;
        Ok(format!("Model information saved."))
    }

    /// Loads a previously [`saved`](ModelArtifact::save) artifact back from `dir`, given the
    /// same `snapshot_id`/`label` pair it was saved under.
    ///
    /// Reads the metadata file first to recover the model's [`ModelArchitecture`], then rebuilds
    /// a same-shaped, untrained model from it (e.g. via [`candle_nn::linear()`]) before loading
    /// the saved weights into it — `candle_nn::VarMap::load` only fills in *already-declared*
    /// variables, it doesn't recreate the architecture that produced them. `device` chooses
    /// where the loaded model's tensors live; it doesn't need to match the device used at
    /// training time (the saved weights are plain numeric data, not tied to any device).
    ///
    /// Returns both the reconstructed [`LoadedModel`] and the [`ModelArtifact`] metadata itself
    /// (read from the same file), so callers don't need to keep the original artifact around
    /// separately just to inspect it again.
    ///
    /// # Errors
    /// Returns [`ArtifactError::ArtifactFileOpenFailed`] if the metadata file doesn't exist,
    /// [`ArtifactError::DeserializationFailed`] if it can't be parsed,
    /// [`ArtifactError::ModelCreationFailed`] if rebuilding the untrained model fails, or
    /// [`ArtifactError::WeightsLoadedFailed`] if reading the weights file fails.
    pub fn load(
        snapshot_id: &SnapshotId,
        label: &str,
        dir: &Path,
        device: &Device,
    ) -> Result<(LoadedModel, ModelArtifact), ArtifactError> {
        let file_name = format!("{}_{}", snapshot_id, label);
        let varmap_file_name = dir.join(format!("{}.safetensors", file_name));
        let artifact_file_name = dir.join(format!("{}.json", file_name));

        let artifact_file = std::fs::File::open(artifact_file_name)
            .map_err(ArtifactError::ArtifactFileOpenFailed)?;
        let reader = BufReader::new(artifact_file);
        let model_artifact: ModelArtifact =
            serde_json::from_reader(reader).map_err(ArtifactError::DeserializationFailed)?;

        match model_artifact.architecture() {
            ModelArchitecture::Linear {
                in_features,
                out_features,
            } => {
                let mut varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
                let linear_model = linear(*in_features, *out_features, vb.pp("linear"))
                    .map_err(ArtifactError::ModelCreationFailed)?;
                varmap
                    .load(varmap_file_name)
                    .map_err(ArtifactError::WeightsLoadedFailed)?;
                Ok((LoadedModel::Linear(linear_model), model_artifact))
            }
        }
    }
}

mod unix_seconds {
    use serde::ser::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).map_err(S::Error::custom)?;
        let seconds = duration.as_secs();
        serializer.serialize_u64(seconds)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let seconds = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs(seconds))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use crate::ml::training::train_linear_regression;
    use candle_nn::Module;

    use crate::{
        feature_store::{FeatureStore, Transformation},
        models::{column::Column, table::ColumnData},
    };

    use super::*;

    #[test]
    fn model_artifact_new_succeeds_with_valid_label() {
        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();
        let architecture = ModelArchitecture::Linear {
            in_features: 1,
            out_features: 1,
        };

        let result = ModelArtifact::new(architecture, snapshot_id, "my-model_v1".to_string());

        assert!(result.is_ok());
        let artifact = result.unwrap();
        assert_eq!(artifact.label(), "my-model_v1");
    }

    #[test]
    fn model_artifact_new_fails_for_empty_label() {
        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();
        let architecture = ModelArchitecture::Linear {
            in_features: 1,
            out_features: 1,
        };

        let result = ModelArtifact::new(architecture, snapshot_id, "".to_string());

        match result {
            Err(ArtifactError::EmptyLabel) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn model_artifact_new_fails_for_too_long_label() {
        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();
        let architecture = ModelArchitecture::Linear {
            in_features: 1,
            out_features: 1,
        };
        let long_label = "a".repeat(99);

        let result = ModelArtifact::new(architecture, snapshot_id, long_label);

        match result {
            Err(ArtifactError::LabelTooLong) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn model_artifact_new_fails_for_invalid_character() {
        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();
        let architecture = ModelArchitecture::Linear {
            in_features: 1,
            out_features: 1,
        };

        let result = ModelArtifact::new(architecture, snapshot_id, "bad/label".to_string());

        match result {
            Err(ArtifactError::InvalidCharacter) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn save_writes_weights_and_metadata_files() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.]], &device).unwrap();
        let y = Tensor::new(&[3f32, 5., 7., 9.], &device).unwrap();
        let training_output = train_linear_regression(&x, &y, 5, 0.05, &device).unwrap();

        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();

        let artifact = ModelArtifact::new(
            training_output.architecture,
            snapshot_id,
            "test-model".to_string(),
        )
        .unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let result = artifact.save(&training_output.varmap, temp_dir.path());

        assert!(result.is_ok());
        let weights_path = temp_dir.path().join(format!(
            "{}_{}.safetensors",
            artifact.snapshot_id(),
            artifact.label()
        ));
        let metadata_path = temp_dir.path().join(format!(
            "{}_{}.json",
            artifact.snapshot_id(),
            artifact.label()
        ));
        assert!(weights_path.exists());
        assert!(metadata_path.exists());
    }

    #[test]
    fn save_fails_when_directory_does_not_exist() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.]], &device).unwrap();
        let y = Tensor::new(&[3f32, 5., 7., 9.], &device).unwrap();
        let training_output = train_linear_regression(&x, &y, 5, 0.05, &device).unwrap();

        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();

        let artifact = ModelArtifact::new(
            training_output.architecture,
            snapshot_id,
            "test-model".to_string(),
        )
        .unwrap();

        let bad_dir = std::path::Path::new("/this/does/not/exist/at/all");
        let result = artifact.save(&training_output.varmap, bad_dir);

        match result {
            Err(ArtifactError::WeightsSaveFailed(_)) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn load_restores_the_saved_model_and_metadata() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1f32], [2.], [3.], [4.]], &device).unwrap();
        let y = Tensor::new(&[3f32, 5., 7., 9.], &device).unwrap();
        let training_output = train_linear_regression(&x, &y, 5, 0.05, &device).unwrap();
        let original_predictions = training_output.model.forward(&x).unwrap();

        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();

        let artifact = ModelArtifact::new(
            training_output.architecture,
            snapshot_id.clone(),
            "test-model".to_string(),
        )
        .unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        artifact
            .save(&training_output.varmap, temp_dir.path())
            .unwrap();

        let result = ModelArtifact::load(&snapshot_id, "test-model", temp_dir.path(), &device);

        assert!(result.is_ok());
        let (loaded_model, loaded_artifact) = result.unwrap();
        assert_eq!(loaded_artifact.label(), "test-model".to_string());
        assert_eq!(loaded_artifact.snapshot_id(), &snapshot_id);

        match loaded_model {
            LoadedModel::Linear(linear) => {
                let loaded_predictions = linear.forward(&x).unwrap();
                let diff = (loaded_predictions - original_predictions)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_vec0::<f32>()
                    .unwrap();
                assert!(diff < 0.0001);
            }
        }
    }

    #[test]
    fn load_fails_when_metadata_file_does_not_exist() {
        let device = Device::Cpu;
        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        let snapshot_id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let result =
            ModelArtifact::load(&snapshot_id, "nonexistent-label", temp_dir.path(), &device);

        match result {
            Err(ArtifactError::ArtifactFileOpenFailed(_)) => {}
            _ => panic!("Unexpected result."),
        }
    }
}
