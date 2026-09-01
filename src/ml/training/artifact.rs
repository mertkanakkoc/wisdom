use std::time::{SystemTime, UNIX_EPOCH};

use crate::feature_store::SnapshotId;
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

pub const MAX_LABEL_CHARACTER: usize = 64;

#[derive(Debug)]
pub enum ArtifactError {
    EmptyLabel,
    LabelTooLong,
    InvalidCharacter,
    TimeSerializationError,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Linear {
        in_features: usize,
        out_features: usize,
    },
}

pub struct TrainingOutput<M> {
    pub model: M,
    pub varmap: VarMap,
    pub architecture: ModelArchitecture,
}

#[derive(Serialize, Deserialize)]
pub struct ModelArtifact {
    architecture: ModelArchitecture,
    snapshot_id: SnapshotId,
    label: String,
    #[serde(with = "unix_seconds")]
    timestamp: SystemTime,
}

impl ModelArtifact {
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

    pub fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }

    pub fn snapshot_id(&self) -> &SnapshotId {
        &self.snapshot_id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn timestamp(&self) -> SystemTime {
        self.timestamp
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
}
