use std::time::SystemTime;

use sha2::{Digest, Sha256};

use crate::feature_store::FeatureStoreError;

use super::FeatureStore;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SnapshotId(String);

pub struct Snapshot {
    id: SnapshotId,
    ordered_features: Vec<(String, usize)>,
    timestamp: SystemTime,
    label: Option<String>,
}

fn compute_snapshot_id(ordered_features: &[(String, usize)]) -> SnapshotId {
    let mut hasher = Sha256::new();
    for (name, version) in ordered_features.iter() {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&(*version as u64).to_le_bytes());
    }
    let hash256 = hasher.finalize();
    let hex_string: String = hash256.iter().map(|byte| format!("{:02x}", byte)).collect();
    SnapshotId(hex_string)
}

impl FeatureStore {
    pub fn create_snapshot(
        &mut self,
        ordered_features: Vec<(String, usize)>,
        label: Option<String>,
    ) -> Result<SnapshotId, FeatureStoreError> {
        for (name, version) in ordered_features.iter() {
            self.get_version(name, *version)?;
        }

        let id = compute_snapshot_id(&ordered_features);

        let snapshot = Snapshot {
            id: id.clone(),
            ordered_features,
            timestamp: SystemTime::now(),
            label,
        };

        self.snapshots.insert(id.clone(), snapshot);

        Ok(id)
    }

    pub fn get_snapshot(&self, id: &SnapshotId) -> Result<&Snapshot, FeatureStoreError> {
        self.snapshots
            .get(id)
            .ok_or_else(|| FeatureStoreError::SnapshotNotFound { id: id.clone() })
    }
}
