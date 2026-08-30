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

#[cfg(test)]
mod tests {
    use crate::{
        feature_store::Transformation,
        models::{column::Column, table::ColumnData},
    };

    use super::*;

    #[test]
    fn create_snapshot_succeeds_and_can_be_retrieved() {
        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("init".to_string()),
            )
            .unwrap();

        let result = store.create_snapshot(vec![("age".to_string(), 1)], Some("v1".to_string()));

        assert!(result.is_ok());
        let id = result.unwrap();
        let fetched = store.get_snapshot(&id);
        assert!(fetched.is_ok());
    }

    #[test]
    fn create_snapshot_is_deterministic_for_same_order() {
        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("init".to_string()),
            )
            .unwrap();

        let id1 = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();
        let id2 = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();

        assert_eq!(id1, id2);
    }

    #[test]
    fn create_snapshot_differs_by_order() {
        let mut store = FeatureStore::new(5).unwrap();
        let age = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        let income = Column::new_from_parsed(vec![Some(2.0)], "income".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(age),
                Transformation::Custom("init".to_string()),
            )
            .unwrap();
        store
            .commit(
                ColumnData::Float(income),
                Transformation::Custom("init".to_string()),
            )
            .unwrap();

        let id1 = store
            .create_snapshot(
                vec![("age".to_string(), 1), ("income".to_string(), 1)],
                None,
            )
            .unwrap();
        let id2 = store
            .create_snapshot(
                vec![("income".to_string(), 1), ("age".to_string(), 1)],
                None,
            )
            .unwrap();

        assert_ne!(id1, id2);
    }

    #[test]
    fn create_snapshot_fails_for_nonexistent_version() {
        let mut store = FeatureStore::new(5).unwrap();

        let result = store.create_snapshot(vec![("age".to_string(), 1)], None);

        match result {
            Err(FeatureStoreError::VersionNotFound { name, version }) => {
                assert_eq!(name, "age".to_string());
                assert_eq!(version, 1);
            }
            _ => panic!("Unexpected result."),
        }
        assert!(store.snapshots.is_empty());
    }

    #[test]
    fn get_snapshot_fails_for_unknown_id() {
        let store = FeatureStore::new(5).unwrap();
        let fake_id = SnapshotId("does-not-exist".to_string());

        let result = store.get_snapshot(&fake_id);

        match result {
            Err(FeatureStoreError::SnapshotNotFound { id }) => {
                assert_eq!(id, fake_id);
            }
            _ => panic!("Unexpected result."),
        }
    }
}
