use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    feature_store::FeatureStoreError,
    models::table::{ColumnData, Table},
};

use super::FeatureStore;

/// A content-addressed identifier for a [`Snapshot`] — a hex-encoded SHA-256 digest computed
/// over the snapshot's `(name, version)` pairs *and their order* (see
/// [`FeatureStore::create_snapshot`]). The same combination, given in the same order, always
/// produces the same `SnapshotId`; a different order produces a different one, since feature
/// order is significant for reconstructing the exact tensor layout a model was trained on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SnapshotId(String);

/// A named, ordered combination of feature versions — the "dataset version" a model is actually
/// trained or served against, as opposed to [`ColumnVersion`]'s per-feature history.
///
/// Created via [`FeatureStore::create_snapshot`] and looked up via
/// [`FeatureStore::get_snapshot`]. `ordered_features`' order is preserved exactly as given at
/// creation time, so it can later drive [`Table::to_tensor`]'s `feature_columns` argument
/// without the caller having to remember or re-derive the order themselves.
///
/// [`ColumnVersion`]: crate::feature_store::ColumnVersion
/// [`Table::to_tensor`]: crate::models::table::Table::to_tensor
pub struct Snapshot {
    id: SnapshotId,
    ordered_features: Vec<(String, usize)>,
    timestamp: SystemTime,
    label: Option<String>,
}

impl Snapshot {
    /// This snapshot's content-addressed identifier.
    pub fn id(&self) -> &SnapshotId {
        &self.id
    }

    /// The `(feature name, version)` pairs this snapshot pins down, in the order they were
    /// given to [`FeatureStore::create_snapshot`].
    pub fn ordered_features(&self) -> &[(String, usize)] {
        &self.ordered_features
    }

    /// When this snapshot was created.
    pub fn timestamp(&self) -> SystemTime {
        self.timestamp
    }

    /// The optional human-readable tag given to this snapshot at creation time, if any.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
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
    /// Records `ordered_features` — an ordered list of `(feature name, version)` pairs — as a
    /// single, named [`Snapshot`], and returns its [`SnapshotId`].
    ///
    /// The order of `ordered_features` is significant and preserved as given: it becomes part
    /// of the computed `SnapshotId` (the same pairs in a different order produce a different
    /// snapshot), and is later handed back unchanged by [`FeatureStore::get_snapshot`] so a
    /// caller can reconstruct the exact same tensor layout without having to remember or
    /// re-derive the order themselves. `label` is an optional human-readable tag (e.g.
    /// `"v1-training-set"`); pass `None` if it isn't needed.
    ///
    /// Every `(name, version)` pair is validated against [`FeatureStore::get_version`] *before*
    /// the snapshot is computed or stored, so a failing call leaves the store unchanged — no
    /// partial or invalid snapshot is ever recorded. `ordered_features` also can't repeat the
    /// same feature name twice — this is checked before any version lookup happens.
    ///
    /// # Errors
    /// Returns [`FeatureStoreError::DuplicateFeatureInSnapshot`] if the same feature name
    /// appears more than once in `ordered_features`, or [`FeatureStoreError::VersionNotFound`]
    /// if any `(name, version)` pair doesn't exist.
    pub fn create_snapshot(
        &mut self,
        ordered_features: Vec<(String, usize)>,
        label: Option<String>,
    ) -> Result<SnapshotId, FeatureStoreError> {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (name, version) in ordered_features.iter() {
            if !seen.insert(name.as_str()) {
                return Err(FeatureStoreError::DuplicateFeatureInSnapshot { name: name.clone() });
            }
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

    /// Looks up a previously created [`Snapshot`] by its [`SnapshotId`].
    ///
    /// # Errors
    /// Returns [`FeatureStoreError::SnapshotNotFound`] if `id` doesn't match any snapshot
    /// created by [`FeatureStore::create_snapshot`].
    pub fn get_snapshot(&self, id: &SnapshotId) -> Result<&Snapshot, FeatureStoreError> {
        self.snapshots
            .get(id)
            .ok_or_else(|| FeatureStoreError::SnapshotNotFound { id: id.clone() })
    }

    /// Rebuilds a fresh, standalone [`Table`] from a previously created [`Snapshot`], walking
    /// `ordered_features` in order and adding each feature's recorded version as a column via
    /// [`Table::add_column`].
    ///
    /// The returned `Table` is a brand-new copy — the underlying data is reconstructed from
    /// each [`ColumnVersion`] (via [`ColumnData::select_rows`], since `Column<T>` isn't
    /// `Clone`), so mutating it never affects the `FeatureStore`'s stored history.
    ///
    /// # Errors
    /// Returns [`FeatureStoreError::SnapshotNotFound`] if `id` doesn't exist,
    /// [`FeatureStoreError::VersionNotFound`] if a recorded `(name, version)` pair no longer
    /// exists (e.g. pruned since the snapshot was created), or
    /// [`FeatureStoreError::TableBuildFailed`] if adding a reconstructed column to the new
    /// `Table` fails.
    ///
    /// [`Table`]: crate::models::table::Table
    /// [`Table::add_column`]: crate::models::table::Table::add_column
    /// [`ColumnVersion`]: crate::feature_store::ColumnVersion
    /// [`ColumnData::select_rows`]: crate::models::table::ColumnData::select_rows
    pub fn reconstruct_table(&self, id: &SnapshotId) -> Result<Table, FeatureStoreError> {
        let snapshot = self.get_snapshot(id)?;
        let mut table = Table::default();

        for (name, version) in snapshot.ordered_features.iter() {
            let column_version = self.get_version(name, *version)?;
            let full_indices: Vec<usize> = (0..column_version.data.len()).collect();
            let owned_data = column_version.data.select_rows(&full_indices);

            match owned_data {
                ColumnData::Int(c) => table
                    .add_column(c)
                    .map_err(FeatureStoreError::TableBuildFailed)?,
                ColumnData::Float(c) => table
                    .add_column(c)
                    .map_err(FeatureStoreError::TableBuildFailed)?,
                ColumnData::Text(c) => table
                    .add_column(c)
                    .map_err(FeatureStoreError::TableBuildFailed)?,
                ColumnData::Bool(c) => table
                    .add_column(c)
                    .map_err(FeatureStoreError::TableBuildFailed)?,
                ColumnData::Raw(_) => unreachable!("commit rejects Raw columns"),
            };
        }

        Ok(table)
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
    fn create_snapshot_fails_for_duplicate_feature_name() {
        let mut store = FeatureStore::new(5).unwrap();
        let column1 = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        let column2 = Column::new_from_parsed(vec![Some(2.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column1),
                Transformation::Custom("v1".to_string()),
            )
            .unwrap();
        store
            .commit(
                ColumnData::Float(column2),
                Transformation::Custom("v2".to_string()),
            )
            .unwrap();

        let result =
            store.create_snapshot(vec![("age".to_string(), 1), ("age".to_string(), 2)], None);

        match result {
            Err(FeatureStoreError::DuplicateFeatureInSnapshot { name }) => {
                assert_eq!(name, "age".to_string());
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

    #[test]
    fn reconstruct_table_succeeds_and_rebuilds_table() {
        let mut store = FeatureStore::new(5).unwrap();
        let column =
            Column::new_from_parsed(vec![Some(1.0), Some(2.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("init".to_string()),
            )
            .unwrap();
        let id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();

        let result = store.reconstruct_table(&id);

        assert!(result.is_ok());
        let table = result.unwrap();
        assert_eq!(table.row_count(), 2);
        assert_eq!(table.column_count(), 1);
    }

    #[test]
    fn reconstruct_table_preserves_column_values() {
        let mut store = FeatureStore::new(5).unwrap();
        let column =
            Column::new_from_parsed(vec![Some(1.0), Some(2.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("init".to_string()),
            )
            .unwrap();
        let id = store
            .create_snapshot(vec![("age".to_string(), 1)], None)
            .unwrap();

        let mut table = store.reconstruct_table(&id).unwrap();
        let rebuilt_column = table.get_column::<f64>("age").unwrap();

        assert_eq!(rebuilt_column.get(0), Some(&Some(1.0)));
        assert_eq!(rebuilt_column.get(1), Some(&Some(2.0)));
    }

    #[test]
    fn reconstruct_table_fails_for_unknown_snapshot() {
        let store = FeatureStore::new(5).unwrap();
        let fake_id = SnapshotId("does-not-exist".to_string());

        let result = store.reconstruct_table(&fake_id);

        match result {
            Err(FeatureStoreError::SnapshotNotFound { id }) => {
                assert_eq!(id, fake_id);
            }
            _ => panic!("Unexpected result."),
        }
    }
}
