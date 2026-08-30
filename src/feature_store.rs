mod drift;
mod snapshot;

use std::{collections::HashMap, time::SystemTime};

use crate::{
    feature_store::snapshot::{Snapshot, SnapshotId},
    models::table::{ColumnData, TableError},
};

/// Errors that can occur while creating or using a [`FeatureStore`].
#[derive(Debug)]
pub enum FeatureStoreError {
    /// [`FeatureStore::new`] was called with `max_versions < 2` — at least 2 is required so the
    /// permanently-kept first version and at least one more recent version can coexist.
    MaxVersionsTooLow {
        min: usize,
        actual: usize,
    },
    /// [`FeatureStore::commit`] was given a [`ColumnData::Raw`] value, which has no name of its
    /// own to commit under (see [`ColumnData::extract_name`]).
    RawColumnExists,
    /// The requested `(name, version)` pair doesn't exist — either the feature was never
    /// committed, or that specific version number doesn't exist for it (pruned or never
    /// existed). `version` always echoes back exactly what was asked for.
    VersionNotFound {
        name: String,
        version: usize,
    },
    /// [`FeatureStore::get_latest_version`] was called for a feature that was never committed.
    FeatureNotFound {
        name: String,
    },
    /// [`FeatureStore::get_snapshot`] was called with an `id` that doesn't match any snapshot
    /// created by [`FeatureStore::create_snapshot`].
    SnapshotNotFound {
        id: SnapshotId,
    },
    /// [`FeatureStore::create_snapshot`] was given `ordered_features` with the same feature
    /// name listed more than once — a snapshot can only reference each feature at most once.
    DuplicateFeatureInSnapshot {
        name: String,
    },
    /// [`FeatureStore::reconstruct_table`] failed while adding a reconstructed column back to
    /// the new [`Table`] (e.g. a duplicate name slipping past [`FeatureStore::create_snapshot`]'s
    /// own check, or a row-count mismatch between features).
    ///
    /// [`Table`]: crate::models::table::Table
    TableBuildFailed(TableError),
    /// [`FeatureStore::detect_drift`] was asked to compare a feature whose committed data is
    /// [`ColumnData::Text`] or [`ColumnData::Raw`] — drift detection only supports the same
    /// numeric types [`Table::to_tensor`] accepts (`Int`/`Float`/`Bool`).
    ///
    /// [`Table::to_tensor`]: crate::models::table::Table::to_tensor
    NonNumericFeature {
        name: String,
    },
    /// [`FeatureStore::detect_drift`] was asked to compare a version whose values are entirely
    /// missing (`None`) — a mean/standard deviation can't be computed with zero present values.
    AllValuesMissing {
        name: String,
        version: usize,
    },
}

/// The lineage record attached to a committed [`ColumnVersion`] — what was done to produce it.
/// Supplied explicitly by the caller of [`FeatureStore::commit`]; never inferred by the store.
pub enum Transformation {
    FillWith,
    FillMean,
    ForwardFill,
    BackwardFill,
    MinMaxScale,
    ZScoreStandardization,
    /// An escape hatch for any transformation not covered by the other variants.
    Custom(String),
}

/// A single, immutable, historical snapshot of one feature (column), as recorded by
/// [`FeatureStore::commit`].
pub struct ColumnVersion {
    data: ColumnData,
    version_number: usize,
    name: String,
    transformation: Transformation,
    timestamp: SystemTime,
}

/// A version history and lineage store for features (columns), separate from [`Table`]'s live,
/// mutable working copy.
///
/// [`Table`] always holds the current/working state of a column; `FeatureStore` only records a
/// snapshot when the caller explicitly calls [`FeatureStore::commit`] — there's no automatic or
/// implicit syncing, so "the latest version" here means "the last thing someone committed," not
/// necessarily "what `Table` currently holds."
///
/// On top of per-feature versioning, `FeatureStore` also lets a caller pin down a whole
/// *combination* of feature versions as a single, content-addressed [`Snapshot`] (see
/// [`FeatureStore::create_snapshot`]) — the unit a model is actually trained/served against —
/// and compare any two versions of the same feature for drift (see
/// [`FeatureStore::detect_drift`]).
///
/// [`Table`]: crate::models::table::Table
pub struct FeatureStore {
    versions: HashMap<String, Vec<ColumnVersion>>,
    max_versions: usize,
    snapshots: HashMap<SnapshotId, Snapshot>,
}

impl FeatureStore {
    /// Creates a new, empty `FeatureStore`. `max_versions` is a hard, store-wide cap on how
    /// many versions of any single feature are kept at once — the first version committed for a
    /// feature always survives pruning, so `max_versions` must be at least 2 (one slot for the
    /// first version, at least one more for recent history).
    ///
    /// # Errors
    /// Returns [`FeatureStoreError::MaxVersionsTooLow`] if `max_versions < 2`.
    pub fn new(max_versions: usize) -> Result<Self, FeatureStoreError> {
        if max_versions < 2 {
            return Err(FeatureStoreError::MaxVersionsTooLow {
                min: 2,
                actual: max_versions,
            });
        }

        Ok(Self {
            versions: HashMap::new(),
            max_versions,
            snapshots: HashMap::new(),
        })
    }

    /// Records `data` as a new version of the feature named by `data.extract_name()`, tagged
    /// with `transformation` and the current time. The feature's name is derived from `data`
    /// itself (see [`ColumnData::extract_name`]) rather than taken as a separate parameter, so
    /// it can never drift out of sync with the column's own name; to commit under a different
    /// name, rename the underlying `Column` first.
    ///
    /// Version numbers auto-increment per feature name, starting at 1. If this commit pushes
    /// the feature's version count past `max_versions`, the oldest version *after* the first
    /// one is pruned — the very first version committed for a feature is never pruned.
    ///
    /// Returns the new version's number.
    ///
    /// # Errors
    /// Returns [`FeatureStoreError::RawColumnExists`] if `data` is [`ColumnData::Raw`]
    /// (unmaterialized, with no name to commit under).
    pub fn commit(
        &mut self,
        data: ColumnData,
        transformation: Transformation,
    ) -> Result<usize, FeatureStoreError> {
        let name: String;
        match data.extract_name() {
            Some(n) => {
                name = n.to_string();
            }
            None => return Err(FeatureStoreError::RawColumnExists),
        }
        let versions = self.versions.entry(name.clone()).or_insert_with(Vec::new);

        let max_version = versions.iter().map(|v| v.version_number).max().unwrap_or(0);

        versions.push(ColumnVersion {
            data,
            version_number: max_version + 1,
            name,
            transformation,
            timestamp: SystemTime::now(),
        });

        if versions.len() > self.max_versions {
            versions.remove(1);
        }

        Ok(max_version + 1)
    }

    /// Looks up a specific historical version of a feature by name and version number.
    ///
    /// # Errors
    /// Returns [`FeatureStoreError::VersionNotFound`] if `name` was never committed, or if it
    /// was but doesn't have this particular `version` (pruned or never existed).
    pub fn get_version(
        &self,
        name: &str,
        version: usize,
    ) -> Result<&ColumnVersion, FeatureStoreError> {
        self.versions
            .get(name)
            .and_then(|versions| versions.iter().find(|v| v.version_number == version))
            .ok_or_else(|| FeatureStoreError::VersionNotFound {
                name: name.to_string(),
                version,
            })
    }

    /// Looks up the most recently committed version of a feature.
    ///
    /// # Errors
    /// Returns [`FeatureStoreError::FeatureNotFound`] if `name` was never committed.
    pub fn get_latest_version(&self, name: &str) -> Result<&ColumnVersion, FeatureStoreError> {
        let max_version = self
            .versions
            .get(name)
            .map(|versions| versions.iter().map(|v| v.version_number).max().unwrap_or(0))
            .ok_or_else(|| FeatureStoreError::FeatureNotFound {
                name: name.to_string(),
            })?;
        self.get_version(name, max_version)
    }
}

#[cfg(test)]
mod tests {
    use crate::models::column::Column;

    use super::*;

    #[test]
    fn new_succeeds_with_valid_max_versions() {
        let store = FeatureStore::new(5);

        assert!(store.is_ok());
        let store = store.unwrap();
        assert_eq!(store.max_versions, 5);
        assert!(store.versions.is_empty());
        assert!(store.snapshots.is_empty());
    }

    #[test]
    fn new_fails_when_max_versions_is_zero() {
        let result = FeatureStore::new(0);

        match result {
            Err(FeatureStoreError::MaxVersionsTooLow { min, actual }) => {
                assert_eq!(min, 2);
                assert_eq!(actual, 0);
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn new_fails_when_max_versions_is_one() {
        let result = FeatureStore::new(1);

        match result {
            Err(FeatureStoreError::MaxVersionsTooLow { min, actual }) => {
                assert_eq!(min, 2);
                assert_eq!(actual, 1);
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn commit_returns_incrementing_version_numbers() {
        let mut store = FeatureStore::new(5).unwrap();
        let column1 = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        let column2 = Column::new_from_parsed(vec![Some(2.0)], "age".to_string()).unwrap();

        let first = store.commit(
            ColumnData::Float(column1),
            Transformation::Custom("v1".to_string()),
        );
        let second = store.commit(
            ColumnData::Float(column2),
            Transformation::Custom("v2".to_string()),
        );

        assert_eq!(first.unwrap(), 1);
        assert_eq!(second.unwrap(), 2);
    }

    #[test]
    fn commit_fails_for_raw_column() {
        let mut store = FeatureStore::new(5).unwrap();

        let result = store.commit(
            ColumnData::Raw(vec![Some("25".to_string())]),
            Transformation::Custom("v1".to_string()),
        );

        match result {
            Err(FeatureStoreError::RawColumnExists) => {}
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn commit_tracks_versions_independently_per_column() {
        let mut store = FeatureStore::new(5).unwrap();
        let age = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        let score = Column::new_from_parsed(vec![Some(2.0)], "score".to_string()).unwrap();

        let age_version = store.commit(
            ColumnData::Float(age),
            Transformation::Custom("v1".to_string()),
        );
        let score_version = store.commit(
            ColumnData::Float(score),
            Transformation::Custom("v1".to_string()),
        );

        assert_eq!(age_version.unwrap(), 1);
        assert_eq!(score_version.unwrap(), 1);
    }

    #[test]
    fn commit_prunes_middle_versions_beyond_max_versions() {
        let mut store = FeatureStore::new(3).unwrap();

        for i in 1..=5 {
            let column = Column::new_from_parsed(vec![Some(i as f64)], "age".to_string()).unwrap();
            store
                .commit(
                    ColumnData::Float(column),
                    Transformation::Custom(format!("v{i}")),
                )
                .unwrap();
        }

        assert!(store.get_version("age", 1).is_ok());
        assert!(store.get_version("age", 2).is_err());
        assert!(store.get_version("age", 3).is_err());
        assert!(store.get_version("age", 4).is_ok());
        assert!(store.get_version("age", 5).is_ok());
    }

    #[test]
    fn commit_never_exceeds_max_versions() {
        let mut store = FeatureStore::new(3).unwrap();

        for i in 1..=10 {
            let column = Column::new_from_parsed(vec![Some(i as f64)], "age".to_string()).unwrap();
            store
                .commit(
                    ColumnData::Float(column),
                    Transformation::Custom(format!("v{i}")),
                )
                .unwrap();
        }

        assert_eq!(store.versions.get("age").unwrap().len(), 3);
    }

    #[test]
    fn get_version_returns_committed_version() {
        let mut store = FeatureStore::new(5).unwrap();
        let column =
            Column::new_from_parsed(vec![Some(1.0), Some(2.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("initial".to_string()),
            )
            .unwrap();

        let result = store.get_version("age", 1);
        assert!(result.is_ok());
        let version = result.unwrap();
        assert_eq!(version.version_number, 1);
        assert_eq!(version.name, "age".to_string());
    }

    #[test]
    fn get_version_fails_for_nonexistent_version() {
        let mut store = FeatureStore::new(5).unwrap();
        let column = Column::new_from_parsed(vec![Some(1.0)], "age".to_string()).unwrap();
        store
            .commit(
                ColumnData::Float(column),
                Transformation::Custom("initial".to_string()),
            )
            .unwrap();

        let result = store.get_version("age", 5);

        match result {
            Err(FeatureStoreError::VersionNotFound { name, version }) => {
                assert_eq!(name, "age".to_string());
                assert_eq!(version, 5);
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn get_version_fails_for_uncommitted_column() {
        let store = FeatureStore::new(5).unwrap();

        let result = store.get_version("city", 1);

        match result {
            Err(FeatureStoreError::VersionNotFound { name, version }) => {
                assert_eq!(name, "city".to_string());
                assert_eq!(version, 1);
            }
            _ => panic!("Unexpected result."),
        }
    }

    #[test]
    fn get_latest_version_returns_highest_version() {
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

        let result = store.get_latest_version("age");

        assert!(result.is_ok());
        assert_eq!(result.unwrap().version_number, 2);
    }

    #[test]
    fn get_latest_version_fails_for_uncommitted_column() {
        let store = FeatureStore::new(5).unwrap();

        let result = store.get_latest_version("city");

        match result {
            Err(FeatureStoreError::FeatureNotFound { name }) => {
                assert_eq!(name, "city".to_string())
            }
            _ => panic!("Unexpected result."),
        }
    }
}
