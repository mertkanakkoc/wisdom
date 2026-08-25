use std::{collections::HashMap, time::SystemTime};

use crate::models::table::ColumnData;

#[derive(Debug)]
pub enum FeatureStoreError {
    MaxVersionsTooLow { min: usize, actual: usize },
    RawColumnExists,
    VersionNotFound { name: String, version: usize },
    FeatureNotFound { name: String },
}

pub enum Transformation {
    FillWith,
    FillMean,
    ForwardFill,
    BackwardFill,
    MinMaxScale,
    ZScoreStandardization,
    Custom(String),
}

pub struct ColumnVersion {
    data: ColumnData,
    version_number: usize,
    name: String,
    transformation: Transformation,
    timestamp: SystemTime,
}

pub struct FeatureStore {
    versions: HashMap<String, Vec<ColumnVersion>>,
    max_versions: usize,
}

impl FeatureStore {
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
        })
    }

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

        Ok(max_version + 1)
    }

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
