use std::{collections::HashMap, time::SystemTime};

use crate::models::table::ColumnData;

#[derive(Debug)]
pub enum FeatureStoreError {
    MaxVersionsTooLow { min: usize, actual: usize },
    RawColumnExists,
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
        if max_versions == 0 {
            return Err(FeatureStoreError::MaxVersionsTooLow {
                min: 1,
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
}

#[cfg(test)]
mod tests {
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
                assert_eq!(min, 1);
                assert_eq!(actual, 0);
            }
            _ => panic!("Unexpected result."),
        }
    }
}
